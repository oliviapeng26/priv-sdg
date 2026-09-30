#!/usr/bin/env python3
"""Seeded DP-2Stage generation (variant O or U) on the full 21,523-row adult_train.csv.

Companion to sdg/generate_great.py, and shaped like it: 5 seeds from
seeds.RUN_SEEDS, one CSV per seed, resumable, per-seed try/except, one process per
GPU. All the method lives in sdg/dp2stage_driver.py (read its docstring first: the
fixed hyperparameters, the four deliberate deviations from the released scripts,
and why max_new_tokens is 115); this script only loops seeds and validates output.

RUN IT IN ~/venv-dp2stage
    The driver needs DP-2Stage's pinned stack (python 3.9, torch 2.5.1, transformers
    4.30.2, opacus 1.4.1), so this script runs in that venv and calls the driver with
    the same interpreter (sys.executable). It imports nothing heavy itself.

WHAT ONE SEED IS
    Stage 2 (DP, eps=1, delta=1e-5) on all 21,523 rows, starting from the ONE shared
    Stage 1 checkpoint (Airline, non-DP, trained once by `dp2stage_driver.py stage1`),
    then 21,523 rows sampled from the fitted model. The seed sets BOTH Stage 2's
    training seed and the generation seed, so the five runs differ in Stage 2 only.

Outputs:
    synthetic_data/runs/{method}_seed{seed}.csv                 one file per seed
    evaluation/results/{method}/generation_info_seed{seed}.json spent epsilon, sigma,
                                                                 delta, steps, timings,
                                                                 acceptance, md5 (tracked:
                                                                 the formal-epsilon record)
    sdg/generate_dp2stage_log.txt                               start/end/elapsed, errors

    {method} is "dp2stageo" for variant O and "dp2stageu" for variant U;
    run_path() in evaluation/eval_{fidelity,utility}.py resolves both.

VARIANTS (--variant, default O)
    O  Stage 1 on public Airline data; the guarantee is exactly (eps, delta).
    U  Stage 1 on uniform pseudo data drawn from the PRIVATE training set's own ranges
       and categories, with no noise -- the paper's DP-2Stage-U, whose guarantee the
       paper itself says is weakened (Sec. 6). Built ONCE from the full training set
       (`dp2stage_driver.py stage1 --variant U --train-csv data/adult_train.csv`) and
       shared by all 5 seeds, the same pattern as O. Everything in Stage 2 and
       sampling is identical between the variants.

REFUSES TO PUBLISH A FIT THAT OVERSPENT: the driver reports the epsilon Opacus's RDP
accountant says was spent; a seed whose spent epsilon exceeds the 1.0 target by more
than Opacus's solver tolerance (0.01) is an invalid DP output and is not written.

DISK: no checkpoints. The driver's per-call scratch dir is deleted on exit (and the
per-seed dir here after each seed, success or failure); Stage 1's 0.5 GB checkpoint is
the only large file and lives outside this repo.

GPU: this script does not select a GPU. One process per GPU, e.g.
  CUDA_VISIBLE_DEVICES=0 python -u sdg/generate_dp2stage.py --seeds 100 102 104 >> ~/dp2_gpu0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -u sdg/generate_dp2stage.py --seeds 101 103     >> ~/dp2_gpu1.log 2>&1 &
  wait
Check nvidia-smi first: czha4500's job has filled both cards before.

Resumable: an existing {method}_seed{seed}.csv is reused and skipped (--regenerate to
re-fit). Exit code 0 only if every requested seed has an output file, else 1.

After all five exist, ALWAYS md5 them (this script also does, and logs an ERROR on any
duplicate) and check the eval summary's std is not 0 -- the GReaT seeding bug made all
five byte-identical once.

No PROBE-DEPENDENT settings remain in this file (FULL_GEN_SAMPLE_BATCH was settled by the
seed-100 run, see its comment). Grep the repo for PROBE-DEPENDENT for the rest.

Run from repo root (in ~/venv-dp2stage):
  python sdg/generate_dp2stage.py                 # all 5 seeds, sequential
  python sdg/generate_dp2stage.py --seed 100      # one seed only
  python sdg/generate_dp2stage.py --seeds 100 101 # a subset, sequential
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SDG_DIR = REPO_ROOT / "sdg"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDG_DIR))

from seeds import RUN_SEEDS                                                # noqa: E402
from dp2stage_driver import (EPSILON, DELTA, INTEGER_COLS, STAGE1_DIRS,    # noqa: E402
                             EXIT_HARD_FAILURE, EXIT_SHORTFALL, DP2STAGE_COMMIT)

DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
RUNS_DIR = REPO_ROOT / "synthetic_data" / "runs"
INFO_ROOT = REPO_ROOT / "evaluation" / "results"
METHOD = {"O": "dp2stageo", "U": "dp2stageu"}   # eval method key, file prefix, info dir
LABEL = {"O": "DP-2Stage-O", "U": "DP-2Stage-U"}
DRIVER = SDG_DIR / "dp2stage_driver.py"
EXPECTED_TRAIN_N = 21_523

# Opacus's make_private_with_epsilon solves for sigma to within this of the target.
EPS_TOLERANCE = 0.01

# Rows drawn per model.generate() call when sampling the FULL 21,523 rows. 100 is
# DP-2Stage's own default. SETTLED by the variant-O seed-100 run (2026-09-26): sampling
# took 114 s of a 3,259 s seed (3.5%; the DP fit is the other 96%), at 84% acceptance and
# 3.6 GB peak, so a larger batch could save at most ~2 min per seed. Unlike GReaT, where
# sampling dominated and k=2000 halved wall clock, there is nothing to win here.
FULL_GEN_SAMPLE_BATCH = 100

log = logging.getLogger("generate_dp2stage")


def _configure_logging() -> None:
    """Called from main(), not at import time (same reason as generate_great.py)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(SDG_DIR / "generate_dp2stage_log.txt"),
                  logging.StreamHandler()],
    )


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_train_df() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    assert len(df) == EXPECTED_TRAIN_N, f"train is {len(df)}, expected {EXPECTED_TRAIN_N}"
    return df


def generate_one(train_df: pd.DataFrame, seed: int, regenerate: bool,
                 sample_batch: int, stage1_dir: str, variant: str) -> bool:
    """DP Stage 2 + sample for one seed. Returns True if a fit actually happened."""
    method = METHOD[variant]
    path = RUNS_DIR / f"{method}_seed{seed}.csv"
    if path.exists() and not regenerate:
        log.info(f"  cached at {path.relative_to(REPO_ROOT)}, skipping (--regenerate to re-fit)")
        return False

    # gitignored (.dp2stage_fit_*); O keeps its original name
    scratch = SDG_DIR / (f".dp2stage_fit_seed{seed}" if variant == "O"
                         else f".dp2stage_fit_{variant}_seed{seed}")
    scratch.mkdir(exist_ok=True)
    out_csv, info_json = scratch / "synth.csv", scratch / "info.json"

    t_start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        cmd = [sys.executable, "-u", str(DRIVER), "fit-sample",
               "--train-csv", str(TRAIN_CSV), "--out-csv", str(out_csv),
               "--info-json", str(info_json), "--n-samples", str(len(train_df)),
               "--seed", str(seed), "--gen-seed", str(seed),
               "--sample-batch", str(sample_batch), "--stage1-dir", stage1_dir,
               "--variant", variant,
               "--scratch-root", str(scratch)]
        rc = subprocess.call(cmd)                 # inherits the GPU pin and stdout
        if rc != 0:
            meaning = {EXIT_HARD_FAILURE: "hard failure -- CUDA OOM or a zero-row generation "
                                          "round; check nvidia-smi for another job, then re-run "
                                          "(this seed is resumable)",
                       EXIT_SHORTFALL: "quota not filled after the driver's generation rounds "
                                       "(model problem, not contention)"}.get(rc, "unexpected failure")
            raise RuntimeError(f"driver exited {rc}: {meaning}")

        info = json.loads(info_json.read_text())
        if info["epsilon_spent"] > EPSILON + EPS_TOLERANCE:
            raise RuntimeError(f"seed {seed}: spent epsilon {info['epsilon_spent']:.4f} exceeds "
                               f"the {EPSILON} target -- invalid DP output, NOT written")

        synthetic = pd.read_csv(out_csv)
        assert len(synthetic) == len(train_df), \
            f"seed {seed}: generated {len(synthetic)} rows, expected {len(train_df)}"
        assert list(synthetic.columns) == list(train_df.columns), \
            f"seed {seed}: columns {list(synthetic.columns)} != {list(train_df.columns)}"
        assert all(str(t) == "int64" for t in synthetic[INTEGER_COLS].dtypes), \
            f"seed {seed}: integer columns are not int64 (float text would have leaked through)"

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_csv), str(path))
        elapsed_s = time.perf_counter() - t0
        info.update(seed=seed, run_csv=str(path.relative_to(REPO_ROOT)),
                    md5=_md5(path), wall_clock_s=round(elapsed_s, 1),
                    started=t_start_wall, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    variant=LABEL[variant], stage2_rows=len(train_df),
                    delta_note=f"delta={DELTA} (DP-2Stage's own); differs from AIM 1e-9, "
                               f"DP-CTGAN 1/(n*sqrt(n)), DPGAN 1/n -- limitation")
        info_dir = INFO_ROOT / method
        info_dir.mkdir(parents=True, exist_ok=True)
        (info_dir / f"generation_info_seed{seed}.json").write_text(json.dumps(info, indent=2))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log.info(f"  seed {seed}: start={t_start_wall} end={time.strftime('%Y-%m-%d %H:%M:%S')} "
             f"elapsed={elapsed_s:.1f}s ({elapsed_s / 3600:.2f}h) "
             f"eps_spent={info['epsilon_spent']:.4f} sigma={info['noise_multiplier']:.3f} "
             f"acceptance={info['acceptance']} -> {path.relative_to(REPO_ROOT)}")
    return True


def check_distinct(method: str) -> None:
    """md5 every output that exists; duplicates mean a seed is being overridden."""
    md5s = {s: _md5(RUNS_DIR / f"{method}_seed{s}.csv") for s in RUN_SEEDS
            if (RUNS_DIR / f"{method}_seed{s}.csv").exists()}
    for s, m in md5s.items():
        log.info(f"  md5 {method}_seed{s}.csv  {m}")
    if len(set(md5s.values())) < len(md5s):
        log.error("DUPLICATE MD5s among the seed outputs -- a seed is being silently "
                  "overridden (the GReaT bug's shape). Do not evaluate these.")
    elif len(md5s) > 1:
        log.info(f"  all {len(md5s)} existing seed outputs are distinct")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="run a single seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="run a subset of seeds, sequentially")
    parser.add_argument("--regenerate", action="store_true",
                        help="re-fit even when a cached run CSV exists")
    parser.add_argument("--sample-batch", type=int, default=FULL_GEN_SAMPLE_BATCH,
                        help=f"rows per model.generate() call (default {FULL_GEN_SAMPLE_BATCH})")
    parser.add_argument("--variant", choices=["O", "U"], default="O",
                        help="O: Airline Stage 1 (default). U: uniform pseudo data from the "
                             "private data's stats (weaker guarantee, see docstring)")
    parser.add_argument("--stage1-dir", default=None,
                        help="the shared Stage 1 checkpoint (default: the driver's STAGE1_DIRS[variant])")
    args = parser.parse_args()
    _configure_logging()

    if args.seed is not None and args.seeds is not None:
        parser.error("pass either --seed or --seeds, not both")
    seeds = [args.seed] if args.seed is not None else (args.seeds or RUN_SEEDS)
    unknown = set(seeds) - set(RUN_SEEDS)
    if unknown:
        parser.error(f"unknown seed(s): {sorted(unknown)}. Known: {RUN_SEEDS}")

    stage1_dir = args.stage1_dir or str(STAGE1_DIRS[args.variant])
    log.info(f"=== generate_dp2stage: seeds={seeds}, {LABEL[args.variant]}, eps={EPSILON}, delta={DELTA}, "
             f"sample_batch={args.sample_batch}, commit={DP2STAGE_COMMIT[:7]}, "
             f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} ===")

    train_df = load_train_df()
    log.info(f"Loaded training split: {train_df.shape}")

    fitted = 0
    for seed in seeds:
        log.info(f"--- seed {seed} ---")
        try:
            fitted += generate_one(train_df, seed, args.regenerate,
                                   args.sample_batch, stage1_dir, args.variant)
        except Exception:
            log.error(f"  seed {seed} FAILED:\n{traceback.format_exc()}")

    missing = [s for s in seeds
               if not (RUNS_DIR / f"{METHOD[args.variant]}_seed{s}.csv").exists()]
    log.info(f"=== Done: {fitted} run(s) generated. "
             f"Synthetic data in {RUNS_DIR.relative_to(REPO_ROOT)}/ ===")
    check_distinct(METHOD[args.variant])
    if missing:
        log.error(f"missing outputs for seeds {missing} -- re-run to resume them")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
