#!/usr/bin/env python3
"""Seeded GReaT (be-great) generation on the full 21,523-row adult_train.csv.

Companion to sdg/generate_runs.py's four Synthcity methods -- GReaT is a
separate LLM-based generator (fine-tuned GPT-2) with its own dependency
(be-great) and its own, much longer, per-seed cost, so it gets its own script
rather than joining METHOD_SPEC there.

Hyperparameters (gpt2, batch_size=32, epochs=10, fp16=True,
dataloader_num_workers=4, sample k=2000) match the 500-row sanity check in
sdg/great_sanity_check.py -- same config, more rows -- with ONE deliberate
difference: float_precision=0 here (integer text, `age is 47`), which
great_sanity_check.py does not set (float text, `age is 47.0`). gpt2 (not
distilgpt2) because it sampled ~20x faster and matched DP-2Stage's reference
setup; see the README's GReaT timing table for the sanity-check numbers behind
this.

float_precision=0 is what the TAPAS audit's wrapper (benchmark_tapas/
tapas_wrappers/great_generator.py) uses, so the model scored for utility and
fidelity is serialised the same way as the model audited for privacy. The
earlier runs used be_great's default (float_precision=None), which writes the
CSV's floats as `47.0`: measured with GPT-2's tokenizer, that makes a row
80-105 tokens against max_length=100 (integer text: 70-96), and once prompt
left-padding is counted up to 25.8% of rows overflow when native_country is
the start column. Those outputs were replaced.

Outputs:
    synthetic_data/runs/great_seed{seed}.csv   one file per seed
    sdg/generate_great_log.txt                 start/end/elapsed per seed, errors

DISK: no model checkpoints are written. save_strategy/logging_strategy="no"
and report_to=[] disable HF Trainer's own checkpoint and tensorboard writes;
the per-seed scratch experiment_dir is removed after each run whether it
succeeds or fails. The only persistent output is one CSV per seed (21,523
rows, a few MB each -- five seeds total stays well under 1 GB).

GPU: this script does not select a GPU itself. Run one process per GPU with
CUDA_VISIBLE_DEVICES, e.g.:
  CUDA_VISIBLE_DEVICES=0 python sdg/generate_great.py --seeds 100 102 104
  CUDA_VISIBLE_DEVICES=1 python sdg/generate_great.py --seeds 101 103
Each process works through its own seed list sequentially, so once a GPU
finishes its seeds it's simply done -- no manual relaunching needed.

Resumable: an existing synthetic_data/runs/great_seed{seed}.csv is reused and
skipped, so re-running after a crash or overnight interruption picks up where
it stopped. --regenerate forces a re-fit.

Errors on one seed are logged and the script moves to the next seed rather
than stopping the whole batch.

Run from repo root:
  python sdg/generate_great.py                 # all 5 seeds, sequential
  python sdg/generate_great.py --seed 100       # one seed only
  python sdg/generate_great.py --seeds 100 101  # a subset, sequential
"""

import argparse
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SDG_DIR = REPO_ROOT / "sdg"
sys.path.insert(0, str(REPO_ROOT))

from seeds import RUN_SEEDS, set_all_seeds

DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
RUNS_DIR = REPO_ROOT / "synthetic_data" / "runs"
EXPECTED_TRAIN_N = 21_523

LLM = "gpt2"
BATCH_SIZE = 32
EPOCHS = 10
FP16 = True
DATALOADER_NUM_WORKERS = 4
FLOAT_PRECISION = 0   # integer text ("age is 47"); see the module docstring
# efficient_finetuning intentionally left unset below -- full fine-tuning,
# no LoRA, to match DP-2Stage's GPT-2 setup exactly.

# GReaT.sample()'s own generation batch size (separate from fit's BATCH_SIZE
# above). be_great's default k=100 left the GPU ~0% utilized during sampling
# -- confirmed via sdg/great_sanity_check.py, where k=2000 roughly halved
# sample() wall clock (581s -> 291s for 500 rows) with no change to fit() or
# output quality. See that script's history for the diagnosis.
SAMPLE_K = 2000

log = logging.getLogger("generate_great")


def _configure_logging() -> None:
    """Called from main(), NOT at import time.

    benchmark_tapas/tapas_wrappers/great_generator.py imports the hyperparameter
    constants above so the audited model cannot drift from the one these runs
    produce. A module-level basicConfig would fire on that import and attach this
    file's FileHandler to the root logger before the audit configures its own, so
    every audit log line would land in generate_great_log.txt. sdg/aim.py is
    import-safe for the same reason (aim_generator.py imports its BIN_EDGES).
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(SDG_DIR / "generate_great_log.txt"),
                  logging.StreamHandler()],
    )


def load_train_df() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    assert len(df) == EXPECTED_TRAIN_N, f"train is {len(df)}, expected {EXPECTED_TRAIN_N}"
    return df


def generate_one(train_df: pd.DataFrame, seed: int, regenerate: bool) -> bool:
    """Fine-tune + sample GReaT for one seed. Returns True if a fit actually happened."""
    path = RUNS_DIR / f"great_seed{seed}.csv"
    if path.exists() and not regenerate:
        log.info(f"  cached at {path.relative_to(REPO_ROOT)}, skipping (--regenerate to re-fit)")
        return False

    from be_great import GReaT

    scratch_dir = SDG_DIR / f".great_scratch_seed{seed}"
    set_all_seeds(seed)

    t_start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        model = GReaT(
            llm=LLM,
            experiment_dir=str(scratch_dir),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            fp16=FP16,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            float_precision=FLOAT_PRECISION,
            seed=seed,   # HF's Trainer defaults TrainingArguments.seed=42 and
                         # calls set_seed(42) internally, silently overwriting
                         # the set_all_seeds(seed) above -- without this, every
                         # one of the 5 "seeded" runs trains and samples from
                         # the same internal state. Confirmed: all 5 committed
                         # great_seed*.csv were byte-identical (same MD5) before
                         # this fix.
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
        )
        model.fit(train_df)
        synthetic = model.sample(n_samples=len(train_df), k=SAMPLE_K)
        assert len(synthetic) == len(train_df), \
            f"seed {seed}: generated {len(synthetic)} rows, expected {len(train_df)}"
        synthetic.to_csv(path, index=False)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    elapsed_s = time.perf_counter() - t0
    t_end_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"  seed {seed}: start={t_start_wall} end={t_end_wall} "
             f"elapsed={elapsed_s:.1f}s ({elapsed_s / 3600:.2f}h) -> "
             f"{path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="run a single seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="run a subset of seeds, sequentially")
    parser.add_argument("--regenerate", action="store_true",
                        help="re-fit even when a cached run CSV exists")
    args = parser.parse_args()
    _configure_logging()

    if args.seed is not None and args.seeds is not None:
        parser.error("pass either --seed or --seeds, not both")
    seeds = [args.seed] if args.seed is not None else (args.seeds or RUN_SEEDS)
    unknown = set(seeds) - set(RUN_SEEDS)
    if unknown:
        parser.error(f"unknown seed(s): {sorted(unknown)}. Known: {RUN_SEEDS}")

    log.info(f"=== generate_great: seeds={seeds}, llm={LLM}, epochs={EPOCHS}, "
             f"batch_size={BATCH_SIZE}, fp16={FP16} ===")

    train_df = load_train_df()
    log.info(f"Loaded training split: {train_df.shape}")

    fitted = 0
    for seed in seeds:
        log.info(f"--- seed {seed} ---")
        try:
            fitted += generate_one(train_df, seed, args.regenerate)
        except Exception:
            log.error(f"  seed {seed} FAILED:\n{traceback.format_exc()}")

    log.info(f"=== Done: {fitted} run(s) generated. "
             f"Synthetic data in {RUNS_DIR.relative_to(REPO_ROOT)}/ ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
