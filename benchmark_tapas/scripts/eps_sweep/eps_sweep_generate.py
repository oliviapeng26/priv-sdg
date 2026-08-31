#!/usr/bin/env python3
"""Phase 1a of the formal-epsilon sweep: generate DPGAN data at each eps.

GENERATION ONLY. Scoring is eps_sweep_evaluate.py, deliberately a separate
process -- this script imports torch and never xgboost, exactly as
sdg/generate_runs.py does, so the two OpenMP runtimes never meet (README).

WHY THIS EXISTS INSTEAD OF sdg/generate_runs.py
    Two blockers, neither worth patching into a script the whole repo depends on:

    1. generate_runs.py hardcodes FORMAL_EPSILON = 1.0 at module scope (line 74)
       and exposes no --epsilon flag.
    2. Its outputs are not keyed by eps. It writes
       synthetic_data/runs/{method}_seed{seed}.csv and the eval scripts read that
       same path, so four eps values would silently overwrite each other and the
       last one written would be scored as all four.

    So the outputs here are namespaced:
        synthetic_data/eps_sweep/eps{eps}/dpgan_seed{seed}.csv

    Everything else is held identical to generate_runs.py on purpose -- same
    plugin, same n_iter, same per-seed random_state, same fit+generate timing
    region, same row-count and NaN asserts -- so the only difference between an
    eps arm here and the repo's existing dpgan numbers is eps itself.

EPS = 1.0 IS NOT REGENERATED
    That arm already exists at synthetic_data/runs/dpgan_seed{100..104}.csv,
    generated 2026-08-24 under the same configuration, and its utility/fidelity
    are already in results/{utility,fidelity}_summary.csv. Re-fitting it would
    produce the same draws and buy nothing, so 1.0 is excluded from the default
    eps list. eps_sweep_evaluate.py picks that arm up from synthetic_data/runs/.

WHAT VARIES AND WHAT DOES NOT
    varies:      epsilon (hence the Opacus noise multiplier sigma; see
                 eps_sweep_sigma_check.py for the sigma each eps buys)
    fixed:       n_iter=DPGAN_N_ITER, batch_size, every other plugin default,
                 the 21,523-row training split, RUN_SEEDS = [100..104], device

COST
    ~70-80 s per (eps, seed) on the workstation GPU, from
    results/computational_cost.csv -- so ~6 min per eps, ~20 min for all three.
    Negligible next to the ~13 h privacy half.

Outputs:
    synthetic_data/eps_sweep/eps{eps}/dpgan_seed{seed}.csv
    benchmark_tapas/results/eps_sweep/generation_cost.csv   (upserted per eps+seed)

Run from the repo root, venv active:
  python benchmark_tapas/scripts/eps_sweep_generate.py
  python benchmark_tapas/scripts/eps_sweep_generate.py --epsilons 10
  python benchmark_tapas/scripts/eps_sweep_generate.py --runs 2 --regenerate

Resumable: an existing run CSV is reused and the fit skipped, so an interrupted
session picks up where it stopped. --regenerate forces a re-fit.
"""

import argparse
import logging
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

import pandas as pd

# benchmark_tapas/, found by walking up to config.py rather than counting parents --
# these scripts live in a subfolder now and may move again.
BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import (RESULTS_DIR, TRAIN_CSV, DPGAN_N_ITER,        # noqa: E402
                    CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL)
from seeds import RUN_SEEDS, NUM_RUNS, set_all_seeds            # noqa: E402

METHOD = "dpgan"
# 1.0 is reused from the existing generation run (see module docstring).
SWEEP_EPSILONS = [0.1, 10.0, 100.0]
EXPECTED_TRAIN_N = 21_523

SWEEP_DIR = RESULTS_DIR / "eps_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_ROOT = REPO_ROOT / "synthetic_data" / "eps_sweep"
COST_CSV = SWEEP_DIR / "generation_cost.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SWEEP_DIR / "eps_sweep_generate_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("eps_sweep_generate")


def eps_slug(eps: float) -> str:
    """Path-safe, round-trippable label for one budget: 0.1 -> 'eps0.1', 10.0 -> 'eps10'.

    %g rather than str() so 10.0 and 10 land in the same directory instead of
    silently splitting one arm across 'eps10' and 'eps10.0'.
    """
    return f"eps{eps:g}"


def load_train_df() -> pd.DataFrame:
    """The 21,523-row training split -- the only data any generator ever sees.

    Same assert as sdg/generate_runs.py: the full check that these CSVs really are
    the DATA_SPLIT_SEED partition lives in data_preprocessing.ipynb and is repeated
    in evaluation/eval_utility.py, which is where a stale split would do real damage.
    """
    df = pd.read_csv(TRAIN_CSV)
    assert len(df) == EXPECTED_TRAIN_N, f"train is {len(df)}, expected {EXPECTED_TRAIN_N}"
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def generate_dpgan(train_data: pd.DataFrame, seed: int, device: str,
                   epsilon: float) -> pd.DataFrame:
    """Fit + sample the dpgan plugin at one (seed, epsilon).

    Byte-for-byte the kwargs sdg/generate_runs.generate_synthcity builds for dpgan,
    with epsilon swapped from the module constant to the argument. random_state is
    not a hyperparameter change -- it selects which draw we get, not how the model
    is configured -- and is consumed by Plugin.fit and passed again to
    Plugin.generate, both of which route through enable_reproducible_results.
    """
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    df = train_data.copy()
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

    kwargs = {"random_state": seed, "epsilon": epsilon,
              "device": device, "n_iter": DPGAN_N_ITER}
    log.info(f"  plugin=dpgan, kwargs={kwargs}")
    plugin = Plugins().get(METHOD, **kwargs)
    plugin.fit(GenericDataLoader(df, target_column=TARGET_COL))
    return plugin.generate(count=len(df), random_state=seed).dataframe()


def record_cost(row: dict) -> None:
    """Upsert one (epsilon, seed) row into results/eps_sweep/generation_cost.csv.

    A sweep-local cost table rather than results/computational_cost.csv: that file
    is keyed on (method, seed, stage) with no eps column, so three eps arms would
    collapse onto the same three keys and overwrite both each other and the
    existing eps=1.0 dpgan rows the repo already reports.
    """
    df = pd.DataFrame([row])
    if COST_CSV.exists():
        existing = pd.read_csv(COST_CSV)
        if {"formal_epsilon", "seed", "stage"}.issubset(existing.columns):
            mask = ~((existing["formal_epsilon"] == row["formal_epsilon"])
                     & (existing["seed"] == row["seed"])
                     & (existing["stage"] == row["stage"]))
            existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(COST_CSV, index=False)


def generate_one(train_data: pd.DataFrame, epsilon: float, seed: int,
                 device: str, regenerate: bool) -> bool:
    """Generate (or reuse) one (eps, seed) run. Returns True if a fit happened."""
    out_dir = SYNTH_ROOT / eps_slug(epsilon)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{METHOD}_seed{seed}.csv"

    if path.exists() and not regenerate:
        log.info(f"  cached at {path.relative_to(REPO_ROOT)}, skipping "
                 f"(--regenerate to re-fit)")
        return False

    set_all_seeds(seed)
    tracemalloc.start()
    t0 = time.perf_counter()
    synthetic = generate_dpgan(train_data, seed, device, epsilon)
    wall_clock_s = round(time.perf_counter() - t0, 2)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    synthetic = synthetic[train_data.columns.tolist()]
    assert len(synthetic) == len(train_data), \
        f"eps={epsilon} seed={seed}: generated {len(synthetic)} rows, expected {len(train_data)}"
    assert not synthetic.isna().any().any(), f"eps={epsilon} seed={seed}: NaNs in output"
    synthetic.to_csv(path, index=False)

    record_cost({"method": METHOD, "formal_epsilon": epsilon, "seed": seed,
                 "stage": "generation", "wall_clock_s": wall_clock_s,
                 "peak_memory_mb": round(peak_bytes / 1e6, 2), "device": device,
                 "n_iter": DPGAN_N_ITER})
    log.info(f"  {len(synthetic)} rows | {wall_clock_s}s | "
             f"{round(peak_bytes / 1e6, 2)} MB peak | device={device} "
             f"-> {path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SWEEP_EPSILONS,
                        help=f"budgets to generate (default: {SWEEP_EPSILONS}; 1.0 is "
                             f"excluded because synthetic_data/runs/ already has it)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"how many of the {NUM_RUNS} run seeds to use (default: all)")
    parser.add_argument("--regenerate", action="store_true",
                        help="re-fit even when a cached run CSV exists")
    args = parser.parse_args()

    run_seeds = RUN_SEEDS[:args.runs]
    if 1.0 in args.epsilons:
        log.warning("eps=1.0 requested. synthetic_data/runs/dpgan_seed*.csv already holds "
                    "that arm under the same configuration; this will write a SECOND copy "
                    "under synthetic_data/eps_sweep/eps1/. Harmless but redundant.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"=== eps sweep generation: eps={args.epsilons}, seeds={run_seeds}, "
             f"n_iter={DPGAN_N_ITER}, device={device} "
             f"(torch.cuda.is_available()={torch.cuda.is_available()}) ===")

    train_df = load_train_df()
    log.info(f"Loaded training split: {train_df.shape}")

    # Warm up synthcity BEFORE any timed region. Without this the first run of a
    # session pays the import + plugin-discovery cost inside its timed block --
    # measured at a 5x inflation on run 0 in sdg/generate_runs.py, which corrupts
    # both the mean and the std of the cost table.
    t0 = time.perf_counter()
    from synthcity.plugins import Plugins
    Plugins().get(METHOD)
    log.info(f"synthcity warmed up in {time.perf_counter() - t0:.1f}s (untimed)")

    fitted = 0
    for epsilon in args.epsilons:
        log.info(f"--- eps={epsilon:g} ---")
        for run_idx, seed in enumerate(run_seeds):
            log.info(f"  run {run_idx}/{len(run_seeds) - 1}, seed {seed}")
            try:
                fitted += generate_one(train_df, epsilon, seed, device, args.regenerate)
            except Exception:
                log.error(f"  eps={epsilon:g} seed={seed} FAILED:\n{traceback.format_exc()}")

    log.info(f"=== Done: {fitted} run(s) generated under "
             f"{SYNTH_ROOT.relative_to(REPO_ROOT)}/ ===")
    if COST_CSV.exists():
        cost = pd.read_csv(COST_CSV)
        gen = cost[cost.stage == "generation"]
        print("\n=== generation cost (mean over seeds) ===")
        print(gen.groupby(["formal_epsilon", "device"], sort=True).agg(
            n_runs=("seed", "count"),
            wall_clock_s_mean=("wall_clock_s", "mean"),
            wall_clock_s_std=("wall_clock_s", "std"),
        ).reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
