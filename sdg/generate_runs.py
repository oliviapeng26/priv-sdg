#!/usr/bin/env python3
"""Seeded, repeated synthetic data generation for the four Synthcity methods.

Replaces the per-method sdg/*.ipynb notebooks (now *_LEGACY.ipynb). Same
plugins, same hyperparameters -- what is new is that each method is fitted and
sampled once per seed in seeds.RUN_SEEDS, so downstream fidelity and utility
get mean +/- std instead of a single draw, and every draw is reproducible.

Outputs:
    synthetic_data/runs/{method}_seed{seed}.csv   one file per (method, run)
    results/computational_cost.csv                wall clock + peak memory per run

This script does GENERATION ONLY. Scoring lives in evaluation/eval_fidelity.py
and evaluation/eval_utility.py, both of which read the CSVs written here. The
split is not just tidiness: generation is the expensive GPU-bound half and
evaluation is cheap and CPU-bound, so they belong in separate processes and
separate machines. It also means this script never imports xgboost, and the
evaluation scripts never import torch -- which is exactly the OpenMP collision
the README documents a workaround for.

DEVICE
    Statistical generators (bayesian_network, privbayes) have no device
    parameter at all -- pgmpy/numpy, CPU only, no GPU path exists. The neural
    ones (ctgan, dpgan) take `device` and get the detected CUDA device; note
    synthcity's own default is already `DEVICE` (cuda when available), but it
    is passed explicitly here so the value recorded in the cost CSV is the one
    actually used rather than an assumption.

    So on the GPU workstation a full run legitimately reports a mix of cpu and
    cuda rows. That is the intended setup, not a misconfiguration: the neural
    methods are on GPU purely for the speedup, and the statistical ones are
    CPU-bound by their libraries. The `device` column records which, so the
    write-up can state it.

PEAK MEMORY CAVEAT
    tracemalloc sees Python-level allocations only. It does not see torch CUDA
    buffers or memory allocated inside C extensions, so peak_memory_mb
    understates the real footprint for ctgan/dpgan (torch), and is only
    comparable in a like-for-like sense within the statistical methods. Wall clock is the trustworthy cross-method column.
    This is the same caveat the legacy sdg/computational_overhead_LEGACY.csv carried.

Run from repo root:
  python sdg/generate_runs.py                          # all 4 methods x 5 seeds
  python sdg/generate_runs.py bayesian_network         # one method
  python sdg/generate_runs.py ctgan dpgan              # the GPU half
  python sdg/generate_runs.py privbayes --runs 2       # partial
  python sdg/generate_runs.py ctgan --regenerate       # ignore cached CSVs

THE SMARTNOISE METHODS LIVE IN sdg/generate_smartnoise.py
    aim used to be a fifth method here. It moved out when dpctgan joined it: both
    are smartnoise-synth rather than Synthcity plugins, neither goes through
    Plugin.fit/generate, and their outputs are keyed by epsilon
    (synthetic_data/smartnoise/{method}/eps{eps}/seed{seed}.csv) so a budget sweep
    cannot overwrite an arm. evaluation/eval_{fidelity,utility}.py score all six
    methods and know which tree each one lives in.

Resumable: an existing synthetic_data/runs/{method}_seed{seed}.csv is reused and
the run skipped, so an interrupted session picks up where it stopped. --regenerate
forces a re-fit (and overwrites that run's cost row).
"""

import argparse
import logging
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SDG_DIR = REPO_ROOT / "sdg"
sys.path.insert(0, str(REPO_ROOT))

from seeds import RUN_SEEDS, NUM_RUNS, set_all_seeds

FORMAL_EPSILON = 1.0      # DP budget, unchanged from the legacy notebooks

# method -> synthcity plugin name
METHOD_SPEC = {
    "bayesian_network": "bayesian_network",
    "privbayes":        "privbayes",
    "ctgan":            "ctgan",
    "dpgan":            "dpgan",
}
ALL_METHODS = list(METHOD_SPEC)
DP_METHODS = {"privbayes", "dpgan"}       # take an `epsilon` kwarg
NEURAL_METHODS = {"ctgan", "dpgan"}       # take a `device` kwarg

# Both GANs are capped here, NOT left at the synthcity default of 2000.
#
# MUST MATCH benchmark_tapas/config.py's CTGAN_N_ITER / DPGAN_N_ITER. That file
# controls the generators TAPAS re-fits during the privacy audit; this one controls
# the generators we measure utility and fidelity on. If the two drift apart we are
# back to reporting utility for one model and privacy for another -- the exact
# confound this cap removes. (Deliberately duplicated rather than imported: sdg/
# should not depend on the privacy experiment's config.)
#
# Justification for the value lives in benchmark_tapas/config.py. In short: CTGAN's
# utility has converged by 50, and for DPGAN n_iter is a parameter of the DP
# mechanism rather than a training length, with all tested caps statistically
# indistinguishable -- so 50 is chosen as the cheapest.
GAN_N_ITER = 50

CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]
TARGET_COL = "income"

DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
RUNS_DIR = REPO_ROOT / "synthetic_data" / "runs"
RESULTS_DIR = REPO_ROOT / "results"
COST_CSV = RESULTS_DIR / "computational_cost.csv"

EXPECTED_TRAIN_N = 21_523

RESULTS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SDG_DIR / "generate_runs_log.txt"), logging.StreamHandler()],
)
log = logging.getLogger("generate_runs")


def load_train_df() -> pd.DataFrame:
    """The 21,523-row training split -- the only data any generator ever sees.

    Row count is asserted rather than assumed; the full check that these CSVs
    really are the DATA_SPLIT_SEED partition lives in data_preprocessing.ipynb
    and is repeated in evaluation/eval_utility.py, where a stale split would
    silently corrupt TSTR.
    """
    df = pd.read_csv(TRAIN_CSV)
    assert len(df) == EXPECTED_TRAIN_N, f"train is {len(df)}, expected {EXPECTED_TRAIN_N}"
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def device_for(method: str) -> str:
    """The device this method will actually use -- recorded, not assumed."""
    if method in NEURAL_METHODS:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return "cpu"      # bayesian_network / privbayes: no GPU path exists


def generate_synthcity(method: str, train_data: pd.DataFrame, seed: int,
                       device: str) -> pd.DataFrame:
    """Fit + sample one Synthcity plugin at a fixed seed.

    Hyperparameters are untouched from the legacy notebooks: plugin defaults
    throughout, plus epsilon=1.0 for the DP methods. `random_state` is not a
    hyperparameter change -- it selects which draw we get, not how the model is
    configured. It is consumed by Plugin.fit and passed again to
    Plugin.generate; both route through enable_reproducible_results, which
    seeds numpy, torch and random.
    """
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    df = train_data.copy()
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

    kwargs = {"random_state": seed}
    if method in DP_METHODS:
        kwargs["epsilon"] = FORMAL_EPSILON
    if method in NEURAL_METHODS:
        kwargs["device"] = device
        kwargs["n_iter"] = GAN_N_ITER

    log.info(f"  plugin={METHOD_SPEC[method]}, kwargs={kwargs}")
    plugin = Plugins().get(METHOD_SPEC[method], **kwargs)
    plugin.fit(GenericDataLoader(df, target_column=TARGET_COL))
    return plugin.generate(count=len(df), random_state=seed).dataframe()


def record_cost(row: dict) -> None:
    """Upsert one (method, seed, stage) row into results/computational_cost.csv.

    Keyed on stage as well as method+seed -- the eval scripts write fidelity and
    utility rows into this same file, so matching on method+seed alone would
    delete their rows every time a generation run was repeated.
    """
    df = pd.DataFrame([row])
    if COST_CSV.exists():
        existing = pd.read_csv(COST_CSV)
        if {"method", "seed", "stage"}.issubset(existing.columns):
            mask = ~((existing["method"] == row["method"])
                     & (existing["seed"] == row["seed"])
                     & (existing["stage"] == row["stage"]))
            existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(COST_CSV, index=False)


def generate_one(method: str, train_data: pd.DataFrame, seed: int,
                 regenerate: bool) -> bool:
    """Generate (or reuse) one run. Returns True if a fit actually happened."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{method}_seed{seed}.csv"

    if path.exists() and not regenerate:
        log.info(f"  cached at {path.relative_to(REPO_ROOT)}, skipping "
                 f"(--regenerate to re-fit)")
        return False

    device = device_for(method)
    set_all_seeds(seed)

    # Timed region covers fit + generate together, matching what the legacy
    # notebooks measured, so the numbers stay comparable to
    # sdg/computational_overhead_LEGACY.csv.
    tracemalloc.start()
    t0 = time.perf_counter()
    synthetic = generate_synthcity(method, train_data, seed, device)
    wall_clock_s = round(time.perf_counter() - t0, 2)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = round(peak_bytes / 1e6, 2)

    synthetic = synthetic[train_data.columns.tolist()]
    assert len(synthetic) == len(train_data), \
        f"{method} seed {seed}: generated {len(synthetic)} rows, expected {len(train_data)}"
    assert not synthetic.isna().any().any(), f"{method} seed {seed}: NaNs in output"
    synthetic.to_csv(path, index=False)

    record_cost({
        "method": method, "seed": seed, "stage": "generation",
        "wall_clock_s": wall_clock_s, "peak_memory_mb": peak_memory_mb,
        "device": device,
    })
    log.info(f"  {len(synthetic)} rows | {wall_clock_s}s | {peak_memory_mb} MB peak "
             f"| device={device} -> {path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", default=None,
                        help=f"methods to generate (default: all of {ALL_METHODS})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"how many of the {NUM_RUNS} run seeds to use (default: all)")
    parser.add_argument("--regenerate", action="store_true",
                        help="re-fit even when a cached run CSV exists")
    args = parser.parse_args()

    methods = args.methods or ALL_METHODS
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        parser.error(f"unknown method(s): {sorted(unknown)}. Known: {ALL_METHODS}")
    run_seeds = RUN_SEEDS[:args.runs]

    log.info(f"=== generate_runs: methods={methods}, seeds={run_seeds} ===")
    log.info("devices: " + ", ".join(f"{m}={device_for(m)}" for m in methods))

    train_df = load_train_df()
    log.info(f"Loaded training split: {train_df.shape}")

    # Warm up the heavy imports BEFORE any timed region. Without this the first
    # run of a session pays synthcity's import cost inside its timed block --
    # measured at 40.1s vs ~7.3s for the other four BN runs, a 5x inflation that
    # lands entirely on run 0 and corrupts both the mean and the std.
    t0 = time.perf_counter()
    from synthcity.plugins import Plugins
    # Building the registry AND instantiating each plugin once: both the module
    # import and the first Plugins().get() carry one-off cost (plugin discovery,
    # pgmpy/torch lazy imports) that would otherwise land inside run 0's timed block.
    for m in methods:
        Plugins().get(METHOD_SPEC[m])
    log.info(f"synthcity warmed up in {time.perf_counter() - t0:.1f}s (untimed)")

    fitted = 0
    for method in methods:
        log.info(f"--- {method} ---")
        for run_idx, seed in enumerate(run_seeds):
            log.info(f"  run {run_idx}/{len(run_seeds) - 1}, seed {seed}")
            try:
                fitted += generate_one(method, train_df, seed, args.regenerate)
            except Exception:
                log.error(f"  {method} seed {seed} FAILED:\n{traceback.format_exc()}")

    log.info(f"=== Done: {fitted} run(s) generated. "
             f"Synthetic data in {RUNS_DIR.relative_to(REPO_ROOT)}/ ===")
    if COST_CSV.exists():
        cost = pd.read_csv(COST_CSV)
        print("\n=== computational cost (mean over runs) ===")
        summary = cost.groupby(["method", "device"], sort=False).agg(
            n_runs=("seed", "count"),
            wall_clock_s_mean=("wall_clock_s", "mean"),
            wall_clock_s_std=("wall_clock_s", "std"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
        ).reset_index()
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
