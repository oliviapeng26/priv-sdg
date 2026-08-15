#!/usr/bin/env python3
"""SDMetrics fidelity, scored per run over the seeded generation runs.

Reads the per-run synthetic CSVs written by sdg/generate_runs.py and scores
each against the training data. Fidelity metrics and their definitions are
UNCHANGED from the single-draw evaluation_LEGACY/eval_sdmetrics_LEGACY.py -- the only
difference is that each method is now scored once per seed, so the results
carry mean +/- std rather than one number.

Fidelity is measured against the TRAINING data, deliberately, not the holdout:
the question fidelity asks is how faithfully the generator reproduced the
distribution it was fitted on. (Utility is the one that needs a holdout, and
that lives in evaluation/eval_utility.py.)

Outputs:
    results/fidelity_per_run.csv    one row per (method, run)
    results/fidelity_summary.csv    mean/std per method

Run from repo root, after sdg/generate_runs.py:
  python evaluation/eval_fidelity.py                     # every method with run CSVs
  python evaluation/eval_fidelity.py bayesian_network    # one method
  python evaluation/eval_fidelity.py --runs 2            # first 2 seeds only

Cheap and CPU-only -- it never imports torch or synthcity, so it can be re-run
freely on a laptop against run CSVs generated elsewhere.
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
sys.path.insert(0, str(REPO_ROOT))

from seeds import RUN_SEEDS, NUM_RUNS

ALL_METHODS = ["bayesian_network", "privbayes", "ctgan", "dpgan", "aim"]

CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "synthetic_data" / "runs"
RESULTS_DIR = REPO_ROOT / "results"
EVAL_DIR = REPO_ROOT / "evaluation"

PER_RUN_CSV = RESULTS_DIR / "fidelity_per_run.csv"
SUMMARY_CSV = RESULTS_DIR / "fidelity_summary.csv"
COST_CSV = RESULTS_DIR / "computational_cost.csv"   # shared with sdg/generate_runs.py

EXPECTED_TRAIN_N = 21_523
FIDELITY_METRICS = ["KSComplement", "TVComplement",
                    "CorrelationSimilarity", "ContingencySimilarity"]

# SDMetrics metadata: column sdtypes derived from our schema
METADATA = {"columns": {
    **{c: {"sdtype": "numerical"} for c in CONTINUOUS_COLS},
    **{c: {"sdtype": "categorical"} for c in CATEGORICAL_COLS},
}}

RESULTS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(EVAL_DIR / "eval_fidelity_log.txt"), logging.StreamHandler()],
)
log = logging.getLogger("eval_fidelity")


def load_train_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "adult_train.csv")
    assert len(df) == EXPECTED_TRAIN_N, f"train is {len(df)}, expected {EXPECTED_TRAIN_N}"
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def load_run(method: str, seed: int):
    """The synthetic CSV for one run, or None if it hasn't been generated."""
    path = RUNS_DIR / f"{method}_seed{seed}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def compute_fidelity(train_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict:
    """The four SDMetrics fidelity scores (higher = better), synthetic vs training.

    KSComplement   per-column KS similarity, continuous columns
    TVComplement   per-column total-variation similarity, categorical columns
    CorrelationSimilarity   pairwise Pearson correlation between numerical columns
    ContingencySimilarity   joint distribution of categorical column pairs
    """
    from sdmetrics.single_table import (
        KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity,
    )
    return {
        "KSComplement": float(KSComplement.compute(
            train_data, synthetic_data, metadata=METADATA)),
        "TVComplement": float(TVComplement.compute(
            train_data, synthetic_data, metadata=METADATA)),
        "CorrelationSimilarity": float(CorrelationSimilarity.compute(
            train_data, synthetic_data, metadata=METADATA)),
        "ContingencySimilarity": float(ContingencySimilarity.compute(
            train_data, synthetic_data, metadata=METADATA)),
    }


def record_cost(row: dict) -> None:
    """Upsert one (method, seed, stage) row into results/computational_cost.csv.

    Same file sdg/generate_runs.py writes, keyed on stage -- so one table carries
    generation, fidelity and utility cost side by side. This replaces the old
    evaluation_LEGACY/eval_overhead_LEGACY.csv, which timed whole script invocations rather
    than per-method work and only covered the superseded eval scripts.
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


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean and std of each metric across runs."""
    rows = []
    for method, g in per_run.groupby("method", sort=False):
        row = {"method": method, "n_runs": len(g)}
        for col in FIDELITY_METRICS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", default=None,
                        help=f"methods to score (default: all of {ALL_METHODS})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"how many of the {NUM_RUNS} run seeds to score (default: all)")
    args = parser.parse_args()

    methods = args.methods or ALL_METHODS
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        parser.error(f"unknown method(s): {sorted(unknown)}. Known: {ALL_METHODS}")
    run_seeds = RUN_SEEDS[:args.runs]

    log.info(f"=== eval_fidelity: methods={methods}, seeds={run_seeds} ===")
    train_df = load_train_df()
    log.info(f"Scoring against training data {train_df.shape}")

    # Warm up SDMetrics before any timed region -- otherwise run 0 pays the
    # import inside its timed block (measured 4.8s vs 0.57s for the rest).
    import sdmetrics.single_table  # noqa: F401

    rows, missing = [], []
    for method in methods:
        log.info(f"--- {method} ---")
        for run_idx, seed in enumerate(run_seeds):
            synthetic = load_run(method, seed)
            if synthetic is None:
                missing.append(f"{method}_seed{seed}")
                continue
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                fidelity = compute_fidelity(train_df, synthetic)
                wall_clock_s = round(time.perf_counter() - t0, 2)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                rows.append({"method": method, "run_idx": run_idx, "seed": seed, **fidelity})
                record_cost({"method": method, "seed": seed, "stage": "fidelity",
                             "wall_clock_s": wall_clock_s,
                             "peak_memory_mb": round(peak_bytes / 1e6, 2),
                             "device": "cpu"})   # SDMetrics is numpy/pandas, CPU only
                log.info(f"  seed {seed}: " +
                         ", ".join(f"{k}={v:.4f}" for k, v in fidelity.items()) +
                         f"  ({wall_clock_s}s)")
            except Exception:
                log.error(f"  {method} seed {seed} FAILED:\n{traceback.format_exc()}")

    if missing:
        log.warning(f"{len(missing)} run(s) not generated yet, skipped: "
                    f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if not rows:
        log.warning("No runs scored -- run sdg/generate_runs.py first.")
        return 1

    per_run = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    per_run = per_run.sort_values(
        ["method", "run_idx"], key=lambda s: s.map(order) if s.name == "method" else s
    ).reset_index(drop=True)
    per_run.to_csv(PER_RUN_CSV, index=False)

    summary = summarise(per_run)
    summary.to_csv(SUMMARY_CSV, index=False)
    log.info(f"Wrote {PER_RUN_CSV.relative_to(REPO_ROOT)} ({len(per_run)} rows) and "
             f"{SUMMARY_CSV.relative_to(REPO_ROOT)}")

    print("\n=== fidelity summary (mean over runs) ===")
    show = ["method", "n_runs"] + [f"{m}_mean" for m in FIDELITY_METRICS]
    print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
