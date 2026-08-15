#!/usr/bin/env python3
"""Standalone n_iter convergence + timing check for CTGAN / DPGAN (Step 1).

NOT through TAPAS. Trains each GAN on the FULL training set at several n_iter
caps, over several seeds, and records (a) wall-clock fit+generate time and
(b) TSTR utility. Two goals:

  1. Pick the smallest n_iter where TSTR utility has plateaued (so the capped
     generator still "learned enough" -- an undertrained GAN would deflate MIA
     AUC artificially, making it look private when it just leaked nothing; see
     README caveat).
  2. Measure seconds-per-fit so we can project the TAPAS shadow-model budget:
     fits = (num_train + num_test) x 2, and decide whether the neural counts
     fit inside one ~2 hr Colab T4 session.

UTILITY IS NOW IN-HOUSE (2026-08-15). This script used to call synthcity's
`Metrics.evaluate(metrics={"performance": ["xgb"]})` and read
performance.xgb.syn_id. That metric is leaky: synthcity splits the real loader
it is handed -- here the training set every GAN was just fitted on -- and scores
the synthetic-trained classifier on that internal holdout, so memorisation is
rewarded. It now calls `compute_tstr` from evaluation/eval_utility.py, which
scores on data/adult_test.csv, the 5,381 records no generator has seen.

This matters because the constants it feeds -- CTGAN_N_ITER=50 and
DPGAN_N_ITER=100 in config.py -- were chosen from the leaky numbers.

SEEDS. Each (method, n_iter) is now run over several seeds from seeds.RUN_SEEDS
and reported as mean +/- std. The previous single-seed sweep is what produced
DPGAN's 50=.40 -> 100=.60 -> 200=.55 -> default=.54 curve, and DPGAN_N_ITER=100
is the peak of it. At eps=1.0 the sibling DP generator (PrivBayes) swings
+/- 0.057 TSTR across seeds, which is wider than the .60-vs-.55 gap that pick
rests on -- so with n=1 that peak may be a draw, not a plateau.

The default plan (SEED_PLAN) is 3 seeds each at n_iter 50/100/200 -- the three
the choice is actually between -- and 1 seed at "default", which is only the
converged ceiling to compare the caps against and is the most expensive setting
by far. One plain invocation runs the whole plan; --seeds N overrides it
uniformly, and --seeds 1 reproduces the old design and warns.

Writes incrementally to results/convergence/convergence_check_tstr.csv (a NEW
file -- the old results are kept at convergence_check_LEGACY.csv for provenance,
and reusing that path would have made every setting look already-done) and is
resumable: an already-recorded (method, n_iter, seed) row is skipped.

Meant to run on the Colab T4 (device auto-detected).

Run:
  python benchmark_tapas/neural_tuning/convergence_check.py            # the full plan, both GANs
  python benchmark_tapas/neural_tuning/convergence_check.py ctgan      # one method
  python benchmark_tapas/neural_tuning/convergence_check.py --n-iter 50,100 --seeds 5
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent   # benchmark_tapas/
REPO_ROOT = BENCHMARK_DIR.parent                         # priv-sdg/
sys.path.insert(0, str(BENCHMARK_DIR))
from config import (
    TRAIN_CSV, TEST_CSV, CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL,
    CONVERGENCE_DIR, FORMAL_EPSILON, DP_METHODS, METHOD_CONFIG,
)

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ITER_CANDIDATES = [50, 100, 200, "default"]   # "default" -> plugin default (2000 max + early stop)

# Seeds per candidate. The three real candidates get 3 seeds each, because the
# choice between them is the whole point and at n=1 a 0.05 gap is inside the
# noise floor (see the SEEDS note in the module docstring). "default" gets 1:
# it is only the converged ceiling to compare the caps against, and it is by far
# the most expensive setting -- on the T4 it was 1744s (CTGAN) + 2835s (DPGAN),
# 63% of a whole single-seed sweep. --seeds N overrides this uniformly.
SEED_PLAN = {50: 3, 100: 3, 200: 3, "default": 1}
DEFAULT_SEEDS = 3   # used for any candidate not named in SEED_PLAN

# NEW OUTPUT FILE, deliberately not the old one. Two reasons:
#   1. already_done() keys on the recorded rows, and the old CSV has all 8
#      (method, n_iter) combinations from the leaky run. Reusing the path would
#      make every setting report "already recorded, skipping" and the GPU
#      session would do nothing.
#   2. The old numbers stay on disk as provenance for the CTGAN_N_ITER=50 /
#      DPGAN_N_ITER=100 choices currently baked into config.py.
OUT_CSV = CONVERGENCE_DIR / "convergence_check_tstr.csv"
LEGACY_CSV = CONVERGENCE_DIR / "convergence_check_LEGACY.csv"

CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(CONVERGENCE_DIR / "convergence_check_tstr_log.txt"),
                              logging.StreamHandler()])
log = logging.getLogger("convergence")

# Imported AFTER basicConfig on purpose: eval_utility configures the root logger
# at import time, so importing it first would divert this script's output into
# evaluation/eval_utility_log.txt.
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))
from eval_utility import (build_feature_spec, compute_tstr, compute_trtr,
                          compute_retention, UTILITY_MODELS)
from seeds import RUN_SEEDS, set_all_seeds


def load_real_dfs():
    """Training data for fitting, held-out test data for scoring.

    The old version loaded ONLY the training set, because synthcity's
    Metrics.evaluate made its own internal split of whatever it was handed --
    which is precisely the leak. TSTR needs a real holdout, so the 5,381-record
    test split is loaded here too.
    """
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    for df in (train, test):
        df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    fit_df = train.copy()
    fit_df[CATEGORICAL_COLS] = fit_df[CATEGORICAL_COLS].astype("category")
    return fit_df, train, test


def already_done(method, n_iter, seed):
    if not OUT_CSV.exists():
        return False
    done = pd.read_csv(OUT_CSV)
    return ((done["method"] == method)
            & (done["n_iter"].astype(str) == str(n_iter))
            & (done["seed"] == seed)).any()


def append_row(row: dict):
    CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if OUT_CSV.exists():
        df = pd.concat([pd.read_csv(OUT_CSV), df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)


def run_one(method, n_iter, seed, fit_df, test_df, spec, trtr):
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    kwargs = {"device": DEVICE, "random_state": seed}
    if n_iter != "default":
        kwargs["n_iter"] = int(n_iter)
    if method in DP_METHODS:
        kwargs["epsilon"] = FORMAL_EPSILON

    log.info(f"[{method} n_iter={n_iter} seed={seed}] fitting on {len(fit_df)} rows, "
             f"device={DEVICE}, kwargs={kwargs}")
    set_all_seeds(seed)
    plugin = Plugins().get(method, **kwargs)

    t0 = time.time()
    plugin.fit(GenericDataLoader(fit_df, target_column=TARGET_COL))
    fit_s = round(time.time() - t0, 2)

    t1 = time.time()
    syn = plugin.generate(count=len(fit_df), random_state=seed).dataframe()
    gen_s = round(time.time() - t1, 2)

    # In-house TSTR: train on synthetic, score on the 5,381-record held-out test
    # split that no generator has seen. Replaces synthcity's Metrics.evaluate,
    # whose performance.xgb.syn_id scored on an internal split of the SAME rows
    # the generator was fitted on (see evaluation_LEGACY/eval_synthcity_LEGACY.py).
    set_all_seeds(seed)
    tstr = compute_tstr(syn, test_df, spec, seed)
    retention = compute_retention(tstr, trtr)

    row = {
        "method": method, "n_iter": n_iter, "device": DEVICE, "seed": seed,
        "fit_time_s": fit_s, "generate_time_s": gen_s,
    }
    for model in UTILITY_MODELS:
        row[f"tstr_{model}_auc"] = tstr[model]
        row[f"trtr_{model}_auc"] = trtr[model]
        row[f"retention_{model}"] = retention[model]

    log.info(f"[{method} n_iter={n_iter} seed={seed}] fit={fit_s}s gen={gen_s}s  " +
             ", ".join(f"{m} TSTR={tstr[m]:.4f} (retention {retention[m]:.3f})"
                       for m in UTILITY_MODELS))
    append_row(row)
    return row


def summarise():
    """Per (method, n_iter): mean +/- std TSTR across seeds -- the convergence read.

    Pick the smallest n_iter whose mean TSTR is within noise of the best, where
    "noise" is the across-seed std in this very table. With one seed per setting
    the std column is NaN and there is no way to tell a plateau from a lucky
    draw -- which is exactly how the current DPGAN_N_ITER=100 was chosen.
    """
    if not OUT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(OUT_CSV)
    g = df.groupby(["method", "n_iter"], sort=False)
    out = g.agg(
        n_seeds=("seed", "count"),
        fit_time_s=("fit_time_s", "mean"),
        generate_time_s=("generate_time_s", "mean"),
        tstr_mean=("tstr_xgboost_auc", "mean"),
        tstr_std=("tstr_xgboost_auc", "std"),
        retention_mean=("retention_xgboost", "mean"),
    ).reset_index()
    out.to_csv(CONVERGENCE_DIR / "convergence_check_tstr_summary.csv", index=False)
    log.info("=== TSTR by n_iter (XGBoost, scored on the held-out test split) ===")
    log.info(out.to_string(index=False))
    return out


def project_budget(summary):
    """Print the TAPAS shadow-model budget projection from measured fit times."""
    if summary.empty:
        return
    log.info("=== TAPAS budget projection (fits = (num_train+num_test) x 2) ===")
    for _, r in summary.iterrows():
        method = r["method"]
        cfg = METHOD_CONFIG.get(method, {"num_train": 10, "num_test": 20})
        fits = (cfg["num_train"] + cfg["num_test"]) * 2
        per_fit = r["fit_time_s"] + r["generate_time_s"]
        total_min = fits * per_fit / 60.0
        fits_2hr = "FITS" if total_min <= 120 else "OVER"
        std = r["tstr_std"]
        std_str = "  +/- n/a (1 seed)" if pd.isna(std) else f"  +/- {std:.4f}"
        log.info(f"  {method:16s} n_iter={str(r['n_iter']):>7s}  "
                 f"{per_fit:7.1f}s/fit x {fits} fits = {total_min:6.1f} min "
                 f"[{fits_2hr} 2h @ {cfg['num_train']}/{cfg['num_test']}]  "
                 f"TSTR={r['tstr_mean']:.4f}{std_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", default=None,
                        help="methods to sweep (default: ctgan dpgan)")
    parser.add_argument("--seeds", type=int, default=None,
                        help=f"override seeds per (method, n_iter), from seeds.RUN_SEEDS. "
                             f"Default is the per-candidate plan {SEED_PLAN}. "
                             f"--seeds 1 reproduces the old single-draw design and warns.")
    parser.add_argument("--n-iter", default=None,
                        help="comma-separated n_iter candidates (default: 50,100,200,default). "
                             "'default' means the plugin default, 2000 max with early stopping.")
    args = parser.parse_args()

    methods = args.methods or ["ctgan", "dpgan"]
    candidates = N_ITER_CANDIDATES
    if args.n_iter:
        candidates = [c if c == "default" else int(c) for c in args.n_iter.split(",")]

    # seeds per candidate: uniform if --seeds given, else the per-candidate plan
    seeds_for = {
        c: RUN_SEEDS[:(args.seeds if args.seeds else SEED_PLAN.get(c, DEFAULT_SEEDS))]
        for c in candidates
    }

    log.info(f"=== convergence check (in-house TSTR): methods={methods}, device={DEVICE} ===")
    log.info("Plan: " + ", ".join(f"n_iter={c} x {len(s)} seed(s)" for c, s in seeds_for.items()))
    if DEVICE == "cpu":
        log.warning("Running on CPU -- GAN fits will be slow. This is meant for a GPU.")
    singles = [c for c, s in seeds_for.items() if len(s) == 1 and c != "default"]
    if singles:
        log.warning(f"Single seed at n_iter={singles}: the std column will be NaN and a plateau "
                    f"will be indistinguishable from a lucky draw. This is how the current "
                    f"DPGAN_N_ITER=100 was picked.")

    fit_df, train_df, test_df = load_real_dfs()
    spec = build_feature_spec(train_df, test_df)
    log.info(f"Fitting on {len(fit_df)} rows; TSTR scored on {len(test_df)} held-out records.")

    # TRTR reference: fixed data, so compute once per seed rather than per
    # setting. This is the denominator of every retention number below.
    trtr_by_seed = {}
    for seed in sorted({s for ss in seeds_for.values() for s in ss}):
        set_all_seeds(seed)
        trtr_by_seed[seed] = compute_trtr(train_df, test_df, spec, seed)
    first = next(iter(trtr_by_seed.values()))
    log.info("TRTR (real -> real): " +
             ", ".join(f"{m}={first[m]:.4f}" for m in UTILITY_MODELS))

    for method in methods:
        for n_iter in candidates:
            for seed in seeds_for[n_iter]:
                if already_done(method, n_iter, seed):
                    log.info(f"[{method} n_iter={n_iter} seed={seed}] already recorded, skipping")
                    continue
                try:
                    run_one(method, n_iter, seed, fit_df, test_df, spec, trtr_by_seed[seed])
                except Exception:
                    import traceback
                    log.error(f"[{method} n_iter={n_iter} seed={seed}] FAILED:\n"
                              f"{traceback.format_exc()}")

    project_budget(summarise())
    log.info(f"=== Done. Results at {OUT_CSV} ===")
    if LEGACY_CSV.exists():
        log.info(f"Old leaky-TSTR results left untouched at {LEGACY_CSV.name} for provenance.")


if __name__ == "__main__":
    main()
