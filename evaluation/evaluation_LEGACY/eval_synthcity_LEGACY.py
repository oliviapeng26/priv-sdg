#!/usr/bin/env python3
"""Synthcity metric evaluation on pre-generated synthetic datasets.

SUPERSEDED FOR UTILITY -- USE evaluation/eval_utility.py INSTEAD.
    The performance.* numbers this script writes to results/synthcity_results.csv
    are contaminated by data leakage. Metrics.evaluate() takes the real loader it
    is handed -- here data/adult_train.csv, the 21,523 rows every generator was
    fitted on -- and makes its own internal 80/20 split of it, training the
    classifier on synthetic data and scoring it on that internal holdout. Those
    "held-out" records were in the generator's training set, so memorisation is
    rewarded. performance.*.gt is scored on the same leaky split, so the
    syn_id/gt ratio does not correct for it either.

    evaluation/eval_utility.py replaces this: TSTR trains on synthetic and
    scores on data/adult_test.csv (the 5,381 records no generator has seen),
    TRTR trains on all 21,523 training rows and scores on the same 5,381, and
    every seeded run from sdg/generate_runs.py is scored for mean +/- std.

    results/synthcity_results.csv has been DELETED, along with the figures
    derived from it (utility_metric_comparison.png, utility_privacy_tradeoff.png,
    utility_comparison_b32.png) -- the numbers were not salvageable. This script
    is kept only so the leak is documented at the site that produced it. Its
    inputs were renamed to synthetic_data/{method}_synthetic_LEGACY.csv, which
    the path below follows. Do not produce new utility numbers with it.

Uses Metrics.evaluate() directly (not Benchmarks.evaluate) since synthetic
data has already been generated. Benchmarks.evaluate() re-runs generators
internally; Metrics.evaluate() accepts pre-computed DataLoaders.

Utility metrics are computed (see README).

Run from repo root:
  python evaluation/eval_synthcity.py                   # all methods
  python evaluation/eval_synthcity.py bayesian_network  # one method (lower RAM)
"""

import sys
import time
import tracemalloc
import logging
import traceback
from pathlib import Path

import pandas as pd

# NOTE (2026-08-09) — if this script dies with `OMP: Error #179` or a bare segfault
# during performance.xgb / performance.feat_rank_distance, the OpenMP runtimes have
# collided again. xgboost ships no libomp of its own: libxgboost.dylib has a single
# rpath, /opt/homebrew/opt/libomp/lib, so it uses Homebrew's. torch bundles its own at
# torch/.dylibs/libomp.dylib, and the two builds cannot coexist in one process — the
# crash happens the moment XGBoost starts a thread pool after torch is imported.
# (sklearn's bundled copy is fine; only torch's conflicts. XGBoost alone, 8 threads,
# is also fine.) Fixed here by replacing torch's copy with a symlink to Homebrew's, so
# one runtime is shared; the original is kept at libomp.dylib.bak. A `pip install -U
# torch` restores torch's own copy and re-breaks it — redo the symlink (see README).
# Quick workaround if you can't: run with OMP_NUM_THREADS=1, which costs ~0.7s.

ALL_METHODS = ['bayesian_network', 'privbayes', 'ctgan', 'dpgan', 'aim']
METHODS = [sys.argv[1]] if len(sys.argv) > 1 else ALL_METHODS

CONTINUOUS_COLS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'native_country', 'income']
TARGET_COL = 'income'

DATA_DIR   = Path('data')
SYN_DIR    = Path('synthetic_data')
RESULTS_DIR = Path('results')
EVAL_DIR   = Path('evaluation') / 'evaluation_LEGACY'

METRICS_CFG = {
    'performance': ['linear_model', 'xgb', 'feat_rank_distance'],
}

# ---------- logging ----------
EVAL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
log_path = EVAL_DIR / 'eval_synthcity_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ---------- load real data ----------
def load_df(path):
    df = pd.read_csv(path)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype('category')
    return df

train_df = load_df(DATA_DIR / 'adult_train.csv')
log.info(f'Loaded real training data: {train_df.shape}')

# ---------- run ----------
from synthcity.metrics.eval import Metrics
from synthcity.plugins.core.dataloader import GenericDataLoader

real_loader = GenericDataLoader(train_df, target_column=TARGET_COL)

tracemalloc.start()
t_start = time.time()
results = []
methods_succeeded = 0
methods_failed = 0

for method in METHODS:
    log.info(f'--- {method} ---')
    try:
        syn_path = SYN_DIR / f'{method}_synthetic_LEGACY.csv'
        if not syn_path.exists():
            raise FileNotFoundError(
                f'{syn_path} not found. This is the legacy single-draw path; '
                f'current runs live in synthetic_data/runs/ — use eval_utility.py')

        syn_df = load_df(syn_path)
        log.info(f'{method}: loaded {syn_df.shape}')

        score = Metrics.evaluate(
            X_gt=real_loader,
            X_syn=syn_df,
            task_type='classification',
            metrics=METRICS_CFG,
            workspace=EVAL_DIR / 'workspace_synthcity',
        )
        score.insert(0, 'method', method)
        results.append(score)
        methods_succeeded += 1
        log.info(f'{method}: done')

    except Exception:
        log.error(f'{method} FAILED:\n{traceback.format_exc()}')
        methods_failed += 1

# ---------- save results ----------
if results:
    out = pd.concat(results)
    csv_path = RESULTS_DIR / 'synthcity_results.csv'
    if csv_path.exists():
        existing = pd.read_csv(csv_path, index_col=0)
        existing = existing[~existing['method'].isin(out['method'].unique())]
        out = pd.concat([existing, out])
    out.to_csv(csv_path)
    log.info(f'Saved results to results/synthcity_results.csv  shape={out.shape}')
else:
    log.warning('No results to save.')

# ---------- overhead ----------
wall_time = round(time.time() - t_start, 2)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

overhead_path = EVAL_DIR / 'eval_overhead_LEGACY.csv'
script_key = f'eval_synthcity_{METHODS[0]}' if len(METHODS) == 1 else 'eval_synthcity'
row = pd.DataFrame([{
    'script': script_key,
    'runtime_sec': wall_time,
    'peak_memory_mb': round(peak / 1e6, 2),
    'methods_succeeded': methods_succeeded,
    'methods_failed': methods_failed,
}])
if overhead_path.exists():
    existing = pd.read_csv(overhead_path)
    existing = existing[existing['script'] != script_key]
    row = pd.concat([existing, row], ignore_index=True)
row.to_csv(overhead_path, index=False)

log.info(f'Finished in {wall_time}s | peak {round(peak/1e6,2)} MB | '
         f'{methods_succeeded} succeeded, {methods_failed} failed')