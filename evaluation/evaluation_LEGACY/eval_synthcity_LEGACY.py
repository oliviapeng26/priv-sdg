#!/usr/bin/env python3
"""Synthcity metric evaluation on pre-generated synthetic datasets.

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

ALL_METHODS = ['bayesian_network', 'privbayes', 'ctgan', 'dpgan']
METHODS = [sys.argv[1]] if len(sys.argv) > 1 else ALL_METHODS

CONTINUOUS_COLS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'native_country', 'income']
TARGET_COL = 'income'

DATA_DIR   = Path('data')
SYN_DIR    = Path('synthetic_data')
RESULTS_DIR = Path('results')
EVAL_DIR   = Path('evaluation')

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
        syn_path = SYN_DIR / f'{method}_synthetic.csv'
        if not syn_path.exists():
            raise FileNotFoundError(f'{syn_path} not found — run SDG notebook first')

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

overhead_path = EVAL_DIR / 'eval_overhead.csv'
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