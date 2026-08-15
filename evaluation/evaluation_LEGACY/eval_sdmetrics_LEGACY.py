#!/usr/bin/env python3
"""SDMetrics evaluation on pre-generated synthetic datasets.

SUPERSEDED -- USE evaluation/eval_fidelity.py INSTEAD.
    Same four metrics, same definitions, same comparison against training data.
    The difference is the input: this script scores the single unseeded draw per
    method, while eval_fidelity.py scores every seeded run from
    sdg/generate_runs.py and reports mean +/- std.

    Kept for provenance because results/sdmetrics_results.csv and the figures
    built from it came from here. Its inputs were renamed to
    synthetic_data/{method}_synthetic_LEGACY.csv, and the paths below follow
    that rename so the script still reproduces its own numbers.

Covers fidelity metrics only (see README — utility cross-check metrics
BinaryLogisticRegression_F1 and LogisticDetection were removed; utility is
reported via synthcity's eval_performance instead, and detection is covered
by synthcity's detection.detection_xgb/detection_linear).

Run from repo root: python evaluation/eval_sdmetrics.py
"""

import time
import tracemalloc
import logging
import traceback
from pathlib import Path

import pandas as pd

METHODS = ['bayesian_network', 'privbayes', 'ctgan', 'dpgan', 'aim']

CONTINUOUS_COLS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'native_country', 'income']
TARGET_COL = 'income'

DATA_DIR    = Path('data')
SYN_DIR     = Path('synthetic_data')
RESULTS_DIR = Path('results')
EVAL_DIR    = Path('evaluation') / 'evaluation_LEGACY'

# SDMetrics metadata: column sdtypes derived from our schema
METADATA = {
    'columns': {
        **{col: {'sdtype': 'numerical'} for col in CONTINUOUS_COLS},
        **{col: {'sdtype': 'categorical'} for col in CATEGORICAL_COLS},
    }
}

# ---------- logging ----------
EVAL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
log_path = EVAL_DIR / 'eval_sdmetrics_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ---------- imports ----------
from sdmetrics.single_table import (
    KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity,
)

# ---------- load real data ----------
def load_df(path):
    df = pd.read_csv(path)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df

train_df = load_df(DATA_DIR / 'adult_train.csv')
test_df  = load_df(DATA_DIR / 'adult_test.csv')
log.info(f'Loaded train {train_df.shape}, test {test_df.shape}')

# ---------- run ----------
tracemalloc.start()
t_start = time.time()
rows = []
methods_succeeded = 0
methods_failed = 0

for method in METHODS:
    log.info(f'--- {method} ---')
    try:
        syn_path = SYN_DIR / f'{method}_synthetic_LEGACY.csv'
        if not syn_path.exists():
            raise FileNotFoundError(
                f'{syn_path} not found. This is the legacy single-draw path; '
                f'current runs live in synthetic_data/runs/ — use eval_fidelity.py')

        syn_df = load_df(syn_path)
        log.info(f'{method}: loaded {syn_df.shape}')

        row = {'method': method}

        # --- Fidelity ---
        # KSComplement: per-column KS similarity for continuous columns (higher = better)
        row['KSComplement'] = KSComplement.compute(train_df, syn_df, metadata=METADATA)
        log.info(f'{method} KSComplement: {row["KSComplement"]:.4f}')

        # TVComplement: per-column TV similarity for categorical columns (higher = better)
        row['TVComplement'] = TVComplement.compute(train_df, syn_df, metadata=METADATA)
        log.info(f'{method} TVComplement: {row["TVComplement"]:.4f}')

        # CorrelationSimilarity: pairwise Pearson correlation between numerical columns
        row['CorrelationSimilarity'] = CorrelationSimilarity.compute(
            train_df, syn_df, metadata=METADATA)
        log.info(f'{method} CorrelationSimilarity: {row["CorrelationSimilarity"]:.4f}')

        # ContingencySimilarity: joint distribution of categorical column pairs
        row['ContingencySimilarity'] = ContingencySimilarity.compute(
            train_df, syn_df, metadata=METADATA)
        log.info(f'{method} ContingencySimilarity: {row["ContingencySimilarity"]:.4f}')

        rows.append(row)
        methods_succeeded += 1

    except Exception:
        log.error(f'{method} FAILED:\n{traceback.format_exc()}')
        methods_failed += 1

# ---------- save ----------
if rows:
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'sdmetrics_results.csv', index=False)
    log.info(f'Saved results to results/sdmetrics_results.csv')
else:
    log.warning('No results to save.')

# ---------- overhead ----------
wall_time = round(time.time() - t_start, 2)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

overhead_path = EVAL_DIR / 'eval_overhead_LEGACY.csv'
row_oh = pd.DataFrame([{
    'script': 'eval_sdmetrics',
    'runtime_sec': wall_time,
    'peak_memory_mb': round(peak / 1e6, 2),
    'methods_succeeded': methods_succeeded,
    'methods_failed': methods_failed,
}])
if overhead_path.exists():
    existing = pd.read_csv(overhead_path)
    existing = existing[existing['script'] != 'eval_sdmetrics']
    row_oh = pd.concat([existing, row_oh], ignore_index=True)
row_oh.to_csv(overhead_path, index=False)

log.info(f'Finished in {wall_time}s | peak {round(peak/1e6,2)} MB | '
         f'{methods_succeeded} succeeded, {methods_failed} failed')