#!/usr/bin/env python3
"""TAPAS privacy evaluation on pre-generated synthetic datasets.

Attack strategy with pre-generated data:
  MIA  — ClosestDistanceMIA: score each target record by its negative
          minimum L2 distance to any synthetic record. Members (train set)
          tend to have closer synthetic neighbours than non-members (test set).
  AIA  — SynInference: train a LogisticRegression classifier on synthetic
          (features → income), then apply to real test records. Measures how
          much the synthetic data leaks the sensitive attribute.

Reports: MIAttackReport, BinaryAIAttackReport, EffectiveEpsilonReport, ROCReport.
         Numeric summary → results/tapas_results.csv
         Plots / epsilon CSV → results/tapas/{method}/

Run from repo root: python evaluation/eval_tapas.py
"""

import time
import tracemalloc
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

METHODS = ['bayesian_network', 'privbayes', 'ctgan', 'dpgan']

CONTINUOUS_COLS  = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_COLS = ['workclass', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'native_country', 'income']
TARGET_COL       = 'income'
POSITIVE_INCOME  = '>50K'

DATA_DIR    = Path('data')
SYN_DIR     = Path('synthetic_data')
RESULTS_DIR = Path('results')
TAPAS_DIR   = RESULTS_DIR / 'tapas'
EVAL_DIR    = Path('evaluation')

MIA_N_TARGETS = 300   # members + non-members each (capped by set size)

# ---------- logging ----------
EVAL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TAPAS_DIR.mkdir(exist_ok=True)
log_path = EVAL_DIR / 'eval_tapas_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ---------- TAPAS imports ----------
from tapas.report import (
    MIAttackSummary, MIAttackReport,
    BinaryAIAttackSummary, BinaryAIAttackReport,
    EffectiveEpsilonReport, ROCReport,
)

# ---------- helpers ----------
def load_df(path):
    df = pd.read_csv(path)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def encode_for_distance(ref_df, *dfs):
    """One-hot encode categoricals + standardise continuous. Fit on ref_df.
    Returns encoded arrays for each df, aligned to ref_df's vocabulary."""
    feature_cols = CONTINUOUS_COLS + [c for c in CATEGORICAL_COLS if c != TARGET_COL]

    # One-hot encode using combined vocab from ref_df (training data)
    cat_cols = [c for c in CATEGORICAL_COLS if c != TARGET_COL]

    # Fit dummies on reference
    ref_dummies = pd.get_dummies(ref_df[cat_cols], dtype=float)
    dummy_cols = ref_dummies.columns

    def transform(df):
        num = df[CONTINUOUS_COLS].copy().values.astype(float)
        cat = pd.get_dummies(df[cat_cols], dtype=float).reindex(
            columns=dummy_cols, fill_value=0.0).values
        return np.hstack([num, cat])

    ref_enc = transform(ref_df)
    scaler = StandardScaler().fit(ref_enc)
    return (scaler.transform(ref_enc),) + tuple(scaler.transform(transform(d)) for d in dfs)


def mia_closest_distance(train_enc, test_enc, syn_enc, n_targets, rng):
    """Membership Inference Attack via closest distance to synthetic records.

    Members (from train set) tend to have smaller minimum distance to synthetic
    neighbours than non-members (from test set).

    Returns labels (True=member), predictions, scores.
    """
    n = min(n_targets, len(train_enc), len(test_enc))
    member_idx    = rng.choice(len(train_enc), size=n, replace=False)
    nonmember_idx = rng.choice(len(test_enc),  size=n, replace=False)

    targets = np.vstack([train_enc[member_idx], test_enc[nonmember_idx]])
    labels  = np.array([True] * n + [False] * n)

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='ball_tree')
    nn.fit(syn_enc)
    distances, _ = nn.kneighbors(targets)
    scores = -distances.flatten()
    threshold = np.median(scores)
    predictions = scores >= threshold

    # Shuffle so validation split in EffectiveEpsilonReport sees both classes
    perm = rng.permutation(len(labels))
    return labels[perm], predictions[perm], scores[perm]


def aia_syn_inference(syn_df, test_df):
    """Attribute Inference Attack: train classifier on synthetic, infer income on real.

    Returns binary labels (1 = >50K), predictions, probability scores.
    """
    feat_cols = [c for c in syn_df.columns if c != TARGET_COL]
    cat_cols  = [c for c in feat_cols if c in CATEGORICAL_COLS]

    # Encode synthetic data (train set for attack)
    syn_cat  = pd.get_dummies(syn_df[cat_cols], dtype=float)
    syn_feat = pd.concat([syn_df[CONTINUOUS_COLS], syn_cat], axis=1)
    syn_label = (syn_df[TARGET_COL].str.strip() == POSITIVE_INCOME).astype(int)

    # Encode real test data — align to synthetic columns
    real_cat  = pd.get_dummies(test_df[cat_cols], dtype=float).reindex(
        columns=syn_cat.columns, fill_value=0.0)
    real_feat = pd.concat([test_df[CONTINUOUS_COLS], real_cat], axis=1)
    real_label = (test_df[TARGET_COL].str.strip() == POSITIVE_INCOME).astype(int)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(syn_feat.values, syn_label.values)

    scores      = clf.predict_proba(real_feat.values)[:, 1]
    predictions = clf.predict(real_feat.values)
    return real_label.values, predictions, scores


# ---------- load real data ----------
train_df = load_df(DATA_DIR / 'adult_train.csv')
test_df  = load_df(DATA_DIR / 'adult_test.csv')
log.info(f'Loaded train {train_df.shape}, test {test_df.shape}')
train_enc, test_enc = encode_for_distance(train_df, test_df)

rng = np.random.default_rng(42)

# ---------- run ----------
tracemalloc.start()
t_start = time.time()
rows = []
all_mia_summaries = []
all_aia_summaries = []
methods_succeeded = 0
methods_failed = 0

for method in METHODS:
    log.info(f'--- {method} ---')
    method_dir = TAPAS_DIR / method
    method_dir.mkdir(exist_ok=True)

    try:
        syn_path = SYN_DIR / f'{method}_synthetic.csv'
        if not syn_path.exists():
            raise FileNotFoundError(f'{syn_path} not found — run SDG notebook first')

        syn_df = load_df(syn_path)
        log.info(f'{method}: loaded {syn_df.shape}')

        # Encode synthetic for distance computation (using train_df as reference vocab)
        _, syn_enc = encode_for_distance(train_df, syn_df)

        # ---- MIA ----
        mia_labels, mia_preds, mia_scores = mia_closest_distance(
            train_enc, test_enc, syn_enc, MIA_N_TARGETS, rng)

        mia_summary = MIAttackSummary(
            labels=mia_labels, predictions=mia_preds, scores=mia_scores,
            generator_info=method, attack_info='ClosestDistanceMIA',
            dataset_info='AdultCensus', target_id='aggregate',
        )
        all_mia_summaries.append(mia_summary)
        log.info(f'{method} MIA — AUC={mia_summary.auc:.4f}  '
                 f'advantage={mia_summary.mia_advantage:.4f}  '
                 f'eff_epsilon={mia_summary.effective_epsilon:.4f}')

        # ---- AIA ----
        aia_labels, aia_preds, aia_scores = aia_syn_inference(syn_df, test_df)

        aia_summary = BinaryAIAttackSummary(
            labels=aia_labels, predictions=aia_preds, scores=aia_scores,
            generator_info=method, attack_info='SynInference-LR',
            dataset_info='AdultCensus', target_id='income',
            sensitive_attribute='income',
        )
        all_aia_summaries.append(aia_summary)
        log.info(f'{method} AIA — AUC={aia_summary.auc:.4f}  '
                 f'advantage={aia_summary.mia_advantage:.4f}  '
                 f'eff_epsilon={aia_summary.effective_epsilon:.4f}')

        # ---- collect core metrics first so they're always saved ----
        row = {
            'method': method,
            'mia_auc':          mia_summary.auc,
            'mia_advantage':    mia_summary.mia_advantage,
            'mia_privacy_gain': mia_summary.privacy_gain,
            'mia_eff_epsilon':  mia_summary.effective_epsilon,
            'mia_tp':           mia_summary.tp,
            'mia_fp':           mia_summary.fp,
            'aia_auc':          aia_summary.auc,
            'aia_advantage':    aia_summary.mia_advantage,
            'aia_privacy_gain': aia_summary.privacy_gain,
            'aia_eff_epsilon':  aia_summary.effective_epsilon,
            'eps_low_90':       np.nan,
            'eps_high_90':      np.nan,
        }

        # ---- EffectiveEpsilonReport (MIA) ----
        try:
            eps_report = EffectiveEpsilonReport(
                [mia_summary], validation_split=0.1,
                confidence_levels=(0.9, 0.95, 0.99), suffix=method
            )
            eps_df = eps_report.publish(str(method_dir))
            row['eps_low_90']  = eps_df.iloc[0].epsilon_low
            row['eps_high_90'] = eps_df.iloc[0].epsilon_high
            log.info(f'{method} effective epsilon (90% CI): '
                     f'[{row["eps_low_90"]:.4f}, {row["eps_high_90"]:.4f}]')
        except Exception:
            log.warning(f'{method} EffectiveEpsilonReport failed:\n{traceback.format_exc()}')

        # ---- ROCReport ----
        try:
            ROCReport([mia_summary, aia_summary], suffix=method).publish(str(method_dir))
        except Exception:
            log.warning(f'{method} ROCReport failed:\n{traceback.format_exc()}')

        rows.append(row)
        methods_succeeded += 1

    except Exception:
        log.error(f'{method} FAILED:\n{traceback.format_exc()}')
        methods_failed += 1

# ---- aggregate reports across all methods ----
if all_mia_summaries:
    MIAttackReport(all_mia_summaries).publish(str(TAPAS_DIR))
    log.info('MIAttackReport saved to results/tapas/')

if all_aia_summaries:
    BinaryAIAttackReport(all_aia_summaries).publish(str(TAPAS_DIR))
    log.info('BinaryAIAttackReport saved to results/tapas/')

# ---- save numeric results ----
if rows:
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'tapas_results.csv', index=False)
    log.info(f'Saved results to results/tapas_results.csv')
else:
    log.warning('No results to save.')

# ---------- overhead ----------
wall_time = round(time.time() - t_start, 2)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

overhead_path = EVAL_DIR / 'eval_overhead.csv'
row_oh = pd.DataFrame([{
    'script': 'eval_tapas',
    'runtime_sec': wall_time,
    'peak_memory_mb': round(peak / 1e6, 2),
    'methods_succeeded': methods_succeeded,
    'methods_failed': methods_failed,
}])
if overhead_path.exists():
    existing = pd.read_csv(overhead_path)
    existing = existing[existing['script'] != 'eval_tapas']
    row_oh = pd.concat([existing, row_oh], ignore_index=True)
row_oh.to_csv(overhead_path, index=False)

log.info(f'Finished in {wall_time}s | peak {round(peak/1e6,2)} MB | '
         f'{methods_succeeded} succeeded, {methods_failed} failed')
