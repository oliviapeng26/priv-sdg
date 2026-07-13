#!/usr/bin/env python3
"""Target strategy: OUTLIER (Houssiau et al. 2022, original TAPAS paper Experiment 2).

Selects the record with lowest log-likelihood under independent empirical
marginals from a random subsample of training records not in the background.
This is the "outlier" heuristic the TAPAS paper uses — the intuition is that
rare records are harder to hide in a synthetic dataset, making MIA easier.

Alternate t' is selected the same way as random (arbitrary non-background,
non-target record).
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    REPO_ROOT, NUM_TRAIN, NUM_TEST, FORMAL_EPSILON,
    CONTINUOUS_COLS, CATEGORICAL_COLS,
    strategy_results_dir,
)
import common

STRATEGY = "outlier"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / f"{STRATEGY}_strategy"
RESULTS_DIR = strategy_results_dir(STRATEGY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / f"{STRATEGY}_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("target_strategy")


# ── Target selection ───────────────────────────────────────────────────

def _log_likelihood_independent_marginals(df: pd.DataFrame, num_bins: int = 20):
    """Compute per-row log-likelihood assuming all attributes are independent.

    Continuous columns: discretised into num_bins equal-width bins, probability
    estimated from bin frequency. Categorical columns: probability from value
    frequency. Returns array of log-likelihoods (one per row).
    """
    n = len(df)
    log_liks = np.zeros(n)

    for c in CONTINUOUS_COLS:
        vals = df[c].astype(float).values
        # Equal-width binning on the [0,1]-scaled data
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_idx = np.clip(np.digitize(vals, bin_edges) - 1, 0, num_bins - 1)
        counts = np.bincount(bin_idx, minlength=num_bins)
        probs = (counts + 1) / (n + num_bins)  # Laplace smoothing
        log_liks += np.log(probs[bin_idx])

    for c in CATEGORICAL_COLS:
        vals = df[c].astype(str).values
        unique, inverse = np.unique(vals, return_inverse=True)
        counts = np.bincount(inverse)
        probs = (counts + 1) / (n + len(unique))  # Laplace smoothing
        log_liks += np.log(probs[inverse])

    return log_liks


def select_target(train_dataset, background_indices, subsample_size=1000, seed=44):
    """Pick the lowest log-likelihood record from a subsample of non-background
    training records. Alternate t' is a random non-background, non-target record.

    Matches the TAPAS paper's Experiment 1 target selection (Houssiau et al.
    2022, sec. 4): 'we select an "outlier" record, by choosing the record with
    lowest log-likelihood from a random sample of 1000 records.'
    """
    rng = np.random.RandomState(seed)
    n = len(train_dataset.data)
    available = np.array([i for i in range(n) if i not in background_indices])

    # Subsample for log-likelihood computation
    subsample_idx = rng.choice(available, size=min(subsample_size, len(available)), replace=False)
    subsample_df = train_dataset.data.iloc[subsample_idx].reset_index(drop=True)

    log_liks = _log_likelihood_independent_marginals(subsample_df)
    outlier_pos = np.argmin(log_liks)  # lowest log-likelihood = rarest record
    outlier_global_idx = int(subsample_idx[outlier_pos])

    target = train_dataset.get_records([outlier_global_idx])
    log.info(f"[{STRATEGY}] outlier target at index {outlier_global_idx} "
             f"(log-lik = {log_liks[outlier_pos]:.3f})")

    # Alternate t': random record excluding background and target
    remaining = np.array([i for i in available if i != outlier_global_idx])
    tprime_idx = int(rng.choice(remaining))
    alternate = train_dataset.get_records([tprime_idx])
    log.info(f"[{STRATEGY}] alternate t' at index {tprime_idx}")

    return target, alternate


# ── Main ───────────────────────────────────────────────────────────────

def main():
    log.info(f"=== Target strategy experiment: {STRATEGY} ===")

    train_dataset, test_dataset, description = common.load_adult_datasets()
    background_dataset, background_indices = common.sample_background(train_dataset)
    target_record, alternate_record = select_target(train_dataset, background_indices)

    threat_model = common.build_or_load_threat_model(
        cache_dir=CACHE_DIR,
        background_dataset=background_dataset,
        target_record=target_record,
        alternate_record=alternate_record,
        description=description,
    )

    attacks = common.build_attacks(target_record, background_dataset)

    rows = []
    summaries = {}
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model,
            num_train=NUM_TRAIN, num_test=NUM_TEST,
            cache_dir=CACHE_DIR, results_dir=RESULTS_DIR,
        )
        result["strategy"] = STRATEGY
        result["formal_epsilon"] = FORMAL_EPSILON
        rows.append(result)
        if summary is not None:
            summaries[attack.label] = summary
        threat_model.save(str(CACHE_DIR / "threat_model"))

    common.save_roc_report(summaries, RESULTS_DIR, f"{STRATEGY}_strategy")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"effeps_{STRATEGY}.csv", index=False)
    log.info(f"Saved results to {RESULTS_DIR / f'effeps_{STRATEGY}.csv'}")
    log.info(f"=== Done: {STRATEGY} ===")


if __name__ == "__main__":
    main()