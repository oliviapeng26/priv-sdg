#!/usr/bin/env python3
"""Target strategy: ARTIFICIAL worst-case (Chida et al. 2024).

Constructs a synthetic target record by selecting the rarest (minimum
frequency) value for each attribute independently. This is designed to be
maximally distinguishable from typical records — a worst-case for the
generator's privacy protection.

Alternate t' is a random non-background record (same as other strategies).
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
from tapas.datasets import TabularDataset

STRATEGY = "artificial"
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

def select_target(train_dataset, background_indices, seed=45):
    """Construct an artificial worst-case target by picking the rarest value
    per attribute from the training data.

    Continuous columns: pick the value from the training data that falls in the
    least-populated bin (after [0,1] scaling). This ensures the value is a real
    observed value, not a fabricated out-of-range number.

    Categorical columns: pick the category with the lowest frequency.

    Alternate t' is a random non-background record.
    """
    rng = np.random.RandomState(seed)
    df = train_dataset.data.copy()
    n = len(df)

    target_values = {}

    # Continuous: find the value in the least-populated bin
    num_bins = 20
    for c in CONTINUOUS_COLS:
        vals = df[c].astype(float).values
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_idx = np.clip(np.digitize(vals, bin_edges) - 1, 0, num_bins - 1)
        counts = np.bincount(bin_idx, minlength=num_bins)
        rarest_bin = np.argmin(counts)
        # Pick a real value from that bin
        mask = bin_idx == rarest_bin
        if mask.any():
            target_values[c] = float(rng.choice(vals[mask]))
        else:
            target_values[c] = float(vals[np.argmin(counts[bin_idx])])

    # Categorical: pick the least frequent category
    for c in CATEGORICAL_COLS:
        vals = df[c].astype(str)
        freq = vals.value_counts()
        rarest_cat = freq.idxmin()
        target_values[c] = rarest_cat

    # Build the target as a single-row TabularDataset
    target_df = pd.DataFrame([target_values])
    target_df = target_df[list(train_dataset.description.columns)]
    target_record = TabularDataset(target_df, train_dataset.description)

    log.info(f"[{STRATEGY}] constructed artificial worst-case target:")
    for col, val in target_values.items():
        log.info(f"  {col}: {val}")

    # Alternate t': random non-background record
    available = np.array([i for i in range(n) if i not in background_indices])
    tprime_idx = int(rng.choice(available))
    alternate = train_dataset.get_records([tprime_idx])
    log.info(f"[{STRATEGY}] alternate t' at index {tprime_idx}")

    return target_record, alternate


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