#!/usr/bin/env python3
"""Target strategy: RANDOM.

Selects target t and alternate t' uniformly at random from training records
not in the background. Baseline — no deliberate outlier or worst-case selection.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from parent (target_strategy/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    REPO_ROOT, NUM_TRAIN, NUM_TEST, FORMAL_EPSILON,
    strategy_results_dir,
)
import common

STRATEGY = "random"
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

def select_target(train_dataset, background_indices, seed=43):
    """Pick target t and alternate t' uniformly at random from records
    not in the background."""
    n = len(train_dataset.data)
    rng = np.random.RandomState(seed)
    available = np.array([i for i in range(n) if i not in background_indices])
    t_idx, tprime_idx = rng.choice(available, size=2, replace=False)
    target = train_dataset.get_records([int(t_idx)])
    alternate = train_dataset.get_records([int(tprime_idx)])
    log.info(f"[{STRATEGY}] target t at index {t_idx}, alternate t' at index {tprime_idx}")
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