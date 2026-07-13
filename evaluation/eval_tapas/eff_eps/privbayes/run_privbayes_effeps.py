#!/usr/bin/env python3
"""TAPAS effective-epsilon audit for PrivBayes (RQ1, sub-question 1).

Threat model: exact-knowledge data prior (a fixed 499-record background,
reused across every simulated dataset) + black-box generator knowledge,
membership inference only, aligned with the TAPAS paper's own Experiment 2
(Stadler et al. 2022, sec. 4): member world D+ = background+target(t),
non-member world D- = background+alternate(t'), both size 500, deterministic
on every draw (see common.SwapMIALabeller). Runs 4 attacks (Groundhog,
shadow-modelling with random-query features, local-neighbourhood,
probability-estimation) and compares each attack's effective epsilon to the
formal DP guarantee (eps=1.0).

Caching (resumable — safe to re-run after a crash or Ctrl-C):
  - The fitted threat model (incl. every simulated dataset generated so far) is
    pickled to cache/threat_model.pkl and reloaded if present.
  - Each attack's result is cached to cache/result_<attack>.json as soon as it
    finishes, and skipped on the next run if already present.
  - The per-attack cache is self-invalidating on sample count: if you raise
    NUM_TRAIN_SIMULATIONS/NUM_TEST_SIMULATIONS below and rerun, any attack
    whose cached result used fewer samples is automatically retrained/retested
    (see common.run_attack) -- no manual deletion needed. Lowering the
    constants back down just reuses the (more reliable) existing cache.
    threat_model.pkl's own memoisation means scaling up only generates the
    *delta* of new simulations, not a full redo.

  NOTE: the background size and target-selection method changed (full
  ~21,523-row background + single-outlier target -> fixed 499-row background +
  random target/alternate pair). Any cache/privbayes/threat_model.pkl or
  cache/privbayes/result_*.json from before this change is stale (built
  against the old, differently-shaped threat model) and must be deleted
  before rerunning -- it will NOT be auto-invalidated by common.run_attack's
  sample-count check, since that only compares num_train/num_test, not the
  underlying background/target.

Runtime note / choosing NUM_TRAIN_SIMULATIONS & NUM_TEST_SIMULATIONS:
A single PrivBayes fit+generate on a 500-row dataset (the new background+1
size) benchmarks at ~28.6s (fit ~28.5s, generate ~0.07s) -- see the
benchmarking snippet in this script's git history / PR description. Every
simulated dataset pair in TAPAS requires one fit+generate call, and
NUM_TRAIN + NUM_TEST calls are shared (via the threat model's memoisation)
across all 4 attacks, but SwapMIALabeller always generates *pairs*
(D+ and D- from the same background draw), so the total generator-fit cost
for this whole script is about (NUM_TRAIN + NUM_TEST) x 2 x 28.6s.

Lowering these numbers only widens the confidence interval on eff_epsilon
(Clopper-Pearson CI width scales ~1/sqrt(num_test)) -- it does NOT change the
threat model itself (same fixed background, same target/alternate records,
same generator config, same 4 attacks), so a quick pass is directly
comparable to, just noisier than, a full run.

  tier                | num_train | num_test | fits  | time @ 28.6s/fit | CI width vs. paper's 2500
  quick (default)     |    50     |   100    |  300  | ~2.4 hr          | ~5x wider
  medium               |   200     |   400    | 1200  | ~9.5 hr          | ~2.5x wider
  full (paper's own)  |   1000    |   2500   | 7000  | ~55.5 hr         | 1x (paper's own numbers)

Start with the quick tier to confirm the pipeline runs end-to-end with the
new background/target construction and get a rough directional signal, then
bump NUM_TRAIN_SIMULATIONS/NUM_TEST_SIMULATIONS to medium or full and rerun
-- the cache handles the rest automatically.

Run from repo root: python evaluation/eval_tapas/eff_eps/privbayes/run_privbayes_effeps.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eff_eps/ (for `common`)
import common

METHOD = "privbayes"
NUM_SYNTHETIC_RECORDS = common.BACKGROUND_SIZE + 1  # 500: matches each simulated world's size

# Quick tier (~2.4 hr) -- see the tier table above the imports to scale up.
NUM_TRAIN_SIMULATIONS = 50
NUM_TEST_SIMULATIONS = 100

CACHE_DIR = common.EFF_EPS_DIR / "cache" / METHOD
RESULTS_DIR = common.RESULTS_DIR / METHOD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(common.EFF_EPS_DIR / f"eff_eps_{METHOD}_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("eff_eps")


def main():
    log.info(f"=== TAPAS effective-epsilon audit: {METHOD} ===")

    train_dataset, test_dataset, description = common.load_adult_datasets()
    background_dataset, background_indices = common.sample_background(train_dataset)
    target_record, alternate_record = common.select_random_target_and_alternate(
        train_dataset, background_indices
    )

    threat_model = common.build_or_load_threat_model(
        cache_dir=CACHE_DIR,
        method=METHOD,
        background_dataset=background_dataset,
        target_record=target_record,
        alternate_record=alternate_record,
        description=description,
        num_synthetic_records=NUM_SYNTHETIC_RECORDS,
        epsilon=common.FORMAL_EPSILON,
    )

    attacks = common.build_attacks(target_record, background_dataset)

    rows = []
    summaries = {}
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model,
            num_train=NUM_TRAIN_SIMULATIONS, num_test=NUM_TEST_SIMULATIONS,
            cache_dir=CACHE_DIR, results_dir=RESULTS_DIR,
        )
        result["method"] = METHOD
        result["formal_epsilon"] = common.FORMAL_EPSILON
        result["exceeds_formal_epsilon"] = (
            result.get("eps_low_90") is not None
            and result["eps_low_90"] > common.FORMAL_EPSILON
        )
        rows.append(result)
        if summary is not None:
            summaries[attack.label] = summary
        # Persist the threat model after every attack (new simulations memoised).
        threat_model.save(str(CACHE_DIR / "threat_model"))

    common.save_roc_report(summaries, RESULTS_DIR, METHOD)

    import pandas as pd
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / f"effeps_{METHOD}.csv", index=False)
    log.info(f"Saved per-attack results to {RESULTS_DIR / f'effeps_{METHOD}.csv'}")

    # Merge into the cross-method comparison table.
    combined_path = common.RESULTS_DIR / "effeps_all_methods.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        existing = existing[existing["method"] != METHOD]
        out = pd.concat([existing, out], ignore_index=True)
    out.to_csv(combined_path, index=False)
    log.info(f"Updated combined comparison table at {combined_path}")

    log.info(f"=== Done: {METHOD} ===")


if __name__ == "__main__":
    main()
