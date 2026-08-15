#!/usr/bin/env python3
"""TAPAS effective-epsilon audit for DPGAN (RQ1, sub-question 1).

Threat model: exact-knowledge data prior + black-box generator knowledge,
membership inference only. Runs 4 attacks (Groundhog, shadow-modelling with
random-query features, local-neighbourhood, probability-estimation) and
compares each attack's effective epsilon to the formal DP guarantee (eps=1.0).

Caching (resumable — safe to re-run after a crash or Ctrl-C):
  - The fitted threat model (incl. every simulated dataset generated so far) is
    pickled to cache/threat_model.pkl and reloaded if present.
  - Each attack's result is cached to cache/result_<attack>.json as soon as it
    finishes, and skipped on the next run if already present.

IMPORTANT runtime note (read before running):
Per the README's overhead benchmark, DPGAN fit+generate at the "real" settings
(n_iter=2000, patience=5, stopped at 299 iters) took ~6756s (112 min) on a
laptop CPU. TAPAS needs one fit+generate call per simulated dataset, shared
via memoisation across all 4 attacks -- but NOT shared across separate script
invocations if you increase NUM_TRAIN/NUM_TEST later than a previous run's
cache. At the real settings, (NUM_TRAIN + NUM_TEST) simulations x 112 min
becomes infeasible fast: 20 simulations alone is ~37 hours.

To make this tractable, SHADOW_PLUGIN_KWARGS below caps shadow-model training
to far fewer iterations than the "real" run (n_iter=100). Note that Synthcity's
dpgan plugin has its own internal n_iter_min=100 -- early stopping (patience=5,
also a Synthcity default) can never trigger before that floor. Since our cap
equals that floor, every shadow fit trains for *exactly* 100 iterations, every
time -- early stopping never gets a chance to act. This is a deliberate,
predictable choice (you know exactly how long every fit takes) over letting
Synthcity's real patience=5/n_iter=2000 defaults govern it, which would be
more faithful to the real generator's behavior (converged at 299 iterations
per the README) but with unbounded worst-case time per fit. Revisit this
tradeoff once results look reasonable and GPU-driven cost reduction is
confirmed -- n_iter=100 is a "get it working first" choice, not a final one.

Device: Synthcity's dpgan plugin defaults to device='cpu' and does NOT
auto-detect CUDA -- confirmed by inspecting the plugin's constructor
signature directly. DEVICE below auto-detects via torch.cuda.is_available()
and is passed through SHADOW_PLUGIN_KWARGS, so this script actually uses the
GPU on Colab instead of silently training on CPU there.
"""

import sys
import logging
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eff_eps/ (for `common`)
import common

METHOD = "dpgan"
NUM_SYNTHETIC_RECORDS = 21523  # matches training set size, per README
NUM_TRAIN_SIMULATIONS = 20
NUM_TEST_SIMULATIONS = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cheaper-than-real DPGAN settings for shadow models (see runtime note above).
# Remove/adjust n_iter once you've decided how much compute this experiment
# can spend; device is auto-detected so this line shouldn't need editing.
SHADOW_PLUGIN_KWARGS = {"n_iter": 100, "device": DEVICE}

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
    log.info(f"Shadow-model device: {DEVICE} "
             f"(torch.cuda.is_available()={torch.cuda.is_available()})")

    train_dataset, test_dataset, description = common.load_adult_datasets()
    target_record, background_dataset = common.select_target_and_background(train_dataset)

    threat_model = common.build_or_load_threat_model(
        cache_dir=CACHE_DIR,
        method=METHOD,
        background_dataset=background_dataset,
        target_record=target_record,
        description=description,
        num_synthetic_records=NUM_SYNTHETIC_RECORDS,
        epsilon=common.FORMAL_EPSILON,
        plugin_kwargs=SHADOW_PLUGIN_KWARGS,
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
