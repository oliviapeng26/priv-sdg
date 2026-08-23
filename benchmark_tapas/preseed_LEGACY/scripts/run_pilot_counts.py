#!/usr/bin/env python3
"""Pilot: does raising num_train/num_test actually tighten eps_low_95?

THE PROBLEM THIS TESTS
    Every generator in the 50/100 benchmark returned eps_low_95 = 2.209964 -- the
    same value to six decimal places. That is not four measurements agreeing; it is
    a ceiling. With only 100 test samples the smallest resolvable false-positive
    rate is 1/100, so the 95% lower bound on ln(TP/FP) cannot exceed a fixed value
    no matter how badly a generator leaks. The number describes the sample size,
    not the generators.

    The literature runs far larger: TAPAS's own Experiment 2 and Chida et al. both
    use 1000 train / 2500 test. At the per-fit times measured on this workstation
    that is ~27 h for all four generators -- affordable, but worth checking before
    committing.

WHAT THIS DOES
    Runs ONE generator (default: bayesian_network, the cheapest at ~1.05 s/fit)
    at the target counts, and prints eps_low_95 next to the 50/100 baseline. At
    1000/2500 that is 3500 fits ~= 1 h. If the bound moves substantially the full
    run is justified; if it barely moves, an hour has saved 27.

    Deliberately a separate script, not a change to METHOD_CONFIG: the committed
    50/100 configuration stays untouched until the pilot says otherwise.

FITS ARITHMETIC
    fits = num_train + num_test, NOT x2. Verified against a completed audit --
    the cached threat model from the 50/100 run memoised exactly 50 training and
    100 testing datasets. TAPAS's `num_samples` is the dataset count; the labeller
    halves it into pairs and emits both worlds, returning num_samples datasets,
    one generator fit each.

CACHE
    Writes to its own cache directory (cache_pilot_{method}_{train}_{test}/) so it
    can never collide with or invalidate the main benchmark's caches. Attacks are
    cached individually as they finish, so an interrupted run resumes.

Run from repo root:
  python benchmark_tapas/scripts/run_pilot_counts.py
  python benchmark_tapas/scripts/run_pilot_counts.py --num-train 1000 --num-test 2500
  python benchmark_tapas/scripts/run_pilot_counts.py --method ctgan --attacks GroundhogAttack
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

import common
from config import (
    FORMAL_EPSILON, METHOD_CONFIG, RESULTS_DIR, CACHE_DIR, slug,
)

BASELINE_EPS_LOW_95 = 2.209964      # what every method returned at 50/100
BASELINE_COUNTS = (50, 100)

PILOT_DIR = RESULTS_DIR / "pilot_counts"
PILOT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(PILOT_DIR / "pilot_counts_log.txt"), logging.StreamHandler()],
)
log = logging.getLogger("pilot")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="bayesian_network",
                        choices=list(METHOD_CONFIG),
                        help="generator to pilot (default: bayesian_network, the cheapest)")
    parser.add_argument("--num-train", type=int, default=1000,
                        help="shadow models for attack training (default: 1000, TAPAS Exp 2)")
    parser.add_argument("--num-test", type=int, default=2500,
                        help="shadow models for evaluation (default: 2500, TAPAS Exp 2)")
    parser.add_argument("--attacks", nargs="*", default=None,
                        help="attack labels to run (default: all 5). The first attack pays "
                             "for every shared fit; later ones reuse them.")
    args = parser.parse_args()

    method, num_train, num_test = args.method, args.num_train, args.num_test
    fits = num_train + num_test
    cfg = METHOD_CONFIG[method]

    # Isolated cache: never touches benchmark_tapas/cache/.
    cache_dir = CACHE_DIR.parent / f"cache_pilot_{method}_{num_train}_{num_test}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PILOT_DIR / f"{method}_{num_train}_{num_test}"
    results_dir.mkdir(parents=True, exist_ok=True)

    plugin_kwargs = dict(cfg["plugin_kwargs"])
    if method in ("ctgan", "dpgan"):
        import torch
        plugin_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"=== PILOT: {method} at num_train={num_train}, num_test={num_test} ===")
    log.info(f"fits = num_train + num_test = {fits}  (baseline was "
             f"{sum(BASELINE_COUNTS)} at {BASELINE_COUNTS[0]}/{BASELINE_COUNTS[1]})")
    log.info(f"plugin_kwargs={plugin_kwargs}   cache={cache_dir.name}")

    train_dataset, _, description = common.load_adult_datasets()
    background_dataset, background_indices = common.sample_background(train_dataset)
    target_record, alternate_record = common.select_random_target(
        train_dataset, background_indices)

    threat_model = common.build_or_load_threat_model(
        cache_dir=cache_dir, method=method,
        background_dataset=background_dataset,
        target_record=target_record, alternate_record=alternate_record,
        description=description,
        epsilon=FORMAL_EPSILON, plugin_kwargs=plugin_kwargs,
    )

    attacks = common.build_attacks(target_record, background_dataset)
    if args.attacks:
        wanted = {a.lower() for a in args.attacks}
        attacks = [a for a in attacks
                   if a.label.lower() in wanted or slug(a.label) in wanted]
        if not attacks:
            parser.error(f"no attacks matched {args.attacks}")

    rows = []
    t0 = time.time()
    for attack in attacks:
        result, _ = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=cache_dir, results_dir=results_dir,
        )
        result["method"] = method
        result["num_train"] = num_train
        result["num_test"] = num_test
        rows.append(result)
        threat_model.save(str(cache_dir / "threat_model"))

    out = pd.DataFrame(rows)
    out_path = results_dir / f"pilot_{method}_{num_train}_{num_test}.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}  ({time.time() - t0:.0f}s total)")

    # --- the actual question ---
    print(f"\n=== does the bound move? {method}, {num_train}/{num_test} ===")
    if "eps_low_95" in out:
        best = out["eps_low_95"].max()
        print(f"  baseline (50/100)     eps_low_95 = {BASELINE_EPS_LOW_95:.4f}   <- sample-size ceiling")
        print(f"  pilot ({num_train}/{num_test})   eps_low_95 = {best:.4f}   (max over "
              f"{len(out)} attack(s))")
        delta = best - BASELINE_EPS_LOW_95
        print(f"  change: {delta:+.4f}")
        if delta > 0.5:
            print("  -> the bound was sample-size limited. Higher counts are worth the "
                  "full run.")
        elif delta > 0.1:
            print("  -> modest improvement. Worth weighing against ~27 h for all four.")
        else:
            print("  -> barely moved. The ceiling is NOT mainly sample size; spend the "
                  "budget elsewhere (e.g. seeds) and report 2.21 as a genuine bound.")
        print("\n  per attack:")
        for _, r in out.iterrows():
            print(f"    {r['attack'][:42]:44s} auc={r['auc']:.3f}  "
                  f"eps_low_95={r.get('eps_low_95', float('nan')):.4f}")
    else:
        print("  eps_low_95 missing - check the log for EffectiveEpsilonReport errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
