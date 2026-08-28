#!/usr/bin/env python3
"""Separate attacker strength from measurement precision in the counts sweep.

THE PROBLEM
    run_counts_sweep.py grows num_train and num_test together, and they do different
    jobs. num_test only sharpens the estimate: the trained attack has some fixed TPR
    and FPR, and more test datasets tighten the Clopper-Pearson interval around them,
    raising the certifiable lower bound toward the truth without changing the truth.
    num_train genuinely changes the attacker -- Groundhog fits a random forest on
    num_train shadow models, so more of them make a better classifier and there is
    more leakage to find.

    So DPGAN's eps_low_95 rising 0.00 -> 0.52 -> 0.74 -> 1.76 mixes the two, and the
    claim "DPGAN leaks at least 1.76" cannot be separated from "an attacker with 1000
    shadow models extracts 1.76 from DPGAN".

THE EXPERIMENT
    Hold the test set at the full memoised pool (2500) and vary num_train alone.
    Any movement is then attacker strength, because the measurement is identical
    across rows. Comparing this curve against the sweep's attributes the effect:

      reaches the sweep's endpoint  -> the rise was measurement; a weak attacker
                                       always extracted that much and the small audit
                                       simply could not certify it
      plateaus well below           -> the rise was attacker budget, and eps_eff has
                                       to be reported as a function of shadow models

GROUNDHOG ONLY, DELIBERATELY
    Groundhog is the argmax attack in all 12 cells of the sweep -- every eps_low_95 in
    the figure is its number -- so it is the one the claim rests on. Adding
    ShadowModelling would roughly quadruple the runtime (~21 min per run against
    Groundhog's ~7) for an attack that never sets the bound. The three threshold
    attacks barely use num_train at all: their "training" only picks a scalar cutoff.

WHY IT NEEDS NO GPU AND NO GENERATION
    Every dataset is already memoised in cache/{method}/threat_model.pkl. Only the
    attacks re-run, and all five are sklearn/numpy. As a guardrail the generator's
    .fit is replaced with a raiser, so any accidental generation aborts instead of
    silently spending hours.

READ-ONLY ON THE CACHE
    This script truncates threat_model._memory[True] in place to control num_train and
    restores it in a finally block. It NEVER calls threat_model.save(). Running it
    cannot corrupt the sweep's caches, and it is safe to run while another sweep is
    working on a different generator.

Run from the repo root, venv active:
  python benchmark_tapas/scripts/run_disentangle_numtrain.py
  python benchmark_tapas/scripts/run_disentangle_numtrain.py --methods dpgan
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import common                                              # noqa: E402
import tapas.threat_models as tm                           # noqa: E402
from tapas.report import EffectiveEpsilonReport            # noqa: E402
from config import CACHE_DIR, RESULTS_DIR, METHODS         # noqa: E402
from seeds import SCORE_ATTACK_SEED                        # noqa: E402

from scipy.stats import mannwhitneyu                       # noqa: E402

NUM_TRAIN_VALUES = [50, 200, 500, 1000]
OUT_DIR = RESULTS_DIR / "counts_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "disentangle_log.txt"), logging.StreamHandler()],
)
log = logging.getLogger("disentangle")


def auc_from(labels, scores) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    U, _ = mannwhitneyu(scores[labels == 1], scores[labels == 0], alternative="two-sided")
    return float(U / (pos * neg))


def run(method: str, out_rows: list) -> None:
    path = CACHE_DIR / method / "threat_model"
    if not path.with_suffix(".pkl").exists():
        log.warning(f"{method}: no cache at {path}.pkl -- skipping")
        return

    log.info(f"=== {method} ===")
    threat_model = tm.ThreatModel.load(str(path))

    def _refuse(*a, **k):
        raise RuntimeError(f"{method}: generation attempted. Everything should be memoised.")
    threat_model.atk_know_gen.generator.fit = _refuse

    full_train = threat_model._memory[True]
    n_train_avail = len(full_train[0])
    n_test = len(threat_model._memory[False][0])
    log.info(f"    pools: {n_train_avail} train / {n_test} test (test held at full size)")

    target = threat_model.target_record
    background = threat_model.atk_know_data.attacker_knowledge.training_dataset

    try:
        for num_train in NUM_TRAIN_VALUES:
            if num_train > n_train_avail:
                log.warning(f"    num_train={num_train} > {n_train_avail} available -- skipping")
                continue
            t0 = time.time()

            # Prefix truncation. The pools are nested and SwapMIALabeller emits
            # alternating member/non-member pairs, so an even prefix is exactly balanced
            # and is the same data the sweep used at that stage.
            threat_model._memory[True] = (full_train[0][:num_train], full_train[1][:num_train])

            # Same seed every time: identical forest initialisation and identical random
            # queries, so num_train is the only thing that differs between rows.
            np.random.seed(SCORE_ATTACK_SEED)
            attack = common.build_attacks(target, background)[0]     # Groundhog
            assert attack.label == "Groundhog", attack.label

            attack.train(threat_model, num_samples=None)
            summary = threat_model.test(attack, num_samples=None)

            results_dir = OUT_DIR / "disentangle" / method / f"train{num_train}"
            results_dir.mkdir(parents=True, exist_ok=True)
            split = min(0.5, max(0.1, 15 / len(summary.scores)))
            eps = EffectiveEpsilonReport([summary], validation_split=split,
                                         confidence_levels=(0.95,),
                                         suffix=f"_{method}_train{num_train}").publish(str(results_dir))
            lo, hi = float(eps.epsilon_low.iloc[0]), float(eps.epsilon_high.iloc[0])
            a = auc_from(summary.labels, summary.scores)
            _, p = mannwhitneyu(np.asarray(summary.scores)[np.asarray(summary.labels) == 1],
                                np.asarray(summary.scores)[np.asarray(summary.labels) == 0],
                                alternative="two-sided")

            out_rows.append(dict(method=method, num_train=num_train, num_test=len(summary.scores),
                                 auc=a, p=p, tp=float(summary.tp), fp=float(summary.fp),
                                 eps_low_95=lo, eps_high_95=hi,
                                 wall_time_s=round(time.time() - t0, 1)))
            log.info(f"    num_train={num_train:>4}  auc={a:.3f}  p={p:.2e}  "
                     f"eps_low_95={lo:.3f} [{lo:.3f}, {hi:.3f}]  ({time.time() - t0:.0f}s)")
    finally:
        threat_model._memory[True] = full_train      # never leave the cache truncated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=[m for m in METHODS if m != "privbayes"],
                    choices=list(METHODS))
    args = ap.parse_args()

    log.info(f"=== num_train disentangling: {args.methods}, Groundhog, test set held full ===")
    rows = []
    t0 = time.time()
    for m in args.methods:
        run(m, rows)
    if not rows:
        log.error("no results -- are the sweep caches present under cache/{method}/?")
        return 1

    out = pd.DataFrame(rows)
    path = OUT_DIR / "disentangle_numtrain.csv"
    out.to_csv(path, index=False)
    log.info(f"wrote {path}  ({(time.time() - t0) / 60:.1f} min total)")
    print("\n=== eps_low_95, test set held at full size ===")
    print(out.pivot(index="method", columns="num_train", values="eps_low_95").round(3).to_string())
    print("\n=== AUC (attacker strength, measurement identical across the row) ===")
    print(out.pivot(index="method", columns="num_train", values="auc").round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
