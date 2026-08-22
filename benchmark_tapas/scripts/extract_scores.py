#!/usr/bin/env python3
"""Extract the raw (continuous) attack scores that sit behind TAPAS's binary MIA decisions.

READ THIS FIRST -- WHICH CACHES ARE VALID
    Until 2026-08-22 SynthcityGenerator passed no random_state, which silently gave
    every fit synthcity's default of 0 (Plugin.__init__ random_state=0; Plugin.fit
    calls enable_reproducible_results). With ExactDataKnowledge feeding every
    simulation the same background, all D+ simulations produced ONE byte-identical
    synthetic dataset and all D- another. The 1000/2500 DPGAN pilot's 3500 fits
    yielded 2 distinct datasets.

    Scores extracted from those caches are 2 numbers repeated, and every AUC lands on
    exactly 0.000, 0.500 or 1.000. This script therefore reads ONLY benchmark_tapas/
    cache/, regenerated after the fix (seeds.TAPAS_GENERATOR_SEED_BASE + i per fit).
    cache_LEGACY/ and cache_pilot_* are unreachable from here on purpose.

WHY THIS EXISTS
    The benchmark reports one number per generator x attack: eff-epsilon, computed
    from a binary member/non-member decision. At num_test=100 that number saturates --
    every generator returned eps_low_95 = 2.209964 to six decimals, because with 100
    test samples the smallest resolvable FPR is 1/100 and the Clopper-Pearson bound
    caps there regardless of how badly a generator leaks. Raising the counts does not
    fix it (the 1000/2500 pilot moved the ceiling to 5.546079 and kept TPR=1, FPR=0);
    the threat model, not the sample size, is what forces FP to zero.

    But every attack computes a CONTINUOUS score before thresholding it. That score is
    thrown away by the time the result JSON is written. This script recovers it. The
    continuous scores support two things the binary decision cannot:

      separation   -- how far apart are member and non-member score distributions?
                      (full ROC AUC, not a single operating point)
      decisiveness -- is the attacker confident or guessing? A score cloud pressed
                      against the extremes is a different privacy story from one
                      piled up at the decision boundary, even when both binarise
                      to the same accuracy.

WHERE THE SCORES COME FROM (traced in tapas 1.0.0, commit a7069d7)
    Two binarisation paths, not five:

      TrainableThresholdAttack  attacks/base_classes.py:253-259
          predicts positive iff score >= self._threshold. The threshold is trained
          once in .train() (base_classes.py:220-232) as argmax(tpr - fpr) over the
          ROC of the TRAINING samples. Used by ClosestDistanceMIA,
          LocalNeighbourhoodAttack and ProbabilityEstimationAttack.

      ShadowModellingAttack     attacks/shadow_modelling.py:109
          delegates to sklearn: SetClassifier.predict -> set_classifiers.py:147 ->
          RandomForest argmax, i.e. an implicit and untrained 0.5 cutoff on
          predict_proba[:, 1]. Used by GroundhogAttack and ShadowModelling(RandomQueries).

    The per-attack score functions and their ranges:

      Groundhog                       shadow_modelling.py:129-134  RF P(member)          [0, 1]
      ShadowModelling(RandomQueries)  shadow_modelling.py:129-134  RF P(member)          [0, 1]
      ClosestDistanceMIA              closest_distance.py:87       -min L2 distance      (-inf, 0]
      LocalNeighbourhoodAttack        closest_distance.py:242      fraction in radius    [0, 1]
      ProbabilityEstimationAttack     synthinference.py:118        KDE LOG-density       (-inf, inf)

    The ranges are not commensurable. Do not plot all five on a shared axis without
    per-attack scaling -- only the two RF attacks are probabilities, and only for
    those does "clustered at 0.5" mean "guessing".

    NOTE ON ClosestDistanceMIA's AUC=0.000. This is NOT a sign-convention bug.
    closest_distance.py:84-87 already negates the distance ("larger scores are
    associated with higher probability of membership, whereas distances do the
    opposite"), and attack_summary.py:210 is a bare roc_auc_score with no flip. An
    AUC of exactly 0.000 therefore means the ranking really is perfectly inverted,
    with no ties -- roc_auc_score averages over ties, so a single tie would move it
    off 0.000. Conversely LocalNeighbourhood's AUC=0.500 for three of the four
    generators is consistent with a DEAD attack rather than a coin flip: its score
    is the fraction of synthetic records inside the radius, so if no record ever
    lands inside, every score is exactly 0.0, everything ties, and roc_auc_score
    returns 0.5. The extracted scores distinguish these two cases immediately --
    look for a spike at exactly 0.0.

WHY WE SUBSET DPGAN BY HAND
    TAPAS does NOT truncate a memoised pool to the requested size. In
    attacker_knowledge.py:_generate_samples, when memory is in use:

        num_samples -= len(self._memory[training][0])      # line 591, goes negative
        ...
        mem_datasets, mem_labels = self._memory[training]  # line 623-624, returns ALL

    So asking a 2500-dataset cache for 100 silently returns all 2500. BN, PrivBayes
    and CTGAN cache exactly 50/100 and are unaffected, but DPGAN's only cache at the
    corrected n_iter=50 is the 1000/2500 pilot. Passing num_test=100 there would have
    produced 25x more DPGAN rows than the other three, with no error -- and a
    difference in the plots caused by sample size rather than by the generator.

    We therefore subsample the memoised pool ourselves before running anything, and
    then pass num_samples=None (which is the documented "use exactly what is in
    memory" path) so TAPAS never generates.

    We subsample PAIRS, not individual datasets. SwapMIALabeller emits strictly
    alternating (member, non-member) worlds built from the same background with one
    record swapped -- that coupling IS the experiment. Sampling datasets independently
    would both unbalance the classes and split pairs across the keep/drop boundary.
    Verified against all four caches: labels alternate 1,0,1,0,... in both pools.

WHY IT NEVER NEEDS A GPU (OR SYNTHCITY)
    Every simulated dataset is already memoised in threat_model.pkl -- the generator
    fits, which are the only expensive and the only GPU-touching part, were paid for
    when the caches were built. common.SynthcityGenerator imports synthcity and torch
    inside .fit() only, and .fit() is never reached here. All five attacks are
    sklearn/numpy. Measured cost is ~95 s per generator: Groundhog ~25 s,
    ShadowModelling ~68 s, the other three <1 s each.

    As a guardrail the generator's .fit is replaced with a raiser, so any code path
    that would start generating fails loudly instead of quietly running for hours.

REPRODUCIBILITY
    Neither the RandomForests nor RandomTargetedQueryFeature take a random_state, so
    both draw from numpy's global RNG. We reseed to seeds.SCORE_ATTACK_SEED before
    building the attacks for EACH generator -- not once at startup. That is deliberate:
    it means all four generators are probed by the same forest initialisation and the
    same 1500 random queries, so a difference between generators is the generator's,
    not the attack's.

    Because of that fresh draw, Groundhog and ShadowModelling will land NEAR their
    cached AUCs rather than exactly on them. The three threshold attacks are
    deterministic given the datasets and should reproduce exactly.

OUTPUT
    results/scores/raw_scores.csv, one row per (generator, attack, test dataset):

        generator          bayesian_network | privbayes | ctgan | dpgan
        attack             TAPAS attack label
        target_id          target record id (identical across all four generators)
        ground_truth       1 = member (D+), 0 = non-member (D-)
        raw_score          the continuous score, BEFORE thresholding
        binary_prediction  1/0, what the attack actually decided
        threshold          the cutoff that produced that decision

    CAVEAT ON `threshold`: only the three TrainableThresholdAttacks have a real
    trained threshold (attack._threshold). Groundhog and ShadowModelling have none --
    sklearn just takes the larger class probability -- so they are recorded as 0.5,
    which is the true cutoff but a fixed one, not a learned one.

    4 generators x 5 attacks x 100 test datasets = 2000 rows. Scores are from the TEST
    pool only; the 50 training datasets are consumed fitting the classifiers and
    choosing thresholds, exactly as in the original benchmark run.

Run from the repo root:
  python benchmark_tapas/scripts/extract_scores.py
  python benchmark_tapas/scripts/extract_scores.py --methods dpgan --attacks Groundhog
  python benchmark_tapas/scripts/extract_scores.py --subset-seed 46 --out raw_scores_seed46.csv
"""

import argparse
import hashlib
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

import common                       # noqa: E402  (registers the classes the pickles need)
import tapas.threat_models as tm    # noqa: E402
from config import CACHE_DIR, SCORES_DIR   # noqa: E402
from seeds import SCORE_SUBSET_SEED, SCORE_ATTACK_SEED  # noqa: E402

# The benchmark's shadow-model counts. Every generator is cut to exactly this so the
# four score distributions are the same size and directly comparable.
NUM_TRAIN, NUM_TEST = 50, 100

# The regenerated caches under benchmark_tapas/cache/, written by run_method after the
# per-fit seeding fix. The cache_LEGACY/* and cache_pilot_* directories are deliberately
# NOT reachable from here: every simulation in them is a byte-identical duplicate (see
# the seeding note above), so scores extracted from them are meaningless.
CACHE_PATHS = {m: CACHE_DIR / m / "threat_model"
               for m in ("bayesian_network", "privbayes", "ctgan", "dpgan")}

SCORES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SCORES_DIR / "extract_scores_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("extract_scores")


def fingerprint(dataset) -> str:
    """Stable short hash of a TabularDataset's contents (for comparability checks)."""
    return hashlib.sha256(dataset.data.to_csv(index=False).encode()).hexdigest()[:12]


def subsample_pairs(pool, n_wanted: int, rng: np.random.RandomState, what: str):
    """Cut a memoised (datasets, labels) pool down to n_wanted, sampling whole D+/D- pairs.

    Returns the pool unchanged when it is already the requested size, so BN/PrivBayes/
    CTGAN are untouched and only DPGAN is actually subsampled.
    """
    datasets, labels = pool
    n_have = len(datasets)

    if n_have < n_wanted:
        raise SystemExit(
            f"{what}: cache holds {n_have} datasets but {n_wanted} were requested. "
            f"Refusing to run -- filling the gap would mean re-fitting the generator."
        )
    if n_have % 2 or n_wanted % 2:
        raise SystemExit(f"{what}: pool sizes must be even (got have={n_have}, want={n_wanted}).")

    # The whole pair-sampling argument rests on this ordering. Check it, don't assume it.
    if not all(bool(l) == (i % 2 == 0) for i, l in enumerate(labels)):
        raise SystemExit(
            f"{what}: labels are not strictly alternating member/non-member. "
            f"SwapMIALabeller should guarantee this; the cache may be from other code."
        )

    if n_have == n_wanted:
        log.info(f"    {what}: {n_have} datasets, exactly what is needed -- no subsampling")
        return pool

    chosen = np.sort(rng.choice(n_have // 2, size=n_wanted // 2, replace=False))
    idx = np.concatenate([[2 * p, 2 * p + 1] for p in chosen])
    log.info(f"    {what}: {n_have} -> {n_wanted} datasets "
             f"({n_wanted // 2} of {n_have // 2} pairs, seed={SCORE_SUBSET_SEED})")
    return ([datasets[i] for i in idx], [labels[i] for i in idx])


def load_and_prepare(method: str, subset_seed: int):
    """Load one cached threat model, cut its pools to NUM_TRAIN/NUM_TEST, disarm generation."""
    path = CACHE_PATHS[method]
    if not path.with_suffix(".pkl").exists():
        raise SystemExit(f"{method}: no cache at {path}.pkl")

    log.info(f"  loading {path.relative_to(BENCHMARK_DIR)}.pkl")
    threat_model = tm.ThreatModel.load(str(path))

    # Separate RandomState per method, seeded identically -- so the pairs kept for a
    # given method do not depend on how many methods ran before it.
    rng = np.random.RandomState(subset_seed)
    threat_model._memory[True] = subsample_pairs(
        threat_model._memory[True], NUM_TRAIN, rng, "train pool")
    threat_model._memory[False] = subsample_pairs(
        threat_model._memory[False], NUM_TEST, rng, "test pool")

    # Guardrail: the datasets are all memoised, so nothing should ever fit. If some
    # code path tries, fail immediately rather than silently starting a multi-hour run
    # (and, off the GPU, importing synthcity/torch that may not even be installed).
    def _refuse(*args, **kwargs):
        raise RuntimeError(
            f"{method}: the generator was asked to fit. Nothing in raw-score extraction "
            f"should generate data -- every dataset is memoised. This is a bug."
        )
    threat_model.atk_know_gen.generator.fit = _refuse

    return threat_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=list(CACHE_PATHS),
                        choices=list(CACHE_PATHS), help="generators to run (default: all 4)")
    parser.add_argument("--attacks", nargs="*", default=None,
                        help="attack labels to run (default: all 5)")
    parser.add_argument("--subset-seed", type=int, default=SCORE_SUBSET_SEED,
                        help=f"seed for the D+/D- pair subsample (default: {SCORE_SUBSET_SEED})")
    parser.add_argument("--attack-seed", type=int, default=SCORE_ATTACK_SEED,
                        help=f"seed for attack-internal randomness (default: {SCORE_ATTACK_SEED})")
    parser.add_argument("--out", default="raw_scores.csv", help="output filename in results/scores/")
    args = parser.parse_args()

    log.info("=== raw score extraction ===")
    log.info(f"methods={args.methods}  counts={NUM_TRAIN}/{NUM_TEST}  "
             f"subset_seed={args.subset_seed}  attack_seed={args.attack_seed}")

    rows, setup_fingerprints = [], {}
    t_start = time.time()

    for method in args.methods:
        log.info(f"--- {method} ---")
        threat_model = load_and_prepare(method, args.subset_seed)

        # Take the target/alternate/background from the cache itself rather than
        # re-deriving them from the CSVs: this compares what was actually audited.
        target_record = threat_model.target_record
        background = threat_model.atk_know_data.attacker_knowledge.training_dataset
        setup_fingerprints[method] = (
            fingerprint(target_record),
            fingerprint(threat_model.alternate_record),
            fingerprint(background),
        )

        # Same seed before every method: identical forests and identical random
        # queries across generators, so differences are the generator's.
        np.random.seed(args.attack_seed)
        attacks = common.build_attacks(target_record, background)
        if args.attacks:
            wanted = {a.lower() for a in args.attacks}
            attacks = [a for a in attacks if a.label.lower() in wanted]
            if not attacks:
                parser.error(f"no attacks matched {args.attacks}")

        for attack in attacks:
            t0 = time.time()
            attack.train(threat_model, num_samples=None)
            summary = threat_model.test(attack, num_samples=None)

            raw_threshold = getattr(attack, "_threshold", None)
            threshold = 0.5 if raw_threshold is None else float(raw_threshold)

            for truth, pred, score in zip(summary.labels, summary.predictions, summary.scores):
                rows.append({
                    "generator": method,
                    "attack": attack.label,
                    "target_id": summary.target_id,
                    "ground_truth": int(truth),
                    "raw_score": float(score),
                    "binary_prediction": int(pred),
                    "threshold": threshold,
                })

            log.info(f"    [{attack.label}] auc={summary.auc:.3f}  "
                     f"adv={summary.mia_advantage:.3f}  n={len(summary.scores)}  "
                     f"thr={threshold:.4g}  ({time.time() - t0:.1f}s)")

    # All four must have audited the same target against the same background, or the
    # score distributions are not comparable and neither plot means anything.
    distinct = set(setup_fingerprints.values())
    if len(distinct) > 1:
        for m, fp in setup_fingerprints.items():
            log.error(f"  {m}: target/alternate/background = {fp}")
        raise SystemExit("Caches disagree on the target, alternate or background. Aborting.")
    log.info(f"comparability OK -- all {len(setup_fingerprints)} methods share "
             f"target/alternate/background {distinct.pop()}")

    out = pd.DataFrame(rows)
    out_path = SCORES_DIR / args.out
    out.to_csv(out_path, index=False)
    log.info(f"wrote {len(out)} rows to {out_path}  ({time.time() - t_start:.0f}s total)")

    print(f"\n=== AUC over the full ROC (from raw scores) ===")
    pivot = (out.groupby(["attack", "generator"])
                .apply(lambda g: _auc(g.ground_truth.values, g.raw_score.values),
                       include_groups=False)
                .unstack())
    print(pivot.round(3).to_string())
    return 0


def _auc(labels, scores) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), tie-corrected. Avoids an sklearn import."""
    pos, neg = (labels == 1).sum(), (labels == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, so an all-identical score gives exactly 0.5.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


if __name__ == "__main__":
    sys.exit(main())
