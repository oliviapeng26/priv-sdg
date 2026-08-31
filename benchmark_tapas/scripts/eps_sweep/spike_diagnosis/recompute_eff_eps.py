#!/usr/bin/env python3
"""[1] Recompute DPGAN's effective epsilon at eps=1.0 from scratch, and check TAPAS.

THE QUESTION
    The eps sweep reports eps_eff = 1.761 for DPGAN at eps=1.0 (1000/2500), against
    0.000 at 0.1, 10 and 100. Component [0] established the leakage is in the
    generated data rather than invented by the attacks. This component tests the one
    remaining link neither of those touch: the arithmetic that turns an attack's
    true/false positive counts into a certified bound.

    If an independent implementation reproduces 1.761 from the same raw scores,
    TAPAS's bound is sound and the spike is a property of the generator. If it does
    not, the answer is here and no GPU time is needed.

NO NEW DATA, NO GPU
    Everything is read off disk. TAPAS wrote every attack's per-dataset score and the
    true D+/D- label to raw_scores_dpgan_{stage}.csv, so both the counts and the bound
    can be rebuilt. All four stages of the eps=1.0 arm are used (50/100 through
    1000/2500), because a bound that behaves correctly as the sample size grows is
    better evidence than a single agreeing number.

WHAT TAPAS ACTUALLY COMPUTES (read from tapas/report/report.py)
    delta is not used -- the bound is the delta=0 form. For a threshold tau:

        eps_low  = max(0, log(TPR_low / FPR_high))
        eps_high = log(TPR_high / FPR_low)

    with Clopper-Pearson intervals on TPR and FPR, each taken at
    1 - (1-confidence)/2 so the confidence is split across the two estimates. An
    `inverse` flag swaps positives and negatives, giving the TN/FN branch.

    This script rebuilds that from the definition rather than importing it:
    Clopper-Pearson via the Beta quantile directly, both branches evaluated
    explicitly, and the whole thing swept over thresholds.

WHY A THRESHOLD SWEEP RATHER THAN ONE NUMBER
    TAPAS does not use the attack's own decision threshold. Its report re-thresholds
    the same scores on a held-out validation split, choosing tau to maximise the
    certifiable bound, and that tau never reaches the result CSV. So comparing a
    single number would be comparing two different thresholds and any gap would be
    uninformative.

    Instead this sweeps every threshold the scores admit and reports the BEST
    achievable eps_low. That upper-envelopes whatever tau TAPAS picked, which makes
    the comparison decisive in the direction that matters:

        TAPAS <= best achievable   -> its bound is attainable from this data. Sound.
        TAPAS >  best achievable   -> it reports a bound no threshold can support. Bug.

    The value at the attack's own threshold is reported alongside as a reference
    point, not as the comparison.

OUTPUT
    results/eps_sweep/spike_diagnosis/recomputed_eff_eps.csv
        one row per (stage, attack): TAPAS's bound, the best achievable bound, the
        threshold that achieves it, which branch won, and the bound at the attack's
        own threshold.

Run from the repo root (CPU, seconds):
  python benchmark_tapas/scripts/eps_sweep/spike_diagnosis/recompute_eff_eps.py
  python benchmark_tapas/scripts/eps_sweep/spike_diagnosis/recompute_eff_eps.py --confidence 0.9
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from config import RESULTS_DIR                                   # noqa: E402

METHOD = "dpgan"
STAGES = [(50, 100), (200, 500), (500, 1000), (1000, 2500)]
COUNTS_DIR = RESULTS_DIR / "counts_sweep"
OUT_DIR = RESULTS_DIR / "eps_sweep" / "spike_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "recomputed_eff_eps.csv"


def clopper_pearson(k: int, n: int, confidence: float):
    """Exact binomial interval for k successes in n trials.

    The textbook Beta-quantile form, written out rather than imported, since the
    point of this script is to not rely on the library being checked. Matches
    scipy's binomtest(...).proportion_ci, which is what TAPAS calls.
    """
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def eff_eps_at(scores, labels, tau: float, confidence: float, inverse: bool):
    """Certified (low, high) effective epsilon at one threshold, one branch.

    inverse=False scores the positives (TP/FP); inverse=True scores the negatives
    (TN/FN), which is the second branch of the differential-privacy inequality.
    TAPAS splits the confidence across the two rate estimates, so each Clopper-Pearson
    interval is taken at 1 - (1-confidence)/2; that is reproduced here.
    """
    positive_label = not inverse
    predicted = (scores <= tau) if inverse else (scores >= tau)

    pos = labels == positive_label
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0, np.inf

    tp = int(predicted[pos].sum())
    fp = int(predicted[neg].sum())

    half = 1 - (1 - confidence) / 2
    tpr_lo, tpr_hi = clopper_pearson(tp, n_pos, half)
    fpr_lo, fpr_hi = clopper_pearson(fp, n_neg, half)

    # tpr_lo of exactly 0 is normal (an attack with no certified true positives);
    # log(0) -> -inf -> clipped to 0 by the max. Silence the warning, keep the maths.
    with np.errstate(divide="ignore"):
        low = max(0.0, float(np.log(tpr_lo / fpr_hi))) if fpr_hi > 0 else np.inf
        high = float(np.log(tpr_hi / fpr_lo)) if fpr_lo > 0 else np.inf
    return low, high


def best_over_thresholds(scores, labels, confidence: float):
    """Sweep every threshold the scores admit, both branches, keep the best eps_low.

    Candidate thresholds are the distinct score values (plus a point beyond each end),
    which is sufficient: the bound only changes where the predicted set changes.
    """
    cands = np.unique(scores)
    step = np.diff(cands).min() if len(cands) > 1 else 1.0
    cands = np.concatenate([[cands[0] - step], cands, [cands[-1] + step]])

    best = {"eps_low": -1.0, "eps_high": np.nan, "tau": np.nan, "inverse": None}
    for inverse in (False, True):
        for tau in cands:
            low, high = eff_eps_at(scores, labels, float(tau), confidence, inverse)
            if np.isfinite(low) and low > best["eps_low"]:
                best = {"eps_low": low, "eps_high": high,
                        "tau": float(tau), "inverse": inverse}
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="confidence level for the certified bound (default 0.95)")
    args = ap.parse_args()
    conf = args.confidence
    tag = f"{int(round(conf * 100))}"

    rows = []
    for ntr, nte in STAGES:
        stage = f"{ntr}_{nte}"
        scores_path = COUNTS_DIR / f"raw_scores_{METHOD}_{stage}.csv"
        tapas_path = COUNTS_DIR / METHOD / stage / f"effeps_{METHOD}_{stage}.csv"
        if not scores_path.exists() or not tapas_path.exists():
            print(f"skipping {stage}: missing {scores_path.name} or {tapas_path.name}")
            continue

        sc = pd.read_csv(scores_path)
        sc["attack_short"] = sc.attack.str.split("(").str[0]
        tp_tab = pd.read_csv(tapas_path)
        tp_tab["attack_short"] = tp_tab.attack.str.split("(").str[0]

        for atk, grp in sc.groupby("attack_short"):
            scores = grp.raw_score.to_numpy(dtype=float)
            labels = grp.ground_truth.to_numpy().astype(bool)
            own_tau = float(grp.threshold.iloc[0])

            best = best_over_thresholds(scores, labels, conf)
            own_low, own_high = max(
                (eff_eps_at(scores, labels, own_tau, conf, inv) for inv in (False, True)),
                key=lambda t: t[0])

            ref = tp_tab[tp_tab.attack_short == atk]
            tapas_low = float(ref[f"eps_low_{tag}"].iloc[0]) if len(ref) else np.nan
            tapas_high = float(ref[f"eps_high_{tag}"].iloc[0]) if len(ref) else np.nan

            rows.append({
                "stage": f"{ntr}/{nte}", "attack": atk, "n_scored": len(grp),
                "tapas_eps_low": tapas_low, "tapas_eps_high": tapas_high,
                "best_eps_low": best["eps_low"], "best_eps_high": best["eps_high"],
                "best_tau": best["tau"], "best_branch": "TN/FN" if best["inverse"] else "TP/FP",
                "own_threshold": own_tau, "own_eps_low": own_low,
                # The decisive column: TAPAS should never exceed what any threshold
                # can support. A positive value here would mean it reports a bound
                # the data cannot justify.
                "tapas_exceeds_best": tapas_low - best["eps_low"],
                "confidence": conf,
            })

    if not rows:
        print("nothing to compare -- no counts-sweep raw scores found")
        return 1

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    show = ["stage", "attack", "tapas_eps_low", "best_eps_low", "best_tau",
            "best_branch", "own_eps_low", "tapas_exceeds_best"]
    print(f"DPGAN at eps=1.0 -- TAPAS's certified bound vs an independent recomputation "
          f"({int(conf*100)}% confidence)\n")
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    over = out[out.tapas_exceeds_best > 1e-9]
    print()
    if len(over):
        print(f"!! {len(over)} of {len(out)} cells: TAPAS reports MORE than any threshold "
              f"supports. Its bound is not attainable from these scores.")
        print(over[["stage", "attack", "tapas_eps_low", "best_eps_low"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print(f"OK: TAPAS's bound is attainable from the raw scores in all {len(out)} "
              f"cells -- its arithmetic is sound, and the eps=1.0 spike is a property "
              f"of the generated data, not of the bound.")
        gap = out.best_eps_low - out.tapas_eps_low
        print(f"   TAPAS sits {gap.min():.3f} to {gap.max():.3f} below the best achievable "
              f"bound, which is expected: it picks tau on a held-out split rather than "
              f"on the scores it reports.")
    print(f"\nwrote {OUT_CSV.relative_to(BENCHMARK_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
