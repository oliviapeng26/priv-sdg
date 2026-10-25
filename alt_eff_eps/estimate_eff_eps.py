#!/usr/bin/env python3
"""Recompute effective epsilon from saved TAPAS attack scores with two independent libraries.

NO NEW DATA
    Reads the raw_scores_*.csv that every audit already wrote (one row per test
    simulation: ground_truth 1 = target in training = member, raw_score higher = more
    member-like) and the matching effeps_*.csv for TAPAS's own EffectiveEpsilonReport
    numbers. Nothing is regenerated and TAPAS is not re-run.

METHODS (one output row per run x attack x confidence x method x delta)
    tapas_replica       TAPAS's report re-implemented: threshold + branch picked on the first
                        validation_split rows by position (cp heuristic at 90%), then the
                        two Clopper-Pearson intervals on the remaining rows, each at
                        1-(1-c)/2. Should equal the saved TAPAS value; it is the check that
                        the split and counts below are TAPAS's.
    jax_cp_split        jax_privacy CanaryScoreAuditor.epsilon_clopper_pearson, Split
                        (random partition, threshold_estimation_frac = TAPAS's formula),
                        significance = 1-c, one_sided=False.
    jax_cp_split_tapas_tails
                        Same, but significance = (1-c)/2. jax_privacy puts significance/2
                        on each ONE-sided rate bound; TAPAS takes two-SIDED intervals at
                        1-(1-c)/2, i.e. (1-c)/4 per used tail. Halving significance makes
                        the tails identical, so this row is the like-for-like comparison.
    jax_cp_bonferroni   Bonferroni over every (Pareto-frontier) threshold, all data.
    jax_cp_multisplit   MultiSplit: median of 100 Split bounds at significance/2.
    jax_raw_counts      ln(TPR/FPR) point estimate with no confidence; reference only. Same
                        value repeated on each confidence row so it joins like the others.
    jax_gdp             epsilon_from_gdp, AIM only (Gaussian mechanism), delta = 1e-9.
    pe_joint_beta       privacy-estimates joint-beta (Bayesian, Jeffreys priors on FNR/FPR),
    pe_clopper_pearson  and the Clopper-Pearson and Jeffreys rows estimate-epsilon.py prints,
    pe_jeffreys         all on TP/FP/TN/FN counted on the rows after TAPAS's validation
                        slice, at the threshold TAPAS's heuristic picks on that slice.
                        alpha = 1-c, as the script's "two-sided" row: eps_lo is then a
                        one-sided bound at (1-c)/2, the same budget TAPAS's eps_low spends.

    Not used: epsilon_one_run / epsilon_one_run_fdp. They assume canaries randomly included
    in ONE training run; here every simulation is a separate fit.

DELTA
    Primary rows use delta = 0, because TAPAS ignores delta. AIM (meta.json delta, 1e-9) and
    DP-CTGAN (SmartNoise's internal 1/n^1.5, n = 500 -> 8.94e-5) get a second row at their
    formal delta, labelled "formal". jax_gdp needs delta > 0 and uses 1e-9 only.

SAMPLE-SIZE CAP AND ZERO FALSE POSITIVES
    A finite sample cannot certify more than the bound it would give on perfectly separated
    scores of the same size. `cap` is exactly that: the same call on perfectly separated
    data with the same member/non-member counts. `capped` = eps_lower has reached it.
    `zero_fp` = no false positives in the branch used (for the TN/FN branch, no false
    negatives); the lower bound stays finite, the upper bound becomes inf. jax rows do not
    expose counts, so for them only `capped` is reported. joint-beta cannot find eps above
    10 (its internal root bracket) and errors when FP or FN is 0, because its upper bound
    is inf; then only eps_lo is computed and eps_upper is inf.

RUNS
    Default = Tier 1 (headline) + Tier 2 (smaller sample sizes) + secondary (seed_fix=False
    DPGAN arms, whose fits share training noise, so the CI independence assumption is
    broken; labelled in `ci_caveat`). Pass --folders to run any other run folders
    (tuning/, extras/, ...): a folder needs one effeps_*.csv, and its raw_scores_*.csv is
    looked up in the folder, its parent, then its grandparent.

Run from the repo root, in the alt_eff_eps venv (see requirements.txt):
  alt_eff_eps/.venv/bin/python alt_eff_eps/estimate_eff_eps.py
  alt_eff_eps/.venv/bin/python alt_eff_eps/estimate_eff_eps.py --tiers 1
  alt_eff_eps/.venv/bin/python alt_eff_eps/estimate_eff_eps.py --folders tuning/ctgan/bs100/1000_2500
"""

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from jax_privacy import auditing
from privacy_estimates import AttackResults, Beta, compute_eps_lo_hi

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*IntegrationWarning.*")
warnings.filterwarnings("ignore", module="scipy.integrate")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmark_tapas" / "results"
OUT_CSV = Path(__file__).resolve().parent / "results" / "alt_eff_eps.csv"

CONFIDENCE_LEVELS = (0.9, 0.95, 0.99)   # common.py's EffectiveEpsilonReport call
SELECTION_CONFIDENCE = 0.9              # EffectiveEpsilonReport._select_attack_cp default
SEED = 0                                # Split / MultiSplit partitions
MULTISPLIT_SAMPLES = 100
# epsilon_raw_counts only considers thresholds with >= min_count FP (and FN). The default 50
# is 4% of each class at 1000/2500 but 20% at 200/500 and all of it at 50/100 (where it
# returns 0), so keep the default's 4% and scale it with n, floor 5. 1% was tried: on the
# Tier 1 cells it raised the null median from 0.25 to 0.43 (low-FPR noise, upward bias).
MIN_COUNT_FRAC, MIN_COUNT_FLOOR = 0.04, 5
CAP_TOL = 0.05      # eps_lower within this of the cap counts as capped (threshold transfer
                    # from the validation slice can drop a member or two)
NUM_SYNTHETIC = 500                     # records each DP-CTGAN fit trains on
GDP_DELTA = 1e-9                        # AIM's formal delta

SS, EPS = "sample_size_sweep", "eps_sweep"
TIERS = {
    "1": [f"{SS}/{g}/1000_2500" for g in ("bayesian_network", "ctgan", "aim", "dp_ctgan")]
         + [f"{EPS}/aim/eps{e}/1000_2500" for e in ("0.1", "10")]
         + [f"{EPS}/dp_ctgan/eps{e}/1000_2500" for e in ("0.1", "10", "100")]
         + [f"{EPS}/dpgan/eps{e}_seedfix" for e in ("0.1", "0.5", "1", "2", "5", "10", "100")],
    "2": [f"{SS}/{g}/{s}" for g in ("bayesian_network", "ctgan", "aim")
          for s in ("50_100", "200_500", "500_1000")],
    "secondary": [f"{EPS}/dpgan/eps{e}" for e in ("0.1", "0.5", "1", "2", "5", "10", "100")]
                 + [f"{EPS}/dpgan/eps1_{t}" for t in ("seedbase10000", "tseed44", "bg44")]
                 + [f"{SS}/dpgan/{s}" for s in ("50_100", "200_500", "500_1000", "1000_2500")],
}
CI_CAVEAT = "seed_fix=False: fits share training noise, so CI independence is violated"
# jax_privacy 2.0.0 epsilon_from_gdp takes max |mu| over thresholds. At near-trivial thresholds
# (e.g. FN = n-1, FP = 0) FPR_ub + FNR_ub > 1 gives mu < 0, which upper bounds cannot support
# in reverse, yet abs() counts it: on AIM eps10 ClosestDistance, mu = -2.07 -> eps 14 at
# delta 1e-9 while every CP method gives 0.
GDP_NOTE = "UNRELIABLE: library takes abs(mu); inflated by trivial thresholds"


# -- TAPAS's report, re-implemented (tapas/report/report.py) ------------------

def tapas_cp(scores, labels, threshold, confidence, inverse):
    """EffectiveEpsilonReport._estimate_effective_epsilon, line for line."""
    half = 1 - (1 - confidence) / 2
    positive_label = not inverse
    predicted = (scores <= threshold) if inverse else (scores >= threshold)
    pos = labels == positive_label
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    tp, fp = int(predicted[pos].sum()), int(predicted[~pos].sum())
    ci_tpr = binomtest(k=tp, n=n_pos, p=tp / n_pos).proportion_ci(half)
    ci_fpr = binomtest(k=fp, n=n_neg, p=fp / n_neg).proportion_ci(half)
    low = max(0, np.log(ci_tpr.low / ci_fpr.high)) if ci_fpr.high > 0 else np.inf
    high = np.log(ci_tpr.high / ci_fpr.low) if ci_fpr.low > 0 else np.inf
    return float(low), float(high)


def tapas_select(scores, labels):
    """EffectiveEpsilonReport._select_attack_cp for one attack: (threshold, inverse)."""
    best_eps, best = -1, None
    for threshold in np.sort(np.unique(scores)):
        for inverse in (False, True):
            low = tapas_cp(scores, labels, threshold, SELECTION_CONFIDENCE, inverse)[0]
            if not np.isnan(low) and low > best_eps:
                best_eps, best = low, (float(threshold), inverse)
    return best


def validation_split(n):
    return min(0.5, max(0.1, 15 / n))           # common.py:515


# -- Estimators --------------------------------------------------------------

_CAPS = {}


def cap_of(key, fn, n_in, n_out):
    """fn evaluated on perfectly separated data of the same size (memoised)."""
    key = (key, n_in, n_out)
    if key not in _CAPS:
        try:
            _CAPS[key] = float(fn(np.ones(n_in), np.zeros(n_out)))
        except Exception:
            _CAPS[key] = np.nan
    return _CAPS[key]


def pe_estimate(count, delta, alpha, method):
    """privacy-estimates (eps_lo, eps_hi, note).

    joint-beta special cases, both for cost (~100 s a call otherwise, against ~4 s):
    - trivial rule (the threshold puts every row in one class, which TAPAS's heuristic picks
      when no threshold certifies anything): the counts carry no membership signal, so
      eps_lo = 0 (TAPAS also reports 0) and joint-beta is not called.
    - any zero count: its eps_hi needs eps > 10 (the root bracket) and errors, so eps_lo
      alone is computed, at alpha/2 as eps_lo_hi would, and eps_hi = inf.
    """
    if method == "joint-beta":
        if count.TP + count.FP == 0 or count.TN + count.FN == 0:
            return 0.0, np.nan, "trivial rule (one class for every row): joint-beta skipped, 0"
        if min(count.TP, count.FP, count.FN, count.TN) == 0:
            try:
                return float(Beta(count).eps_lo(delta=delta, alpha=alpha / 2)), np.inf, \
                    "zero count: eps_hi unbounded (joint-beta root bracket); eps_lo only"
            except Exception as e:
                return np.nan, np.inf, f"failed: {e}"
    try:
        lo, hi = compute_eps_lo_hi(count=count, delta=delta, alpha=alpha, method=method)
        return float(lo), float(hi), ""
    except Exception as e:
        return np.nan, np.nan, f"failed: {e}"


def estimate_cell(scores, labels, formal_delta=None, gdp=False):
    """All method rows for one attack's scores. labels: True = member. Rows carry no run
    metadata; the caller adds it."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n = len(scores)
    frac = validation_split(n)
    k = int(frac * n)
    deltas = [(0.0, "0 (TAPAS-equivalent)")]
    if formal_delta:
        deltas.append((float(formal_delta), "formal"))
    rows = []

    # Threshold on the validation slice, counts on the rest (TAPAS's positional split).
    val_s, val_l, ev_s, ev_l = scores[:k], labels[:k], scores[k:], labels[k:]
    threshold, inverse = tapas_select(val_s, val_l)
    pred = (ev_s > threshold) if inverse else (ev_s >= threshold)   # predicted member
    tp, fp = int((pred & ev_l).sum()), int((pred & ~ev_l).sum())
    fn, tn = int((~pred & ev_l).sum()), int((~pred & ~ev_l).sum())
    count = AttackResults(FN=fn, FP=fp, TN=tn, TP=tp)
    perfect = AttackResults(FN=0, FP=0, TN=fp + tn, TP=tp + fn)
    counted = dict(threshold_strategy=f"positional first {frac:.3g} (TAPAS cp heuristic)",
                   n_members=tp + fn, n_nonmembers=fp + tn, n_select=k,
                   TP=tp, FP=fp, TN=tn, FN=fn, branch="TN/FN" if inverse else "TP/FP",
                   threshold=threshold, zero_fp=(fn == 0) if inverse else (fp == 0))

    in_s, out_s = scores[labels], scores[~labels]
    n_in, n_out = len(in_s), len(out_s)
    auditor = auditing.CanaryScoreAuditor(in_s, out_s)
    n_in_ev, n_out_ev = n_in - int(n_in * frac), n_out - int(n_out * frac)
    min_count = max(MIN_COUNT_FLOOR, math.ceil(MIN_COUNT_FRAC * min(n_in, n_out)))

    def jax_cp(strategy, sig, delta):
        return lambda i, o: auditing.CanaryScoreAuditor(i, o).epsilon_clopper_pearson(
            sig, delta, one_sided=False, threshold_strategy=strategy)

    for c in CONFIDENCE_LEVELS:
        for delta, dlabel in deltas:
            base = dict(confidence=c, delta=delta, delta_label=dlabel)

            if delta == 0:   # TAPAS has no delta, so the replica has one row
                lo, hi = tapas_cp(ev_s, ev_l, threshold, c, inverse)
                cap = tapas_cp(np.r_[np.ones(n_in_ev), np.zeros(n_out_ev)],
                               np.r_[np.ones(n_in_ev, bool), np.zeros(n_out_ev, bool)],
                               0.5, c, False)[0]
                rows.append(dict(base, method="tapas_replica", significance=1 - c,
                                 eps_lower=lo, eps_upper=hi, cap=cap, **counted))

            for name, sig, strategy, label, n_ev in [
                ("jax_cp_split", 1 - c, auditing.Split(frac, seed=SEED),
                 f"Split(frac={frac:.3g}, seed={SEED})", (n_in_ev, n_out_ev)),
                ("jax_cp_split_tapas_tails", (1 - c) / 2, auditing.Split(frac, seed=SEED),
                 f"Split(frac={frac:.3g}, seed={SEED})", (n_in_ev, n_out_ev)),
                ("jax_cp_bonferroni", 1 - c, auditing.Bonferroni(), "Bonferroni()",
                 (n_in, n_out)),
                ("jax_cp_multisplit", 1 - c,
                 auditing.MultiSplit(MULTISPLIT_SAMPLES, frac, seed=SEED),
                 f"MultiSplit(n={MULTISPLIT_SAMPLES}, frac={frac:.3g}, seed={SEED})",
                 (n_in_ev, n_out_ev)),
            ]:
                fn_ = jax_cp(strategy, sig, delta)
                rows.append(dict(base, method=name, significance=sig, threshold_strategy=label,
                                 eps_lower=float(fn_(in_s, out_s)),
                                 cap=cap_of((name, sig, delta, label), fn_, n_in, n_out),
                                 n_members=n_ev[0], n_nonmembers=n_ev[1]))

            raw = lambda i, o: auditing.CanaryScoreAuditor(i, o).epsilon_raw_counts(
                min_count, delta, one_sided=False)
            rows.append(dict(base, method="jax_raw_counts", threshold_strategy="all thresholds",
                             eps_lower=float(raw(in_s, out_s)), min_count=min_count,
                             cap=cap_of(("raw", min_count, delta), raw, n_in, n_out),
                             n_members=n_in, n_nonmembers=n_out,
                             note="point estimate, no confidence; reference only"))

            for name, method in [("pe_joint_beta", "joint-beta"),
                                 ("pe_clopper_pearson", "beta"), ("pe_jeffreys", "jeffreys")]:
                lo, hi, note = pe_estimate(count, delta, 1 - c, method)
                key = ("pe", method, 1 - c, delta, perfect.TP, perfect.TN)
                if key not in _CAPS:
                    _CAPS[key] = pe_estimate(perfect, delta, 1 - c, method)[0]
                rows.append(dict(base, method=name, significance=1 - c, eps_lower=lo,
                                 eps_upper=hi, cap=_CAPS[key], note=note, **counted))

        if gdp:
            g = lambda i, o: auditing.CanaryScoreAuditor(i, o).epsilon_from_gdp(1 - c, GDP_DELTA)
            rows.append(dict(confidence=c, delta=GDP_DELTA, delta_label="formal (GDP)",
                             method="jax_gdp", significance=1 - c,
                             threshold_strategy="Bonferroni over 2n (built in)",
                             eps_lower=float(g(in_s, out_s)),
                             cap=cap_of(("gdp", 1 - c), g, n_in, n_out),
                             n_members=n_in, n_nonmembers=n_out, note=GDP_NOTE))

    # Plain rank AUC; < 0.5 means the higher=member orientation fails. (auditor.attack_auroc
    # uses the ROC convex hull, which is floored at 0.5, so it cannot show a flip.)
    ranks = rankdata(scores)
    auc = float((ranks[labels].sum() - n_in * (n_in + 1) / 2) / (n_in * n_out))
    for r in rows:
        r["auc"] = auc
        r["capped"] = bool(np.isfinite(r["cap"]) and r["cap"] > 0
                           and r["eps_lower"] >= r["cap"] - CAP_TOL)
    return rows


# -- Runs --------------------------------------------------------------------

def load_run(folder):
    """(scores DataFrame, TAPAS effeps DataFrame, meta dict) for one run folder."""
    effeps = list(folder.glob("effeps_*.csv"))
    if len(effeps) != 1:
        raise FileNotFoundError(f"{folder}: expected one effeps_*.csv, found {len(effeps)}")
    raw_name = "raw_scores_" + effeps[0].name[len("effeps_"):]
    raw = next((d / raw_name for d in (folder, folder.parent, folder.parent.parent)
                if (d / raw_name).exists()), None)
    if raw is None:
        raise FileNotFoundError(f"{folder}: no {raw_name} in the folder or its two parents")
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pd.read_csv(raw), pd.read_csv(effeps[0]), meta


def run_rows(rel, tier):
    scores, tapas, meta = load_run(RESULTS_DIR / rel)
    generator = scores.generator.iloc[0]
    seed_fix = meta.get("seed_fix", meta.get("config", {}).get("seed_fix"))
    if seed_fix is None and generator == "dpgan":
        seed_fix = False        # runs from before the flag existed (run_dpgan_eps_sweep.py)
    formal_delta = {"aim": meta.get("delta"), "dpctgan": 1 / NUM_SYNTHETIC ** 1.5}.get(generator)
    tapas["attack_short"] = tapas.attack.str.split("(").str[0]
    run = dict(tier=tier, headline=tier == "1", run=rel, generator=generator,
               formal_epsilon=tapas.formal_epsilon.iloc[0] if "formal_epsilon" in tapas else None,
               seed_fix=seed_fix, ci_caveat=CI_CAVEAT if seed_fix is False else "",
               num_train=int(tapas.num_train.iloc[0]), num_test=int(tapas.num_test.iloc[0]))

    rows = []
    for attack, grp in scores.groupby("attack", sort=False):
        short = attack.split("(")[0]
        ref = tapas[tapas.attack_short == short]
        for r in estimate_cell(grp.raw_score.to_numpy(), grp.ground_truth.to_numpy() == 1,
                               formal_delta=formal_delta, gdp=generator == "aim"):
            c = int(round(r["confidence"] * 100))
            rows.append({**run, "attack": short, **r,
                         "tapas_eps_lower": float(ref[f"eps_low_{c}"].iloc[0]) if len(ref) else np.nan,
                         "tapas_eps_upper": float(ref[f"eps_high_{c}"].iloc[0]) if len(ref) else np.nan})
    return rows


COLUMNS = ["tier", "headline", "run", "generator", "formal_epsilon", "seed_fix", "ci_caveat",
           "num_train", "num_test", "attack", "confidence", "method", "threshold_strategy",
           "significance", "delta", "delta_label", "eps_lower", "eps_upper", "tapas_eps_lower",
           "tapas_eps_upper", "n_members", "n_nonmembers", "n_select", "TP", "FP", "TN", "FN",
           "branch", "threshold", "zero_fp", "cap", "capped", "min_count", "auc", "note"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", nargs="+", choices=list(TIERS), default=list(TIERS))
    ap.add_argument("--folders", nargs="+", help="run folders relative to "
                    "benchmark_tapas/results/ (overrides --tiers; tier = custom)")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--workers", type=int, default=8, help="runs processed in parallel")
    args = ap.parse_args()

    runs = ([(f, "custom") for f in args.folders] if args.folders
            else [(f, t) for t in args.tiers for f in TIERS[t]])
    print(f"raw_counts min_count = max({MIN_COUNT_FLOOR}, ceil({MIN_COUNT_FRAC} x smaller class)); "
          f"Split/MultiSplit seed = {SEED}; {len(runs)} runs, {args.workers} workers", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, ((rel, tier), r) in enumerate(
                zip(runs, pool.map(run_rows, *zip(*runs))), 1):
            rows.extend(r)
            print(f"  [{i}/{len(runs)}] tier {tier:9s} {rel}  "
                  f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} rows to {args.out}  ({(time.time() - t0) / 60:.1f} min)")

    rep = out[(out.method == "tapas_replica")]
    gap = (rep.eps_lower - rep.tapas_eps_lower).abs()
    print(f"tapas_replica vs saved TAPAS eps_low: max |diff| = {gap.max():.2e} "
          f"over {gap.notna().sum()} cells")
    if (gap > 1e-6).any():
        print(rep[gap > 1e-6][["run", "attack", "confidence", "eps_lower", "tapas_eps_lower"]]
              .to_string(index=False))

    show = out[(out.confidence == 0.95) & (out.delta_label != "formal")
               & (out.headline | (out.tier == "custom"))]
    if len(show):
        table = show.pivot_table(index=["run", "attack"], columns="method", values="eps_lower",
                                 aggfunc="first", sort=False)
        tapas = show.groupby(["run", "attack"], sort=False).tapas_eps_lower.first()
        table.insert(0, "TAPAS", tapas)
        table.index = table.index.set_levels(
            table.index.levels[0].str.replace(f"{SS}/", "ss/").str.replace(f"{EPS}/", "eps/"),
            level=0)
        table.columns = [c.replace("jax_cp_", "jax_").replace("_tapas_tails", "_tt")
                         for c in table.columns]
        print("\neps_lower at 95%, delta=0 (jax_gdp at delta=1e-9) -- headline runs; "
              "jax_split_tt = jax Split at TAPAS's tail allocation")
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            print(table.round(3).to_string())
        n_cap = show[show.capped].groupby("method").size()
        if len(n_cap):
            print("\ncells at the sample-size cap:", n_cap.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
