#!/usr/bin/env python3
"""Phase 3 of the formal-epsilon sweep: one table, one row per eps.

Joins the three halves of the sweep -- the DP accounting, the utility/fidelity
arm, and the TAPAS audit -- into a single wide summary, and emits the per-attack
long table the notebook plots from.

WHERE EACH EPS COMES FROM
    eps in {0.1, 10, 100}   this sweep: results/eps_sweep/privacy/eps{eps}/
    eps = 1.0               reused from the counts sweep, which already audited
                            dpgan at exactly 1000/2500 under the same threat model
                            and the same per-fit seeding:
                              results/counts_sweep/dpgan/1000_2500/effeps_dpgan_1000_2500.csv

    Mixing the two sources is safe precisely because run_eps_sweep.py is
    run_counts_sweep.py with the loop swapped: same background, same target,
    same SCORE_ATTACK_SEED, same battery, same counts. The one thing that is NOT
    comparable is wall clock -- see below.

WALL CLOCK IS SPLIT INTO TWO COLUMNS, DELIBERATELY
    wall_clock_privacy_attacks_s   sum of the 5 attacks' wall_time_s. Comparable
                                   across every eps, including 1.0.
    wall_clock_privacy_total_s     that plus the time spent generating the 3,500
                                   simulated datasets. NaN at eps=1.0: the counts
                                   sweep built that pool incrementally across four
                                   nested stages, so no single figure for "the cost
                                   of the 1000/2500 pool" exists. Reporting the
                                   attack-only column as if it were the total would
                                   understate eps=1.0 by ~4 h against the others.

WORST CASE IS ARGMAX OVER eps_low_95, NOT OVER AUC
    The claim the sweep makes is a certified lower bound on leakage, so the attack
    that matters is the one with the highest certifiable bound -- not the one that
    separates member from non-member best on average. In the counts sweep those
    coincide (Groundhog wins both at every cell), but they need not.

DEGRADES RATHER THAN FAILS
    Every input is optional. Missing utility, missing privacy, missing sigma table
    each leave their columns NaN with a warning, so this is runnable mid-sweep to
    watch arms land instead of only at the end.

Outputs:
    benchmark_tapas/results/eps_sweep/eps_sweep_summary.csv        one row per eps
    benchmark_tapas/results/eps_sweep/privacy/eps_sweep_privacy_per_attack.csv  long form

Run from the repo root, after the other three phases:
  python benchmark_tapas/scripts/eps_sweep_aggregate.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# benchmark_tapas/, found by walking up to config.py rather than counting parents --
# these scripts live in a subfolder now and may move again.
BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import RESULTS_DIR                                    # noqa: E402

METHOD = "dpgan"
SWEEP_EPSILONS = [0.1, 1.0, 10.0, 100.0]
REUSED_EPSILON = 1.0
NUM_TRAIN, NUM_TEST = 1000, 2500

SWEEP_DIR = RESULTS_DIR / "eps_sweep"
PRIVACY_DIR = SWEEP_DIR / "privacy"
COUNTS_DIR = RESULTS_DIR / "counts_sweep"

SIGMA_CSV = SWEEP_DIR / "noise_multipliers.csv"
UTIL_SUMMARY_CSV = SWEEP_DIR / "utility_fidelity_summary.csv"
COST_CSV = SWEEP_DIR / "generation_cost.csv"
SUMMARY_CSV = SWEEP_DIR / "eps_sweep_summary.csv"
PER_ATTACK_CSV = PRIVACY_DIR / "eps_sweep_privacy_per_attack.csv"

FIDELITY_METRICS = ["KSComplement", "TVComplement",
                    "CorrelationSimilarity", "ContingencySimilarity"]
UTILITY_MODELS = ["xgboost", "logistic_regression"]

SWEEP_DIR.mkdir(parents=True, exist_ok=True)
PRIVACY_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SWEEP_DIR / "eps_sweep_aggregate_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("eps_sweep_aggregate")


def eps_slug(eps: float) -> str:
    """Must match run_eps_sweep.eps_slug / eps_sweep_generate.eps_slug."""
    return f"eps{eps:g}"


def short_attack(label: str) -> str:
    """'LocalNeighbourhood(L_2, 0.2504..., accuracy)' -> 'LocalNeighbourhood'.

    The full TAPAS labels embed the radius and the criterion, which makes them
    unusable as column-name stems and different between runs if the radius ever
    moves. The family name is what the comparison is over.
    """
    return label.split("(")[0]


def privacy_paths(eps: float) -> Path:
    """The per-attack effeps CSV for one eps, wherever it lives."""
    if eps == REUSED_EPSILON:
        return COUNTS_DIR / METHOD / f"{NUM_TRAIN}_{NUM_TEST}" / \
            f"effeps_{METHOD}_{NUM_TRAIN}_{NUM_TEST}.csv"
    return PRIVACY_DIR / eps_slug(eps) / f"effeps_{METHOD}_{eps_slug(eps)}.csv"


def load_privacy(eps: float):
    """Per-attack privacy rows for one eps, or None if that arm has not landed."""
    path = privacy_paths(eps)
    if not path.exists():
        log.warning(f"eps={eps:g}: no privacy table at "
                    f"{path.relative_to(REPO_ROOT)} -- privacy columns left NaN")
        return None
    df = pd.read_csv(path)
    df["formal_epsilon"] = eps
    df["attack_short"] = df["attack"].map(short_attack)
    df["source"] = "counts_sweep" if eps == REUSED_EPSILON else "eps_sweep"
    # counts_sweep predates these columns; fill so the long table has one schema.
    for col, val in (("replacement_epsilon", 2 * eps), ("degenerate_pool", False),
                     ("eff_eps_tau", np.nan), ("eff_eps_tau_inverse", np.nan),
                     ("eff_eps_validation_split", np.nan)):
        if col not in df.columns:
            df[col] = val
    if "tn" not in df.columns:
        df["tn"] = 1.0 - df["fp"]
    if "fn" not in df.columns:
        df["fn"] = 1.0 - df["tp"]
    return df


def load_meta(eps: float) -> dict:
    """Timings + guard fractions for one eps. Only run_eps_sweep.py writes these."""
    if eps == REUSED_EPSILON:
        return {}
    path = PRIVACY_DIR / eps_slug(eps) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def utility_row(util: pd.DataFrame, eps: float):
    if util is None:
        return None
    row = util[np.isclose(util.formal_epsilon, eps)]
    if row.empty:
        log.warning(f"eps={eps:g}: no utility/fidelity row -- those columns left NaN")
        return None
    return row.iloc[0]


def sigma_row(sig: pd.DataFrame, eps: float):
    if sig is None:
        return None
    row = sig[np.isclose(sig.formal_epsilon, eps)]
    if row.empty:
        log.warning(f"eps={eps:g}: not in noise_multipliers.csv -- sigma/delta left NaN. "
                    f"Re-run eps_sweep_sigma_check.py --epsilons {eps:g}")
        return None
    return row.iloc[0]


def generation_seconds(cost: pd.DataFrame, eps: float) -> float:
    """Mean per-seed generation wall clock for one eps, or NaN."""
    if cost is None or "formal_epsilon" not in cost.columns:
        return float("nan")
    g = cost[(np.isclose(cost.formal_epsilon, eps)) & (cost.stage == "generation")]
    return float(g.wall_clock_s.mean()) if len(g) else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SWEEP_EPSILONS,
                        help=f"budgets to aggregate (default: {SWEEP_EPSILONS})")
    args = parser.parse_args()

    def read(path, what):
        if path.exists():
            return pd.read_csv(path)
        log.warning(f"{path.relative_to(REPO_ROOT)} missing -- {what} columns left NaN")
        return None

    sig = read(SIGMA_CSV, "sigma/delta")
    util = read(UTIL_SUMMARY_CSV, "utility/fidelity")
    cost = read(COST_CSV, "generation wall-clock")

    rows, per_attack = [], []
    for eps in args.epsilons:
        row = {"method": METHOD, "formal_epsilon": eps, "replacement_epsilon": 2 * eps,
               "num_train": NUM_TRAIN, "num_test": NUM_TEST,
               "source": "counts_sweep" if eps == REUSED_EPSILON else "eps_sweep"}

        s = sigma_row(sig, eps)
        if s is not None:
            for c in ("sigma_audit", "sigma_utility", "delta_audit", "delta_utility",
                      "sigma_audit_nominal", "sigma_utility_nominal",
                      "sample_rate_audit", "sample_rate_utility", "accountant"):
                row[c] = s.get(c, np.nan)

        u = utility_row(util, eps)
        if u is not None:
            n_runs = u.get("n_runs", np.nan)
            row["n_runs"] = int(n_runs) if pd.notna(n_runs) else np.nan
            for model in UTILITY_MODELS:
                short = "xgb" if model == "xgboost" else "lr"
                row[f"tstr_auc_{short}_mean"] = u.get(f"tstr_{model}_auc_mean", np.nan)
                row[f"tstr_auc_{short}_std"] = u.get(f"tstr_{model}_auc_std", np.nan)
                row[f"trtr_auc_{short}"] = u.get(f"trtr_{model}_auc_mean", np.nan)
                row[f"retention_{short}_mean"] = u.get(f"retention_{model}_mean", np.nan)
            for metric in FIDELITY_METRICS:
                row[f"{metric}_mean"] = u.get(f"{metric}_mean", np.nan)
                row[f"{metric}_std"] = u.get(f"{metric}_std", np.nan)

        priv = load_privacy(eps)
        meta = load_meta(eps)
        if priv is not None:
            per_attack.append(priv)
            for _, r in priv.iterrows():
                stem = short_attack(r["attack"])
                row[f"{stem}_auc"] = r["auc"]
                row[f"{stem}_tp"] = r["tp"]
                row[f"{stem}_fp"] = r["fp"]
                row[f"{stem}_advantage"] = r["mia_advantage"]
                row[f"{stem}_eff_eps_low"] = r["eps_low_95"]
                row[f"{stem}_eff_eps_high"] = r["eps_high_95"]
            best = priv.loc[priv["eps_low_95"].idxmax()]
            row["worst_case_eff_eps_low"] = float(best["eps_low_95"])
            row["worst_case_eff_eps_high"] = float(best["eps_high_95"])
            row["worst_case_attack"] = short_attack(best["attack"])
            row["mean_mia_advantage"] = float(priv["mia_advantage"].mean())
            row["max_mia_auc"] = float(priv["auc"].max())
            row["degenerate_pool"] = bool(priv["degenerate_pool"].any())
            row["wall_clock_privacy_attacks_s"] = float(priv["wall_time_s"].sum())
            # NaN at eps=1.0 on purpose: its pool was built across four nested
            # counts stages, so there is no single comparable figure. See docstring.
            row["wall_clock_privacy_total_s"] = meta.get("total_wall_clock_s", np.nan)
            row["wall_clock_privacy_pool_s"] = meta.get("pool_wall_clock_s", np.nan)

        row["wall_clock_generation_s_mean"] = generation_seconds(cost, eps)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("formal_epsilon").reset_index(drop=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    log.info(f"Wrote {SUMMARY_CSV.relative_to(REPO_ROOT)} "
             f"({len(summary)} rows x {len(summary.columns)} columns)")

    if per_attack:
        long = pd.concat(per_attack, ignore_index=True).sort_values(
            ["formal_epsilon", "attack_short"]).reset_index(drop=True)
        long.to_csv(PER_ATTACK_CSV, index=False)
        log.info(f"Wrote {PER_ATTACK_CSV.relative_to(REPO_ROOT)} ({len(long)} rows)")

    show = [c for c in ["formal_epsilon", "replacement_epsilon", "sigma_audit",
                        "sigma_utility", "tstr_auc_xgb_mean", "tstr_auc_xgb_std",
                        "KSComplement_mean", "worst_case_eff_eps_low",
                        "worst_case_eff_eps_high", "worst_case_attack",
                        "mean_mia_advantage", "degenerate_pool"]
            if c in summary.columns]
    print("\n=== DPGAN formal-epsilon sweep ===")
    print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    missing = summary[summary.get("worst_case_eff_eps_low", pd.Series(dtype=float)).isna()] \
        if "worst_case_eff_eps_low" in summary.columns else summary
    if len(missing):
        print(f"\nStill missing privacy: eps = "
              f"{', '.join(f'{e:g}' for e in missing.formal_epsilon)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
