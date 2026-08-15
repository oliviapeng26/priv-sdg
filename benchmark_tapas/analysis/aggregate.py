#!/usr/bin/env python3
"""Aggregate the benchmark into report-ready summary tables.

Joins three sources (privacy from this experiment; utility + fidelity already
computed elsewhere in the repo) into:
  results/summary_privacy.csv   -- per method: worst-case & mean MIA AUC across
                                   the 5 attacks, worst-case eff-epsilon (max
                                   eps_low_95, the tightest defensible lower
                                   bound on leakage), which attack was worst.
  results/summary_utility.csv   -- per method: in-house TSTR/TRTR/retention
                                   (results/utility_summary.csv) + SDMetrics
                                   fidelity (results/fidelity_summary.csv),
                                   both as mean +/- std over seeds.RUN_SEEDS.
  results/summary_combined.csv  -- one row per method with the tradeoff axes
                                   (tstr_xgboost_auc_mean + worst_case_auc)
                                   side by side.

Run from repo root, after the four run_*.py have populated
benchmark_privacy_per_attack.csv and evaluation/eval_{utility,fidelity}.py have
populated the utility/fidelity tables (missing inputs degrade to NaN columns,
they do not fail):
  python benchmark_tapas/aggregate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/ (script now in a subfolder)
from config import (
    TABLES_DIR, METHOD_CONFIG, METHODS, DP_METHODS,
    UTILITY_RESULTS, FIDELITY_RESULTS,
)

PER_ATTACK = TABLES_DIR / "benchmark_privacy_per_attack.csv"

# Pretty labels for tables/plots.
LABELS = {
    "bayesian_network": "BN (no-DP)",
    "privbayes": "PrivBayes (DP)",
    "ctgan": "CTGAN (no-DP)",
    "dpgan": "DPGAN (DP)",
}


def summarize_privacy() -> pd.DataFrame:
    if not PER_ATTACK.exists():
        print(f"[warn] {PER_ATTACK} not found -- run the run_*.py scripts first.")
        return pd.DataFrame()
    df = pd.read_csv(PER_ATTACK)
    rows = []
    for method, g in df.groupby("method"):
        worst = g.loc[g["auc"].idxmax()]
        # Worst-case eff-epsilon = highest 95% lower bound across attacks.
        eps_low = g["eps_low_95"] if "eps_low_95" in g else pd.Series(dtype=float)
        max_eps_low = float(eps_low.max()) if len(eps_low) else np.nan
        eps_high_at_max = np.nan
        if len(eps_low) and eps_low.notna().any():
            eps_high_at_max = float(g.loc[eps_low.idxmax(), "eps_high_95"])
        rows.append({
            "method": method,
            "label": LABELS.get(method, method),
            "dp": bool(g["dp"].iloc[0]),
            "kind": g["kind"].iloc[0],
            "n_attacks": len(g),
            "worst_case_auc": float(g["auc"].max()),
            "worst_auc_attack": worst["attack"],
            "mean_auc": float(g["auc"].mean()),
            # Mean membership advantage (tp-fp) across the 5 attacks: the privacy
            # axis of the tradeoff plot. Unlike mean AUC it anchors "no leakage"
            # (random/broken attack) at 0 rather than 0.5, so a method isn't
            # penalised for successfully resisting an attack. Here it equals
            # (# attacks that fully separate)/5 since advantage is binary {0,1}.
            "mean_advantage": float(g["mia_advantage"].mean()),
            "n_attacks_succeed": int((g["mia_advantage"] > 0.5).sum()),
            "worst_case_eff_eps_low95": max_eps_low,
            "worst_case_eff_eps_high95": eps_high_at_max,
            "n_pointwise_inf": int(g.get("eff_epsilon_pointwise_is_inf", pd.Series(dtype=bool)).sum()),
            "formal_epsilon": g["formal_epsilon"].iloc[0] if g["dp"].iloc[0] else np.nan,
            # eff-eps gap (only meaningful for DP): worst-case empirical eps_low_95
            # minus the formal budget. Positive => empirical lower bound EXCEEDS the
            # claimed budget (apparent under-protection); negative => the eps_eff<<eps
            # gap. For non-DP methods there is no formal eps to compare against.
            "eff_eps_gap": (max_eps_low - float(g["formal_epsilon"].iloc[0]))
                           if g["dp"].iloc[0] else np.nan,
            "num_train": METHOD_CONFIG.get(method, {}).get("num_train"),
            "num_test": METHOD_CONFIG.get(method, {}).get("num_test"),
        })
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    out = out.sort_values("method", key=lambda s: s.map(order)).reset_index(drop=True)
    out.to_csv(TABLES_DIR / "summary_privacy.csv", index=False)
    print(f"Wrote {TABLES_DIR / 'summary_privacy.csv'}")
    return out


# Columns pulled from the multi-run tables. Every one carries a _std alongside,
# because both source tables aggregate over seeds.RUN_SEEDS.
UTILITY_COLS = [
    "tstr_xgboost_auc", "trtr_xgboost_auc", "retention_xgboost",
    "tstr_logistic_regression_auc", "retention_logistic_regression",
]
FIDELITY_COLS = ["KSComplement", "TVComplement",
                 "CorrelationSimilarity", "ContingencySimilarity"]


def summarize_utility() -> pd.DataFrame:
    """Join the multi-run utility + fidelity tables into one row per method.

    Sources changed 2026-08-15. Utility was results/synthcity_results.csv
    (performance.xgb.syn_id etc., now evaluation_LEGACY/eval_synthcity_LEGACY.py),
    whose TSTR was scored on an internal split of
    the generators' own training data; that file is deleted and the metric is
    replaced by tstr_xgboost_auc from evaluation/eval_utility.py, scored on the
    5,381-record held-out split. Fidelity moved from the single-draw
    sdmetrics_results.csv to the multi-run fidelity_summary.csv at the same
    time, so both halves of this table are means over the same runs.

    Dropped in the move: performance.feat_rank_distance.*, which had no in-house
    equivalent -- it was a synthcity-only metric, not part of the tradeoff axes.

    Both inputs are optional: until sdg/generate_runs.py and the two eval
    scripts have run, the corresponding columns come back as NaN rather than
    failing, so the privacy half of the benchmark can still be aggregated.
    """
    util = pd.read_csv(UTILITY_RESULTS) if UTILITY_RESULTS.exists() else None
    fid = pd.read_csv(FIDELITY_RESULTS) if FIDELITY_RESULTS.exists() else None
    if util is None:
        print(f"[warn] {UTILITY_RESULTS.name} not found -- run evaluation/eval_utility.py. "
              f"Utility columns will be blank.")
    if fid is None:
        print(f"[warn] {FIDELITY_RESULTS.name} not found -- run evaluation/eval_fidelity.py. "
              f"Fidelity columns will be blank.")

    rows = []
    for method in METHODS:
        row = {"method": method, "label": LABELS.get(method, method)}
        for src, cols, n_key in ((util, UTILITY_COLS, "n_utility_runs"),
                                 (fid, FIDELITY_COLS, "n_fidelity_runs")):
            if src is None:
                continue
            sub = src[src["method"] == method]
            if not len(sub):
                continue
            row[n_key] = int(sub["n_runs"].iloc[0]) if "n_runs" in sub else np.nan
            for col in cols:
                for suffix in ("_mean", "_std"):
                    key = f"{col}{suffix}"
                    row[key] = float(sub[key].iloc[0]) if key in sub else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "summary_utility.csv", index=False)
    print(f"Wrote {TABLES_DIR / 'summary_utility.csv'}")
    return out


def combine(priv: pd.DataFrame, util: pd.DataFrame) -> pd.DataFrame:
    if priv.empty:
        return priv
    keep_p = ["method", "label", "dp", "kind", "worst_case_auc", "mean_auc",
              "mean_advantage", "n_attacks_succeed",
              "worst_case_eff_eps_low95", "worst_case_eff_eps_high95", "formal_epsilon",
              "eff_eps_gap"]
    keep_u = ["method", "tstr_xgboost_auc_mean", "tstr_xgboost_auc_std",
              "trtr_xgboost_auc_mean", "retention_xgboost_mean",
              "tstr_logistic_regression_auc_mean",
              "KSComplement_mean", "KSComplement_std", "TVComplement_mean"]
    keep_u = [c for c in keep_u if c in util.columns]
    out = priv[keep_p].merge(util[keep_u], on="method", how="left")
    out.to_csv(TABLES_DIR / "summary_combined.csv", index=False)
    print(f"Wrote {TABLES_DIR / 'summary_combined.csv'}")
    return out


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    priv = summarize_privacy()
    util = summarize_utility()
    combined = combine(priv, util)
    if not combined.empty:
        cols = [c for c in ["label", "kind", "dp", "tstr_xgboost_auc_mean",
                            "tstr_xgboost_auc_std", "worst_case_auc",
                            "worst_case_eff_eps_low95", "formal_epsilon"] if c in combined.columns]
        print("\n=== benchmark summary (tradeoff axes) ===")
        print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
