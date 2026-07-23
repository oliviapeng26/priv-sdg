#!/usr/bin/env python3
"""Aggregate the benchmark into report-ready summary tables.

Joins three sources (privacy from this experiment; utility + fidelity already
computed elsewhere in the repo) into:
  results/summary_privacy.csv   -- per method: worst-case & mean MIA AUC across
                                   the 5 attacks, worst-case eff-epsilon (max
                                   eps_low_95, the tightest defensible lower
                                   bound on leakage), which attack was worst.
  results/summary_utility.csv   -- per method: TSTR performance.* (from
                                   results/synthcity_results.csv) + SDMetrics
                                   fidelity (from results/sdmetrics_results.csv).
  results/summary_combined.csv  -- one row per method with the tradeoff axes
                                   (xgb_syn_id + worst_case_auc) side by side.

Run from repo root, after the four run_*.py have populated
benchmark_privacy_per_attack.csv:
  python benchmark_tapas/aggregate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/ (script now in a subfolder)
from config import (
    TABLES_DIR, METHOD_CONFIG, METHODS, DP_METHODS,
    SYNTHCITY_RESULTS, SDMETRICS_RESULTS,
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


def summarize_utility() -> pd.DataFrame:
    rows = []
    perf = None
    if SYNTHCITY_RESULTS.exists():
        # long format: first col = metric name (index), plus 'method','mean'
        perf = pd.read_csv(SYNTHCITY_RESULTS, index_col=0)
    fid = pd.read_csv(SDMETRICS_RESULTS) if SDMETRICS_RESULTS.exists() else None

    for method in METHODS:
        row = {"method": method, "label": LABELS.get(method, method)}
        if perf is not None:
            sub = perf[perf["method"] == method]
            for metric in ["performance.xgb.gt", "performance.xgb.syn_id",
                           "performance.xgb.syn_ood", "performance.linear_model.syn_id",
                           "performance.feat_rank_distance.corr"]:
                key = metric.replace("performance.", "").replace(".", "_")
                row[key] = float(sub.loc[metric, "mean"]) if metric in sub.index else np.nan
        if fid is not None:
            fsub = fid[fid["method"] == method]
            if len(fsub):
                for col in ["KSComplement", "TVComplement", "CorrelationSimilarity",
                            "ContingencySimilarity"]:
                    if col in fsub:
                        row[col] = float(fsub[col].iloc[0])
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
    keep_u = ["method", "xgb_syn_id", "xgb_gt", "linear_model_syn_id",
              "feat_rank_distance_corr", "KSComplement", "TVComplement"]
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
        cols = [c for c in ["label", "kind", "dp", "xgb_syn_id", "worst_case_auc",
                            "worst_case_eff_eps_low95", "formal_epsilon"] if c in combined.columns]
        print("\n=== benchmark summary (tradeoff axes) ===")
        print(combined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
