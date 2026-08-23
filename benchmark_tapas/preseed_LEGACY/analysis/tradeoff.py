#!/usr/bin/env python3
"""Privacy-utility tradeoff scatter -- the key benchmark deliverable figure.

One point per generator (4 total). Axes:
  X = performance.xgb.syn_id  (TSTR in-distribution XGBoost; higher = more useful
      as a drop-in replacement for real data). Dashed vertical = real-data
      baseline performance.xgb.gt.
  Y = MEAN MEMBERSHIP ADVANTAGE (tp-fp) across the 5 attacks (higher = broader
      attack surface / more leakage). We use the *mean*, not the worst case,
      because worst-case AUC saturates at 1.0 for all four generators
      (Groundhog/ShadowModelling perfectly separate every generator regardless of
      DP -- see summary_privacy.csv and the companion table). Advantage, not AUC:
      a random or broken attack (AUC 0.5, or AUC 0 with tp=fp=0) means the method
      *resisted* that attack, so it must contribute 0 leakage -- but mean AUC would
      credit it 0.5 (or 0.0), penalising a successful defence. Advantage anchors
      "no leakage" at 0 (tp=fp) and "fully separated" at 1 (tp=1,fp=0), so it
      counts only genuine attack success. Here it equals (# of 5 attacks that
      succeed)/5 since advantage is binary {0,1}. Dashed horizontal at 0 = attacks
      no better than random.

Encoding: marker shape = generator family (circle=statistical, square=neural);
fill = DP status (filled=DP, hollow=non-DP).

Reads results/summary_combined.csv (produced by aggregate.py). Writes
results/privacy_utility_tradeoff.png.

Run from repo root, after aggregate.py:
  python benchmark_tapas/tradeoff.py
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/ (script now in a subfolder)
from config import TABLES_DIR, FIGURES_DIR

COMBINED = TABLES_DIR / "summary_combined.csv"
OUT_PNG = FIGURES_DIR / "privacy_utility_tradeoff.png"

# Repo palette (from the reference chart): orange + grey primary, teal + pink accent.
ORANGE, GREY, TEAL, PINK = "#CE4E36", "#3A3A3A", "#8CA9A0", "#D2B4B1"

# Minimal 2-channel encoding that mirrors the 2x2 grid (point labels already
# identify each generator, so per-generator colour was redundant):
#   marker SHAPE  = generator family  (circle = statistical, square = neural)
#   marker COLOUR = DP status         (grey = non-DP, orange = DP)
DP_COLOR = {False: GREY, True: ORANGE}
SHAPE = {"statistical": "o", "neural": "s"}

# Per-method label placement (dx, dy in points; ha; va). CTGAN + PrivBayes sit
# just left of the right-edge utility-ceiling line, so their labels go LEFT to
# clear it; BN + PrivBayes share height (0.40), so PrivBayes drops BELOW its
# marker to avoid BN's left-going label. DPGAN is far left -> label right.
LABEL_PLACE = {
    "dpgan":            (12, 9, "left", "bottom"),
    "ctgan":            (-12, 9, "right", "bottom"),
    "bayesian_network": (-12, 9, "right", "bottom"),
    "privbayes":        (-22, -12, "center", "top"),
}


def main():
    if not COMBINED.exists():
        print(f"[error] {COMBINED} not found -- run aggregate.py first.")
        return
    df = pd.read_csv(COMBINED)
    if df.empty or "tstr_xgboost_auc_mean" not in df or "mean_advantage" not in df:
        print("[error] combined summary missing required columns.")
        print("        privacy axis -> benchmark_tapas/scripts/run_*.py")
        print("        utility axis -> evaluation/eval_utility.py, then aggregate.py")
        return
    if df["tstr_xgboost_auc_mean"].isna().all():
        print("[error] utility column is all-NaN -- run evaluation/eval_utility.py, then aggregate.py.")
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    # Reference line: advantage 0 = attacks no better than random (private).
    ax.axhline(0.0, ls="--", lw=1, color=TEAL, zorder=1)
    ax.text(0.01, 0.0, " privacy leakage floor (AUC = 0.0)", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color=TEAL)
    if "trtr_xgboost_auc_mean" in df and df["trtr_xgboost_auc_mean"].notna().any():
        gt = float(df["trtr_xgboost_auc_mean"].dropna().iloc[0])
        ax.axvline(gt, ls=":", lw=1, color=TEAL, zorder=1)
        ax.text(gt, 0.07, f" utility ceiling from real data (TRTR = {gt:.2f})", rotation=90,
                transform=ax.get_xaxis_transform(), va="bottom", ha="right",
                fontsize=8, color=TEAL)

    for _, r in df.iterrows():
        color = DP_COLOR[bool(r["dp"])]
        n_succeed = int(r["n_attacks_succeed"]) if "n_attacks_succeed" in r else None
        # x error bar = across-seed std of TSTR (seeds.RUN_SEEDS). It is load-bearing:
        # at eps=1.0 PrivBayes spans ~0.11 AUC across seeds, wide enough to overlap
        # its non-DP sibling, so a bare point would overstate the separation.
        xerr = r.get("tstr_xgboost_auc_std", float("nan"))
        if pd.notna(xerr):
            ax.errorbar(r["tstr_xgboost_auc_mean"], r["mean_advantage"], xerr=xerr,
                        fmt="none", ecolor=color, elinewidth=1.2, capsize=3, zorder=2)
        ax.scatter(r["tstr_xgboost_auc_mean"], r["mean_advantage"],
                   marker=SHAPE.get(r["kind"], "o"), s=200,
                   facecolors=color, edgecolors="black", linewidths=0.8, zorder=3)
        tag = f"\n({n_succeed}/5 attacks)" if n_succeed is not None else ""
        # drop the "(DP)"/"(no-DP)" suffix here -- colour already encodes DP status
        name = str(r["label"]).split(" (")[0]
        dx, dy, ha, va = LABEL_PLACE.get(r["method"], (12, 9, "left", "bottom"))
        ax.annotate(f"{name}{tag}",
                    (r["tstr_xgboost_auc_mean"], r["mean_advantage"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                    ha=ha, va=va, color="#222222")

    ax.set_xlabel("TSTR XGBoost AUC, mean $\\pm$ s.d. over runs (Utility)")
    ax.set_ylabel("Mean membership advantage (Privacy leakage)")
    ax.set_ylim(-0.05, 0.75)
    ax.margins(x=0.18)

    # 2 channels: grey markers show the shape=family mapping; colour swatches show
    # the colour=DP mapping. (Grey on the shape entries so colour isn't implied there.)
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=10, label="Statistical"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=10, label="Neural"),
        Patch(facecolor=DP_COLOR[False], edgecolor="black", label="non-DP"),
        Patch(facecolor=DP_COLOR[True], edgecolor="black", label="DP (ε=1)"),
    ]
    # Anchor the legend's bottom-left ~12% up the axes so it clears the 0.0
    # baseline (and its label) sitting at the very bottom.
    ax.legend(handles=legend_elems, loc="lower left", bbox_to_anchor=(0.0, 0.12),
              fontsize=8, framealpha=0.9,
              title="shape = family · colour = DP", title_fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 1])   # leave room at the bottom for the caption
    fig.text(0.02, 0.02, r"$\mathbf{Figure\ 1.}$ Privacy–utility Tradeoff",
             ha="left", va="bottom", fontsize=9)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
