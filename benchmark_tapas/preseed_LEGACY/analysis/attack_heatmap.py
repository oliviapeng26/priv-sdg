#!/usr/bin/env python3
"""Complementary figure: MIA AUC heatmap (attack x method).

Visualises the two headline findings at a glance:
  - Groundhog + ShadowModelling separate EVERY generator (AUC=1.0) -> the
    worst-case saturation, DP or not.
  - ProbabilityEstimation succeeds only on the NEURAL generators (CTGAN, DPGAN)
    -> the "broader attack surface" of neural methods that mean AUC captures.

Diverging colormap centred at 0.5 (random): red = leakage (AUC>0.5), blue =
below random (failed attack). Reads results/benchmark_privacy_per_attack.csv,
writes results/attack_auc_heatmap.png.

Run from repo root, after the four run_*.py:
  python benchmark_tapas/attack_heatmap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/ (script now in a subfolder)
from config import TABLES_DIR, FIGURES_DIR

PER_ATTACK = TABLES_DIR / "benchmark_privacy_per_attack.csv"
OUT_PNG = FIGURES_DIR / "attack_auc_heatmap.png"

# Repo palette (from the reference chart): orange + grey primary.
# Diverging map for AUC 0..1 centred at 0.5: grey (0.0 = failed) -> white
# (0.5 = random) -> orange (1.0 = perfect separation / leaked).
ORANGE, GREY = "#CE4E36", "#3A3A3A"
CMAP = LinearSegmentedColormap.from_list("repo_div", [GREY, "#FFFFFF", ORANGE])

ATTACK_ORDER = ["Groundhog", "ShadowModelling", "LocalNeighbourhood",
                "ProbabilityEstimation", "ClosestDistance"]
METHOD_ORDER = ["bayesian_network", "privbayes", "ctgan", "dpgan"]
METHOD_LABELS = ["BN", "PrivBayes", "CTGAN", "DPGAN"]


def main():
    if not PER_ATTACK.exists():
        print(f"[error] {PER_ATTACK} not found -- run the run_*.py scripts first.")
        return
    df = pd.read_csv(PER_ATTACK)
    df["a"] = df["attack"].str.split("(").str[0]
    mat = (df.pivot_table(index="a", columns="method", values="auc")
             .reindex(index=ATTACK_ORDER, columns=METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(mat.values, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_LABELS, fontsize=9)
    ax.set_yticks(range(len(ATTACK_ORDER)))
    ax.set_yticklabels(ATTACK_ORDER, fontsize=9)

    # Annotate each cell with its AUC; white text on saturated cells.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if (v > 0.8 or v < 0.2) else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MIA AUC", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])   # leave room at the bottom for the caption
    fig.text(0.02, 0.02, r"$\mathbf{Figure\ 2.}$ MIA AUC by Attack × Generator",
             ha="left", va="bottom", fontsize=9)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
