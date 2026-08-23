#!/usr/bin/env python3
"""Separation and decisiveness plots from the raw pre-threshold attack scores.

INPUT   benchmark_tapas/results/scores/raw_scores_{method}.csv, written by
        common.run_method (7 columns; see common._score_rows).
OUTPUT  benchmark_tapas/results/figures/score_separation.png
        benchmark_tapas/results/figures/score_decisiveness.png

SEPARATION asks: does the attack tell members from non-members? Member and
non-member score densities are overlaid per generator x attack, with the full-ROC
AUC annotated. AUC is the summary because it reads the WHOLE score distribution
rather than the single operating point the binary decision collapses to.

DECISIVENESS asks: how confident is the attacker, ignoring whether it is right?
Score spread per generator, labels discarded. A tight pile means the attack cannot
tell the datasets apart at all; a wide spread means it is making strong calls
(correct or not).

READING THE AUC -- the null band matters more than the point estimate.
    With 50 members and 50 non-members, an attack with NO signal still scatters
    around 0.5 with s.d. sqrt((n1+n2+1)/(12*n1*n2)) = 0.058, so the 95% null band
    is 0.5 +- 0.114, i.e. [0.386, 0.614]. An AUC inside that band is indistinguishable
    from guessing. Panels are annotated "ns" when they fall inside it. This band is
    why the counts, not the generators, are currently the binding constraint.

SCALES ARE PER-ATTACK, DELIBERATELY.
    RF probabilities are [0,1]; ClosestDistance is a negated L2 distance;
    ProbabilityEstimation is a KDE log-density. Sharing an x-axis across attacks
    would be meaningless, so each ROW gets its own scale and columns within a row
    share one (that is the comparison that matters: generators, same attack).

DEGENERATE ATTACKS.
    LocalNeighbourhood scores the fraction of synthetic records inside a radius.
    When no record ever lands inside, every score is 0.0 and a KDE is undefined
    (zero variance). Those panels fall back to a bar at the observed value and are
    labelled "degenerate (k distinct values)" -- that is a finding about the attack,
    not a plotting failure, so it is shown rather than hidden.

COLOUR (repo palette, inlined per convention -- no shared palette module).
    Member = terracotta, non-member = charcoal. Validated as a categorical pair:
    CVD dE 17.8 (protan), normal-vision dE 29.5, contrast >= 3:1 on white.
    The sage/rose accents are deliberately NOT used to distinguish the four
    generators -- they fail CVD separation against each other (dE 5.9 protan,
    10.8 normal). Generators are separated by POSITION instead.

Run from repo root:
  python benchmark_tapas/scripts/plot_scores.py
"""

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCORES_DIR = REPO_ROOT / "benchmark_tapas" / "results" / "scores"
FIGURES_DIR = REPO_ROOT / "benchmark_tapas" / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Repo palette (see the module docstring; inlined on purpose).
ORANGE, CHARCOAL = "#CE4E36", "#3A3A3A"
INK, MUTED, GRID = "#3A3A3A", "#6B6B6B", "#DDDDDD"

GENERATORS = ["bayesian_network", "privbayes", "ctgan", "dpgan"]
GEN_LABELS = {"bayesian_network": "BayesNet", "privbayes": "PrivBayes",
              "ctgan": "CTGAN", "dpgan": "DPGAN"}
DP = {"bayesian_network": False, "privbayes": True, "ctgan": False, "dpgan": True}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged so an all-identical
    score column returns exactly 0.5 rather than an arbitrary value."""
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def null_halfwidth(n1: int, n2: int) -> float:
    """Half-width of the 95% band around AUC=0.5 for an attack with no signal."""
    return 1.96 * np.sqrt((n1 + n2 + 1) / (12 * n1 * n2))


def density(values: np.ndarray, grid: np.ndarray):
    """KDE, or None when the sample has no spread for a KDE to be defined."""
    if len(np.unique(values)) < 3 or np.std(values) == 0:
        return None
    try:
        return gaussian_kde(values)(grid)
    except np.linalg.LinAlgError:
        return None


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(SCORES_DIR / "raw_scores_*.csv")))
    if not files:
        raise SystemExit(f"No raw_scores_*.csv in {SCORES_DIR}. Run the benchmark first.")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["attack_short"] = d["attack"].str.split("(").str[0]
    missing = set(GENERATORS) - set(d.generator)
    if missing:
        print(f"WARNING: no scores for {sorted(missing)} -- those columns will be blank")
    return d


def plot_separation(d: pd.DataFrame) -> Path:
    attacks = sorted(d.attack_short.unique())
    gens = [g for g in GENERATORS if g in set(d.generator)]
    fig, axes = plt.subplots(len(attacks), len(gens),
                             figsize=(3.4 * len(gens), 2.25 * len(attacks)),
                             squeeze=False)

    for r, atk in enumerate(attacks):
        row = d[d.attack_short == atk]
        lo, hi = row.raw_score.min(), row.raw_score.max()
        pad = (hi - lo) * 0.12 if hi > lo else 0.5
        grid = np.linspace(lo - pad, hi + pad, 400)

        for c, gen in enumerate(gens):
            ax = axes[r][c]
            g = row[row.generator == gen]
            mem = g[g.ground_truth == 1].raw_score.values
            non = g[g.ground_truth == 0].raw_score.values
            a = auc(g.ground_truth.values, g.raw_score.values)
            ns = abs(a - 0.5) <= null_halfwidth(len(mem), len(non))

            dm, dn = density(mem, grid), density(non, grid)
            if dm is None or dn is None:
                # Zero-variance: a KDE is undefined, so show where the mass sits.
                # Bars are drawn on the ROW's scale (set below) so the panel stays
                # comparable with its neighbours instead of auto-zooming to itself.
                w = (grid[-1] - grid[0]) / 60
                for vals, col in ((non, CHARCOAL), (mem, ORANGE)):
                    u, cts = np.unique(vals, return_counts=True)
                    ax.bar(u, cts / cts.sum(), width=w, color=col, alpha=0.55,
                           edgecolor="white", lw=0.5)
                ax.set_ylim(0, 1.35)
                ax.text(0.5, 0.55, f"degenerate\n{g.raw_score.nunique()} distinct value(s)",
                        transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)
            else:
                ax.fill_between(grid, dn, color=CHARCOAL, alpha=0.30, lw=0)
                ax.fill_between(grid, dm, color=ORANGE, alpha=0.30, lw=0)
                ax.plot(grid, dn, color=CHARCOAL, lw=2)
                ax.plot(grid, dm, color=ORANGE, lw=2)

            thr = g.threshold.iloc[0]
            if np.isfinite(thr) and grid[0] <= thr <= grid[-1]:
                ax.axvline(thr, color=MUTED, ls=(0, (4, 3)), lw=1.2)

            ax.set_xlim(grid[0], grid[-1])   # one scale per row: the comparison is across generators
            ax.set_yticks([])
            tidy(ax)
            # White bbox so the threshold rule can't strike through the label.
            ax.text(0.03, 0.93, f"AUC {a:.3f}" + ("  ns" if ns else ""),
                    transform=ax.transAxes, va="top", fontsize=9,
                    color=MUTED if ns else INK,
                    fontweight="normal" if ns else "bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
            if r == 0:
                ax.set_title(f"{GEN_LABELS[gen]}" + ("  (DP)" if DP[gen] else ""),
                             fontsize=10, color=INK, pad=8)
            if c == 0:
                ax.set_ylabel(atk, fontsize=9, color=INK, labelpad=8)

    handles = [plt.Line2D([], [], color=ORANGE, lw=3, label="member (D+)"),
               plt.Line2D([], [], color=CHARCOAL, lw=3, label="non-member (D-)"),
               plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)), lw=1.2, label="decision threshold")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("Attack score separation: member vs non-member  "
                 "(n=50/50 per panel; \"ns\" = inside the 95% null band for AUC)",
                 fontsize=11, color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0.025, 1, 0.985])
    out = FIGURES_DIR / "score_separation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_decisiveness(d: pd.DataFrame) -> Path:
    """Score spread per generator, labels ignored. Generators are separated by
    POSITION (one row each) rather than hue -- the palette's accent colours fail
    CVD separation against one another."""
    attacks = sorted(d.attack_short.unique())
    gens = [g for g in GENERATORS if g in set(d.generator)]
    fig, axes = plt.subplots(len(attacks), 1, figsize=(9, 2.1 * len(attacks)), squeeze=False)

    for r, atk in enumerate(attacks):
        ax = axes[r][0]
        row = d[d.attack_short == atk]
        lo, hi = row.raw_score.min(), row.raw_score.max()
        pad = (hi - lo) * 0.12 if hi > lo else 0.5
        grid = np.linspace(lo - pad, hi + pad, 400)

        for i, gen in enumerate(gens):
            vals = row[row.generator == gen].raw_score.values
            base = len(gens) - 1 - i           # top row = first generator
            dens = density(vals, grid)
            if dens is not None:
                dens = dens / dens.max() * 0.78
                ax.fill_between(grid, base, base + dens, color=ORANGE, alpha=0.32, lw=0)
                ax.plot(grid, base + dens, color=ORANGE, lw=1.8)
            # Rug: the actual observations, so a degenerate panel is visibly degenerate.
            ax.plot(vals, np.full_like(vals, base - 0.06), "|", color=CHARCOAL,
                    ms=6, alpha=0.5, mew=1)
            iqr = np.percentile(vals, [25, 75])
            ax.plot(iqr, [base - 0.16] * 2, color=CHARCOAL, lw=2.5, solid_capstyle="butt")
            ax.text(1.005, base + 0.18, f"{vals.std():.3g}", transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color=MUTED, va="center")

        ax.set_xlim(grid[0], grid[-1])
        ax.set_yticks(range(len(gens)))
        ax.set_yticklabels([GEN_LABELS[g] + ("  (DP)" if DP[g] else "")
                            for g in reversed(gens)], fontsize=9, color=INK)
        ax.set_ylim(-0.45, len(gens) - 0.05)
        ax.set_title(atk, fontsize=10, color=INK, loc="left", pad=6)
        tidy(ax)
        ax.grid(axis="y", visible=False)

    handles = [plt.Line2D([], [], color=ORANGE, lw=3, label="score density (area-normalised)"),
               plt.Line2D([], [], color=CHARCOAL, lw=2.5, label="interquartile range"),
               plt.Line2D([], [], color=CHARCOAL, marker="|", ls="none", label="individual datasets")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.008))
    fig.suptitle("Attacker decisiveness: score spread per generator, ground truth ignored\n"
                 "(right-hand number is the standard deviation)",
                 fontsize=11, color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = FIGURES_DIR / "score_decisiveness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    d = load()
    print(f"loaded {len(d)} rows  "
          f"({d.generator.nunique()} generators x {d.attack_short.nunique()} attacks "
          f"x {len(d) // max(d.generator.nunique() * d.attack_short.nunique(), 1)} datasets)")

    hw = null_halfwidth(50, 50)
    tbl = (d.groupby(["attack_short", "generator"])
             .apply(lambda g: auc(g.ground_truth.values, g.raw_score.values),
                    include_groups=False)
             .unstack())
    tbl = tbl[[g for g in GENERATORS if g in tbl.columns]]
    print(f"\n=== AUC (full ROC) -- 95% null band is [{0.5 - hw:.3f}, {0.5 + hw:.3f}] ===")
    print(tbl.round(3).to_string())
    outside = (tbl - 0.5).abs() > hw
    print(f"\npanels outside the null band: {int(outside.values.sum())} of {outside.size}")
    if outside.values.any():
        for atk, gen in zip(*np.where(outside.values)):
            print(f"  {tbl.index[atk]} x {tbl.columns[gen]}: AUC {tbl.values[atk, gen]:.3f}")

    for p in (plot_separation(d), plot_decisiveness(d)):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
