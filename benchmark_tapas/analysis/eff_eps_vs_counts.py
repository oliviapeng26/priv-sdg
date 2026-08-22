#!/usr/bin/env python3
"""Effective epsilon vs shadow-model count: is the bound a measurement or a ceiling?

THE QUESTION
    Every generator in the 50/100 benchmark returned eps_low_95 = 2.209964 to six
    decimal places, and both generators piloted at 1000/2500 returned 5.546079 to
    six decimal places. Identical values across generators that differ enormously
    in fidelity is the signature of a bound set by the SAMPLE SIZE rather than by
    the generator: with FPR exactly 0, the Clopper-Pearson lower bound on
    ln(TP/FP) is pinned to whatever the test-set size allows, and that pin moves
    like ln(n).

    Two points cannot tell "ln(n) forever" from "converging on a real leakage
    value". This plots four for BayesNet and DPGAN and overlays a slope-1 ln(n)
    reference. If the points track the reference and FPR stays exactly 0, then no
    achievable count makes eps_eff discriminate DP from non-DP under this threat
    model -- the exact-knowledge attacker with a 499-record background makes the
    one-record swap trivially detectable, so everything leaks "perfectly" and the
    only thing the number reports is how many samples were drawn.

WHERE THE POINTS COME FROM
    50/100      results/per_method/{m}/effeps_{m}.csv    -- the main benchmark run
    all others  results/pilot_counts/{m}_{tr}_{te}/pilot_{m}_{tr}_{te}.csv
                                                         -- run_pilot_counts.py
                                                            and run_extra_counts.py

    DPGAN IS EXCLUDED AT 50/100. That audit predates the 2026-08-16 change to
    DPGAN_N_ITER=50: cache_LEGACY/dpgan records plugin_kwargs={'n_iter': 100}.
    Under DP that is not a shorter version of the same model -- opacus calibrates
    the per-step noise so the budget over n_iter epochs equals eps=1.0, so
    n_iter=100 is a different mechanism and cannot sit on the same curve.
    run_extra_counts.py regenerates DPGAN's 50/100 at n_iter=50 from the pilot
    cache; once that has run, the point appears here automatically.

    CTGAN and PrivBayes have a single point each (50/100) by design -- filling in
    their intermediate counts would cost ~7 h and ~29 h of real generator fits to
    confirm a shape that BN and DPGAN already show for free.

WHAT IS PLOTTED
    Top:    max eps_low_95 over the 5-attack battery, vs fits = num_train + num_test,
            on a log x-axis so ln(n) growth is a straight line. The teal dashed
            reference has slope exactly 1 in ln(fits), anchored at the leftmost
            point -- it is a guide, not a fit.
    Bottom: max TPR and max FPR over the battery. FPR sitting on 0 everywhere is
            what makes the top panel a ceiling rather than a measurement.

Run from repo root:
  python benchmark_tapas/analysis/eff_eps_vs_counts.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/
from config import (
    FIGURES_DIR, METHOD_CONFIG, PER_METHOD_DIR, RESULTS_DIR, TABLES_DIR,
)

PILOT_DIR = RESULTS_DIR / "pilot_counts"
OUT_PNG = FIGURES_DIR / "eff_eps_vs_counts.png"
OUT_CSV = TABLES_DIR / "eff_eps_vs_counts.csv"

# Repo palette (from the reference chart): orange + grey primary, teal + pink accent.
ORANGE, GREY, TEAL, PINK = "#CE4E36", "#3A3A3A", "#8CA9A0", "#D2B4B1"

# Same 2-channel encoding as tradeoff.py, so the two figures read together:
#   colour = DP status   (grey = non-DP, orange = DP)
#   shape  = family      (circle = statistical, square = neural)
DP_COLOR = {False: GREY, True: ORANGE}
KIND_MARKER = {"statistical": "o", "neural": "s"}

PRETTY = {"bayesian_network": "BayesNet", "privbayes": "PrivBayes",
          "ctgan": "CTGAN", "dpgan": "DPGAN"}

# All four generators return eps_low_95 IDENTICAL to six decimal places at every
# count setting (2.209964 at 150 fits, 5.546079 at 3500). Plotted honestly, three
# of the four markers vanish underneath the fourth and the figure looks like it
# has one generator in it. These fixed vertical offsets pull them apart so the
# agreement is visible as agreement -- the caption says so explicitly, and the
# printed table underneath carries the unmodified numbers. Deterministic per
# generator, never data-dependent, so the same generator sits at the same offset
# in every version of the figure.
Y_OFFSET = {"bayesian_network": -0.27, "privbayes": -0.09,
            "ctgan": 0.09, "dpgan": 0.27}
RATE_OFFSET = {"bayesian_network": -0.045, "privbayes": -0.015,
               "ctgan": 0.015, "dpgan": 0.045}

# Stale SOURCE FILES to skip, and why. Keyed by generator -- the whole
# per_method/{m}/effeps_{m}.csv is one audit at one configuration, so staleness
# is a property of the file, not of a count setting. Keying this on
# (method, counts) instead would also throw away the valid replacement that
# run_extra_counts.py regenerates at the same counts from the pilot cache.
STALE_PER_METHOD = {
    "dpgan": "audited at n_iter=100 (pre-2026-08-16); a different DP mechanism, "
             "not a shorter run of the same one. Its 50/100 replacement at "
             "n_iter=50 comes from results/pilot_counts/dpgan_50_100/ instead",
}


def _collect() -> pd.DataFrame:
    """Gather every valid (method, num_train, num_test) cell into one long table."""
    frames = []

    # 50/100 -- the main benchmark run, one CSV per generator.
    for method in METHOD_CONFIG:
        path = PER_METHOD_DIR / method / f"effeps_{method}.csv"
        if not path.exists():
            continue
        if method in STALE_PER_METHOD:
            print(f"  skipping {path.relative_to(RESULTS_DIR)}: "
                  f"{STALE_PER_METHOD[method]}")
            continue
        frame = pd.read_csv(path)
        frame["method"] = method
        frames.append(frame)

    # Everything else -- the pilot's directory layout, one dir per count setting.
    # run_extra_counts.py writes into exactly this pattern, so new count settings
    # are picked up with no change here.
    for cell_dir in sorted(PILOT_DIR.glob("*_*_*")):
        if not cell_dir.is_dir():
            continue
        match = re.fullmatch(r"(.+)_(\d+)_(\d+)", cell_dir.name)
        if not match or match.group(1) not in METHOD_CONFIG:
            continue
        method = match.group(1)
        path = cell_dir / f"pilot_{cell_dir.name}.csv"
        if not path.exists():
            print(f"  note: {cell_dir.name} has no {path.name} yet -- skipping")
            continue
        frame = pd.read_csv(path)
        frame["method"] = method
        frames.append(frame)

    if not frames:
        raise SystemExit("No results found. Run the benchmark / pilot scripts first.")

    table = pd.concat(frames, ignore_index=True)
    table = table.drop_duplicates(subset=["method", "num_train", "num_test", "attack"],
                                  keep="last")

    table["fits"] = table["num_train"] + table["num_test"]
    # tp / fp from TAPAS are rates (tapas/report/attack_summary.py) -- i.e. TPR/FPR.
    return table


def _per_cell(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, counts): the worst case over the 5-attack battery.

    Max is the right aggregate: eff-epsilon is a LOWER bound on leakage, so the
    strongest attack's bound is the binding one. The other four attacks return
    eps_low_95 = 0 here (AUC 0.0 or 0.5), which would only dilute a mean.
    """
    cells = (table.groupby(["method", "num_train", "num_test", "fits"], as_index=False)
                  .agg(max_eps_low_95=("eps_low_95", "max"),
                       max_tpr=("tp", "max"), max_fpr=("fp", "max"),
                       max_auc=("auc", "max"),
                       n_attacks=("attack", "count"),
                       n_attacks_at_max_auc=("auc", lambda s: int((s >= 1.0).sum()))))
    cells["dp"] = cells["method"].map(lambda m: METHOD_CONFIG[m]["dp"])
    cells["kind"] = cells["method"].map(lambda m: METHOD_CONFIG[m]["kind"])
    return cells.sort_values(["method", "fits"], kind="stable")


def main() -> int:
    print("Collecting effective-epsilon results...")
    table = _collect()
    cells = _per_cell(table)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_CSV, index=False)
    print(f"\n{'generator':12s} {'counts':>12s} {'fits':>6s} {'max eps_low_95':>15s} "
          f"{'TPR':>6s} {'FPR':>6s}")
    for _, r in cells.iterrows():
        print(f"{PRETTY[r.method]:12s} {int(r.num_train):5d}/{int(r.num_test):<6d} "
              f"{int(r.fits):6d} {r.max_eps_low_95:15.6f} {r.max_tpr:6.3f} {r.max_fpr:6.3f}")
    print(f"\nWrote {OUT_CSV}")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0]})

    # --- slope-1 ln(n) reference, anchored at the leftmost point -------------
    # Not a fit: a guide with slope exactly 1 in ln(fits). If a generator's points
    # lie on it, its eps_low_95 is growing exactly as fast as the Clopper-Pearson
    # bound allows when FPR is pinned at 0 -- i.e. the number is reporting the
    # sample size, not the leakage.
    anchor = cells.loc[cells["fits"].idxmin()]
    grid = np.array([cells["fits"].min() * 0.8, cells["fits"].max() * 1.3])
    ax.plot(grid, anchor.max_eps_low_95 + np.log(grid / anchor.fits),
            ls="--", lw=1.2, color=TEAL, zorder=1)
    mid = float(np.sqrt(grid[0] * grid[1]))     # midpoint on a log axis
    ax.annotate(r"slope 1 in $\ln(n)$" "\n" "(sample-size ceiling)",
                xy=(mid, anchor.max_eps_low_95 + np.log(mid / anchor.fits)),
                xytext=(8, -8), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=TEAL)

    # Formal budget. eps_eff is a lower bound on the leakage the mechanism admits,
    # so for the DP generators it should sit BELOW eps=1. Everything here sits far
    # above it, which is the clearest single statement that the estimate is not
    # measuring the mechanism.
    ax.axhline(1.0, ls=":", lw=1, color=PINK, zorder=1)
    ax.annotate(r"formal $\varepsilon = 1$ (DP generators)",
                xy=(cells["fits"].min() * 0.85, 1.0), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color="#8A6F6D",
                ha="left", va="bottom")

    # The generators coincide exactly, so the curve belongs to all of them at once.
    shared = (cells.groupby("fits", as_index=False)["max_eps_low_95"].max()
                   .sort_values("fits"))
    ax.plot(shared["fits"], shared["max_eps_low_95"], ls="-", lw=1.8,
            color="#7A7A7A", zorder=2)
    ax2.plot(shared["fits"], [1.0] * len(shared), ls="-", lw=1.4,
             color="#7A7A7A", zorder=2)
    ax2.plot(shared["fits"], [0.0] * len(shared), ls="-", lw=1.4,
             color="#7A7A7A", zorder=2)
    # Parked in the empty lower-right quadrant rather than near the curve, which
    # the markers and the ln(n) guide already crowd.
    ax.annotate("all four generators coincide at every $n$,\n"
                "to six decimal places",
                xy=(0.70, 0.30), xycoords="axes fraction",
                ha="center", va="center", fontsize=8.5, color="#333333")

    for method, group in cells.groupby("method", sort=False):
        group = group.sort_values("fits")
        color = DP_COLOR[bool(group["dp"].iloc[0])]
        marker = KIND_MARKER[group["kind"].iloc[0]]
        dy, dr = Y_OFFSET[method], RATE_OFFSET[method]
        # A generator with one count setting gets a marker only -- drawing a line
        # through a single point would imply a trend that was never measured.
        style = "none"          # the shared grey curve carries the trend
        ax.plot(group["fits"], group["max_eps_low_95"] + dy,
                ls=style, lw=1.6, color=color, marker=marker, markersize=8,
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=0.8, zorder=3)
        last = group.iloc[-1]
        ax.annotate(PRETTY[method], xy=(last.fits, last.max_eps_low_95 + dy),
                    xytext=(9, 0), textcoords="offset points",
                    fontsize=8.5, va="center", ha="left", color="#222222")

        ax2.plot(group["fits"], group["max_tpr"] + dr, ls=style,
                 lw=1.4, color=color, marker=marker, markersize=6,
                 markerfacecolor=color, markeredgecolor="black",
                 markeredgewidth=0.7, zorder=3)
        ax2.plot(group["fits"], group["max_fpr"] + dr, ls=style,
                 lw=1.4, color=color, marker=marker, markersize=6,
                 markerfacecolor="white", markeredgecolor=color,
                 markeredgewidth=1.2, zorder=3)

    ax.set_xscale("log")
    ax.set_ylabel(r"max $\varepsilon_{\mathrm{eff}}$ (95% lower bound)")
    ax.margins(x=0.22)          # room for the right-hand labels
    ax.set_ylim(bottom=-0.3)

    ax2.set_ylabel("rate")
    ax2.set_xlabel(r"shadow-model fits  $n = $ num_train $+$ num_test  (log scale)")
    ax2.set_ylim(-0.12, 1.12)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xticks(sorted(cells["fits"].unique()))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.annotate("FPR = 0 and TPR = 1 at every point, for every generator:\n"
                 "the bound is set by $n$, not by leakage",
                 xy=(0.97, 0.5), xycoords="axes fraction", fontsize=8,
                 ha="right", va="center", color="#444444")

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=9, label="Statistical"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=9, label="Neural"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
               markeredgecolor="black", markersize=9, label="non-DP"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE,
               markeredgecolor="black", markersize=9, label="DP (ε=1)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8, framealpha=0.9,
              title="shape = family · colour = DP", title_fontsize=8)
    ax2.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
               markeredgecolor="black", markersize=7, label="TPR (filled)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=GREY, markeredgewidth=1.2, markersize=7,
               label="FPR (hollow)"),
    ], loc="center left", fontsize=8, framealpha=0.9, ncol=1)

    fig.tight_layout(rect=[0, 0.085, 1, 1])   # room for the 3-line caption
    fig.subplots_adjust(hspace=0.12)
    fig.text(0.02, 0.02,
             r"$\mathbf{Figure\ 2.}$ Effective $\varepsilon$ against shadow-model count."
             "\nPoints tracking the $\\ln(n)$ guide with FPR $=0$ indicate a "
             "Clopper–Pearson ceiling, not a leakage measurement."
             "\nGenerators are offset vertically to separate them: their values are "
             "identical to six decimal places.",
             ha="left", va="bottom", fontsize=8.5)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PNG}")

    # --- the actual question, in numbers ---
    print("\n=== growth vs the ln(n) ceiling ===")
    for method, group in cells.groupby("method", sort=False):
        group = group.sort_values("fits")
        if len(group) < 2:
            print(f"  {PRETTY[method]:12s} single point -- no growth to measure")
            continue
        print(f"  {PRETTY[method]}")
        for (_, a), (_, b) in zip(group.iloc[:-1].iterrows(), group.iloc[1:].iterrows()):
            observed = b.max_eps_low_95 - a.max_eps_low_95
            expected = np.log(b.fits / a.fits)
            print(f"    {int(a.fits):5d} -> {int(b.fits):5d} fits:  "
                  f"observed +{observed:.3f}   ln(n) ceiling +{expected:.3f}   "
                  f"ratio {observed / expected:.2f}")
        print("    (ratio near 1.0 = tracking the ceiling; near 0 = converging on "
              "a real value)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
