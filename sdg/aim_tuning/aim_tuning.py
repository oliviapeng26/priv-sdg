#!/usr/bin/env python3
"""AIM bin-count sweep: does discretisation granularity decide whether AIM ever
buys a numerical x numerical marginal at eps = 1.0?

KNOWN ISSUE (2026-08-15) -- TSTR COLUMNS HERE ARE LEAKY, NOT YET FIXED
    This script scores utility with synthcity's
    `Metrics.evaluate(metrics={"performance": [...]})`, which splits the real
    loader it is handed -- the training data AIM was just fitted on -- and scores
    on that internal holdout. Records the generator saw are being used as the
    test set, so a memorising generator is rewarded. This is the same leak that
    was fixed elsewhere in the repo (see evaluation/evaluation_LEGACY/eval_synthcity_LEGACY.py).

    AFFECTED: the `xgb_syn_id`, `xgb_gt` and `linear_syn_id` columns of
    aim_tuning_results.csv / aim_tuning_repeats.csv, and the TSTR panel of
    aim_tuning_utility.png. `results/utility_comparison_b32.png` was built from
    them and has already been deleted.

    NOT AFFECTED: every fidelity column (KSComplement, TVComplement,
    CorrelationSimilarity, ContingencySimilarity) and the marginal-selection
    counts, which is what this sweep's actual conclusion rests on -- so the
    bins=32 choice does not obviously need revisiting.

    TO FIX: swap the Metrics.evaluate call for `compute_tstr` / `compute_trtr`
    from evaluation/eval_utility.py, scored on data/adult_test.csv, and re-run
    the sweep. Deliberately deferred -- flagged so the numbers are not quoted in
    the write-up as they stand.

WHY THIS EXISTS
---------------
AIM at 20 bins produced a split profile on Adult: best-of-five on categorical
fidelity (TVComplement 0.983, ContingencySimilarity 0.912) but worst-of-five on
CorrelationSimilarity (0.796, below even DPGAN) and 4th on TSTR (xgb 0.724).

The cause is visible in the run log: across all 208 rounds AIM selected 31
marginals and NOT ONE of them paired two continuous columns. It never measured
age x hours_per_week, age x education_num, capital_gain x capital_loss, etc., so
the model has no information about numerical relationships -- exactly what a
correlation metric and a tree model punish.

Binning was ruled out as the cause: round-tripping the real data through
encode->decode with AIM removed scores CorrelationSimilarity 0.9984, i.e. the
discretisation is near-lossless. AIM is also already AT the binning ceiling for
KSComplement (0.9095 vs 0.9133).

Why AIM skips them: its selection score subtracts a bias proportional to the
marginal's cell count. At 20 bins, age x hours = 400 cells while
(relationship, sex) = 12. Under a tight budget the big candidates are
systematically outbid. Bin count is therefore the lever that decides whether a
numerical pair is ever affordable.

LITERATURE
----------
Ganev, Annamalai, Mahiou & De Cristofaro, "The Importance of Being Discrete:
Measuring the Impact of Discretization in End-to-End Differentially Private
Synthetic Data", ACM CCS 2025 (arXiv:2504.06923):
  - p2, Finding 2: utility follows "an inverted u-shaped trend as the number of
    bins increases: it initially improves but then degrades when too many bins
    introduce too much noise" (Figure 1, at eps = 1).
  - p2, Finding 3: optimising discretizer + bin count beats "the default
    discretization (uniform with 20 bins)" by 9.28%-43.54%; AIM gains +16.22%
    (Figure 2). Our 20-bin uniform choice IS that default.
  - p8: "for models with relatively low variability (i.e., PrivBayes, MST, and
    AIM) ... At lower epsilon values, the best results are achieved with fewer
    bins." We are at eps = 1.
  - p11: "AIM takes a similar approach to MST -- it uses uniform with 32 bins,
    assuming the domain is provided." Hence 32 is in the grid.
  - p11: their sweep spans "from 5 to 250".
They include Adult among three real datasets (p5) but publish no Adult-only,
AIM-only bin curve -- Figures 1/2 average over Adult+Gas+Wine and the per-model
Figure 8 is Gas. This sweep fills that gap for our exact setup.

DESIGN
------
Bin count is the ONLY variable. Everything else is pinned to sdg/aim.py:
eps=1.0, delta=1e-9, degree=2, max_cells=10000, max_model_size=80, rounds=208,
public-codebook bounds (so preprocessor_eps=0 and the full eps reaches AIM),
uniform-within-bin decoding, 21,523 generated rows.

One draw per setting -- consistent with the rest of the benchmark, where every
generator is a single run. The DP noise is unseedable (opendp), so treat small
metric gaps as noise; the numerical-pair count is the robust signal.

250 bins is excluded: a numerical pair would be 62,500 cells and AIM's
max_cells=10000 filter drops it from the candidate workload before selection
runs, so the outcome would measure our filter rather than AIM's behaviour.

WRITES NOTHING OUTSIDE THIS FOLDER. synthetic_data/aim_synthetic.csv and
results/*.csv are untouched; only the final chosen bin count, applied later in
sdg/aim.py, changes the real artefacts.

Run from anywhere, in the priv-sdg env (~40 min):
    python sdg/aim_tuning/aim_tuning.py
    python sdg/aim_tuning/aim_tuning.py 5 10 20   # subset of the grid
"""

import io
import re
import sys
import time
import warnings
import contextlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Sweep grid ----------------------------------------------------------
# 5/10/25/50/100 = the paper's grid; 20 = current baseline (= their "default");
# 32 = AIM's own default (p11). 250 excluded (see docstring).
BIN_GRID = [5, 10, 20, 25, 32, 50, 100]

# -- Pinned to sdg/aim.py (do not vary) ----------------------------------
EPSILON = 1.0
DELTA = 1e-9
SEED = 42
DEGREE = 2
MAX_CELLS = 10_000
MAX_MODEL_SIZE = 80
ROUNDS = None            # -> 16 * 13 = 208

TUNING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TUNING_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
SYN_DIR = TUNING_DIR / "synthetic"        # gitignored
WORKSPACE = TUNING_DIR / "workspace"      # gitignored (synthcity metric cache)
RESULTS_CSV = TUNING_DIR / "aim_tuning_results.csv"
LOG_TXT = TUNING_DIR / "aim_tuning_log.txt"
CURVE_PNG = TUNING_DIR / "aim_tuning_curve.png"

CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]
TARGET_COL = "income"

# Repo palette: orange + grey primary, teal + pink accent.
ORANGE, GREY, TEAL, PINK = "#CE4E36", "#3A3A3A", "#8CA9A0", "#D2B4B1"


def build_edges(b: int) -> dict:
    """Bin edges for every continuous column at sweep setting `b`.

    Bounds are fixed public-codebook constants, never estimated from the data,
    so no privacy budget is spent on preprocessing at any point in the sweep.

    Two columns deviate from plain equal-width, deliberately and identically at
    every b (so the sweep stays single-variable):

      capital_gain / capital_loss are 90.7% / 94.8% EXACT ZEROS. They get a
      dedicated zero bin [0, 1) plus (b-1) log-spaced bins over the positive
      range. Plain equal-width at b=5 would put a [0, 20000) bin around those
      zeros and decode them uniformly across it, destroying both columns -- we
      would be measuring that, not AIM. Log spacing handles the right skew.

      education_num holds only 16 distinct integers, so it caps at 16 bins;
      beyond that the extra bins are empty and just add noise-carrying cells.
    """
    return {
        "age": np.linspace(17, 91, b + 1),
        "hours_per_week": np.linspace(1, 100, b + 1),
        "education_num": np.linspace(1, 17, min(b, 16) + 1),
        "capital_gain": np.concatenate([[0.0], np.geomspace(1, 100_000, b)]),
        "capital_loss": np.concatenate([[0.0], np.geomspace(1, 5_000, b)]),
    }


def encode(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Continuous column -> integer bin index in [0, len(edges) - 2]."""
    clipped = values.clip(edges[0], np.nextafter(edges[-1], edges[0]))
    return pd.Series(np.searchsorted(edges, clipped, side="right") - 1,
                     index=values.index, dtype=int)


def decode(codes: pd.Series, edges: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """Bin index -> value drawn uniformly inside the bin, floored to an integer.

    Flooring matches the integer support of every continuous column in Adult and
    makes the zero bin [0, 1) decode to exactly 0, preserving the spike.
    """
    lo, hi = edges[codes.to_numpy()], edges[codes.to_numpy() + 1]
    return pd.Series(np.floor(rng.uniform(lo, hi)), index=codes.index, dtype=float)


def fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    from sdmetrics.single_table import (
        KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity)
    meta = {"columns": {**{c: {"sdtype": "numerical"} for c in CONTINUOUS_COLS},
                        **{c: {"sdtype": "categorical"} for c in CATEGORICAL_COLS}}}
    return {
        "KSComplement": KSComplement.compute(real, syn, metadata=meta),
        "TVComplement": TVComplement.compute(real, syn, metadata=meta),
        "CorrelationSimilarity": CorrelationSimilarity.compute(real, syn, metadata=meta),
        "ContingencySimilarity": ContingencySimilarity.compute(real, syn, metadata=meta),
    }


def utility(real_loader, syn: pd.DataFrame) -> dict:
    """TSTR via synthcity, same metrics eval_synthcity.py reports."""
    from synthcity.metrics.eval import Metrics
    syn = syn.copy()
    syn[CATEGORICAL_COLS] = syn[CATEGORICAL_COLS].astype("category")
    score = Metrics.evaluate(
        X_gt=real_loader, X_syn=syn, task_type="classification",
        metrics={"performance": ["linear_model", "xgb"]}, workspace=WORKSPACE,
    )["mean"]
    return {
        "xgb_syn_id": score["performance.xgb.syn_id"],
        "xgb_gt": score["performance.xgb.gt"],
        "linear_syn_id": score["performance.linear_model.syn_id"],
    }


def run_one(b: int, real: pd.DataFrame, real_loader, log) -> dict:
    """One full sweep point: bin at b, fit AIM, generate, decode, score."""
    from snsynth.aim import AIMSynthesizer

    edges = build_edges(b)
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)          # marginal selection + mbi's rounding (NOT the DP noise)
    n_rows = len(real)

    discrete = real.copy()
    for col in CONTINUOUS_COLS:
        discrete[col] = encode(real[col], edges[col])
    cards = {c: len(edges[c]) - 1 for c in CONTINUOUS_COLS}
    log(f"  bins: {cards}")

    t0 = time.time()
    synth = AIMSynthesizer(epsilon=EPSILON, delta=DELTA, rounds=ROUNDS, degree=DEGREE,
                           max_cells=MAX_CELLS, max_model_size=MAX_MODEL_SIZE, verbose=True)
    # AIM prints "Selected ('colI', 'colJ') Size .. Budget Used .." per round; that
    # log is the only place the chosen marginals are exposed, so capture stdout.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        synth.fit(discrete, categorical_columns=list(discrete.columns), preprocessor_eps=0.0)
        sampled = synth.sample(n_rows)
    fit_secs = round(time.time() - t0, 1)

    # colN -> real column name (snsynth renames in the input column order)
    names = list(real.columns)
    selected = [tuple(names[int(i)] for i in re.findall(r"col(\d+)", m))
                for m in re.findall(r"Selected \(([^)]*)\)", buf.getvalue())]
    pairs = [s for s in selected if len(s) == 2]
    numeric_pairs = [s for s in pairs if all(c in CONTINUOUS_COLS for c in s)]
    log(f"  selected {len(selected)} marginals, {len(numeric_pairs)} numerical-numerical")
    for s in selected:
        mark = "  <-- numerical pair" if s in numeric_pairs else ""
        log(f"      {s}{mark}")

    syn = sampled[names].copy()
    for col in CONTINUOUS_COLS:
        syn[col] = decode(syn[col].astype(int), edges[col], rng)
    syn[CATEGORICAL_COLS] = syn[CATEGORICAL_COLS].astype(str)
    assert len(syn) == n_rows and not syn.isna().any().any()

    SYN_DIR.mkdir(parents=True, exist_ok=True)
    syn.to_csv(SYN_DIR / f"aim_bins{b}_synthetic.csv", index=False)

    # Binning ceiling: real data round-tripped through the SAME bins with AIM
    # removed. Separates "the bins lost it" from "AIM lost it" at every point.
    rt = real.copy()
    rt_rng = np.random.default_rng(SEED)
    for col in CONTINUOUS_COLS:
        rt[col] = decode(encode(real[col], edges[col]), edges[col], rt_rng)

    row = {"bins": b, "n_marginals": len(selected), "n_numeric_pairs": len(numeric_pairs),
           "numeric_pairs": "; ".join("x".join(s) for s in numeric_pairs) or "-",
           "fit_secs": fit_secs}
    row.update(fidelity(real, syn))
    row.update(utility(real_loader, syn))
    row.update({f"ceiling_{k}": v for k, v in fidelity(real, rt).items()})
    return row


def plot(df: pd.DataFrame):
    """Three stacked panels sharing the bin axis.

    Deliberately NOT a dual-axis chart: utility (0-1), correlation (0-1) and a
    raw count are different scales, so they get their own panels. Reading down
    the column answers the causal chain -- do numerical pairs get bought, does
    correlation recover, does utility follow?
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.sort_values("bins")
    x = d["bins"]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True)

    # 1. the U curve
    ax = axes[0]
    ax.plot(x, d["xgb_syn_id"], "-o", color=ORANGE, lw=2, ms=8, label="TSTR XGBoost")
    ax.plot(x, d["linear_syn_id"], "-s", color=GREY, lw=2, ms=8, label="TSTR logistic")
    if d["xgb_gt"].notna().any():
        gt = float(d["xgb_gt"].dropna().iloc[0])
        ax.axhline(gt, ls=":", lw=1, color=TEAL)
        ax.text(0.99, gt, "real-data ceiling (TRTR) ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=TEAL)
    ax.set_ylabel("TSTR AUC")
    ax.set_title("Utility vs bin count (AIM, ε = 1.0, single draw)", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # 2. correlation vs its binning ceiling -- the gap is AIM's loss, not binning's
    ax = axes[1]
    ax.plot(x, d["CorrelationSimilarity"], "-o", color=ORANGE, lw=2, ms=8, label="AIM")
    ax.plot(x, d["ceiling_CorrelationSimilarity"], "--", color=TEAL, lw=1.5,
            label="binning ceiling (no DP)")
    ax.set_ylabel("CorrelationSimilarity")
    ax.legend(fontsize=8, frameon=False)

    # 3. the diagnostic that explains the rest
    ax = axes[2]
    ax.bar(x.astype(str), d["n_numeric_pairs"], color=ORANGE, width=0.55)
    for xi, v in zip(range(len(d)), d["n_numeric_pairs"]):
        ax.text(xi, v, str(int(v)), ha="center", va="bottom", fontsize=8, color="#222222")
    ax.set_ylabel("numerical × numerical\nmarginals selected")
    ax.set_xlabel("Number of bins")

    for a in axes:
        a.grid(axis="y", lw=0.5, alpha=0.35)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=200)
    print(f"Wrote {CURVE_PNG}")


def main() -> int:
    grid = [int(a) for a in sys.argv[1:]] or BIN_GRID
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    real = pd.read_csv(DATA_DIR / "adult_train.csv")
    real[CONTINUOUS_COLS] = real[CONTINUOUS_COLS].astype(float)
    log(f"Loaded {real.shape} from data/adult_train.csv | sweeping bins {grid}")

    from synthcity.plugins.core.dataloader import GenericDataLoader
    loader_df = real.copy()
    loader_df[CATEGORICAL_COLS] = loader_df[CATEGORICAL_COLS].astype("category")
    real_loader = GenericDataLoader(loader_df, target_column=TARGET_COL)

    rows = []
    for b in grid:
        log(f"\n=== bins = {b} ===")
        try:
            row = run_one(b, real, real_loader, log)
            rows.append(row)
            log(f"  -> pairs={row['n_numeric_pairs']} xgb={row['xgb_syn_id']:.4f} "
                f"corr={row['CorrelationSimilarity']:.4f} ({row['fit_secs']}s)")
            # write after every setting so a late failure doesn't lose the sweep
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
            LOG_TXT.write_text("\n".join(lines))
        except Exception:
            log(f"  FAILED at bins={b}:\n{traceback.format_exc()}")

    if not rows:
        log("No successful runs.")
        return 1

    df = pd.DataFrame(rows)
    plot(df)
    cols = ["bins", "n_numeric_pairs", "xgb_syn_id", "linear_syn_id",
            "CorrelationSimilarity", "KSComplement", "TVComplement",
            "ContingencySimilarity", "fit_secs"]
    log("\n=== SUMMARY ===")
    log(df[cols].round(4).to_string(index=False))
    LOG_TXT.write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_CSV}\nWrote {LOG_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
