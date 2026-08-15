#!/usr/bin/env python3
"""AIM (Adaptive and Iterative Mechanism) synthetic data generation, eps = 1.0.

AIM is a marginal-based DP generator: it adaptively selects low-order marginals,
measures them under the Gaussian mechanism, and fits a graphical model (Private-PGM)
to the noisy measurements, then samples from that model.

Unlike the other four generators, AIM is NOT a Synthcity plugin, so it does not use
the Plugin.fit()/generate() interface. This script stands alone and writes the same
artefacts the sdg/*.ipynb notebooks do:
    synthetic_data/aim_synthetic_LEGACY.csv  (21,523 rows, same 13 columns, same dtypes)
    sdg/computational_overhead_LEGACY.csv (one row: wall time + peak memory)
    NOTE: this standalone mode is SUPERSEDED by sdg/generate_runs.py, which runs
    AIM once per seed and records cost to results/computational_cost.csv. This
    module is still imported by generate_runs.py for BIN_EDGES/encode/decode.
    sdg/aim_logs.txt                      (traceback on failure)

Implementation: SmartNoise SDK's `snsynth.aim.AIMSynthesizer`, a port of the
reference implementation at ryan112358/private-pgm `mechanisms/aim.py`, running on
`mbi` 1.1.0 (the jax Private-PGM inference engine) underneath. The mbi package
itself ships only the inference engine — the AIM mechanism lives in the repo's
`mechanisms/` folder, which pip does not install — so SmartNoise is the shortest
path to a working AIM.

ENVIRONMENT: runs in `priv-sdg`, the same env as everything else. smartnoise-synth
pins opacus<0.15 while synthcity's DPGAN needs opacus 1.x, so it must be installed as
`pip install --no-deps smartnoise-synth==1.0.8` (see requirements.txt). That pin is a
packaging artefact: AIM's real dependency chain is mbi + jax + opendp + smartnoise-sql,
and importing snsynth.aim pulls neither opacus nor torch. Verified — AIM and DPGAN both
run with opacus 1.4.1.

    conda activate priv-sdg
    python sdg/aim.py                                # tens of minutes, local CPU
    python evaluation/eval_fidelity.py aim           # fidelity, per run
    python evaluation/eval_utility.py aim            # utility, per run

Use `conda activate`, not `conda run -n ...`: conda run buffers stdout, so the long
generation step would print nothing until it finished.

DISCRETISATION: AIM requires an all-discrete domain, so the 5 continuous columns are
binned before fitting and decoded back afterwards. Bin edges are FIXED CONSTANTS taken
from the public Adult codebook (age 17-90, education_num 1-16, hours_per_week 1-99,
capital_gain/loss non-negative), never estimated from the training data — so no part of
the eps = 1.0 budget is spent on preprocessing, and the reported epsilon is the whole
story. capital_gain/capital_loss are ~91%/95% exact zeros, so they get a dedicated
zero bin plus log-spaced bins over the positive range; equal-width bins there would
smear the zero spike across a 5,000-wide bin and wreck both fidelity and utility.
Decoding draws uniformly inside the chosen bin and floors to an integer, matching the
integer support of the real columns (the zero bin decodes to exactly 0).

PRIVACY ACCOUNTING: AIM is natively zCDP (Gaussian measurements + exponential-mechanism
marginal selection), so SmartNoise converts the (eps, delta) budget to rho via opendp.
AIM cannot be pure-DP: rho-zCDP converts to (eps, delta)-DP only for delta > 0, and
delta=0 sets rho=0, i.e. infinite noise. At eps=1.0 the grid's deltas are NOT matched:
PrivBayes 0 (pure DP), AIM 1e-9 (rho=0.0150), DPGAN 4.65e-5 = 1/n (synthcity's silent
default, gan.py:591; rho=0.0367).

delta=1e-9 is deliberate, not merely inherited. It is the strictest delta in the grid,
so it keeps AIM as close as the algorithm allows to its primary contrast (PrivBayes,
same quadrant, delta=0) and no utility win can be blamed on a loose parameter. It also
matches the AIM paper (McKenna et al., arXiv:2201.12677 — "varying eps in [0.01,100.0]
and setting delta=1e-9"), the reference implementation's default_params(), and
SmartNoise's default, so results stay comparable to published AIM numbers. The cost is
that AIM absorbs ~56% more noise than DPGAN's budget permits (initial sigma 87.8 vs
56.1) — matching DPGAN's delta would be equivalent to giving AIM eps=1.6. That confound
runs AGAINST AIM, so a utility win is unattackable.

TODO if AIM underperforms the others on utility: rerun with DELTA = 4.65e-5 and report
it as a sensitivity line, to separate the stricter budget from the algorithm itself.

REPRODUCIBILITY: this run is NOT bit-reproducible, and SEED does not make it so. The
Gaussian measurements come from opendp's `make_gaussian` (via snsynth's gaussian_noise),
which draws from opendp's own CSPRNG with no seeding hook — verified empirically, two
calls under the same np.random.seed give different noise. That is deliberate on opendp's
part: a seedable DP noise source is a security hazard. SEED therefore controls only the
exponential mechanism's marginal selection, mbi's randomised rounding at sampling time,
and the decode-time uniform draw inside each bin. Since the selected marginals depend on
the noisy errors, reruns will differ end to end. Treat the saved CSV as one draw, and
keep it rather than regenerating if a number needs to be traced back.

MEMORY MEASUREMENT: tracemalloc only sees Python-level allocations, so peak_memory_mb
understates AIM's real footprint — the graphical model lives in jax buffers outside
Python's allocator. Same caveat as the torch-based CTGAN/DPGAN rows in
computational_overhead_LEGACY.csv; wall time is the trustworthy column for cross-method
comparison.
"""

import sys
import time
import tracemalloc
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

METHOD = "aim"
EPSILON = 1.0
DELTA = 1e-9          # strictest delta in the grid — see PRIVACY ACCOUNTING above
SEED = 42             # partial only — see REPRODUCIBILITY above

# AIM hyperparameters (snsynth defaults, listed explicitly for the writeup)
DEGREE = 2            # workload = all 2-way marginals
MAX_CELLS = 10_000    # drop workload marginals bigger than this
MAX_MODEL_SIZE = 80   # MB budget for the graphical model
ROUNDS = None         # None -> 16 * n_columns = 208 rounds (reference default)

REPO_ROOT = Path(__file__).resolve().parent.parent
SDG_DIR = REPO_ROOT / "sdg"
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "synthetic_data"

CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]
TARGET_COL = "income"

# Bin edges per continuous column: length = n_bins + 1, half-open [edge[i], edge[i+1]).
# Public-codebook bounds only — nothing here is read off the training data.
BIN_EDGES = {
    # 17-90 in 20 equal-width bins (upper edge 91 so age 90 lands inside the last bin)
    "age": np.linspace(17, 91, 21),
    # 1-16, one integer per bin -> decoded exactly
    "education_num": np.arange(1, 18, dtype=float),
    # zero bin [0, 1) + 19 log-spaced bins over [1, 100000): 91% of rows are exact 0
    "capital_gain": np.concatenate([[0.0], np.geomspace(1, 100_000, 20)]),
    # zero bin [0, 1) + 19 log-spaced bins over [1, 5000): 95% of rows are exact 0
    "capital_loss": np.concatenate([[0.0], np.geomspace(1, 5_000, 20)]),
    # 1-99 in 20 equal-width bins (upper edge 100 so 99 lands inside the last bin)
    "hours_per_week": np.linspace(1, 100, 21),
}


def encode(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Continuous column -> integer bin index in [0, len(edges) - 2]."""
    clipped = values.clip(edges[0], np.nextafter(edges[-1], edges[0]))
    return pd.Series(np.searchsorted(edges, clipped, side="right") - 1,
                     index=values.index, dtype=int)


def decode(codes: pd.Series, edges: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """Integer bin index -> float value drawn uniformly inside the bin, floored.

    Flooring matches the integer support of every continuous column in Adult and makes
    the zero bin [0, 1) decode to exactly 0, preserving the capital_gain/loss spike.
    """
    lo = edges[codes.to_numpy()]
    hi = edges[codes.to_numpy() + 1]
    return pd.Series(np.floor(rng.uniform(lo, hi)), index=codes.index, dtype=float)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    # Seeds marginal selection + mbi's randomised rounding, NOT the Gaussian noise.
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)   # decode-only: uniform draw inside each bin

    from snsynth.aim import AIMSynthesizer

    df = pd.read_csv(DATA_DIR / "adult_train.csv")
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    n_rows = len(df)
    print(f"Loaded {df.shape} from data/adult_train.csv")

    # --- discretise: all 13 columns become categorical for AIM ---
    discrete = df.copy()
    for col in CONTINUOUS_COLS:
        discrete[col] = encode(df[col], BIN_EDGES[col])
        n_bins = len(BIN_EDGES[col]) - 1
        print(f"  {col}: {n_bins} bins, {discrete[col].nunique()} occupied")

    # --- fit + generate ---
    tracemalloc.start()
    t0 = time.time()

    synth = AIMSynthesizer(
        epsilon=EPSILON, delta=DELTA, rounds=ROUNDS, degree=DEGREE,
        max_cells=MAX_CELLS, max_model_size=MAX_MODEL_SIZE, verbose=True,
    )
    # preprocessor_eps=0.0: every column reaching snsynth is already discrete, so the
    # transformer is pure label encoding and consumes none of the budget.
    synth.fit(discrete, categorical_columns=list(discrete.columns), preprocessor_eps=0.0)
    sampled = synth.sample(n_rows)

    wall_time = round(time.time() - t0, 2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = round(peak / 1e6, 2)

    # --- decode back to the original schema ---
    assert set(sampled.columns) == set(df.columns), \
        f"column mismatch: {set(df.columns) ^ set(sampled.columns)}"
    synthetic = sampled[df.columns.tolist()].copy()   # restore the original column order
    for col in CONTINUOUS_COLS:
        synthetic[col] = decode(synthetic[col].astype(int), BIN_EDGES[col], rng)
    synthetic[CATEGORICAL_COLS] = synthetic[CATEGORICAL_COLS].astype(str)

    assert len(synthetic) == n_rows, f"row count {len(synthetic)} != {n_rows}"
    assert not synthetic.isna().any().any(), "NaNs in decoded output"

    synthetic.to_csv(OUT_DIR / f"{METHOD}_synthetic_LEGACY.csv", index=False)
    print(f"\nDone — {len(synthetic)} rows | {wall_time}s | {peak_mb} MB peak")
    print(synthetic.head())
    print(synthetic[CONTINUOUS_COLS].describe().T[["min", "mean", "max"]])

    # --- overhead log (same schema/file the notebooks append to) ---
    overhead_path = SDG_DIR / "computational_overhead_LEGACY.csv"
    row = pd.DataFrame([{"method": METHOD, "rows_generated": len(synthetic),
                         "wall_time_s": wall_time, "peak_memory_mb": peak_mb}])
    if overhead_path.exists():
        existing = pd.read_csv(overhead_path)
        existing = existing[existing["method"] != METHOD]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(overhead_path, index=False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log = f"ERROR running {METHOD} (epsilon={EPSILON}, delta={DELTA}):\n{traceback.format_exc()}"
        print(log)
        (SDG_DIR / f"{METHOD}_logs.txt").write_text(log)
        sys.exit(1)
