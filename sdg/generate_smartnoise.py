#!/usr/bin/env python3
"""Seeded, repeated synthetic data generation for the two SmartNoise generators.

The SmartNoise counterpart of sdg/generate_runs.py: same 21,523-row training
split, same RUN_SEEDS, same fit+generate timing region, same row-count/NaN
asserts, same resumability -- but the models come from smartnoise-synth rather
than Synthcity, and the outputs are keyed by epsilon so a later budget sweep
cannot overwrite this one.

    aim       snsynth.aim.AIMSynthesizer      (marginal-based, zCDP)
    dpctgan   snsynth.pytorch.nn.DPCTGAN      (CTGAN + DP-SGD on the discriminator)

Outputs:
    synthetic_data/smartnoise/{generator}/eps{eps}/seed{seed}.csv
    results/smartnoise/generation_cost.csv    wall clock, peak memory, device

GENERATION ONLY. Scoring is evaluation/eval_fidelity.py and
evaluation/eval_utility.py, which carry `aim` and `dpctgan` alongside the four
Synthcity methods and know to look for them in the eps-keyed tree above (their
--epsilon flag picks the arm). Keeping generation separate is deliberate: this
script imports torch and jax and never xgboost, exactly as sdg/generate_runs.py
does, so the two OpenMP runtimes never meet (README).

WHY A SEPARATE SCRIPT AND NOT A METHOD IN generate_runs.py
    Three things generate_runs.py cannot express without being rewritten around
    them, none worth forcing into a script the committed results/ tables depend on:

    1. FORMAL_EPSILON is a module constant there (no --epsilon flag), and its
       outputs are not keyed by eps -- synthetic_data/runs/{method}_seed{seed}.csv
       is read by both eval scripts, so two budgets would silently overwrite each
       other. Same blocker eps_sweep_generate.py was split out for.
    2. Its `aim` row and this one's are the SAME configuration but land in
       different trees, and this file's whole point is that the SmartNoise pair is
       reported together, at a stated eps, as its own arm.
    3. dpctgan needs a transformer built by hand (see PUBLIC BOUNDS below); no
       other method in the repo takes a `transformer` argument at all.

EPSILON DOES NOT MEAN THE SAME THING FOR THE TWO GENERATORS -- READ BEFORE COMPARING
    AIM is calibrated: (eps, delta) is converted to rho-zCDP and the Gaussian noise
    is scaled so exactly that budget is spent over its 208 rounds. eps is the input.

    DP-CTGAN inverts it. sigma is FIXED at 5 and eps is a STOPPING RULE: it trains,
    asks the Opacus accountant what has been spent, and breaks out of the epoch loop
    the first time the spend exceeds the target (snsynth/pytorch/nn/dpctgan.py:327,
    the `if self.epsilon < epsilon` branch). It does this SILENTLY -- a budget
    exhausted in three epochs looks exactly like a completed 300-epoch run from the
    outside. delta is not passed either: it is derived internally as 1/(n*sqrt(n)),
    which on the 21,523-row training split is 3.17e-7 (versus AIM's 1e-9 here, and
    synthcity DPGAN's 1/n = 4.65e-5).

    THE STOPPING RULE OVERSHOOTS, AND THE OVERSHOOT IS WHAT WAS ACTUALLY SPENT.
    The accountant is queried at the TOP of each epoch (dpctgan.py:307-327, before
    that epoch's steps at :335), so at epoch i it reports the cost of epochs 0..i-1.
    Breaking there means i epochs have already been trained at a cost of
    epsilon_list[i], and epsilon_list[i] is by definition GREATER than the requested
    budget -- that is the condition that fired. The released synthesiser has
    therefore spent `epsilon_spent`, not `epsilon`, and the excess is up to one
    epoch's worth of budget. Verified against the control flow, not assumed.

    So `epsilon_spent` is the number to report for this generator; the requested eps
    is a target it steps past. Both go into results/smartnoise/generation_cost.csv,
    alongside `epochs_run`.

    Both generators are taken exactly as the libraries ship them, and "eps = 1.0"
    therefore labels two different mechanisms, with two different deltas and only one
    of them exact. That is deliberate -- the same choice
    benchmark_tapas/scripts/eps_sweep/spike_diagnosis/dpctgan_generator.py documents
    -- but it means the epochs actually trained are a REPORTED QUANTITY rather than a
    configuration detail. A run that stopped in single-digit epochs is an undertrained
    network whose weak utility says nothing about DP-SGD. Check both columns before
    reading the utility table.

PUBLIC BOUNDS, AND WHY preprocessor_eps IS ZERO FOR BOTH
    Neither generator may spend budget learning column ranges, so the ranges are
    supplied from sdg/aim.py's BIN_EDGES -- fixed public-codebook constants (age
    17-91, education_num 1-17, hours_per_week 1-100, a dedicated zero bin for
    capital_gain/loss), never read off the training data. AIM consumes them as bin
    edges via encode/decode; DP-CTGAN consumes the first and last edge as the
    lower/upper of a MinMaxTransformer(epsilon=0.0), which requires explicit bounds
    precisely because it will not estimate them for free.

    They are imported from sdg/aim.py rather than restated so the two generators
    cannot drift apart on the one thing that keeps the reported epsilon honest.

REPRODUCIBILITY IS ASYMMETRIC
    dpctgan is seeded: set_all_seeds(seed) runs with torch already imported, and
    Opacus draws its DP-SGD noise from the torch RNG, so a rerun on the same device
    reproduces the draw. Across devices it does not -- the same caveat the README
    records for synthcity's GANs.

    aim is NOT seeded and cannot be. Its Gaussian measurements come from opendp's
    CSPRNG, which has no seeding hook by design; the seed reaches only the
    exponential mechanism's marginal selection, mbi's randomised rounding and the
    decode-time uniform draw. Reruns differ end to end. Treat each CSV as one draw.

    NOTE: synthetic_data/runs/aim_seed{100..104}.csv already holds five draws of
    this exact configuration (generate_runs.py, eps=1.0). Because AIM is not
    reproducible these are INDEPENDENT draws, not copies -- useful as a replication,
    but if you would rather score the existing ones, copy them in instead of fitting:
        for s in 100 101 102 103 104; do
          cp synthetic_data/runs/aim_seed$s.csv synthetic_data/smartnoise/aim/eps1/seed$s.csv
        done
    This script skips any (generator, eps, seed) whose CSV already exists.

PEAK MEMORY CAVEAT
    tracemalloc sees Python-level allocations only, so peak_memory_mb understates
    both generators badly -- AIM's graphical model lives in jax buffers and
    DP-CTGAN's network in torch buffers, neither of which goes through Python's
    allocator. Wall clock is the trustworthy column.

Run from the repo root, env active:
  python sdg/generate_smartnoise.py                       # both generators, eps=1.0, 5 seeds
  python sdg/generate_smartnoise.py aim                   # one generator
  python sdg/generate_smartnoise.py dpctgan --epsilon 10  # a different budget
  python sdg/generate_smartnoise.py --runs 2 --regenerate

Then score them with the ordinary eval scripts:
  python evaluation/eval_fidelity.py aim dpctgan
  python evaluation/eval_utility.py aim dpctgan
  python evaluation/eval_fidelity.py --epsilon 10 aim dpctgan   # a different arm

Resumable: an existing run CSV is reused and the fit skipped, so an interrupted
session picks up where it stopped. --regenerate forces a re-fit (and overwrites
that run's cost row).

ENVIRONMENT NOTE FOR dpctgan
    snsynth 1.0.8's DPCTGAN is written against the Opacus 0.x API
    (`opacus.PrivacyEngine(batch_size=...).attach(optimizer)`). The `priv-sdg`
    conda env carries opacus 1.4.1, which synthcity's DPGAN requires, and DP-CTGAN
    raises `TypeError: PrivacyEngine.__init__() got an unexpected keyword argument
    'batch_size'` there. `aim` is unaffected -- snsynth.aim imports neither opacus
    nor torch, which is the whole reason smartnoise-synth goes in with --no-deps
    (requirements.txt). Run dpctgan in the environment the DP-CTGAN TAPAS audit
    used; --probe fails fast and says so if the API is wrong.
"""

import argparse
import logging
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SDG_DIR = REPO_ROOT / "sdg"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDG_DIR))          # sibling import of aim.py

from seeds import RUN_SEEDS, NUM_RUNS, set_all_seeds        # noqa: E402

DEFAULT_EPSILON = 1.0
ALL_GENERATORS = ["aim", "dpctgan"]

# DP-CTGAN: every parameter at the SmartNoise default, listed explicitly so the
# write-up can state them without reading the library. Only `verbose` deviates,
# and that is a logging flag rather than a mechanism parameter.
DPCTGAN_SIGMA = 5
DPCTGAN_EPOCH_CAP = 300         # a CAP, not a schedule -- see the epsilon note above
DPCTGAN_BATCH_SIZE = 500
DPCTGAN_MAX_PER_SAMPLE_GRAD_NORM = 1.0

CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]

DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
SYNTH_ROOT = REPO_ROOT / "synthetic_data" / "smartnoise"
RESULTS_DIR = REPO_ROOT / "results" / "smartnoise"
COST_CSV = RESULTS_DIR / "generation_cost.csv"

EXPECTED_TRAIN_N = 21_523

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RESULTS_DIR / "generate_smartnoise_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("generate_smartnoise")


def eps_slug(eps: float) -> str:
    """Path-safe, round-trippable label for one budget: 0.1 -> 'eps0.1', 10.0 -> 'eps10'.

    %g rather than str(), matching eps_sweep_generate.eps_slug, so 10 and 10.0 land
    in one directory instead of splitting an arm across 'eps10' and 'eps10.0'.
    """
    return f"eps{eps:g}"


def load_train_df() -> pd.DataFrame:
    """The 21,523-row training split -- the only data any generator ever sees.

    Same row-count assert as sdg/generate_runs.py. The full check that these CSVs
    really are the DATA_SPLIT_SEED partition lives in data_preprocessing.ipynb and
    is repeated in evaluation/eval_utility.py -- that is where a stale split would
    do real damage, by making the TSTR holdout not held out.
    """
    df = pd.read_csv(TRAIN_CSV)
    assert len(df) == EXPECTED_TRAIN_N, f"train is {len(df)}, expected {EXPECTED_TRAIN_N}"
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def device_for(generator: str) -> str:
    """The device this generator will actually use -- recorded, not assumed."""
    if generator == "dpctgan":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    try:                                   # aim runs its inference engine on jax
        import jax
        return jax.default_backend()
    except Exception:
        return "cpu"


# ------------------------------------------------------------ generators ----

def generate_aim(train_data: pd.DataFrame, seed: int, epsilon: float):
    """Fit + sample AIM at one (seed, epsilon).

    The round trip lives in sdg/aim.py:fit_sample -- the same call
    sdg/generate_runs.py makes, with epsilon lifted from that script's module
    constant to an argument here. Calling it rather than restating it is what makes
    "the same AIM configuration at a different budget" true by construction instead
    of by inspection.

    Returns (synthetic, diagnostics) to match generate_dpctgan's signature. AIM's
    diagnostics are empty because its budget is calibrated rather than a stopping
    rule, so there is no epochs_run/epsilon_spent to report.
    """
    from aim import fit_sample
    return fit_sample(train_data, seed, epsilon=epsilon), {}


def dpctgan_transformer(columns):
    """A TableTransformer with PUBLIC bounds, so preprocessor_eps can stay 0.

    MinMaxTransformer(epsilon=0.0) refuses to estimate lower/upper and requires them
    to be given; they come from the codebook bin edges, not from the data.

    Categoricals are CHAINED, not bare. LabelTransformer alone hands the network an
    integer code, so the generator's output comes back as a float and its inverse
    does categories[np.float32] -> TypeError. The GAN-family synthesisers want
    one-hot: label -> integer code -> one-hot on the way in, and the inverse unwinds
    both. (Same construction as the TAPAS wrapper in
    benchmark_tapas/scripts/eps_sweep/spike_diagnosis/dpctgan_generator.py.)
    """
    from aim import BIN_EDGES
    from snsynth.transform import (TableTransformer, MinMaxTransformer,
                                   LabelTransformer, OneHotEncoder, ChainTransformer)
    parts = []
    for col in columns:
        if col in CONTINUOUS_COLS:
            edges = BIN_EDGES[col]
            parts.append(MinMaxTransformer(lower=float(edges[0]),
                                           upper=float(edges[-1]), epsilon=0.0))
        else:
            parts.append(ChainTransformer([LabelTransformer(), OneHotEncoder()]))
    return TableTransformer(parts)


def generate_dpctgan(train_data: pd.DataFrame, seed: int, epsilon: float,
                     device: str, epochs: int = DPCTGAN_EPOCH_CAP):
    """Fit + sample DP-CTGAN at one (seed, epsilon).

    Returns (synthetic, diagnostics). The diagnostics are not decoration: epsilon is
    a stopping rule here, so `epochs_run` is the only evidence of how much training
    the budget actually bought, and the library will not tell you otherwise.

    The accountant is queried at the TOP of each epoch, before that epoch's steps, so
    epsilon_list gains one entry per epoch ENTERED and entry i is the cost of epochs
    0..i-1. Hence epochs_run = len(epsilon_list) - 1 when it broke early (the last
    entry belongs to an epoch that was never trained), and the full cap when it did
    not -- and epsilon_spent, the last entry, exceeds `epsilon` on an early break.

    When the CAP binds instead and all 300 epochs train, that last entry is the cost
    of the first 299: the accountant is never queried again, so epsilon_spent then
    understates the final spend by one epoch rather than overstating it. Which case
    a run is in is legible from epochs_run.
    """
    from snsynth.pytorch.nn import DPCTGAN

    raw = train_data.copy()
    cols = list(raw.columns)

    synth = DPCTGAN(
        epsilon=epsilon,
        sigma=DPCTGAN_SIGMA,
        epochs=epochs,
        batch_size=DPCTGAN_BATCH_SIZE,
        max_per_sample_grad_norm=DPCTGAN_MAX_PER_SAMPLE_GRAD_NORM,
        cuda=(device == "cuda"),
        verbose=False,          # logging only; the default prints every epoch
    )
    synth.fit(raw, transformer=dpctgan_transformer(cols), preprocessor_eps=0.0,
              categorical_columns=[c for c in cols if c in CATEGORICAL_COLS],
              continuous_columns=[c for c in cols if c in CONTINUOUS_COLS])

    spent = getattr(synth, "epsilon_list", None) or getattr(
        getattr(synth, "gan", None), "epsilon_list", None)
    diagnostics = {}
    if spent:
        broke_early = float(spent[-1]) > epsilon
        diagnostics = {"epochs_run": len(spent) - 1 if broke_early else len(spent),
                       "epsilon_spent": float(spent[-1])}

    sampled = synth.sample(len(train_data))
    synthetic = sampled[train_data.columns.tolist()].copy()
    synthetic[CONTINUOUS_COLS] = synthetic[CONTINUOUS_COLS].astype(float)
    synthetic[CATEGORICAL_COLS] = synthetic[CATEGORICAL_COLS].astype(str)
    return synthetic, diagnostics


# ------------------------------------------------------------- bookkeeping --

# Fixed column order, so the header does not reshuffle between runs.
#
# Generation cost lands here rather than in results/computational_cost.csv because
# that table is keyed (method, seed, stage) with no eps column, so two budgets would
# collapse onto the same keys. The fidelity/utility rows for these methods DO go to
# results/computational_cost.csv, written by the eval scripts alongside every other
# method -- which means those eval rows are not eps-keyed and record whichever arm
# was scored last. Generation, the expensive half, is the one that needed the split.
COST_COLUMNS = ["method", "formal_epsilon", "seed", "stage", "wall_clock_s",
                "peak_memory_mb", "device", "epochs_run", "epsilon_spent"]


def record_cost(row: dict) -> None:
    """Upsert one (method, formal_epsilon, seed, stage) row into generation_cost.csv.

    A SmartNoise-local cost table rather than results/computational_cost.csv: that
    file is keyed on (method, seed, stage) with no eps column, so two budgets would
    collapse onto the same keys, and its `aim` rows already belong to the
    generate_runs.py arm. Same reasoning as eps_sweep_generate.record_cost.
    """
    df = pd.DataFrame([row])
    if COST_CSV.exists():
        existing = pd.read_csv(COST_CSV)
        if {"method", "formal_epsilon", "seed", "stage"}.issubset(existing.columns):
            mask = ~((existing["method"] == row["method"])
                     & (existing["formal_epsilon"] == row["formal_epsilon"])
                     & (existing["seed"] == row["seed"])
                     & (existing["stage"] == row["stage"]))
            existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.reindex(columns=COST_COLUMNS).to_csv(COST_CSV, index=False)


def generate_one(generator: str, train_data: pd.DataFrame, epsilon: float,
                 seed: int, device: str, regenerate: bool) -> bool:
    """Generate (or reuse) one (generator, eps, seed) run. True if a fit happened."""
    out_dir = SYNTH_ROOT / generator / eps_slug(epsilon)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed{seed}.csv"

    if path.exists() and not regenerate:
        log.info(f"  cached at {path.relative_to(REPO_ROOT)}, skipping "
                 f"(--regenerate to re-fit)")
        return False

    set_all_seeds(seed)

    # Timed region covers fit + generate together, matching sdg/generate_runs.py so
    # the numbers stay comparable to results/computational_cost.csv.
    tracemalloc.start()
    t0 = time.perf_counter()
    if generator == "aim":
        synthetic, diagnostics = generate_aim(train_data, seed, epsilon)
    else:
        synthetic, diagnostics = generate_dpctgan(train_data, seed, epsilon, device)
    wall_clock_s = round(time.perf_counter() - t0, 2)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = round(peak_bytes / 1e6, 2)

    synthetic = synthetic[train_data.columns.tolist()]
    assert len(synthetic) == len(train_data), \
        f"{generator} eps={epsilon:g} seed={seed}: generated {len(synthetic)} rows, " \
        f"expected {len(train_data)}"
    assert not synthetic.isna().any().any(), \
        f"{generator} eps={epsilon:g} seed={seed}: NaNs in output"
    synthetic.to_csv(path, index=False)

    record_cost({"method": generator, "formal_epsilon": float(epsilon), "seed": seed,
                 "stage": "generation", "wall_clock_s": wall_clock_s,
                 "peak_memory_mb": peak_memory_mb, "device": device, **diagnostics})
    extra = (f" | epochs_run={diagnostics['epochs_run']}/{DPCTGAN_EPOCH_CAP}, "
             f"eps_spent={diagnostics['epsilon_spent']:.3f}" if diagnostics else "")
    log.info(f"  {len(synthetic)} rows | {wall_clock_s}s | {peak_memory_mb} MB peak "
             f"| device={device}{extra} -> {path.relative_to(REPO_ROOT)}")
    return True


def probe(train_data: pd.DataFrame, epsilon: float, device: str, epochs: int) -> int:
    """Fit DP-CTGAN once at a low epoch cap and report what the budget bought.

    Cheap insurance against the silent break: if a full-cap run would stop after a
    handful of epochs, or if the installed Opacus has the wrong API for snsynth's
    DP-CTGAN (see the ENVIRONMENT NOTE), this says so in a minute rather than
    partway through five 300-epoch fits. Writes nothing.
    """
    log.info(f"--- probe: dpctgan, eps={epsilon:g}, epochs={epochs}, device={device} ---")
    set_all_seeds(RUN_SEEDS[0])
    t0 = time.perf_counter()
    _, diagnostics = generate_dpctgan(train_data, RUN_SEEDS[0], epsilon, device, epochs)
    log.info(f"  fit+sample in {time.perf_counter() - t0:.1f}s; diagnostics={diagnostics}")
    if not diagnostics:
        log.warning("  no epsilon_list on the fitted synthesiser -- the accountant "
                    "diagnostics will be blank in generation_cost.csv")
    elif diagnostics["epochs_run"] < epochs:
        log.warning(f"  budget exhausted after {diagnostics['epochs_run']} epoch(s) at "
                    f"eps={epsilon:g}. A full run will stop there too: the cap is not "
                    f"binding, the budget is. Report this next to the utility numbers.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generators", nargs="*", default=None,
                        help=f"generators to run (default: all of {ALL_GENERATORS})")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"formal DP budget (default: {DEFAULT_EPSILON})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"how many of the {NUM_RUNS} run seeds to use (default: all)")
    parser.add_argument("--regenerate", action="store_true",
                        help="re-fit even when a cached run CSV exists")
    parser.add_argument("--probe", type=int, metavar="EPOCHS", default=None,
                        help="fit dpctgan once at EPOCHS and report epochs_run / "
                             "epsilon_spent, then exit. Writes nothing.")
    args = parser.parse_args()

    generators = args.generators or ALL_GENERATORS
    unknown = set(generators) - set(ALL_GENERATORS)
    if unknown:
        parser.error(f"unknown generator(s): {sorted(unknown)}. Known: {ALL_GENERATORS}")
    run_seeds = RUN_SEEDS[:args.runs]

    train_df = load_train_df()
    log.info(f"Loaded training split: {train_df.shape}")

    if args.probe is not None:
        return probe(train_df, args.epsilon, device_for("dpctgan"), args.probe)

    devices = {g: device_for(g) for g in generators}
    log.info(f"=== generate_smartnoise: generators={generators}, "
             f"eps={args.epsilon:g}, seeds={run_seeds} ===")
    log.info("devices: " + ", ".join(f"{g}={d}" for g, d in devices.items()))

    # Warm up the heavy imports BEFORE any timed region. Without this the first run
    # of a session pays the import cost inside its timed block -- measured at a 5x
    # inflation on run 0 in sdg/generate_runs.py, which corrupts both the mean and
    # the std of the cost table.
    for gen in generators:
        t0 = time.perf_counter()
        if gen == "aim":
            import snsynth.aim  # noqa: F401
        else:
            from snsynth.pytorch.nn import DPCTGAN  # noqa: F401
        log.info(f"{gen} imports warmed up in {time.perf_counter() - t0:.1f}s (untimed)")

    fitted = 0
    for gen in generators:
        log.info(f"--- {gen}, eps={args.epsilon:g} ---")
        for run_idx, seed in enumerate(run_seeds):
            log.info(f"  run {run_idx}/{len(run_seeds) - 1}, seed {seed}")
            try:
                fitted += generate_one(gen, train_df, args.epsilon, seed,
                                       devices[gen], args.regenerate)
            except Exception:
                log.error(f"  {gen} eps={args.epsilon:g} seed={seed} FAILED:\n"
                          f"{traceback.format_exc()}")

    log.info(f"=== Done: {fitted} run(s) generated under "
             f"{SYNTH_ROOT.relative_to(REPO_ROOT)}/ ===")
    if COST_CSV.exists():
        cost = pd.read_csv(COST_CSV)
        gen_rows = cost[cost.stage == "generation"]
        if not gen_rows.empty:
            print("\n=== generation cost (mean over seeds) ===")
            print(gen_rows.groupby(["method", "formal_epsilon", "device"], sort=True).agg(
                n_runs=("seed", "count"),
                wall_clock_s_mean=("wall_clock_s", "mean"),
                wall_clock_s_std=("wall_clock_s", "std"),
                peak_memory_mb_mean=("peak_memory_mb", "mean"),
                epochs_run_mean=("epochs_run", "mean"),
            ).reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
