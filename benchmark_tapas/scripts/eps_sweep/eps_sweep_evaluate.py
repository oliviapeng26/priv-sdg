#!/usr/bin/env python3
"""Phase 1b of the formal-epsilon sweep: fidelity + utility at each eps.

Scores the per-seed DPGAN draws written by eps_sweep_generate.py, plus the
existing eps=1.0 arm, on exactly the metrics the rest of the repo reports.

THE METRIC DEFINITIONS ARE IMPORTED, NOT REIMPLEMENTED
    compute_fidelity comes from evaluation/eval_fidelity.py and
    build_feature_spec / compute_tstr / compute_trtr / compute_retention /
    load_and_split from evaluation/eval_utility.py. Copying them would let this
    sweep's numbers drift away from results/{utility,fidelity}_summary.csv while
    both tables still claimed to measure the same thing -- and the whole point of
    the sweep is that the eps=1.0 row is comparable to the repo's headline dpgan
    row. Importing makes that comparability structural instead of a promise.
    (benchmark_tapas/neural_tuning/convergence_check.py already imports
    compute_tstr/compute_trtr the same way.)

    In particular load_and_split() re-derives the 80/20 partition from
    adult_clean.csv and refuses to run if data/adult_{train,test}.csv are not the
    DATA_SPLIT_SEED split. TSTR is only meaningful if the 5,381 test records were
    genuinely held out, so that check is inherited deliberately.

EPS = 1.0 IS RE-SCORED, NOT RE-GENERATED
    Its draws are read from synthetic_data/runs/dpgan_seed{100..104}.csv -- the
    same files results/{utility,fidelity}_summary.csv were built from. Scoring is
    a couple of seconds per seed and gives the sweep per-seed rows for all four
    eps in one table, rather than three per-seed arms next to one pre-aggregated
    row. Because the inputs and the metric code are identical, the eps=1.0 rows
    must reproduce the committed dpgan summary; --check verifies exactly that and
    prints the deltas, so a silent divergence cannot pass unnoticed.

CPU ONLY
    Never imports torch or synthcity -- xgboost and SDMetrics only -- so it is
    re-runnable anywhere against run CSVs generated on the workstation, and the
    OpenMP collision the README documents cannot arise.

Outputs:
    benchmark_tapas/results/eps_sweep/utility_fidelity_per_run.csv
    benchmark_tapas/results/eps_sweep/utility_fidelity_summary.csv
    benchmark_tapas/results/eps_sweep/generation_cost.csv   (eval rows upserted)

Run from the repo root, after eps_sweep_generate.py:
  python benchmark_tapas/scripts/eps_sweep_evaluate.py
  python benchmark_tapas/scripts/eps_sweep_evaluate.py --epsilons 10 100
  python benchmark_tapas/scripts/eps_sweep_evaluate.py --no-check
"""

import argparse
import logging
import sys
import time
import tracemalloc
import traceback
from pathlib import Path

import pandas as pd

# benchmark_tapas/, found by walking up to config.py rather than counting parents --
# these scripts live in a subfolder now and may move again.
BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from config import RESULTS_DIR                                   # noqa: E402
from seeds import RUN_SEEDS, NUM_RUNS, set_all_seeds             # noqa: E402

METHOD = "dpgan"
SWEEP_EPSILONS = [0.1, 1.0, 10.0, 100.0]

SWEEP_DIR = RESULTS_DIR / "eps_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_ROOT = REPO_ROOT / "synthetic_data" / "eps_sweep"
LEGACY_RUNS = REPO_ROOT / "synthetic_data" / "runs"     # holds the eps=1.0 arm

PER_RUN_CSV = SWEEP_DIR / "utility_fidelity_per_run.csv"
SUMMARY_CSV = SWEEP_DIR / "utility_fidelity_summary.csv"
COST_CSV = SWEEP_DIR / "generation_cost.csv"            # shared with eps_sweep_generate.py

# Configure logging BEFORE importing the evaluation modules: both call
# logging.basicConfig at import time, and basicConfig is a no-op once the root
# logger already has handlers. Without this ordering their FileHandlers would win
# and this sweep's log would be written into evaluation/*_log.txt instead.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SWEEP_DIR / "eps_sweep_evaluate_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("eps_sweep_evaluate")

import eval_fidelity                                             # noqa: E402
import eval_utility                                              # noqa: E402

FIDELITY_METRICS = eval_fidelity.FIDELITY_METRICS
UTILITY_MODELS = eval_utility.UTILITY_MODELS
CONTINUOUS_COLS = eval_utility.CONTINUOUS_COLS


def eps_slug(eps: float) -> str:
    """Must match eps_sweep_generate.eps_slug -- same %g formatting, same paths."""
    return f"eps{eps:g}"


def run_path(eps: float, seed: int) -> Path:
    """Where this (eps, seed) draw lives.

    eps=1.0 is the pre-existing arm in synthetic_data/runs/; everything else was
    written by eps_sweep_generate.py under synthetic_data/eps_sweep/.
    """
    if eps == 1.0:
        return LEGACY_RUNS / f"{METHOD}_seed{seed}.csv"
    return SYNTH_ROOT / eps_slug(eps) / f"{METHOD}_seed{seed}.csv"


def load_run(eps: float, seed: int):
    """One synthetic draw, or None if it has not been generated yet."""
    path = run_path(eps, seed)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


def record_cost(row: dict) -> None:
    """Upsert one (epsilon, seed, stage) row into results/eps_sweep/generation_cost.csv."""
    df = pd.DataFrame([row])
    if COST_CSV.exists():
        existing = pd.read_csv(COST_CSV)
        if {"formal_epsilon", "seed", "stage"}.issubset(existing.columns):
            mask = ~((existing["formal_epsilon"] == row["formal_epsilon"])
                     & (existing["seed"] == row["seed"])
                     & (existing["stage"] == row["stage"]))
            existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(COST_CSV, index=False)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    """One row per eps: mean and std of every metric across run seeds."""
    metric_cols = [c for c in per_run.columns
                   if c not in ("formal_epsilon", "method", "run_idx", "seed")]
    rows = []
    for eps, g in per_run.groupby("formal_epsilon", sort=True):
        row = {"formal_epsilon": float(eps), "method": METHOD, "n_runs": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def check_against_committed(summary: pd.DataFrame, tol: float = 1e-9) -> None:
    """Verify the eps=1.0 rows reproduce the repo's committed dpgan numbers.

    Same synthetic CSVs and the same imported metric code, so any difference at all
    means something in the scoring path has moved -- a changed split, a different
    xgboost/sklearn version, a stale summary. Reported, never fatal: the sweep is
    still valid internally if the whole column shifted, and stopping the run would
    hide the size of the shift.
    """
    row = summary[summary.formal_epsilon == 1.0]
    if row.empty:
        log.info("--check: eps=1.0 not scored this run, nothing to compare")
        return
    row = row.iloc[0]

    pairs = []
    for src, cols in ((REPO_ROOT / "results" / "utility_summary.csv",
                       [f"tstr_{m}_auc" for m in UTILITY_MODELS]
                       + [f"trtr_{m}_auc" for m in UTILITY_MODELS]),
                      (REPO_ROOT / "results" / "fidelity_summary.csv",
                       list(FIDELITY_METRICS))):
        if not src.exists():
            log.warning(f"--check: {src.relative_to(REPO_ROOT)} missing, skipped")
            continue
        committed = pd.read_csv(src)
        committed = committed[committed.method == METHOD]
        if committed.empty:
            log.warning(f"--check: no {METHOD} row in {src.name}, skipped")
            continue
        committed = committed.iloc[0]
        for col in cols:
            key = f"{col}_mean"
            if key in row.index and key in committed.index:
                pairs.append((col, float(row[key]), float(committed[key])))

    if not pairs:
        log.warning("--check: no comparable columns found")
        return
    worst = max(abs(a - b) for _, a, b in pairs)
    log.info(f"--check: eps=1.0 vs committed dpgan summary, {len(pairs)} metrics, "
             f"max |delta| = {worst:.3e}")
    for name, got, want in pairs:
        flag = "" if abs(got - want) <= tol else "   <-- DIFFERS"
        log.info(f"    {name:<34} sweep={got:.6f}  committed={want:.6f}{flag}")
    if worst > tol:
        log.warning(
            f"--check: eps=1.0 does NOT reproduce the committed dpgan row "
            f"(max |delta| {worst:.3e}).\n"
            f"    The metric code is imported, not copied, and compute_fidelity/compute_tstr "
            f"are deterministic given their inputs -- so the inputs differ. By far the most "
            f"likely cause is that synthetic_data/ is gitignored, so the local "
            f"synthetic_data/runs/dpgan_seed*.csv are a DIFFERENT draw from the ones the "
            f"committed summary was built on (a CPU re-generation, or a pre-n_iter-cap "
            f"draw). dpgan is not bit-reproducible across devices at a fixed seed.\n"
            f"    Confirmed benign only if this run is on the machine that produced "
            f"results/*_summary.csv. Elsewhere, treat the sweep as internally consistent "
            f"(all four eps arms scored on one machine) but do NOT quote its eps=1.0 row "
            f"interchangeably with the repo's committed dpgan row.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SWEEP_EPSILONS,
                        help=f"budgets to score (default: {SWEEP_EPSILONS})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"how many of the {NUM_RUNS} run seeds to score (default: all)")
    parser.add_argument("--no-check", dest="check", action="store_false",
                        help="skip the eps=1.0 vs committed-summary comparison")
    args = parser.parse_args()
    run_seeds = RUN_SEEDS[:args.runs]

    log.info(f"=== eps sweep evaluation: eps={args.epsilons}, seeds={run_seeds} ===")

    train_df, test_df = eval_utility.load_and_split()
    spec = eval_utility.build_feature_spec(train_df, test_df)
    log.info(f"Feature layout: {len(spec['columns'])} columns, fitted on real data only")

    # Warm up SDMetrics before any timed region (run 0 otherwise pays the import
    # inside its timed block: 4.8s vs 0.57s, per eval_fidelity.py).
    import sdmetrics.single_table                              # noqa: F401

    # TRTR is a property of the real data, not of eps -- computed once per seed and
    # shared across every arm, so retention divides matched-seed numerator and
    # denominator. Expect trtr_*_std == 0: on fixed data both classifiers are
    # deterministic, so random_state has nothing to move.
    trtr_by_seed = {}
    for seed in run_seeds:
        set_all_seeds(seed)
        trtr_by_seed[seed] = eval_utility.compute_trtr(train_df, test_df, spec, seed)
    log.info("TRTR (real -> real): " + ", ".join(
        f"{m}={trtr_by_seed[run_seeds[0]][m]:.4f}" for m in UTILITY_MODELS))

    rows, missing = [], []
    for eps in args.epsilons:
        log.info(f"--- eps={eps:g} ---")
        for run_idx, seed in enumerate(run_seeds):
            synthetic = load_run(eps, seed)
            if synthetic is None:
                missing.append(f"eps{eps:g}_seed{seed}")
                continue
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                fidelity = eval_fidelity.compute_fidelity(train_df, synthetic)
                fid_s = round(time.perf_counter() - t0, 2)

                set_all_seeds(seed)
                t1 = time.perf_counter()
                tstr = eval_utility.compute_tstr(synthetic, test_df, spec, seed)
                util_s = round(time.perf_counter() - t1, 2)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                trtr = trtr_by_seed[seed]
                retention = eval_utility.compute_retention(tstr, trtr)

                row = {"formal_epsilon": float(eps), "method": METHOD,
                       "run_idx": run_idx, "seed": seed}
                for model in UTILITY_MODELS:
                    row[f"tstr_{model}_auc"] = tstr[model]
                    row[f"trtr_{model}_auc"] = trtr[model]
                    row[f"retention_{model}"] = retention[model]
                row.update(fidelity)
                rows.append(row)

                for stage, secs in (("fidelity", fid_s), ("utility", util_s)):
                    record_cost({"method": METHOD, "formal_epsilon": float(eps),
                                 "seed": seed, "stage": stage, "wall_clock_s": secs,
                                 "peak_memory_mb": round(peak_bytes / 1e6, 2),
                                 "device": "cpu", "n_iter": None})

                log.info(f"  seed {seed}: TSTR xgb={tstr['xgboost']:.4f} "
                         f"lr={tstr['logistic_regression']:.4f} | "
                         + ", ".join(f"{k}={v:.4f}" for k, v in fidelity.items()))
            except Exception:
                log.error(f"  eps={eps:g} seed={seed} FAILED:\n{traceback.format_exc()}")

    if missing:
        log.warning(f"{len(missing)} run(s) not generated yet, skipped: "
                    f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if not rows:
        log.warning("No runs scored -- run eps_sweep_generate.py first.")
        return 1

    per_run = pd.DataFrame(rows).sort_values(["formal_epsilon", "run_idx"]).reset_index(drop=True)
    per_run.to_csv(PER_RUN_CSV, index=False)
    summary = summarise(per_run)
    summary.to_csv(SUMMARY_CSV, index=False)
    log.info(f"Wrote {PER_RUN_CSV.relative_to(REPO_ROOT)} ({len(per_run)} rows) and "
             f"{SUMMARY_CSV.relative_to(REPO_ROOT)}")

    if args.check:
        check_against_committed(summary)

    show = ["formal_epsilon", "n_runs", "tstr_xgboost_auc_mean", "tstr_xgboost_auc_std",
            "tstr_logistic_regression_auc_mean", "retention_xgboost_mean",
            "KSComplement_mean", "TVComplement_mean",
            "CorrelationSimilarity_mean", "ContingencySimilarity_mean"]
    print("\n=== DPGAN utility + fidelity by formal epsilon (mean over seeds) ===")
    print(summary[[c for c in show if c in summary.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
