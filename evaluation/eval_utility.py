#!/usr/bin/env python3
"""TSTR / TRTR / retention utility, scored per run on a genuine holdout.

Reads the per-run synthetic CSVs written by sdg/generate_runs.py
(the four Synthcity methods) and sdg/generate_smartnoise.py (aim,
dpctgan), trains
classifiers on them, and scores AUC on data/adult_test.csv -- the 5,381 records
produced by the DATA_SPLIT_SEED split that no generator has ever seen.

WHY NOT SYNTHCITY'S TSTR (the leak this script exists to fix)
    `Metrics.evaluate(..., metrics={"performance": [...]})` reports
    performance.xgb.syn_id / syn_ood. Internally it takes the REAL loader it is
    handed -- data/adult_train.csv, the same 21,523 rows every generator was
    fitted on -- and makes its OWN 80/20 split of it, training the classifier on
    synthetic data and scoring on that internal holdout. Those "held-out"
    records were in the generator's training set, so a generator that memorises
    is rewarded. performance.*.gt is scored on the same leaky split, so the
    syn_id/gt ratio does not correct for it either.

    Here TSTR scores on the real holdout, TRTR trains on the full 21,523-row
    training set and scores on the SAME 5,381 records, and retention =
    TSTR/TRTR is therefore a like-for-like ratio.

Outputs:
    results/utility_per_run.csv    one row per (method, run)
    results/utility_summary.csv    mean/std per method

Run from repo root, after sdg/generate_runs.py:
  python evaluation/eval_utility.py                     # every method with run CSVs
  python evaluation/eval_utility.py bayesian_network    # one method
  python evaluation/eval_utility.py --runs 2            # first 2 seeds only

Cheap and CPU-only -- it never imports torch or synthcity, so it can be re-run
freely against run CSVs generated on another machine. `compute_tstr` /
`compute_trtr` are also imported by benchmark_tapas/neural_tuning/
convergence_check.py, which uses them for the n_iter sweep.
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
sys.path.insert(0, str(REPO_ROOT))

from seeds import DATA_SPLIT_SEED, RUN_SEEDS, NUM_RUNS, set_all_seeds

ALL_METHODS = ["bayesian_network", "privbayes", "ctgan", "dpgan",
               "aim", "dpctgan"]

CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]
TARGET_COL = "income"
POSITIVE_CLASS = ">50K"
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_COLS if c != TARGET_COL]

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "synthetic_data" / "runs"

# Where each method's per-seed CSVs live. The four Synthcity methods come from
# sdg/generate_runs.py's flat tree; aim and dpctgan come from
# sdg/generate_smartnoise.py, whose outputs are keyed by epsilon so a budget sweep
# cannot overwrite an earlier arm. --epsilon selects which arm to score and is
# ignored by the Synthcity methods, which have exactly one.
SMARTNOISE_METHODS = {"aim", "dpctgan"}
SMARTNOISE_DIR = REPO_ROOT / "synthetic_data" / "smartnoise"
DEFAULT_EPSILON = 1.0
RESULTS_DIR = REPO_ROOT / "results"
EVAL_DIR = REPO_ROOT / "evaluation"

PER_RUN_CSV = RESULTS_DIR / "utility_per_run.csv"
SUMMARY_CSV = RESULTS_DIR / "utility_summary.csv"
COST_CSV = RESULTS_DIR / "computational_cost.csv"   # shared with sdg/generate_runs.py

EXPECTED_TRAIN_N = 21_523
EXPECTED_TEST_N = 5_381
UTILITY_MODELS = ["xgboost", "logistic_regression"]

RESULTS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(EVAL_DIR / "eval_utility_log.txt"), logging.StreamHandler()],
)
log = logging.getLogger("eval_utility")


# ---------------------------------------------------------------- data ----

def load_and_split(seed: int = DATA_SPLIT_SEED):
    """Load the 80/20 split, and verify it really is the `seed` split.

    The split is materialised once by data_preprocessing.ipynb into
    data/adult_{train,test}.csv -- generators were fitted on those exact files
    -- so this reads them rather than re-splitting. What it does re-derive is
    the partition from data/adult_clean.csv, to confirm the files on disk are
    the `seed` split and not a stale artefact of some other seed.

    This check is heavier than the row-count asserts in sdg/generate_runs.py and
    eval_fidelity.py, and it lives here on purpose: this is the script whose
    correctness depends on the test set being genuinely held out. A silently
    re-split test set would reintroduce exactly the leak this file exists to fix.
    """
    train = pd.read_csv(DATA_DIR / "adult_train.csv")
    test = pd.read_csv(DATA_DIR / "adult_test.csv")

    from sklearn.model_selection import train_test_split
    clean = pd.read_csv(DATA_DIR / "adult_clean.csv")
    exp_train, exp_test = train_test_split(
        clean, test_size=0.2, random_state=seed, stratify=clean[TARGET_COL]
    )
    if not (exp_train.reset_index(drop=True).equals(train)
            and exp_test.reset_index(drop=True).equals(test)):
        raise RuntimeError(
            f"data/adult_train.csv + adult_test.csv are NOT the seed={seed} split of "
            f"adult_clean.csv. Re-run data_preprocessing.ipynb, and note that doing so "
            f"invalidates all synthetic data and TAPAS caches."
        )

    assert len(train) == EXPECTED_TRAIN_N, f"train is {len(train)}, expected {EXPECTED_TRAIN_N}"
    assert len(test) == EXPECTED_TEST_N, f"test is {len(test)}, expected {EXPECTED_TEST_N}"

    for df in (train, test):
        df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)

    log.info(f"Split seed {seed} verified. Train {train.shape}, held-out test {test.shape} "
             f"-- TSTR and TRTR both score on these {len(test)} records.")
    return train, test


def eps_slug(eps: float) -> str:
    """Must match sdg/generate_smartnoise.eps_slug -- same %g, same directories."""
    return f"eps{eps:g}"


def run_path(method: str, seed: int, epsilon: float) -> Path:
    """Where this run's CSV lives, per the method's generator script."""
    if method in SMARTNOISE_METHODS:
        return SMARTNOISE_DIR / method / eps_slug(epsilon) / f"seed{seed}.csv"
    return RUNS_DIR / f"{method}_seed{seed}.csv"


def load_run(method: str, seed: int, epsilon: float = DEFAULT_EPSILON):
    """The synthetic CSV for one run, or None if it hasn't been generated."""
    path = run_path(method, seed, epsilon)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    return df


# ------------------------------------------------------- feature encoding --

def build_feature_spec(*real_frames):
    """Fixed feature layout, fitted on the REAL data only.

    One shared layout for real and synthetic alike, so a category maps to the
    same column everywhere. Fitted on train + test (never on synthetic) so the
    layout does not shift between methods or runs -- a per-method layout would
    make TSTR scores incomparable across methods.

    One-hot rather than LabelEncoder: an ordinal code for `occupation` would
    invent an ordering that logistic regression reads as real. XGBoost tolerates
    either; LR does not, and both share this matrix.
    """
    categories = {
        c: sorted(set().union(*(set(f[c].astype(str).unique()) for f in real_frames)))
        for c in CATEGORICAL_FEATURES
    }
    columns = CONTINUOUS_COLS + [
        f"{c}={v}" for c in CATEGORICAL_FEATURES for v in categories[c]
    ]
    return {"categories": categories, "columns": columns}


def prepare_features(data: pd.DataFrame, spec: dict, label: str = ""):
    """Split off the target and encode features onto the shared layout.

    Categories absent from `spec` (synthetic data can invent them in principle;
    the Synthcity plugins sample from the training schema so in practice they do
    not) are dropped with a warning -- such a row simply gets zeros across that
    column group, which is the honest encoding of "value the real data never
    had". Missing categories are back-filled with zeros by the reindex.
    """
    y = (data[TARGET_COL].astype(str).str.strip() == POSITIVE_CLASS).astype(int)

    X = data[CONTINUOUS_COLS].astype(float).copy()
    for col in CATEGORICAL_FEATURES:
        values = data[col].astype(str)
        unseen = set(values.unique()) - set(spec["categories"][col])
        if unseen:
            log.warning(f"  {label}: {col} has categories absent from the real data, "
                        f"encoded as all-zero: {sorted(unseen)}")
        dummies = pd.get_dummies(values, prefix=col, prefix_sep="=")
        X = pd.concat([X, dummies], axis=1)

    X = X.reindex(columns=spec["columns"], fill_value=0).astype(float)
    return X, y


# ------------------------------------------------------------- scoring ----

def _fit_score(X_train, y_train, X_test, y_test, seed: int):
    """AUC on (X_test, y_test) for XGBoost and logistic regression."""
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if y_train.nunique() < 2:
        log.warning("  training set has a single class -- AUC undefined, recording NaN")
        return {m: float("nan") for m in UTILITY_MODELS}

    # `use_label_encoder` was removed in XGBoost 2.0 (this env runs 2.1.4);
    # passing it now only emits an "unused parameter" warning, so it is omitted.
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=seed)
    xgb_clf.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])

    # StandardScaler is not decoration: capital_gain spans 0-99,999 while the
    # one-hot columns are 0/1, and unscaled LR does not converge in 1000
    # iterations. The scaler is fitted inside the pipeline on the classifier's
    # OWN training data, so no information crosses from test to train.
    lr_clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )
    lr_clf.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr_clf.predict_proba(X_test)[:, 1])

    return {"xgboost": float(xgb_auc), "logistic_regression": float(lr_auc)}


def compute_tstr(synthetic_data, test_data, spec, seed: int):
    """Train on Synthetic, Test on Real.

    Classifiers are trained on synthetic_data and evaluated on test_data -- the
    5,381-record held-out split that no generator ever saw. Returns AUC for both
    XGBoost and logistic regression.
    """
    X_syn, y_syn = prepare_features(synthetic_data, spec, label="synthetic")
    X_test, y_test = prepare_features(test_data, spec, label="test")
    return _fit_score(X_syn, y_syn, X_test, y_test, seed)


def compute_trtr(train_data, test_data, spec, seed: int):
    """Train on Real, Test on Real.

    Trained on the full train_data (21,523 records -- not a CV fold, not a
    subsample), evaluated on the same 5,381-record test set TSTR uses, so the
    two are directly comparable. Returns AUC for both models.
    """
    X_train, y_train = prepare_features(train_data, spec, label="train")
    X_test, y_test = prepare_features(test_data, spec, label="test")
    return _fit_score(X_train, y_train, X_test, y_test, seed)


def compute_retention(tstr_aucs, trtr_aucs):
    """AUC retention = TSTR / TRTR, both on the same test set."""
    return {model: tstr_aucs[model] / trtr_aucs[model] for model in tstr_aucs}


def record_cost(row: dict) -> None:
    """Upsert one (method, seed, stage) row into results/computational_cost.csv.

    Same file sdg/generate_runs.py writes, keyed on stage -- so one table carries
    generation, fidelity and utility cost side by side. This replaces the old
    evaluation_LEGACY/eval_overhead_LEGACY.csv, which timed whole script invocations rather
    than per-method work and only covered the superseded eval scripts.
    """
    df = pd.DataFrame([row])
    if COST_CSV.exists():
        existing = pd.read_csv(COST_CSV)
        if {"method", "seed", "stage"}.issubset(existing.columns):
            mask = ~((existing["method"] == row["method"])
                     & (existing["seed"] == row["seed"])
                     & (existing["stage"] == row["stage"]))
            existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(COST_CSV, index=False)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean and std of every metric across runs."""
    metric_cols = [c for c in per_run.columns if c not in ("method", "run_idx", "seed")]
    rows = []
    for method, g in per_run.groupby("method", sort=False):
        row = {"method": method, "n_runs": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="*", default=None,
                        help=f"methods to score (default: all of {ALL_METHODS})")
    parser.add_argument("--runs", type=int, default=NUM_RUNS,
                        help=f"how many of the {NUM_RUNS} run seeds to score (default: all)")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"which epsilon arm of the SmartNoise methods "
                             f"({sorted(SMARTNOISE_METHODS)}) to score "
                             f"(default: {DEFAULT_EPSILON}); ignored for the rest")
    args = parser.parse_args()

    methods = args.methods or ALL_METHODS
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        parser.error(f"unknown method(s): {sorted(unknown)}. Known: {ALL_METHODS}")
    run_seeds = RUN_SEEDS[:args.runs]

    log.info(f"=== eval_utility: methods={methods}, seeds={run_seeds}, "
             f"eps={args.epsilon:g} (SmartNoise methods only) ===")
    train_df, test_df = load_and_split()
    spec = build_feature_spec(train_df, test_df)
    log.info(f"Feature layout: {len(spec['columns'])} columns "
             f"({len(CONTINUOUS_COLS)} continuous + one-hot categoricals), "
             f"fitted on real data only")

    # TRTR baseline: recomputed per run seed (only the classifier seed varies --
    # the data is fixed), so retention divides matched-seed numerators and
    # denominators. Expect trtr_*_std == 0: on fixed data with fixed
    # hyperparameters both classifiers are deterministic (LR's objective is
    # convex; XGBoost's default `exact` tree method does no subsampling), so
    # random_state has nothing to move. That is a property of the models, not a
    # seeding bug -- all of the across-run spread in retention comes from the
    # generator, which is what we want to measure.
    trtr_by_seed = {}
    for seed in run_seeds:
        set_all_seeds(seed)
        trtr_by_seed[seed] = compute_trtr(train_df, test_df, spec, seed)
        log.info(f"TRTR (real -> real, seed {seed}, trained on {len(train_df)} rows, "
                 f"scored on {len(test_df)}): " +
                 ", ".join(f"{m}={v:.4f}" for m, v in trtr_by_seed[seed].items()))

    rows, missing = [], []
    for method in methods:
        log.info(f"--- {method} ---")
        for run_idx, seed in enumerate(run_seeds):
            synthetic = load_run(method, seed, args.epsilon)
            if synthetic is None:
                missing.append(str(run_path(method, seed, args.epsilon)
                                   .relative_to(REPO_ROOT)))
                continue
            try:
                set_all_seeds(seed)
                tracemalloc.start()
                t0 = time.perf_counter()
                tstr = compute_tstr(synthetic, test_df, spec, seed)
                wall_clock_s = round(time.perf_counter() - t0, 2)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                trtr = trtr_by_seed[seed]
                retention = compute_retention(tstr, trtr)
                # TSTR only: the TRTR baseline is computed once per seed above and
                # is not per-method, so charging it to a method would double-count.
                record_cost({"method": method, "seed": seed, "stage": "utility",
                             "wall_clock_s": wall_clock_s,
                             "peak_memory_mb": round(peak_bytes / 1e6, 2),
                             "device": "cpu"})   # xgboost/sklearn, CPU only

                row = {"method": method, "run_idx": run_idx, "seed": seed}
                for model in UTILITY_MODELS:
                    row[f"tstr_{model}_auc"] = tstr[model]
                    row[f"trtr_{model}_auc"] = trtr[model]
                    row[f"retention_{model}"] = retention[model]
                rows.append(row)
                log.info(f"  seed {seed}: " + ", ".join(
                    f"{m} TSTR={tstr[m]:.4f} / TRTR={trtr[m]:.4f} = {retention[m]:.3f}"
                    for m in UTILITY_MODELS))
            except Exception:
                log.error(f"  {method} seed {seed} FAILED:\n{traceback.format_exc()}")

    if missing:
        log.warning(f"{len(missing)} run(s) not generated yet, skipped: "
                    f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if not rows:
        log.warning("No runs scored -- run sdg/generate_runs.py / "
                    "sdg/generate_smartnoise.py first.")
        return 1

    per_run = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(ALL_METHODS)}
    per_run = per_run.sort_values(
        ["method", "run_idx"], key=lambda s: s.map(order) if s.name == "method" else s
    ).reset_index(drop=True)
    per_run.to_csv(PER_RUN_CSV, index=False)

    summary = summarise(per_run)
    summary.to_csv(SUMMARY_CSV, index=False)
    log.info(f"Wrote {PER_RUN_CSV.relative_to(REPO_ROOT)} ({len(per_run)} rows) and "
             f"{SUMMARY_CSV.relative_to(REPO_ROOT)}")

    show = ["method", "n_runs", "tstr_xgboost_auc_mean", "tstr_xgboost_auc_std",
            "retention_xgboost_mean", "tstr_logistic_regression_auc_mean",
            "retention_logistic_regression_mean"]
    print("\n=== utility summary (mean over runs) ===")
    print(summary[[c for c in show if c in summary.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
