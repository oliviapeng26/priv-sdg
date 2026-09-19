#!/usr/bin/env python3
"""[5] WHERE in the synthetic data does the eps=1.0 membership signal live?

THE QUESTION
    [0] measured that the signal exists (AUC 0.87 at eps=1.0 against ~0.51 at 10 and
    100). It did not say what the classifier is looking at. That matters because the
    candidate explanations make different predictions about WHICH columns leak:

    - the TabularEncoder fits a BayesianGMM per CONTINUOUS column on the raw 500
      rows, outside DP (tabular_encoder.py:58-59, fitted at tabular_gan.py:184)
          -> predicts numeric columns
    - the ConditionalDatasetSampler stores per-category row lists and frequencies
      from the raw rows, outside DP (tabular_gan.py:204, samplers.py:231-240), and
      the generator's own extra penalty compares its fake categorical output against
      the REAL batch with no clipping or noise (gan.py:359-363, tabular_gan.py:212-249)
          -> predicts categorical columns
    - the target/alternate records simply differ in some columns and not others
          -> predicts exactly the differing columns, whatever their type

    None of these is tested here. This script localises the signal so that the next
    experiment can be aimed.

NO NEW FITS, NO GPU
    Reads the exported pools. Every dataset carries its D+/D- label in the
    ground_truth column (written by run_dpgan_eps_sweep.export_pools), so the whole
    analysis is a supervised problem over datasets-as-rows.

THREE VIEWS, BECAUSE ONE OF THEM IS MISLEADING ON ITS OWN
    A. per feature       each of the 76 summaries ranked alone, by |AUC - 0.5|, with
                         a permutation bar: the largest deviation any feature reaches
                         when the labels are shuffled, 95th percentile over N_PERM
                         shuffles. Without that bar, testing 76 features guarantees
                         a "winner".
    B. per column group  all levels of one categorical column, or all numeric means,
                         or all numeric stds -- scored ONLY-this and EXCEPT-this.
    C. blocks + numeric  numeric-as-a-block vs categorical-as-a-block, then each
                         numeric column (its mean and std together).

    A alone is actively misleading here, and that is why all three are kept: at
    eps=1.0 the best single feature reaches only AUC 0.598, which reads as "diffuse,
    nothing much anywhere", while the ten numeric features TOGETHER reach 0.872. The
    signal is a joint pattern that per-feature ranking cannot see.

    The classifier and CV are the same as [0] (RandomForest 200 trees, 5-fold
    stratified, SCORE_ATTACK_SEED) so the numbers are comparable to signal_scan.csv.
    Note signal_scan.csv used n=400 datasets per eps and the audit pools here hold
    1000, which is why eps=1.0 reads 0.892 here against 0.874 there.

RESULTS WHEN THIS WAS RUN (2026-09-19)
    eps=1.0 (1000 datasets, 76 features, all features AUC 0.892):
        blocks          numeric-only 0.872   categorical-only 0.712
        groups only/except   numeric means 0.785/0.814   numeric stds 0.739/0.847
                             marital_status 0.613/0.893, every other categorical
                             column 0.51-0.55 alone and ~0.89 when removed
        numeric columns only/except, against [4]'s profile of the pair:
             education_num (differs, 6 vs 9)   0.701 / 0.766   <- biggest drop
             age           (differs, 36 vs 47) 0.591 / 0.868
             capital_gain  (same, 0)           0.530 / 0.888
             capital_loss  (same, 0)           0.500 / 0.889
             hours_per_week(same, 40)          0.497 / 0.888
    eps=10: every block, group and column sits at 0.48-0.55 against a shuffled-label
        baseline of 0.528. No signal anywhere, consistent with TAPAS's eps_high_95
        of 0.0087 for the same arm.

    So: the signal is concentrated in the numeric block, strongest in the two numeric
    columns where the target and alternate actually differ, and the categorical block
    carries a real but weaker share (0.712, not 0.5).

CAVEATS
    One target/alternate pair. capital_gain and capital_loss are near-constant zero
    across the whole dataset and hours_per_week is identical in the two records, so
    "the columns that differ carry the signal" and "those three columns are
    uninformative for any pair" both fit -- [6] is the experiment that separates
    them. The shuffled-label baseline in view C is a single draw (0.463 at eps=1.0,
    0.528 at eps=10), so differences below ~0.05 are not readable.

OUTPUT
    results/extras/dpgan_spike_diagnosis/signal_anatomy_features.csv
    results/extras/dpgan_spike_diagnosis/signal_anatomy_groups.csv
    results/extras/dpgan_spike_diagnosis/signal_anatomy_blocks.csv

Run from the repo root, env active (CPU, ~5 min):
  python benchmark_tapas/diagnostics/dpgan_signal_anatomy.py
  python benchmark_tapas/diagnostics/dpgan_signal_anatomy.py --epsilons 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import CACHE_DIR, RESULTS_DIR, CONTINUOUS_COLS        # noqa: E402
from seeds import SCORE_ATTACK_SEED                               # noqa: E402

# eps -> cache folder; eps=1.0 is the counts-sweep arm, as in [3].
POOLS = {0.1: "dpgan_eps0.1", 1.0: "dpgan", 10.0: "dpgan_eps10", 100.0: "dpgan_eps100"}
DEFAULT_EPSILONS = [1.0, 10.0]      # the spike and a flat arm; 0.1/100 on request
CV_FOLDS = 5
TOP_FEATURES = 12
N_PERM = 50                         # label shuffles for view A's significance bar

OUT_DIR = RESULTS_DIR / "extras" / "dpgan_spike_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per synthetic dataset: numeric mean/std + categorical level shares.

    Copied from eps_sweep_signal_scan.features rather than imported, deliberately:
    importing that module runs its logging.basicConfig and starts appending to
    signal_scan_log.txt, a committed artefact of a different experiment. The feature
    set must stay identical to it or the AUCs stop being comparable.
    """
    g = df.groupby("dataset_idx")
    num = [c for c in CONTINUOUS_COLS if c in df.columns]
    cat = [c for c in df.columns if c not in set(num) | {"dataset_idx", "ground_truth"}]
    parts = [g[num].mean().add_suffix(" (mean)"), g[num].std().add_suffix(" (std)")]
    for c in cat:
        parts.append(pd.crosstab(df.dataset_idx, df[c], normalize="index").add_prefix(f"{c}="))
    return pd.concat(parts, axis=1).fillna(0.0)


def cv_auc(X: pd.DataFrame, y: np.ndarray) -> float:
    """Cross-validated AUC for D+ vs D-. Same estimator and folds as [0]."""
    rf = RandomForestClassifier(n_estimators=200, random_state=SCORE_ATTACK_SEED, n_jobs=-1)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SCORE_ATTACK_SEED)
    return float(cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean())


def univariate_auc(X: pd.DataFrame, member: np.ndarray) -> pd.Series:
    """AUC of every feature on its own, via the rank-sum identity.

    No model and no cross-validation: a single feature's AUC is a closed-form
    function of its ranks, which makes the 50 permutation replicates affordable.
    """
    r = X.rank()
    n1 = int(member.sum())
    n0 = len(member) - n1
    return (r[member].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def run_one(eps: float, folder: str) -> tuple:
    path = CACHE_DIR / folder / "datasets" / "synthetic_train.csv.gz"
    df = pd.read_csv(path)
    X = features(df)
    member = (df.groupby("dataset_idx").ground_truth.first().loc[X.index] == 1).values
    y = member.astype(int)

    all_auc = cv_auc(X, y)
    rng = np.random.default_rng(0)
    null_auc = cv_auc(X, rng.permutation(y))
    print(f"\n=== eps={eps:g}: {member.sum()} D+ / {(~member).sum()} D- datasets, "
          f"{X.shape[1]} features ===")
    print(f"    all features: AUC {all_auc:.3f}   (shuffled-label baseline {null_auc:.3f})")

    # -- View A: one feature at a time, against a permutation bar ---------
    auc1 = univariate_auc(X, member)
    bar = float(np.quantile([(univariate_auc(X, rng.permutation(member)) - 0.5).abs().max()
                             for _ in range(N_PERM)], 0.95))
    dev = (auc1 - 0.5).abs()
    feat = pd.DataFrame({
        "formal_epsilon": eps, "feature": auc1.index, "auc_alone": auc1.values,
        "higher_when": np.where(auc1.values > 0.5, "target IN", "alternate IN"),
        "mean_if_target_in": X.loc[member].mean().values,
        "mean_if_alternate_in": X.loc[~member].mean().values,
        "beyond_chance": (dev > bar).values, "chance_bar": bar,
    }).sort_values("auc_alone", key=lambda s: (s - 0.5).abs(), ascending=False)
    print(f"\n  [A] single features. Shuffling the labels, the best of all {X.shape[1]} "
          f"reaches |AUC-0.5| = {bar:.3f};\n      {int(feat.beyond_chance.sum())} real "
          f"features beat that. Top {TOP_FEATURES}:")
    print(feat.head(TOP_FEATURES)[["feature", "auc_alone", "higher_when",
                                   "mean_if_target_in", "mean_if_alternate_in"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -- View B: column groups, only-this and except-this -----------------
    groups: dict = {}
    for name in X.columns:
        key = (name.split("=")[0] if "=" in name
               else "numeric means" if "(mean)" in name else "numeric stds")
        groups.setdefault(key, []).append(name)
    grp = pd.DataFrame([
        {"formal_epsilon": eps, "column_group": key, "n_features": len(cols),
         "auc_only_this": cv_auc(X[cols], y),
         "auc_except_this": cv_auc(X[[c for c in X.columns if c not in cols]], y)}
        for key, cols in groups.items()
    ]).sort_values("auc_only_this", ascending=False)
    print(f"\n  [B] column groups (all features together = {all_auc:.3f}):")
    print(grp.drop(columns="formal_epsilon")
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -- View C: blocks, then each numeric column -------------------------
    numeric = [c for c in X.columns if c.endswith(" (mean)") or c.endswith(" (std)")]
    categorical = [c for c in X.columns if c not in numeric]
    blk = [{"formal_epsilon": eps, "block": "all features", "n_features": X.shape[1],
            "auc_only_this": all_auc, "auc_except_this": np.nan},
           {"formal_epsilon": eps, "block": "numeric (all)", "n_features": len(numeric),
            "auc_only_this": cv_auc(X[numeric], y), "auc_except_this": cv_auc(X[categorical], y)},
           {"formal_epsilon": eps, "block": "categorical (all)", "n_features": len(categorical),
            "auc_only_this": cv_auc(X[categorical], y), "auc_except_this": cv_auc(X[numeric], y)},
           {"formal_epsilon": eps, "block": "shuffled labels", "n_features": X.shape[1],
            "auc_only_this": null_auc, "auc_except_this": np.nan}]
    for col in CONTINUOUS_COLS:
        cols = [f"{col} (mean)", f"{col} (std)"]
        if not set(cols) <= set(X.columns):
            continue
        blk.append({"formal_epsilon": eps, "block": f"numeric: {col}", "n_features": 2,
                    "auc_only_this": cv_auc(X[cols], y),
                    "auc_except_this": cv_auc(X[[c for c in X.columns if c not in cols]], y)})
    blocks = pd.DataFrame(blk)
    print("\n  [C] blocks, then each numeric column (its mean and std together).")
    print("      Cross-check the numeric rows against [4]: does the signal sit in the")
    print("      columns where the target and alternate actually differ?")
    print(blocks.drop(columns="formal_epsilon")
                .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    return feat, grp, blocks


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS,
                    help=f"arms to analyse (default: {DEFAULT_EPSILONS}; "
                         f"available: {sorted(POOLS)})")
    args = ap.parse_args()

    feats, grps, blks = [], [], []
    for eps in args.epsilons:
        if eps not in POOLS:
            print(f"eps={eps:g}: no pool mapping, skipped"); continue
        if not (CACHE_DIR / POOLS[eps] / "datasets" / "synthetic_train.csv.gz").exists():
            print(f"eps={eps:g}: no exported pool under cache/{POOLS[eps]}/, skipped"); continue
        f, g, b = run_one(eps, POOLS[eps])
        feats.append(f); grps.append(g); blks.append(b)

    if not blks:
        print("\nNo pools analysed.")
        return 1

    for frames, name in ((feats, "features"), (grps, "groups"), (blks, "blocks")):
        path = OUT_DIR / f"signal_anatomy_{name}.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"wrote {path.relative_to(BENCHMARK_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
