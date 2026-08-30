#!/usr/bin/env python3
"""Is DPGAN's membership signal a smooth function of eps, or is eps=1.0 a one-off?

THE QUESTION
    The formal-eps sweep produced a shape DP cannot: leakage is flat at the floor at
    eps = 0.1, 10 and 100, and elevated only at eps = 1.0. Measuring the membership
    signal directly in the generated pools (a classifier on per-dataset summary
    features, cross-validated within the training pool -- no TAPAS attack involved):

        eps = 0.1  ->  0.569
        eps = 1.0  ->  0.892      <- the odd one
        eps = 10   ->  0.519
        eps = 100  ->  0.512

    Two explanations fit those four numbers equally well, and they need different
    responses:

    (1) REAL INTERIOR MAXIMUM. Two effects pull opposite ways as sigma rises. The
        DP-SGD discriminator degrades, so the generator's output is increasingly
        driven by CTGAN's data transformer and conditional sampler -- neither of
        which is privatised, and both of which are fit directly on the 500 input
        rows including the target. But rising sigma also makes the output noisier,
        which buries any signal. Strongest imprint at sigma = 80, cleanest output at
        sigma = 0.53, best signal-to-noise somewhere between; sigma = 11.7 (eps = 1)
        sitting at the peak is consistent. If true, this is a finding: DPGAN's
        EMPIRICAL privacy is non-monotone in eps because a non-DP component takes
        over once the DP component is noisy enough to be useless.

    (2) NOT ABOUT EPS AT ALL. Something specific to that one run -- a package
        version, machine state, chance -- and eps = 1.0 is incidental. Then the row
        should be re-run and the sweep reported as flat, matching Chida et al.

    One point cannot separate these: an isolated elevated measurement is equally
    consistent with the top of a smooth curve and with a spike. This script adds
    neighbours at eps = 0.3 and eps = 3.0, which bracket 1.0 closely on a log scale.

        both neighbours elevated (~0.7-0.8)  -> smooth peak, hypothesis (1)
        both at ~0.51                        -> spike with flat ground, hypothesis (2)

    eps = 1.0 is in the default scan as well, and does double duty: it is also the
    replication. A fresh eps = 1.0 coming back at ~0.51 settles the question on its
    own, without needing the neighbours at all.

WHY NOT JUST RE-RUN THE FULL AUDIT
    Effective epsilon needs 3,500 fits, five attacks and a test pool. The question
    here is narrower -- does the generated data carry membership information at all
    -- and that is answerable from the training pool alone. Telling 0.51 from 0.89
    takes a few hundred datasets, not thousands. So this generates N datasets per
    eps (default 400) and stops.

WHAT IS HELD FIXED
    Everything the audit fixes: the same background (TAPAS_BG_SEED), the same
    target/alternate (TAPAS_TARGET_SEED), the same per-fit generator seeding
    (TAPAS_GENERATOR_SEED_BASE + i), the same n_iter and plugin defaults, the same
    500 records per simulation, the same SwapTargetedMIA construction. Only eps
    varies. The pools are therefore drawn exactly as run_eps_sweep.py draws them,
    which is what makes the new points comparable to the four already measured.

    Caches live in cache/dpgan_signal_eps{eps}/ and are never shared with the audit
    caches. In particular a fresh eps = 1.0 must NOT reuse cache/dpgan/ -- that is
    the archived counts-sweep pool whose reproducibility is the thing in question.

THE MEASUREMENT
    Per dataset: numeric column means and standard deviations, plus categorical level
    frequencies -- the naive Groundhog feature set in spirit. Then a RandomForest,
    5-fold stratified CV, scored by AUC on the D+/D- label.

    A label-permutation baseline is reported alongside. At N = 400 the null is not
    exactly 0.5, and the permutation score says what "no signal" actually looks like
    for this feature set at this sample size -- so the comparison is against a
    measured null rather than an assumed one.

OUTPUTS
    cache/dpgan_signal_eps{eps}/threat_model.pkl    pool, resumable
    results/eps_sweep/signal_scan.csv               one row per eps
    results/eps_sweep/signal_scan_log.txt

Run from the repo root, env active:
  python benchmark_tapas/scripts/eps_sweep_signal_scan.py
  python benchmark_tapas/scripts/eps_sweep_signal_scan.py --epsilons 1.0
  python benchmark_tapas/scripts/eps_sweep_signal_scan.py --n 200 --epsilons 0.3 3.0

Cost: N fits per eps at the 4.4-8 s/fit DPGAN manages on a 500-row background, so
~30-55 min per eps and ~1.5-2.5 h for the default three. Resumable -- an interrupted
run reuses the memoised fits.
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import tapas.threat_models as tm                                  # noqa: E402
import common                                                     # noqa: E402
from config import (CACHE_DIR, RESULTS_DIR, METHOD_CONFIG,        # noqa: E402
                    NUM_SYNTHETIC, CONTINUOUS_COLS)
from seeds import SCORE_ATTACK_SEED                               # noqa: E402

METHOD = "dpgan"
# 0.3 and 3.0 bracket 1.0 on a log scale; 1.0 itself is the replication.
SCAN_EPSILONS = [0.3, 1.0, 3.0]
N_DATASETS = 400            # enough to separate 0.51 from 0.89; not an audit
CHECKPOINT_EVERY = 100      # a crash costs at most this many fits
CV_FOLDS = 5

SWEEP_DIR = RESULTS_DIR / "eps_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
SCAN_CSV = SWEEP_DIR / "signal_scan.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SWEEP_DIR / "signal_scan_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("signal_scan")


def eps_slug(eps: float) -> str:
    """Matches run_eps_sweep.eps_slug: 0.3 -> 'eps0.3', 3.0 -> 'eps3'."""
    return f"eps{eps:g}"


# -- The measurement -----------------------------------------------------

def features(datasets, labels):
    """Per-dataset feature vector + D+/D- label.

    Numeric mean/std plus categorical level frequencies -- the naive Groundhog
    feature set in spirit, and deliberately NOT TAPAS's implementation: the point is
    to establish whether the signal is in the DATA, independently of the attack code
    the audit runs.
    """
    frames = []
    for i, d in enumerate(datasets):
        f = d.data.copy()
        f.insert(0, "dataset_idx", i)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)

    g = df.groupby("dataset_idx")
    num = [c for c in CONTINUOUS_COLS if c in df.columns]
    cat = [c for c in df.columns if c not in set(num) | {"dataset_idx"}]
    parts = [g[num].mean().add_suffix("_mean"), g[num].std().add_suffix("_std")]
    for c in cat:
        parts.append(pd.crosstab(df.dataset_idx, df[c], normalize="index").add_prefix(f"{c}="))
    X = pd.concat(parts, axis=1).fillna(0.0)
    y = np.array([int(bool(l)) for l in labels])
    return X, y


def signal_auc(X, y, seed: int = SCORE_ATTACK_SEED):
    """Cross-validated AUC for predicting D+ vs D-, and a permuted-label baseline.

    The permutation is the honest null: at a few hundred datasets, 0.5 is the
    expectation but not what a finite sample returns, so the baseline says what "no
    signal" measures as for this feature set at this size.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    def score(labels):
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
        s = cross_val_score(rf, X, labels, cv=cv, scoring="roc_auc")
        return float(s.mean()), float(s.std())

    auc, auc_sd = score(y)
    rng = np.random.default_rng(seed)
    null, null_sd = score(rng.permutation(y))
    return auc, auc_sd, null, null_sd


# -- Pool construction ---------------------------------------------------

def build_pool(epsilon: float, n_datasets: int, device_kwargs: dict):
    """Grow a training pool of n_datasets at this eps, checkpointing as it goes.

    Constructed exactly as run_eps_sweep.py does -- same threat model class, same
    knowledge classes, same generator, same seeds -- so the resulting datasets are
    the same draws the audit would have made.
    """
    cache_dir = CACHE_DIR / f"{METHOD}_signal_{eps_slug(epsilon)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "threat_model"

    plugin_kwargs = {**METHOD_CONFIG[METHOD]["plugin_kwargs"], **device_kwargs}
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)

    threat_model = common.build_or_load_threat_model(
        cache_dir=cache_dir, method=METHOD, background_dataset=background,
        target_record=target, alternate_record=alternate, description=description,
        epsilon=epsilon, plugin_kwargs=plugin_kwargs,
    )
    gen = threat_model.atk_know_gen.generator
    cached_eps = getattr(gen, "epsilon", None)
    if cached_eps is not None and not np.isclose(float(cached_eps), epsilon):
        raise RuntimeError(
            f"{eps_slug(epsilon)}: cached pool at {cache_dir} was built at "
            f"epsilon={cached_eps}, not {epsilon}. Delete that directory and re-run.")

    t0 = time.time()
    while True:
        have = len(threat_model._memory[True][0])
        if have >= n_datasets:
            break
        step = min(n_datasets, have + CHECKPOINT_EVERY)
        threat_model.generate_training_samples(step)
        threat_model.save(str(cache_path))
        log.info(f"    pool {len(threat_model._memory[True][0])}/{n_datasets} "
                 f"checkpointed ({time.time() - t0:.0f}s elapsed)")

    datasets, labels = threat_model._memory[True]
    return datasets[:n_datasets], labels[:n_datasets], round(time.time() - t0, 1)


# -- Per-eps driver ------------------------------------------------------

def run_one(epsilon: float, n_datasets: int, device_kwargs: dict) -> dict:
    log.info(f"=== eps={epsilon:g}: {n_datasets} datasets, "
             f"plugin_kwargs={METHOD_CONFIG[METHOD]['plugin_kwargs']} ===")
    datasets, labels, pool_s = build_pool(epsilon, n_datasets, device_kwargs)

    n_pos = int(sum(bool(l) for l in labels))
    X, y = features(datasets, labels)
    auc, auc_sd, null, null_sd = signal_auc(X, y)

    log.info(f"    in-pool membership signal: AUC {auc:.3f} +- {auc_sd:.3f}   "
             f"(permuted-label baseline {null:.3f} +- {null_sd:.3f})")
    log.info(f"    {n_pos} D+ / {len(labels) - n_pos} D-, {X.shape[1]} features, "
             f"pool {pool_s / 60:.1f} min")

    return {
        "method": METHOD, "formal_epsilon": epsilon, "n_datasets": len(datasets),
        "n_positive": n_pos, "n_features": int(X.shape[1]),
        "signal_auc": auc, "signal_auc_cv_std": auc_sd,
        "permuted_auc": null, "permuted_auc_cv_std": null_sd,
        "num_synthetic": NUM_SYNTHETIC, "cv_folds": CV_FOLDS,
        "pool_wall_clock_s": pool_s,
    }


def upsert(row: dict) -> None:
    """One row per (epsilon, n_datasets), so a re-run at a different N adds rather
    than silently replaces."""
    df = pd.DataFrame([row])
    if SCAN_CSV.exists():
        existing = pd.read_csv(SCAN_CSV)
        if {"formal_epsilon", "n_datasets"}.issubset(existing.columns):
            mask = ~((existing.formal_epsilon == row["formal_epsilon"])
                     & (existing.n_datasets == row["n_datasets"]))
            existing = existing[mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.sort_values("formal_epsilon").to_csv(SCAN_CSV, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SCAN_EPSILONS,
                        help=f"budgets to scan (default: {SCAN_EPSILONS})")
    parser.add_argument("--n", type=int, default=N_DATASETS,
                        help=f"datasets per eps (default: {N_DATASETS})")
    args = parser.parse_args()

    import torch
    device_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    log.info(f"=== signal scan: eps={args.epsilons}, n={args.n} per eps, "
             f"device={device_kwargs['device']} ===")
    log.info("    reference, from the audit pools already measured: "
             "eps=0.1 -> 0.569, eps=1.0 -> 0.892, eps=10 -> 0.519, eps=100 -> 0.512")

    failed = []
    for epsilon in args.epsilons:
        try:
            upsert(run_one(epsilon, args.n, device_kwargs))
        except Exception:
            log.error(f"eps={epsilon:g} FAILED:\n{traceback.format_exc()}")
            failed.append(epsilon)

    if SCAN_CSV.exists():
        out = pd.read_csv(SCAN_CSV)
        print("\n=== membership signal vs formal epsilon ===")
        print(out[["formal_epsilon", "n_datasets", "signal_auc",
                   "signal_auc_cv_std", "permuted_auc"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nRead it as: neighbours of eps=1.0 clearly above their permuted "
              "baseline => smooth interior peak, a real property of the mechanism. "
              "Neighbours at the baseline with only eps=1.0 elevated => not a "
              "function of eps; re-run that arm and report a flat sweep.")

    if failed:
        log.warning(f"Incomplete: {', '.join(f'eps={e:g}' for e in failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
