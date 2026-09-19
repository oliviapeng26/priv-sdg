#!/usr/bin/env python3
"""[3] Does the eps=10/100 floor mean privacy, or just a worse generator?

THE QUESTION
    Components [0]-[2] establish that the spike is real, reproducible, and in the
    generated data rather than in the attacks or the bound. They do not say why the
    curve has an interior maximum. Two stories fit the four points equally well:

    (A) REAL INTERIOR MAXIMUM. A non-DP component's imprint of the target is always
        present; what varies with eps is how legibly it survives into the output.
        Buried under sigma=80 at eps=0.1, visible at sigma=11.7, washed out once the
        DP discriminator is good enough to drive the generator towards the bulk
        distribution at sigma=2 and 0.53.

    (B) NOT ABOUT MEMBERSHIP AT ALL. At low sigma the GAN trains properly and
        collapses onto the population mean. A one-record difference cannot survive
        that, so leakage vanishes for a reason that has nothing to do with privacy.

    Both predict the same eff-epsilon curve. They differ in what the eps=100 output
    should LOOK like: (B) requires it to be closer to the real data and steadier
    across fits than eps=1, because that tightness is the thing doing the hiding.

NO NEW FITS, NO GPU
    run_dpgan_eps_sweep.export_pools already wrote every generated dataset to
    cache/dpgan_eps{eps}/datasets/synthetic_train.csv.gz, so both quantities are
    measurable from disk in seconds. eps=1.0 comes from cache/dpgan/ (the counts
    sweep arm that eps_sweep_aggregate.py also reads); the other three from their
    own eps caches.

WHAT IS MEASURED
    fidelity   how far the pooled synthetic data sits from the real training
               population. Continuous columns: 1-D Wasserstein distance, computed in
               the [0,1] units the pools are stored in (scalers.csv carries the
               bounds). Categorical columns: total-variation distance between level
               frequencies. Both averaged over columns.
    stability  the standard deviation, ACROSS datasets, of each numeric column's
               mean, averaged over columns. This is between-fit spread: how much the
               output moves when only the seed changes.

    Deliberately crude. The question is a direction (is eps=100 tighter or looser
    than eps=1), not a fidelity benchmark -- evaluation/eval_fidelity.py is the
    place for the real thing.

RESULT WHEN THIS WAS RUN (2026-09-19, N_DATASETS=200)
    eps     cont_wasserstein  cat_tv_distance  between_fit_std
    0.1               0.1007           0.2702           0.1076
    1.0               0.1327           0.5951           0.1647
    10.0              0.1002           0.6399           0.1797
    100.0             0.2317           0.6682           0.3107

    eps=100 is the FARTHEST from the real data and the LEAST stable, which is the
    opposite of what (B) requires. Story (B) is not supported. Note this does not
    confirm (A) -- it removes a rival, nothing more. Note also that between-fit
    spread alone cannot explain the spike: eps=10 is as variable as eps=1 (0.180 vs
    0.165) while carrying no membership signal at all (see [5]).

CAVEATS
    One fixed background and target/alternate pair. The comparison is against the
    full 21.5k-row training set while each fit saw 500 rows, so the absolute
    distances are not meaningful -- only their ORDER across eps is.

OUTPUT
    results/extras/dpgan_spike_diagnosis/pool_fidelity.csv

Run from the repo root, env active (CPU, ~1 min):
  python benchmark_tapas/diagnostics/dpgan_pool_fidelity.py
  python benchmark_tapas/diagnostics/dpgan_pool_fidelity.py --n 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# benchmark_tapas/, found by walking up to config.py rather than counting parents --
# these scripts live in a subfolder now and may move again.
BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import (CACHE_DIR, RESULTS_DIR, TRAIN_CSV,            # noqa: E402
                    CONTINUOUS_COLS, CATEGORICAL_COLS)

# eps -> cache folder. eps=1.0 is the counts-sweep arm (cache/dpgan/), not an
# eps-sweep arm, for the same reason eps_sweep_aggregate.py reads it from there:
# re-running it would spend 4.3 h reproducing a number the repo already has.
POOLS = {0.1: "dpgan_eps0.1", 1.0: "dpgan", 10.0: "dpgan_eps10", 100.0: "dpgan_eps100"}
N_DATASETS = 200        # 200 x 500 rows = 100k per eps; the distances are stable well below this

OUT_DIR = RESULTS_DIR / "extras" / "dpgan_spike_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "pool_fidelity.csv"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_DATASETS,
                    help=f"datasets per eps to pool (default: {N_DATASETS})")
    args = ap.parse_args()

    real = pd.read_csv(TRAIN_CSV)
    rows, missing = [], []

    for eps, folder in POOLS.items():
        d = CACHE_DIR / folder / "datasets"
        try:
            # scalers.csv is written beside the pool by export_pools; the pools are
            # stored in TAPAS's min-max scaled representation, so the real data has
            # to be put into the same units before the distances mean anything.
            scalers = pd.read_csv(d / "scalers.csv").set_index("column")
            syn = pd.read_csv(d / "synthetic_train.csv.gz")
        except FileNotFoundError:
            missing.append(f"eps={eps:g} ({d.relative_to(REPO_ROOT)})")
            continue
        syn = syn[syn.dataset_idx < args.n]

        w = []
        for c in CONTINUOUS_COLS:
            lo, hi = scalers.loc[c, "min"], scalers.loc[c, "max"]
            w.append(wasserstein_distance((real[c] - lo) / (hi - lo), syn[c]))

        tv = []
        for c in CATEGORICAL_COLS:
            p = real[c].astype(str).value_counts(normalize=True)
            q = syn[c].astype(str).value_counts(normalize=True)
            idx = p.index.union(q.index)
            tv.append(0.5 * (p.reindex(idx, fill_value=0)
                             - q.reindex(idx, fill_value=0)).abs().sum())

        per_dataset_mean = syn.groupby("dataset_idx")[CONTINUOUS_COLS].mean()
        rows.append({"formal_epsilon": eps, "cache": folder,
                     "n_datasets": int(syn.dataset_idx.nunique()),
                     "cont_wasserstein": float(np.mean(w)),
                     "cat_tv_distance": float(np.mean(tv)),
                     "between_fit_std": float(per_dataset_mean.std().mean())})

    if not rows:
        print("No exported pools found. Run the eps sweep first, or copy the caches "
              "down from the workstation.")
        return 1

    out = pd.DataFrame(rows).sort_values("formal_epsilon")
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if missing:
        print("\nno exported pool for: " + ", ".join(missing))
    print("\nRead it as: story (B) predicts eps=100 is the CLOSEST to real and the "
          "LEAST variable.\nIf it is instead the farthest and the most variable, (B) "
          "is not supported and the\neps=10/100 floor is not the generator collapsing "
          "onto the population.")
    print(f"\nwrote {OUT_CSV.relative_to(BENCHMARK_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
