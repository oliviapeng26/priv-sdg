#!/usr/bin/env python3
"""Phase 0 of the formal-epsilon sweep: what noise does each eps actually buy?

Prints the DP-SGD noise multiplier sigma that Opacus derives for every eps in the
sweep, at BOTH dataset sizes the sweep uses -- the 500-row TAPAS audit world and
the 21,523-row utility world. Five seconds, no GPU, no generation. Run it before
committing a night to the sweep, so a surprise in the accounting shows up now
rather than in the results.

WHY SIGMA DIFFERS BETWEEN THE TWO WORLDS AT THE SAME EPS
    sigma is not a function of eps alone. Opacus solves for it from
    (eps, delta, sample_rate, epochs), and synthcity derives delta and
    sample_rate from the data it was handed:

        delta        = 1 / len(X)                      gan.py:590-591
        sample_rate  = 1 / len(data_loader)            opacus/privacy_engine.py:461
                     = 1 / ceil(len(X) / batch_size)
        epochs       = generator_n_iter                gan.py:602

    A 500-row audit dataset therefore gets a much larger sample_rate (fewer,
    proportionally bigger batches) and a much looser delta than the 21,523-row
    training set, and both push sigma up. At eps=1.0 that is sigma ~11.7 in the
    audit against ~2.7 on full data -- the audited model is a substantially
    noisier mechanism than the model we report utility for, at the same nominal
    budget. That is a limitation of auditing a 500-row world, not a bug, and it
    is why the sweep reports both columns rather than one "the" sigma.

WHAT len(X) ACTUALLY IS (the 80/20 wrinkle)
    gan.py reassigns X on line 583:

        X, X_val, cond, cond_val = self._train_test_split(X, cond)

    BEFORE reading len(X) for delta on line 591 and before building the loader on
    line 587. _train_test_split is a no-op only when patience_metric is None, and
    plugin_dpgan.py:157-158 installs a default WeightedMetrics when the caller
    passes none -- which we do. So the split is always live for dpgan, and both
    delta and sample_rate are computed on the 80% training portion, not on the
    full frame:

        audit    n=500    -> X is 400    -> delta = 2.50e-3, sample_rate = 1/2
        utility  n=21,523 -> X is 17,218 -> delta = 5.81e-5, sample_rate = 1/87

    This script reports both conventions side by side. The `_post_split` columns
    are what Opacus is actually handed; the `_nominal` columns are the 1/n figure
    the README quotes for dpgan (4.65e-5). They differ by 25%, which moves sigma
    by ~1-9% -- small, but the post-split value is the true one and the sweep
    reports that.

NEITHER DELTA IS PINNED, DELIBERATELY
    synthcity picks delta silently and it is left alone: pinning it would make
    this sweep a different mechanism from every dpgan number already in the repo.
    The consequence is that delta MOVES WITH n across this sweep's two worlds, so
    the audit and the utility arm are not the same (eps, delta) pair. Documented,
    not fixed.

POISSON SAMPLING IS OFF
    gan.py:607 passes poisson_sampling=False while the accountant that produced
    sigma assumes Poisson subsampling (opacus warns about exactly this). The
    reported sigma is therefore calibrated for a sampling scheme the training
    loop does not use, so the formal guarantee is an approximation rather than a
    proof. Left as-is: fixing it would change the mechanism under audit. Recorded
    here so the write-up can state it.

Outputs:
    results/eps_sweep/noise_multipliers.csv   read by eps_sweep_aggregate.py

Run from the repo root, venv active:
  python benchmark_tapas/scripts/eps_sweep_sigma_check.py
  python benchmark_tapas/scripts/eps_sweep_sigma_check.py --epsilons 0.05 0.5 5
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# benchmark_tapas/, found by walking up to config.py rather than counting parents --
# these scripts live in a subfolder now and may move again.
BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import RESULTS_DIR, NUM_SYNTHETIC, DPGAN_N_ITER    # noqa: E402

# Every eps the sweep reports, including the 1.0 arm reused from counts_sweep.
SWEEP_EPSILONS = [0.1, 1.0, 10.0, 100.0]

# The two worlds sigma is asked for. NUM_SYNTHETIC (500) is the TAPAS audit's
# D+/D- size (BACKGROUND_SIZE 499 + one target); 21,523 is the real training split.
AUDIT_N = NUM_SYNTHETIC
UTILITY_N = 21_523

# synthcity's dpgan defaults, plugin_dpgan.py:125 and config.DPGAN_N_ITER.
BATCH_SIZE = 200
VAL_FRACTION = 0.2          # gan.py:562, `split = int(len(total) * 0.8)`

SWEEP_DIR = RESULTS_DIR / "eps_sweep"


def noise_multiplier(eps: float, n_rows: int, batch_size: int = BATCH_SIZE,
                     epochs: int = DPGAN_N_ITER, post_split: bool = True) -> dict:
    """Mirror synthcity + Opacus exactly for one (eps, n) pair.

    post_split=True reproduces what the code really does (delta and sample_rate
    from the 80% training portion); post_split=False reproduces the nominal 1/n
    reading the README quotes. Both are reported so the gap is visible.
    """
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier

    n = int(n_rows * (1 - VAL_FRACTION)) if post_split else n_rows
    delta = 1.0 / n                                     # gan.py:591
    sample_rate = 1.0 / math.ceil(n / batch_size)       # privacy_engine.py:461
    mechanism = PrivacyEngine().accountant.mechanism()  # "prv" on opacus 1.4.1
    sigma = get_noise_multiplier(
        target_epsilon=eps, target_delta=delta,
        sample_rate=sample_rate, epochs=epochs, accountant=mechanism,
    )
    return {"delta": delta, "sample_rate": sample_rate, "n_effective": n,
            "sigma": float(sigma), "accountant": mechanism}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SWEEP_EPSILONS,
                        help=f"budgets to report (default: {SWEEP_EPSILONS})")
    parser.add_argument("--n-iter", type=int, default=DPGAN_N_ITER,
                        help=f"epochs handed to the accountant (default: {DPGAN_N_ITER}, "
                             f"config.DPGAN_N_ITER)")
    args = parser.parse_args()

    rows = []
    for eps in args.epsilons:
        row = {"formal_epsilon": eps, "replacement_epsilon": 2 * eps,
               "n_iter": args.n_iter, "batch_size": BATCH_SIZE}
        for world, n_rows in (("audit", AUDIT_N), ("utility", UTILITY_N)):
            for tag, post in (("", True), ("_nominal", False)):
                r = noise_multiplier(eps, n_rows, epochs=args.n_iter, post_split=post)
                row[f"sigma_{world}{tag}"] = r["sigma"]
                row[f"delta_{world}{tag}"] = r["delta"]
                if not tag:
                    row[f"n_rows_{world}"] = n_rows
                    row[f"n_effective_{world}"] = r["n_effective"]
                    row[f"sample_rate_{world}"] = r["sample_rate"]
                    row["accountant"] = r["accountant"]
        rows.append(row)

    df = pd.DataFrame(rows)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out = SWEEP_DIR / "noise_multipliers.csv"
    df.to_csv(out, index=False)

    show = ["formal_epsilon", "replacement_epsilon",
            "sigma_audit", "delta_audit", "sigma_utility", "delta_utility",
            "sigma_audit_nominal", "sigma_utility_nominal"]
    print(f"\n=== DP-SGD noise multipliers  (n_iter={args.n_iter}, batch={BATCH_SIZE}, "
          f"accountant={rows[0]['accountant']}, poisson_sampling=False) ===")
    print(f"audit world:   n={AUDIT_N:,} rows -> {rows[0]['n_effective_audit']:,} after the 80/20 "
          f"split, sample_rate={rows[0]['sample_rate_audit']:.4f}")
    print(f"utility world: n={UTILITY_N:,} rows -> {rows[0]['n_effective_utility']:,} after the "
          f"80/20 split, sample_rate={rows[0]['sample_rate_utility']:.4f}")
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nsigma_* are post-split (what Opacus is actually handed); *_nominal use the "
          f"README's 1/n reading.\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
