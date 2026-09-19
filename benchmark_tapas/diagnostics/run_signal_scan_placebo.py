#!/usr/bin/env python3
"""[6] Is the eps=1.0 spike a property of DPGAN, or of ONE target/alternate pair?

THE QUESTION
    Every DPGAN privacy number in this repo -- the 1.761 eff-epsilon, the 0.874
    membership signal, the whole spike -- is measured against a single
    target/alternate pair, fixed by TAPAS_TARGET_SEED = 43 and profiled in [4].
    Nothing so far distinguishes:

        "DPGAN at eps~1 leaks membership"            (a claim about the mechanism)
        "DPGAN at eps~1 leaks THIS record"           (a claim about one draw)

    Those are different papers. [5] sharpens the stakes: the signal concentrates in
    education_num and age, which are exactly the two numeric columns where this pair
    differs, while the three columns identical in both records carry nothing. That
    is either the mechanism following whatever the pair differs in -- in which case a
    new pair moves the signal to ITS differing columns and the spike survives -- or
    an artefact of this particular draw.

    Re-drawing the pair is the only thing that separates them.

NO NEW CODE PATH -- THIS IS eps_sweep_signal_scan.py
    Same reasoning as [2]: the point is that nothing about the pipeline changes, so
    this script adds no logic of its own. It imports eps_sweep_signal_scan.run_one
    and calls it once per seed. Same background (BACKGROUND_SEED, unchanged -- only
    the pair is redrawn), same generator and per-fit seeding, same 500 records per
    simulation, same feature set, same RandomForest under SCORE_ATTACK_SEED, same
    N datasets. The only thing that differs is the seed passed to
    common.select_random_target.

    The --target-seed plumbing lives in eps_sweep_signal_scan.py so that both that
    script and this one reach it; this script exists to run the SET of seeds, log
    the comparison, and keep the placebo output away from the committed scan.

WHY THE SIGNAL SCAN AND NOT A FULL AUDIT
    A full eff-epsilon arm is 3,500 fits and five attacks, ~4.3 h per pair. The
    question here is narrower -- does the generated data carry membership signal at
    all -- and [0] established that N=400 separates 0.51 from 0.89 comfortably. So
    this is 400 fits per pair, ~30-55 min each, and three pairs fit in an evening.

    If the placebo pairs come back elevated, the eff-epsilon claim is about the
    mechanism and the existing 1.761 stands as one draw from a distribution. If they
    come back at baseline, the headline number is a property of record 12435 and the
    sweep should be reported with that stated plainly.

NOTHING IS OVERWRITTEN
    Each seed gets its own cache (cache/dpgan_signal_eps1_target{seed}/) and its own
    results file (signal_scan_target{seed}.csv). The published pair keeps
    signal_scan.csv and cache/dpgan_signal_eps1/ untouched -- eps_sweep_signal_scan.
    scan_csv() routes on the seed, so a placebo run cannot land in the committed
    scan even if it is invoked with the default seed by mistake.

    seeds.py is NOT touched. TAPAS_TARGET_SEED stays 43; these are extra draws
    alongside it, not a change to the audited pair.

READING THE RESULT
    Reference: the audited pair scores signal_auc = 0.874 at eps=1.0, n=400
    (results/extras/dpgan_spike_diagnosis/signal_scan.csv), against a permuted-label
    baseline of 0.501.

        all placebo pairs elevated (~0.8+)  -> the spike is a property of DPGAN at
                                               eps~1; report 1.761 as one draw and
                                               say the pair-to-pair spread is unknown
        all at baseline (~0.5)              -> the spike is a property of this pair;
                                               the headline number must be qualified
        mixed                               -> it depends on the record. That is
                                               itself the finding, and the
                                               eff-epsilon figure needs an error bar
                                               over pairs, not just over simulations

    Three seeds is a first look, not a distribution. Do not quote a mean over three.

COST
    400 fits per seed at the 4.4-8 s/fit DPGAN manages on a 500-row background, so
    ~30-55 min per seed and ~1.5-3 h for the default three. Resumable: an interrupted
    run reuses the memoised fits in that seed's cache.

OUTPUT
    cache/dpgan_signal_eps{eps}_target{seed}/threat_model.pkl
    results/extras/dpgan_spike_diagnosis/signal_scan_target{seed}.csv
    results/extras/dpgan_spike_diagnosis/signal_scan_placebo_log.txt

Run from the repo root, env active (GPU workstation):
  python benchmark_tapas/diagnostics/run_signal_scan_placebo.py
  python benchmark_tapas/diagnostics/run_signal_scan_placebo.py --target-seeds 7
  python benchmark_tapas/diagnostics/run_signal_scan_placebo.py --epsilons 1.0 10.0 --n 200
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR / "diagnostics"))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import eps_sweep_signal_scan as scan                              # noqa: E402
from seeds import TAPAS_TARGET_SEED                               # noqa: E402

# Arbitrary and fixed. Any three seeds that are not TAPAS_TARGET_SEED would do; these
# are recorded here so the run is reproducible rather than chosen per invocation.
PLACEBO_SEEDS = [7, 11, 19]
PLACEBO_EPSILONS = [1.0]        # the spike itself; other arms are flat for every pair so far

# eps_sweep_signal_scan configures logging at import, pointing at the committed
# signal_scan_log.txt. Swap that handler so a placebo run does not append to the
# published scan's log. Same manoeuvre as [2].
for h in list(scan.log.handlers) + list(logging.getLogger().handlers):
    if isinstance(h, logging.FileHandler):
        logging.getLogger().removeHandler(h)
        scan.log.removeHandler(h)
        h.close()
_fh = logging.FileHandler(scan.SWEEP_DIR / "signal_scan_placebo_log.txt")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(_fh)
log = scan.log


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-seeds", nargs="+", type=int, default=PLACEBO_SEEDS,
                    help=f"pairs to redraw (default: {PLACEBO_SEEDS})")
    ap.add_argument("--epsilons", nargs="+", type=float, default=PLACEBO_EPSILONS,
                    help=f"budgets to scan (default: {PLACEBO_EPSILONS})")
    ap.add_argument("--n", type=int, default=scan.N_DATASETS,
                    help=f"datasets per (eps, seed) (default: {scan.N_DATASETS})")
    args = ap.parse_args()

    if TAPAS_TARGET_SEED in args.target_seeds:
        log.error(f"--target-seeds includes {TAPAS_TARGET_SEED}, the AUDITED pair. That is "
                  f"not a placebo: it would re-run the published arm. Refusing.")
        return 1

    import torch
    device_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    log.info(f"=== placebo pairs: target_seeds={args.target_seeds}, eps={args.epsilons}, "
             f"n={args.n}, device={device_kwargs['device']} ===")
    log.info(f"    audited pair (seed {TAPAS_TARGET_SEED}) scored signal_auc = 0.874 at "
             f"eps=1.0, n=400, permuted baseline 0.501")
    log.info(f"    projected ~{0.75 * len(args.target_seeds) * len(args.epsilons):.1f}-"
             f"{1.0 * len(args.target_seeds) * len(args.epsilons):.1f} h total")

    t0, failed = time.time(), []
    for seed in args.target_seeds:
        for eps in args.epsilons:
            try:
                row = scan.run_one(eps, args.n, device_kwargs, seed)
                scan.upsert(row, scan.scan_csv(seed))
            except Exception:
                log.error(f"target_seed={seed} eps={eps:g} FAILED:\n{traceback.format_exc()}")
                failed.append((seed, eps))

    log.info(f"=== done in {(time.time() - t0) / 3600:.2f} h ===")

    summary = []
    for seed in args.target_seeds:
        path = scan.scan_csv(seed)
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "target_seed", seed)
            summary.append(df)
    if summary:
        out = pd.concat(summary, ignore_index=True)
        print("\n=== membership signal, placebo pairs vs the audited pair ===")
        print(out[["target_seed", "formal_epsilon", "n_datasets",
                   "signal_auc", "signal_auc_cv_std", "permuted_auc"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\naudited pair (seed {TAPAS_TARGET_SEED}): signal_auc 0.874, permuted 0.501 "
              f"(n=400, signal_scan.csv)")
        print("\nAll placebo pairs elevated  -> the spike is a property of DPGAN at eps~1.")
        print("All at their permuted baseline -> it is a property of the audited record.")
        print("Mixed -> it depends on the record, and the eff-epsilon figure needs an")
        print("error bar over PAIRS, not only over simulations. Three seeds is a first")
        print("look, not a distribution -- do not quote a mean over three.")

    if failed:
        log.warning("Incomplete: " + ", ".join(f"seed={s} eps={e:g}" for s, e in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
