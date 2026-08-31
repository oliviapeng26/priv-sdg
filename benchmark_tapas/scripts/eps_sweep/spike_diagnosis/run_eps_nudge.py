#!/usr/bin/env python3
"""[2] DPGAN at eps = 0.999 and 1.001, same setup, to test whether 1.0 is special.

THE QUESTION
    Is the spike attached to the NUMBER 1.0 -- a default, a branch, a special case
    somewhere in synthcity or Opacus -- or to the region around it? Nudging eps by
    a thousandth changes the mechanism by an amount no accountant would notice, so
    if the spike survives it belongs to the region, and if it vanishes something is
    keying on the literal value.

    Note what we already know: eps = 0.3 and 3.0 both showed clearly elevated
    membership signal (0.757 and 0.615 against baselines of 0.554 and 0.424). A
    special case on the value 1.0 would have left those at baseline. So the expected
    result here is confirmation, and the value of running it is that "we checked"
    beats "we reasoned about it".

NO NEW CODE PATH -- THIS IS run_eps_sweep.py
    The whole point is that nothing about the pipeline changes, so this script adds
    no logic of its own. It imports run_eps_sweep.run_one_epsilon and calls it. Same
    Synthcity-to-TAPAS adapter, same fixed background and target/alternate, same
    5-attack battery under the same SCORE_ATTACK_SEED, same per-fit generator
    seeding, same distinctness guard, same 1000/2500 counts, same resumable caches.
    The only thing that differs from the committed sweep is the number passed as eps.

    Two module-level constants are redirected before the call so the outputs land in
    the diagnosis folder rather than mixed into the headline sweep's results. That is
    a redirection of where files are written, not a change to what is computed.

CACHES
    cache/dpgan_eps0.999/ and cache/dpgan_eps1.001/, created fresh by the eps slug --
    no collision with the existing arms, and in particular no reuse of cache/dpgan/,
    which holds the archived eps=1.0 pool whose behaviour is the thing in question.

COST
    3500 fits per eps. DPGAN managed 4.2 s/fit at eps=1.0 on the workstation, so
    roughly 4 h per arm and ~8 h for both. Resumable: an interrupted run reuses the
    memoised fits.

OUTPUT
    results/eps_sweep/spike_diagnosis/privacy/eps{0.999,1.001}/
        effeps_dpgan_eps*.csv, effective_epsilon_*.csv, meta.json
    results/eps_sweep/spike_diagnosis/privacy/raw_scores_dpgan_eps*.csv
    results/eps_sweep/spike_diagnosis/privacy/eps_nudge_log.txt

Run from the repo root, env active:
  python benchmark_tapas/scripts/eps_sweep/spike_diagnosis/run_eps_nudge.py
  python benchmark_tapas/scripts/eps_sweep/spike_diagnosis/run_eps_nudge.py --epsilons 0.999
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
SWEEP_SCRIPTS = BENCHMARK_DIR / "scripts" / "eps_sweep"
sys.path.insert(0, str(SWEEP_SCRIPTS))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import run_eps_sweep as sweep                                    # noqa: E402
from config import RESULTS_DIR                                   # noqa: E402

# Either side of 1.0 by a thousandth: far too small to change the mechanism, large
# enough that no float comparison would treat it as 1.0.
NUDGE_EPSILONS = [0.999, 1.001]

DIAG_DIR = RESULTS_DIR / "eps_sweep" / "spike_diagnosis"
PRIVACY_DIR = DIAG_DIR / "privacy"
PRIVACY_DIR.mkdir(parents=True, exist_ok=True)

# Redirect the imported module's output paths. run_one_epsilon reads both of these at
# call time, so reassigning them here is enough -- no function is copied or rewritten.
# CACHE_DIR is deliberately NOT redirected: the eps slug already namespaces the pools
# (cache/dpgan_eps0.999/), and keeping them beside the other arms means the same
# diagnostics run over all of them.
sweep.PRIVACY_DIR = PRIVACY_DIR
sweep.SWEEP_DIR = DIAG_DIR

# run_eps_sweep configures logging at import, pointing at the headline sweep's log.
# Swap that handler so this run does not append to a committed artefact of a
# different experiment.
for h in list(sweep.log.handlers) + list(logging.getLogger().handlers):
    if isinstance(h, logging.FileHandler):
        logging.getLogger().removeHandler(h)
        sweep.log.removeHandler(h)
        h.close()
_fh = logging.FileHandler(PRIVACY_DIR / "eps_nudge_log.txt")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(_fh)
log = sweep.log


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epsilons", nargs="+", type=float, default=NUDGE_EPSILONS,
                    help=f"budgets to audit (default: {NUDGE_EPSILONS})")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="record an arm whose pools failed the distinctness guard")
    args = ap.parse_args()

    import torch
    device_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    log.info(f"=== eps nudge: {args.epsilons} at "
             f"{sweep.NUM_TRAIN}/{sweep.NUM_TEST}, device={device_kwargs['device']} ===")
    log.info(f"    writing to {PRIVACY_DIR.relative_to(REPO_ROOT)}")
    log.info(f"    reference: the committed eps=1.0 arm scored eps_low_95 = 1.761 "
             f"[1.761, 2.780] via Groundhog")
    log.info(f"    projected ~4 h per eps at 4.2 s/fit x "
             f"{sweep.NUM_TRAIN + sweep.NUM_TEST} fits")

    t0 = time.time()
    failed = []
    for eps in args.epsilons:
        try:
            sweep.run_one_epsilon(eps, device_kwargs, args.allow_degenerate)
        except sweep.DegeneratePool as exc:
            log.error(f"DISTINCTNESS GUARD FAILED for eps={eps:g}:\n{exc}")
            failed.append((eps, "degenerate pool"))
        except Exception:
            log.error(f"eps={eps:g} FAILED:\n{traceback.format_exc()}")
            failed.append((eps, "exception"))

    log.info(f"=== done in {(time.time() - t0) / 3600:.2f} h ===")
    if failed:
        log.warning("Incomplete: " + ", ".join(f"eps={e:g} ({why})" for e, why in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
