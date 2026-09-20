#!/usr/bin/env python3
"""[7] Does the eff-epsilon SPIKE reproduce on other target/alternate pairs?

THE QUESTION
    [6] answered a weaker question than the headline needs. It showed that three
    fresh target/alternate pairs all carry membership signal at eps = 1.0:

        pair          in-pool signal AUC        permuted-label baseline
        seed 43       0.874  (the audited one)  0.501
        seed 7        0.902 +- 0.040            0.512 +- 0.053
        seed 11       0.946 +- 0.027            0.462 +- 0.063
        seed 19       0.877 +- 0.050            0.511 +- 0.015

    But that AUC is NOT the published quantity. It comes from one RandomForest over
    per-dataset summary statistics, deliberately not TAPAS's implementation
    (eps_sweep_signal_scan.features). The number in the paper is eff-epsilon:
    eps_low_95 = 1.761 [1.761, 2.780] via Groundhog, produced by the five-attack
    battery and EffectiveEpsilonReport's Clopper-Pearson bound at 1000/2500.

    "Signal AUC is elevated on other pairs" and "eff-epsilon spikes on other pairs"
    are different claims, and only the second one supports the figure. A high AUC
    under a bespoke classifier does not have to survive the report's re-thresholding
    on a held-out 10%, and the bound is a worst-case over five attacks rather than
    a mean over folds. This script closes that gap: same five attacks, same counts,
    same report, new pair.

NO NEW CODE PATH -- THIS IS run_dpgan_eps_sweep.py
    Same reasoning as [2] and [6]: the point is that nothing about the pipeline
    changes, so this script adds no measurement logic of its own. It imports
    run_dpgan_eps_sweep.run_one_epsilon and calls it once per placebo seed. Same
    Synthcity-to-TAPAS adapter, same background (TAPAS_BG_SEED, unchanged), same
    five-attack battery under the same SCORE_ATTACK_SEED, same per-fit generator
    seeding, same distinctness guard, same 1000/2500 counts, same resumable
    per-attack JSON caches, same EffectiveEpsilonReport at the same confidence
    levels.

    The only thing that differs from the committed sweep is which pair
    common.select_random_target draws.

    NUM_TRAIN / NUM_TEST are deliberately NOT exposed as options. The whole value
    of this run is a number comparable to 1.761, and that number was measured at
    1000/2500; auditing a placebo at smaller counts would produce a figure that
    cannot be set beside it.

HOW THE PAIR IS REDIRECTED, AND WHY IT IS DONE HERE RATHER THAN IN THE AUDIT SCRIPT
    run_one_epsilon calls common.select_random_target(train_dataset, background_idx)
    with no seed, so it takes the default TAPAS_TARGET_SEED = 43. Two module
    attributes are rebound before each call:

      common.select_random_target  -> the same function with the placebo seed bound
      sweep.eps_slug               -> returns "eps1_target7" instead of "eps1"

    The second one is what keeps a placebo from ever landing on the published arm.
    run_one_epsilon derives EVERY path from that one label -- the cache directory,
    the per-attack JSON cache, the results directory, effeps_dpgan_<label>.csv and
    raw_scores_dpgan_<label>.csv -- so namespacing the label namespaces all of them
    at once. cache/dpgan/ (the archived counts-sweep pool at eps = 1.0) and
    cache/dpgan_eps1/ are both untouched and unreadable from here.

    run_dpgan_eps_sweep.py itself is not edited. It is committed code that produced
    published results, and a placebo is not a reason to add a parameter to it.

    seeds.py is NOT touched either. TAPAS_TARGET_SEED stays 43; these are extra
    draws alongside it, not a change to the audited pair.

WHY NOT REUSE [6]'s POOLS
    cache/dpgan_signal_eps1_target{seed}/ already holds 400 memoised fits per pair
    under the same background, target and plugin_kwargs, so in principle this audit
    could start 400 fits ahead. It does not, and should not: those pools were grown
    by a different script for a different measurement, and quietly seeding a
    published-style audit from them would make the arm's provenance depend on the
    order two unrelated runs happened to execute in. 400 of 3500 fits is 11% of the
    cost and not worth that.

COST -- READ THIS BEFORE LAUNCHING, THE OLD ESTIMATE WAS WRONG
    run_dpgan_eps_sweep.py's docstring projects 4.4 s/fit, i.e. ~4.3 h per arm. That
    rate was not what this workstation delivered in [6]. Its log is unambiguous:
    400 fits took 4314 s, 4377 s and ~4300 s for seeds 11, 19 and 7, which is

        ~10.8 s/fit  ->  3500 fits  ->  ~10.5 h of POOL per arm, plus attack time

    So one placebo arm is roughly a night, and three arms are roughly three nights.
    Budget accordingly: --max-hours (default 12) refuses to START an arm it projects
    cannot finish, rather than leaving one killed halfway. Everything is resumable,
    so a later invocation continues from the memoised fits and the finished attacks.

    If the 4.4 s/fit rate does come back -- it is the same config, so the difference
    is the machine, not the mechanism -- an arm lands nearer 4.3 h and more than one
    will fit. The rate is re-measured from this run's own fits (see PROGRESS) rather
    than assumed, so the first progress line tells you which world you are in about
    twenty minutes in.

PROGRESS, BECAUSE THE AUDIT IS OTHERWISE SILENT FOR HOURS
    run_one_epsilon grows both pools in two unbroken calls, and TAPAS's dataset
    generation takes a progress tracker that defaults to a silent one
    (attacker_knowledge.py ~431). Nothing is logged between "resuming at fit N" and
    "pools: ...", which at this fit rate is a ten-hour gap with no way to tell a
    working run from a wedged one. [6] did not have this problem: the signal scan
    checkpoints every 100 fits because it drives the pool itself.

    So this script wraps common.SynthcityGenerator.fit to log every --progress-every
    fits (default 100, about one line every 18 minutes) with the elapsed time, the
    measured s/fit and the projected pool time remaining. The wrapper calls the
    original fit and only logs; it changes nothing about what is fitted or seeded.

ORDER, AND WHY 19 GOES FIRST
    Default order is 19, 7, 11 -- deliberately not ascending. If only one arm
    finishes it should be the cleanest replication, and seed 19's signal AUC (0.877)
    is the one closest to the audited pair's (0.874), with a numeric-dominated
    profile of the same shape. Seed 7 is next. Seed 11 goes last precisely because
    it is the interesting one: its alternate carries native_country=France, absent
    from all 499 background records, so an elevated eff-epsilon there is confounded
    with the encoder-layout effect and is the weakest evidence for the plain claim
    "the spike is not about one record".

READING THE RESULT
    Reference: the audited pair scores eps_low_95 = 1.761 [1.761, 2.780] via
    Groundhog at 1000/2500 (results/sample_size_sweep/dpgan/1000_2500/), against
    eps_low_95 = 0 at eps = 0.1, 10 and 100.

        placebo eps_low_95 also >> 0   -> the spike is a property of DPGAN at
                                          eps ~ 1; 1.761 is one draw and the
                                          pair-to-pair spread can finally be quoted
        placebo eps_low_95 ~ 0         -> [6]'s AUC was picking up structure the
                                          five-attack battery cannot certify. The
                                          headline number is about record 12435 and
                                          must be reported with that stated plainly
        mixed across pairs             -> eff-epsilon needs an error bar over PAIRS,
                                          not just over simulations. That is itself
                                          the finding

    Compare eps_low_95 first and the winning attack second: if a placebo reaches a
    similar bound through a different attack, the leak is real but its shape is
    pair-dependent, which is a different sentence in the paper from a clean
    replication. One to three pairs is a first look, not a distribution -- do not
    quote a mean over them.

OUTPUT
    cache/dpgan_eps1_target{seed}/threat_model.pkl        per pair, never shared
    cache/dpgan_eps1_target{seed}/datasets/synthetic_*.csv.gz
    cache/dpgan_eps1_target{seed}/sweep_1000_2500/result_*.json   resumable
    results/extras/dpgan_spike_diagnosis/privacy_placebo/eps1_target{seed}/
        effeps_dpgan_eps1_target{seed}.csv, effective_epsilon_*.csv, meta.json
    results/extras/dpgan_spike_diagnosis/privacy_placebo/
        raw_scores_dpgan_eps1_target{seed}.csv
        full_audit_placebo_log.txt

Run from the repo root, env active (GPU workstation, screen/tmux -- this is hours):
  python benchmark_tapas/diagnostics/run_full_audit_placebo.py
  python benchmark_tapas/diagnostics/run_full_audit_placebo.py --target-seeds 19
  python benchmark_tapas/diagnostics/run_full_audit_placebo.py --max-hours 11.5
  python benchmark_tapas/diagnostics/run_full_audit_placebo.py --target-seeds 19 --dry-run
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
sys.path.insert(0, str(BENCHMARK_DIR / "audits"))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import common                                                      # noqa: E402
import run_dpgan_eps_sweep as sweep                                # noqa: E402
from config import RESULTS_DIR                                     # noqa: E402
from seeds import TAPAS_TARGET_SEED                                # noqa: E402

# The pairs [6] already measured a signal AUC for, so every arm here has a
# pre-registered expectation rather than being a fresh draw. Ordered most-comparable
# first -- see ORDER in the module docstring.
PLACEBO_SEEDS = [19, 7, 11]
PLACEBO_EPSILON = 1.0           # the spike itself; every other budget is flat for every pair so far

# [6]'s signal AUCs, carried here so the log states the prior before the arm runs.
SIGNAL_AUC = {43: 0.874, 7: 0.902, 11: 0.946, 19: 0.877}

# Measured in [6] on this workstation: ~4300-4380 s per 400 fits at eps = 1.0.
# run_dpgan_eps_sweep.py's docstring claims 4.4; both are logged and the projection
# is re-measured from this run's own pool timing.
MEASURED_S_PER_FIT = 10.8
OPTIMISTIC_S_PER_FIT = 4.4

DIAG_DIR = RESULTS_DIR / "extras" / "dpgan_spike_diagnosis"
PRIVACY_DIR = DIAG_DIR / "privacy_placebo"
PRIVACY_DIR.mkdir(parents=True, exist_ok=True)

# Redirect the imported module's output paths. run_one_epsilon reads both at call
# time, so reassigning them here is enough -- no function is copied or rewritten.
# CACHE_DIR is deliberately NOT redirected: the patched eps_slug below already
# namespaces the pools per pair, and keeping them beside the other arms means the
# same diagnostics run over all of them.
sweep.PRIVACY_DIR = PRIVACY_DIR
sweep.SWEEP_DIR = DIAG_DIR

# run_dpgan_eps_sweep configures logging at import, pointing at the headline sweep's
# log. Swap that handler so this run does not append to a committed artefact of a
# different experiment. Same manoeuvre as [2] and [6].
for h in list(sweep.log.handlers) + list(logging.getLogger().handlers):
    if isinstance(h, logging.FileHandler):
        logging.getLogger().removeHandler(h)
        sweep.log.removeHandler(h)
        h.close()
_fh = logging.FileHandler(PRIVACY_DIR / "full_audit_placebo_log.txt")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(_fh)
log = sweep.log

_TOTAL_FITS = sweep.NUM_TRAIN + sweep.NUM_TEST

# Per-arm fit counters for the progress wrapper. Reset at the start of each arm so
# the measured rate covers THIS invocation's fits -- on a resumed run the
# generator's own _fit_counter starts part-way through and would otherwise divide
# this run's elapsed time by the whole pool.
_PROGRESS = {"t0": None, "seen": 0, "every": 100}


def install_fit_progress() -> None:
    """Log every Nth fit, because nothing else does. See PROGRESS in the docstring.

    Wraps the original fit and logs after it returns; it does not touch the
    dataset, the plugin arguments or the per-fit seed. Installed once per process.
    """
    gen_cls = common.SynthcityGenerator
    if getattr(gen_cls, "_placebo_progress_installed", False):
        return
    original_fit = gen_cls.fit

    def fit_with_progress(self, dataset, **kwargs):
        out = original_fit(self, dataset, **kwargs)
        _PROGRESS["seen"] += 1
        seen = _PROGRESS["seen"]
        if _PROGRESS["t0"] is not None and seen % _PROGRESS["every"] == 0:
            elapsed = time.time() - _PROGRESS["t0"]
            rate = elapsed / seen
            done = getattr(self, "_fit_counter", seen)
            left_h = max(0, _TOTAL_FITS - done) * rate / 3600
            log.info(f"    fit {done}/{_TOTAL_FITS} ({elapsed / 60:.0f} min this run, "
                     f"{rate:.1f} s/fit, ~{left_h:.1f} h of pool left)")
        return out

    gen_cls.fit = fit_with_progress
    gen_cls._placebo_progress_installed = True


def bind_target_seed(target_seed: int) -> None:
    """Point run_one_epsilon at a different target/alternate pair.

    Both rebindings are looked up at call time (a module attribute and a module
    global respectively), so rebinding them here reaches run_one_epsilon without
    touching run_dpgan_eps_sweep.py. The eps_slug one is load-bearing for safety,
    not just for tidiness: it is the single point every cache and results path in
    run_one_epsilon is derived from, so it is what guarantees a placebo cannot write
    into the published arm.
    """
    original_select = getattr(common, "_select_random_target_original",
                              common.select_random_target)
    common._select_random_target_original = original_select

    def select_placebo(train_dataset, background_indices, seed=target_seed):
        return original_select(train_dataset, background_indices, seed=seed)

    common.select_random_target = select_placebo
    sweep.eps_slug = lambda eps: f"eps{eps:g}_target{target_seed}"


def projected_hours(s_per_fit: float) -> float:
    """Pool time only. Attack time is on top and is not predictable from fit count:
    it depends on the attack, not on the generator."""
    return _TOTAL_FITS * s_per_fit / 3600


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-seeds", nargs="+", type=int, default=PLACEBO_SEEDS,
                    help=f"placebo target/alternate pairs to audit, in order "
                         f"(default: {PLACEBO_SEEDS})")
    ap.add_argument("--epsilon", type=float, default=PLACEBO_EPSILON,
                    help=f"budget to audit (default: {PLACEBO_EPSILON}, the spike)")
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="wall-clock budget. An arm is not STARTED if its projected "
                         "pool time does not fit in what is left (default: 12)")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="record an arm whose pools failed the distinctness guard")
    ap.add_argument("--progress-every", type=int, default=100,
                    help="log a pool progress line every N fits (default: 100, about "
                         "one line every 18 min at the rate [6] measured)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print each arm's pair, paths and projection, "
                         "then exit without fitting anything")
    args = ap.parse_args()

    # A placebo that re-runs the published pair is not a placebo -- it would spend a
    # night reproducing 1.761 into a parallel cache. Same guard as [6].
    if TAPAS_TARGET_SEED in args.target_seeds:
        log.error(f"--target-seeds includes {TAPAS_TARGET_SEED}, the AUDITED pair. That is "
                  f"not a placebo: it would re-run the published arm. Refusing.")
        return 2

    import torch
    device_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

    log.info(f"=== [7] full TAPAS audit on placebo pairs: seeds={args.target_seeds} "
             f"at eps={args.epsilon:g}, {sweep.NUM_TRAIN}/{sweep.NUM_TEST}, "
             f"device={device_kwargs['device']} ===")
    log.info(f"    writing to {PRIVACY_DIR.relative_to(REPO_ROOT)}")
    log.info(f"    reference: the audited pair (seed {TAPAS_TARGET_SEED}) scored "
             f"eps_low_95 = 1.761 [1.761, 2.780] via Groundhog at these counts")
    log.info(f"    {_TOTAL_FITS} fits per arm. Projection "
             f"{projected_hours(MEASURED_S_PER_FIT):.1f} h at the "
             f"{MEASURED_S_PER_FIT} s/fit measured in [6] on this machine, "
             f"{projected_hours(OPTIMISTIC_S_PER_FIT):.1f} h at the {OPTIMISTIC_S_PER_FIT} s/fit "
             f"the sweep docstring claims. Pool only -- attacks are on top.")
    log.info(f"    budget {args.max_hours:.1f} h; arms that do not fit are skipped, not killed")

    if args.dry_run:
        for seed in args.target_seeds:
            bind_target_seed(seed)
            label = sweep.eps_slug(args.epsilon)
            log.info(f"    [dry-run] seed {seed}: [6] signal AUC "
                     f"{SIGNAL_AUC.get(seed, float('nan')):.3f} -> "
                     f"cache/dpgan_{label}/, "
                     f"{(PRIVACY_DIR / label).relative_to(REPO_ROOT)}/"
                     f"effeps_dpgan_{label}.csv")
        log.info("    [dry-run] nothing fitted.")
        return 0

    t0 = time.time()
    failed, skipped = [], []
    for seed in args.target_seeds:
        elapsed_h = (time.time() - t0) / 3600
        remaining_h = args.max_hours - elapsed_h
        need_h = projected_hours(MEASURED_S_PER_FIT)
        if remaining_h < need_h:
            log.warning(f"seed {seed}: SKIPPED -- {remaining_h:.1f} h left of the "
                        f"{args.max_hours:.1f} h budget, arm projects {need_h:.1f} h of pool. "
                        f"Re-run later; memoised fits and finished attacks are reused.")
            skipped.append(seed)
            continue

        bind_target_seed(seed)
        _PROGRESS.update(t0=time.time(), seen=0, every=args.progress_every)
        install_fit_progress()
        log.info(f"--- placebo pair seed={seed} "
                 f"([6] signal AUC {SIGNAL_AUC.get(seed, float('nan')):.3f}), "
                 f"{remaining_h:.1f} h of budget left ---")
        try:
            sweep.run_one_epsilon(args.epsilon, device_kwargs, args.allow_degenerate)
        except sweep.DegeneratePool as exc:
            log.error(f"DISTINCTNESS GUARD FAILED for seed={seed}:\n{exc}")
            failed.append((seed, "degenerate pool"))
        except Exception:
            log.error(f"seed={seed} FAILED:\n{traceback.format_exc()}")
            failed.append((seed, "exception"))

    log.info(f"=== done in {(time.time() - t0) / 3600:.2f} h ===")
    if failed:
        log.warning("Incomplete arms: " +
                    ", ".join(f"seed={s} ({why})" for s, why in failed))
    if skipped:
        log.warning(f"Not started (out of budget): {skipped}. Re-run to continue.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
