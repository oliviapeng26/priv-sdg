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

COST -- THE RATE DEPENDS ON WHO ELSE IS ON THE GPU, SO MEASURE IT
    run_dpgan_eps_sweep.py's docstring projects 4.4 s/fit, i.e. ~4.3 h per arm. This
    workstation has delivered three different rates, and the difference is load:

        [6]   400 fits in ~4300 s                    ~10.8 s/fit   GPU shared with the AIM jobs
        [7]   first two fits, empty-ish GPU          ~3-4 s/fit
        [7]   fits 400-1100 of the first attempt     ~6.3 s/fit    another user's job on both cards

    Treat ~6.3 s/fit as the working figure: 3500 fits is ~6.1 h of POOL per arm, plus
    attack time, so a 12 h window fits ONE arm and not two. --max-hours (default 12)
    refuses to START an arm it projects cannot finish, rather than leaving one killed
    halfway. The rate is logged from this run's own fits (see PROGRESS) rather than
    assumed, and MEASURED_S_PER_FIT below is only what the start-of-arm guard uses.

CHECKPOINTING, BECAUSE THE FIRST ATTEMPT LOST TWO HOURS TO ONE OUT-OF-MEMORY ERROR
    run_one_epsilon grows both pools in two unbroken calls and saves the threat model
    ONCE, after both are done. TAPAS holds the fitted datasets in memory until then,
    so any crash mid-pool loses the whole pool -- the docstring in
    run_dpgan_eps_sweep.py calls the run "resumable", which is only true across
    finished pools and finished attacks, not across a crash inside one.

    That is what happened on 2026-09-21. Seed 19 had finished its 1000 training fits
    and ~100 test fits when another user's process took 15.5 GB of GPU 0 and this
    process (8 GB in use) died with torch.cuda.OutOfMemoryError. Seeds 7 and 11 then
    failed on their first fit for the same reason. threat_model.pkl was still the
    empty file from 01:28, so nothing was recoverable.

    Fix, without touching run_dpgan_eps_sweep.py: before calling run_one_epsilon,
    prewarm_pool() grows the same pools in chunks of --chunk fits (default 100, about
    10 min), saving after each chunk. run_one_epsilon then loads that cache through
    the same build_or_load_threat_model call, finds both pools already full, and
    generates nothing -- it goes straight to the guard, the export and the attacks.
    The counts-sweep pools were already grown in nested stages, so growing in chunks
    is not new to this repo; and each fit is seeded from a per-fit counter, so a chunk
    boundary does not change which seed any fit uses.

    On torch.cuda.OutOfMemoryError the chunk is retried after --oom-wait seconds, up to
    --oom-retries times. That is the failure a shared GPU produces, and it is usually
    gone in minutes. The retry needs no seed bookkeeping: TAPAS pools each dataset as
    it is made, the fit counter only advances after a fit succeeds, so the retry
    continues from exactly the next unused seed. A crash of any other kind, or
    running out of retries, loses at most one chunk: relaunch the same command and
    it resumes from the last saved chunk.

    Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True when launching (the error
    message itself suggests it); it reduces fragmentation and costs nothing.

PROGRESS, BECAUSE THE AUDIT IS OTHERWISE SILENT FOR HOURS
    TAPAS's dataset generation takes a progress tracker that defaults to a silent one
    (attacker_knowledge.py ~431), so nothing is logged while a pool grows. This
    script wraps common.SynthcityGenerator.fit to log every --progress-every fits
    with the elapsed time, the measured s/fit and the projected pool time remaining.
    The wrapper calls the original fit and only logs; it changes nothing about what
    is fitted or seeded. prewarm_pool() additionally logs each saved chunk.

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
from config import RESULTS_DIR, CACHE_DIR, METHOD_CONFIG           # noqa: E402
from seeds import TAPAS_TARGET_SEED                                # noqa: E402

# The pairs [6] already measured a signal AUC for, so every arm here has a
# pre-registered expectation rather than being a fresh draw. Ordered most-comparable
# first -- see ORDER in the module docstring.
PLACEBO_SEEDS = [19, 7, 11]
PLACEBO_EPSILON = 1.0           # the spike itself; every other budget is flat for every pair so far

# [6]'s signal AUCs, carried here so the log states the prior before the arm runs.
SIGNAL_AUC = {43: 0.874, 7: 0.902, 11: 0.946, 19: 0.877}

# Working rate for the start-of-arm budget guard: 5.9-6.4 s/fit over fits 400-1100 of
# the first [7] attempt, with another user's job on the GPU. [6] saw 10.8 and the first
# fits of [7] saw 3-4 -- see COST in the docstring. The run logs its own rate.
MEASURED_S_PER_FIT = 6.3
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


def prewarm_pool(epsilon: float, device_kwargs: dict, chunk: int,
                 oom_retries: int, oom_wait_s: int) -> None:
    """Grow both pools in chunks, saving after each, so a crash costs one chunk.

    See CHECKPOINTING in the module docstring. Builds the threat model with exactly
    the arguments run_one_epsilon uses (same background, same patched target, same
    plugin_kwargs, same cache dir from the patched eps_slug), so that run_one_epsilon's
    own build_or_load_threat_model call loads this one from disk, finds both pools
    already full, and generates nothing further.

    Train pool first and then test pool, the same order run_one_epsilon uses, so
    per-fit seeds line up as they would have in one unbroken call.
    """
    import torch

    label = sweep.eps_slug(epsilon)
    cache_dir = CACHE_DIR / f"{sweep.METHOD}_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    plugin_kwargs = {**METHOD_CONFIG[sweep.METHOD]["plugin_kwargs"], **device_kwargs}

    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    threat_model = common.build_or_load_threat_model(
        cache_dir=cache_dir, method=sweep.METHOD, background_dataset=background,
        target_record=target, alternate_record=alternate, description=description,
        epsilon=epsilon, plugin_kwargs=plugin_kwargs,
    )
    save_path = str(cache_dir / "threat_model")
    log.info(f"    prewarm {label}: chunks of {chunk}, resuming with "
             f"{len(threat_model._memory[True][0])} train / "
             f"{len(threat_model._memory[False][0])} test already saved")

    t0 = time.time()
    for training, want, name in ((True, sweep.NUM_TRAIN, "train"),
                                 (False, sweep.NUM_TEST, "test")):
        while len(threat_model._memory[training][0]) < want:
            step = min(want, len(threat_model._memory[training][0]) + chunk)
            for attempt in range(oom_retries + 1):
                try:
                    threat_model._generate_samples(step, training=training)
                    break
                except torch.cuda.OutOfMemoryError:
                    # NO counter rewind, and that is deliberate. TAPAS appends each
                    # dataset to memory as it is made (attacker_knowledge.py
                    # _sync_generate_data), so fits that succeeded before the error are
                    # already pooled and their seeds are spent; the retry generates only
                    # the shortfall and carries on from the next seed. SynthcityGenerator
                    # increments its counter AFTER the plugin fit, so the fit that ran
                    # out of memory did not advance it and is simply retried at the same
                    # seed. Rewinding would reuse seeds already in the pool -- duplicate
                    # datasets, the exact failure the distinctness guard exists to catch.
                    torch.cuda.empty_cache()
                    if attempt == oom_retries:
                        log.error(f"    {name} pool {step}/{want}: CUDA out of memory, "
                                  f"giving up after {oom_retries} retries")
                        raise
                    log.warning(f"    {name} pool {step}/{want}: CUDA out of memory "
                                f"(another process is holding the GPU?), retry "
                                f"{attempt + 1}/{oom_retries} in {oom_wait_s}s")
                    time.sleep(oom_wait_s)
            threat_model.save(save_path)
            log.info(f"    {name} pool {step}/{want} checkpointed "
                     f"({(time.time() - t0) / 60:.0f} min this run)")


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
    ap.add_argument("--chunk", type=int, default=100,
                    help="save the pool every N fits, so a crash costs at most one "
                         "chunk (default: 100, about 10 min)")
    ap.add_argument("--oom-retries", type=int, default=30,
                    help="retries of a chunk after a CUDA out-of-memory error, e.g. "
                         "another user's job on the GPU (default: 30)")
    ap.add_argument("--oom-wait", type=int, default=120,
                    help="seconds to wait between those retries (default: 120)")
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
            prewarm_pool(args.epsilon, device_kwargs, args.chunk,
                         args.oom_retries, args.oom_wait)
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
