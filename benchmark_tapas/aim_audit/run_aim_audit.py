#!/usr/bin/env python3
"""TAPAS MIA audit of AIM: the epsilon sweep at num_train=1000 / num_test=2500.

Structurally this is run_counts_sweep.py with one stage and a different
generator. Everything load-bearing is reused from benchmark_tapas/common.py --
the same fixed background (TAPAS_BG_SEED), the same target/alternate
(TAPAS_TARGET_SEED), the same 5-attack battery built under the same
SCORE_ATTACK_SEED, the same per-attack JSON caches, the same raw-score export,
the same distinctness guard. Only the generator differs (see aim_generator.py).

Nothing outside benchmark_tapas/aim_audit/ is modified. The AIM entry that would
otherwise go in config.METHOD_CONFIG is inlined below instead, so the existing
four-method config keeps exactly the shape every committed result was produced
under.

THE EPSILON SWEEP
    eps in {0.1, 1.0, 10, 100} at 1000/2500, matching the budgets the DPGAN eps
    sweep (results/eps_sweep/) and the DP-CTGAN spike diagnosis were run at, so the
    three DP generators can be read on one axis. --epsilon selects the arm; it is
    passed straight to AIMGenerator, which passes it to AIMSynthesizer, and NOTHING
    else moves -- delta stays at 1e-9, rounds/degree/max_cells/max_model_size and
    the bin edges all stay imported from sdg/aim.py.

    Unlike DP-CTGAN, AIM's epsilon is a genuine calibration rather than a stopping
    rule: the (eps, delta) budget is converted to rho-zCDP and the Gaussian noise is
    scaled so that exactly that budget is spent over the 208 rounds. So eps=0.1 is a
    noisier AIM, not a shorter one, and the arms are directly comparable to each
    other in a way the DP-CTGAN arms are not.

    WHERE eps=1.0 LIVES, AND WHY IT IS NOT REGENERATED
        The eps=1.0 arm is ALREADY DONE -- it is the whole existing counts sweep in
        results/aim_audit/{50_100,200_500,500_1000,1000_2500}/, and
        privacy_analysis.ipynb reads those paths. So eps=1.0 keeps them, along with
        its pool at cache/aim_audit, and only the new budgets are namespaced:
        cache/aim_audit_eps{e} and results/aim_audit/eps{e}/{nt}_{nte}/.

        The layout is asymmetric on purpose. Making it uniform would mean moving
        committed results that a notebook reads, and orphaning a pool that cost ~12 h
        to build, in exchange for a tidier tree. Not worth it.

    COST. Each new arm is a fresh 3500-fit pool -- the pools cannot be shared across
    budgets, since epsilon changes the mechanism -- so at the measured 12.1 s/fit
    that is ~12 h per arm, ~35 h for the three new ones. Run them one per night in
    the restart loop below, and note that AIM's per-fit cost is roughly flat in
    epsilon (the round count is fixed), so the estimate holds for all three.

WHY 1000/2500, AND WHY THE PLAN SAID 200/500
    The counts were chosen on a projected cost of ~90 s/fit, reasoning that AIM's
    ~208 rounds of Private-PGM inference run over a domain fixed by the public bin
    edges and so would barely get cheaper on a 500-row background. That reasoning
    was wrong. Measured with --probe 6 on a 500-row background: 12.5 s/fit steady
    state (first fit 24 s, jax compilation). Row count does dominate after all --
    full-data fits are 82-153 s, so the background is ~10x cheaper, not ~1x.

        200/500    =  700 fits ->  2.4 h
        500/1000   = 1500 fits ->  5.2 h
        1000/2500  = 3500 fits -> 12.2 h    <- one night, so this is the default

    1000/2500 is where every other method's headline privacy number lives, so
    auditing AIM there puts it in the same table rather than in a footnote. It is
    also the only stage that resolves much: in the counts sweep all four methods
    returned eps_low_95 = 0.000 at 50/100, with the max-AUC ordering inverted
    (bayesian_network 0.595 above dpgan 0.584); at 200/500 only dpgan separates
    (0.520); by 1000/2500 both GANs do (dpgan 1.761, ctgan 0.846).

    RECOMMENDED SEQUENCE, because it costs almost nothing extra: run 200/500
    first (~2.4 h) to bank a complete result, then run the default. TAPAS's
    memoisation only ever grows the pools, so the second run generates the 2800
    missing fits rather than starting over, and run_attack self-invalidates its
    per-attack cache when num_train/num_test increases. Total fits are still
    3500; the only extra cost is one ~5 min attack pass at the smaller stage.

CHECKPOINTING, WHICH MATTERS MORE HERE THAN ANYWHERE ELSE
    run_eps_sweep.py saves the threat model only after BOTH pools finish, which
    risked 8 h of unsaved work for DPGAN. That would have been far worse here:
    this audit's generator crashes the process every ~112 fits (grow_pools), so
    a single unsaved stretch would never have completed at all. Pools are grown
    in CHECKPOINT_EVERY-fit chunks with a save after each, so no crash costs more
    than one chunk and every restart resumes from disk.

WHAT IS AND IS NOT HELD FIXED
    fixed:   background, target/alternate, attack battery + its internal
             randomness, num_synthetic (500 records per simulation), and AIM's
             entire configuration EXCEPT epsilon -- delta, rounds, degree,
             max_cells, max_model_size and BIN_EDGES are imported from sdg/aim.py,
             not restated
    varies:  epsilon (--epsilon) and the counts (--num-train/--num-test). One
             process runs one (epsilon, counts) stage.

    AIM runs at delta=1e-9 regardless of dataset size, while synthcity's DPGAN
    derives delta = 1/len(X) (= 2e-3 on a 500-row background) and PrivBayes is
    pure DP. So the three DP methods are audited at three different deltas. That
    is already true of the committed counts sweep; it is a limitation to report,
    not something this script should quietly "fix" by pinning a different value.

    AIM is not bit-reproducible at any seed (opendp CSPRNG). See aim_generator.py.

OUTPUTS
    shared across the stages of ONE epsilon arm (the pool is grown, never rebuilt):
      cache/{cache_root}/threat_model.pkl                  pools, resumable
      cache/{cache_root}/datasets/synthetic_{split}.csv.gz every simulation, gzipped
      results/aim_audit/aim_audit_log.txt                  one log for every arm
    per stage, so a later stage never overwrites an earlier one's results:
      cache/{cache_root}/attacks_{nt}_{nte}/result_*.json  per-attack cache
      {results}/effeps_aim_{nt}_{nte}.csv                  per-attack metrics
      {results}/raw_scores_aim_{nt}_{nte}.csv              pre-threshold scores
      {results}/effective_epsilon_*.csv                    TAPAS's own report
      {results}/meta.json                                  timings + guard

    where cache_root is  aim_audit  and {results} is  results/aim_audit/{nt}_{nte}/
    at eps=1.0, and  aim_audit_eps{e}  /  results/aim_audit/eps{e}/{nt}_{nte}/  at
    every other budget -- see WHERE eps=1.0 LIVES above.

RUN IT IN THE RESTART LOOP, NOT DIRECTLY
    The script builds at most MAX_NEW_FITS per process and exits EXIT_INCOMPLETE
    while the pools are unfinished (see grow_pools for why). Exit 0 means that
    stage is done.

    The loop below also retries an UNEXPECTED exit, up to three times in a row.
    That matters because the jax failure kills the process outright rather than
    returning EXIT_INCOMPLETE: 75 fits should stay clear of it, but if one slips
    through, the chunk is already checkpointed and a retry resumes from disk. Three
    consecutive failures means something real is wrong, and the loop stops.

      screen -S aimaudit
      cd ~/priv-sdg && source venv/bin/activate
      run_stage () {
        fails=0
        while true; do
          python -u benchmark_tapas/aim_audit/run_aim_audit.py --num-train $1 --num-test $2
          rc=$?
          [ $rc -eq 0 ] && return 0
          if [ $rc -eq 3 ]; then fails=0; continue; fi
          fails=$((fails + 1)); echo "!! exit $rc (consecutive failure $fails)"
          [ $fails -ge 3 ] && return $rc
        done
      }
      run_stage 50 100 && echo "=== 50/100 DONE ===" && run_stage 200 500

    The epsilon sweep, one arm per night (each is a fresh ~12 h pool):

      run_stage () {            # $1 num_train  $2 num_test  $3.. extra flags
        fails=0
        while true; do
          python -u benchmark_tapas/aim_audit/run_aim_audit.py \
                 --num-train $1 --num-test $2 "${@:3}"
          rc=$?
          [ $rc -eq 0 ] && return 0
          if [ $rc -eq 3 ]; then fails=0; continue; fi
          fails=$((fails + 1)); echo "!! exit $rc (consecutive failure $fails)"
          [ $fails -ge 3 ] && return $rc
        done
      }
      for e in 0.1 10 100; do              # 1.0 is already done, see above
        run_stage 1000 2500 --epsilon $e || break
      done

    Staging like that is near-free: the pool is shared, so 50/100 then 200/500 is
    700 fits total rather than 850, and the smaller stage's results are complete
    and readable while the larger one is still running.

Also:
  python benchmark_tapas/aim_audit/run_aim_audit.py --probe 6                 # time 6 fits
  python benchmark_tapas/aim_audit/run_aim_audit.py --probe 6 --epsilon 0.1   # noisiest arm
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import tapas.threat_models as tm                                 # noqa: E402
import common                                                    # noqa: E402
from aim_generator import AIMGenerator, EPSILON, DELTA            # noqa: E402
from config import CACHE_DIR, RESULTS_DIR, TRAIN_CSV, NUM_SYNTHETIC  # noqa: E402
from seeds import SCORE_ATTACK_SEED                              # noqa: E402

METHOD = "aim"
NUM_TRAIN, NUM_TEST = 1000, 2500

# The config.METHOD_CONFIG entry AIM would have, kept local so config.py is
# untouched. `kind` places it in the statistical quadrant beside PrivBayes;
# there are no plugin_kwargs because AIM is not a Synthcity plugin.
AIM_CONFIG = {"dp": True, "kind": "statistical", "plugin_kwargs": {}}

# Fits per process before exiting for a fresh one. The first attempt died during
# fit 113, so this is set a third below that rather than just under it -- the
# threshold is jax's mapping space, which is not something to tune finely against.
# 75 fits is ~10 min at the 8.1 s/fit measured on the workstation, so ~47 restarts
# for the full 3500 and ~6% overhead from process startup. See grow_pools().
MAX_NEW_FITS = 75
CHECKPOINT_EVERY = 75   # one save per process: a crash costs at most ~10 min of fits
EXIT_INCOMPLETE = 3     # "pools not finished, restart me" -- not an error

# One shared log across every arm; RESULTS_ROOT below is namespaced per epsilon, so
# a sweep never overwrites a finished arm's results.
AIM_DIR = RESULTS_DIR / "aim_audit"
AIM_DIR.mkdir(parents=True, exist_ok=True)

# The POOL cache is shared across the COUNTS stages of one epsilon arm, on purpose:
# TAPAS only ever grows it, so running 50/100 and then 200/500 costs 700 fits in
# total, not 850. It is NOT shared across epsilon -- a different budget is a
# different mechanism, so each arm fits its own pool from scratch. Everything a
# stage WRITES is namespaced by its counts as well -- TAPAS's per-attack
# effective_epsilon_*.csv and meta.json are not keyed by counts, so a later stage
# would otherwise overwrite an earlier one's results while you were reading them.
#
# Set by set_paths() once --epsilon is known. Module-level so the helpers below can
# read them at call time without threading them through every signature.
CACHE_ROOT = RESULTS_ROOT = None


def set_paths(epsilon: float) -> None:
    """Point the pool cache and the results tree at this budget's own directories.

    eps=1.0 keeps the original unnamespaced paths, because that is where its ~12 h
    pool and its four committed counts stages already are and where
    privacy_analysis.ipynb reads them. See WHERE eps=1.0 LIVES in the module
    docstring for why the asymmetry is preferred to migrating them.
    """
    global CACHE_ROOT, RESULTS_ROOT
    if epsilon == EPSILON:
        CACHE_ROOT = CACHE_DIR / "aim_audit"
        RESULTS_ROOT = AIM_DIR
    else:
        CACHE_ROOT = CACHE_DIR / f"aim_audit_{eps_slug(epsilon)}"
        RESULTS_ROOT = AIM_DIR / eps_slug(epsilon)
    for d in (CACHE_ROOT, RESULTS_ROOT):
        d.mkdir(parents=True, exist_ok=True)


def eps_slug(eps: float) -> str:
    """Matches run_eps_sweep.eps_slug: 1.0 -> 'eps1', 0.1 -> 'eps0.1'."""
    return f"eps{eps:g}"


def stage_dirs(num_train: int, num_test: int):
    """(results, attack cache) for one stage, both keyed by its counts."""
    results = RESULTS_ROOT / f"{num_train}_{num_test}"
    attacks = CACHE_ROOT / f"attacks_{num_train}_{num_test}"
    for d in (results, attacks):
        d.mkdir(parents=True, exist_ok=True)
    return results, attacks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(AIM_DIR / "aim_audit_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("aim_audit")


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws. Raised by assert_distinct."""


# -- Copied from run_eps_sweep.py, for the reason stated there ------------
# Verbatim rather than imported: importing that module would run its
# logging.basicConfig and append this run's lines to the eps sweep's committed log.

def dataset_hash(dataset) -> str:
    """Fast content hash; pd.util.hash_pandas_object beats to_csv by a wide margin."""
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model) -> dict:
    """Abort if the memoised simulations are not essentially all distinct.

    Kept even though AIM cannot suffer the seeding bug that motivated it -- its
    CSPRNG makes every fit independent by construction. If this ever fails for
    AIM it means something else is wrong (a collapsed model, a constant table
    from a mis-scaled fit), which is exactly when you want to find out before
    spending the attack time.
    """
    fractions, failures = {}, []
    for training, name in ((True, "train"), (False, "test")):
        datasets, labels = threat_model._memory[training]
        if not datasets:
            continue
        for want in (True, False):
            subset = [d for d, l in zip(datasets, labels) if bool(l) is want]
            if not subset:
                continue
            n_distinct = len({dataset_hash(d) for d in subset})
            frac = n_distinct / len(subset)
            world = "D+" if want else "D-"
            fractions[f"{name}/{world}"] = frac
            log.info(f"    guard {name}/{world}: {n_distinct}/{len(subset)} distinct "
                     f"({frac:.1%})" + ("" if frac >= 0.99 else "   <-- BELOW 99%"))
            if frac < 0.99:
                failures.append(f"{name}/{world}: {n_distinct}/{len(subset)} ({frac:.1%})")

    if failures:
        raise DegeneratePool(
            f"AIM simulations are not independent draws -- {'; '.join(failures)}.\n"
            f"  AIM's noise comes from opendp's CSPRNG, so this is NOT the seeding bug.\n"
            f"  Most likely the scaling round trip is wrong: if unscaling is skipped, "
            f"encode() pushes every row into bin 0 and AIM fits a constant table.\n"
            f"  Check aim_generator.AIMGenerator._unscale against common._apply_scalers, "
            f"then delete {CACHE_ROOT} and re-run."
        )
    return fractions


def export_pools(threat_model) -> None:
    """Write every generated synthetic dataset out as gzipped CSV.

    Same format the counts and eps sweeps use, so the same diagnostics run over it:
    continuous columns are min-max scaled to [0,1] against the training split, and
    scalers.csv carries the bounds to put them back into original units.
    """
    out_dir = CACHE_ROOT / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    scalers_path = out_dir / "scalers.csv"
    if not scalers_path.exists():
        bounds = common._fit_scalers(pd.read_csv(TRAIN_CSV))
        pd.DataFrame([{"column": c, "min": lo, "max": hi} for c, (lo, hi) in bounds.items()]) \
          .to_csv(scalers_path, index=False)

    for training, name in ((True, "train"), (False, "test")):
        datasets, labels = threat_model._memory[training]
        if not datasets:
            continue
        frames = []
        for i, (d, l) in enumerate(zip(datasets, labels)):
            f = d.data.copy()
            f.insert(0, "ground_truth", int(bool(l)))
            f.insert(0, "dataset_idx", i)
            frames.append(f)
        path = out_dir / f"synthetic_{name}.csv.gz"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, compression="gzip")
        log.info(f"    exported {len(datasets)} {name} datasets -> {path.name} "
                 f"[{path.stat().st_size / 1e6:.1f} MB]")


# -- Pool growth, in capped chunks ----------------------------------------

def grow_pools(threat_model, num_train: int, num_test: int) -> int:
    """Grow both pools toward their targets, checkpointing every CHECKPOINT_EVERY
    fits and stopping after MAX_NEW_FITS new fits in this process.

    WHY THIS IS CHUNKED RATHER THAN TWO BIG CALLS
        The first attempt at this audit called generate_training_samples(1000)
        and _generate_samples(2500) once each. It died after 112 fits with

            LLVM compilation error: Cannot allocate memory

        on a box with 88 GB free and no swap in use, so this is not RAM. AIM
        selects a different set of marginals every round, so the graphical model
        changes shape and jax JIT-compiles fresh kernels each time -- roughly 208
        compilations per fit, ~23,000 by fit 112. Each one needs executable
        mappings, and the process exhausts its mapping space long before it
        exhausts memory. Nothing in this repo can stop jax from recompiling; the
        only reliable reset is a new process.

        So: cap the fits per process, checkpoint often, and let a shell loop
        restart. TAPAS's memoisation makes that free -- _generate_samples only
        ever generates the shortfall, so a restarted process resumes exactly
        where the last one stopped rather than redoing work.

        The 6-fit probe and the 10/20 smoke test both passed because neither got
        anywhere near the threshold. Any future generator that JIT-compiles per
        fit will need the same treatment.

    Returns the remaining fit budget: <= 0 means this process stopped early and
    the caller should exit EXIT_INCOMPLETE so the wrapper restarts it.
    """
    budget = MAX_NEW_FITS

    for training, target, name in ((True, num_train, "train"), (False, num_test, "test")):
        while True:
            have = len(threat_model._memory[training][0])
            if have >= target:
                break
            if budget <= 0:
                return 0
            step = min(target, have + CHECKPOINT_EVERY, have + budget)
            t0 = time.time()
            if training:
                threat_model.generate_training_samples(step)
            else:
                threat_model._generate_samples(step, training=False)
            grown = len(threat_model._memory[training][0])
            budget -= grown - have
            threat_model.save(str(CACHE_ROOT / "threat_model"))
            log.info(f"    {name} pool {grown}/{target} checkpointed "
                     f"(+{grown - have} fits, {time.time() - t0:.0f}s, "
                     f"{budget} left in this process)")

    return budget


# -- Shared setup ---------------------------------------------------------

def build_world(epsilon: float = EPSILON):
    """The fixed audit world: scalers, background, target/alternate, generator.

    Identical to what common.run_method builds for the other four methods, so the
    AIM row is comparable to theirs cell for cell. `epsilon` is the ONLY thing an
    arm changes; delta and every other AIM hyperparameter stay imported.
    """
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    scalers = common._fit_scalers(pd.read_csv(TRAIN_CSV))
    generator = AIMGenerator(description, scalers, epsilon=epsilon, delta=DELTA)
    return description, background, target, alternate, generator


class CacheMismatch(RuntimeError):
    """The cached pool was grown at a different epsilon."""


def assert_generator_matches(threat_model, epsilon: float) -> None:
    """Abort if the cached pool was grown at a different budget.

    build_or_load_threat_model unpickles the generator along with the pool and uses
    THAT object for every subsequent fit, so a mis-pointed cache would keep fitting
    at the old epsilon while this run labelled the results with the new one.
    set_paths keys the cache directory by epsilon precisely so this cannot happen --
    this is the belt to that braces, and it costs nothing.
    """
    generator = getattr(threat_model.atk_know_gen, "generator", None)
    got = getattr(generator, "epsilon", None) if generator is not None else None
    if got is not None and float(got) != float(epsilon):
        raise CacheMismatch(
            f"{CACHE_ROOT} was grown at epsilon={got}, but epsilon={epsilon} was "
            f"requested.\n"
            f"  The pool, not this invocation, decides what was actually fitted, so "
            f"continuing would label one budget's simulations as another's.\n"
            f"  Either point at the right cache or delete this one and re-fit."
        )


def build_or_load_threat_model(background, target, alternate, generator):
    """common.build_or_load_threat_model, but constructing an AIMGenerator.

    Not a call into that function: it hardcodes SynthcityGenerator. Everything
    else -- SwapTargetedMIA, ExactDataKnowledge, BlackBoxKnowledge, the cache
    round trip -- is the same objects it uses.
    """
    cache_path = CACHE_ROOT / "threat_model"
    if (CACHE_ROOT / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info("Building new threat model for aim (no cache found)")
    threat_model = common.SwapTargetedMIA(
        attacker_knowledge_data=tm.ExactDataKnowledge(background),
        target_record=target,
        alternate_record=alternate,
        attacker_knowledge_generator=tm.BlackBoxKnowledge(
            generator, num_synthetic_records=NUM_SYNTHETIC),
    )
    threat_model.save(str(cache_path))
    return threat_model


# -- Probe ----------------------------------------------------------------

def probe(n_fits: int, epsilon: float) -> int:
    """Time real fit+generate cycles on the 500-row background, then exit.

    This is the number the whole schedule turns on: 3500 fits at the measured s/fit
    is one epsilon arm's cost. AIM's round count does not depend on the budget, so
    the per-fit time should be roughly flat across arms -- if a low-epsilon arm comes
    back much slower, something other than noise scaling has changed. Nothing is
    cached or written.
    """
    _, background, target, _, generator = build_world(epsilon)
    member = background.copy()
    member.add_records(target, in_place=True)
    log.info(f"=== probe: {n_fits} fit+generate cycles on {len(member.data)} rows, "
             f"eps={epsilon:g}, delta={DELTA} ===")

    times = []
    for i in range(n_fits):
        t0 = time.time()
        generator.fit(member)
        synthetic = generator.generate(NUM_SYNTHETIC)
        dt = time.time() - t0
        times.append(dt)
        log.info(f"  fit {i}: {dt:.1f}s  ({len(synthetic.data)} rows, "
                 f"{len(synthetic.data.drop_duplicates())} unique)")

    mean = float(np.mean(times))
    log.info(f"=== mean {mean:.1f}s/fit (min {min(times):.1f}, max {max(times):.1f}) ===")
    for nt, nte in ((200, 500), (500, 1000), (1000, 2500)):
        log.info(f"    {nt}/{nte} = {nt + nte} fits -> {(nt + nte) * mean / 3600:.1f} h")
    return 0


# -- Audit ----------------------------------------------------------------

def run_audit(num_train: int, num_test: int, epsilon: float = EPSILON) -> int:
    log.info(f"=== TAPAS privacy audit: aim (dp={AIM_CONFIG['dp']}, "
             f"kind={AIM_CONFIG['kind']}, eps={epsilon:g}, delta={DELTA}, "
             f"num_train={num_train}, num_test={num_test}) ===")
    log.info(f"    cache {CACHE_ROOT.relative_to(REPO_ROOT)}   "
             f"results {RESULTS_ROOT.relative_to(REPO_ROOT)}   "
             f"total fits = {num_train + num_test}")

    results_dir, attack_cache = stage_dirs(num_train, num_test)
    _, background, target, alternate, generator = build_world(epsilon)
    threat_model = build_or_load_threat_model(background, target, alternate, generator)
    assert_generator_matches(threat_model, epsilon)

    # Same seed before build_attacks as every other method in the benchmark, so
    # AIM is probed by the same forests and the same 1500 random queries.
    np.random.seed(SCORE_ATTACK_SEED)
    attacks = common.build_attacks(target, background)

    t_pool = time.time()
    n_before = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])

    budget = grow_pools(threat_model, num_train, num_test)

    pool_s = round(time.time() - t_pool, 1)
    n_after = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
    log.info(f"    pools: {len(threat_model._memory[True][0])}/{num_train} train, "
             f"{len(threat_model._memory[False][0])}/{num_test} test  "
             f"(+{n_after - n_before} new fits, {pool_s:.0f}s)")

    if budget <= 0:
        log.info(f"=== fit budget for this process spent. Pools are checkpointed; "
                 f"exit {EXIT_INCOMPLETE} so the wrapper starts a fresh process "
                 f"(see MAX_NEW_FITS in the module docstring). ===")
        return EXIT_INCOMPLETE

    # Guard BEFORE the attacks, so a broken pool is caught before hours of scoring.
    fractions = assert_distinct(threat_model)
    export_pools(threat_model)

    rows, score_rows, no_scores = [], [], []
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=attack_cache, results_dir=results_dir,
        )
        result.update(method=METHOD, dp=AIM_CONFIG["dp"], kind=AIM_CONFIG["kind"],
                      num_train=num_train, num_test=num_test,
                      formal_epsilon=epsilon, delta=DELTA)
        # TAPAS's MIAttackSummary exposes the positive rates only.
        result["tn"] = 1.0 - result["fp"]
        result["fn"] = 1.0 - result["tp"]
        if summary is not None:
            score_rows.extend(common._score_rows(METHOD, attack, summary))
        else:
            no_scores.append(attack.label)
        rows.append(result)
        threat_model.save(str(CACHE_ROOT / "threat_model"))

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / f"effeps_{METHOD}_{num_train}_{num_test}.csv", index=False)
    if score_rows:
        path = results_dir / f"raw_scores_{METHOD}_{num_train}_{num_test}.csv"
        pd.DataFrame(score_rows).to_csv(path, index=False)
        log.info(f"    wrote {len(score_rows)} raw scores -> {path.name}")
    if no_scores:
        log.warning(f"    no raw scores for {no_scores} -- served from the per-attack "
                    f"JSON cache, which stores aggregates only. Delete those JSONs to "
                    f"recompute (cheap: the fits are memoised).")

    meta = {
        "method": METHOD, "formal_epsilon": epsilon, "delta": DELTA,
        "num_train": num_train, "num_test": num_test,
        "num_synthetic": NUM_SYNTHETIC,
        "pool_wall_clock_s": pool_s,
        "attack_wall_clock_s": float(out["wall_time_s"].sum()),
        "new_fits_this_run": n_after - n_before,
        "distinct_fractions": fractions,
        "attacks_without_raw_scores": no_scores,
        "note": "pool_wall_clock_s covers this invocation only; see new_fits_this_run "
                "to tell a fresh run from a resumed one. AIM is not bit-reproducible "
                "(opendp CSPRNG), so a re-run draws different simulations.",
    }
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] "
                 f"via {best['attack']} ===")
    else:
        log.warning(f"=== done, but no usable eps_low_95 across the 5 attacks. "
                    f"Results are written; inspect {results_dir.relative_to(REPO_ROOT)} ===")
    return 0


def main() -> int:
    global MAX_NEW_FITS, CHECKPOINT_EVERY
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=int, metavar="N",
                        help="time N real fit+generate cycles and exit (no caching)")
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num-test", type=int, default=NUM_TEST)
    parser.add_argument("--epsilon", type=float, default=EPSILON,
                        help=f"budget to audit (default {EPSILON}). Caches and results "
                             f"are namespaced per eps, so arms never overwrite each "
                             f"other; {EPSILON} keeps the original unnamespaced paths.")
    parser.add_argument("--max-new-fits", type=int, default=MAX_NEW_FITS,
                        help=f"fits per process before exiting {EXIT_INCOMPLETE} for a "
                             f"fresh one (default {MAX_NEW_FITS}; see grow_pools)")
    args = parser.parse_args()

    MAX_NEW_FITS = args.max_new_fits
    CHECKPOINT_EVERY = min(CHECKPOINT_EVERY, MAX_NEW_FITS)
    set_paths(args.epsilon)
    log.info(f"eps={args.epsilon:g} | cache {CACHE_ROOT.name} | "
             f"results {RESULTS_ROOT.relative_to(REPO_ROOT)}")

    if args.probe:
        return probe(args.probe, args.epsilon)
    try:
        return run_audit(args.num_train, args.num_test, args.epsilon)
    except DegeneratePool as exc:
        log.error(f"DISTINCTNESS GUARD FAILED:\n{exc}")
        return 1
    except CacheMismatch as exc:
        log.error(f"CACHE CONFIGURATION MISMATCH:\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
