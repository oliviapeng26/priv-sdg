#!/usr/bin/env python3
"""TAPAS MIA audit of SmartNoise DP-CTGAN: the counts sweep and the epoch_cap sweep.

Structurally this is aim_audit/run_aim_audit.py with a different generator, and it
is deliberately the same shape as scripts/eps_sweep/spike_diagnosis/
run_dpctgan_audit.py, which ran the eps sweep. Everything load-bearing is reused
from benchmark_tapas/common.py -- the same fixed background (TAPAS_BG_SEED), the
same target/alternate (TAPAS_TARGET_SEED), the same 5-attack battery built under
the same SCORE_ATTACK_SEED, the same per-attack JSON caches, the same raw-score
export, the same distinctness guard, the same 500 records per simulation. Only the
generator differs, and only two of ITS parameters ever move.

Nothing outside this folder is modified. The DP-CTGAN entry that would otherwise go
in config.METHOD_CONFIG is inlined below, so the existing four-method config keeps
exactly the shape every committed result was produced under.

THE TWO EXPERIMENTS THIS SCRIPT RUNS
    counts sweep    eps=1.0, epoch_cap=300, num_train/num_test walking
                    50/100 -> 200/500 -> 500/1000 -> 1000/2500.
                    Answers: how many shadow models does it take before DP-CTGAN's
                    eff-epsilon separates from zero? The synthcity methods have this
                    curve already (results/counts_sweep/); this puts DP-CTGAN on it.

    epoch_cap sweep eps=100, num_train/num_test=1000/2500, epoch_cap in
                    500 / 750 / 1000. epoch_cap=300 is ALREADY DONE -- it is the
                    eps100 arm of the spike diagnosis, and this script does not
                    re-run it (see WHERE 300 LIVES below).
                    Answers: at a budget loose enough that the cap binds instead of
                    the accountant, does more training leak more?

    eps=100 is the right budget for the epoch sweep precisely because it is loose.
    At eps=1.0 the accountant stops training long before 300 epochs, so raising the
    cap to 500 would change nothing and the sweep would measure noise. Confirm with
    --probe before committing a run: if epochs_run comes back well below the cap,
    the cap is not the binding constraint and that arm is a duplicate of a lower one.

EPSILON DOES NOT MEAN WHAT IT MEANS FOR SYNTHCITY -- READ BEFORE COMPARING
    synthcity solves for the noise multiplier so that exactly eps is spent over
    n_iter epochs: eps is the input, sigma is derived. SmartNoise inverts that.
    sigma is FIXED at 5 and eps is a STOPPING RULE: it trains, asks the accountant
    after each epoch what has been spent, and breaks out the first time the spend
    exceeds the target. delta is not passed either -- it is derived internally as
    1/(n*sqrt(n)), which on the 500-row audit background is 8.9e-5.

    So "eps = 1.0" labels two different mechanisms with two different deltas. That is
    deliberate: the library is taken as it ships, exactly as AIM is, because the
    question is whether an independent implementation of DP-SGD-on-CTGAN behaves the
    same way -- not whether a re-parameterised synthcity does. See dpctgan_generator.py.

    One consequence to watch: batch_size defaults to 500 and the audit background is
    500 rows, so the sampling rate is 1.0 and there is NO privacy amplification from
    subsampling. The budget can be exhausted in a handful of epochs, leaving a barely
    trained generator, and the library breaks out SILENTLY. --probe reports how many
    epochs actually ran; that number belongs in the write-up either way.

WHERE THE POOL CACHE LIVES, AND WHY THE COUNTS SWEEP IS NEARLY FREE
    TAPAS memoises simulations and only ever GROWS the pool, so a counts sweep that
    walks upward re-uses everything the previous stage fitted. The eps=1.0 /
    epoch_cap=300 pool already has 3500 fits in it from the spike diagnosis, at
    cache/dpctgan_eps1 -- and this script points at that same directory. So the whole
    counts sweep at eps=1.0 should need ZERO new fits and cost only four attack
    passes. Do not delete that cache to "start clean"; it is the expensive part.

    A cache root is keyed by EVERY generator parameter that moves (eps and
    epoch_cap), because build_or_load_threat_model unpickles the generator from the
    cache and uses THAT one -- a pool grown at epoch_cap=300 would silently keep
    training at 300 however this script was invoked. assert_generator_matches()
    checks the unpickled generator against the requested configuration and aborts on
    a mismatch rather than producing a mislabelled result.

    CACHES ARE SHARED WITH THE SPIKE DIAGNOSIS; RESULTS ARE NOT. At the default
    epoch_cap=300 the pool cache is cache/dpctgan_eps{e} -- the SAME directory the
    spike diagnosis grew -- which is exactly the point: its eps=1.0 pool is already
    at 3500 fits, so the counts sweep re-uses all of them. Results, though, always
    go to the new results/dp-ctgan/ tree, because that is where these experiments
    were asked to land; the spike diagnosis's own results stay untouched at
    results/eps_sweep/spike_diagnosis/dpctgan/, which is where
    privacy_analysis.ipynb reads them from.

    So epoch_cap=300 at eps=100 is NOT re-run by the epoch sweep: that arm is done,
    and its numbers are in the spike diagnosis's eps100 directory. Read the sweep as
    300 (there) against 500/750/1000 (here). If you would rather have all four in
    one tree, `--epsilon 100 --epoch-cap 300` re-scores it into
    results/dp-ctgan/eps100/1000_2500/ for the cost of five attack passes and zero
    new fits, since the pool is already complete.

ATTACK CACHES ARE KEYED BY COUNTS, THE POOL IS NOT
    common.run_attack self-invalidates its per-attack JSON when num_train/num_test
    increase. In a counts sweep that would mean each stage destroying the previous
    stage's cached results as it passed. So attack caches go in
    attacks_{num_train}_{num_test}/ -- one per stage, as run_aim_audit.py does --
    while the pool stays shared. Every stage's results survive and stay readable.

OUTPUTS
    shared across stages of one arm (the pool is grown, never rebuilt):
      cache/{cache_root}/threat_model.pkl                  pool, resumable
      cache/{cache_root}/datasets/synthetic_{split}.csv.gz every simulation, gzipped
      results/dp-ctgan/dpctgan_audit_log.txt               one log for every arm
    per stage, so a later stage never overwrites an earlier one:
      cache/{cache_root}/attacks_{nt}_{nte}/result_*.json  per-attack cache
      {results}/effeps_dpctgan_{nt}_{nte}.csv              per-attack metrics
      {results}/raw_scores_dpctgan_{nt}_{nte}.csv          pre-threshold scores
      {results}/effective_epsilon_*.csv                    TAPAS's own report
      {results}/meta.json                                  timings, guard, epochs_run

    where {results} is  results/dp-ctgan/eps{e}/{nt}_{nte}/                for the
    counts sweep, and results/dp-ctgan/epoch_sweep/eps{e}_ep{cap}/{nt}_{nte}/ when
    epoch_cap != 300.

RUN IT IN THE RESTART LOOP, NOT DIRECTLY
    The script builds at most MAX_NEW_FITS per process and exits EXIT_INCOMPLETE
    while the pools are unfinished. Exit 0 means that stage is done.

      screen -S dpctgan
      cd ~/priv-sdg && source venv/bin/activate
      run_stage () {            # $1 num_train  $2 num_test  $3.. extra flags
        fails=0
        while true; do
          python -u benchmark_tapas/dp-ctgan/run_dpctgan_audit.py \
                 --num-train $1 --num-test $2 "${@:3}"
          rc=$?
          [ $rc -eq 0 ] && return 0
          if [ $rc -eq 3 ]; then fails=0; continue; fi
          fails=$((fails + 1)); echo "!! exit $rc (consecutive failure $fails)"
          [ $fails -ge 3 ] && return $rc
        done
      }

      # 1. counts sweep at eps=1.0 -- walk upward so each stage banks a result
      run_stage 50 100 && run_stage 200 500 && run_stage 500 1000 && run_stage 1000 2500

      # 2. epoch_cap sweep at eps=100 (300 is already done; see WHERE 300 LIVES)
      for cap in 500 750 1000; do
        run_stage 1000 2500 --epsilon 100 --epoch-cap $cap || break
      done

Also:
  python benchmark_tapas/dp-ctgan/run_dpctgan_audit.py --probe 3
  python benchmark_tapas/dp-ctgan/run_dpctgan_audit.py --probe 3 --epsilon 100 --epoch-cap 1000

ENVIRONMENT
    snsynth 1.0.8's DPCTGAN is written against the Opacus 0.x API and raises
    `TypeError: PrivacyEngine.__init__() got an unexpected keyword argument
    'batch_size'` under opacus 1.4.1. Run this where the eps sweep ran, not in a
    conda env carrying synthcity's opacus. --probe fails fast and says so.
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

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import tapas.threat_models as tm                                      # noqa: E402
import common                                                         # noqa: E402
from dpctgan_generator import (DPCTGANGenerator, EPSILON, SIGMA,      # noqa: E402
                               EPOCHS, BATCH_SIZE)
from config import CACHE_DIR, RESULTS_DIR, TRAIN_CSV, NUM_SYNTHETIC   # noqa: E402
from seeds import SCORE_ATTACK_SEED                                   # noqa: E402

METHOD = "dpctgan"
NUM_TRAIN, NUM_TEST = 1000, 2500

# The counts ladder and the epoch ladder, for the docstring's run recipes and for
# --probe's cost projection. Not consumed as a loop: one process runs one stage, so
# a crash never costs more than a stage.
COUNTS_LADDER = [(50, 100), (200, 500), (500, 1000), (1000, 2500)]
EPOCH_LADDER = [500, 750, 1000]         # 300 is the spike diagnosis's arm

# Not in config.METHOD_CONFIG: DP-CTGAN is not a Synthcity plugin, and config.py is
# left exactly as every committed result was produced under.
DPCTGAN_CONFIG = {"dp": True, "kind": "neural"}

MAX_NEW_FITS = 400        # fits per process; a crash costs at most this many
CHECKPOINT_EVERY = 100
EXIT_INCOMPLETE = 3       # "pools not finished, restart me" -- not an error

DPCTGAN_DIR = RESULTS_DIR / "dp-ctgan"
DPCTGAN_DIR.mkdir(parents=True, exist_ok=True)

# Set by set_paths() once --epsilon and --epoch-cap are known. Module-level so the
# helpers can read them at call time without threading them through every signature.
CACHE_ROOT = RESULTS_ROOT = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(DPCTGAN_DIR / "dpctgan_audit_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("dpctgan_audit")


def eps_slug(eps: float) -> str:
    """Matches run_eps_sweep.eps_slug: 1.0 -> 'eps1', 0.1 -> 'eps0.1'."""
    return f"eps{eps:g}"


def set_paths(epsilon: float, epoch_cap: int) -> None:
    """Point the pool cache and the results tree at this arm's own directories.

    epoch_cap == EPOCHS (300, the SmartNoise default) keeps the spike diagnosis's
    original paths, because that is where its 3500-fit pool and its committed
    results already are -- see WHERE 300 LIVES in the module docstring. Any other
    cap is a different generator and gets its own pool and its own results tree.
    """
    global CACHE_ROOT, RESULTS_ROOT
    if epoch_cap == EPOCHS:
        CACHE_ROOT = CACHE_DIR / f"dpctgan_{eps_slug(epsilon)}"
        RESULTS_ROOT = DPCTGAN_DIR / eps_slug(epsilon)
    else:
        CACHE_ROOT = CACHE_DIR / f"dpctgan_{eps_slug(epsilon)}_ep{epoch_cap}"
        RESULTS_ROOT = (DPCTGAN_DIR / "epoch_sweep"
                        / f"{eps_slug(epsilon)}_ep{epoch_cap}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def stage_dirs(num_train: int, num_test: int):
    """(results, attack cache) for one stage, both keyed by its counts.

    The attack cache is per-stage because common.run_attack self-invalidates when
    the counts grow; sharing one directory across a counts sweep would mean each
    stage wiping the last one's cached attacks as it passed.
    """
    results = RESULTS_ROOT / f"{num_train}_{num_test}"
    attacks = CACHE_ROOT / f"attacks_{num_train}_{num_test}"
    for d in (results, attacks):
        d.mkdir(parents=True, exist_ok=True)
    return results, attacks


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws. Raised by assert_distinct."""


class CacheMismatch(RuntimeError):
    """The cached pool was grown by a differently-configured generator."""


# -- Copied from run_aim_audit.py, for the reason stated there ------------
# Verbatim rather than imported: importing that module would run its
# logging.basicConfig and append this run's lines to the AIM audit's log.

def dataset_hash(dataset) -> str:
    """Fast content hash; pd.util.hash_pandas_object beats to_csv by a wide margin."""
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model) -> dict:
    """Abort if the memoised simulations are not essentially all distinct.

    Especially worth keeping here: if DP-CTGAN's budget is exhausted after one or
    two epochs the generator can emit a near-constant table, and this catches that
    before hours of attack time rather than after.
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
            f"DP-CTGAN simulations are not independent draws -- {'; '.join(failures)}.\n"
            f"  Most likely the budget is exhausted almost immediately (sigma={SIGMA}, "
            f"batch_size={BATCH_SIZE} on a 500-row background means sample rate 1.0), "
            f"leaving an untrained generator that emits a near-constant table.\n"
            f"  Run --probe to see how many epochs survive, then decide whether to "
            f"record this as a finding about the library.\n"
            f"  Delete {CACHE_ROOT} before re-running."
        )
    return fractions


def assert_generator_matches(threat_model, epsilon: float, epoch_cap: int) -> None:
    """Abort if the cached pool was grown by a different generator configuration.

    build_or_load_threat_model unpickles the generator along with the pool and uses
    THAT object for every subsequent fit, so a mis-pointed cache would keep training
    at the old epsilon or epoch cap while this run labelled the results with the new
    ones. set_paths keys the cache directory by both parameters precisely so this
    cannot happen -- this check is the belt to that braces, and it costs nothing.
    """
    generator = getattr(threat_model.atk_know_gen, "generator", None)
    if generator is None:
        return
    got_eps = getattr(generator, "epsilon", None)
    got_epochs = getattr(generator, "epochs", None)
    mismatches = []
    if got_eps is not None and float(got_eps) != float(epsilon):
        mismatches.append(f"epsilon: cache has {got_eps}, requested {epsilon}")
    # A pool pickled before `epochs` existed on the class reports None. That pool can
    # only have been grown at the default, so it matches iff the default was asked for.
    effective_epochs = EPOCHS if got_epochs is None else got_epochs
    if int(effective_epochs) != int(epoch_cap):
        mismatches.append(f"epoch_cap: cache has {effective_epochs}, requested {epoch_cap}")
    if mismatches:
        raise CacheMismatch(
            f"{CACHE_ROOT} was grown by a different generator -- {'; '.join(mismatches)}.\n"
            f"  The pool, not this invocation, decides what was actually fitted, so "
            f"continuing would label one configuration's simulations as another's.\n"
            f"  Either point at the right cache or delete this one and re-fit."
        )


def export_pools(threat_model) -> None:
    """Every generated dataset as gzipped CSV, same format as the other audits.

    Continuous columns are min-max scaled to [0,1] against the training split, and
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


# -- Shared setup ---------------------------------------------------------

def build_world(cuda: bool, epsilon: float, epoch_cap: int):
    """The fixed audit world: background, target/alternate, scalers, generator.

    Identical to what common.run_method builds for the four Synthcity methods and to
    what run_aim_audit.build_world builds for AIM, so every row is comparable cell
    for cell. Only the generator differs.
    """
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    scalers = common._fit_scalers(pd.read_csv(TRAIN_CSV))
    generator = DPCTGANGenerator(description, scalers, epsilon=epsilon,
                                 cuda=cuda, epochs=epoch_cap)
    return description, background, target, alternate, generator


def build_or_load_threat_model(background, target, alternate, generator):
    """common.build_or_load_threat_model, but constructing a DPCTGANGenerator.

    Not a call into that function: it hardcodes SynthcityGenerator. Everything else
    -- SwapTargetedMIA, ExactDataKnowledge, BlackBoxKnowledge, the cache round trip
    -- is the same objects it uses.
    """
    cache_path = CACHE_ROOT / "threat_model"
    if (CACHE_ROOT / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info(f"Building new threat model for {METHOD} (no cache found)")
    threat_model = common.SwapTargetedMIA(
        attacker_knowledge_data=tm.ExactDataKnowledge(background),
        target_record=target,
        alternate_record=alternate,
        attacker_knowledge_generator=tm.BlackBoxKnowledge(
            generator, num_synthetic_records=NUM_SYNTHETIC),
    )
    threat_model.save(str(cache_path))
    return threat_model


def grow_pools(threat_model, num_train: int, num_test: int) -> int:
    """Grow both pools toward their targets, checkpointing every CHECKPOINT_EVERY
    fits and stopping after MAX_NEW_FITS new fits in this process.

    TAPAS's memoisation makes restarts free -- _generate_samples only ever generates
    the shortfall, so a restarted process resumes exactly where the last one stopped.
    For the eps=1.0 counts sweep the pool is already complete at 3500, so this
    returns immediately without fitting anything.

    Returns the remaining fit budget: <= 0 means this process stopped early and the
    caller should exit EXIT_INCOMPLETE so the wrapper restarts it.
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


# -- Probe ----------------------------------------------------------------

def probe(n_fits: int, cuda: bool, epsilon: float, epoch_cap: int) -> int:
    """Fit a few times on the real audit background and report what DP happened.

    The number that matters is epochs_run. DP-CTGAN breaks out of training silently
    once the budget is spent, so a small number here means the audit would be
    measuring an untrained network -- and, for the epoch_cap sweep specifically, that
    the cap is not the binding constraint and the arm duplicates a lower one.
    Nothing is cached or written.
    """
    _, background, target, _, generator = build_world(cuda, epsilon, epoch_cap)
    member = background.copy()
    member.add_records(target, in_place=True)
    log.info(f"=== probe: {n_fits} fits on {len(member.data)} rows, eps={epsilon:g}, "
             f"sigma={SIGMA}, batch_size={BATCH_SIZE}, epoch cap={epoch_cap} ===")
    log.info(f"    sample rate = batch_size / n = {BATCH_SIZE}/{len(member.data)} "
             f"= {BATCH_SIZE / len(member.data):.2f}  (1.0 means no subsampling "
             f"amplification)")

    times = []
    for i in range(n_fits):
        t0 = time.time()
        generator.fit(member)
        synthetic = generator.generate(NUM_SYNTHETIC)
        dt = time.time() - t0
        times.append(dt)
        uniq = len(synthetic.data.drop_duplicates())
        log.info(f"  fit {i}: {dt:.1f}s | epochs run "
                 f"{generator.last_epochs_run}/{epoch_cap} | eps spent "
                 f"{generator.last_epsilon_spent} | {uniq}/{NUM_SYNTHETIC} unique rows")

    mean = float(np.mean(times))
    log.info(f"=== mean {mean:.1f}s/fit ===")
    for nt, nte in COUNTS_LADDER:
        log.info(f"    {nt}/{nte} = {nt + nte} fits -> {(nt + nte) * mean / 3600:.1f} h "
                 f"(minus whatever the shared pool already holds)")

    run = generator.last_epochs_run
    if run is not None and run <= 5:
        log.warning(f"    only {run} epochs survived the budget. The audit would "
                    f"measure a barely-trained network. That is a real finding about "
                    f"the library, but decide before committing the run.")
    if run is not None and epoch_cap != EPOCHS and run < epoch_cap:
        log.warning(f"    epochs_run={run} < cap={epoch_cap}: the ACCOUNTANT is the "
                    f"binding constraint here, not the cap. This arm will be "
                    f"statistically identical to any lower cap above {run}, so the "
                    f"epoch sweep learns nothing from it. Raise epsilon or drop the arm.")
    return 0


# -- Audit ----------------------------------------------------------------

def run_audit(num_train: int, num_test: int, cuda: bool,
              epsilon: float, epoch_cap: int) -> int:
    log.info(f"=== TAPAS privacy audit: {METHOD} (SmartNoise) eps={epsilon:g}, "
             f"sigma={SIGMA}, epoch_cap={epoch_cap}, batch={BATCH_SIZE}, "
             f"num_train={num_train}, num_test={num_test} ===")
    log.info(f"    cache {CACHE_ROOT.relative_to(REPO_ROOT)}   "
             f"results {RESULTS_ROOT.relative_to(REPO_ROOT)}   "
             f"total fits = {num_train + num_test}")

    results_dir, attack_cache = stage_dirs(num_train, num_test)
    _, background, target, alternate, generator = build_world(cuda, epsilon, epoch_cap)
    threat_model = build_or_load_threat_model(background, target, alternate, generator)
    assert_generator_matches(threat_model, epsilon, epoch_cap)

    # Same seed before build_attacks as every other method in the benchmark, so
    # DP-CTGAN is probed by the same forests and the same 1500 random queries.
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
        result.update(method=METHOD, dp=DPCTGAN_CONFIG["dp"], kind=DPCTGAN_CONFIG["kind"],
                      num_train=num_train, num_test=num_test,
                      formal_epsilon=epsilon, sigma=SIGMA,
                      epochs_cap=epoch_cap, batch_size=BATCH_SIZE,
                      library="smartnoise")
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
        "method": METHOD, "library": "smartnoise", "formal_epsilon": epsilon,
        "sigma": SIGMA, "epochs_cap": epoch_cap, "batch_size": BATCH_SIZE,
        "num_train": num_train, "num_test": num_test, "num_synthetic": NUM_SYNTHETIC,
        # Read off the THREAT MODEL's generator, not the local one: on a resumed
        # process the pool is already complete, so the local object never fits and
        # its diagnostics stay None. The pickled one carries the last fit's values.
        "last_epochs_run": getattr(threat_model.atk_know_gen.generator,
                                   "last_epochs_run", None),
        "last_epsilon_spent": getattr(threat_model.atk_know_gen.generator,
                                      "last_epsilon_spent", None),
        "pool_wall_clock_s": pool_s, "new_fits_this_run": n_after - n_before,
        "attack_wall_clock_s": float(out["wall_time_s"].sum()),
        "distinct_fractions": fractions,
        "attacks_without_raw_scores": no_scores,
        "note": "epsilon is a STOPPING RULE in SmartNoise (fixed sigma, break when the "
                "accountant says the budget is spent), not a noise calibration as in "
                "synthcity. delta is derived internally as 1/(n*sqrt(n)). epochs_cap is "
                "a cap, not a schedule -- compare it against last_epochs_run to see "
                "which of the two actually bound. See dpctgan_generator.py.",
    }
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] "
                 f"via {best['attack']} ===")
    else:
        log.warning(f"=== done, but no usable eps_low_95 across the 5 attacks. "
                    f"Results are written; inspect "
                    f"{results_dir.relative_to(REPO_ROOT)} ===")
    return 0


def main() -> int:
    global MAX_NEW_FITS, CHECKPOINT_EVERY
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=int, metavar="N",
                    help="fit N times, report epochs run and eps spent, then exit")
    ap.add_argument("--num-train", type=int, default=NUM_TRAIN)
    ap.add_argument("--num-test", type=int, default=NUM_TEST)
    ap.add_argument("--epsilon", type=float, default=EPSILON,
                    help=f"budget to audit (default {EPSILON}). Caches and results are "
                         f"namespaced per eps, so arms never overwrite each other.")
    ap.add_argument("--epoch-cap", type=int, default=EPOCHS,
                    help=f"DP-CTGAN epoch cap (default {EPOCHS}, the SmartNoise "
                         f"default). {EPOCH_LADDER} are the epoch sweep's arms; they "
                         f"get their own caches and land under epoch_sweep/.")
    ap.add_argument("--max-new-fits", type=int, default=MAX_NEW_FITS,
                    help=f"fits per process before exiting {EXIT_INCOMPLETE} for a "
                         f"fresh one (default {MAX_NEW_FITS}; see grow_pools)")
    args = ap.parse_args()

    MAX_NEW_FITS = args.max_new_fits
    CHECKPOINT_EVERY = min(CHECKPOINT_EVERY, MAX_NEW_FITS)
    set_paths(args.epsilon, args.epoch_cap)

    import torch
    cuda = torch.cuda.is_available()
    log.info(f"device: {'cuda' if cuda else 'cpu'} "
             f"(torch.cuda.is_available()={cuda}) | eps={args.epsilon:g} | "
             f"epoch_cap={args.epoch_cap} | cache {CACHE_ROOT.name} | "
             f"results {RESULTS_ROOT.relative_to(REPO_ROOT)}")

    if args.probe:
        return probe(args.probe, cuda, args.epsilon, args.epoch_cap)
    try:
        return run_audit(args.num_train, args.num_test, cuda,
                         args.epsilon, args.epoch_cap)
    except DegeneratePool as exc:
        log.error(f"DISTINCTNESS GUARD FAILED:\n{exc}")
        return 1
    except CacheMismatch as exc:
        log.error(f"CACHE CONFIGURATION MISMATCH:\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
