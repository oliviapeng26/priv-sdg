#!/usr/bin/env python3
"""TAPAS MIA audit of GReaT (fine-tuned GPT-2), at num_train=1000 / num_test=2500.

Structurally this is run_aim_audit.py with a different generator and no epsilon
arm. Everything load-bearing is reused from benchmark_tapas/common.py -- the same
fixed background (TAPAS_BG_SEED), the same target/alternate (TAPAS_TARGET_SEED),
the same 5-attack battery built under the same SCORE_ATTACK_SEED, the same
per-attack JSON caches, the same raw-score export. Only the generator differs
(see tapas_wrappers/great_generator.py).

Nothing outside benchmark_tapas/ is modified. The GReaT entry that would
otherwise go in config.METHOD_CONFIG is inlined below instead, so the existing
four-method config keeps exactly the shape every committed result was produced
under.

NO EPSILON ARM
    GReaT is non-DP: GPT-2 fine-tuning carries no formal guarantee, so there is
    no budget to sweep and no eps namespacing in the paths. It sits in the same
    quadrant as CTGAN -- neural, non-DP -- and is read against CTGAN's row.

COST, AND WHY --probe COMES FIRST
    sdg/great_sanity_check.py measured gpt2 at k=2000 on 500 rows: 11.5 s fit +
    14.2 s sample = ~25.7 s/fit. At that rate 1000/2500 = 3500 fits is ~25 h.
    But that was measured standalone, NOT through this wrapper, which adds an
    unscale/rescale round trip and rejection sampling against the TAPAS
    vocabulary (see great_generator.py). --probe measures the real number, and
    also reports the acceptance rate and whether consecutive fits actually
    differ. Run it before committing a night to the full pool.

        200/500    =  700 fits ->  5.0 h
        500/1000   = 1500 fits -> 10.7 h
        1000/2500  = 3500 fits -> 25.0 h   <- default; where every other method's
                                              headline number lives

    TAPAS's memoisation only ever grows the pool, so running a smaller stage
    first and the default afterwards costs 3500 fits in total, not 4200, and
    run_attack self-invalidates its per-attack cache when the counts increase.

CHECKPOINTING AND THE RESTART LOOP
    AIM's audit crashed the process every ~112 fits (jax exhausting its mapping
    space) and had to be chunked. GReaT has no known equivalent threshold -- but
    it rebuilds a GPT-2 and an HF Trainer on every one of 3500 fits, and a slow
    VRAM creep would only surface hours into an unattended run. great_generator
    frees the previous model explicitly, and this script caps fits per process
    anyway: cheap insurance, and it costs only process startup.

    So MAX_NEW_FITS here is a checkpoint cadence, not a crash workaround. At
    ~25.7 s/fit it is ~43 min of work risked per process. Raise it with
    --max-new-fits once a full run has proven stable.

WHAT IS AND IS NOT HELD FIXED
    fixed:   background, target/alternate, attack battery + its internal
             randomness, num_synthetic (500 records per simulation), and GReaT's
             entire configuration -- llm, batch_size, epochs, fp16,
             dataloader_num_workers and the sample-time k are imported from
             sdg/generate_great.py, not restated, so the audited model is the one
             whose utility and fidelity are reported.
    varies:  the counts (--num-train/--num-test), and the per-fit seed, which
             MUST vary -- see the seeding section of great_generator.py.

OUTPUTS
      cache/great_audit/threat_model.pkl                   pools, resumable
      cache/great_audit/attacks_{nt}_{nte}/result_*.json   per-attack cache
      results/sample_size_sweep/great/great_audit_log.txt  one log for every stage
      results/sample_size_sweep/great/{nt}_{nte}/effeps_great_{nt}_{nte}.csv
      results/sample_size_sweep/great/{nt}_{nte}/raw_scores_great_{nt}_{nte}.csv
      results/sample_size_sweep/great/{nt}_{nte}/effective_epsilon_*.csv
      results/sample_size_sweep/great/{nt}_{nte}/meta.json

    The simulated datasets themselves are not exported to CSV (AIM's audit does;
    its diagnostics read them). They live in threat_model.pkl if ever needed --
    3500 x 500 rows of gzipped CSV is not worth the disk on this workstation.

RUN IT IN THE RESTART LOOP, NOT DIRECTLY
    The script builds at most MAX_NEW_FITS per process and exits EXIT_INCOMPLETE
    while the pools are unfinished. Exit 0 means that stage is done.

      screen -S greataudit
      cd ~/priv-sdg && source venv/bin/activate
      export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0
      run_stage () {
        fails=0
        while true; do
          python -u benchmark_tapas/audits/run_great_audit.py \
                 --num-train $1 --num-test $2
          rc=$?
          [ $rc -eq 0 ] && return 0
          if [ $rc -eq 3 ]; then fails=0; continue; fi
          fails=$((fails + 1)); echo "!! exit $rc (consecutive failure $fails)"
          [ $fails -ge 3 ] && return $rc
        done
      }
      run_stage 1000 2500

    NCCL_P2P_DISABLE/NCCL_IB_DISABLE are not optional on the RTX 4090s: accelerate's
    distributed init raises NotImplementedError on consumer cards without them,
    even for a single-GPU run, whenever more than one GPU is visible.

Also:
  python benchmark_tapas/audits/run_great_audit.py --probe 10
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
sys.path.insert(0, str(BENCHMARK_DIR / "tapas_wrappers"))
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import tapas.threat_models as tm                                   # noqa: E402
import common                                                      # noqa: E402
from great_generator import GReaTGenerator, LLM, SAMPLE_K, SAMPLE_MAX_LENGTH  # noqa: E402
from config import CACHE_DIR, RESULTS_DIR, TRAIN_CSV, NUM_SYNTHETIC  # noqa: E402
from seeds import SCORE_ATTACK_SEED                                # noqa: E402

METHOD = "great"
NUM_TRAIN, NUM_TEST = 1000, 2500

# The config.METHOD_CONFIG entry GReaT would have, kept local so config.py is
# untouched. `kind` places it in the neural quadrant beside CTGAN; dp is False
# (GPT-2 fine-tuning carries no formal guarantee) and there are no plugin_kwargs
# because GReaT is not a Synthcity plugin.
GREAT_CONFIG = {"dp": False, "kind": "neural", "plugin_kwargs": {}}

# Checkpoint cadence, NOT a crash workaround -- see the module docstring. ~43 min
# of fits per process at the sanity-check rate; --max-new-fits overrides it.
MAX_NEW_FITS = 100
CHECKPOINT_EVERY = 50   # a crash costs at most ~21 min of fits
EXIT_INCOMPLETE = 3     # "pools not finished, restart me" -- not an error

CACHE_ROOT = CACHE_DIR / "great_audit"
RESULTS_ROOT = RESULTS_DIR / "sample_size_sweep" / "great"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RESULTS_ROOT / "great_audit_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("great_audit")


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws. Raised by assert_distinct."""


def stage_dirs(num_train: int, num_test: int):
    """(results, attack cache) for one stage, both keyed by its counts.

    TAPAS's per-attack effective_epsilon_*.csv and meta.json are not keyed by
    counts, so without this a later stage would overwrite an earlier one's
    results while you were reading them.
    """
    results = RESULTS_ROOT / f"{num_train}_{num_test}"
    attacks = CACHE_ROOT / f"attacks_{num_train}_{num_test}"
    for d in (results, attacks):
        d.mkdir(parents=True, exist_ok=True)
    return results, attacks


# -- Distinctness guard ---------------------------------------------------
# Copied from run_aim_audit.py rather than imported, for the reason stated there:
# importing that module would run its logging.basicConfig and append this run's
# lines to the AIM audit's committed log.

def dataset_hash(dataset) -> str:
    """Fast content hash; pd.util.hash_pandas_object beats to_csv by a wide margin."""
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model) -> dict:
    """Abort if the memoised simulations are not essentially all distinct.

    This is the regression test for the seeding fix in great_generator.py. HF's
    Trainer reseeds random/numpy/torch globally from TrainingArguments.seed on
    every fit, defaulting to 42; ExactDataKnowledge hands every simulation the
    same background, so a GReaT wrapper that forgot to vary the seed would
    reproduce the pre-2026-08-23 Synthcity collapse exactly -- all D+ simulations
    one identical dataset, all D- another, and an audit whose effective sample
    size is 2.
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
            f"GReaT simulations are not independent draws -- {'; '.join(failures)}.\n"
            f"  This is the Synthcity collapse bug's shape. Check that "
            f"GReaTGenerator.fit passes seed=self.seed_base + self._fit_counter "
            f"into GReaT(), that the counter survives __getstate__, and that the "
            f"cached pool was not grown by an earlier build that lacked the fix.\n"
            f"  If the wrapper is correct, delete {CACHE_ROOT} and re-fit."
        )
    return fractions


# -- Pool growth, in capped chunks ----------------------------------------

def grow_pools(threat_model, num_train: int, num_test: int) -> int:
    """Grow both pools toward their targets, checkpointing every CHECKPOINT_EVERY
    fits and stopping after MAX_NEW_FITS new fits in this process.

    TAPAS's memoisation makes the restart free -- _generate_samples only ever
    generates the shortfall, so a restarted process resumes exactly where the
    last one stopped rather than redoing work.

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

def build_world():
    """The fixed audit world: scalers, background, target/alternate, generator.

    Identical to what common.run_method builds for the other methods, so the
    GReaT row is comparable to theirs cell for cell.
    """
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    scalers = common._fit_scalers(pd.read_csv(TRAIN_CSV))
    generator = GReaTGenerator(description, scalers)
    return description, background, target, alternate, generator


def build_or_load_threat_model(background, target, alternate, generator):
    """common.build_or_load_threat_model, but constructing a GReaTGenerator.

    Not a call into that function: it hardcodes SynthcityGenerator. Everything
    else -- SwapTargetedMIA, ExactDataKnowledge, BlackBoxKnowledge, the cache
    round trip -- is the same objects it uses.
    """
    cache_path = CACHE_ROOT / "threat_model"
    if (CACHE_ROOT / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info("Building new threat model for great (no cache found)")
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

def probe(n_fits: int) -> int:
    """Time real fit+generate cycles on the 500-row background, then exit.

    Three numbers come out of this, and all three gate the full run:

      s/fit              the whole schedule turns on it. The ~25.7 s from
                         sdg/great_sanity_check.py did not include this wrapper's
                         unscale/rescale round trip or its rejection sampling.
      acceptance rate    fraction of sampled rows whose categorical values are in
                         the TAPAS vocabulary. Low means rows are being truncated
                         mid-value; raise sample()'s max_length.
      distinct outputs   consecutive fits use consecutive seeds, so they must
                         produce different data. If this is not n_fits/n_fits the
                         seeding fix is not working and the audit would be void.

    Nothing is cached or written.
    """
    _, background, target, _, generator = build_world()
    member = background.copy()
    member.add_records(target, in_place=True)
    log.info(f"=== probe: {n_fits} fit+generate cycles on {len(member.data)} rows, "
             f"llm={LLM}, k={SAMPLE_K}, max_length={SAMPLE_MAX_LENGTH} ===")

    times, hashes, rates = [], [], []
    for i in range(n_fits):
        t0 = time.time()
        generator.fit(member)
        synthetic = generator.generate(NUM_SYNTHETIC)
        dt = time.time() - t0
        times.append(dt)
        hashes.append(dataset_hash(synthetic))
        rates.append(generator.last_acceptance_rate)
        log.info(f"  fit {i}: {dt:.1f}s  ({len(synthetic.data)} rows, "
                 f"{len(synthetic.data.drop_duplicates())} unique, "
                 f"acceptance {generator.last_acceptance_rate:.1%} over "
                 f"{generator.last_sample_rounds} sample round(s))")

    mean = float(np.mean(times))
    n_distinct = len(set(hashes))
    log.info(f"=== mean {mean:.1f}s/fit (min {min(times):.1f}, max {max(times):.1f}), "
             f"mean acceptance {np.mean(rates):.1%} ===")
    log.info(f"=== {n_distinct}/{n_fits} distinct outputs across consecutive seeds "
             + ("(seeding OK) ===" if n_distinct == n_fits else
                "<-- SEEDING BROKEN: identical fits mean the audit would measure "
                "an effective sample size of 2. Do not run the full audit. ==="))
    for nt, nte in ((200, 500), (500, 1000), (1000, 2500)):
        log.info(f"    {nt}/{nte} = {nt + nte} fits -> {(nt + nte) * mean / 3600:.1f} h")
    return 0 if n_distinct == n_fits else 1


# -- Audit ----------------------------------------------------------------

def run_audit(num_train: int, num_test: int) -> int:
    log.info(f"=== TAPAS privacy audit: great (dp={GREAT_CONFIG['dp']}, "
             f"kind={GREAT_CONFIG['kind']}, num_train={num_train}, "
             f"num_test={num_test}) ===")
    log.info(f"    cache {CACHE_ROOT.relative_to(REPO_ROOT)}   "
             f"results {RESULTS_ROOT.relative_to(REPO_ROOT)}   "
             f"total fits = {num_train + num_test}")

    results_dir, attack_cache = stage_dirs(num_train, num_test)
    _, background, target, alternate, generator = build_world()
    threat_model = build_or_load_threat_model(background, target, alternate, generator)

    # Same seed before build_attacks as every other method in the benchmark, so
    # GReaT is probed by the same forests and the same 1500 random queries.
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

    rows, score_rows, no_scores = [], [], []
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=attack_cache, results_dir=results_dir,
        )
        result.update(method=METHOD, dp=GREAT_CONFIG["dp"], kind=GREAT_CONFIG["kind"],
                      num_train=num_train, num_test=num_test, formal_epsilon=None)
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
        "method": METHOD, "formal_epsilon": None,
        "num_train": num_train, "num_test": num_test,
        "num_synthetic": NUM_SYNTHETIC,
        "pool_wall_clock_s": pool_s,
        "attack_wall_clock_s": float(out["wall_time_s"].sum()),
        "new_fits_this_run": n_after - n_before,
        "distinct_fractions": fractions,
        "attacks_without_raw_scores": no_scores,
        "note": "pool_wall_clock_s covers this invocation only; see new_fits_this_run "
                "to tell a fresh run from a resumed one.",
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
    parser.add_argument("--max-new-fits", type=int, default=MAX_NEW_FITS,
                        help=f"fits per process before exiting {EXIT_INCOMPLETE} for a "
                             f"fresh one (default {MAX_NEW_FITS}; see grow_pools)")
    args = parser.parse_args()

    MAX_NEW_FITS = args.max_new_fits
    CHECKPOINT_EVERY = min(CHECKPOINT_EVERY, MAX_NEW_FITS)

    if args.probe:
        return probe(args.probe)
    try:
        return run_audit(args.num_train, args.num_test)
    except DegeneratePool as exc:
        log.error(f"DISTINCTNESS GUARD FAILED:\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
