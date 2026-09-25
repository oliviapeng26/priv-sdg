#!/usr/bin/env python3
"""TAPAS MIA audit of DP-2Stage-O, at num_train=200 / num_test=500 to start.

Structurally this is run_great_audit.py with a different generator plus the formal
privacy record that the DP audits (AIM, DP-CTGAN) carry. Everything load-bearing is
reused from benchmark_tapas/common.py -- the same fixed background (TAPAS_BG_SEED), the
same target/alternate (TAPAS_TARGET_SEED), the same 5-attack battery built under the
same SCORE_ATTACK_SEED, the same per-attack JSON caches, the same raw-score export.
Only the generator differs (see tapas_wrappers/dp2stage_generator.py, and
sdg/dp2stage_driver.py for the method itself).

THE FORMAL EPSILON IS REPORTED NEXT TO THE EMPIRICAL ONE
    DP-2Stage-O has a genuine (eps=1, delta=1e-5) guarantee: Stage 1 (Airline) never
    touches the private data, and Stage 2 is DP-SGD with Opacus's RDP accountant solving
    sigma for eps=1. Every fit's SPENT epsilon, sigma and step count come back from that
    accountant (through the driver) and are logged per fit; meta.json carries their
    summary and each effeps row carries formal_epsilon=1.0 and delta, the same columns
    the AIM/DP-CTGAN audits write. TAPAS's eps_low_95 is then read against 1.0.
    LIMITATION to state in the write-up: delta is 1e-5 (the paper's), whereas AIM uses
    1e-9, DP-CTGAN derives 1/(n*sqrt(n)) and DPGAN 1/n, so "eps=1" is not the same
    guarantee across generators.

WHAT IS HELD FIXED (everything else lives in the driver / wrapper)
    background, target/alternate, attack battery + its internal randomness,
    num_synthetic (500 records per simulation), and DP-2Stage's entire configuration --
    imported from sdg/dp2stage_driver.py, never restated, so the audited model is the one
    whose utility and fidelity are reported. One shared Stage 1 checkpoint for every fit.
    varies: the counts (--num-train/--num-test), and the per-fit seed, which MUST vary.

COST, AND WHY --probe COMES FIRST
    Nothing about per-fit cost is known yet. Each fit here is a fresh subprocess doing:
    process start-up, loading the ~0.5 GB Stage 1 checkpoint, DP Stage 2 on ~500 rows
    (10 epochs, ~160 DP-SGD steps), then sampling 500 rows. The driver's own timings come
    from `python sdg/dp2stage_probe.py fit`; THIS probe measures the whole thing through
    the wrapper, and also reports the spent epsilon, acceptance and whether consecutive
    fits actually differ. Run it before committing a night to the pool.

        200/500    =  700 fits   (the starting stage)
        500/1000   = 1500 fits
        1000/2500  = 3500 fits   (where the other generators' headline numbers live)
    --probe prints the projected hours for each from the measured mean.

    TAPAS's memoisation only ever grows the pool, so running a smaller stage first and a
    larger one afterwards costs the larger total, not the sum, and run_attack
    self-invalidates its per-attack cache when the counts increase.

CHECKPOINTING AND THE RESTART LOOP (same pattern as GReaT)
    czha4500's job shares this workstation's GPUs and has OOM'd fits before.
    dp2stage_generator.HardSampleFailure catches that signature and main() maps it to
    EXIT_INCOMPLETE, so the restart loop retries it indefinitely without spending one of
    its 3 consecutive-failure strikes (reserved for non-transient failures: quota
    unfilled, vocabulary mismatch, overspent epsilon). MAX_NEW_FITS / CHECKPOINT_EVERY
    bound how much a restart repeats: at most one partial CHECKPOINT_EVERY chunk.

    PROBE-DEPENDENT: MAX_NEW_FITS and CHECKPOINT_EVERY below are GReaT's values, carried
    over provisionally. Set them from the measured s/fit so a chunk is ~30 minutes.
    Grep the repo for PROBE-DEPENDENT for the full list.

OUTPUTS
      cache/dp2stage_audit/threat_model.pkl                     pools, resumable
      cache/dp2stage_audit/fit_log.jsonl                        one line per fit: spent eps, sigma, ...
      cache/dp2stage_audit/attacks_{nt}_{nte}/result_*.json     per-attack cache
      results/sample_size_sweep/dp2stage/dp2stage_audit_log.txt one log for every stage
      results/sample_size_sweep/dp2stage/{nt}_{nte}/effeps_dp2stage_{nt}_{nte}.csv
      results/sample_size_sweep/dp2stage/{nt}_{nte}/raw_scores_dp2stage_{nt}_{nte}.csv
      results/sample_size_sweep/dp2stage/{nt}_{nte}/effective_epsilon_*.csv
      results/sample_size_sweep/dp2stage/{nt}_{nte}/fit_log_dp2stage_{nt}_{nte}.jsonl
      results/sample_size_sweep/dp2stage/{nt}_{nte}/meta.json

RUN IT IN THE RESTART LOOP, IN ~/priv-sdg/venv (the TAPAS venv), NOT DIRECTLY
    The script builds at most MAX_NEW_FITS per process and exits EXIT_INCOMPLETE while the
    pools are unfinished. Exit 0 means that stage is done. Each fit shells out to
    ~/venv-dp2stage/bin/python (override with $DP2STAGE_PYTHON); it inherits
    CUDA_VISIBLE_DEVICES.

      screen -S dp2audit
      cd ~/priv-sdg && source venv/bin/activate
      export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0
      nvidia-smi                                   # is czha4500's job on this GPU?
      run_stage () {
        fails=0
        while true; do
          python -u benchmark_tapas/audits/run_dp2stage_audit.py \\
                 --num-train $1 --num-test $2
          rc=$?
          [ $rc -eq 0 ] && return 0
          if [ $rc -eq 3 ]; then fails=0; continue; fi
          fails=$((fails + 1)); echo "!! exit $rc (consecutive failure $fails)"
          [ $fails -ge 3 ] && return $rc
        done
      }
      run_stage 200 500

Also:
  python benchmark_tapas/audits/run_dp2stage_audit.py --probe 10
  python benchmark_tapas/audits/run_dp2stage_audit.py --probe 10 --sample-batch 500   # PROBE-DEPENDENT
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
from dp2stage_generator import (DP2StageGenerator, HardSampleFailure,   # noqa: E402
                                AUDIT_SAMPLE_BATCH, EPSILON, DELTA,
                                EPS_TOLERANCE, MAX_NEW_TOKENS)
from config import CACHE_DIR, RESULTS_DIR, TRAIN_CSV, NUM_SYNTHETIC  # noqa: E402
from seeds import SCORE_ATTACK_SEED                                # noqa: E402

METHOD = "dp2stage"
NUM_TRAIN, NUM_TEST = 200, 500     # the starting stage (700 fits); GReaT defaulted to 1000/2500

# The config.METHOD_CONFIG entry DP-2Stage would have, kept local so config.py is
# untouched. neural: it sits in the neural quadrant; dp is True with a formal budget.
DP2STAGE_CONFIG = {"dp": True, "kind": "neural", "plugin_kwargs": {}}
FORMAL_EPSILON = EPSILON

# PROBE-DEPENDENT: checkpoint cadence and per-process fit budget. These are GReaT's values
# (100 / 50), carried over WITHOUT a measurement. Choose CHECKPOINT_EVERY ~= 1800 / (mean
# s/fit from --probe) so a crash or contention restart repeats at most ~30 min, and
# MAX_NEW_FITS as a small multiple of it. --max-new-fits overrides MAX_NEW_FITS at the CLI.
MAX_NEW_FITS = 100
CHECKPOINT_EVERY = 50
EXIT_INCOMPLETE = 3     # "pools not finished, restart me" -- not an error

CACHE_ROOT = CACHE_DIR / "dp2stage_audit"
RESULTS_ROOT = RESULTS_DIR / "sample_size_sweep" / "dp2stage"
FIT_LOG = CACHE_ROOT / "fit_log.jsonl"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RESULTS_ROOT / "dp2stage_audit_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("dp2stage_audit")


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws. Raised by assert_distinct."""


def stage_dirs(num_train: int, num_test: int):
    """(results, attack cache) for one stage, both keyed by its counts.

    TAPAS's per-attack effective_epsilon_*.csv and meta.json are not keyed by counts, so
    without this a later stage would overwrite an earlier one's results.
    """
    results = RESULTS_ROOT / f"{num_train}_{num_test}"
    attacks = CACHE_ROOT / f"attacks_{num_train}_{num_test}"
    for d in (results, attacks):
        d.mkdir(parents=True, exist_ok=True)
    return results, attacks


# -- Distinctness guard ---------------------------------------------------
# Copied from run_great_audit.py / run_aim_audit.py rather than imported, for the reason
# stated there: importing those modules would run their logging.basicConfig and append
# this run's lines to their audits' logs.

def dataset_hash(dataset) -> str:
    """Fast content hash; pd.util.hash_pandas_object beats to_csv by a wide margin."""
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model) -> dict:
    """Abort if the memoised simulations are not essentially all distinct.

    The regression test for the seeding fix in dp2stage_generator.py: ExactDataKnowledge
    hands every simulation the same background, so a wrapper that forgot to vary the seed
    would reproduce the pre-2026-08-23 Synthcity collapse -- all D+ simulations one
    identical dataset, all D- another, an audit whose effective sample size is 2.
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
            f"DP-2Stage simulations are not independent draws -- {'; '.join(failures)}.\n"
            f"  This is the Synthcity collapse bug's shape. Check that "
            f"DP2StageGenerator passes seed_base + fit index as BOTH --seed and --gen-seed to "
            f"the driver, that the counter survives __getstate__, and that the cached pool "
            f"was not grown by an earlier build that lacked the fix.\n"
            f"  If the wrapper is correct, delete {CACHE_ROOT} and re-fit."
        )
    return fractions


# -- Formal-privacy record ------------------------------------------------

def read_fit_log(path: Path) -> pd.DataFrame:
    """The per-fit privacy records, one row per fit_index (last write wins: a restart
    that repeats an uncheckpointed chunk appends the same indices again)."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
    return df.drop_duplicates("fit_index", keep="last").sort_values("fit_index").reset_index(drop=True)


def privacy_summary(fits: pd.DataFrame) -> dict:
    """Formal epsilon as actually spent, across every fit in the pool."""
    if fits.empty:
        return {"n_fits_logged": 0}
    eps = fits["epsilon_spent"].astype(float)
    return {
        "n_fits_logged": int(len(fits)),
        "epsilon_spent_max": float(eps.max()), "epsilon_spent_min": float(eps.min()),
        "epsilon_spent_mean": float(eps.mean()),
        "noise_multiplier_mean": float(fits["noise_multiplier"].astype(float).mean()),
        "noise_multiplier_min": float(fits["noise_multiplier"].astype(float).min()),
        "noise_multiplier_max": float(fits["noise_multiplier"].astype(float).max()),
        "dp_steps_distinct": sorted({int(s) for s in fits["dp_steps"].dropna()}),
        "fit_s_mean": float(fits["fit_s"].astype(float).mean()),
        "gen_s_mean": float(fits["gen_s"].astype(float).mean()),
        "acceptance_mean": float(fits["acceptance"].astype(float).mean()),
        "peak_gpu_gb_fit_max": float(fits["peak_gpu_gb_fit"].astype(float).max()),
        "within_budget": bool(eps.max() <= FORMAL_EPSILON + EPS_TOLERANCE),
    }


# -- Pool growth, in capped chunks ----------------------------------------

def grow_pools(threat_model, num_train: int, num_test: int) -> int:
    """Grow both pools toward their targets, checkpointing every CHECKPOINT_EVERY fits
    and stopping after MAX_NEW_FITS new fits in this process.

    TAPAS's memoisation makes the restart free -- _generate_samples only ever generates
    the shortfall, so a restarted process resumes exactly where the last one stopped.

    Returns the remaining fit budget: <= 0 means this process stopped early and the
    caller should exit EXIT_INCOMPLETE so the restart loop starts a fresh one.
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

def build_world(sample_batch: int = AUDIT_SAMPLE_BATCH, fit_log_path=FIT_LOG):
    """The fixed audit world: scalers, background, target/alternate, generator.

    Identical to what common.run_method builds for the other methods, so the DP-2Stage
    row is comparable to theirs cell for cell.
    """
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    scalers = common._fit_scalers(pd.read_csv(TRAIN_CSV))
    generator = DP2StageGenerator(description, scalers, fit_log_path=fit_log_path,
                                  sample_batch=sample_batch)
    return description, background, target, alternate, generator


def build_or_load_threat_model(background, target, alternate, generator):
    """common.build_or_load_threat_model, but constructing a DP2StageGenerator.

    Not a call into that function: it hardcodes SynthcityGenerator. Everything else --
    SwapTargetedMIA, ExactDataKnowledge, BlackBoxKnowledge, the cache round trip -- is
    the same objects it uses.
    """
    cache_path = CACHE_ROOT / "threat_model"
    if (CACHE_ROOT / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info("Building new threat model for dp2stage (no cache found)")
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

def probe(n_fits: int, sample_batch: int) -> int:
    """Time real fit+generate cycles on the 500-row background, then exit.

    What comes out gates the full run:

      s/fit              the whole schedule turns on it. It includes what the driver's own
                         fit_s/gen_s do not: process start-up and the Stage 1 checkpoint
                         load, paid on EVERY fit because each fit is its own process.
      spent epsilon      per fit, from Opacus's RDP accountant. Must be <= 1.0 (+0.01
                         solver tolerance) -- the wrapper already refuses a fit above it.
      acceptance         fraction of sampled rows surviving the public-vocabulary and
                         integer filters. Low means the model is not producing the format.
      distinct outputs   consecutive fits use consecutive seeds, so they must produce
                         different data. If this is not n_fits/n_fits the seeding is not
                         working and the audit would be void.

    Nothing is cached, and no privacy record is written.
    """
    _, background, target, _, generator = build_world(sample_batch, fit_log_path=None)
    member = background.copy()
    member.add_records(target, in_place=True)
    log.info(f"=== probe: {n_fits} fit+generate cycles on {len(member.data)} rows, "
             f"eps={EPSILON}, delta={DELTA}, max_new_tokens={MAX_NEW_TOKENS}, "
             f"sample_batch={sample_batch} (PROBE-DEPENDENT) ===")

    times, hashes, infos = [], [], []
    for i in range(n_fits):
        t0 = time.time()
        generator.fit(member)
        synthetic = generator.generate(NUM_SYNTHETIC)
        dt = time.time() - t0
        info = generator.last_info
        times.append(dt)
        infos.append(info)
        hashes.append(dataset_hash(synthetic))
        log.info(f"  fit {i}: {dt:.1f}s wall  (driver: fit {info['fit_s']}s + gen {info['gen_s']}s; "
                 f"overhead {dt - info['fit_s'] - info['gen_s']:.1f}s)  "
                 f"eps_spent={info['epsilon_spent']:.4f} sigma={info['noise_multiplier']:.3f} "
                 f"steps={info['dp_steps']}  acceptance {info['acceptance']}  "
                 f"peak {info['peak_gpu_gb_fit']}/{info['peak_gpu_gb_gen']} GB (torch-allocated)  "
                 f"{len(synthetic.data.drop_duplicates())}/{len(synthetic.data)} unique rows")

    mean = float(np.mean(times))
    n_distinct = len(set(hashes))
    eps_max = max(i["epsilon_spent"] for i in infos)
    log.info(f"=== mean {mean:.1f}s/fit (min {min(times):.1f}, max {max(times):.1f}), "
             f"mean acceptance {np.mean([i['acceptance'] for i in infos]):.1%}, "
             f"max spent eps {eps_max:.4f} "
             + ("(within budget) ===" if eps_max <= FORMAL_EPSILON + EPS_TOLERANCE
                else "<-- OVER BUDGET ==="))
    log.info(f"=== {n_distinct}/{n_fits} distinct outputs across consecutive seeds "
             + ("(seeding OK) ===" if n_distinct == n_fits else
                "<-- SEEDING BROKEN: identical fits mean the audit would measure an "
                "effective sample size of 2. Do not run the full audit. ==="))
    for nt, nte in ((200, 500), (500, 1000), (1000, 2500)):
        log.info(f"    {nt}/{nte} = {nt + nte} fits -> {(nt + nte) * mean / 3600:.1f} h")
    log.info(f"    PROBE-DEPENDENT: set CHECKPOINT_EVERY ~= {max(5, round(1800 / mean))} "
             f"(~30 min of fits) and MAX_NEW_FITS to a small multiple; both are still "
             f"GReaT's provisional {CHECKPOINT_EVERY}/{MAX_NEW_FITS}.")
    ok = n_distinct == n_fits and eps_max <= FORMAL_EPSILON + EPS_TOLERANCE
    return 0 if ok else 1


# -- Audit ----------------------------------------------------------------

def run_audit(num_train: int, num_test: int, sample_batch: int) -> int:
    log.info(f"=== TAPAS privacy audit: dp2stage (dp={DP2STAGE_CONFIG['dp']}, "
             f"kind={DP2STAGE_CONFIG['kind']}, formal eps={FORMAL_EPSILON}, delta={DELTA}, "
             f"num_train={num_train}, num_test={num_test}) ===")
    log.info(f"    cache {CACHE_ROOT.relative_to(REPO_ROOT)}   "
             f"results {RESULTS_ROOT.relative_to(REPO_ROOT)}   "
             f"total fits = {num_train + num_test}   sample_batch={sample_batch}")

    results_dir, attack_cache = stage_dirs(num_train, num_test)
    _, background, target, alternate, generator = build_world(sample_batch)
    threat_model = build_or_load_threat_model(background, target, alternate, generator)

    # Same seed before build_attacks as every other method in the benchmark, so DP-2Stage
    # is probed by the same forests and the same 1500 random queries.
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
                 f"exit {EXIT_INCOMPLETE} so the restart loop starts a fresh process "
                 f"(see MAX_NEW_FITS). ===")
        return EXIT_INCOMPLETE

    # Guard BEFORE the attacks, so a broken pool is caught before hours of scoring.
    fractions = assert_distinct(threat_model)

    fits = read_fit_log(FIT_LOG)
    priv = privacy_summary(fits)
    log.info(f"    formal privacy over {priv.get('n_fits_logged', 0)} logged fits: "
             + (f"spent eps max {priv['epsilon_spent_max']:.4f} (target {FORMAL_EPSILON}), "
                f"sigma mean {priv['noise_multiplier_mean']:.3f}, "
                f"steps {priv['dp_steps_distinct']}, "
                f"{'WITHIN budget' if priv['within_budget'] else 'OVER BUDGET'}"
                if priv.get("n_fits_logged") else "none logged"))
    if priv.get("n_fits_logged") and priv["n_fits_logged"] < num_train + num_test:
        log.warning(f"    fit log has {priv['n_fits_logged']} of {num_train + num_test} fits: "
                    f"a pool grown before the log existed, or a cache from an earlier "
                    f"stage. The per-fit guard still ran on every fit at generation time.")

    rows, score_rows, no_scores = [], [], []
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=attack_cache, results_dir=results_dir,
        )
        result.update(method=METHOD, dp=DP2STAGE_CONFIG["dp"], kind=DP2STAGE_CONFIG["kind"],
                      num_train=num_train, num_test=num_test,
                      formal_epsilon=FORMAL_EPSILON, delta=DELTA)
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
    if not fits.empty:
        fits.to_json(results_dir / f"fit_log_{METHOD}_{num_train}_{num_test}.jsonl",
                     orient="records", lines=True)

    meta = {
        "method": METHOD, "variant": "DP-2Stage-O",
        "formal_epsilon": FORMAL_EPSILON, "delta": DELTA,
        "privacy_spent": priv,
        "num_train": num_train, "num_test": num_test,
        "num_synthetic": NUM_SYNTHETIC,
        "sample_batch": sample_batch, "max_new_tokens": MAX_NEW_TOKENS,
        "pool_wall_clock_s": pool_s,
        "attack_wall_clock_s": float(out["wall_time_s"].sum()),
        "new_fits_this_run": n_after - n_before,
        "distinct_fractions": fractions,
        "attacks_without_raw_scores": no_scores,
        "delta_note": f"delta={DELTA} is DP-2Stage's own; AIM uses 1e-9, DP-CTGAN "
                      f"1/(n*sqrt(n)), DPGAN 1/n, so eps=1 is not the same guarantee across "
                      f"generators (limitation).",
        "note": "pool_wall_clock_s covers this invocation only; see new_fits_this_run "
                "to tell a fresh run from a resumed one.",
    }
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] "
                 f"via {best['attack']}  (formal eps {FORMAL_EPSILON}) ===")
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
                             f"fresh one (default {MAX_NEW_FITS}, PROBE-DEPENDENT; see grow_pools)")
    parser.add_argument("--sample-batch", type=int, default=AUDIT_SAMPLE_BATCH,
                        help=f"rows per model.generate() call inside each fit "
                             f"(default {AUDIT_SAMPLE_BATCH}, PROBE-DEPENDENT)")
    args = parser.parse_args()

    MAX_NEW_FITS = args.max_new_fits
    CHECKPOINT_EVERY = min(CHECKPOINT_EVERY, MAX_NEW_FITS)

    if args.probe:
        return probe(args.probe, args.sample_batch)
    try:
        return run_audit(args.num_train, args.num_test, args.sample_batch)
    except DegeneratePool as exc:
        log.error(f"DISTINCTNESS GUARD FAILED:\n{exc}")
        return 1
    except HardSampleFailure as exc:
        # Transient (GPU contention, most likely) -- NOT one of the restart loop's 3
        # consecutive-failure strikes. The pool is checkpointed as of the last completed
        # CHECKPOINT_EVERY chunk (grow_pools saves after each), so at most one partial
        # chunk of fits is repeated. See HardSampleFailure in dp2stage_generator.py.
        log.warning(f"HARD SAMPLE FAILURE (transient, restarting): {exc}")
        return EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
