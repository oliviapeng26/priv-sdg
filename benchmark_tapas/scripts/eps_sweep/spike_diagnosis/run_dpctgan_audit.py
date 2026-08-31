#!/usr/bin/env python3
"""[3b] TAPAS MIA audit of SmartNoise DP-CTGAN at eps = 1.0, num_train/num_test = 1000/2500.

THE QUESTION
    Does a DIFFERENT library's DP-SGD-on-CTGAN show the same eps=1.0 spike that
    synthcity's DPGAN does? synthcity reported eps_eff = 1.761 [1.761, 2.780] via
    Groundhog at these counts. This runs the identical audit against an independent
    implementation of the same idea.

        same spike  -> the effect belongs to the method, and the finding generalises
        no spike    -> synthcity's implementation is implicated, and the claim becomes
                       "this library's DP-SGD generator" rather than "DP-SGD generators"

    Either answer is publishable; they are different papers.

THE TAPAS SETUP IS UNCHANGED
    Everything load-bearing is reused from benchmark_tapas/common.py: the same fixed
    background (TAPAS_BG_SEED), the same target/alternate (TAPAS_TARGET_SEED), the
    same 5-attack battery built under the same SCORE_ATTACK_SEED, the same
    SwapTargetedMIA / ExactDataKnowledge / BlackBoxKnowledge construction, the same
    500 records per simulation, the same per-attack caches and raw-score export, the
    same distinctness guard, the same 1000/2500 counts. Only the generator differs.

    This is run_aim_audit.py with the generator swapped -- deliberately, since that
    script already solved the "audit a SmartNoise synthesiser through TAPAS" problem.

RUN THE PROBE FIRST. IT IS NOT OPTIONAL HERE.
    DP-CTGAN treats eps as a stopping rule at fixed sigma=5, and its default
    batch_size of 500 equals the audit background, so the sampling rate is 1.0 with
    no amplification from subsampling. The budget may be exhausted within a handful
    of epochs, and the library breaks out of training SILENTLY rather than raising.

    --probe fits once and reports how many of the 300 epochs actually ran plus the
    accountant's final spend. If that comes back as two or three epochs, the audit
    would be measuring an essentially untrained network -- still a real result about
    the library, but you want to know before spending eight hours, and you want it in
    the write-up either way.

CHUNKED, LIKE THE AIM AUDIT
    Pools are grown in CHECKPOINT_EVERY-fit chunks with a save after each, and each
    process stops after MAX_NEW_FITS. The AIM audit needed that because jax exhausted
    its compilation mappings after ~112 fits; torch has no equivalent failure, but
    the pattern costs nothing and caps the loss from any crash at one chunk. Run it
    in the restart loop below.

OUTPUTS
    cache/dpctgan_eps1/threat_model.pkl                       pool, resumable
    cache/dpctgan_eps1/datasets/synthetic_{split}.csv.gz      every simulation
    cache/dpctgan_eps1/attacks/result_*.json                  per-attack cache
    results/eps_sweep/spike_diagnosis/dpctgan/effeps_dpctgan_1000_2500.csv
    results/eps_sweep/spike_diagnosis/dpctgan/raw_scores_dpctgan_1000_2500.csv
    results/eps_sweep/spike_diagnosis/dpctgan/effective_epsilon_*.csv
    results/eps_sweep/spike_diagnosis/dpctgan/meta.json
    results/eps_sweep/spike_diagnosis/dpctgan/dpctgan_audit_log.txt

Run from the repo root, env active:
  python benchmark_tapas/scripts/eps_sweep/spike_diagnosis/run_dpctgan_audit.py --probe 3

  screen -S dpctgan
  cd ~/priv-sdg && source venv/bin/activate
  run_stage () {
    fails=0
    while true; do
      python -u benchmark_tapas/scripts/eps_sweep/spike_diagnosis/run_dpctgan_audit.py
      rc=$?
      [ $rc -eq 0 ] && return 0
      if [ $rc -eq 3 ]; then fails=0; continue; fi
      fails=$((fails + 1)); echo "FAILED: exit $rc (consecutive failure $fails)"
      [ $fails -ge 3 ] && return $rc
    done
  }
  run_stage 2>&1 | tee ~/dpctgan_audit.out
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

import tapas.threat_models as tm                                 # noqa: E402
import common                                                    # noqa: E402
from dpctgan_generator import (DPCTGANGenerator, EPSILON, SIGMA,  # noqa: E402
                               EPOCHS, BATCH_SIZE)
from config import CACHE_DIR, RESULTS_DIR, TRAIN_CSV, NUM_SYNTHETIC  # noqa: E402
from seeds import SCORE_ATTACK_SEED                              # noqa: E402

METHOD = "dpctgan"
NUM_TRAIN, NUM_TEST = 1000, 2500

# Not in config.METHOD_CONFIG: DP-CTGAN is not a Synthcity plugin, and config.py is
# left exactly as every committed result was produced under.
DPCTGAN_CONFIG = {"dp": True, "kind": "neural"}

MAX_NEW_FITS = 400        # fits per process; a crash costs at most this many
CHECKPOINT_EVERY = 100
EXIT_INCOMPLETE = 3       # "pools not finished, restart me" -- not an error

CACHE_ROOT = CACHE_DIR / "dpctgan_eps1"
ATTACK_CACHE = CACHE_ROOT / "attacks"
RESULTS_ROOT = RESULTS_DIR / "eps_sweep" / "spike_diagnosis" / "dpctgan"
for d in (CACHE_ROOT, ATTACK_CACHE, RESULTS_ROOT):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RESULTS_ROOT / "dpctgan_audit_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("dpctgan_audit")


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws."""


# -- Copied from run_aim_audit.py, for the reason stated there -----------
# Verbatim rather than imported: importing that module would run its
# logging.basicConfig and append this run's lines to the AIM audit's log.

def dataset_hash(dataset) -> str:
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model) -> dict:
    """Abort if the memoised simulations are not essentially all distinct.

    Especially worth keeping here: if DP-CTGAN's budget is exhausted after one epoch,
    the generator may emit a near-constant table, and this catches that before hours
    of attack time rather than after.
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


def export_pools(threat_model) -> None:
    """Every generated dataset as gzipped CSV, same format as the other audits."""
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

def build_world(cuda: bool):
    """The fixed audit world, identical to every other method's."""
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    scalers = common._fit_scalers(pd.read_csv(TRAIN_CSV))
    generator = DPCTGANGenerator(description, scalers, epsilon=EPSILON, cuda=cuda)
    return description, background, target, alternate, generator


def build_or_load_threat_model(background, target, alternate, generator):
    """common.build_or_load_threat_model, but constructing a DPCTGANGenerator."""
    cache_path = CACHE_ROOT / "threat_model"
    if (CACHE_ROOT / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info("Building new threat model for dpctgan (no cache found)")
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
    """Grow both pools in chunks, checkpointing, stopping after MAX_NEW_FITS."""
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

def probe(n_fits: int, cuda: bool) -> int:
    """Fit a few times on the real audit background and report what DP happened.

    The number that matters is epochs_run: DP-CTGAN breaks out of training silently
    once the budget is spent, so a small number here means the audit would be
    measuring an untrained network.
    """
    _, background, target, _, generator = build_world(cuda)
    member = background.copy()
    member.add_records(target, in_place=True)
    log.info(f"=== probe: {n_fits} fits on {len(member.data)} rows, "
             f"eps={EPSILON}, sigma={SIGMA}, batch_size={BATCH_SIZE}, "
             f"epochs cap={EPOCHS} ===")
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
                 f"{generator.last_epochs_run}/{EPOCHS} | eps spent "
                 f"{generator.last_epsilon_spent} | {uniq}/{NUM_SYNTHETIC} unique rows")

    mean = float(np.mean(times))
    log.info(f"=== mean {mean:.1f}s/fit -> {NUM_TRAIN + NUM_TEST} fits = "
             f"{(NUM_TRAIN + NUM_TEST) * mean / 3600:.1f} h ===")
    if generator.last_epochs_run is not None and generator.last_epochs_run <= 5:
        log.warning(f"    only {generator.last_epochs_run} epochs survived the budget. "
                    f"The audit would measure a barely-trained network. That is a real "
                    f"finding about the library, but decide before committing the run.")
    return 0


# -- Audit ----------------------------------------------------------------

def run_audit(num_train: int, num_test: int, cuda: bool) -> int:
    log.info(f"=== TAPAS privacy audit: dpctgan (SmartNoise) eps={EPSILON}, "
             f"sigma={SIGMA}, epochs<={EPOCHS}, batch={BATCH_SIZE}, "
             f"num_train={num_train}, num_test={num_test} ===")
    log.info(f"    reference: synthcity DPGAN at eps=1.0 and these counts scored "
             f"eps_low_95 = 1.761 [1.761, 2.780] via Groundhog")

    _, background, target, alternate, generator = build_world(cuda)
    threat_model = build_or_load_threat_model(background, target, alternate, generator)

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
        log.info(f"=== fit budget for this process spent; pools checkpointed. "
                 f"Exit {EXIT_INCOMPLETE} so the wrapper starts a fresh process. ===")
        return EXIT_INCOMPLETE

    fractions = assert_distinct(threat_model)
    export_pools(threat_model)

    rows, score_rows, no_scores = [], [], []
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=ATTACK_CACHE, results_dir=RESULTS_ROOT,
        )
        result.update(method=METHOD, dp=DPCTGAN_CONFIG["dp"], kind=DPCTGAN_CONFIG["kind"],
                      num_train=num_train, num_test=num_test,
                      formal_epsilon=EPSILON, sigma=SIGMA,
                      epochs_cap=EPOCHS, batch_size=BATCH_SIZE,
                      library="smartnoise")
        result["tn"] = 1.0 - result["fp"]
        result["fn"] = 1.0 - result["tp"]
        if summary is not None:
            score_rows.extend(common._score_rows(METHOD, attack, summary))
        else:
            no_scores.append(attack.label)
        rows.append(result)
        threat_model.save(str(CACHE_ROOT / "threat_model"))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_ROOT / f"effeps_{METHOD}_{num_train}_{num_test}.csv", index=False)
    if score_rows:
        path = RESULTS_ROOT / f"raw_scores_{METHOD}_{num_train}_{num_test}.csv"
        pd.DataFrame(score_rows).to_csv(path, index=False)
        log.info(f"    wrote {len(score_rows)} raw scores -> {path.name}")

    meta = {
        "method": METHOD, "library": "smartnoise", "formal_epsilon": EPSILON,
        "sigma": SIGMA, "epochs_cap": EPOCHS, "batch_size": BATCH_SIZE,
        "num_train": num_train, "num_test": num_test, "num_synthetic": NUM_SYNTHETIC,
        "last_epochs_run": generator.last_epochs_run,
        "last_epsilon_spent": generator.last_epsilon_spent,
        "pool_wall_clock_s": pool_s, "new_fits_this_run": n_after - n_before,
        "attack_wall_clock_s": float(out["wall_time_s"].sum()),
        "distinct_fractions": fractions,
        "attacks_without_raw_scores": no_scores,
        "note": "epsilon is a STOPPING RULE in SmartNoise (fixed sigma, break when the "
                "accountant says the budget is spent), not a noise calibration as in "
                "synthcity. delta is derived internally as 1/(n*sqrt(n)). See "
                "dpctgan_generator.py.",
    }
    (RESULTS_ROOT / "meta.json").write_text(json.dumps(meta, indent=2))

    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] "
                 f"via {best['attack']}   (synthcity DPGAN: 1.761 [1.761, 2.780]) ===")
    else:
        log.warning("=== done, but no usable eps_low_95 across the 5 attacks ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=int, metavar="N",
                    help="fit N times, report epochs run and eps spent, then exit")
    ap.add_argument("--num-train", type=int, default=NUM_TRAIN)
    ap.add_argument("--num-test", type=int, default=NUM_TEST)
    args = ap.parse_args()

    import torch
    cuda = torch.cuda.is_available()
    log.info(f"device: {'cuda' if cuda else 'cpu'} "
             f"(torch.cuda.is_available()={cuda})")

    if args.probe:
        return probe(args.probe, cuda)
    try:
        return run_audit(args.num_train, args.num_test, cuda)
    except DegeneratePool as exc:
        log.error(f"DISTINCTNESS GUARD FAILED:\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
