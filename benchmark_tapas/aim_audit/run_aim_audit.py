#!/usr/bin/env python3
"""TAPAS MIA audit of AIM at num_train=1000 / num_test=2500.

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
    run_eps_sweep.py saves the threat model only after BOTH pools finish. For
    DPGAN that risked 8 h of unsaved work; here the test pool alone is ~8.7 h, so
    a reboot late in the run would discard most of the night. This script saves
    after the training pool as well, capping the loss at whichever pool is in
    flight. Both pools are resumable on re-run.

WHAT IS AND IS NOT HELD FIXED
    fixed:   background, target/alternate, attack battery + its internal
             randomness, num_synthetic (500 records per simulation), and AIM's
             entire configuration -- imported from sdg/aim.py, not restated
    varies:  nothing. This is a single-point audit, not a sweep.

    AIM runs at delta=1e-9 regardless of dataset size, while synthcity's DPGAN
    derives delta = 1/len(X) (= 2e-3 on a 500-row background) and PrivBayes is
    pure DP. So the three DP methods are audited at three different deltas. That
    is already true of the committed counts sweep; it is a limitation to report,
    not something this script should quietly "fix" by pinning a different value.

    AIM is not bit-reproducible at any seed (opendp CSPRNG). See aim_generator.py.

OUTPUTS
    cache/aim_audit/threat_model.pkl                 pools, resumable
    cache/aim_audit/datasets/synthetic_{split}.csv.gz  every simulation, gzipped
    cache/aim_audit/attacks/result_*.json            per-attack cache (resumable)
    results/aim_audit/effeps_aim_200_500.csv         per-attack metrics
    results/aim_audit/effective_epsilon_*.csv        TAPAS's own per-attack report
    results/aim_audit/raw_scores_aim_200_500.csv     pre-threshold scores
    results/aim_audit/meta.json                      timings + guard fractions

Run from the repo root, env active:
  python benchmark_tapas/aim_audit/run_aim_audit.py --probe 6                  # time 6 fits, exit
  python benchmark_tapas/aim_audit/run_aim_audit.py --num-train 200 --num-test 500
  python benchmark_tapas/aim_audit/run_aim_audit.py                            # 1000/2500, ~12 h
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

CACHE_ROOT = CACHE_DIR / "aim_audit"
ATTACK_CACHE = CACHE_ROOT / "attacks"
RESULTS_ROOT = RESULTS_DIR / "aim_audit"
for d in (CACHE_ROOT, ATTACK_CACHE, RESULTS_ROOT):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RESULTS_ROOT / "aim_audit_log.txt"),
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


# -- Shared setup ---------------------------------------------------------

def build_world():
    """The fixed audit world: scalers, background, target/alternate, generator.

    Identical to what common.run_method builds for the other four methods, so the
    AIM row is comparable to theirs cell for cell.
    """
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)
    scalers = common._fit_scalers(pd.read_csv(TRAIN_CSV))
    generator = AIMGenerator(description, scalers, epsilon=EPSILON, delta=DELTA)
    return description, background, target, alternate, generator


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

def probe(n_fits: int) -> int:
    """Time real fit+generate cycles on the 500-row background, then exit.

    This is the number the whole schedule turns on: 700 fits at the measured
    s/fit is the audit's cost. Nothing is cached or written.
    """
    _, background, target, _, generator = build_world()
    member = background.copy()
    member.add_records(target, in_place=True)
    log.info(f"=== probe: {n_fits} fit+generate cycles on {len(member.data)} rows ===")

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

def run_audit(num_train: int, num_test: int) -> int:
    log.info(f"=== TAPAS privacy audit: aim (dp={AIM_CONFIG['dp']}, "
             f"kind={AIM_CONFIG['kind']}, eps={EPSILON}, delta={DELTA}, "
             f"num_train={num_train}, num_test={num_test}) ===")
    log.info(f"    cache {CACHE_ROOT.relative_to(REPO_ROOT)}   "
             f"total fits = {num_train + num_test}")

    _, background, target, alternate, generator = build_world()
    threat_model = build_or_load_threat_model(background, target, alternate, generator)

    # Same seed before build_attacks as every other method in the benchmark, so
    # AIM is probed by the same forests and the same 1500 random queries.
    np.random.seed(SCORE_ATTACK_SEED)
    attacks = common.build_attacks(target, background)

    # Two checkpoints rather than one: at AIM's per-fit cost the test pool alone
    # is most of a night, and an unsaved crash there would cost the whole run.
    t_pool = time.time()
    n_before = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])

    threat_model.generate_training_samples(num_train)
    threat_model.save(str(CACHE_ROOT / "threat_model"))
    log.info(f"    train pool done: {len(threat_model._memory[True][0])} datasets, "
             f"checkpointed ({time.time() - t_pool:.0f}s)")

    threat_model._generate_samples(num_test, training=False)
    threat_model.save(str(CACHE_ROOT / "threat_model"))
    pool_s = round(time.time() - t_pool, 1)
    n_after = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
    log.info(f"    pools: {len(threat_model._memory[True][0])} train / "
             f"{len(threat_model._memory[False][0])} test  "
             f"(+{n_after - n_before} new fits, {pool_s:.0f}s)")

    # Guard BEFORE the attacks, so a broken pool is caught before hours of scoring.
    fractions = assert_distinct(threat_model)
    export_pools(threat_model)

    rows, score_rows, no_scores = [], [], []
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=ATTACK_CACHE, results_dir=RESULTS_ROOT,
        )
        result.update(method=METHOD, dp=AIM_CONFIG["dp"], kind=AIM_CONFIG["kind"],
                      num_train=num_train, num_test=num_test,
                      formal_epsilon=EPSILON, delta=DELTA)
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
    out.to_csv(RESULTS_ROOT / f"effeps_{METHOD}_{num_train}_{num_test}.csv", index=False)
    if score_rows:
        path = RESULTS_ROOT / f"raw_scores_{METHOD}_{num_train}_{num_test}.csv"
        pd.DataFrame(score_rows).to_csv(path, index=False)
        log.info(f"    wrote {len(score_rows)} raw scores -> {path.name}")
    if no_scores:
        log.warning(f"    no raw scores for {no_scores} -- served from the per-attack "
                    f"JSON cache, which stores aggregates only. Delete those JSONs to "
                    f"recompute (cheap: the fits are memoised).")

    meta = {
        "method": METHOD, "formal_epsilon": EPSILON, "delta": DELTA,
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
    (RESULTS_ROOT / "meta.json").write_text(json.dumps(meta, indent=2))

    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] "
                 f"via {best['attack']} ===")
    else:
        log.warning(f"=== done, but no usable eps_low_95 across the 5 attacks. "
                    f"Results are written; inspect {RESULTS_ROOT.relative_to(REPO_ROOT)} ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=int, metavar="N",
                        help="time N real fit+generate cycles and exit (no caching)")
    parser.add_argument("--num-train", type=int, default=NUM_TRAIN)
    parser.add_argument("--num-test", type=int, default=NUM_TEST)
    args = parser.parse_args()

    if args.probe:
        return probe(args.probe)
    try:
        return run_audit(args.num_train, args.num_test)
    except DegeneratePool as exc:
        log.error(f"DISTINCTNESS GUARD FAILED:\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
