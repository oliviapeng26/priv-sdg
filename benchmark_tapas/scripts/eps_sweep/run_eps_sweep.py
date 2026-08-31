#!/usr/bin/env python3
"""Phase 2 of the formal-epsilon sweep: TAPAS MIA audit of DPGAN at each eps.

Structurally this is run_counts_sweep.py with the loop swapped: instead of
iterating over (num_train, num_test) at a fixed eps=1.0, it iterates over eps at
a fixed 1000/2500. Everything load-bearing carries over unchanged -- the same
fixed background/target/alternate, the same 5-attack battery built under the same
SCORE_ATTACK_SEED, the same per-fit generator seeding, the same distinctness
guard, the same resumable per-attack JSON caches, the same raw-score CSVs.

WHY EACH EPS COSTS A FULL 3,500 FITS
    The counts sweep was cheap because its stages are nested: TAPAS appends to its
    memoised pool (attacker_knowledge.py:591 generates only the shortfall), so the
    whole curve costs max(num_train)+max(num_test) fits. That does NOT apply here.
    A different eps is a different mechanism -- Opacus derives a different noise
    multiplier -- so the synthetic pool at eps=10 tells you nothing about eps=100
    and cannot be reused. Each eps therefore needs its own threat model, its own
    cache directory, and its own 3,500 fits.

    At the 4.4 s/fit measured for DPGAN at 500 rows that is ~4.3 h per eps, so
    ~13 h for the three new values. One night.

WHY EPS=1.0 IS NOT IN THE DEFAULT LIST
    It has already been audited at exactly these counts, under exactly this threat
    model, with correct per-fit seeding:

        results/counts_sweep/dpgan/1000_2500/effeps_dpgan_1000_2500.csv
        results/counts_sweep/raw_scores_dpgan_1000_2500.csv
        cache/dpgan/threat_model.pkl

    Re-running it would spend 4.3 h reproducing a number the repo already has
    (worst-case eps_low_95 = 1.761 via Groundhog). eps_sweep_aggregate.py reads
    that arm straight from counts_sweep. Passing --epsilons 1.0 is allowed but
    warns loudly, because it builds a SEPARATE cache under cache/dpgan_eps1/
    rather than reusing cache/dpgan/.

WHAT IS AND IS NOT HELD FIXED ACROSS EPS
    fixed:   background (TAPAS_BG_SEED), target/alternate (TAPAS_TARGET_SEED),
             attack battery + its internal randomness (SCORE_ATTACK_SEED),
             LocalNeighbourhood radius (a function of target+background only),
             per-fit generator seeds (TAPAS_GENERATOR_SEED_BASE + i),
             n_iter, batch_size, every other dpgan default, num_train/num_test
    varies:  epsilon -- and, through it, the Opacus noise multiplier sigma

    NOT fixed, and this is a limitation rather than a choice: delta. synthcity
    sets dp_delta = 1/len(X) (gan.py:590-591) from whatever frame it is handed, so
    the audited 500-row world runs at delta ~2.5e-3 while the utility arm runs at
    ~5.8e-5. The audit and the utility arm are therefore not the same (eps, delta)
    pair at any eps. Left unpinned deliberately: pinning it would make this sweep a
    different mechanism from every other dpgan number in the repo. See
    eps_sweep_sigma_check.py. Also note gan.py:607 passes poisson_sampling=False
    while the accountant that produced sigma assumes Poisson subsampling, so the
    formal guarantee is an approximation throughout.

    eps is reported as passed to synthcity. Opacus accounts under add/remove,
    while this threat model swaps one record for another, so the comparable
    replacement-relation budget is 2*eps -- carried as a separate column rather
    than folded into the reported eps.

DISTINCTNESS GUARD, AND WHY IT IS PER-EPS RECOVERABLE
    Inherited verbatim from run_counts_sweep.py: every memoised dataset is hashed
    after the pools grow, and the run aborts if they are not ~all distinct. That is
    the check that would have caught the pre-2026-08-23 seeding bug immediately.

    Two changes for this sweep. First, a guard failure aborts only ITS eps: the
    remaining budgets still run, so one degenerate arm cannot cost the whole night.
    Second, the failure is genuinely ambiguous here in a way it was not in the
    counts sweep. At eps=0.1 the audit runs at sigma ~80 against a clip norm of 2,
    and a generator that has collapsed to a near-constant output is a FINDING about
    DP at that budget, not a seeding regression -- but it looks identical to the
    bug. So the guard reports the distinct fraction either way, refuses to write
    results by default, and --allow-degenerate records the arm with a
    `degenerate_pool` flag if you decide the collapse is real.

OUTPUTS
    cache/dpgan_eps{eps}/threat_model.pkl           per-eps, never shared
    cache/dpgan_eps{eps}/datasets/synthetic_{split}.csv.gz
                                                    every generated dataset, gzipped
    cache/dpgan_eps{eps}/sweep_1000_2500/result_*.json
                                                    per-attack cache (resumable)
    results/eps_sweep/privacy/eps{eps}/effeps_dpgan_eps{eps}.csv
    results/eps_sweep/privacy/eps{eps}/effective_epsilon_*.csv
    results/eps_sweep/privacy/eps{eps}/meta.json    timings + guard fractions
    results/eps_sweep/privacy/raw_scores_dpgan_eps{eps}.csv
                                                    raw pre-threshold scores

Run from the repo root, venv active:
  python benchmark_tapas/scripts/run_eps_sweep.py
  python benchmark_tapas/scripts/run_eps_sweep.py --epsilons 10 100
  python benchmark_tapas/scripts/run_eps_sweep.py --epsilons 0.1 --allow-degenerate

Resumable: finished attacks and memoised fits are skipped on re-run, so a crash
or a disconnect costs only the attack that was in flight.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# benchmark_tapas/, found by walking up to config.py rather than counting parents --
# these scripts live in a subfolder now and may move again.
BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import common                                                    # noqa: E402
from tapas.report import EffectiveEpsilonReport                  # noqa: E402
from config import CACHE_DIR, RESULTS_DIR, METHOD_CONFIG, slug   # noqa: E402
from seeds import SCORE_ATTACK_SEED                              # noqa: E402

METHOD = "dpgan"
# 1.0 is reused from counts_sweep (see module docstring).
SWEEP_EPSILONS = [0.1, 10.0, 100.0]
# Chida et al. and TAPAS Experiment 2 both sit here; the counts sweep showed the
# eff-epsilon interval flattening beyond it.
NUM_TRAIN, NUM_TEST = 1000, 2500

SWEEP_DIR = RESULTS_DIR / "eps_sweep"
PRIVACY_DIR = SWEEP_DIR / "privacy"
PRIVACY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(PRIVACY_DIR / "eps_sweep_privacy_log.txt"),
              logging.StreamHandler()],
)
log = logging.getLogger("eps_sweep_privacy")


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws. Raised by assert_distinct."""


def eps_slug(eps: float) -> str:
    """Must match eps_sweep_generate.eps_slug: 0.1 -> 'eps0.1', 10.0 -> 'eps10'."""
    return f"eps{eps:g}"


# -- Copied from run_counts_sweep.py -------------------------------------
# Verbatim rather than imported: importing that module would run its
# logging.basicConfig and start appending this sweep's lines to
# results/counts_sweep/counts_sweep_log.txt, which is a committed artefact of a
# different experiment.

def dataset_hash(dataset) -> str:
    """Fast content hash. pd.util.hash_pandas_object beats to_csv by a wide margin,
    which matters when this runs over 3500 datasets."""
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model, label: str, allow_degenerate: bool) -> dict:
    """Abort if the memoised simulations are not essentially all distinct.

    Under correct per-fit seeding, duplicates should not occur at all: each dataset is
    500 rows sampled from a freshly fitted model. A handful of collisions could in
    principle happen for a very low-entropy generator, so the bar is 99% rather than
    100%. Returns the per-world distinct fractions so they can be recorded even when
    the guard passes -- at low eps the trend toward degeneracy is itself a result.
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

    if failures and not allow_degenerate:
        raise DegeneratePool(
            f"{label}: the simulations are not independent draws -- {'; '.join(failures)}.\n"
            f"  Two causes look identical here, and they need different responses:\n"
            f"    (a) the pre-2026-08-23 seeding bug returning. Check that "
            f"SynthcityGenerator._plugin_args() still passes a per-fit random_state and "
            f"that the fit counter is advancing. If so this is a BUG -- fix it, delete "
            f"cache/dpgan_{label}/, re-run.\n"
            f"    (b) DPGAN genuinely collapsing at this budget. At low eps the audit runs "
            f"at a very large noise multiplier against a clip norm of 2, and a generator "
            f"that outputs a near-constant table is a real property of DP at that eps. If "
            f"so this is a RESULT -- re-run with --allow-degenerate to record the arm with "
            f"a degenerate_pool flag.\n"
            f"  Distinguish them by checking whether the OTHER eps arms are distinct: a "
            f"seeding bug would flatten all of them, collapse would not.\n"
            f"  Refusing to write results for this eps."
        )
    if failures:
        log.warning(f"{label}: --allow-degenerate set, recording anyway. Pools: "
                    f"{'; '.join(failures)}")
    return fractions


def export_pools(threat_model, cache_dir: Path) -> None:
    """Write every generated synthetic dataset out in a portable form.

    gzipped CSV rather than parquet on purpose: pandas reads it anywhere with no extra
    dependency. Overwritten each call, so the file always reflects the full pool.
    """
    out_dir = cache_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The datasets are in TAPAS's representation: continuous columns min-max scaled to
    # [0,1] against the TRAINING split (common._fit_scalers), categoricals as strings.
    # Write the bounds so the export can be put back into original units:
    #   original = scaled * (max - min) + min
    scalers_path = out_dir / "scalers.csv"
    if not scalers_path.exists():
        from config import TRAIN_CSV
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
        log.info(f"    exported {len(datasets)} {name} datasets "
                 f"({len(datasets) * len(datasets[0].data):,} rows) -> {path.name} "
                 f"[{path.stat().st_size / 1e6:.1f} MB]")


# -- Selection threshold tau ---------------------------------------------

def selected_threshold(summary) -> dict:
    """Recover the (threshold, inverse) EffectiveEpsilonReport picked for this attack.

    run_attack's report chooses tau inside publish() via _select_attack and only
    prints it, so it never reaches the result CSV. Rebuilding the report with the
    SAME validation_split formula run_attack uses and calling _select_attack
    directly reproduces the choice exactly, at the cost of a few hundred
    Clopper-Pearson evaluations over the validation split -- well under a second.

    tau is not the `threshold` column of the raw-score CSV: that one is the attack's
    own decision cutoff (learned argmax(tpr-fpr), or 0.5 for the shadow-modelling
    attacks). tau is the report's separate re-thresholding of the same scores,
    chosen on a held-out 10% specifically to maximise the certifiable eps.
    `inverse` records whether it scored the negatives (TN/FN) instead.
    """
    validation_split = min(0.5, max(0.1, 15 / len(summary.scores)))
    report = EffectiveEpsilonReport([summary], validation_split=validation_split,
                                    confidence_levels=(0.9, 0.95, 0.99))
    selection = report._select_attack()
    if selection is None:
        return {"eff_eps_tau": None, "eff_eps_tau_inverse": None,
                "eff_eps_validation_split": validation_split}
    _, tau, inverse = selection
    return {"eff_eps_tau": float(tau), "eff_eps_tau_inverse": bool(inverse),
            "eff_eps_validation_split": float(validation_split)}


# -- Per-eps driver ------------------------------------------------------

def run_one_epsilon(epsilon: float, device_kwargs: dict, allow_degenerate: bool) -> None:
    label = eps_slug(epsilon)
    cfg = METHOD_CONFIG[METHOD]
    plugin_kwargs = {**cfg["plugin_kwargs"], **device_kwargs}
    cache_dir = CACHE_DIR / f"{METHOD}_{label}"
    stage_cache = cache_dir / f"sweep_{NUM_TRAIN}_{NUM_TEST}"
    results_dir = PRIVACY_DIR / label
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage_cache.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {METHOD} eps={epsilon:g} "
             f"(num_train={NUM_TRAIN}, num_test={NUM_TEST}, "
             f"plugin_kwargs={plugin_kwargs}) ===")
    log.info(f"    cache {cache_dir.relative_to(REPO_ROOT)}   "
             f"total fits = {NUM_TRAIN + NUM_TEST}")

    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)

    threat_model = common.build_or_load_threat_model(
        cache_dir=cache_dir, method=METHOD, background_dataset=background,
        target_record=target, alternate_record=alternate, description=description,
        epsilon=epsilon, plugin_kwargs=plugin_kwargs,
    )
    gen = threat_model.atk_know_gen.generator
    # A cache built at another eps would silently audit the wrong mechanism, and the
    # per-eps cache directory is the only thing preventing it. Verify rather than trust.
    cached_eps = getattr(gen, "epsilon", None)
    if cached_eps is not None and not np.isclose(float(cached_eps), epsilon):
        raise RuntimeError(
            f"{label}: cached threat model at {cache_dir} was built with "
            f"epsilon={cached_eps}, not {epsilon}. Delete that directory and re-run.")
    log.info(f"    resuming at fit {getattr(gen, '_fit_counter', 0)} "
             f"(next seed {gen.seed_base + gen._fit_counter}), generator eps={cached_eps}")

    # Same seed before build_attacks at every eps, so all arms are probed by the same
    # forests and the same 1500 random queries and the only thing changing is eps.
    # Attacks are stateful (they learn a threshold in train), so they are rebuilt per
    # eps rather than shared.
    np.random.seed(SCORE_ATTACK_SEED)
    attacks = common.build_attacks(target, background)

    t_pool = time.time()
    n_before = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
    threat_model.generate_training_samples(NUM_TRAIN)
    threat_model._generate_samples(NUM_TEST, training=False)
    n_after = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
    pool_s = round(time.time() - t_pool, 1)
    log.info(f"    pools: {len(threat_model._memory[True][0])} train / "
             f"{len(threat_model._memory[False][0])} test  "
             f"(+{n_after - n_before} new fits, {pool_s:.0f}s)")
    threat_model.save(str(cache_dir / "threat_model"))

    # Guard BEFORE hours of attack time, so a degenerate cache is caught immediately.
    fractions = assert_distinct(threat_model, label, allow_degenerate)
    degenerate = any(f < 0.99 for f in fractions.values())
    export_pools(threat_model, cache_dir)

    rows, score_rows, no_scores = [], [], []
    for attack in attacks:
        result, summary = common.run_attack(
            attack, threat_model, num_train=NUM_TRAIN, num_test=NUM_TEST,
            cache_dir=stage_cache, results_dir=results_dir,
        )
        result.update(method=METHOD, dp=cfg["dp"], kind=cfg["kind"],
                      num_train=NUM_TRAIN, num_test=NUM_TEST,
                      formal_epsilon=epsilon, replacement_epsilon=2 * epsilon,
                      degenerate_pool=degenerate)
        # TAPAS's MIAttackSummary exposes the positive rates only; the negatives are
        # their complements, and the report's `inverse` branch is scored on them.
        result["tn"] = 1.0 - result["fp"]
        result["fn"] = 1.0 - result["tp"]
        if summary is not None:
            result.update(selected_threshold(summary))
            score_rows.extend(common._score_rows(METHOD, attack, summary))
        else:
            no_scores.append(attack.label)
        rows.append(result)
        threat_model.save(str(cache_dir / "threat_model"))

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / f"effeps_{METHOD}_{label}.csv", index=False)
    if score_rows:
        path = PRIVACY_DIR / f"raw_scores_{METHOD}_{label}.csv"
        pd.DataFrame(score_rows).to_csv(path, index=False)
        log.info(f"    wrote {len(score_rows)} raw scores -> {path.name}")
    if no_scores:
        log.warning(f"    no raw scores for {no_scores} -- served from the per-attack JSON "
                    f"cache, which stores aggregates only. Delete those JSONs to recompute "
                    f"(cheap: the fits are memoised).")

    meta = {
        "formal_epsilon": epsilon, "replacement_epsilon": 2 * epsilon,
        "method": METHOD, "num_train": NUM_TRAIN, "num_test": NUM_TEST,
        "plugin_kwargs": {k: str(v) for k, v in plugin_kwargs.items()},
        "pool_wall_clock_s": pool_s,
        "attack_wall_clock_s": float(out["wall_time_s"].sum()),
        "total_wall_clock_s": pool_s + float(out["wall_time_s"].sum()),
        "new_fits_this_run": n_after - n_before,
        "distinct_fractions": fractions,
        "degenerate_pool": degenerate,
        "attacks_without_raw_scores": no_scores,
        # On a RESUMED run the pools already exist, so pool_wall_clock_s covers only
        # the fits done in this invocation -- read it together with new_fits_this_run
        # before quoting total_wall_clock_s as the cost of the arm.
        "note": "pool_wall_clock_s and total_wall_clock_s cover this invocation only; "
                "see new_fits_this_run to tell a fresh arm from a resumed one",
    }
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Defensive: the CSV and meta.json are already on disk by this point, so a
    # degenerate eps_low_95 column must not turn a finished 4-hour arm into a
    # traceback that main() records as a failure.
    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== eps={epsilon:g} done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] via {best['attack']} "
                 f"({meta['total_wall_clock_s'] / 3600:.2f} h) ===")
    else:
        log.warning(f"=== eps={epsilon:g} done but no usable eps_low_95 across the 5 attacks "
                    f"({meta['total_wall_clock_s'] / 3600:.2f} h). Results are written; "
                    f"inspect {results_dir.relative_to(REPO_ROOT)} ===")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SWEEP_EPSILONS,
                        help=f"budgets to audit (default: {SWEEP_EPSILONS}; 1.0 is "
                             f"excluded because counts_sweep already has it)")
    parser.add_argument("--allow-degenerate", action="store_true",
                        help="record an eps whose pools failed the distinctness guard, "
                             "flagged as degenerate_pool (see the module docstring)")
    args = parser.parse_args()

    if 1.0 in args.epsilons:
        log.warning("eps=1.0 requested. That arm is already audited at 1000/2500 in "
                    "results/counts_sweep/dpgan/1000_2500/ with cache/dpgan/threat_model.pkl. "
                    "This will build a SEPARATE cache under cache/dpgan_eps1/ and spend "
                    "~4.3 h reproducing it. Ctrl-C now if that was not intended.")

    import torch
    device_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    log.info(f"=== eps sweep privacy: eps={args.epsilons} at {NUM_TRAIN}/{NUM_TEST}, "
             f"device={device_kwargs['device']} "
             f"(torch.cuda.is_available()={torch.cuda.is_available()}) ===")
    log.info(f"    projected ~4.3 h per eps at 4.4 s/fit x {NUM_TRAIN + NUM_TEST} fits "
             f"-> ~{4.3 * len(args.epsilons):.1f} h total")

    t0 = time.time()
    failed = []
    for epsilon in args.epsilons:
        # Per-eps isolation: a guard failure or a crash in one arm must not cost the
        # others, which by then may represent many hours of finished compute.
        try:
            run_one_epsilon(epsilon, device_kwargs, args.allow_degenerate)
        except DegeneratePool as exc:
            log.error(f"DISTINCTNESS GUARD FAILED for eps={epsilon:g}:\n{exc}")
            failed.append((epsilon, "degenerate pool"))
        except Exception:
            log.error(f"eps={epsilon:g} FAILED:\n{traceback.format_exc()}")
            failed.append((epsilon, "exception"))

    log.info(f"=== all done in {(time.time() - t0) / 3600:.2f} h ===")
    if failed:
        log.warning("Incomplete arms: " +
                    ", ".join(f"eps={e:g} ({why})" for e, why in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
