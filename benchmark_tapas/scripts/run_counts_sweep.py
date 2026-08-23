#!/usr/bin/env python3
"""Ascending num_train/num_test sweep for the TAPAS audit.

WHY ASCENDING AND WHY IT IS CHEAP
    TAPAS appends to its memoised pool rather than regenerating it:
    `attacker_knowledge.py:591` does `num_samples -= len(self._memory[training][0])`
    and generates only the shortfall. SynthcityGenerator's fit counter survives
    pickling, so fit i keeps running at seeds.TAPAS_GENERATOR_SEED_BASE + i across
    stages. The whole curve therefore costs max(num_train) + max(num_test) fits per
    generator -- 3500 at 1000/2500 -- not the sum of the stages. Every intermediate
    stage is a real measurement on a prefix of the same pool, so you get a
    convergence curve for free and can stop early if the intervals tighten sooner
    than projected.

WHY IT IS WORTH RUNNING AT ALL (it was not, before 2026-08-23)
    Until the per-fit seed was added, every simulation was byte-identical and more
    samples added literally no information -- eps_low_95 grew as ln(n) purely because
    Clopper-Pearson could bound a structurally-zero FPR more tightly. Now FP is real,
    and larger counts buy two things:

      A nonzero eff-epsilon. Holding BayesNet/Groundhog's measured TPR=0.38 /
      FPR=0.14, the 95% lower bound goes 0.000 (num_test=100) -> 0.491 (500) ->
      0.646 (1000) -> 0.775 (2500). The first real empirical leakage bound here.

      A resolvable DP comparison. The Mann-Whitney null band on AUC is +-0.114 at
      num_test=100 but +-0.023 at 2500, so the BayesNet-vs-DPGAN gap of 0.068 goes
      from invisible to measurable.

    Returns flatten hard after 2500 (8000 buys only +0.10 for 3x the compute), which
    is also where TAPAS Experiment 2 and Chida et al. sit.

MEASURED COST, from the 2026-08-23 GPU run (150 fits per generator)
    bayesian_network  0.8 s/fit ->  1.2 h        ctgan      3.2 s/fit ->  3.4 h
    dpgan            11.1 s/fit -> 11.2 h        privbayes 59.8 s/fit -> 58.6 h
    BN + CTGAN + DPGAN is about 16 h, one night. PrivBayes alone is 2.5 days and is
    the weaker comparison anyway (BN vs PrivBayes is not a clean DP ablation --
    different encoders, and PrivBayes redraws its DAG every fit while BayesNet's
    structure is deterministic). CTGAN vs DPGAN is the clean one.

DISTINCTNESS GUARD
    After growing the pools at each stage, every memoised dataset is hashed and the
    run aborts if they are not essentially all distinct. This is the check that would
    have caught the seeding bug immediately: synthcity defaults random_state=0 and
    reseeds globally in Plugin.fit(), so with ExactDataKnowledge feeding an identical
    background to every simulation, all D+ datasets came out identical and all D-
    datasets came out identical. Nothing downstream can tell -- the attacks still run
    and still report numbers -- so the guard is the only thing standing between a
    silent regression and another invalid sweep.

OUTPUTS
    cache/{method}/threat_model.pkl               shared across stages, grows
    cache/{method}/datasets/synthetic_{split}.csv.gz
                                                  every generated dataset, gzipped,
                                                  readable with plain pandas
    cache/{method}/sweep_{ntr}_{nte}/result_*.json per-stage attack cache (resumable)
    results/counts_sweep/{method}/{ntr}_{nte}/     per-stage eff-epsilon + per-attack CSV
    results/counts_sweep/raw_scores_{method}_{ntr}_{nte}.csv
                                                  raw pre-threshold scores per stage

Run from the repo root, venv active:
  python benchmark_tapas/scripts/run_counts_sweep.py --methods bayesian_network ctgan dpgan
  python benchmark_tapas/scripts/run_counts_sweep.py --methods privbayes
  python benchmark_tapas/scripts/run_counts_sweep.py --methods ctgan --max-counts 500 1000
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import common                                            # noqa: E402
from config import (CACHE_DIR, RESULTS_DIR, METHOD_CONFIG, METHODS,   # noqa: E402
                    FORMAL_EPSILON, slug)
from seeds import SCORE_ATTACK_SEED                      # noqa: E402

# Ascending, nested. Each is a prefix of the next, so the total cost is the last row.
STAGES = [(50, 100), (200, 500), (500, 1000), (1000, 2500)]

SWEEP_DIR = RESULTS_DIR / "counts_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(SWEEP_DIR / "counts_sweep_log.txt"), logging.StreamHandler()],
)
log = logging.getLogger("counts_sweep")


def dataset_hash(dataset) -> str:
    """Fast content hash. pd.util.hash_pandas_object beats to_csv by a wide margin,
    which matters when this runs over 3500 datasets at every stage."""
    h = pd.util.hash_pandas_object(dataset.data, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def assert_distinct(threat_model, method: str, stage: str) -> None:
    """Abort if the memoised simulations are not essentially all distinct.

    Under correct per-fit seeding, duplicates should not occur at all: each dataset is
    500 rows sampled from a freshly fitted model. A handful of collisions could in
    principle happen for a very low-entropy generator, so the bar is 99% rather than
    100%, but anything near 1 distinct value is the seeding bug returning.
    """
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
            if frac < 0.99:
                raise SystemExit(
                    f"\n{method} [{stage}] {name}/{world}: only {n_distinct} distinct datasets "
                    f"out of {len(subset)} ({frac:.1%}).\n"
                    f"The simulations are not independent draws. This is the signature of the "
                    f"pre-2026-08-23 seeding bug: check that SynthcityGenerator._plugin_args() "
                    f"still passes a per-fit random_state and that the fit counter is advancing. "
                    f"Aborting rather than producing another invalid sweep."
                )
            log.info(f"    guard {name}/{world}: {n_distinct}/{len(subset)} distinct ({frac:.1%}) OK")


def export_pools(threat_model, cache_dir: Path) -> None:
    """Write every generated synthetic dataset out in a portable form.

    gzipped CSV rather than parquet on purpose: pandas reads it anywhere with no extra
    dependency, and the workstation's pyarrow is not available in every environment
    these files might later be opened in. Overwritten each stage, so the file always
    reflects the full pool generated so far.
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
        log.info(f"    wrote {scalers_path.name} (min/max to invert the scaling)")

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


def run_method_sweep(method: str, stages: list, extra_plugin_kwargs: dict = None) -> None:
    cfg = METHOD_CONFIG[method]
    plugin_kwargs = {**cfg["plugin_kwargs"], **(extra_plugin_kwargs or {})}
    cache_dir = CACHE_DIR / method
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {method} (dp={cfg['dp']}, plugin_kwargs={plugin_kwargs}) ===")
    log.info(f"    stages: {stages}   total fits = {stages[-1][0] + stages[-1][1]}")

    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx)

    threat_model = common.build_or_load_threat_model(
        cache_dir=cache_dir, method=method, background_dataset=background,
        target_record=target, alternate_record=alternate, description=description,
        epsilon=FORMAL_EPSILON, plugin_kwargs=plugin_kwargs,
    )
    gen = threat_model.atk_know_gen.generator
    log.info(f"    resuming at fit {getattr(gen, '_fit_counter', 0)} "
             f"(next seed {gen.seed_base + gen._fit_counter})")

    # Same seed before build_attacks for every method and every stage, so all
    # generators are probed by the same forests and the same 1500 random queries and
    # the only thing changing across stages is the sample size.
    np.random.seed(SCORE_ATTACK_SEED)
    attacks = common.build_attacks(target, background)

    for num_train, num_test in stages:
        stage = f"{num_train}_{num_test}"
        t_stage = time.time()
        log.info(f"  -- stage {num_train}/{num_test} --")

        stage_cache = cache_dir / f"sweep_{stage}"
        stage_results = SWEEP_DIR / method / stage
        stage_cache.mkdir(parents=True, exist_ok=True)
        stage_results.mkdir(parents=True, exist_ok=True)

        # Grow the pools first, so the guard runs before hours of attack time and a
        # degenerate cache is caught immediately. generate_training_samples is public;
        # there is no public equivalent for the testing pool, hence _generate_samples.
        n_before = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
        threat_model.generate_training_samples(num_train)
        threat_model._generate_samples(num_test, training=False)
        n_after = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
        log.info(f"    pools: {len(threat_model._memory[True][0])} train / "
                 f"{len(threat_model._memory[False][0])} test  (+{n_after - n_before} new fits, "
                 f"{time.time() - t_stage:.0f}s)")
        threat_model.save(str(cache_dir / "threat_model"))

        assert_distinct(threat_model, method, stage)
        export_pools(threat_model, cache_dir)

        rows, score_rows, no_scores = [], [], []
        for attack in attacks:
            result, summary = common.run_attack(
                attack, threat_model, num_train=num_train, num_test=num_test,
                cache_dir=stage_cache, results_dir=stage_results,
            )
            result.update(method=method, dp=cfg["dp"], kind=cfg["kind"],
                          num_train=num_train, num_test=num_test,
                          formal_epsilon=FORMAL_EPSILON if cfg["dp"] else None)
            rows.append(result)
            if summary is not None:
                score_rows.extend(common._score_rows(method, attack, summary))
            else:
                no_scores.append(attack.label)
            threat_model.save(str(cache_dir / "threat_model"))

        pd.DataFrame(rows).to_csv(stage_results / f"effeps_{method}_{stage}.csv", index=False)
        if score_rows:
            path = SWEEP_DIR / f"raw_scores_{method}_{stage}.csv"
            pd.DataFrame(score_rows).to_csv(path, index=False)
            log.info(f"    wrote {len(score_rows)} raw scores -> {path.name}")
        if no_scores:
            log.warning(f"    no raw scores for {no_scores} -- served from the per-attack JSON "
                        f"cache, which stores aggregates only. Delete those JSONs to recompute.")
        log.info(f"  -- stage {stage} done in {(time.time() - t_stage) / 60:.1f} min --")

    log.info(f"=== {method} complete ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                        help="generators to sweep (default: all four)")
    parser.add_argument("--max-counts", nargs=2, type=int, metavar=("NUM_TRAIN", "NUM_TEST"),
                        default=None, help="stop after this stage (default: 1000 2500)")
    args = parser.parse_args()

    stages = STAGES
    if args.max_counts:
        cap = tuple(args.max_counts)
        stages = [s for s in STAGES if s[0] <= cap[0] and s[1] <= cap[1]]
        if cap not in stages:
            stages = stages + [cap]
        if not stages:
            parser.error(f"--max-counts {cap} is below the smallest stage {STAGES[0]}")

    device_kwargs = {}
    if any(m in ("ctgan", "dpgan") for m in args.methods):
        import torch
        device_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        log.info(f"neural device: {device_kwargs['device']} "
                 f"(torch.cuda.is_available()={torch.cuda.is_available()})")

    log.info(f"=== counts sweep: {args.methods} through {stages[-1]} ===")
    t0 = time.time()
    for method in args.methods:
        extra = device_kwargs if method in ("ctgan", "dpgan") else None
        run_method_sweep(method, stages, extra)
    log.info(f"=== all done in {(time.time() - t0) / 3600:.2f} h ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
