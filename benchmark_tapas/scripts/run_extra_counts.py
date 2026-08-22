#!/usr/bin/env python3
"""Fill in the intermediate count settings for BayesNet and DPGAN -- for free.

WHY THIS EXISTS
    The eff-epsilon-vs-sample-size picture currently has two columns and a hole:

        counts      fits    BN    PrivBayes   CTGAN   DPGAN
        50/100       150    yes      yes       yes    STALE (n_iter=100)
        200/500      700     -        -         -       -
        500/1000    1500     -        -         -       -
        1000/2500   3500    yes       -         -      yes

    Two points per generator cannot separate "eps_low_95 is converging on a real
    leakage measurement" from "eps_low_95 is a Clopper-Pearson ceiling that grows
    like ln(n) forever". Four points can, and BN and DPGAN can have four at no
    compute cost at all (see below). PrivBayes and CTGAN stay at their single
    50/100 point -- filling those in would mean 3500 real generator fits each
    (~7 h CTGAN, ~29 h PrivBayes), which is not worth it to confirm a shape that
    two generators will already have shown.

WHY IT COSTS NOTHING
    run_pilot_counts.py left fully-populated threat models for bayesian_network
    and dpgan at 1000/2500 -- 3500 memoised synthetic datasets each, one
    generator fit apiece, already paid for. Every smaller count setting is a
    PREFIX of that memory, so 200/500 and 500/1000 are obtained by truncating a
    loaded copy. No generator is ever re-fit. The only cost is attack compute
    (Groundhog feature extraction, the RandomForest, the KDE), on the order of
    an hour for all six cells.

    Truncation is statistically sound here because SwapMIALabeller appends the
    memoised datasets in member/non-member PAIRS -- index 2i is D+ (background +
    target), 2i+1 is D- (background + alternate) -- and each pair is an
    independent draw. A prefix of even length is therefore itself a balanced
    i.i.d. sample. Verified against both pickles: truncating 2500 -> 100 leaves
    exactly 50 member / 50 non-member.

THE BUG THIS AVOIDS  (read before changing anything below)
    LabelInferenceThreatModel._generate_samples does

        num_samples -= len(self._memory[training][0])
        ... generate only if positive ...
        return self._memory[training]          # ALL of it, not the first n

    So a threat model holding 2500 memoised test datasets, asked for 500,
    silently hands back all 2500. Without the explicit truncation in
    `_truncate_memory`, every cell here would be labelled 200/500 or 500/1000
    while actually being evaluated on the full 1000/2500 memory -- producing the
    1000/2500 answer four times and looking like convergence. That is precisely
    the artefact this script exists to rule out, so the truncation is load-bearing,
    not a tidy-up. There is an assert after it.

DPGAN AT 50/100
    Included by default, because the existing 50/100 DPGAN audit is stale: its
    cache (cache_LEGACY/dpgan) records plugin_kwargs={'n_iter': 100}, from before
    the 2026-08-16 change to DPGAN_N_ITER=50. That is not "the same model trained
    longer" -- opacus calibrates the per-step noise so the budget over n_iter
    epochs equals eps=1.0, so a different n_iter is a different DP mechanism and
    its numbers cannot sit on the same curve. Regenerating it from the 1000/2500
    pilot cache costs nothing and gives a 50/100 point at the correct n_iter=50.
    (BN, CTGAN and PrivBayes at 50/100 are fine and are left alone --
    cache_LEGACY records plugin_kwargs={} / {'n_iter': 50} / {} respectively.)

SAFETY
    The pilot caches are opened READ-ONLY: the loaded threat model is truncated
    in memory and never saved back, so cache_pilot_*_1000_2500/threat_model.pkl
    cannot be shrunk by a crash or a mistake here. Every generator's configured
    plugin_kwargs is checked against what the pickle actually recorded before any
    of its data is used; a mismatch skips that generator loudly rather than
    quietly mixing mechanisms.

    Results are written to the pilot's own directory layout
    (results/pilot_counts/{method}_{train}_{test}/, cache_pilot_{method}_{train}_{test}/)
    because these are the same experiment at more settings. Each attack's result
    JSON is cached as it finishes and the CSV is rewritten after every attack, so
    the run is resumable: re-run the same command and finished cells are skipped.

Run from repo root, inside the venv:
  python benchmark_tapas/scripts/run_extra_counts.py --dry-run
  nohup python -u benchmark_tapas/scripts/run_extra_counts.py \
      > benchmark_tapas/results/pilot_counts/extra_counts_nohup.out 2>&1 &

Then plot with:
  python benchmark_tapas/analysis/eff_eps_vs_counts.py
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

import common
from config import CACHE_DIR, FORMAL_EPSILON, METHOD_CONFIG, RESULTS_DIR, slug

# -- What to run ---------------------------------------------------------
#
# Only the cells that are missing AND free. Keyed by generator, in run order:
# BN first (its attacks are the cheapest), DPGAN second. Ascending counts within
# a generator so the quick cells report before the slow ones.
#
# 1000/2500 is absent on purpose -- run_pilot_counts.py already produced it for
# both, and those results are reused as-is by the plotting script.
TO_RUN = {
    "bayesian_network": [(200, 500), (500, 1000)],
    "dpgan":            [(50, 100), (200, 500), (500, 1000)],
}
METHOD_ORDER = ["bayesian_network", "dpgan"]

# The fully-populated threat models left by the pilot. (cache dir, train, test)
# that each is expected to hold; re-verified against the pickle before use.
PILOT_CACHES = {
    "bayesian_network": ("cache_pilot_bayesian_network_1000_2500", 1000, 2500),
    "dpgan":            ("cache_pilot_dpgan_1000_2500",            1000, 2500),
}

PILOT_DIR = RESULTS_DIR / "pilot_counts"
PILOT_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_CSV = PILOT_DIR / "extra_counts.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(PILOT_DIR / "extra_counts_log.txt"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("extra_counts")


class StaleCacheError(RuntimeError):
    """A cached threat model was built with plugin kwargs we no longer audit at."""


# -- Threat-model memory surgery -----------------------------------------
#
# These two helpers reach into LabelInferenceThreatModel._memory directly.
# Deliberate: TAPAS exposes no public way to ask how many datasets a threat
# model has memoised, nor to shrink that memory. The structure is documented in
# tapas/threat_models/attacker_knowledge.py -- `_memory` maps training=True/False
# to (list_of_datasets, list_of_labels), appended in generation order.

def _memory_counts(threat_model) -> tuple:
    """(num memoised training datasets, num memoised testing datasets)."""
    return (len(threat_model._memory[True][0]), len(threat_model._memory[False][0]))


def _truncate_memory(threat_model, target: int, training: bool) -> int:
    """Shrink one memory bucket to exactly `target` datasets; return how many
    were dropped. `target` must be even so the D+/D- pairing divides exactly."""
    assert target % 2 == 0, f"count {target} must be even to keep D+/D- pairs balanced"
    datasets, labels = threat_model._memory[training]
    dropped = max(0, len(datasets) - target)
    if dropped:
        threat_model._memory[training] = (datasets[:target], labels[:target])
    return dropped


def _check_not_stale(threat_model, method: str, expected_kwargs: dict, where: Path):
    """Refuse a threat model whose generator was configured differently.

    SynthcityGenerator.__getstate__ drops only the fitted plugin, so the pickle
    still records the plugin_kwargs the memoised synthetic data was actually
    produced under. `device` is ignored -- CPU vs CUDA changes the wall clock,
    not the mechanism.
    """
    generator = getattr(threat_model.atk_know_gen, "generator", None)
    if generator is None:
        raise StaleCacheError(f"{where}: no generator on the pickled threat model")
    if getattr(generator, "method", None) != method:
        raise StaleCacheError(
            f"{where}: cached generator is {getattr(generator, 'method', None)!r}, "
            f"expected {method!r}")
    cached = {k: v for k, v in (getattr(generator, "plugin_kwargs", {}) or {}).items()
              if k != "device"}
    wanted = {k: v for k, v in expected_kwargs.items() if k != "device"}
    if cached != wanted:
        raise StaleCacheError(
            f"{where}: this cache's synthetic data was generated with "
            f"plugin_kwargs={cached}, but the audit config is {wanted}. A different "
            f"n_iter is a different mechanism (opacus recalibrates DPGAN's per-step "
            f"noise to hit eps=1.0 over n_iter epochs), so it cannot go on the same "
            f"curve. Quarantine it:\n    mv {where} {where}_LEGACY")


# -- One (method, counts) cell -------------------------------------------

def run_cell(method: str, num_train: int, num_test: int, cfg: dict,
             plugin_kwargs: dict, shared) -> tuple:
    """Run the 5-attack battery for one count setting off the pilot cache.

    `shared` is the (background, target, alternate, description) tuple, drawn
    once for the whole run -- identical to what the pilot and the main benchmark
    use. That is what makes these cells comparable to the existing ones, and what
    makes the pilot's memoised datasets applicable at all.
    """
    background_dataset, target_record, alternate_record, description = shared

    # Same layout the pilot itself writes to: these are the same experiment at
    # more settings, so they belong in the same place, named the same way.
    cache_dir = CACHE_DIR.parent / f"cache_pilot_{method}_{num_train}_{num_test}"
    results_dir = PILOT_DIR / f"{method}_{num_train}_{num_test}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"--- {method} @ {num_train}/{num_test} "
             f"(fits={num_train + num_test}, all reused from the pilot cache) ---")

    attacks = common.build_attacks(target_record, background_dataset)
    all_cached = all((cache_dir / f"result_{slug(a.label)}.json").exists() for a in attacks)

    threat_model = None
    if all_cached:
        log.info("    all 5 attacks already cached -- nothing to load")
    else:
        pilot_name, have_train, have_test = PILOT_CACHES[method]
        pilot_cache = CACHE_DIR.parent / pilot_name
        if not (pilot_cache / "threat_model.pkl").exists():
            raise FileNotFoundError(
                f"{pilot_cache}/threat_model.pkl is missing -- pull it over from the "
                f"GPU workstation. Without it this cell would need "
                f"{num_train + num_test} real generator fits.")

        # Reloaded fresh for every cell. Truncation mutates the memory in place,
        # so carrying one object across cells would mean each cell inherited the
        # previous cell's (smaller) memory. The pickle loads in ~0.5 s, so there
        # is nothing to gain by being clever here.
        log.info(f"    loading {pilot_name} (read-only, never saved back)")
        threat_model = common.build_or_load_threat_model(
            cache_dir=pilot_cache, method=method,
            background_dataset=background_dataset,
            target_record=target_record, alternate_record=alternate_record,
            description=description, epsilon=FORMAL_EPSILON,
            plugin_kwargs=plugin_kwargs)
        _check_not_stale(threat_model, method, plugin_kwargs, pilot_cache)

        got_train, got_test = _memory_counts(threat_model)
        if got_train < num_train or got_test < num_test:
            raise RuntimeError(
                f"{pilot_name} holds {got_train}/{got_test} memoised datasets, "
                f"too few for {num_train}/{num_test} (expected {have_train}/{have_test})")

        # Load-bearing -- see the module docstring. Without this the attacks would
        # silently run on all 1000/2500 memoised datasets while the row said 200/500.
        dropped_train = _truncate_memory(threat_model, num_train, True)
        dropped_test = _truncate_memory(threat_model, num_test, False)
        log.info(f"    truncated {got_train}/{got_test} -> {num_train}/{num_test} "
                 f"(dropped {dropped_train} train, {dropped_test} test)")
        assert _memory_counts(threat_model) == (num_train, num_test)

    rows = []
    t0 = time.time()
    for attack in attacks:
        result_path = cache_dir / f"result_{slug(attack.label)}.json"
        was_cached = result_path.exists()
        result, _ = common.run_attack(
            attack, threat_model, num_train=num_train, num_test=num_test,
            cache_dir=cache_dir, results_dir=results_dir)
        result["method"] = method
        result["dp"] = cfg["dp"]
        result["kind"] = cfg["kind"]
        result["formal_epsilon"] = FORMAL_EPSILON if cfg["dp"] else None
        result["fits"] = num_train + num_test
        # summary.tp / summary.fp are RATES (tapas/report/attack_summary.py), so
        # they are the TPR and FPR. Aliased under both names: tp/fp keeps
        # continuity with the existing effeps_*.csv and pilot_*.csv tables.
        result["tpr"] = result["tp"]
        result["fpr"] = result["fp"]
        result["source"] = "cached" if was_cached else "pilot_cache_reuse"
        result["timestamp"] = datetime.now().isoformat(timespec="seconds")
        rows.append(result)
        _write_csv(rows[-1:])

    # The pilot's own per-cell CSV, same filename pattern, so the plotting script
    # can glob one directory for every count setting.
    cell_csv = results_dir / f"pilot_{method}_{num_train}_{num_test}.csv"
    pd.DataFrame(rows).to_csv(cell_csv, index=False)

    eps = [r["eps_low_95"] for r in rows if r.get("eps_low_95") is not None]
    best = max(eps) if eps else float("nan")
    best_attack = next((r["attack"] for r in rows if r.get("eps_low_95") == best), "")
    log.info(f"    max eps_low_95 = {best:.4f} ({best_attack})   "
             f"max TPR = {max(r['tp'] for r in rows):.3f}  "
             f"max FPR = {max(r['fp'] for r in rows):.3f}   "
             f"({time.time() - t0:.0f}s)   -> {cell_csv}")
    return rows, best


# -- Output --------------------------------------------------------------

_COL_ORDER = [
    "method", "dp", "kind", "formal_epsilon", "num_train", "num_test", "fits",
    "attack", "eps_low_95", "tpr", "fpr", "wall_time_s",
    "auc", "mia_advantage", "privacy_gain", "tp", "fp",
    "eff_epsilon_pointwise", "eff_epsilon_pointwise_is_inf",
    "eps_low_90", "eps_high_90", "eps_high_95", "eps_low_99", "eps_high_99",
    "peak_memory_mb", "source", "timestamp",
]


def _write_csv(new_rows):
    """Merge rows into extra_counts.csv and rewrite it after EVERY attack, so a
    run killed partway still keeps everything it earned. Keyed on
    (method, num_train, num_test, attack) -- re-running replaces, never duplicates."""
    frame = pd.DataFrame(new_rows)
    if EXTRA_CSV.exists():
        frame = pd.concat([pd.read_csv(EXTRA_CSV), frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["method", "num_train", "num_test", "attack"], keep="last")
    lead = [c for c in _COL_ORDER if c in frame.columns]
    frame = frame[lead + [c for c in frame.columns if c not in lead]]
    frame.sort_values(["method", "fits", "attack"], kind="stable").to_csv(
        EXTRA_CSV, index=False)


def _parse_combo(text: str) -> tuple:
    try:
        a, b = text.replace(",", "/").split("/")
        return int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"combo {text!r} must look like 200/500")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=METHOD_ORDER, choices=METHOD_ORDER,
                        help="generators to fill in (default: both). Only these two "
                             "have a populated pilot cache; anything else would need "
                             "thousands of real generator fits.")
    parser.add_argument("--combos", nargs="*", type=_parse_combo, default=None,
                        help="override the count settings (default: per-generator, "
                             "see TO_RUN -- BN 200/500 500/1000, DPGAN 50/100 200/500 "
                             "500/1000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run, run nothing")
    args = parser.parse_args()

    plan = []
    for method in args.methods:
        combos = sorted(set(args.combos)) if args.combos else TO_RUN[method]
        for num_train, num_test in combos:
            plan.append((method, num_train, num_test))

    if args.dry_run:
        print("\n=== cells to run (all reuse the pilot caches -- 0 generator fits) ===")
        for method, num_train, num_test in plan:
            done = (CACHE_DIR.parent / f"cache_pilot_{method}_{num_train}_{num_test}")
            n_done = len(list(done.glob("result_*.json"))) if done.is_dir() else 0
            state = "already complete" if n_done == 5 else f"{n_done}/5 attacks cached"
            print(f"  {method:18s} {num_train:5d}/{num_test:<5d} "
                  f"({num_train + num_test:5d} fits, reused)   {state}")
        print(f"\n  {len(plan)} cells x 5 attacks. Attack compute only -- no generator "
              f"is fit.\n  Expect roughly an hour in total; the two shadow-modelling "
              f"attacks dominate.\n")
        return 0

    log.info("=== extra count settings from the pilot caches ===")
    log.info(f"cells: {[f'{m} {a}/{b}' for m, a, b in plan]}")

    # One background / target / alternate draw for everything, exactly as the
    # pilot and the main benchmark use. If this differed, the pilot's memoised
    # datasets would not apply and the new points would not be comparable to the
    # existing ones.
    train_dataset, _, description = common.load_adult_datasets()
    background_dataset, background_indices = common.sample_background(train_dataset)
    target_record, alternate_record = common.select_random_target(
        train_dataset, background_indices)
    shared = (background_dataset, target_record, alternate_record, description)

    failures = []
    t0 = time.time()
    for method in args.methods:
        cfg = METHOD_CONFIG[method]
        plugin_kwargs = dict(cfg["plugin_kwargs"])
        if method in ("ctgan", "dpgan"):
            # Imported here rather than at module scope so the statistical
            # generators never pull torch in.
            import torch
            plugin_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"=== {method} (dp={cfg['dp']}, plugin_kwargs={plugin_kwargs}) ===")

        for m, num_train, num_test in plan:
            if m != method:
                continue
            try:
                run_cell(method, num_train, num_test, cfg, plugin_kwargs, shared)
            except StaleCacheError as exc:
                log.error(f"STALE CACHE -- skipping the rest of {method}:\n{exc}")
                failures.append((method, f"{num_train}/{num_test}", "stale cache"))
                break                   # every cell would hit the same cache
            except Exception as exc:    # noqa: BLE001 -- unattended background run
                log.error(f"FAILED {method} {num_train}/{num_test}: {exc}")
                log.error(traceback.format_exc())
                failures.append((method, f"{num_train}/{num_test}", repr(exc)))
                continue                # one bad cell must not end the run

    log.info(f"=== done in {(time.time() - t0) / 60:.1f} min -> {EXTRA_CSV} ===")

    if EXTRA_CSV.exists():
        table = pd.read_csv(EXTRA_CSV)
        agg = (table.groupby(["method", "num_train", "num_test", "fits"], as_index=False)
                    .agg(max_eps_low_95=("eps_low_95", "max"),
                         max_tpr=("tpr", "max"), max_fpr=("fpr", "max")))
        print("\n=== new points (existing 50/100 and 1000/2500 not shown) ===")
        for method, group in agg.sort_values(["method", "fits"]).groupby("method"):
            print(f"\n  {method}")
            print(f"    {'counts':>12s}  {'fits':>5s}  {'max eps_low_95':>14s}"
                  f"  {'max TPR':>7s}  {'max FPR':>7s}")
            for _, r in group.iterrows():
                print(f"    {int(r.num_train):5d}/{int(r.num_test):<6d}  {int(r.fits):5d}"
                      f"  {r.max_eps_low_95:14.4f}  {r.max_tpr:7.3f}  {r.max_fpr:7.3f}")
        print("\n  Plot all four count settings together with:"
              "\n    python benchmark_tapas/analysis/eff_eps_vs_counts.py\n")

    if failures:
        print("=== failures ===")
        for method, cell, why in failures:
            print(f"  {method:18s} {cell:12s} {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
