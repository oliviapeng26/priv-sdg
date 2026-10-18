#!/usr/bin/env python3
"""Phase 2 of the formal-epsilon sweep: TAPAS MIA audit of DPGAN at each eps.

Structurally this is run_synthcity_sample_size_sweep.py with the loop swapped: instead of
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

        results/sample_size_sweep/dpgan/1000_2500/effeps_dpgan_1000_2500.csv
        results/sample_size_sweep/raw_scores_dpgan_1000_2500.csv
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
    Inherited verbatim from run_synthcity_sample_size_sweep.py: every memoised dataset is hashed
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

TAGGED RUNS -- VARYING THE PAIR, THE BACKGROUND OR THE SEEDS AT A FIXED EPS
    With no new options the script does exactly what it always did. --seed-base,
    --background-seed, --target-seed and --target-index/--alternate-index each
    change WHAT is audited while leaving eps alone, and every path is derived from
    eps alone, so each of them requires --tag. The tag is appended to the label,
    eps1 -> eps1_<tag>, and from there to every path below plus a separate log file
    (eps_sweep_privacy_log_<tag>.txt). eps_slug never contains "_", so a tagged
    label cannot equal an untagged one and two different tags cannot collide.

    --target-index/--alternate-index name the pair directly (train-set positions,
    the indices select_random_target logs), for re-auditing a known pair against a
    different background. Neither may be in the sampled background: that is checked
    up front and the run refuses. --target-seed and the explicit indices are
    mutually exclusive. SCORE_ATTACK_SEED is deliberately not an option.

    The resolved configuration (eps, counts, seed base, background seed and a hash
    of its indices, target seed or explicit indices, the t/t' indices actually used,
    attack seed, n_iter, tag) is written to meta.json and attached to the pickled
    threat model. On resume the pickle's configuration is compared with this run's
    and the arm stops on any difference. Pickles from before this was recorded are
    treated as the default configuration, corrected by what the pickle itself still
    shows (eps, seed base, n_iter, the background and the t/t' records).

    --seed-fix switches on SynthcityGenerator(seed_fix=True): every reseed inside a
    fit and its generate uses that fit's seed (seed_base + i), and generate() passes
    it to the plugin (see common.forced_synthcity_seed). It changes the mechanism's
    randomness, so like the options above it requires --tag. The flag is recorded as
    "seed_fix" in the configuration (meta.json and the pickled threat model), and a
    cache is resumed only if its configuration AND its generator agree with it, so a
    seed-fix run never loads a non-seed-fix cache or the reverse. Pickles from before
    the flag existed count as seed_fix=False.

    --dry-run prints the resolved configuration and every path the run would write,
    marking the ones that already exist, and exits without generating anything.

OUTPUTS   (<label> = eps{eps}, or eps{eps}_<tag> for a tagged run)
    cache/dpgan_eps{eps}/threat_model.pkl           per-eps, never shared
    cache/dpgan_eps{eps}/datasets/synthetic_{split}.csv.gz
                                                    every generated dataset, gzipped
    cache/dpgan_eps{eps}/sweep_1000_2500/result_*.json
                                                    per-attack cache (resumable)
    results/eps_sweep/dpgan/eps{eps}/effeps_dpgan_eps{eps}.csv
    results/eps_sweep/dpgan/eps{eps}/effective_epsilon_*.csv
    results/eps_sweep/dpgan/eps{eps}/meta.json    timings + guard fractions
    results/eps_sweep/dpgan/raw_scores_dpgan_eps{eps}.csv
                                                    raw pre-threshold scores

Run from the repo root, venv active:
  python benchmark_tapas/audits/run_dpgan_eps_sweep.py
  python benchmark_tapas/audits/run_dpgan_eps_sweep.py --epsilons 10 100
  python benchmark_tapas/audits/run_dpgan_eps_sweep.py --epsilons 0.1 --allow-degenerate
  python benchmark_tapas/audits/run_dpgan_eps_sweep.py --epsilons 1.0 --seed-base 10000 --tag seedbase10000
  python benchmark_tapas/audits/run_dpgan_eps_sweep.py --epsilons 1.0 --tag seedbase10000 --seed-base 10000 --dry-run
  python benchmark_tapas/audits/run_dpgan_eps_sweep.py --epsilons 0.5 1 10 --tag seedfix --seed-fix

Resumable: the pools are saved every --chunk fits (default 100, the same chunked
growth run_full_audit_placebo.py uses), and finished attacks are cached, so a
crash or a disconnect costs at most one chunk or the attack in flight. Re-run the
same command to resume. A CUDA out-of-memory error (another job on a shared GPU)
is retried from the last checkpoint after --oom-wait seconds, up to --oom-retries
times. Saves are atomic (temp file + rename). Chunking does not change any result:
fits run in the same order at the same per-fit seeds as one unbroken call.
"""

import argparse
import hashlib
import json
import logging
import os
import re
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
from config import (CACHE_DIR, RESULTS_DIR, METHOD_CONFIG, slug,  # noqa: E402
                    BACKGROUND_SEED, TARGET_SEED)
from seeds import SCORE_ATTACK_SEED, TAPAS_GENERATOR_SEED_BASE   # noqa: E402

METHOD = "dpgan"
# 1.0 is reused from counts_sweep (see module docstring).
SWEEP_EPSILONS = [0.1, 10.0, 100.0]
# Chida et al. and TAPAS Experiment 2 both sit here; the counts sweep showed the
# eff-epsilon interval flattening beyond it.
NUM_TRAIN, NUM_TEST = 1000, 2500

SWEEP_DIR = RESULTS_DIR / "eps_sweep" / "dpgan"
PRIVACY_DIR = SWEEP_DIR                     # eps arms sit directly under eps_sweep/dpgan/
PRIVACY_DIR.mkdir(parents=True, exist_ok=True)

# delay=True: the file is opened on the first record, not here, so --dry-run and a
# tagged run (both swap this handler out in main before logging anything) never
# touch the shared log. An untagged run writes to it exactly as before.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(PRIVACY_DIR / "eps_sweep_privacy_log.txt", delay=True),
              logging.StreamHandler()],
)
log = logging.getLogger("eps_sweep_privacy")


class DegeneratePool(RuntimeError):
    """The memoised simulations are not independent draws. Raised by assert_distinct."""


class ConfigMismatch(RuntimeError):
    """A cached threat model was built under a different configuration than this run."""


def eps_slug(eps: float) -> str:
    """Must match eps_sweep_generate.eps_slug: 0.1 -> 'eps0.1', 10.0 -> 'eps10'."""
    return f"eps{eps:g}"


# -- Run options, labels, paths and the recorded configuration -------------

# What a tagged run may vary. None means the default, which is all an untagged run
# (and run_eps_nudge.py / run_full_audit_placebo.py, which call run_one_epsilon with
# no options) ever gets.
RUN_OPTIONS = ("tag", "seed_base", "background_seed", "target_seed",
               "target_index", "alternate_index", "seed_fix")
TAG_PATTERN = re.compile(r"[A-Za-z0-9-]+")

# Captured at import, before any caller can rebind it (run_full_audit_placebo.py
# does), so a rebound selector is recorded as such instead of as TARGET_SEED.
_SELECT_RANDOM_TARGET = common.select_random_target


def arm_label(epsilon: float, tag: str = None) -> str:
    """eps_slug, plus '_<tag>' for a tagged run. Looks eps_slug up at call time, so
    run_full_audit_placebo.py's rebinding of it still applies."""
    label = eps_slug(epsilon)
    return f"{label}_{tag}" if tag else label


def arm_paths(label: str) -> dict:
    """Every per-arm path run_one_epsilon writes, from the label alone. Module
    globals are read at call time, so run_eps_nudge.py's PRIVACY_DIR redirection
    still applies. Per-attack files hang off stage_cache and results_dir."""
    cache_dir = CACHE_DIR / f"{METHOD}_{label}"
    results_dir = PRIVACY_DIR / label
    return {
        "cache_dir": cache_dir,
        "threat_model": cache_dir / "threat_model.pkl",
        "stage_cache": cache_dir / f"sweep_{NUM_TRAIN}_{NUM_TEST}",
        "datasets": cache_dir / "datasets",
        "results_dir": results_dir,
        "effeps": results_dir / f"effeps_{METHOD}_{label}.csv",
        "meta": results_dir / "meta.json",
        "raw_scores": PRIVACY_DIR / f"raw_scores_{METHOD}_{label}.csv",
    }


def log_path(tag: str = None) -> Path:
    return PRIVACY_DIR / (f"eps_sweep_privacy_log_{tag}.txt" if tag else
                          "eps_sweep_privacy_log.txt")


def resolve_pair(train_dataset, options: dict):
    """Background, then t/t'. With no options this makes exactly the calls the script
    always made, so callers that rebind common.select_random_target still work.
    Returns (background, background_idx, target, alternate)."""
    if options.get("background_seed") is None:
        background, background_idx = common.sample_background(train_dataset)
    else:
        background, background_idx = common.sample_background(
            train_dataset, seed=options["background_seed"])
    if options.get("target_index") is not None:
        target, alternate = common.select_fixed_target(
            train_dataset, background_idx, options["target_index"], options["alternate_index"])
    elif options.get("target_seed") is not None:
        target, alternate = common.select_random_target(
            train_dataset, background_idx, seed=options["target_seed"])
    else:
        target, alternate = common.select_random_target(train_dataset, background_idx)
    return background, background_idx, target, alternate


def background_fingerprint(indices) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in sorted(indices)).encode()).hexdigest()


def _default_target_selection() -> tuple:
    if common.select_random_target is _SELECT_RANDOM_TARGET:
        return "random", TARGET_SEED
    return "random (select_random_target rebound by caller)", None


def build_config(epsilon: float, options: dict, background_idx, target, alternate) -> dict:
    """The full configuration of one arm, with the t/t' indices actually used."""
    if options.get("target_index") is not None:
        selection, target_seed = "explicit", None
    elif options.get("target_seed") is not None:
        selection, target_seed = "random", options["target_seed"]
    else:
        selection, target_seed = _default_target_selection()
    seed_base, bg_seed = options.get("seed_base"), options.get("background_seed")
    return {
        "method": METHOD,
        "formal_epsilon": float(epsilon),
        "num_train": NUM_TRAIN,
        "num_test": NUM_TEST,
        "seed_base": TAPAS_GENERATOR_SEED_BASE if seed_base is None else seed_base,
        "background_seed": BACKGROUND_SEED if bg_seed is None else bg_seed,
        "background_sha256": background_fingerprint(background_idx),
        "target_selection": selection,
        "target_seed": target_seed,
        "target_index": int(target.id),
        "alternate_index": int(alternate.id),
        "attack_seed": SCORE_ATTACK_SEED,
        "n_iter": METHOD_CONFIG[METHOD]["plugin_kwargs"].get("n_iter"),
        "tag": options.get("tag"),
        "seed_fix": bool(options.get("seed_fix")),
    }


def stored_config(threat_model, current: dict) -> dict:
    """The configuration a cached threat model was built under. Pickles from before
    it was recorded are taken as the default configuration, overridden by whatever
    the pickle itself still shows: eps, seed base and n_iter from the generator,
    and the background and t/t' records it actually holds."""
    stored = getattr(threat_model, "audit_config", None)
    if stored is not None:
        # Recorded before seed_fix existed: that run was seed_fix=False.
        return {"seed_fix": False, **stored}
    selection, target_seed = _default_target_selection()
    legacy = dict(current, num_train=NUM_TRAIN, num_test=NUM_TEST,
                  background_seed=BACKGROUND_SEED, target_selection=selection,
                  target_seed=target_seed, attack_seed=SCORE_ATTACK_SEED, tag=None)
    gen = threat_model.atk_know_gen.generator
    legacy["formal_epsilon"] = float(gen.epsilon)
    legacy["seed_base"] = getattr(gen, "seed_base", TAPAS_GENERATOR_SEED_BASE)
    legacy["n_iter"] = gen.plugin_kwargs.get("n_iter")
    legacy["seed_fix"] = bool(getattr(gen, "seed_fix", False))
    background = threat_model.atk_know_data.attacker_knowledge.training_dataset
    legacy["background_sha256"] = background_fingerprint(background.data.index)
    legacy["target_index"] = int(threat_model.target_record.id)
    legacy["alternate_index"] = int(threat_model.alternate_record.id)
    return legacy


def checkpoint(threat_model, cache_dir: Path) -> None:
    """Save the threat model atomically: write a temp pickle, then rename over the
    real one, so a crash mid-save cannot leave a truncated threat_model.pkl."""
    tmp = cache_dir / "threat_model.tmp"
    threat_model.save(str(tmp))
    os.replace(f"{tmp}.pkl", cache_dir / "threat_model.pkl")


def grow_pools(threat_model, cache_dir: Path, chunk: int, oom_retries: int,
               oom_wait_s: int):
    """Grow the train pool to NUM_TRAIN, then the test pool to NUM_TEST, saving every
    `chunk` fits, so a crash costs at most one chunk and re-running resumes.

    Same order as the unchunked calls (train first, then test), and every fit takes
    the next per-fit seed, so the pools are the ones one unbroken call makes.

    On CUDA out-of-memory (another job on a shared GPU) the chunk is retried after
    oom_wait_s, up to oom_retries times, FROM THE LAST CHECKPOINT. TAPAS appends each
    dataset to memory as it is made (attacker_knowledge.py _sync_generate_data), so
    an in-place retry would keep a partial chunk, possibly half a D+/D- pair, and
    shift every later seed onto a different world. Reloading discards that partial
    chunk and its seeds are reused, which is exactly what a crash plus a re-run does.
    Returns the threat model (a new object if it was reloaded)."""
    import torch

    t0 = time.time()
    fitted = 0
    for training, want, name in ((True, NUM_TRAIN, "train"), (False, NUM_TEST, "test")):
        while len(threat_model._memory[training][0]) < want:
            have = len(threat_model._memory[training][0])
            step = min(want, have + chunk)
            for attempt in range(oom_retries + 1):
                try:
                    threat_model._generate_samples(step, training=training)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if attempt == oom_retries:
                        log.error(f"    {name} pool {step}/{want}: CUDA out of memory, giving "
                                  f"up after {oom_retries} retries. Re-run the same command "
                                  f"to resume from the last saved chunk.")
                        raise
                    log.warning(f"    {name} pool {step}/{want}: CUDA out of memory (another "
                                f"job on the GPU?). Back to the last checkpoint, retry "
                                f"{attempt + 1}/{oom_retries} in {oom_wait_s}s")
                    time.sleep(oom_wait_s)
                    threat_model = common.tm.ThreatModel.load(str(cache_dir / "threat_model"))
            checkpoint(threat_model, cache_dir)
            fitted += step - have
            elapsed = time.time() - t0
            n_tr, n_te = (len(threat_model._memory[t][0]) for t in (True, False))
            left = (NUM_TRAIN - n_tr) + (NUM_TEST - n_te)
            rate = elapsed / fitted
            log.info(f"    {name} pool {step}/{want} saved  ({fitted} fits in "
                     f"{elapsed / 60:.0f} min, {rate:.1f} s/fit, {left} fits left, "
                     f"~{left * rate / 3600:.1f} h)")
    return threat_model


def check_config(stored: dict, current: dict, where: Path) -> None:
    diffs = []
    for key in sorted(set(stored) | set(current)):
        a, b = stored.get(key), current.get(key)
        same = (np.isclose(float(a), float(b)) if key == "formal_epsilon" and None not in (a, b)
                else a == b)
        if not same:
            diffs.append(f"{key}: cached={a!r} this run={b!r}")
    if diffs:
        raise ConfigMismatch(
            f"{where} was built under a different configuration:\n    " + "\n    ".join(diffs)
            + "\n  Use a different --tag, or delete that cache directory if it is yours "
              "to discard.")


# -- Copied from run_synthcity_sample_size_sweep.py -------------------------------------
# Verbatim rather than imported: importing that module would run its
# logging.basicConfig and start appending this sweep's lines to
# results/sample_size_sweep/counts_sweep_log.txt, which is a committed artefact of a
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

def run_one_epsilon(epsilon: float, device_kwargs: dict, allow_degenerate: bool,
                    options: dict = None, chunk: int = 100, oom_retries: int = 30,
                    oom_wait_s: int = 120) -> None:
    """One arm. options holds the RUN_OPTIONS of a tagged run; None (what
    run_eps_nudge.py and run_full_audit_placebo.py pass) is the default arm.
    chunk / oom_retries / oom_wait_s only control checkpointing (see grow_pools)."""
    options = options or {}
    tag = options.get("tag")
    label = arm_label(epsilon, tag)
    paths = arm_paths(label)
    cfg = METHOD_CONFIG[METHOD]
    plugin_kwargs = {**cfg["plugin_kwargs"], **device_kwargs}
    cache_dir = paths["cache_dir"]
    stage_cache = paths["stage_cache"]
    results_dir = paths["results_dir"]

    log.info(f"=== {METHOD} eps={epsilon:g}{f' tag={tag}' if tag else ''} "
             f"(num_train={NUM_TRAIN}, num_test={NUM_TEST}, "
             f"plugin_kwargs={plugin_kwargs}) ===")
    log.info(f"    cache {cache_dir.relative_to(REPO_ROOT)}   "
             f"total fits = {NUM_TRAIN + NUM_TEST}")

    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx, target, alternate = resolve_pair(train_dataset, options)
    config = build_config(epsilon, options, background_idx, target, alternate)
    log.info(f"    config {json.dumps(config)}")

    resuming = paths["threat_model"].exists()
    threat_model = common.build_or_load_threat_model(
        cache_dir=cache_dir, method=METHOD, background_dataset=background,
        target_record=target, alternate_record=alternate, description=description,
        epsilon=epsilon, plugin_kwargs=plugin_kwargs,
        seed_base=config["seed_base"], audit_config=config,
        seed_fix=config["seed_fix"],
    )
    if resuming:
        # Before any directory is created or any fit run: a cache made under another
        # seed base, background, pair or tag must not be extended or re-attacked.
        check_config(stored_config(threat_model, config), config, paths["threat_model"])
        # Saved now, so every checkpoint grow_pools may reload from carries the record.
        threat_model.audit_config = config
        checkpoint(threat_model, cache_dir)
    stage_cache.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    gen = threat_model.atk_know_gen.generator
    # A cache built at another eps would silently audit the wrong mechanism, and the
    # per-eps cache directory is the only thing preventing it. Verify rather than trust.
    cached_eps = getattr(gen, "epsilon", None)
    if cached_eps is not None and not np.isclose(float(cached_eps), epsilon):
        raise RuntimeError(
            f"{label}: cached threat model at {cache_dir} was built with "
            f"epsilon={cached_eps}, not {epsilon}. Delete that directory and re-run.")
    # check_config compares the recorded configuration; this compares the generator that
    # will actually do the fits, so the flag cannot be wrong in the record and right
    # in the pickle, or the reverse.
    gen_seed_fix = bool(getattr(gen, "seed_fix", False))
    if gen_seed_fix != config["seed_fix"]:
        raise ConfigMismatch(
            f"{label}: cached generator has seed_fix={gen_seed_fix}, this run has "
            f"seed_fix={config['seed_fix']}. Use a different --tag, or delete "
            f"{cache_dir} if it is yours to discard.")
    log.info(f"    seed_fix={gen_seed_fix}")
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
    threat_model = grow_pools(threat_model, cache_dir, chunk, oom_retries, oom_wait_s)
    n_after = len(threat_model._memory[True][0]) + len(threat_model._memory[False][0])
    pool_s = round(time.time() - t_pool, 1)
    log.info(f"    pools: {len(threat_model._memory[True][0])} train / "
             f"{len(threat_model._memory[False][0])} test  "
             f"(+{n_after - n_before} new fits, {pool_s:.0f}s)")
    checkpoint(threat_model, cache_dir)

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
        checkpoint(threat_model, cache_dir)

    out = pd.DataFrame(rows)
    out.to_csv(paths["effeps"], index=False)
    if score_rows:
        path = paths["raw_scores"]
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
        "seed_fix": config["seed_fix"],
        "attacks_without_raw_scores": no_scores,
        # On a RESUMED run the pools already exist, so pool_wall_clock_s covers only
        # the fits done in this invocation -- read it together with new_fits_this_run
        # before quoting total_wall_clock_s as the cost of the arm.
        "note": "pool_wall_clock_s and total_wall_clock_s cover this invocation only; "
                "see new_fits_this_run to tell a fresh arm from a resumed one",
        "config": config,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))

    # Defensive: the CSV and meta.json are already on disk by this point, so a
    # degenerate eps_low_95 column must not turn a finished 4-hour arm into a
    # traceback that main() records as a failure.
    arm = f"eps={epsilon:g}{f' tag={tag}' if tag else ''}"
    if "eps_low_95" in out.columns and out["eps_low_95"].notna().any():
        best = out.loc[out["eps_low_95"].idxmax()]
        log.info(f"=== {arm} done: worst-case eps_low_95={best['eps_low_95']:.3f} "
                 f"[{best['eps_low_95']:.3f}, {best['eps_high_95']:.3f}] via {best['attack']} "
                 f"({meta['total_wall_clock_s'] / 3600:.2f} h) ===")
    else:
        log.warning(f"=== {arm} done but no usable eps_low_95 across the 5 attacks "
                    f"({meta['total_wall_clock_s'] / 3600:.2f} h). Results are written; "
                    f"inspect {results_dir.relative_to(REPO_ROOT)} ===")


def _drop_file_handlers() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()


def dry_run(epsilons: list, options: dict) -> int:
    """Resolve everything a run would use and list every path it would write, then
    stop. Reads the data and, where one exists, the cached threat model; writes
    nothing (the log FileHandler has already been removed by main)."""
    tag = options.get("tag")
    rel = lambda p: p.relative_to(REPO_ROOT)                      # noqa: E731
    mark = lambda p: "EXISTS" if p.exists() else "new   "         # noqa: E731

    train_dataset, _, _ = common.load_adult_datasets()
    try:
        background, background_idx, target, alternate = resolve_pair(train_dataset, options)
    except ValueError as exc:
        print(f"[dry-run] REFUSED: {exc}")
        return 2
    np.random.seed(SCORE_ATTACK_SEED)
    attack_slugs = [slug(a.label) for a in common.build_attacks(target, background)]

    print(f"[dry-run] options: {json.dumps(options)}")
    print(f"[dry-run] {mark(log_path(tag))} {rel(log_path(tag))}   (log, appended)")
    for epsilon in epsilons:
        label = arm_label(epsilon, tag)
        paths = arm_paths(label)
        config = build_config(epsilon, options, background_idx, target, alternate)
        print(f"\n[dry-run] --- eps={epsilon:g}  label={label} ---")
        print("[dry-run] config " + json.dumps(config, indent=2).replace("\n", "\n[dry-run] "))
        files = [paths["cache_dir"], paths["threat_model"], paths["stage_cache"]]
        files += [paths["stage_cache"] / f"result_{s}.json" for s in attack_slugs]
        files += [paths["datasets"] / n for n in
                  ("scalers.csv", "synthetic_train.csv.gz", "synthetic_test.csv.gz")]
        files += [paths["results_dir"], paths["effeps"], paths["meta"]]
        files += [paths["results_dir"] / f"effective_epsilon_{s}.csv" for s in attack_slugs]
        files += [paths["raw_scores"]]
        for p in files:
            print(f"[dry-run]   {mark(p)} {rel(p)}")
        if paths["threat_model"].exists():
            cached = common.tm.ThreatModel.load(str(paths["cache_dir"] / "threat_model"))
            try:
                check_config(stored_config(cached, config), config, paths["threat_model"])
                n = (len(cached._memory[True][0]), len(cached._memory[False][0]))
                print(f"[dry-run]   would RESUME a matching cache ({n[0]} train / {n[1]} test "
                      f"already fitted)")
            except ConfigMismatch as exc:
                print(f"[dry-run]   would STOP: {exc}")
        else:
            print(f"[dry-run]   fresh arm: {NUM_TRAIN + NUM_TEST} fits, seeds "
                  f"{config['seed_base']}..{config['seed_base'] + NUM_TRAIN + NUM_TEST - 1}")
    print("\n[dry-run] nothing generated, nothing written.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epsilons", nargs="+", type=float, default=SWEEP_EPSILONS,
                        help=f"budgets to audit (default: {SWEEP_EPSILONS}; 1.0 is "
                             f"excluded because counts_sweep already has it)")
    parser.add_argument("--allow-degenerate", action="store_true",
                        help="record an eps whose pools failed the distinctness guard, "
                             "flagged as degenerate_pool (see the module docstring)")
    parser.add_argument("--tag", default=None,
                        help="appended to the label (eps1 -> eps1_<tag>) and so to every "
                             "cache, result and log path. Required by every option below. "
                             "Letters, digits and '-' only")
    parser.add_argument("--seed-base", type=int, default=None,
                        help=f"generator seed base: fit i runs at seed_base + i "
                             f"(default {TAPAS_GENERATOR_SEED_BASE})")
    parser.add_argument("--background-seed", type=int, default=None,
                        help=f"seed of the 499-record background (default {BACKGROUND_SEED})")
    parser.add_argument("--target-seed", type=int, default=None,
                        help=f"seed of the random t/t' draw (default {TARGET_SEED})")
    parser.add_argument("--target-index", type=int, default=None,
                        help="explicit train-set index of the target t (with --alternate-index)")
    parser.add_argument("--alternate-index", type=int, default=None,
                        help="explicit train-set index of the replacement t' (with --target-index)")
    parser.add_argument("--seed-fix", dest="seed_fix", action="store_const", const=True,
                        default=None,
                        help="every reseed inside a fit and its generate uses that fit's "
                             "seed, and generate() passes it to the plugin (default: off, "
                             "the old behaviour). Changes the mechanism's randomness, so "
                             "it requires --tag")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved configuration and every output path, "
                             "then exit without generating or writing anything")
    parser.add_argument("--chunk", type=int, default=100,
                        help="save the pools every N fits, so a crash costs at most one "
                             "chunk; even, since fits come in D+/D- pairs (default: 100)")
    parser.add_argument("--oom-retries", type=int, default=30,
                        help="retries of a chunk after CUDA out-of-memory (default: 30)")
    parser.add_argument("--oom-wait", type=int, default=120,
                        help="seconds between those retries (default: 120)")
    args = parser.parse_args()
    if args.chunk < 2 or args.chunk % 2:
        parser.error("--chunk must be an even number >= 2 (fits come in D+/D- pairs)")

    options = {k: getattr(args, k) for k in RUN_OPTIONS}
    overrides = [k for k in RUN_OPTIONS if k != "tag" and options[k] is not None]
    if overrides and options["tag"] is None:
        parser.error(", ".join("--" + k.replace("_", "-") for k in overrides)
                     + " changes what is audited but not the eps label, so it would share "
                       "(and resume) the untagged arm's cache and results. Add --tag.")
    if options["tag"] is not None and not TAG_PATTERN.fullmatch(options["tag"]):
        parser.error(f"--tag {options['tag']!r}: letters, digits and '-' only")
    if (options["target_index"] is None) != (options["alternate_index"] is None):
        parser.error("--target-index and --alternate-index must be given together")
    if options["target_index"] is not None and options["target_seed"] is not None:
        parser.error("--target-seed and --target-index/--alternate-index are alternatives; "
                     "give one or the other")

    if args.dry_run:
        _drop_file_handlers()
        return dry_run(args.epsilons, options)

    if options["tag"] is not None:
        # Before the first record, so nothing from a tagged run reaches the shared log.
        _drop_file_handlers()
        handler = logging.FileHandler(log_path(options["tag"]))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)

    if options["target_index"] is not None:
        # Refuse before any arm starts, rather than failing each arm in turn.
        try:
            resolve_pair(common.load_adult_datasets()[0], options)
        except ValueError as exc:
            log.error(f"refusing to run: {exc}")
            return 2

    if 1.0 in args.epsilons and options["tag"] is None:
        log.warning("eps=1.0 requested. That arm is already audited at 1000/2500 in "
                    "results/sample_size_sweep/dpgan/1000_2500/ with cache/dpgan/threat_model.pkl. "
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
            run_one_epsilon(epsilon, device_kwargs, args.allow_degenerate, options,
                            chunk=args.chunk, oom_retries=args.oom_retries,
                            oom_wait_s=args.oom_wait)
        except DegeneratePool as exc:
            log.error(f"DISTINCTNESS GUARD FAILED for eps={epsilon:g}:\n{exc}")
            failed.append((epsilon, "degenerate pool"))
        except ConfigMismatch as exc:
            log.error(f"CONFIG MISMATCH for eps={epsilon:g}, arm not run:\n{exc}")
            failed.append((epsilon, "config mismatch"))
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
