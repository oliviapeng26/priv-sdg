#!/usr/bin/env python3
"""Does the per-fit seed reach DPGAN's training? Three fits on one 500-row dataset.

Fits, in order, with the audit's own SynthcityGenerator (DPGAN, eps = FORMAL_EPSILON,
n_iter = DPGAN_N_ITER, --device cpu|cuda, default cpu): random_state 1, 2, 1. A fresh
generator per fit with seed_base = s, so its first fit runs at random_state = s.

--seed-fix builds the generators with seed_fix=True (common.forced_synthcity_seed);
without it the generators are built exactly as the audit builds them by default.

Every call to synthcity's enable_reproducible_results is logged by a pass-through
wrapper installed in this process only: the seed it was called with, the caller
file:line, the order, and a short hash of torch.get_rng_state() and
np.random.get_state() taken right AFTER the call. In --seed-fix mode the seed logged
is the one that reached synthcity (the override has already been applied), and the
caller is the original call site. A marker line is added at the first call of
GAN._train_epoch, i.e. where the training loop starts, with the RNG hashes at that
moment. Opacus's noise multiplier is captured the same way (wrapper around
get_noise_multiplier). Nothing is changed by any wrapper.

Output: results/extras/dpgan_seed_check/default/  or  .../seed_fix/
Run from the repo root, env active:
  CUDA_VISIBLE_DEVICES="" python benchmark_tapas/diagnostics/check_dpgan_seed.py
  CUDA_VISIBLE_DEVICES="" python benchmark_tapas/diagnostics/check_dpgan_seed.py --seed-fix
  (add --device cuda and CUDA_VISIBLE_DEVICES=<free card> to use the GPU)
"""

import os
import sys

# Parsed before torch/synthcity import: synthcity fixes its DEVICE at import, so for
# cpu the GPUs must be hidden first. For cuda, CUDA_VISIBLE_DEVICES is left as the
# caller set it (use it to pick the card).
DEVICE = "cuda" if "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1] == "cuda" else "cpu"
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import argparse                                                   # noqa: E402
import functools                                                  # noqa: E402
import hashlib                                                    # noqa: E402
import json                                                       # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

import common                                                     # noqa: E402
from config import (DPGAN_N_ITER, FORMAL_EPSILON, METHOD_CONFIG,  # noqa: E402
                    NUM_SYNTHETIC, RESULTS_DIR)

METHOD = "dpgan"
SEEDS = [1, 2, 1]
BASE_DIR = RESULTS_DIR / "extras" / "dpgan_seed_check"

# -- Instrumentation (pass-through: log, then call the original unchanged) --

import synthcity.utils.reproducibility as _repro                  # noqa: E402
_ORIG_REPRO = _repro.enable_reproducible_results
# One list per fit, in order. Entries: (kind, seed, where, torch_hash, numpy_hash)
# with kind "reseed" or "train_start" (seed None).
_LOG = []
_SIGMA_LOG = []          # noise multipliers returned to Opacus; reset per fit
_TRAIN_SEEN = {"flag": False}


def _short(path: str) -> str:
    return path.split("site-packages/")[-1]


def _rng_hashes() -> tuple:
    t = hashlib.sha1(torch.get_rng_state().numpy().tobytes()).hexdigest()[:10]
    st = np.random.get_state()
    n = hashlib.sha1(st[1].tobytes() + repr((st[2], st[3], st[4])).encode()).hexdigest()[:10]
    return t, n


@functools.wraps(_ORIG_REPRO)
def _logged_repro(*args, **kwargs):
    seed = args[0] if args else kwargs.get("random_state", 0)
    fr = sys._getframe(1)
    # In --seed-fix mode the direct caller is common.forced_synthcity_seed's override;
    # report the call site behind it.
    while (fr is not None and fr.f_code.co_name == "forced"
           and fr.f_code.co_filename.endswith("common.py")):
        fr = fr.f_back
    out = _ORIG_REPRO(*args, **kwargs)
    th, nh = _rng_hashes()
    _LOG.append(("reseed", seed, f"{_short(fr.f_code.co_filename)}:{fr.f_lineno}", th, nh))
    return out


def install_repro_wrapper() -> list:
    """Replace the function in its home module and in every loaded module that
    imported it by name. Returns the names of the modules patched."""
    patched = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            if getattr(mod, "enable_reproducible_results", None) is _ORIG_REPRO:
                mod.enable_reproducible_results = _logged_repro
                patched.append(name)
        except Exception:
            pass
    return patched


def install_train_marker() -> None:
    """Log the RNG hashes at the first GAN._train_epoch call of each fit."""
    from synthcity.plugins.core.models.gan import GAN
    orig = GAN._train_epoch

    @functools.wraps(orig)
    def marked(self, *args, **kwargs):
        if not _TRAIN_SEEN["flag"]:
            _TRAIN_SEEN["flag"] = True
            th, nh = _rng_hashes()
            _LOG.append(("train_start", None, "GAN._train_epoch (first call)", th, nh))
        return orig(self, *args, **kwargs)

    GAN._train_epoch = marked


def install_sigma_wrapper() -> list:
    import opacus.privacy_engine as pe
    import opacus.accountants.utils as au
    orig = au.get_noise_multiplier
    patched = []

    @functools.wraps(orig)
    def logged(*args, **kwargs):
        sigma = orig(*args, **kwargs)
        _SIGMA_LOG.append(float(sigma))
        return sigma

    for mod in (au, pe):
        if getattr(mod, "get_noise_multiplier", None) is orig:
            mod.get_noise_multiplier = logged
            patched.append(mod.__name__)
    return patched


def last_reseed_before_training(log: list):
    """(seed, where, torch_hash, numpy_hash) of the last reseed before train_start."""
    last = None
    for kind, seed, where, th, nh in log:
        if kind == "train_start":
            return last
        last = (seed, where, th, nh)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-fix", action="store_true",
                    help="build the generators with seed_fix=True")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="(read before imports; listed here for --help)")
    args = ap.parse_args()
    mode = "seed_fix" if args.seed_fix else "default"
    out_dir = BASE_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up: import everything the DPGAN fit imports (lazy imports included) BEFORE
    # patching, so "imported by name" sites exist and can be found. This instance
    # is discarded and its reseed calls are not logged.
    from synthcity.plugins import Plugins
    Plugins().get(METHOD, device=DEVICE, epsilon=FORMAL_EPSILON)
    import synthcity.plugins.core.plugin                          # noqa: F401
    import synthcity.plugins.core.models.gan                      # noqa: F401
    import synthcity.plugins.core.models.mlp                      # noqa: F401
    import synthcity.plugins.core.models.tabular_gan              # noqa: F401
    import synthcity.plugins.privacy.plugin_dpgan                 # noqa: F401
    import synthcity.metrics.weighted_metrics                     # noqa: F401
    import synthcity.metrics.eval_detection                       # noqa: F401

    patched_repro = install_repro_wrapper()
    patched_sigma = install_sigma_wrapper()
    install_train_marker()
    (out_dir / "patched_sites.txt").write_text(
        "enable_reproducible_results:\n  " + "\n  ".join(patched_repro)
        + "\nget_noise_multiplier:\n  " + "\n  ".join(patched_sigma) + "\n")

    # Same 500 rows for every fit: default-seed background (499) + the fixed target,
    # exactly the D+ world SwapMIALabeller builds.
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, _alternate = common.select_random_target(train_dataset, background_idx)
    dataset = background.copy()
    dataset.add_records(target, in_place=True)
    assert len(dataset.data) == NUM_SYNTHETIC, len(dataset.data)

    plugin_kwargs = {**METHOD_CONFIG[METHOD]["plugin_kwargs"], "device": DEVICE}
    assert plugin_kwargs["n_iter"] == DPGAN_N_ITER

    outputs, logs, sigmas = [], [], []
    for k, seed in enumerate(SEEDS, start=1):
        _LOG.clear()
        _SIGMA_LOG.clear()
        _TRAIN_SEEN["flag"] = False
        gen = common.SynthcityGenerator(METHOD, description, FORMAL_EPSILON,
                                        plugin_kwargs, seed_base=seed,
                                        seed_fix=args.seed_fix)
        gen.fit(dataset)
        n_fit_calls = sum(1 for e in _LOG if e[0] == "reseed")
        synth = gen.generate(NUM_SYNTHETIC).data.copy()
        outputs.append(synth)
        logs.append(list(_LOG))
        sigmas.append(list(_SIGMA_LOG))

        synth.to_csv(out_dir / f"fit{k}_seed{seed}_synthetic.csv", index=False)
        lines = [f"mode={mode}  fit {k}  random_state={seed}  "
                 f"(reseed calls 1..{n_fit_calls} during fit, {n_fit_calls + 1}.. during generate)"]
        i = 0
        for kind, s, where, th, nh in _LOG:
            if kind == "reseed":
                i += 1
                lines.append(f"{i:>3}  seed={s}  torch={th} numpy={nh}  {where}")
            else:
                lines.append(f"     -- training loop starts: torch={th} numpy={nh}  ({where})")
        (out_dir / f"fit{k}_seed{seed}_reseed_log.txt").write_text("\n".join(lines) + "\n")

    # -- Printed output: numbers only --------------------------------------
    print(f"mode={mode} device={DEVICE}")
    # (a) ordered reseed calls
    for k, seed in enumerate(SEEDS, start=1):
        print(f"[a][fit {k}] random_state={seed} reseed calls (order, seed, caller, torch, numpy):")
        i = 0
        for kind, s, where, th, nh in logs[k - 1]:
            if kind == "reseed":
                i += 1
                print(f"  {i} {s} {where} {th} {nh}")
            else:
                print(f"  -- training loop starts {th} {nh}")
    # (b) hash after the last reseed before the training loop
    last = [last_reseed_before_training(lg) for lg in logs]
    for k, entry in enumerate(last, start=1):
        if entry is None:
            print(f"[b][fit {k}] last reseed before training loop: not found")
        else:
            print(f"[b][fit {k}] last reseed before training loop: seed={entry[0]} "
                  f"caller={entry[1]} torch={entry[2]} numpy={entry[3]}")
    hash_match = {}
    for name, (a, b) in {"fit1_vs_fit2": (0, 1), "fit1_vs_fit3": (0, 2)}.items():
        if last[a] is None or last[b] is None:
            hash_match[name] = None
            print(f"[b][{name}] hashes match: not found")
        else:
            hash_match[name] = {"torch": last[a][2] == last[b][2],
                                "numpy": last[a][3] == last[b][3]}
            print(f"[b][{name}] hashes match: torch={hash_match[name]['torch']} "
                  f"numpy={hash_match[name]['numpy']}")
    # (d) sigma
    for k in range(1, len(SEEDS) + 1):
        print(f"[d][fit {k}] opacus sigma: " + ", ".join(repr(x) for x in sigmas[k - 1]))

    # (c) output comparison
    summary = {"mode": mode, "device": DEVICE, "seeds": SEEDS, "sigma": sigmas,
               "last_reseed_before_training": last, "hash_match": hash_match,
               "comparisons": {}}
    num_cols = outputs[0].select_dtypes(include="number").columns.tolist()
    for name, (a, b) in {"fit1_vs_fit2": (0, 1), "fit1_vs_fit3": (0, 2)}.items():
        A, B = outputs[a], outputs[b]
        equal = bool(A.equals(B))
        same_shape = A.shape == B.shape
        row_match = (float((A.values == B.values).all(axis=1).mean())
                     if same_shape else float("nan"))
        print(f"[c][{name}] exactly_equal={equal} shapes={A.shape},{B.shape} "
              f"rows_matching_fraction={row_match!r}")
        cmp = {"exactly_equal": equal, "rows_matching_fraction": row_match, "columns": {}}
        if not equal:
            for c in num_cols:
                dm = float(A[c].mean() - B[c].mean())
                ds = float(A[c].std() - B[c].std())
                print(f"  {c} mean_diff={dm!r} std_diff={ds!r}")
                cmp["columns"][c] = {"mean_diff": dm, "std_diff": ds}
        summary["comparisons"][name] = cmp
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
