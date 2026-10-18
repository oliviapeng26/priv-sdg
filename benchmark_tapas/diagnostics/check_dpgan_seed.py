#!/usr/bin/env python3
"""Does the per-fit seed reach DPGAN's training? Three fits on one 500-row dataset.

Fits, in order, with the audit's own SynthcityGenerator (DPGAN, eps = FORMAL_EPSILON,
n_iter = DPGAN_N_ITER, --device cpu|cuda, default cpu): random_state 1, 2, 1. A fresh generator per fit
with seed_base = s, so its first fit runs at random_state = seed_base + 0 = s.

Every call to synthcity's enable_reproducible_results is logged (seed, caller
file:line, order) by a pass-through wrapper installed in this process only. Opacus's
noise multiplier is captured the same way (wrapper around get_noise_multiplier).

Output: results/extras/dpgan_seed_check/
Run from the repo root, env active:
  CUDA_VISIBLE_DEVICES="" python benchmark_tapas/diagnostics/check_dpgan_seed.py
  CUDA_VISIBLE_DEVICES=0 python benchmark_tapas/diagnostics/check_dpgan_seed.py --device cuda
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

import functools                                                  # noqa: E402
import json                                                       # noqa: E402

import numpy as np                                                # noqa: E402

import common                                                     # noqa: E402
from config import (DPGAN_N_ITER, FORMAL_EPSILON, METHOD_CONFIG,  # noqa: E402
                    NUM_SYNTHETIC, RESULTS_DIR)

METHOD = "dpgan"
SEEDS = [1, 2, 1]
OUT_DIR = RESULTS_DIR / "extras" / "dpgan_seed_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Instrumentation (pass-through: log, then call the original unchanged) --

import synthcity.utils.reproducibility as _repro                  # noqa: E402
_ORIG_REPRO = _repro.enable_reproducible_results
_REPRO_LOG = []          # (seed, "file:line") in call order; reset per fit
_SIGMA_LOG = []          # noise multipliers returned to Opacus; reset per fit


def _short(path: str) -> str:
    return path.split("site-packages/")[-1]


@functools.wraps(_ORIG_REPRO)
def _logged_repro(*args, **kwargs):
    seed = args[0] if args else kwargs.get("random_state", 0)
    fr = sys._getframe(1)
    _REPRO_LOG.append((seed, f"{_short(fr.f_code.co_filename)}:{fr.f_lineno}"))
    return _ORIG_REPRO(*args, **kwargs)


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


def main() -> None:
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
    (OUT_DIR / "patched_sites.txt").write_text(
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
        _REPRO_LOG.clear()
        _SIGMA_LOG.clear()
        gen = common.SynthcityGenerator(METHOD, description, FORMAL_EPSILON,
                                        plugin_kwargs, seed_base=seed)
        gen.fit(dataset)
        n_fit_calls = len(_REPRO_LOG)
        synth = gen.generate(NUM_SYNTHETIC).data.copy()
        outputs.append(synth)
        logs.append(list(_REPRO_LOG))
        sigmas.append(list(_SIGMA_LOG))

        synth.to_csv(OUT_DIR / f"fit{k}_seed{seed}_synthetic.csv", index=False)
        lines = [f"fit {k}  random_state={seed}  "
                 f"(calls 1..{n_fit_calls} during fit, {n_fit_calls + 1}.. during generate)"]
        lines += [f"{i:>3}  seed={s}  {where}" for i, (s, where) in enumerate(_REPRO_LOG, 1)]
        (OUT_DIR / f"fit{k}_seed{seed}_reseed_log.txt").write_text("\n".join(lines) + "\n")

    # -- Printed output: numbers only --------------------------------------
    for k, seed in enumerate(SEEDS, start=1):
        print(f"[fit {k}] random_state={seed} reseed calls (order, seed, caller):")
        for i, (s, where) in enumerate(logs[k - 1], 1):
            print(f"  {i} {s} {where}")
        print(f"[fit {k}] opacus sigma: " + ", ".join(repr(x) for x in sigmas[k - 1]))

    summary = {"seeds": SEEDS, "sigma": sigmas, "comparisons": {}}
    num_cols = outputs[0].select_dtypes(include="number").columns.tolist()
    for name, (a, b) in {"fit1_vs_fit2": (0, 1), "fit1_vs_fit3": (0, 2)}.items():
        A, B = outputs[a], outputs[b]
        equal = bool(A.equals(B))
        same_shape = A.shape == B.shape
        row_match = (float((A.values == B.values).all(axis=1).mean())
                     if same_shape else float("nan"))
        print(f"[{name}] exactly_equal={equal} shapes={A.shape},{B.shape} "
              f"rows_matching_fraction={row_match!r}")
        cmp = {"exactly_equal": equal, "rows_matching_fraction": row_match, "columns": {}}
        if not equal:
            for c in num_cols:
                dm = float(A[c].mean() - B[c].mean())
                ds = float(A[c].std() - B[c].std())
                print(f"  {c} mean_diff={dm!r} std_diff={ds!r}")
                cmp["columns"][c] = {"mean_diff": dm, "std_diff": ds}
        summary["comparisons"][name] = cmp
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
