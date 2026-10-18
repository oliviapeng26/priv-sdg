#!/usr/bin/env python3
"""Do DPGAN fits at eps 0.999 / 1.0 / 1.001 give identical synthetic data?

Three fits on one 500-row dataset (default-seed background of 499 + the fixed target,
as in check_dpgan_seed.py), with the audit's own SynthcityGenerator (DPGAN,
n_iter = DPGAN_N_ITER, --device cpu|cuda, default cpu). A fresh generator per fit,
all with seed_base = 1, so every fit runs at random_state = 1. Only epsilon differs
(--epsilons, default 0.999 1.0 1.001). seed_fix=True unless --no-seed-fix.

Pass-through wrappers (installed in this process only, nothing is changed):
  - opacus PrivacyEngine.make_private_with_epsilon: records target_epsilon,
    target_delta, epochs, and the noise_multiplier of the optimizer it returns.
    (It returns (model, optimizer, data_loader), not sigma, so the value used is read
    from the returned optimizer's noise_multiplier attribute.)
  - opacus get_noise_multiplier: records every sigma it returns.
  - synthcity enable_reproducible_results: seed, caller file:line, RNG hashes after.
  - GAN._train_epoch first call: RNG hashes where the training loop starts.

Output: results/extras/dpgan_eps_nudge_check/seed_fix/  or  .../default/
Run from the repo root, env active:
  CUDA_VISIBLE_DEVICES="" python benchmark_tapas/diagnostics/check_dpgan_eps_nudge.py
  CUDA_VISIBLE_DEVICES="" python benchmark_tapas/diagnostics/check_dpgan_eps_nudge.py --no-seed-fix
"""

import os
import sys

# Parsed before torch/synthcity import: synthcity fixes its DEVICE at import, so for
# cpu the GPUs must be hidden first.
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
from config import (DPGAN_N_ITER, METHOD_CONFIG, NUM_SYNTHETIC,  # noqa: E402
                    RESULTS_DIR)

METHOD = "dpgan"
SEED_BASE = 1
DEFAULT_EPSILONS = [0.999, 1.0, 1.001]
BASE_DIR = RESULTS_DIR / "extras" / "dpgan_eps_nudge_check"

# -- Instrumentation (pass-through: log, then call the original unchanged) --

import synthcity.utils.reproducibility as _repro                  # noqa: E402
_ORIG_REPRO = _repro.enable_reproducible_results
# Entries: (kind, seed, where, torch_hash, numpy_hash); kind "reseed" or "train_start".
_LOG = []
_SIGMA_LOG = []          # get_noise_multiplier returns; reset per fit
_MPE_LOG = []            # make_private_with_epsilon calls; reset per fit
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
    # Under seed_fix the direct caller is common.forced_synthcity_seed's override;
    # report the call site behind it.
    while (fr is not None and fr.f_code.co_name == "forced"
           and fr.f_code.co_filename.endswith("common.py")):
        fr = fr.f_back
    out = _ORIG_REPRO(*args, **kwargs)
    th, nh = _rng_hashes()
    _LOG.append(("reseed", seed, f"{_short(fr.f_code.co_filename)}:{fr.f_lineno}", th, nh))
    return out


def install_repro_wrapper() -> list:
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


def install_mpe_wrapper() -> list:
    from opacus import PrivacyEngine
    orig = PrivacyEngine.make_private_with_epsilon

    @functools.wraps(orig)
    def logged(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        # Returns (model, optimizer, data_loader); the optimizer holds the sigma used.
        nm = getattr(out[1], "noise_multiplier", None)
        _MPE_LOG.append({
            "target_epsilon": kwargs.get("target_epsilon"),
            "target_delta": kwargs.get("target_delta"),
            "epochs": kwargs.get("epochs"),
            "noise_multiplier": None if nm is None else float(nm),
        })
        return out

    PrivacyEngine.make_private_with_epsilon = logged
    return ["opacus.PrivacyEngine.make_private_with_epsilon"]


def train_start_hashes(log: list):
    """(torch_hash, numpy_hash) at the training-loop start marker, or None."""
    for kind, _s, _w, th, nh in log:
        if kind == "train_start":
            return th, nh
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-seed-fix", action="store_true",
                    help="build the generators with seed_fix=False (default: True)")
    ap.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS,
                    help="epsilons of the fits, in order")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="(read before imports; listed here for --help)")
    args = ap.parse_args()
    seed_fix = not args.no_seed_fix
    epsilons = list(args.epsilons)
    mode = "seed_fix" if seed_fix else "default"
    out_dir = BASE_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up: import everything the DPGAN fit imports BEFORE patching, so
    # "imported by name" sites exist and can be found. Not logged.
    from synthcity.plugins import Plugins
    Plugins().get(METHOD, device=DEVICE, epsilon=epsilons[0])
    import synthcity.plugins.core.plugin                          # noqa: F401
    import synthcity.plugins.core.models.gan                      # noqa: F401
    import synthcity.plugins.core.models.mlp                      # noqa: F401
    import synthcity.plugins.core.models.tabular_gan              # noqa: F401
    import synthcity.plugins.privacy.plugin_dpgan                 # noqa: F401
    import synthcity.metrics.weighted_metrics                     # noqa: F401
    import synthcity.metrics.eval_detection                       # noqa: F401

    patched_repro = install_repro_wrapper()
    patched_sigma = install_sigma_wrapper()
    patched_mpe = install_mpe_wrapper()
    install_train_marker()
    (out_dir / "patched_sites.txt").write_text(
        "enable_reproducible_results:\n  " + "\n  ".join(patched_repro)
        + "\nget_noise_multiplier:\n  " + "\n  ".join(patched_sigma)
        + "\nmake_private_with_epsilon:\n  " + "\n  ".join(patched_mpe) + "\n")

    # Same 500 rows for every fit: default-seed background (499) + the fixed target.
    train_dataset, _, description = common.load_adult_datasets()
    background, background_idx = common.sample_background(train_dataset)
    target, _alternate = common.select_random_target(train_dataset, background_idx)
    dataset = background.copy()
    dataset.add_records(target, in_place=True)
    assert len(dataset.data) == NUM_SYNTHETIC, len(dataset.data)

    plugin_kwargs = {**METHOD_CONFIG[METHOD]["plugin_kwargs"], "device": DEVICE}
    assert plugin_kwargs["n_iter"] == DPGAN_N_ITER

    outputs, logs, sigmas, mpes = [], [], [], []
    for k, eps in enumerate(epsilons, start=1):
        _LOG.clear()
        _SIGMA_LOG.clear()
        _MPE_LOG.clear()
        _TRAIN_SEEN["flag"] = False
        gen = common.SynthcityGenerator(METHOD, description, eps, plugin_kwargs,
                                        seed_base=SEED_BASE, seed_fix=seed_fix)
        gen.fit(dataset)
        n_fit_calls = sum(1 for e in _LOG if e[0] == "reseed")
        synth = gen.generate(NUM_SYNTHETIC).data.copy()
        outputs.append(synth)
        logs.append(list(_LOG))
        sigmas.append(list(_SIGMA_LOG))
        mpes.append([dict(m) for m in _MPE_LOG])

        tag = f"fit{k}_eps{eps}"
        synth.to_csv(out_dir / f"{tag}_synthetic.csv", index=False)
        lines = [f"mode={mode}  fit {k}  epsilon={eps!r}  random_state={SEED_BASE}  "
                 f"(reseed calls 1..{n_fit_calls} during fit, {n_fit_calls + 1}.. during generate)"]
        for m in mpes[-1]:
            lines.append(f"make_private_with_epsilon: {m}")
        lines.append("get_noise_multiplier returns: " + ", ".join(repr(x) for x in sigmas[-1]))
        i = 0
        for kind, s, where, th, nh in _LOG:
            if kind == "reseed":
                i += 1
                lines.append(f"{i:>3}  seed={s}  torch={th} numpy={nh}  {where}")
            else:
                lines.append(f"     -- training loop starts: torch={th} numpy={nh}  ({where})")
        (out_dir / f"{tag}_log.txt").write_text("\n".join(lines) + "\n")

    # -- Printed output: numbers only --------------------------------------
    print(f"mode={mode} device={DEVICE} seed_base={SEED_BASE} epsilons={epsilons!r}")
    per_fit = []
    for k, eps in enumerate(epsilons, start=1):
        seeds = [e[1] for e in logs[k - 1] if e[0] == "reseed"]
        rec = {"epsilon_requested": eps, "make_private_with_epsilon": mpes[k - 1],
               "get_noise_multiplier": sigmas[k - 1], "reseed_seeds": seeds,
               "reseed_log": [list(e) for e in logs[k - 1]]}
        per_fit.append(rec)
        print(f"[1][fit {k}] epsilon_requested={eps!r}")
        for m in mpes[k - 1]:
            print(f"  opacus target_epsilon={m['target_epsilon']!r} "
                  f"target_delta={m['target_delta']!r} epochs={m['epochs']!r} "
                  f"noise_multiplier(returned optimizer)={m['noise_multiplier']!r}")
        print("  get_noise_multiplier returns: " + ", ".join(repr(x) for x in sigmas[k - 1]))
        print("  reseed seeds (in order): " + ", ".join(repr(s) for s in seeds))
        i = 0
        for kind, s, where, th, nh in logs[k - 1]:
            if kind == "reseed":
                i += 1
                print(f"  {i} {s} {where} {th} {nh}")
            else:
                print(f"  -- training loop starts {th} {nh}")

    # 2. sigma identical across fits (both sources)
    sigma_identical = {
        "get_noise_multiplier": all(s == sigmas[0] for s in sigmas),
        "make_private_with_epsilon": all(
            [m["noise_multiplier"] for m in x] == [m["noise_multiplier"] for m in mpes[0]]
            for x in mpes),
    }
    print(f"[2] sigma identical across fits: get_noise_multiplier="
          f"{sigma_identical['get_noise_multiplier']} "
          f"make_private_with_epsilon={sigma_identical['make_private_with_epsilon']}")

    pairs = [(a, b) for a, b in [(0, 1), (1, 2), (0, 2)] if b < len(epsilons)]

    # 3. pairwise dataframe comparison
    summary = {"mode": mode, "device": DEVICE, "seed_base": SEED_BASE,
               "seed_fix": seed_fix, "epsilons": epsilons, "fits": per_fit,
               "sigma_identical": sigma_identical, "comparisons": {},
               "train_start_hashes": [], "train_start_hash_match": {}}
    for a, b in pairs:
        name = f"eps{epsilons[a]}_vs_eps{epsilons[b]}"
        A, B = outputs[a], outputs[b]
        equal = bool(A.equals(B))
        same_shape = A.shape == B.shape
        row_match = (float((A.values == B.values).all(axis=1).mean())
                     if same_shape else float("nan"))
        print(f"[3][{name}] exactly_equal={equal} shapes={A.shape},{B.shape} "
              f"rows_matching_fraction={row_match!r}")
        cmp = {"exactly_equal": equal, "shapes": [list(A.shape), list(B.shape)],
               "rows_matching_fraction": row_match, "columns": {}}
        if not equal and same_shape:
            for c in A.select_dtypes(include="number").columns:
                dm = float(A[c].mean() - B[c].mean())
                ds = float(A[c].std() - B[c].std())
                print(f"  {c} mean_diff={dm!r} std_diff={ds!r}")
                cmp["columns"][c] = {"mean_diff": dm, "std_diff": ds}
        summary["comparisons"][name] = cmp

    # 4. RNG hashes at the start of training
    starts = [train_start_hashes(lg) for lg in logs]
    summary["train_start_hashes"] = starts
    for k, h in enumerate(starts, start=1):
        print(f"[4][fit {k}] training loop start: "
              + ("not found" if h is None else f"torch={h[0]} numpy={h[1]}"))
    for a, b in pairs:
        name = f"fit{a + 1}_vs_fit{b + 1}"
        if starts[a] is None or starts[b] is None:
            summary["train_start_hash_match"][name] = None
            print(f"[4][{name}] hashes match: not found")
        else:
            m = {"torch": starts[a][0] == starts[b][0], "numpy": starts[a][1] == starts[b][1]}
            summary["train_start_hash_match"][name] = m
            print(f"[4][{name}] hashes match: torch={m['torch']} numpy={m['numpy']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
