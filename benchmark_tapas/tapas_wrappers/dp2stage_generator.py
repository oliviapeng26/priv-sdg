#!/usr/bin/env python3
"""DP-2Stage-O (Afonja et al., TMLR 2025) wrapped as a TAPAS Generator.

Structurally this mirrors tapas_wrappers/{great,aim,dpctgan}_generator.py and reuses
everything they reuse from benchmark_tapas/common.py. It is closest to GReaT's
wrapper (both serialise rows to text and need the scaling round trip and a
vocabulary check) and to DP-CTGAN's (both carry a formal epsilon that has to be
reported next to the empirical one). Read sdg/dp2stage_driver.py first: the method,
every fixed hyperparameter, and the four deliberate deviations from the released
scripts all live there, and this wrapper takes the model exactly as the driver runs
it for the utility/fidelity runs, so the audited model is the reported one.

WHY EVERY FIT IS A SUBPROCESS, AND WHAT THAT COSTS
    DP-2Stage needs python 3.9, torch 2.5.1, transformers 4.30.2 and opacus 1.4.1;
    the TAPAS audit runs in ~/priv-sdg/venv (python 3.10, torch 2.2.2). They cannot
    share a process, so each fit calls `~/venv-dp2stage/bin/python sdg/dp2stage_driver.py
    fit-sample` (override the interpreter with $DP2STAGE_PYTHON). Consequences:
      - GPU memory is freed by process exit, so the GReaT wrapper's del-model /
        empty_cache dance is unnecessary and VRAM cannot creep across thousands of fits.
      - every fit pays process start-up and loading the shared Stage 1 checkpoint
        (~0.5 GB from disk) on top of the DP fit itself. `--probe` measures the total.
      - the fitted model never exists in this process, so nothing needs pickling:
        __getstate__ only has to keep the seed counter.

TWO-STAGE STRUCTURE
    Stage 1 (Airline, non-DP) is trained ONCE and shared by every fit in the audit,
    as it is by the five utility/fidelity seeds. It never sees the audit's data, so it
    is a fixed public component of the mechanism and does not enter the DP accounting.
    Each fit is Stage 2 only: DP fine-tuning (eps=1, delta=1e-5, C=1) of a copy of that
    checkpoint on the ~500 rows TAPAS hands over, then sampling.

fit() DEFERS THE WORK TO generate()
    Generator.__call__ is `self.fit(dataset); return self.generate(n)`, and the driver
    fits and samples in one process (the fitted model is never written to disk), so
    fit() stashes the unscaled rows and generate() runs the subprocess. generate()
    called twice after one fit returns the cached result rather than refitting.

THE SEEDING FIX, WHICH IS LOAD-BEARING
    Fit i runs at TAPAS_GENERATOR_SEED_BASE + i as BOTH the Stage 2 training seed and
    the generation seed, so consecutive fits -- including the two halves of one D+/D-
    pair -- never share a draw (ExactDataKnowledge hands every simulation the SAME
    background, so a fixed seed would collapse the pool to two datasets, the
    pre-2026-08-23 Synthcity bug). Unlike GReaT there is no HF Trainer to reseed
    globally: ft_opacus.main() calls set_seed(args.seed) once. The two OTHER traps in
    this code are removed in the driver, not here: resume-from-checkpoint (every fit
    gets a fresh temp dir) and the saved generation RNG state (fresh save_name per
    round). The counter survives pickling. run_dp2stage_audit.py's distinctness guard
    is the regression test.

THE SCALING ROUND TRIP
    TAPAS min-max scales the 5 continuous columns to [0,1]. The text serialiser needs
    integers ("age is 47"), so each fit runs: unscale -> round to int -> DP-2Stage ->
    rescale. Rounding is not optional: unscaling through float arithmetic can land on
    46.99999999999999, which the driver (rightly) refuses to truncate. Output is NOT
    clipped to [0,1], as in the GReaT/AIM wrappers.

VOCABULARY
    The driver already restricts categoricals to the PUBLIC codebook (adult_train U
    adult_test, the same lists TAPAS gives every generator) and rejects malformed rows
    inside its sampling loop, refilling to the full quota. This wrapper re-checks the
    result against description[col]["representation"] and RAISES rather than dropping
    rows, so a mismatch between the two vocabularies cannot silently shrink a dataset.

EPSILON
    Every fit's spent epsilon is read back from Opacus's RDP accountant (via the driver)
    and must be <= 1.0 + 0.01 (its solver tolerance) or the fit is rejected as an invalid
    DP output. Per-fit records (spent epsilon, sigma, steps, timings, acceptance,
    peak GPU memory) are appended to `fit_log_path` (JSONL) so the audit can put the
    formal epsilon in meta.json next to TAPAS's empirical eps_low_95. delta is the
    paper's 1e-5; AIM uses 1e-9, DP-CTGAN 1/(n*sqrt(n)), DPGAN 1/n -- a stated
    limitation.

PROBE-DEPENDENT setting in this file: AUDIT_SAMPLE_BATCH (below). Grep PROBE-DEPENDENT.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tapas.datasets import TabularDataset
from tapas.generators import Generator

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import CATEGORICAL_COLS, CACHE_DIR                     # noqa: E402
from seeds import TAPAS_GENERATOR_SEED_BASE                         # noqa: E402
# Import, never copy: these keep the audited model identical to the one
# sdg/generate_dp2stage.py produced the utility and fidelity numbers with. The driver
# imports only stdlib + pandas at module level, so this works in the TAPAS venv.
sys.path.insert(0, str(REPO_ROOT / "sdg"))
from dp2stage_driver import (EPSILON, DELTA, INTEGER_COLS, STAGE1_DIR,    # noqa: E402
                             EXIT_HARD_FAILURE, EXIT_SHORTFALL, MAX_NEW_TOKENS)

DRIVER = REPO_ROOT / "sdg" / "dp2stage_driver.py"
# The interpreter with DP-2Stage's pinned stack. NOT this process's own.
DP2STAGE_PYTHON = Path(os.environ.get("DP2STAGE_PYTHON", "~/venv-dp2stage/bin/python")).expanduser()

# Opacus make_private_with_epsilon solves for sigma to within this of the target.
EPS_TOLERANCE = 0.01

# PROBE-DEPENDENT: rows per model.generate() call inside each 500-row audit fit. 100 is
# DP-2Stage's own default and has never been timed here. Each fit asks for NUM_SYNTHETIC
# (500) rows, so any value >= 500 is one call; the k sweep in
# `python sdg/dp2stage_probe.py fit` (rows/s and peak GPU memory at k=100/500/2000)
# says whether 500 is faster than 100 and whether it fits beside czha4500's job.
AUDIT_SAMPLE_BATCH = 100

_OOM_SIGNATURES = ("out of memory", "cuda error", "cublas", "cudnn", "nccl")


class HardSampleFailure(RuntimeError):
    """The fit/sample subprocess died in a way that looks transient (CUDA OOM from GPU
    contention, a zero-row generation round, or the process being killed) rather than
    the model genuinely failing to produce valid rows. run_dp2stage_audit.py maps this
    to EXIT_INCOMPLETE so the restart loop retries it without spending one of its
    3 consecutive-failure strikes; a plain RuntimeError (vocabulary mismatch, quota
    unfilled, overspent epsilon) is non-transient and does count."""


class DP2StageGenerator(Generator):
    """DP-2Stage-O as a TAPAS Generator (one Stage 2 fit + sample per simulation)."""

    def __init__(self, description, scalers: dict,
                 seed_base: int = TAPAS_GENERATOR_SEED_BASE,
                 stage1_dir=None, fit_log_path=None,
                 sample_batch: int = AUDIT_SAMPLE_BATCH):
        super().__init__()
        self.description = description
        self.scalers = scalers          # {column: (min, max)} from common._fit_scalers
        self.seed_base = seed_base
        self.stage1_dir = str(stage1_dir or STAGE1_DIR)
        self.fit_log_path = Path(fit_log_path) if fit_log_path else None
        self.sample_batch = sample_batch
        self._raw = None                # unscaled rows waiting for generate()
        self._synthetic = None
        self._fit_counter = 0
        # Diagnostics from the most recent fit, read by --probe.
        self.last_info = None

        # Valid values per categorical column, straight off the TAPAS schema.
        self._vocab = {c: set(description[c]["representation"]) for c in CATEGORICAL_COLS}

    # -- scaling round trip ----------------------------------------------

    def _unscale(self, df: pd.DataFrame) -> pd.DataFrame:
        """[0,1] -> original units, inverting common._apply_scalers."""
        out = df.copy()
        for col, (lo, hi) in self.scalers.items():
            span = hi - lo
            out[col] = out[col].astype(float) * span + lo if span > 0 else float(lo)
        return out

    def _rescale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Original units -> [0,1], matching common._apply_scalers exactly."""
        out = df.copy()
        for col, (lo, hi) in self.scalers.items():
            span = hi - lo
            out[col] = (out[col].astype(float) - lo) / span if span > 0 else 0.0
        return out

    # -- TAPAS Generator interface ---------------------------------------

    def fit(self, dataset, **kwargs):
        raw = self._unscale(dataset.data)
        # Unscaling goes through floats; round to the integers they were, or the
        # driver refuses to truncate "46.99999999999999".
        raw[INTEGER_COLS] = np.rint(raw[INTEGER_COLS].astype(float)).astype("int64")
        self._raw = raw[list(self.description.columns)]
        self._synthetic = None
        self._fit_counter += 1
        self.trained = True

    def _run_subprocess(self, scratch: Path, num_samples: int, seed: int):
        train_csv, out_csv = scratch / "private.csv", scratch / "synth.csv"
        info_json, driver_log = scratch / "info.json", scratch / "driver.log"
        self._raw.to_csv(train_csv, index=False)
        cmd = [str(DP2STAGE_PYTHON), "-u", str(DRIVER), "fit-sample",
               "--train-csv", str(train_csv), "--out-csv", str(out_csv),
               "--info-json", str(info_json), "--n-samples", str(num_samples),
               "--seed", str(seed), "--gen-seed", str(seed),
               "--sample-batch", str(self.sample_batch),
               "--stage1-dir", self.stage1_dir, "--scratch-root", str(scratch)]
        # Do not leak this venv's interpreter settings into the other one.
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
        with open(driver_log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        tail = "".join(driver_log.read_text(errors="replace").splitlines(keepends=True)[-25:])

        if rc == 0:
            return out_csv, json.loads(info_json.read_text())
        transient = (rc == EXIT_HARD_FAILURE or rc < 0
                     or any(s in tail.lower() for s in _OOM_SIGNATURES))
        msg = f"DP-2Stage fit-sample exited {rc} (seed {seed}). Last driver output:\n{tail}"
        if rc == EXIT_SHORTFALL:
            raise RuntimeError(msg + "\nquota not filled after the driver's generation rounds -- "
                               "a model problem, not contention; retrying fails the same way.")
        if transient:
            raise HardSampleFailure(msg + "\nlooks transient (CUDA OOM / zero-row round / killed) "
                                    "-- check nvidia-smi for another job.")
        raise RuntimeError(msg)

    def generate(self, num_samples, random_state=None):
        assert self.trained, "Generator must be fit before generate()."
        if self._synthetic is not None and len(self._synthetic) == num_samples:
            return TabularDataset(self._synthetic.copy(), self.description)

        fit_index = self._fit_counter - 1          # the counter was bumped in fit()
        seed = self.seed_base + fit_index
        scratch_root = CACHE_DIR / f".dp2stage_scratch_{os.getpid()}"
        scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="fit_", dir=scratch_root))
        try:
            out_csv, info = self._run_subprocess(scratch, num_samples, seed)
            if info["epsilon_spent"] > EPSILON + EPS_TOLERANCE:
                raise RuntimeError(
                    f"fit {fit_index} (seed {seed}): spent epsilon {info['epsilon_spent']:.4f} "
                    f"exceeds the {EPSILON} target -- invalid DP output, refusing to use it")
            out = pd.read_csv(out_csv)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)   # thousands of fits: never accumulate
            try:
                scratch_root.rmdir()                     # only succeeds when empty
            except OSError:
                pass

        if len(out) != num_samples:
            raise RuntimeError(f"fit {fit_index}: got {len(out)} rows, expected {num_samples}")
        bad = {c: int((~out[c].astype(str).isin(self._vocab[c])).sum()) for c in CATEGORICAL_COLS}
        bad = {c: n for c, n in bad.items() if n}
        if bad:
            raise RuntimeError(f"fit {fit_index}: rows outside the TAPAS vocabulary {bad}; the "
                               f"driver's public vocabulary and description[col]['representation'] "
                               f"have diverged")

        self.last_info = info
        self._record(fit_index, seed, info)

        out = self._rescale(out)
        out[CATEGORICAL_COLS] = out[CATEGORICAL_COLS].astype(str)
        out = out[list(self.description.columns)].reset_index(drop=True)
        self._synthetic = out
        return TabularDataset(out.copy(), self.description)

    def _record(self, fit_index: int, seed: int, info: dict):
        """Append this fit's formal-privacy and cost record (JSONL, one line per fit)."""
        if self.fit_log_path is None:
            return
        keep = ("epsilon_spent", "noise_multiplier", "dp_steps", "sample_rate", "fit_s", "gen_s",
                "acceptance", "peak_gpu_gb_fit", "peak_gpu_gb_gen", "n_returned")
        rec = {"fit_index": fit_index, "seed": seed, **{k: info.get(k) for k in keep}}
        self.fit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fit_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    @property
    def label(self):
        return "dp2stage"

    def __getstate__(self):
        # No fitted model ever lives in this process (the driver fits in a subprocess and
        # writes nothing back but the sample), so there is nothing heavy to drop -- only the
        # transient row buffers. _fit_counter and seed_base are deliberately NOT reset: a
        # resumed run must continue the seed sequence, not replay seeds already used.
        state = self.__dict__.copy()
        state["_raw"] = None
        state["_synthetic"] = None
        state["trained"] = False
        return state
