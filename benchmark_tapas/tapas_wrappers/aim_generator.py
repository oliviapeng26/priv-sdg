#!/usr/bin/env python3
"""A TAPAS Generator wrapping AIM, so the 5-attack MIA battery can audit it.

WHY THIS EXISTS
    common.SynthcityGenerator builds its model with `Plugins().get(method, ...)`.
    AIM is not a Synthcity plugin -- it is SmartNoise's AIMSynthesizer on top of
    mbi/jax -- so it needs its own Generator. Everything else in the audit
    (background, target/alternate, attack battery, caching, guard) is reused
    unchanged from benchmark_tapas/common.py.

    AIM IS TAKEN EXACTLY AS THE REPO ALREADY RUNS IT. epsilon, delta, degree,
    max_cells, max_model_size, rounds and the bin edges are all imported from
    sdg/aim.py rather than restated here, so the model audited for privacy is
    bit-for-bit the configuration whose utility and fidelity are reported. There
    is no tuning knob in this file, deliberately.

THE ONE PIECE OF REAL WORK: THE SCALING ROUND TRIP
    common.load_adult_datasets min-max scales the 5 continuous columns to [0,1]
    against the training split (common._fit_scalers), because that is the
    representation every other generator and every attack sees. But sdg/aim.py's
    BIN_EDGES are in ORIGINAL units (age 17-91, hours_per_week 1-100, a dedicated
    zero bin for capital_gain/loss). Handing scaled data straight to `encode`
    would push every row into bin 0 and silently produce a constant table.

    So each fit runs: unscale -> encode -> AIM -> decode -> rescale. The scalers
    are the same (min, max) pairs load_adult_datasets used, passed in by the run
    script, so the round trip is exact and the generator's output lands back in
    the representation the threat model expects.

    Decoded values stay inside the codebook range by construction (encode clips,
    decode draws within a bin), so the rescaled output stays in [0,1] wherever the
    codebook bound and the training bound agree. It is NOT clipped: the other
    generators' outputs are not clipped either, and forcing a bound here would
    make AIM the only method whose tails are truncated.

NO SEEDING, AND WHY THAT IS CORRECT HERE
    SynthcityGenerator runs fit i at TAPAS_GENERATOR_SEED_BASE + i, because
    Synthcity's Plugin.fit reseeds numpy/torch/random globally and identical input
    would otherwise give byte-identical output (the pre-2026-08-23 bug that made
    all 3500 simulations collapse to 2 distinct datasets).

    AIM has the opposite problem and therefore needs none of that machinery: its
    Gaussian measurements come from opendp's CSPRNG, which has no seeding hook by
    design, so every fit is an independent draw whether we like it or not. The
    collapse bug is structurally impossible here. The price is that an AIM audit
    is not bit-reproducible -- rerunning it gives different simulations. That is a
    property of the mechanism, not of this wrapper, and belongs in the writeup's
    limitations next to the delta mismatch.

    The decode-time uniform draw uses one Generator held on the instance, so it
    advances across fits rather than repeating a draw per simulation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tapas.datasets import TabularDataset
from tapas.generators import Generator

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT / "sdg"))

from config import CONTINUOUS_COLS, CATEGORICAL_COLS          # noqa: E402
# Import, never copy: these constants are what keep preprocessor_eps at 0 and the
# whole eps=1.0 budget on AIM itself.
from aim import (BIN_EDGES, encode, decode, EPSILON, DELTA,   # noqa: E402
                 DEGREE, MAX_CELLS, MAX_MODEL_SIZE, ROUNDS)


class AIMGenerator(Generator):
    """SmartNoise AIM as a TAPAS Generator.

    TAPAS calls this once per simulated dataset via Generator.__call__, which is
    `self.fit(dataset); return self.generate(n)` -- so every simulation is a full
    refit from scratch, exactly as for the Synthcity methods.
    """

    def __init__(self, description, scalers: dict,
                 epsilon: float = EPSILON, delta: float = DELTA):
        super().__init__()
        self.description = description
        self.scalers = scalers            # {column: (min, max)} from common._fit_scalers
        self.epsilon = epsilon
        self.delta = delta
        self._synth = None
        self._fit_counter = 0             # provenance only; AIM cannot be seeded
        self._rng = np.random.default_rng()

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
        from snsynth.aim import AIMSynthesizer

        raw = self._unscale(dataset.data)
        discrete = raw.copy()
        for col in CONTINUOUS_COLS:
            discrete[col] = encode(raw[col], BIN_EDGES[col])

        synth = AIMSynthesizer(
            epsilon=self.epsilon, delta=self.delta, rounds=ROUNDS, degree=DEGREE,
            max_cells=MAX_CELLS, max_model_size=MAX_MODEL_SIZE, verbose=False,
        )
        # Every column is discrete after encoding, and preprocessor_eps=0 because
        # the bin edges are public constants -- no budget is spent on binning.
        synth.fit(discrete, categorical_columns=list(discrete.columns),
                  preprocessor_eps=0.0)

        self._synth = synth
        self._fit_counter += 1
        self.trained = True

    def generate(self, num_samples, random_state=None):
        assert self.trained, "Generator must be fit before generate()."
        sampled = self._synth.sample(num_samples)

        out = sampled[list(self.description.columns)].copy()
        for col in CONTINUOUS_COLS:
            out[col] = decode(out[col].astype(int), BIN_EDGES[col], self._rng)
        out = self._rescale(out)
        out[CATEGORICAL_COLS] = out[CATEGORICAL_COLS].astype(str)
        out = out[list(self.description.columns)].reset_index(drop=True)
        return TabularDataset(out, self.description)

    @property
    def label(self):
        return "aim"

    def __getstate__(self):
        # Same reason SynthcityGenerator drops its plugin: a fitted AIMSynthesizer
        # holds jax/mbi state that pickle cannot serialise, which would break
        # ThreatModel.save(). TAPAS refits per simulation anyway, and the memoised
        # synthetic datasets -- the expensive part -- are unaffected.
        state = self.__dict__.copy()
        state["_synth"] = None
        state["trained"] = False
        return state
