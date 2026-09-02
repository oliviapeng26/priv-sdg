#!/usr/bin/env python3
"""[3a] SmartNoise DP-CTGAN wrapped as a TAPAS Generator.

WHY THIS EXISTS
    Every diagnosis so far shares one component with the result being questioned:
    synthcity's DPGAN produced the data. Component [0] swapped out the attacks and
    [1] swaps out the bound, but neither touches the generator. This does.

    SmartNoise's DP-CTGAN is the genuine like-for-like: synthcity's "DPGAN" is itself
    CTGAN-architecture with DP-SGD on the discriminator, so this is the same idea in
    a completely separate codebase. If the eps=1.0 spike appears here too, it belongs
    to the method. If it does not, synthcity's implementation is implicated.

    Structurally this mirrors aim_audit/aim_generator.py, which already wraps a
    SmartNoise synthesiser for TAPAS. Same shape, same scaling round trip.

EPSILON DOES NOT MEAN THE SAME THING HERE -- READ THIS BEFORE COMPARING
    synthcity solves for the noise multiplier so that exactly eps is spent over
    n_iter epochs: eps is the input, sigma is derived.

    SmartNoise inverts that. sigma is FIXED (default 5) and eps is a STOPPING RULE:
    it trains, asks the accountant after each epoch how much has been spent, and
    breaks out when the spend exceeds the target (dpctgan.py, the `if self.epsilon <
    epsilon` branch). delta is not passed either -- it is derived internally as
    1/(n*sqrt(n)), which on a 500-row audit background is 8.9e-5.

    So "eps = 1.0" labels two different mechanisms. That is deliberate: the library
    is taken as it ships, exactly as AIM was, because the question is whether an
    independent implementation of DP-SGD-on-CTGAN shows the same behaviour -- not
    whether a re-parameterised synthcity does. A difference in result could be the
    accounting rather than a bug in either, and that belongs in the write-up.

    One consequence to watch: batch_size defaults to 500 and the audit background is
    500 rows, so the sampling rate is 1.0 and there is no privacy amplification from
    subsampling. The budget may be exhausted within a few epochs, leaving a barely
    trained generator. run_dpctgan_audit.py --probe reports how many epochs actually
    ran, and that number should be checked before committing to a full audit.

WHAT IS FIXED, AND ON WHOSE AUTHORITY
    Every DP-CTGAN parameter stays at the SmartNoise default except two:
      epsilon = 1.0   the budget under test
      verbose = False a logging flag, not a mechanism parameter; the default would
                      print per-epoch output across 3500 fits
    In particular sigma stays at 5, epochs at 300, batch_size at 500,
    max_per_sample_grad_norm at 1.0, pac at 1.

THE SCALING ROUND TRIP, AND WHY preprocessor_eps IS ZERO
    TAPAS min-max scales the continuous columns to [0,1] against the training split.
    DP-CTGAN needs its own transformer, and with preprocessor_eps = 0 it cannot spend
    budget learning column bounds -- so the bounds must be supplied. They come from
    sdg/aim.py's BIN_EDGES, which are fixed public-codebook constants (age 17-91,
    hours 1-100, and so on), never read off the training data. Same choice as AIM,
    for the same reason: no part of the eps = 1.0 budget leaks into preprocessing,
    so the reported epsilon is the whole story.

    Each fit therefore runs: unscale -> DP-CTGAN with public bounds -> sample ->
    rescale, returning data in the representation the threat model expects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tapas.datasets import TabularDataset
from tapas.generators import Generator

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT / "sdg"))

from config import CONTINUOUS_COLS, CATEGORICAL_COLS            # noqa: E402
# Public codebook bounds, imported rather than restated so they cannot drift from
# the values AIM was audited under.
from aim import BIN_EDGES                                        # noqa: E402

EPSILON = 1.0
# SmartNoise defaults, listed explicitly so the write-up can state them.
SIGMA = 5
EPOCHS = 300
BATCH_SIZE = 500
MAX_PER_SAMPLE_GRAD_NORM = 1.0


class DPCTGANGenerator(Generator):
    """SmartNoise DP-CTGAN as a TAPAS Generator.

    TAPAS calls this once per simulated dataset through Generator.__call__, which is
    `self.fit(dataset); return self.generate(n)` -- a full refit per simulation,
    exactly as for every other generator in the benchmark.
    """

    def __init__(self, description, scalers: dict, epsilon: float = EPSILON,
                 cuda: bool = True):
        super().__init__()
        self.description = description
        self.scalers = scalers          # {column: (min, max)} from common._fit_scalers
        self.epsilon = epsilon
        self.cuda = cuda
        self._synth = None
        self._fit_counter = 0
        # Diagnostics from the most recent fit, read by --probe: how many epochs
        # survived the budget, and what the accountant said was spent.
        self.last_epochs_run = None
        self.last_epsilon_spent = None

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

    def _transformer(self, columns):
        """A TableTransformer with PUBLIC bounds, so preprocessor_eps can stay 0.

        MinMaxTransformer(epsilon=0.0) requires lower/upper to be given rather than
        estimated. They come from the codebook bin edges, not from the data.
        """
        from snsynth.transform import (TableTransformer, MinMaxTransformer,
                                       LabelTransformer, OneHotEncoder,
                                       ChainTransformer)
        parts = []
        for col in columns:
            if col in CONTINUOUS_COLS:
                edges = BIN_EDGES[col]
                parts.append(MinMaxTransformer(lower=float(edges[0]),
                                               upper=float(edges[-1]),
                                               epsilon=0.0))
            else:
                # Chained, not bare. LabelTransformer alone hands the network an
                # integer code, so the generator's output comes back as a float and
                # its inverse does categories[np.float32] -> TypeError. The GAN-family
                # synthesisers expect categoricals one-hot encoded: label -> integer
                # code -> one-hot on the way in, and the inverse unwinds both.
                parts.append(ChainTransformer([LabelTransformer(), OneHotEncoder()]))
        return TableTransformer(parts)

    # -- TAPAS Generator interface ---------------------------------------

    def fit(self, dataset, **kwargs):
        from snsynth.pytorch.nn import DPCTGAN

        raw = self._unscale(dataset.data)
        cols = list(raw.columns)

        synth = DPCTGAN(
            epsilon=self.epsilon,
            sigma=SIGMA,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            max_per_sample_grad_norm=MAX_PER_SAMPLE_GRAD_NORM,
            cuda=self.cuda,
            verbose=False,          # logging only; the default spams across 3500 fits
        )
        synth.fit(raw, transformer=self._transformer(cols), preprocessor_eps=0.0,
                  categorical_columns=[c for c in cols if c in CATEGORICAL_COLS],
                  continuous_columns=[c for c in cols if c in CONTINUOUS_COLS])

        # DP-CTGAN stops early once the budget is spent, silently. Record how far it
        # actually got so a barely-trained generator cannot pass unnoticed.
        spent = getattr(synth, "epsilon_list", None) or getattr(
            getattr(synth, "gan", None), "epsilon_list", None)
        if spent:
            self.last_epochs_run = len(spent)
            self.last_epsilon_spent = float(spent[-1])

        self._synth = synth
        self._fit_counter += 1
        self.trained = True

    def generate(self, num_samples, random_state=None):
        assert self.trained, "Generator must be fit before generate()."
        sampled = self._synth.sample(num_samples)

        out = sampled[list(self.description.columns)].copy()
        out = self._rescale(out)
        out[CATEGORICAL_COLS] = out[CATEGORICAL_COLS].astype(str)
        out = out[list(self.description.columns)].reset_index(drop=True)
        return TabularDataset(out, self.description)

    @property
    def label(self):
        return "dpctgan"

    def __getstate__(self):
        # Same reason SynthcityGenerator and AIMGenerator drop theirs: a fitted torch
        # model does not pickle cleanly and would break ThreatModel.save(). TAPAS
        # refits per simulation, and the memoised datasets are unaffected.
        state = self.__dict__.copy()
        state["_synth"] = None
        state["trained"] = False
        return state
