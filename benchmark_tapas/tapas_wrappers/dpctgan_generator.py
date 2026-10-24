#!/usr/bin/env python3
"""SmartNoise DP-CTGAN wrapped as a TAPAS Generator.

THE ONE COPY. Both audit scripts import this class rather than restating it:
benchmark_tapas/audits/run_dpctgan_audit.py (the counts sweep and the epoch_cap
sweep) and benchmark_tapas/archive/dp_ctgan_sep2_attack_rerun/run_dpctgan_audit_sep2.py
(the completed eps sweep). That matters more than tidiness here -- the epoch_cap
sweep is read against the committed epoch_cap=300 arm the spike diagnosis produced,
so if the two scripts fitted different generators the comparison would be void.

Structurally this mirrors tapas_wrappers/aim_generator.py, which already wraps a
SmartNoise synthesiser for TAPAS. Same shape, same scaling round trip.

WHY IT EXISTS AT ALL
    Every diagnosis so far shares one component with the result being questioned:
    synthcity's DPGAN produced the data. Component [0] swapped out the attacks and
    [1] swaps out the bound, but neither touches the generator. This does.

    SmartNoise's DP-CTGAN is the genuine like-for-like: synthcity's "DPGAN" is itself
    CTGAN-architecture with DP-SGD on the discriminator, so this is the same idea in
    a completely separate codebase. If the eps=1.0 spike appears here too, it belongs
    to the method. If it does not, synthcity's implementation is implicated.

    Structurally this mirrors tapas_wrappers/aim_generator.py, which already wraps a
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
    Every DP-CTGAN parameter stays at the SmartNoise default except three:
      epsilon = 1.0   the budget under test
      epochs  = 300   the SmartNoise default, and the default here; the epoch_cap
                      sweep is the one experiment that moves it, and it moves
                      nothing else
      verbose = False a logging flag, not a mechanism parameter; the default would
                      print per-epoch output across 3500 fits
    In particular sigma stays at 5, batch_size at 500, max_per_sample_grad_norm at
    1.0, pac at 1.

    The exception is the reviewer-response tuning arms, which move ONE of these per
    arm and say so in their tag: batch_size (250, 100), or sigma calibrated to spend
    eps over all 300 epochs (calibrate=True). They exist to show the benchmark's
    conclusions do not hinge on the defaults, not to find a better setting. At the
    defaults this class fits exactly what every committed result was produced under.

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
EPOCHS = 300            # the DEFAULT epoch cap; overridable per instance, see below
BATCH_SIZE = 500
MAX_PER_SAMPLE_GRAD_NORM = 1.0


# -- The accountant, reproduced (tuning arms only) ------------------------
#
# snsynth 1.0.8 DPCTGAN.train builds an opacus 0.14 PrivacyEngine with
# sample_rate = batch_size / n, the alphas below, and delta = 1/(n*sqrt(n)), and
# calls optimizerD.step() TWICE per iteration (once after the fake batch, once after
# the real one). The engine counts every step(), so one epoch is 2 * (n // batch_size)
# privacy steps, not n // batch_size. Checked 2026-10-25 against every logged spend:
# 2 steps per iteration reproduces 0.99996 / 1.4735 / 10.066 (500 rows) and 1.00379 /
# 3.98199 (21,523 rows) bit-for-bit; 1 step per iteration matches none of them.
#
# Needs opacus 0.14 (opacus.privacy_analysis) -- the same env DP-CTGAN itself needs.
# MUST MATCH sdg/generate_smartnoise.py's copy, which is duplicated rather than
# imported because importing this module from sdg/ would pull in TAPAS.
ACCOUNTANT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def accountant_epsilon(n: int, batch_size: int, sigma: float, epochs: int) -> float:
    """What the snsynth accountant reports after `epochs` completed epochs."""
    from opacus import privacy_analysis
    steps = 2 * epochs * max(n // batch_size, 1)
    rdp = np.asarray(privacy_analysis.compute_rdp(
        batch_size / n, sigma, 1, ACCOUNTANT_ALPHAS)) * steps
    return float(privacy_analysis.get_privacy_spent(
        ACCOUNTANT_ALPHAS, rdp, 1 / (n * np.sqrt(n)))[0])


def calibrate_sigma(n: int, batch_size: int, epochs: int, epsilon: float) -> float:
    """The smallest sigma whose spend after ALL `epochs` epochs is <= epsilon.

    This turns eps from a stopping rule into a calibrated budget, as synthcity does
    for DPGAN. The library still checks the accountant at the top of each epoch, but
    epoch e's check sees the cost of e < epochs epochs, which is below the full
    spend and so below epsilon: the break never fires and every epoch trains.
    Bisection on a monotone function; returns the upper bracket, so the result
    never overspends.
    """
    hi = float(SIGMA)
    while accountant_epsilon(n, batch_size, hi, epochs) > epsilon:
        hi *= 2
        if hi > 1e4:
            raise ValueError(f"no sigma reaches eps={epsilon} at n={n}: the accountant's "
                             f"zero-step floor at delta=1/(n*sqrt(n)) is above it")
    lo = hi / 2
    while accountant_epsilon(n, batch_size, lo, epochs) <= epsilon:
        hi, lo = lo, lo / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if accountant_epsilon(n, batch_size, mid, epochs) <= epsilon:
            hi = mid
        else:
            lo = mid
    return hi


class DPCTGANGenerator(Generator):
    """SmartNoise DP-CTGAN as a TAPAS Generator.

    TAPAS calls this once per simulated dataset through Generator.__call__, which is
    `self.fit(dataset); return self.generate(n)` -- a full refit per simulation,
    exactly as for every other generator in the benchmark.
    """

    def __init__(self, description, scalers: dict, epsilon: float = EPSILON,
                 cuda: bool = True, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                 sigma: float = SIGMA, calibrate: bool = False):
        super().__init__()
        if batch_size % 2:
            raise ValueError(f"batch_size={batch_size}: snsynth asserts an even batch size")
        self.description = description
        self.scalers = scalers          # {column: (min, max)} from common._fit_scalers
        self.epsilon = epsilon
        self.cuda = cuda
        # The epoch CAP, not a schedule. Which of the two binds -- the cap or the
        # budget -- is the question the epoch_cap sweep exists to answer, and
        # last_epochs_run below is the answer for any single fit.
        self.epochs = epochs
        # The reviewer-response tuning knobs, one per arm (see run_dpctgan_audit.py's
        # TUNING ARMS). At their defaults the fit is exactly the committed one.
        # calibrate=True ignores `sigma` and solves for it per training size n, so
        # that eps is spent over all `epochs` rather than used as a stopping rule.
        self.batch_size = batch_size
        self.sigma = sigma
        self.calibrate = calibrate
        self._calibrated_sigma = {}     # n -> sigma; the audit has one n (500)
        self._synth = None
        self._fit_counter = 0
        # Diagnostics from the most recent fit, read by --probe: how many epochs
        # survived the budget, and what the accountant said was spent.
        # last_epochs_run is len(epsilon_list), which counts the epoch that triggered
        # an early break as well, so it is one too high whenever the budget binds.
        # Kept as is because committed logs and meta.json files report it;
        # last_epochs_trained is the corrected count. last_epsilon_spent_final is
        # the spend after the last trained epoch -- the last epsilon_list entry on an
        # early break, but one epoch more than it when every epoch trained.
        self.last_epochs_run = None
        self.last_epsilon_spent = None
        self.last_epochs_trained = None
        self.last_epsilon_spent_final = None
        self.last_sigma = None

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
        n = len(raw)
        # Pools pickled before the tuning knobs existed were grown at the defaults.
        batch_size = getattr(self, "batch_size", BATCH_SIZE)
        sigma = getattr(self, "sigma", SIGMA)
        if getattr(self, "calibrate", False):
            if n not in self._calibrated_sigma:
                self._calibrated_sigma[n] = calibrate_sigma(n, batch_size, self.epochs,
                                                            self.epsilon)
            sigma = self._calibrated_sigma[n]

        synth = DPCTGAN(
            epsilon=self.epsilon,
            sigma=sigma,
            epochs=self.epochs,
            batch_size=batch_size,
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
        self.last_sigma = float(sigma)
        if spent:
            self.last_epochs_run = len(spent)
            self.last_epsilon_spent = float(spent[-1])
            if float(spent[-1]) > self.epsilon:      # broke early: last epoch untrained
                self.last_epochs_trained = len(spent) - 1
                self.last_epsilon_spent_final = float(spent[-1])
            else:                                    # every epoch trained
                self.last_epochs_trained = len(spent)
                self.last_epsilon_spent_final = accountant_epsilon(
                    n, batch_size, sigma, len(spent))

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
