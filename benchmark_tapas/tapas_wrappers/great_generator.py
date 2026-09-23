#!/usr/bin/env python3
"""GReaT (be-great, fine-tuned GPT-2) wrapped as a TAPAS Generator.

Structurally this mirrors tapas_wrappers/{aim,dpctgan}_generator.py: like those
two, GReaT is not a Synthcity plugin, so common.SynthcityGenerator cannot build
it and it needs its own Generator. Everything else in the audit -- background,
target/alternate, the 5-attack battery, caching, the distinctness guard -- is
reused unchanged from benchmark_tapas/common.py.

GReaT IS TAKEN EXACTLY AS sdg/generate_great.py RUNS IT
    llm, batch_size, epochs, fp16, dataloader_num_workers and the sample-time k
    are imported from that module rather than restated, so the model audited for
    privacy is the configuration whose utility and fidelity are reported. The one
    addition is `seed` -- see below, it is a correctness fix, not a tuning knob.

THE SEEDING FIX, WHICH IS LOAD-BEARING
    SynthcityGenerator passes a VARYING random_state per fit because Plugin.fit()
    reseeds numpy/torch/random globally from a default, so identical input gave
    byte-identical output -- the pre-2026-08-23 bug that collapsed 3500
    simulations into 2 distinct datasets (see seeds.py).

    be_great has the same hazard by a different route. GReaT.fit() builds
    `TrainingArguments(..., **self.train_hyperparameters)` and passes no seed
    unless the caller supplies one; transformers defaults `seed=42`, and HF's
    Trainer.__init__ calls set_seed(args.seed), reseeding random/numpy/torch
    globally to the same value on every fit. ExactDataKnowledge hands every
    simulation the SAME background, so without an explicit varying seed each D+
    fit would be a deterministic function of identical input -- the Synthcity
    collapse, reproduced.

    So fit i runs at TAPAS_GENERATOR_SEED_BASE + i, passed into GReaT() as a
    train kwarg and forwarded to TrainingArguments. Consecutive fits -- including
    the two halves of one D+/D- pair -- never share a draw. The counter survives
    pickling (see __getstate__), so a resumed run continues the sequence rather
    than replaying seeds it has already used. run_great_audit.py's distinctness
    guard is the regression test for this.

THE SCALING ROUND TRIP, AND WHY IT IS NOT OPTIONAL HERE
    common.load_adult_datasets min-max scales the 5 continuous columns to [0,1],
    because that is the representation every other generator and every attack
    sees. AIM needs to undo that because its bin edges are in original units.
    GReaT needs to undo it for a different reason: it serialises each row to
    TEXT ("age is 39, workclass is Private, ..."), and be_great's float_precision
    defaults to None -- full precision. A scaled age is `0.3013698630136986`,
    roughly 30 characters against 2, across 5 numeric columns. sample() generates
    at most max_length=100 tokens per row, so the scaled representation would
    truncate most rows mid-value and the generator would return almost nothing
    usable.

    So each fit runs: unscale -> GReaT -> sample -> rescale. The scalers are the
    same (min, max) pairs load_adult_datasets used, passed in by the run script,
    so the round trip is exact and the output lands back in the representation
    the threat model expects. In original units the serialised text is also
    identical in form to the 500-row sanity check in sdg/great_sanity_check.py,
    which is where the ~25.7 s/fit cost estimate comes from.

    float_precision=0 is set for the same token-budget reason: all five
    continuous columns are integer-valued in Adult, and unscaling is exact, so
    this costs no information and writes `age is 39` rather than `age is 39.0`.

    Output is NOT clipped to [0,1]. aim_generator.py makes the same choice for
    the same reason: no other generator's output is clipped, and bounding this
    one would make GReaT the only method whose tails are truncated.

REJECTION SAMPLING ON TOP OF be_great's OWN
    be_great's _legacy_sample already drops rows it could not parse (unfilled
    placeholders, all-NaN, non-numeric values in numeric columns). It does NOT
    check categorical values against a vocabulary, so well-formed nonsense gets
    through -- the sanity check produced `Married-civ-spouse-spouse`, `Hus`,
    `Husband-king`, and a `sex` of `White`. TAPAS's DataDescription declares each
    categorical column's representation as a fixed list, and the attacks one-hot
    encode against it, so those rows cannot be handed on.

    generate() therefore filters to rows whose categorical values are all in the
    description's vocabulary and re-samples the shortfall, up to MAX_SAMPLE_ROUNDS
    rounds. The acceptance rate is recorded on the instance (last_acceptance_rate)
    so --probe can report it: a low rate is the signal to raise max_length, which
    is the direct cause of the truncated values above.
"""

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from tapas.datasets import TabularDataset
from tapas.generators import Generator

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import CONTINUOUS_COLS, CATEGORICAL_COLS, CACHE_DIR    # noqa: E402
from seeds import TAPAS_GENERATOR_SEED_BASE                        # noqa: E402
# Import, never copy: these are what keep the audited model identical to the one
# sdg/generate_great.py produced the utility and fidelity numbers with.
sys.path.insert(0, str(REPO_ROOT / "sdg"))
from generate_great import (LLM, BATCH_SIZE, EPOCHS, FP16,         # noqa: E402
                            DATALOADER_NUM_WORKERS, SAMPLE_K)

# Give up after this many sample() rounds without filling the quota. be_great's
# own loop has a similar guard (_cnt > 13 with nothing generated); this one
# catches the different failure of a model that generates steadily but almost
# never inside the vocabulary.
MAX_SAMPLE_ROUNDS = 20

# be_great's sample() default is 100 tokens/row. Probed at k=2000/max_length=100:
# mean acceptance 82%, because rows with several long category values (e.g.
# "Married-civ-spouse", "Machine-op-inspct") get cut off mid-value before all
# columns are written, and the truncated tail fails the vocabulary check in
# _valid_rows below. Raised here to test whether more headroom per row raises
# acceptance and so lowers the number of sample() rounds -- and therefore the
# per-fit wall clock -- the full audit costs.
SAMPLE_MAX_LENGTH = 150


class GReaTGenerator(Generator):
    """GReaT (fine-tuned GPT-2) as a TAPAS Generator.

    TAPAS calls this once per simulated dataset through Generator.__call__, which
    is `self.fit(dataset); return self.generate(n)` -- a full refit per
    simulation, exactly as for every other generator in the benchmark.
    """

    def __init__(self, description, scalers: dict,
                 seed_base: int = TAPAS_GENERATOR_SEED_BASE):
        super().__init__()
        self.description = description
        self.scalers = scalers          # {column: (min, max)} from common._fit_scalers
        self.seed_base = seed_base
        self._model = None
        self._fit_counter = 0
        # Diagnostics from the most recent generate(), read by --probe.
        self.last_acceptance_rate = None
        self.last_sample_rounds = None
        # Per-process so two audits on two GPUs cannot share a scratch directory.
        self._scratch_dir = CACHE_DIR / f".great_scratch_{os.getpid()}"

        # Valid values per categorical column, straight off the TAPAS schema --
        # the same list the attacks one-hot encode against.
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
        import torch
        from be_great import GReaT

        # Release the previous fit's weights before loading the next set. Python
        # would drop them on reassignment anyway, but the CUDA caching allocator
        # holds the freed blocks, and this generator is refit thousands of times
        # in one process. Cheap insurance against the slow VRAM creep that would
        # otherwise only show up hours into an unattended audit.
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        raw = self._unscale(dataset.data)

        model = GReaT(
            llm=LLM,
            experiment_dir=str(self._scratch_dir),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            fp16=FP16,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            float_precision=0,                        # see module docstring
            seed=self.seed_base + self._fit_counter,  # THE seeding fix
            save_strategy="no",                       # no checkpoints, ever
            logging_strategy="no",
            report_to=[],
        )
        try:
            model.fit(raw)
        finally:
            # save_strategy="no" leaves this essentially empty, but it is created
            # per fit and there are thousands of fits.
            shutil.rmtree(self._scratch_dir, ignore_errors=True)

        self._model = model
        self._fit_counter += 1
        self.trained = True

    def _valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows whose categorical values are all in the TAPAS vocabulary.

        Continuous columns are only checked for being parseable and non-null --
        NOT for lying inside [0,1]. See the no-clipping note in the module
        docstring.
        """
        keep = pd.Series(True, index=df.index)
        for col in CATEGORICAL_COLS:
            keep &= df[col].astype(str).isin(self._vocab[col])
        for col in CONTINUOUS_COLS:
            keep &= pd.to_numeric(df[col], errors="coerce").notna()
        return df[keep]

    def generate(self, num_samples, random_state=None):
        assert self.trained, "Generator must be fit before generate()."

        collected, drawn, have, rounds = [], 0, 0, 0
        while have < num_samples:
            if rounds >= MAX_SAMPLE_ROUNDS:
                rate = f"{have / drawn:.1%}" if drawn else "no rows drawn at all"
                raise RuntimeError(
                    f"GReaT produced only {have}/{num_samples} rows inside the "
                    f"TAPAS vocabulary after {rounds} sample() rounds ({drawn} "
                    f"rows drawn, acceptance {rate}). The model is generating "
                    f"well-formed nonsense rather than failing outright -- most "
                    f"likely rows are truncated mid-value, so raising sample()'s "
                    f"max_length is the first thing to try."
                )
            # Always ask for the FULL quota, not the shortfall: be_great generates
            # a whole k-row batch per call regardless, so a request for 3 rows
            # costs the same as a request for num_samples and harvests far fewer.
            batch = self._model.sample(n_samples=num_samples, k=SAMPLE_K,
                                       max_length=SAMPLE_MAX_LENGTH)
            drawn += len(batch)
            valid = self._valid_rows(batch)
            collected.append(valid)
            have += len(valid)
            rounds += 1

        self.last_sample_rounds = rounds
        self.last_acceptance_rate = have / drawn if drawn else 0.0

        out = pd.concat(collected, ignore_index=True).head(num_samples)
        out = self._rescale(out)
        out[CATEGORICAL_COLS] = out[CATEGORICAL_COLS].astype(str)
        out = out[list(self.description.columns)].reset_index(drop=True)
        return TabularDataset(out, self.description)

    @property
    def label(self):
        return "great"

    def __getstate__(self):
        # Same reason SynthcityGenerator, AIMGenerator and DPCTGANGenerator drop
        # theirs: a fitted torch model + tokenizer does not pickle cleanly (and is
        # ~500 MB), which would break ThreatModel.save(). TAPAS refits per
        # simulation, so a reloaded generator just refits on next use and the
        # memoised synthetic datasets -- the expensive part -- are unaffected.
        # _fit_counter and seed_base are deliberately NOT reset: a resumed run
        # must continue the seed sequence, not replay seeds already used.
        state = self.__dict__.copy()
        state["_model"] = None
        state["trained"] = False
        return state
