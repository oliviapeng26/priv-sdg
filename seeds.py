"""
Reproducibility seeds for the evaluation pipeline.

Fixed seeds: ensure all generators/evaluations use the same experimental setup.
Run seeds: varied across runs for variance estimates.

WHY THIS SITS AT THE REPO ROOT AND NOT IN config/
    `benchmark_tapas/common.py` and every script under `benchmark_tapas/` and
    `target_strategy/` already import their experiment constants as the
    top-level module name `config` (they put their own directory on sys.path
    first). A repo-root `config/` package would shadow-collide with those:
    `from config.seeds import ...` would resolve `config` to
    benchmark_tapas/config.py and fail. A flat `seeds.py` at the root has no
    such collision, and matches the user's preference for a flat repo.

    Consumers add the repo root to sys.path and `from seeds import ...`.

THE FIXED VALUES ARE THE ONES THE REPO ALREADY USED
    42 / 42 / 43 are not new numbers — they are what data_preprocessing.ipynb
    and benchmark_tapas/config.py were already using, hoisted here so there is
    one source of truth. Changing them would invalidate data/adult_{train,test}.csv,
    all five synthetic_data/*.csv, every benchmark_tapas/cache/*/threat_model.pkl,
    and every committed table in results/ — for no scientific gain, since the
    numeric value of a seed is arbitrary. Treat them as frozen.

DELIBERATE EXCEPTION — the TAPAS generator is NOT given a fixed seed
    TAPAS re-fits the generator from scratch once per simulated dataset
    (2 x (num_train + num_test) fits per attack), on D+ and D- pairs that differ
    in exactly one record. Pinning random_state to a constant across those fits
    would give D+ and D- common random numbers, collapsing the generator's own
    sampling variance and inflating attack accuracy — a methodological change,
    not a reproducibility fix. So `SynthcityGenerator` stays seedless; what is
    seeded on the privacy side is the *setup* (background sample, target choice),
    which is exactly TAPAS_BG_SEED / TAPAS_TARGET_SEED below.
"""

import random
import sys

import numpy as np

# ── Fixed across all experiments (never change) ──
DATA_SPLIT_SEED = 42       # 80/20 train/test split (data_preprocessing.ipynb)
TAPAS_BG_SEED = 42         # 499-record background sample (benchmark_tapas)
TAPAS_TARGET_SEED = 43     # random target/alternate selection (benchmark_tapas)

# ── Varied across runs (for mean ± std) ──
NUM_RUNS = 5
RUN_SEEDS = [100 + i for i in range(NUM_RUNS)]


def set_all_seeds(seed: int) -> None:
    """Seed numpy, torch and the stdlib `random` module for one run.

    Synthcity plugins do this internally when given `random_state` (see
    `synthcity.utils.reproducibility.enable_reproducible_results`, called from
    `Plugin.fit` and from `Plugin.generate(random_state=...)`), so for those
    four generators passing `random_state=seed` is sufficient and this function
    is belt-and-braces. It is load-bearing for AIM, which is not a Synthcity
    plugin and takes no seed argument.

    torch is seeded only if it is ALREADY imported — deliberately, not out of
    laziness. The statistical-only paths (BN, PrivBayes, AIM, the sklearn /
    xgboost classifiers) have no torch randomness to seed, and importing torch
    into a process that later starts an xgboost thread pool is exactly the
    OpenMP collision documented in the README. If torch is loaded, the GAN path
    is live and it gets seeded; if it is not, there is nothing to seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
