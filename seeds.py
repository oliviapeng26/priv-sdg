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

THE TAPAS GENERATOR GETS A *VARYING* SEED — NOT A FIXED ONE, AND NOT NONE
    TAPAS re-fits the generator from scratch once per simulated dataset, on D+/D-
    pairs differing in exactly one record. Pinning random_state to a constant
    across those fits gives D+ and D- common random numbers, collapsing the
    generator's own sampling variance — a methodological change, not a
    reproducibility fix.

    `SynthcityGenerator` originally passed no random_state at all, on the
    assumption that this left the generator free-running. IT DID NOT. Synthcity's
    `Plugin.__init__` defaults `random_state: int = 0`, and `Plugin.fit()` calls
    `enable_reproducible_results(self.random_state)`, which reseeds numpy, torch
    and `random` globally. Every fit therefore ran at seed 0 on an identical input
    (ExactDataKnowledge hands the same background to every simulation), so every
    simulation produced a byte-identical synthetic dataset. Verified 2026-08-22
    against all six caches: the 1000/2500 DPGAN pilot's 3500 fits yielded exactly
    two distinct datasets, one per class, each duplicated 1250 times. That is what
    forced TPR=1 / FPR=0 and the identical eff-epsilon across generators — an
    effective sample size of 2, not a property of the threat model.

    The fix is an incrementing per-fit seed: fit i runs at
    TAPAS_GENERATOR_SEED_BASE + i. Consecutive fits — including the two halves of
    a D+/D- pair — never share a draw, so the sampling variance is real while the
    audit stays reproducible end to end.

    `Plugin.generate()` reseeds only when explicitly passed random_state, which
    SynthcityGenerator does not do, so sampling continues from the post-fit RNG
    state and inherits the per-fit variation.

    NOTE: this is the PRIVACY path only. sdg/generate_runs.py already passes a
    varying random_state per run across RUN_SEEDS, so utility and fidelity stand.
"""

import random
import sys

import numpy as np

# ── Fixed across all experiments (never change) ──
DATA_SPLIT_SEED = 42       # 80/20 train/test split (data_preprocessing.ipynb)
TAPAS_BG_SEED = 42         # 499-record background sample (benchmark_tapas)
TAPAS_TARGET_SEED = 43     # random target/alternate selection (benchmark_tapas)
TAPAS_GENERATOR_SEED_BASE = 1000   # per-fit generator seed in the TAPAS audit: fit i runs
                                   # at TAPAS_GENERATOR_SEED_BASE + i, so no two simulations
                                   # (and no D+/D- pair) ever share a draw

# ── Raw-score extraction (benchmark_tapas/scripts/extract_scores.py) ──
# These are NOT frozen the way the three above are: they seed a read-only
# re-analysis of already-generated data, so changing them re-shuffles which
# cached D+/D- pairs are read and re-draws the attacks' internal randomness,
# but cannot invalidate any cache, dataset or committed table.
SCORE_SUBSET_SEED = 44     # which memoised D+/D- pairs to keep when a cache holds
                           # more than the benchmark's 50/100 (only DPGAN, at 1000/2500)
SCORE_ATTACK_SEED = 45     # attack-internal randomness: RandomForest trees and the
                           # RandomTargetedQueryFeature draw. Applied identically before
                           # every generator so all four are probed by the SAME attack.

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
