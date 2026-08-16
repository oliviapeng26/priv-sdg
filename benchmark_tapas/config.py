"""Shared constants for the 2x2 generator-grid privacy benchmark (benchmark_tapas).

This experiment runs the full 5-attack TAPAS MIA battery against all four
generators in the 2x2 grid (statistical/neural x non-DP/DP) at a fixed formal
budget of eps=1.0 for the DP methods, under the same exact-knowledge +
black-box threat model and the SAME fixed background / target / alternate
records across all four methods (so any difference is the generator's, not the
draw's).

All four methods use the SAME shadow-model counts (num_train=50, num_test=100),
so their MIA-AUC / eff-epsilon confidence intervals are directly comparable.
The neural methods were originally planned at a reduced 10/20 for the Colab T4
budget, but probe_fit_time.py measured the real 500-row per-fit time at ~6.8s
(CTGAN) / ~18.3s (DPGAN) -- fast enough that 50/100 runs comfortably (CTGAN
~34 min, DPGAN ~92 min for the first attack), so we matched the statistical
methods and dropped the count asymmetry. Colab free-tier sessions run up to
~12 h, so both neural audits fit in one sitting; the binding limit is the idle
timeout (~30-60 min of inactivity), so keep the tab open + machine awake until
DPGAN's ~92-min Groundhog checkpoints (see README).
"""

import sys
from pathlib import Path

# -- Paths ---------------------------------------------------------------
BENCHMARK_DIR = Path(__file__).resolve().parent          # benchmark_tapas/
REPO_ROOT = BENCHMARK_DIR.parent                         # priv-sdg/
DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
TEST_CSV = DATA_DIR / "adult_test.csv"

CACHE_DIR = BENCHMARK_DIR / "cache"                      # gitignored
RESULTS_DIR = BENCHMARK_DIR / "results"
# results/ is organised into subfolders by artefact type:
FIGURES_DIR     = RESULTS_DIR / "figures"       # tradeoff + heatmap PNGs
TABLES_DIR      = RESULTS_DIR / "tables"        # summary_*.csv + benchmark_privacy_per_attack.csv
CONVERGENCE_DIR = RESULTS_DIR / "convergence"   # convergence_check_tstr.csv (+ _LEGACY)
PER_METHOD_DIR  = RESULTS_DIR / "per_method"    # per-generator effeps CSVs, ROC PNG, log

# Utility/fidelity tables that aggregate.py reads. Both are the multi-run,
# in-house tables: utility from evaluation/eval_utility.py (TSTR scored on the
# held-out test split) and fidelity from evaluation/eval_fidelity.py, each
# aggregated over seeds.RUN_SEEDS so every column carries a _mean and a _std.
#
# These replace results/synthcity_results.csv, which has been deleted: its
# performance.* metrics were scored on an internal split of the generators'
# OWN training data (see evaluation_LEGACY/eval_synthcity_LEGACY.py). Fidelity moved to
# multi-run table at the same time so the two halves of summary_utility.csv are
# not a 5-run mean next to a single draw.
UTILITY_RESULTS = REPO_ROOT / "results" / "utility_summary.csv"
FIDELITY_RESULTS = REPO_ROOT / "results" / "fidelity_summary.csv"

# -- Dataset schema (matches common.py / evaluation/eval_utility.py / target_strategy) --
CONTINUOUS_COLS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_COLS = [
    "workclass", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country", "income",
]
TARGET_COL = "income"

# -- Threat-model design (shared, fixed across all 4 methods) -------------
# The two seeds come from the repo-root seeds.py (single source of truth for
# every fixed seed in the pipeline); the values are unchanged from what this
# benchmark already ran with, so the cached threat models stay valid.
sys.path.insert(0, str(REPO_ROOT))
from seeds import TAPAS_BG_SEED, TAPAS_TARGET_SEED

FORMAL_EPSILON = 1.0    # DP budget for privbayes/dpgan (benchmark setting)
BACKGROUND_SIZE = 499   # |d_{-t}|, fixed background sampled once
NUM_SYNTHETIC = 500     # records per simulated synthetic dataset (499+1)
BACKGROUND_SEED = TAPAS_BG_SEED
TARGET_SEED = TAPAS_TARGET_SEED

# -- Per-method configuration --------------------------------------------
# dp:            passes epsilon=FORMAL_EPSILON to the plugin iff True (non-DP
#                plugins reject an `epsilon` kwarg entirely).
# kind:          "statistical" | "neural" (for tradeoff-plot marker shape).
# num_train/test: shadow-model counts (asymmetric, see module docstring).
# plugin_kwargs: extra Synthcity kwargs. GANs get an n_iter cap (provisional --
#                confirm via convergence_check.py) to stay inside the Colab T4
#                budget; `device` is injected at runtime by the run script.
# Set from convergence_check.py (2026-07-18, Colab T4, full-data TSTR sweep):
#   CTGAN  utility is flat across n_iter (50=.874, 100=.876, 200=.880, full=.874)
#          -> 50 already matches the converged model. Lowest converged n_iter.
#   DPGAN  utility jumps 50=.40 (undertrained) -> 100=.60, then 200=.55, full=.54
#          (noisy DP plateau ~.55). 100 = lowest n_iter that reaches convergence;
#          under fixed eps=1 more steps spread the noise thinner and don't help.
# Both GANs capped at 50 (DPGAN was 100 until 2026-08-16). Three reasons, and the
# leak-free convergence sweep supports all three:
#
#   CTGAN   n_iter is pure training length (no DP). 50/100/200 all score ~0.865
#           against a 0.874 ceiling, so 50 is where utility has converged.
#   DPGAN   n_iter is NOT just training length -- opacus calibrates the per-step
#           noise so the budget over n_iter epochs equals eps=1.0 (gan.py passes
#           `epochs=self.generator_n_iter` to make_private_with_epsilon). Each cap
#           is a different mechanism, so there is no converged ceiling to approach.
#           The three caps are indistinguishable (pairwise differences <= 0.022,
#           inside their s.d.), and the default's 5 draws span [0.354, 0.632],
#           containing every cap mean. Utility gives no basis to prefer one, so we
#           take the cheapest.
#   BOTH    matching the two makes CTGAN vs DPGAN a clean DP ablation: same
#           architecture, same training length, DP the only difference. (The
#           statistical pair is NOT such an ablation -- BN and PrivBayes use
#           different encoders and structure learning; see the README correction.)
#
# sdg/generate_runs.py caps generation at the same value, so the model we report
# utility for and the model we audit share a configuration.
CTGAN_N_ITER = 50
DPGAN_N_ITER = 50

METHOD_CONFIG = {
    "bayesian_network": {
        "dp": False, "kind": "statistical",
        "num_train": 50, "num_test": 100, "plugin_kwargs": {},
    },
    "privbayes": {
        "dp": True, "kind": "statistical",
        "num_train": 50, "num_test": 100, "plugin_kwargs": {},
    },
    "ctgan": {
        "dp": False, "kind": "neural",
        "num_train": 50, "num_test": 100, "plugin_kwargs": {"n_iter": CTGAN_N_ITER},
    },
    "dpgan": {
        "dp": True, "kind": "neural",
        "num_train": 50, "num_test": 100, "plugin_kwargs": {"n_iter": DPGAN_N_ITER},
    },
}

METHODS = list(METHOD_CONFIG.keys())
DP_METHODS = {m for m, c in METHOD_CONFIG.items() if c["dp"]}

# -- Attack battery (5 MIA attacks, built in common.build_attacks) --------
ATTACK_NAMES = [
    "GroundhogAttack",
    "ShadowModelling(RandomQueries)",
    "LocalNeighbourhoodAttack",
    "ProbabilityEstimationAttack",
    "ClosestDistanceMIA",
]


# -- Helpers -------------------------------------------------------------
def slug(name: str) -> str:
    """Filesystem-safe identifier for attack labels (cache/result filenames)."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def method_cache_dir(method: str) -> Path:
    d = CACHE_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def method_results_dir(method: str) -> Path:
    d = PER_METHOD_DIR / method
    d.mkdir(parents=True, exist_ok=True)
    return d
