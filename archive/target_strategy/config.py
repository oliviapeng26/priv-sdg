"""
Shared constants for the target-strategy experiment.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent  # priv-sdg/
DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
TEST_CSV = DATA_DIR / "adult_test.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Dataset schema (matches common.py / eval_synthcity.py) ─────────────
CONTINUOUS_COLS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_COLS = [
    "workclass", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country", "income",
]
TARGET_COL = "income"
ALL_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS

# ── Generator settings ─────────────────────────────────────────────────
GENERATOR_NAME = "privbayes"
FORMAL_EPSILON = 10.0   # DP budget for PrivBayes (matches original TAPAS and Chida papers)
FORMAL_DELTA = 1e-5     # δ=1e-5 to match TAPAS original paper, note Chida et al. uses δ=0.

# ── Experiment design ──────────────────────────────────────────────────
BACKGROUND_SIZE = 499   # |d_{-t}|, the fixed background dataset
NUM_SYNTHETIC = 500     # records per synthetic dataset (499+1)
NUM_TRAIN = 50          # train pairs for attack training
NUM_TEST = 100          # test pairs for eff_eps estimation
RANDOM_SEED = 42

# ── Target strategies to run ───────────────────────────────────────────
STRATEGIES = ["random", "outlier", "artificial"]

# ── Attack families ────────────────────────────────────────────────────
ATTACK_NAMES = [
    "GroundhogAttack",
    "ShadowModellingAttack",       # with RandomTargetedQueryFeature
    "LocalNeighbourhoodAttack",
    "ProbabilityEstimationAttack",
    "ClosestDistanceMIA",
]

# ── Helpers ────────────────────────────────────────────────────────────
def slug(name: str) -> str:
    """Turn an attack label into a filesystem-safe string."""
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

def strategy_results_dir(strategy: str) -> Path:
    """Return and create the results directory for a given strategy."""
    d = RESULTS_DIR / f"{strategy}_strategy"
    d.mkdir(parents=True, exist_ok=True)
    return d