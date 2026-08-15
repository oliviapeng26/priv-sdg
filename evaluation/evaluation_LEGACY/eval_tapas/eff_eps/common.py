"""SUPERSEDED -- iteration 2 of the TAPAS privacy evaluation. Use benchmark_tapas/.

    Effective epsilon across the MIA battery, but for PrivBayes and DPGAN ONLY.
    benchmark_tapas/ generalised this to all four generators in the 2x2 grid and
    is the version that goes into the write-up. Outputs live in
    results/tapas_LEGACY/tapas_results/.

Shared, cacheable infrastructure for TAPAS effective-epsilon (RQ1, sub-q 1) experiments.

Reused by both privbayes/run_privbayes_effeps.py and dpgan/run_dpgan_effeps.py:
  - Adult Census loading + a TAPAS DataDescription (continuous cols min-max scaled
    to [0,1], categorical cols one-hot via a fixed category list).
  - A Generator wrapper wiring Synthcity plugins into TAPAS's black-box interface.
  - Background + target-record selection, aligned with the TAPAS paper's own
    Experiment 2 (Stadler et al. 2022, sec. 4): a fixed 499-record background is
    sampled once (sample_background) and reused everywhere; a target t and an
    alternate t' are then selected disjointly from it (currently
    select_random_target_and_alternate; select_outlier_target is a placeholder
    for a later run using the paper's Experiment 1 outlier heuristic instead).
  - Threat-model construction with disk caching (ThreatModel.save/.load memoises
    every simulated (real, synthetic) dataset pair, so re-running the script does
    not repeat generator fits already paid for). Uses a custom SwapMIALabeller /
    SwapTargetedMIA (not TAPAS's stock TargetedMIA) so the member world
    (background+t) and non-member world (background+t') are the same size on
    every draw -- see those classes' docstrings for why TAPAS's own
    replace_target mechanism can't do this exactly.
  - A per-attack runner that caches its result (JSON) as soon as it finishes, so a
    crash partway through the 4-attack battery does not lose completed attacks.
    Each cached result also records wall_time_s/peak_memory_mb for that attack's
    train+test step, matching the overhead-tracking convention used elsewhere in
    the repo (eval_synthcity.py, eval_tapas.py, sdg/*.ipynb), just scoped
    per-attack here since attacks are cached/resumed independently.

Threat model: exact-knowledge data prior (a fixed background sampled once from
the real training set) + black-box generator knowledge + membership inference
only.
"""

import json
import logging
import re
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from tapas.datasets import TabularDataset
from tapas.datasets.data_description import DataDescription
from tapas.generators import Generator
import tapas.threat_models as tm
from tapas.attacks import (
    GroundhogAttack,
    ShadowModellingAttack,
    FeatureBasedSetClassifier,
    RandomTargetedQueryFeature,
    LocalNeighbourhoodAttack,
    ProbabilityEstimationAttack,
    LpDistance,
)
from tapas.report import EffectiveEpsilonReport, ROCReport
from tapas.report.utils import plot_roc_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity

log = logging.getLogger("eff_eps")

# ---------- paths ----------
# parents[4], not [3]: this subtree moved into evaluation/evaluation_LEGACY/ on
# 2026-08-15, adding one directory level between here and the repo root.
#   eff_eps -> eval_tapas -> evaluation_LEGACY -> evaluation -> priv-sdg
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data"
EFF_EPS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results" / "tapas_LEGACY" / "tapas_results" / "eff_eps_results"

# ---------- schema (matches README / eval_synthcity.py / eval_tapas.py) ----------
CONTINUOUS_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship",
                    "race", "sex", "native_country", "income"]
TARGET_COL = "income"
FORMAL_EPSILON = 1.0


def slug(text: str) -> str:
    """Filesystem-safe identifier for attack labels (used in cache/result filenames)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


# ---------- data loading + TAPAS description ----------
"""
TAPAS's TabularDataset.as_numeric does NOT normalise 'real' columns (despite
what its docstring says) -- it passes them through raw. Distance-based attacks
(LocalNeighbourhoodAttack) and density estimation (ProbabilityEstimationAttack)
need comparable scales across columns, so we normalise ourselves and treat the
result as already being in [0, 1] (schema type 'real', bounds (0,1) is also
HistSetFeature's default, so this keeps Groundhog's F_Hist feature meaningful).

So from the training data only, _fit_scalers computes min/max per continuous column and _apply_scalers maps everything to [0,1].
"""
def _fit_scalers(train_df: pd.DataFrame) -> dict:
    return {c: (float(train_df[c].min()), float(train_df[c].max())) for c in CONTINUOUS_COLS}

def _apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for c, (lo, hi) in scalers.items():
        span = hi - lo
        df[c] = (df[c].astype(float) - lo) / span if span > 0 else 0.0
    return df


def build_description(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DataDescription:
    """Build a fixed TAPAS schema. Categorical vocab (unique categories for each categorical attribute) 
    is the union of train+test, so the one-hot encoding used internally never sees an out-of-vocabulary 
    value for the real data. Synthetic data is generated by refitting on (subsets of) the
    real training data, so it draws from the same vocabulary."""
    schema = []
    for c in CONTINUOUS_COLS:
        schema.append({"name": c, "type": "real", "representation": "number"})
    for c in CATEGORICAL_COLS:
        cats = sorted(set(train_df[c].astype(str).unique()) | set(test_df[c].astype(str).unique()))
        schema.append({"name": c, "type": "finite", "representation": cats})
    return DataDescription(schema, label="AdultCensus")


def load_adult_datasets():
    """Load train/test CSVs, scale continuous columns (fit on train), and wrap both
    as TAPAS TabularDatasets sharing one DataDescription.

    Returns
    -------
    train_dataset, test_dataset : TabularDataset (TAPAS's data container)
    description : DataDescription (fixed schema from train+test)
    """
    train_df = pd.read_csv(DATA_DIR / "adult_train.csv")
    test_df = pd.read_csv(DATA_DIR / "adult_test.csv")

    scalers = _fit_scalers(train_df)
    description = build_description(train_df, test_df)

    def to_tabular(df):
        d = _apply_scalers(df, scalers)
        d[CATEGORICAL_COLS] = d[CATEGORICAL_COLS].astype(str)
        d = d[list(description.columns)].reset_index(drop=True)
        return TabularDataset(d, description)

    return to_tabular(train_df), to_tabular(test_df), description


# ---------- background + target-record selection ----------
#
# Aligned with the TAPAS paper's Experiment 2 (Stadler et al. 2022, sec. 4):
# a fixed 499-record background "d" is reused across every simulated dataset;
# TAPAS forms the two 500-record worlds D+ = d+t (member) and D- = d+t'
# (non-member) by adding a specific record on top. Background sampling is
# deliberately decoupled from target selection (below) so that swapping the
# target-selection strategy later (e.g. to the outlier heuristic, see
# select_outlier_target) never requires resampling or invalidating the
# background.
BACKGROUND_SIZE = 499


def sample_background(train_dataset: TabularDataset, background_size: int = BACKGROUND_SIZE,
                       seed: int = 42):
    """Sample a fixed background of `background_size` records once, to be
    reused as-is across all attacks and all runs (do not resample per-attack
    or per-run -- the whole point is that this exact background is shared).

    Returns (background_dataset, background_indices) -- indices (into
    train_dataset.data, 0-based after reset_index) are returned so target
    selection can exclude them.
    """
    n = len(train_dataset.data)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=background_size, replace=False)
    background_dataset = train_dataset.get_records(idx.tolist())
    log.info(f"Sampled fixed background of {background_size} records (seed={seed})")
    return background_dataset, set(idx.tolist())


def select_random_target_and_alternate(train_dataset: TabularDataset, exclude_indices: set,
                                        seed: int = 43):
    """Randomly select the target record t and an alternate record t', both
    excluded from `exclude_indices` (the background) and from each other.

    Returns (target_record, alternate_record).
    """
    n = len(train_dataset.data)
    rng = np.random.RandomState(seed)
    available = np.array([i for i in range(n) if i not in exclude_indices])
    t_idx, tprime_idx = rng.choice(available, size=2, replace=False)
    target_record = train_dataset.get_records([int(t_idx)])
    alternate_record = train_dataset.get_records([int(tprime_idx)])
    log.info(f"Selected random target t at index {t_idx} and alternate t' at index {tprime_idx}")
    return target_record, alternate_record


def select_outlier_target(train_dataset: TabularDataset, exclude_indices: set,
                           subsample_size: int = 1000, num_bins: int = 20, seed: int = 44):
    """NOT YET WIRED IN -- for a later run only (see task notes). Will pick the
    "outlier" record (lowest log-likelihood under independent empirical
    marginals) from a `subsample_size`-record random subsample of the training
    data excluding the fixed background, matching the TAPAS paper's Experiment
    1 (AIA) target-selection method. Reuses the same `sample_background`
    output -- the background does not need to be regenerated to swap the
    target-selection strategy.

    TODO before use: also select a corresponding alternate record t' (excluded
    from the subsample/background/outlier) the same way
    select_random_target_and_alternate does.
    """
    raise NotImplementedError(
        "select_outlier_target is a placeholder for a later run -- see task notes."
    )


# ---------- Synthcity -> TAPAS generator wrapper ----------
class SynthcityGenerator(Generator):
    """Wraps a Synthcity plugin (e.g. 'privbayes', 'dpgan') as a tapas.generators.Generator.

    Called by TAPAS's BlackBoxKnowledge once per simulated dataset: .fit() retrains
    the plugin from scratch on the (real, subsetted) training data each time, then
    .generate() draws a fresh synthetic dataset. This retraining is the expensive
    step TAPAS's threat-model caching amortises across attacks (see run scripts).

    Defined at module level (not as a closure) so that instances -- and the
    ThreatModel that holds one -- can be pickled for on-disk caching.

    plugin_kwargs lets a shadow-model fit use cheaper settings than the "real" run
    documented in the README (e.g. fewer training iterations for DPGAN) to make
    the hundreds of simulations tractable -- see NOTE in dpgan/run_dpgan_effeps.py.
    """

    def __init__(self, method: str, description: DataDescription, 
                 epsilon: float = FORMAL_EPSILON, plugin_kwargs: dict = None):
        super().__init__()
        self.method = method
        self.description = description
        self.epsilon = epsilon
        self.plugin_kwargs = plugin_kwargs or {}
        self._plugin = None

    def fit(self, dataset, **kwargs):
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        df = dataset.data.copy()
        df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)  # still in [0,1]
        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

        loader = GenericDataLoader(df, target_column=TARGET_COL)
        self._plugin = Plugins().get(self.method, epsilon=self.epsilon, **self.plugin_kwargs)
        self._plugin.fit(loader)
        self.trained = True

    def generate(self, num_samples, random_state=None):
        assert self.trained, "Generator must be fit before generate()."
        synth_df = self._plugin.generate(count=num_samples).dataframe()
        # NOTE: no _apply_scalers() call here. fit() feeds the plugin data
        # that's already scaled to [0,1] (see comment above), so its output
        # is already on that same scale -- reapplying the real-data min/max
        # scaler here would double-transform it (e.g. age=0.178 -> (0.178-17)/73
        # ~= -0.23), corrupting every synthetic dataset. Only dtype/column
        # bookkeeping is needed below.
        synth_df[CATEGORICAL_COLS] = synth_df[CATEGORICAL_COLS].astype(str)
        synth_df = synth_df[list(self.description.columns)].reset_index(drop=True)
        return TabularDataset(synth_df, self.description)

    @property
    def label(self):
        return self.method

def make_generator_wrapper(method: str, description: DataDescription, 
                            epsilon: float = FORMAL_EPSILON, plugin_kwargs: dict = None):
    return SynthcityGenerator(method, description, epsilon, plugin_kwargs)


# ---------- exact-swap MIA labeller ----------
class SwapMIALabeller(tm.AttackerKnowledgeWithLabel):
    """Labels datasets by adding `target_record` (label=True, "member") or
    `alternate_record` (label=False, "non-member") on top of copies of the
    *same* fixed background -- both worlds end up the same size
    (len(background)+1), matching the TAPAS paper's Experiment 2 setup
    (D+ = d+t, D- = d+t') exactly on every draw.

    TAPAS's own MIALabeller has no equivalent of this: with replace_target=
    False it leaves the non-member world as the untouched background (size
    len(background), not len(background)+1 -- mismatched sizes vs. the member
    world); with replace_target=True it evicts a *uniformly random* background
    record to keep size constant for the member world only, and never inserts
    anything for the non-member world, so there is no way to plug in a fixed,
    specific t' via the built-in mechanism. This class exists to give exact,
    deterministic D+/D- pairs instead of that approximation.
    """

    def __init__(self, attacker_knowledge, target_record, alternate_record):
        self.attacker_knowledge = attacker_knowledge
        self.target_record = target_record
        self.alternate_record = alternate_record

    def generate_datasets_with_label(self, num_samples: int, training: bool = True):
        if num_samples % 2 == 1:
            num_samples += 1
        datasets = self.attacker_knowledge.generate_datasets(num_samples // 2, training)

        mod_datasets = []
        mod_labels = []
        for dataset in datasets:
            member = dataset.copy()
            member.add_records(self.target_record, in_place=True)
            mod_datasets.append(member)
            mod_labels.append(True)

            non_member = dataset.copy()
            non_member.add_records(self.alternate_record, in_place=True)
            mod_datasets.append(non_member)
            mod_labels.append(False)

        return mod_datasets, mod_labels

    @property
    def label(self):
        return self.attacker_knowledge.label


class SwapTargetedMIA(tm.TargetedMIA):
    """A TargetedMIA that adds `alternate_record` for the non-member world
    instead of leaving it as bare background (see SwapMIALabeller). Subclasses
    TargetedMIA -- rather than building a bare LabelInferenceThreatModel --
    purely so that attacks which hard-check `isinstance(threat_model,
    TargetedMIA)` (e.g. LocalNeighbourhoodAttack, tapas/attacks/
    closest_distance.py:227) still recognise this as a valid MIA threat model.
    Bypasses TargetedMIA.__init__ (which always builds a plain MIALabeller)
    and calls the grandparent LabelInferenceThreatModel.__init__ directly with
    a SwapMIALabeller instead.
    """

    def __init__(self, attacker_knowledge_data, target_record, alternate_record,
                 attacker_knowledge_generator, memorise_datasets=True,
                 iterator_tracker=None, num_concurrent=1):
        labeller = SwapMIALabeller(attacker_knowledge_data, target_record, alternate_record)
        tm.LabelInferenceThreatModel.__init__(
            self,
            labeller,
            attacker_knowledge_generator,
            memorise_datasets,
            iterator_tracker=iterator_tracker,
            num_labels=1,
            num_concurrent=num_concurrent,
        )
        self.target_record = target_record
        self.alternate_record = alternate_record


# ---------- threat model (cached) ----------
def build_or_load_threat_model(cache_dir: Path, method: str, background_dataset,
                                target_record, alternate_record, description,
                                num_synthetic_records: int, epsilon: float = FORMAL_EPSILON,
                                plugin_kwargs: dict = None):
    """Load a cached threat model if present (with any simulations already
    memoised inside it), otherwise build a fresh one. Always call
    threat_model.save(...) again after running attacks to persist newly-generated
    simulations -- the run scripts handle that.
    """
    cache_path = cache_dir / "threat_model"
    if (cache_dir / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info(f"Building new threat model for {method} (no cache found)")
    generator = make_generator_wrapper(method, description, epsilon, plugin_kwargs)
    threat_model = SwapTargetedMIA(
        attacker_knowledge_data=tm.ExactDataKnowledge(background_dataset),
        target_record=target_record,
        alternate_record=alternate_record,
        attacker_knowledge_generator=tm.BlackBoxKnowledge(
            generator, num_synthetic_records=num_synthetic_records
        ),
    )
    if target_record in background_dataset or alternate_record in background_dataset:
        log.warning(
            "Target record was found in the background dataset -- this makes "
            "membership inference less meaningful (see TAPAS's own "
            "TargetedMIA._assert_non_membership)."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    threat_model.save(str(cache_path))
    return threat_model


# ---------- the 4-attack battery ----------

def compute_radius(target_record, background_dataset, distance, k: int = 5) -> float:
    """Radius for LocalNeighbourhoodAttack: distance to the k-th nearest real
    record to the target (excludes the target itself, which isn't in
    background_dataset -- background and target are sampled disjointly, see
    sample_background / select_random_target_and_alternate)."""
    distances = np.sort(distance(target_record, background_dataset)[0])
    k = min(k, len(distances) - 1)
    return float(distances[k])


def build_attacks(target_record, background_dataset):
    """Instantiate the 4 attacks spanning 3 families."""
    groundhog = GroundhogAttack()

    query_feature = (
        RandomTargetedQueryFeature(target=target_record, order=1, number=500)
        + RandomTargetedQueryFeature(target=target_record, order=2, number=500)
        + RandomTargetedQueryFeature(target=target_record, order=3, number=500)
    )
    shadow_random_queries = ShadowModellingAttack(
        FeatureBasedSetClassifier(query_feature, RandomForestClassifier(n_estimators=100)),
        label="ShadowModelling(RandomQueries)",
    )

    distance = LpDistance(p=2)
    radius = compute_radius(target_record, background_dataset, distance)
    local_neighbourhood = LocalNeighbourhoodAttack(
        distance=distance, radius=radius, criterion="accuracy"
    )

    probability_estimation = ProbabilityEstimationAttack(
        estimator=KernelDensity(), criterion="accuracy"
    )

    return [groundhog, shadow_random_queries, local_neighbourhood, probability_estimation]


# ---------- per-attack run + cache ----------

def run_attack(attack, threat_model, num_train: int, num_test: int,
               cache_dir: Path, results_dir: Path):
    """Train + test one attack against threat_model, with a JSON cache keyed by
    attack label so a crash partway through the battery doesn't lose finished
    attacks. Also writes the EffectiveEpsilonReport CSV for this attack to
    results_dir (Clopper-Pearson CI at 90/95/99% confidence).

    The cache is self-invalidating on sample count: each cached JSON records
    the num_train/num_test it was computed with. If a later call requests
    *more* samples than what's cached (e.g. you started with a quick tier and
    raised NUM_TRAIN_SIMULATIONS/NUM_TEST_SIMULATIONS to scale up), the cache
    is treated as stale and this attack is retrained/retested -- no manual
    deletion needed. threat_model.pkl's own memoisation means this only
    generates the *delta* of new simulations, not a full redo. Requesting the
    same or fewer samples than what's cached reuses the cached result as-is
    (it's at least as reliable as what's being asked for). Cache files from
    before this field existed default to num_train=num_test=0, so they're
    always treated as stale and recomputed once.

    Returns (result_dict, summary). summary is None if the result was loaded
    from cache (the raw MIAttackSummary isn't persisted in the JSON cache, only
    its derived metrics) -- ROC plotting is skipped for those in the caller.
    """
    attack_slug = slug(attack.label)
    result_path = cache_dir / f"result_{attack_slug}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        cached_num_train = cached.get("num_train", 0)
        cached_num_test = cached.get("num_test", 0)
        if num_train <= cached_num_train and num_test <= cached_num_test:
            log.info(f"  [{attack.label}] cached result found "
                     f"(num_train={cached_num_train}, num_test={cached_num_test}), skipping")
            return cached, None
        log.info(
            f"  [{attack.label}] cached result is stale (cached at "
            f"num_train={cached_num_train}/num_test={cached_num_test}, "
            f"requested num_train={num_train}/num_test={num_test}) -- recomputing"
        )

    # Wall-time + peak-memory for this attack's train+test (dominated by the
    # generator fit/generate calls), matching the overhead-tracking convention
    # of eval_synthcity.py/eval_tapas.py (time.time() + tracemalloc), just
    # scoped per-attack here since attacks are independently cached/resumable
    # rather than the script running start-to-finish in one shot.
    tracemalloc.start()
    t0 = time.time()

    log.info(f"  [{attack.label}] training on {num_train} simulated datasets...")
    attack.train(threat_model, num_samples=num_train)

    log.info(f"  [{attack.label}] testing on {num_test} simulated datasets...")
    summary = threat_model.test(attack, num_samples=num_test)

    wall_time_s = round(time.time() - t0, 2)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = round(peak_bytes / 1e6, 2)

    results_dir.mkdir(parents=True, exist_ok=True)
    # EffectiveEpsilonReport carves off the first `validation_split` fraction of
    # scores to select a threshold, and needs both labels present in that slice
    # (else scipy's binomtest raises on n=0). At small num_test (e.g. DPGAN's
    # cheaper runs), the default 0.1 can leave too few validation points to
    # reliably include both labels -- scale it up to guarantee >=15 expected
    # validation samples, capped at 0.5 so the CP-bound portion isn't starved.
    validation_split = min(0.5, max(0.1, 15 / len(summary.scores)))
    eps_report = EffectiveEpsilonReport(
        [summary], validation_split=validation_split, confidence_levels=(0.9, 0.95, 0.99),
        suffix=attack_slug,
    )
    eps_df = eps_report.publish(str(results_dir))

    pointwise_eps = summary.effective_epsilon
    if np.isinf(pointwise_eps):
        log.warning(
            f"  [{attack.label}] effective_epsilon is inf (FPR hit 0 or TPR hit 1 "
            "at some threshold) -- see README 'eps_eff vs theoretical eps' note."
        )

    result = {
        "attack": attack.label,
        "num_train": num_train,
        "num_test": num_test,
        "wall_time_s": wall_time_s,
        "peak_memory_mb": peak_memory_mb,
        "auc": float(summary.auc),
        "mia_advantage": float(summary.mia_advantage),
        "privacy_gain": float(summary.privacy_gain),
        "tp": float(summary.tp),
        "fp": float(summary.fp),
        "eff_epsilon_pointwise": None if np.isinf(pointwise_eps) else float(pointwise_eps),
        "eff_epsilon_pointwise_is_inf": bool(np.isinf(pointwise_eps)),
    }
    for _, row in eps_df.iterrows():
        c = int(row.confidence * 100)
        result[f"eps_low_{c}"] = float(row.epsilon_low)
        result[f"eps_high_{c}"] = float(row.epsilon_high)

    result_path.write_text(json.dumps(result, indent=2))
    log.info(f"  [{attack.label}] done -- eps_low_90={result.get('eps_low_90'):.3f}  "
             f"eps_high_90={result.get('eps_high_90'):.3f}  "
             f"({wall_time_s}s, {peak_memory_mb} MB peak)")
    return result, summary


def save_roc_report(summaries_by_attack: dict, results_dir: Path, method: str):
    """summaries_by_attack: dict {attack_label: MIAttackSummary}. Needs the raw
    summary objects (not the JSON cache), so this only plots for attacks run in
    the current process -- rerun with cache cleared if you need the full set.

    Calls TAPAS's own plot_roc_curve directly instead of going through
    ROCReport: ROCReport's default curve_label="attack" reads MIAttackSummary's
    full attack.label (e.g. "LocalNeighbourhood(L_2, 2.058..., accuracy)"),
    and TAPAS's own legend (fontsize=20, drawn inside the axes, saved without
    bbox_inches='tight') clips long labels on save. Passing shortened names
    directly avoids that without touching attack.label itself (which is also
    used for cache filenames elsewhere -- shortening it there would silently
    invalidate the per-attack JSON cache).
    """
    if not summaries_by_attack:
        return
    summaries = list(summaries_by_attack.values())
    short_names = [label.split("(")[0] for label in summaries_by_attack.keys()]
    plot_roc_curve(
        [(s.labels, s.scores) for s in summaries],
        short_names,
        f"Comparison of ROC curves\n({method})",
        str(results_dir),
        suffix=f"_{method}",
    )
