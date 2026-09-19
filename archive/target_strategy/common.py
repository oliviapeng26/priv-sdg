"""Shared pipeline infrastructure for the target-strategy experiment.
Contains everything that is identical across all 3 strategy scripts:
data loading, scaling, the SynthcityGenerator wrapper, the SwapMIALabeller /
SwapTargetedMIA (exact-swap D+/D- construction), threat-model caching,
the 5-attack battery, and per-attack run+cache logic.

Adapted from evaluation/eval_tapas/eff_eps/common.py with these changes:
  - FORMAL_EPSILON = 10.0 (investigating the εeff gap at ε=10)
  - 5th attack added: ClosestDistanceMIA
  - Paths/imports pulled from target_strategy/config.py
  - Target selection functions removed (instead, in strategies/*.py)

The only thing each strategy script provides is the target selection function.
"""

import json
import logging
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
    ClosestDistanceMIA,
    LpDistance,
)
from tapas.report import EffectiveEpsilonReport
from tapas.report.utils import plot_roc_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity

from config import (
    DATA_DIR, TRAIN_CSV, TEST_CSV,
    CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL,
    FORMAL_EPSILON, BACKGROUND_SIZE, NUM_SYNTHETIC,
    GENERATOR_NAME,
    slug,
)

log = logging.getLogger("target_strategy")


# ── Data loading + scaling ─────────────────────────────────────────────

def _fit_scalers(train_df: pd.DataFrame) -> dict:
    """Min/max from training data only — never test."""
    return {c: (float(train_df[c].min()), float(train_df[c].max())) for c in CONTINUOUS_COLS}

def _apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for c, (lo, hi) in scalers.items():
        span = hi - lo
        df[c] = (df[c].astype(float) - lo) / span if span > 0 else 0.0
    return df


def build_description(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DataDescription:
    """TAPAS schema. Categorical vocab = union of train+test to avoid OOV crashes."""
    schema = []
    for c in CONTINUOUS_COLS:
        schema.append({"name": c, "type": "real", "representation": "number"})
    for c in CATEGORICAL_COLS:
        cats = sorted(set(train_df[c].astype(str).unique()) | set(test_df[c].astype(str).unique()))
        schema.append({"name": c, "type": "finite", "representation": cats})
    return DataDescription(schema, label="AdultCensus")


def load_adult_datasets():
    """Load train/test CSVs, scale continuous columns, wrap as TabularDatasets.

    Returns: (train_dataset, test_dataset, description)
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    scalers = _fit_scalers(train_df)
    description = build_description(train_df, test_df)

    def to_tabular(df):
        d = _apply_scalers(df, scalers)
        d[CATEGORICAL_COLS] = d[CATEGORICAL_COLS].astype(str)
        d = d[list(description.columns)].reset_index(drop=True)
        return TabularDataset(d, description)

    return to_tabular(train_df), to_tabular(test_df), description


# ── Background sampling ────────────────────────────────────────────────

def sample_background(train_dataset: TabularDataset, seed: int = 42):
    """Sample a fixed background of BACKGROUND_SIZE records once.

    Returns: (background_dataset, background_indices)
    """
    n = len(train_dataset.data)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=BACKGROUND_SIZE, replace=False)
    background_dataset = train_dataset.get_records(idx.tolist())
    log.info(f"Sampled fixed background of {BACKGROUND_SIZE} records (seed={seed})")
    return background_dataset, set(idx.tolist())


# ── Synthcity-TAPAS generator wrapper ────────────────────────────────

class SynthcityGenerator(Generator):
    """Wraps a Synthcity plugin as a TAPAS Generator.

    Called by BlackBoxKnowledge once per simulated dataset: .fit() retrains
    from scratch, .generate() draws fresh synthetic data.
    """
    def __init__(self, method: str, description: DataDescription,
                 epsilon: float = FORMAL_EPSILON):
        super().__init__()
        self.method = method
        self.description = description
        self.epsilon = epsilon
        self._plugin = None

    def fit(self, dataset, **kwargs):
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        df = dataset.data.copy()
        df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

        loader = GenericDataLoader(df, target_column=TARGET_COL)
        self._plugin = Plugins().get(self.method, epsilon=self.epsilon)
        self._plugin.fit(loader)
        self.trained = True

    def generate(self, num_samples, random_state=None):
        assert self.trained, "Generator must be fit before generate()."
        synth_df = self._plugin.generate(count=num_samples).dataframe()
        synth_df[CATEGORICAL_COLS] = synth_df[CATEGORICAL_COLS].astype(str)
        synth_df = synth_df[list(self.description.columns)].reset_index(drop=True)
        return TabularDataset(synth_df, self.description)

    @property
    def label(self):
        return self.method


# ── Exact-swap MIA labeller ────────────────────────────────────────────

class SwapMIALabeller(tm.AttackerKnowledgeWithLabel):
    """D+ = background + target (member), D- = background + alternate (non-member).

    Both worlds are the same size (BACKGROUND_SIZE + 1) on every draw.
    TAPAS's built-in MIALabeller can't do this exactly — see common.py docstring.
    """
    def __init__(self, attacker_knowledge, target_record, alternate_record):
        self.attacker_knowledge = attacker_knowledge
        self.target_record = target_record
        self.alternate_record = alternate_record

    def generate_datasets_with_label(self, num_samples: int, training: bool = True):
        if num_samples % 2 == 1:
            num_samples += 1
        datasets = self.attacker_knowledge.generate_datasets(num_samples // 2, training)

        mod_datasets, mod_labels = [], []
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
    """TargetedMIA subclass using SwapMIALabeller.

    Subclasses TargetedMIA (not bare LabelInferenceThreatModel) so attacks
    that isinstance-check for TargetedMIA still work (e.g. LocalNeighbourhoodAttack,
    ClosestDistanceMIA that require knowing which record is target).
    """
    def __init__(self, attacker_knowledge_data, target_record, alternate_record,
                 attacker_knowledge_generator, memorise_datasets=True,
                 iterator_tracker=None, num_concurrent=1):
        labeller = SwapMIALabeller(attacker_knowledge_data, target_record, alternate_record)
        tm.LabelInferenceThreatModel.__init__(
            self, labeller, attacker_knowledge_generator, memorise_datasets,
            iterator_tracker=iterator_tracker, num_labels=1, num_concurrent=num_concurrent,
        )
        self.target_record = target_record
        self.alternate_record = alternate_record


# ── Threat model (cached) ──────────────────────────────────────────────

def build_or_load_threat_model(cache_dir: Path, background_dataset,
                                target_record, alternate_record, description):
    """Load cached threat model if present, otherwise build fresh."""
    cache_path = cache_dir / "threat_model"
    if (cache_dir / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info("Building new threat model (no cache found)")
    generator = SynthcityGenerator(GENERATOR_NAME, description, FORMAL_EPSILON)
    threat_model = SwapTargetedMIA(
        attacker_knowledge_data=tm.ExactDataKnowledge(background_dataset),
        target_record=target_record,
        alternate_record=alternate_record,
        attacker_knowledge_generator=tm.BlackBoxKnowledge(
            generator, num_synthetic_records=NUM_SYNTHETIC
        ),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    threat_model.save(str(cache_path))
    return threat_model


# ── 5-attack battery ───────────────────────────────────────────────────

def compute_radius(target_record, background_dataset, distance, k: int = 5) -> float:
    """Radius for LocalNeighbourhoodAttack: distance to k-th nearest background record."""
    distances = np.sort(distance(target_record, background_dataset)[0])
    k = min(k, len(distances) - 1)
    return float(distances[k])


def build_attacks(target_record, background_dataset):
    """Instantiate 5 MIA attacks spanning 4 families."""
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

    closest_distance = ClosestDistanceMIA(distance=distance)

    return [groundhog, shadow_random_queries, local_neighbourhood,
            probability_estimation, closest_distance]


# ── Per-attack run + cache ─────────────────────────────────────────────

def run_attack(attack, threat_model, num_train: int, num_test: int,
               cache_dir: Path, results_dir: Path):
    """Train + test one attack. Caches result JSON per attack so crashes
    don't lose finished attacks. Self-invalidates if num_train/num_test increases.

    Returns: (result_dict, summary_or_None)
    """
    attack_slug = slug(attack.label)
    result_path = cache_dir / f"result_{attack_slug}.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if num_train <= cached.get("num_train", 0) and num_test <= cached.get("num_test", 0):
            log.info(f"  [{attack.label}] cached, skipping")
            return cached, None
        log.info(f"  [{attack.label}] cache stale, recomputing")

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
    validation_split = min(0.5, max(0.1, 15 / len(summary.scores)))
    eps_report = EffectiveEpsilonReport(
        [summary], validation_split=validation_split, confidence_levels=(0.9, 0.95, 0.99),
        suffix=attack_slug,
    )
    eps_df = eps_report.publish(str(results_dir))

    pointwise_eps = summary.effective_epsilon

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

    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    log.info(f"  [{attack.label}] done — eps_low_95={result.get('eps_low_95', 'N/A'):.3f}  "
             f"({wall_time_s}s, {peak_memory_mb} MB)")
    return result, summary


def save_roc_report(summaries_by_attack: dict, results_dir: Path, label: str):
    """Plot combined ROC curve for all attacks that ran (not cached)."""
    if not summaries_by_attack:
        return
    summaries = list(summaries_by_attack.values())
    short_names = [name.split("(")[0] for name in summaries_by_attack.keys()]
    plot_roc_curve(
        [(s.labels, s.scores) for s in summaries],
        short_names,
        f"ROC curves ({label})",
        str(results_dir),
        suffix=f"_{label}",
    )