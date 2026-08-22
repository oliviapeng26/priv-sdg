
"""Shared pipeline for the 2x2 generator-grid privacy benchmark (benchmark_tapas).

Adapted from target_strategy/common.py (iteration 3, the 5-attack battery) with
only these changes:
  1. SynthcityGenerator passes `epsilon` ONLY for DP methods (non-DP plugins
     -- bayesian_network, ctgan -- reject an `epsilon` kwarg), and takes an
     optional `plugin_kwargs` passthrough (for the GANs' n_iter / device).
  2. `method` and `epsilon` are per-call parameters, not a hardcoded
     GENERATOR_NAME/eps=10 -- one common module serves all four methods.
  3. A thin `run_method(method, ...)` helper holds the per-method loop so the
     four scripts/ files stay ~10 lines each.

Everything else -- SwapMIALabeller / SwapTargetedMIA (exact-swap D+/D-
construction), threat-model disk caching, the 5-attack battery, per-attack
JSON caching in run_attack, save_roc_report -- is reused verbatim from
target_strategy/common.py. See that file / evaluation/eval_tapas/eff_eps/common.py
for the fuller docstrings on why those classes exist.

Threat model: exact-knowledge data prior (a fixed BACKGROUND_SIZE background
sampled once, shared across all four methods) + black-box generator knowledge +
membership inference only. Member world D+ = background+target(t), non-member
world D- = background+alternate(t'), both size BACKGROUND_SIZE+1, deterministic
on every draw.
"""

import json
import logging
import os
import sys
import time
import traceback
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

# config.py sits next to this file (benchmark_tapas/); make it importable
# whether we're run from scripts/ or from benchmark_tapas/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TRAIN_CSV, TEST_CSV,
    CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL,
    FORMAL_EPSILON, BACKGROUND_SIZE, NUM_SYNTHETIC,
    BACKGROUND_SEED, TARGET_SEED,
    DP_METHODS, METHOD_CONFIG, TABLES_DIR, SCORES_DIR,
    method_cache_dir, method_results_dir,
    slug,
)
# Must come after `config`: importing config is what puts REPO_ROOT on sys.path,
# so `seeds` is not importable before that line has run.
from seeds import TAPAS_GENERATOR_SEED_BASE

log = logging.getLogger("benchmark_tapas")


# -- Data loading + scaling ----------------------------------------------

def _fit_scalers(train_df: pd.DataFrame) -> dict:
    """Min/max from training data only -- never test."""
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


# -- Background + target selection (shared, fixed across all 4 methods) ---

def sample_background(train_dataset: TabularDataset, seed: int = BACKGROUND_SEED):
    """Sample a fixed background of BACKGROUND_SIZE records once (same seed ->
    same background for every method). Returns (background_dataset, index_set)."""
    n = len(train_dataset.data)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=BACKGROUND_SIZE, replace=False)
    background_dataset = train_dataset.get_records(idx.tolist())
    log.info(f"Sampled fixed background of {BACKGROUND_SIZE} records (seed={seed})")
    return background_dataset, set(idx.tolist())


def select_random_target(train_dataset: TabularDataset, background_indices: set,
                         seed: int = TARGET_SEED):
    """Pick target t and alternate t' uniformly at random from records not in
    the background (same seed -> same t/t' for every method). Random selection
    is the natural default here; the target-selection *effect* is studied
    separately in target_strategy/. Returns (target_record, alternate_record)."""
    n = len(train_dataset.data)
    rng = np.random.RandomState(seed)
    available = np.array([i for i in range(n) if i not in background_indices])
    t_idx, tprime_idx = rng.choice(available, size=2, replace=False)
    target = train_dataset.get_records([int(t_idx)])
    alternate = train_dataset.get_records([int(tprime_idx)])
    log.info(f"Selected random target t at index {t_idx}, alternate t' at index {tprime_idx}")
    return target, alternate


# -- Synthcity-TAPAS generator wrapper -----------------------------------

class SynthcityGenerator(Generator):
    """Wraps a Synthcity plugin as a TAPAS Generator. Called by BlackBoxKnowledge
    once per simulated dataset: .fit() retrains from scratch, .generate() draws
    fresh synthetic data.

    epsilon is passed to the plugin ONLY when method is a DP method -- non-DP
    plugins (bayesian_network, ctgan) raise on an unexpected `epsilon` kwarg.
    plugin_kwargs carries the GANs' n_iter cap and device (see config.py).

    A VARYING random_state, ONE PER FIT. This class is called
    (num_train + num_test) times per attack, on D+/D- pairs differing in exactly
    one record, and each call must be an independent draw or the audit has no
    sampling variance to measure.

    This code used to pass no random_state at all, believing that left the
    generator free-running. It did not. Synthcity's `Plugin.__init__` defaults
    `random_state: int = 0` and `Plugin.fit()` calls
    `enable_reproducible_results(self.random_state)`, reseeding numpy/torch/random
    globally. Since ExactDataKnowledge hands every simulation the SAME background,
    every fit was a deterministic function of identical input: all D+ simulations
    produced one byte-identical synthetic dataset, all D- another. Confirmed
    2026-08-22 across all six caches -- the 1000/2500 DPGAN pilot's 3500 fits
    yielded 2 distinct datasets. That, not the threat model, is why every
    generator returned TPR=1, FPR=0 and the same eff-epsilon.

    So fit i now runs at seeds.TAPAS_GENERATOR_SEED_BASE + i. Consecutive fits --
    including the two halves of one D+/D- pair -- never share a draw, so the
    variance is real, while the audit remains reproducible end to end. The counter
    survives pickling (see __getstate__), so a resumed run continues the sequence
    rather than replaying seeds already used.

    `generate()` deliberately does NOT pass random_state: synthcity reseeds there
    only when one is given, so sampling continues from the post-fit RNG state and
    inherits the per-fit variation.
    """
    def __init__(self, method: str, description: DataDescription,
                 epsilon: float = FORMAL_EPSILON, plugin_kwargs: dict = None,
                 seed_base: int = TAPAS_GENERATOR_SEED_BASE):
        super().__init__()
        self.method = method
        self.description = description
        self.epsilon = epsilon
        self.plugin_kwargs = plugin_kwargs or {}
        self.seed_base = seed_base
        self._fit_counter = 0
        self._plugin = None

    def _plugin_args(self) -> dict:
        args = dict(self.plugin_kwargs)
        if self.method in DP_METHODS:
            args["epsilon"] = self.epsilon
        # Distinct per fit. Read before the counter is incremented in fit(), so
        # the i-th fit of this generator runs at seed_base + i.
        args["random_state"] = self.seed_base + self._fit_counter
        return args

    def fit(self, dataset, **kwargs):
        from synthcity.plugins import Plugins
        from synthcity.plugins.core.dataloader import GenericDataLoader

        df = dataset.data.copy()
        df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
        df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")

        loader = GenericDataLoader(df, target_column=TARGET_COL)
        self._plugin = Plugins().get(self.method, **self._plugin_args())
        self._plugin.fit(loader)
        self._fit_counter += 1
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

    def __getstate__(self):
        # A fitted Synthcity neural plugin (CTGAN/DPGAN) contains dynamically
        # created classes (e.g. SkipConnection(LinearLayer)) that pickle can't
        # serialise, which otherwise breaks ThreatModel.save() for the neural
        # methods (statistical plugins pickle fine, which is why this only bit
        # the GANs). Drop the fitted plugin from the pickled state: TAPAS re-fits
        # the generator from scratch per simulated dataset, so a reloaded
        # generator just re-fits on next use, and the threat model's already
        # memoised synthetic datasets (the expensive part) are unaffected.
        # _fit_counter and seed_base are deliberately NOT reset: a resumed run
        # must continue the seed sequence, not replay seeds it has already used.
        state = self.__dict__.copy()
        state["_plugin"] = None
        state["trained"] = False
        return state


# -- Exact-swap MIA labeller ---------------------------------------------

class SwapMIALabeller(tm.AttackerKnowledgeWithLabel):
    """D+ = background + target (member), D- = background + alternate (non-member).
    Both worlds are the same size (BACKGROUND_SIZE + 1) on every draw. See
    target_strategy/common.py for why TAPAS's built-in MIALabeller can't do this."""
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
    """TargetedMIA subclass using SwapMIALabeller (so isinstance(TargetedMIA)
    checks in LocalNeighbourhoodAttack / ClosestDistanceMIA still pass)."""
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


# -- Threat model (cached) -----------------------------------------------

def build_or_load_threat_model(cache_dir: Path, method: str, background_dataset,
                               target_record, alternate_record, description,
                               epsilon: float = FORMAL_EPSILON, plugin_kwargs: dict = None):
    """Load cached threat model if present (with any memoised simulations),
    otherwise build fresh. Run scripts must call threat_model.save() again after
    each attack to persist newly-generated simulations."""
    cache_path = cache_dir / "threat_model"
    if (cache_dir / "threat_model.pkl").exists():
        log.info(f"Loading cached threat model from {cache_path}.pkl")
        return tm.ThreatModel.load(str(cache_path))

    log.info(f"Building new threat model for {method} (no cache found)")
    generator = SynthcityGenerator(method, description, epsilon, plugin_kwargs)
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


# -- 5-attack battery ----------------------------------------------------

def compute_radius(target_record, background_dataset, distance, k: int = 5) -> float:
    """Radius for LocalNeighbourhoodAttack: distance to k-th nearest background record."""
    distances = np.sort(distance(target_record, background_dataset)[0])
    k = min(k, len(distances) - 1)
    return float(distances[k])


def build_attacks(target_record, background_dataset):
    """Instantiate the 5 MIA attacks spanning 4 families (matches target_strategy)."""
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


# -- Per-attack run + cache ----------------------------------------------

def run_attack(attack, threat_model, num_train: int, num_test: int,
               cache_dir: Path, results_dir: Path):
    """Train + test one attack. Caches result JSON per attack so a crash (or a
    Colab disconnect) doesn't lose finished attacks. Self-invalidates if
    num_train/num_test increases. Returns (result_dict, summary_or_None)."""
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
    log.info(f"  [{attack.label}] done -- eps_low_95={result.get('eps_low_95', float('nan')):.3f}  "
             f"auc={result['auc']:.3f}  ({wall_time_s}s, {peak_memory_mb} MB)")
    return result, summary


def save_roc_report(summaries_by_attack: dict, results_dir: Path, label: str):
    """Plot combined ROC curve for all attacks that ran this process (not cached)."""
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


def _score_rows(method: str, attack, summary) -> list:
    """Flatten one attack summary into per-test-dataset raw-score records.

    `summary.scores` is the CONTINUOUS score (attack.attack_score), `summary.predictions`
    the binary decision it was thresholded into. The two binarisation paths differ:
    TrainableThresholdAttack (ClosestDistance / LocalNeighbourhood / ProbabilityEstimation)
    learns `attack._threshold` as argmax(tpr-fpr) over the training ROC, while the
    shadow-modelling attacks (Groundhog, ShadowModelling) have no TAPAS-level threshold
    at all -- sklearn simply takes the larger class probability. The latter are recorded
    at 0.5, which is their true cutoff but a fixed one, not a learned one.

    Score scales are NOT comparable across attacks: RF probabilities are [0,1],
    ClosestDistance is a negated L2 distance, ProbabilityEstimation is a KDE log-density.
    """
    threshold = getattr(attack, "_threshold", None)
    threshold = 0.5 if threshold is None else float(threshold)
    return [
        {
            "generator": method,
            "attack": attack.label,
            "target_id": summary.target_id,
            "ground_truth": int(truth),
            "raw_score": float(score),
            "binary_prediction": int(pred),
            "threshold": threshold,
        }
        for truth, pred, score in zip(summary.labels, summary.predictions, summary.scores)
    ]


# -- Per-method driver (keeps the 4 scripts thin) ------------------------

def run_method(method: str, extra_plugin_kwargs: dict = None):
    """Full per-method benchmark run: same fixed background/target/alternate for
    every method, build/load the cached threat model, run the 5 attacks (each
    cached as it finishes), write per-method + combined CSVs and the ROC plot.

    extra_plugin_kwargs is merged over config's plugin_kwargs -- run scripts use
    it to inject the runtime-detected `device` for the GANs. Safe to re-run:
    finished attacks and memoised simulations are skipped.
    """
    cfg = METHOD_CONFIG[method]
    num_train, num_test = cfg["num_train"], cfg["num_test"]
    plugin_kwargs = {**cfg["plugin_kwargs"], **(extra_plugin_kwargs or {})}
    cache_dir = method_cache_dir(method)
    results_dir = method_results_dir(method)

    log.info(f"=== TAPAS privacy benchmark: {method} "
             f"(dp={cfg['dp']}, num_train={num_train}, num_test={num_test}, "
             f"plugin_kwargs={plugin_kwargs}) ===")

    train_dataset, _, description = load_adult_datasets()
    background_dataset, background_indices = sample_background(train_dataset)
    target_record, alternate_record = select_random_target(train_dataset, background_indices)

    threat_model = build_or_load_threat_model(
        cache_dir=cache_dir, method=method,
        background_dataset=background_dataset,
        target_record=target_record, alternate_record=alternate_record,
        description=description,
        epsilon=FORMAL_EPSILON, plugin_kwargs=plugin_kwargs,
    )

    attacks = build_attacks(target_record, background_dataset)

    rows, summaries, score_rows, no_scores = [], {}, [], []
    for attack in attacks:
        result, summary = run_attack(
            attack, threat_model,
            num_train=num_train, num_test=num_test,
            cache_dir=cache_dir, results_dir=results_dir,
        )
        result["method"] = method
        result["dp"] = cfg["dp"]
        result["kind"] = cfg["kind"]
        result["formal_epsilon"] = FORMAL_EPSILON if cfg["dp"] else None
        rows.append(result)
        if summary is not None:
            summaries[attack.label] = summary
            score_rows.extend(_score_rows(method, attack, summary))
        else:
            # run_attack short-circuited on its per-attack JSON cache, which stores
            # only aggregates. The scores cannot be recovered without re-running the
            # attack -- cheap, since the fits are memoised (see scripts/extract_scores.py).
            no_scores.append(attack.label)
        threat_model.save(str(cache_dir / "threat_model"))

    save_roc_report(summaries, results_dir, method)

    # Raw pre-threshold scores: the continuous value each attack computed for each
    # simulated dataset, before it was collapsed to a member/non-member decision.
    if score_rows:
        SCORES_DIR.mkdir(parents=True, exist_ok=True)
        scores_path = SCORES_DIR / f"raw_scores_{method}.csv"
        pd.DataFrame(score_rows).to_csv(scores_path, index=False)
        log.info(f"Saved {len(score_rows)} raw attack scores to {scores_path}")
    if no_scores:
        log.warning(f"No raw scores for {no_scores} -- served from the per-attack JSON "
                    f"cache, which does not store them. Re-run those attacks (or use "
                    f"scripts/extract_scores.py) if the scores are needed.")

    out = pd.DataFrame(rows)
    out.to_csv(results_dir / f"effeps_{method}.csv", index=False)
    log.info(f"Saved per-attack results to {results_dir / f'effeps_{method}.csv'}")

    # Merge into the cross-method comparison table.
    combined_path = TABLES_DIR / "benchmark_privacy_per_attack.csv"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        existing = existing[existing["method"] != method]
        out = pd.concat([existing, out], ignore_index=True)
    out.to_csv(combined_path, index=False)
    log.info(f"Updated combined table at {combined_path}")
    log.info(f"=== Done: {method} ===")
    return rows
