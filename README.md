# priv-sdg

Benchmarking synthetic data generation methods for privacy-preserving ML — comparing DP-guaranteed vs non-DP methods on utility and empirical privacy.

**Core research question:** Do non-DP synthesis methods achieve comparable utility and privacy empirically to DP-guaranteed methods? If so, under what conditions (data domain, synthesis paradigm, attack type)?

---

## Setup

Python 3.10 via conda:

```bash
conda create -n priv-sdg python=3.10 -y
conda activate priv-sdg
pip install -r requirements.txt
pip install "pandas>=2.1,<3.0"  # TAPAS downgrades pandas; re-pin after install
```

**Dependencies:** Synthcity (generation + utility eval), TAPAS (adversarial privacy attacks), SDMetrics (privacy proxies).

---

## Dataset

**Adult Census** (UCI ML Repository) — ~32,500 rows, 15 columns, mixed categorical + continuous.

- Raw file: `data/adult.data` — no header, spaces after commas, missing values as `?`
- `data/` is gitignored

**Preprocessing** (see `data_preprocessing.ipynb`):
- Dropped ~7% of rows with missing values (`?`)
- Dropped `fnlwgt` (census sampling weight — not a real feature) and `education` (redundant with `education_num`)
- Deduplicated **after** column drops — `fnlwgt` was acting as a unique-ifier, so rows that were actually identical only became duplicates once it was removed
- Stratified 80/20 train/test split on `income` - the sensitive/secret target attribute for attackers
    - synthesisers only see the train split. holdout provides genuine non-members for TAPAS membership inference attacks.
- Output: `adult_clean.csv`, `adult_train.csv`, `adult_test.csv`

**Column schema (post-processing, 13 columns):**
- Continuous: `age, education_num, capital_gain, capital_loss, hours_per_week`
- Categorical: `workclass, marital_status, occupation, relationship, race, sex, native_country, income`
- Target (for utility eval + TAPAS attribute inference): `income`

---

## Method

## Synthesis Methods

| | **Statistical** | **Neural** |
|---|---|---|
| **No DP** | Bayesian Network | CTGAN |
| **DP** (fixed ε = 1.0, vary later) | PrivBayes | DPGAN |

- **Bayesian Network** - learns conditional dependencies (as conditional probability tables CPTs) between attributes as a DAG, then samples new data points by traversing the graph and drawing from each attribute's conditional distribution given its parents. Handles mixed typed attributes naturally since each node of DAG has its own distribution.
- **PrivBayes** - Bayesian Network with DP guarantee by using the exponential mechanism to select edges/attribute dependencies to include in the network, and then adds Laplace noise to CPTs. 

PrivBayes has same architecture as BN, so any metric difference is purely the DP cost.

- **CTGAN** (Conditional Tabular GAN) — handles mixed-type columns by using mode-specific normalisation for continuous columns (clusters values then normalises within each cluster) and conditional generation for categorical columns (specifically oversamples rare categories so the GAN doesn't ignore them).
- **DPGAN** — CTGAN with DP-SGD applied to the discriminator's training (noisy gradients). 

DPGAN has same architecture as CTGAN, so any metric difference is purely the DP cost.


# Synthetic Data Generation

Uses the Adult Census training data at `data/adult_train.csv`. 

In sdg/, each `{method}.ipynb notebook `
  - Loads `data/adult_train.csv` with explicit dtypes (category/float)
  - Wraps everything in try/except — errors write to `sdg/{method}_logs.txt`
  - Saves synthetic data as `synthetic_data/{method_name}_synthetic.csv` 
  - Times wall-clock + peak memory via tracemalloc and appends a row of computational overhead to `sdg/computational_overhead.csv`


# Note
`bayesian_network.ipynb` and `privbayes.ipynb` ran locally on my Macbook's CPU. 
`ctgan.ipynb` and `dpgan.ipynb` ran on Google Colab's T4 GPU. 

---

## Evaluation

## Evaluation Metrics

### Fidelity
- **synthcity**: 
    - *eval_statistical* - JensenShannonDistance (per-column distribution comparison), MaximumMeanDiscrepancy (whole-dataset distribution comparison), AlphaPrecision (manifold/data pattern in high dimension space quality).
- **SDMetrics**: 
    - *CorrelationSimilarity* - compares correlation matrices of numerical column pairs. Tells you if pairwise relationships are preserved.
    - *ContingencySimilarity* - compares joint frequency tables of categorical column pairs. captures whether the generator preserves multi-column categorical structure, not just individual columns.
    - *KSComplement, TVComplement* - similarity of a real column vs. a synthetic column in terms of the column shapes - aka the marginal distribution or 1D histogram of the column. KSComplement for continuous, numerical data; TVComplement for discrete, categorical data.

### Utility
- **synthcity**: 
    - *eval_performance (Train on Synthetic, Test on Real)* - PerformanceEvaluatorLinear/MLP/XGB, FeatureImportanceRankDistance
    - *eval_detection (distinguish real from synthetic)* - detection_linear/MLP/XGB
- **SDMetrics** (for cross-checking synthcity): 
    - *ML Efficacy: Single Table* - BinaryLogisticRegression
    - *Detection: Single Table* - LogisticDetection

### Privacy
- **SDMetrics**: 
    - *DCRBaselineProtection* - measures distance between synthetic and closest real record
    - *NewRowSynthesis* - whether each row in the synthetic data is novel, or exactly matches an original row in the real data
- **TAPAs**: 
    - *MIAttackReport, AIAttackReport* - membership- or attribute-inference attacks (modifiable via TM framework)
        - reports accuracy, true_positive_rate, false_positive_rate, mia_advantage, privacy_gain, auc, effective_epsilon
    - *ROCReport* - aggregates summaries by plotting a ROC (receiver operating characteristic) curve for each attack
    - *EffectiveEpsilonReport* - effective epsilon of the worst privacy leakage across all simulated attacks

---

## Key Concepts

- **DP is a guarantee, not a technique.** Noise mechanisms (Laplace, Gaussian, exponential) achieve it — noise is added to aggregate statistics, not raw rows.
- **Utility vs fidelity:** Utility = downstream task performance. Fidelity = statistical similarity (marginals, correlations).
- **ε_eff vs theoretical ε:** ε_eff = `ln(TP/FP)` (empirical lower bound on privacy leakage) should be ≤ theoretical ε. If it exceeds it, the implementation doesn't guarantee the theoretical ε.
- **TAPAS threat model:** attacker knowledge (no-box / black-box / white-box) × generator knowledge × target.

---

## Commit convention

`feat:` `fix:` `chore:` `docs:` `refactor:` `test:`
