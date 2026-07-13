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

## Implementation (via Synthcity)

All methods use `Plugins().get(METHOD, ...)` with a `GenericDataLoader(df, target_column='income')` and generate N=21,523 rows (matching training set size). Only explicitly non-default parameters are listed — everything else is Synthcity defaults.

| Method | Synthcity plugin | Explicit params |
|---|---|---|
| Bayesian Network | `bayesian_network` | — |
| PrivBayes | `privbayes` | `epsilon=1.0` |
| CTGAN | `ctgan` | — |
| DPGAN | `dpgan` | `epsilon=1.0` |

**Bayesian Network**
- Uses `pgmpy` under the hood. Learns structure via the Chow-Liu algorithm (greedy tree maximising pairwise mutual information). Fits CPTs from data, samples by traversing the DAG.
- Continuous columns are discretised for structure learning then un-discretised on generation (explains small floating-point offsets in output).
- Runtime: 12.6s, 43.6 MB peak (local CPU)

**PrivBayes**
- Same Chow-Liu structure learning as BN, but the exponential mechanism selects which edges to include (DP structure learning). Laplace noise added to CPTs before sampling.
- Runtime: 165.7s, 21.5 MB peak (local CPU) — overhead is DP noise calibration

**CTGAN**
- Continuous columns: VGM clustering then per-mode Gaussian normalisation (mode-specific normalisation). Categorical columns: conditional vector explicitly oversamples rare categories during training so the GAN doesn't ignore them.
- Training: WGAN-GP loss, `n_iter=2000` max, `patience=5` early stopping. Stopped at **399 iterations**.
- Runtime: 4039.8s (~67 min), 64.8 MB peak (local CPU)

**DPGAN**
- CTGAN architecture with DP-SGD on the discriminator's gradient updates (via Opacus). The generator is not directly noised — privacy comes from the discriminator never memorising individuals.
- Same training loop: `n_iter=2000` max, `patience=5`. Stopped at **299 iterations**.
- Requires `opacus<1.5` — Opacus 1.5.x introduced `nn.RMSNorm` which requires PyTorch ≥ 2.4.
- Runtime: 6755.5s (~112 min), 134.0 MB peak (local CPU) — slower than CTGAN due to per-sample gradient clipping overhead

## Synthetic Data Generation

Uses the Adult Census training data at `data/adult_train.csv`. 

In sdg/, each `{method}.ipynb notebook `
  - Loads `data/adult_train.csv` with explicit dtypes (category/float)
  - Wraps everything in try/except — errors write to `sdg/{method}_logs.txt`
  - Saves synthetic data as `synthetic_data/{method_name}_synthetic.csv` 
  - Times wall-clock + peak memory via tracemalloc and appends a row of computational overhead to `sdg/computational_overhead.csv`


# Notes
`bayesian_network.ipynb` and `privbayes.ipynb` ran locally on my Macbook's CPU. 
`ctgan.ipynb` and `dpgan.ipynb` ran on my Macbook's CPU and Google Colab's T4 GPU. 

---

## Evaluation

## Evaluation Metrics
CSV column names in `evaluation/{tool}.py` are listed as `metrics`.

### Fidelity
**SDMetrics**: 
- *CorrelationSimilarity* - compares correlation matrices of numerical column pairs. Tells you if pairwise relationships are preserved.
    - `CorrelationSimilarity`
- *ContingencySimilarity* - compares joint frequency tables of categorical column pairs. captures whether the generator preserves multi-column categorical structure, not just individual columns.
    - `ContingencySimilarity`
- *KSComplement, TVComplement* - similarity of a real column vs. a synthetic column in terms of the column shapes - aka the marginal distribution or 1D histogram of the column. KSComplement for continuous, numerical data; TVComplement for discrete, categorical data.
    - `KSComplement`, `TVComplement`

### Utility
**synthcity**: 
- *eval_performance (Train on Synthetic, Test on Real)* - PerformanceEvaluatorLinear/XGB, FeatureImportanceRankDistance
    - `performance.linear_model.gt`, `performance.linear_model.syn_id`, `performance.linear_model.syn_ood`, `performance.xgb.gt`, `performance.xgb.syn_id`, `performance.xgb.syn_ood`, `performance.feat_rank_distance.corr`, `performance.feat_rank_distance.pvalue`

### Privacy
**TAPAs**: 
- *MIAttackReport, BinaryAIAttackReport* - membership- or attribute-inference attacks (modifiable via TM framework)
    - `mia_auc`, `mia_advantage`, `mia_privacy_gain`, `mia_eff_epsilon`, `mia_tp`, `mia_fp`
    - `aia_auc`, `aia_advantage`, `aia_privacy_gain`, `aia_eff_epsilon`
- *ROCReport* - aggregates summaries by plotting a ROC (receiver operating characteristic) curve for each attack
    - outputs plots saved to `results/tapas/{method}/`
- *EffectiveEpsilonReport* - effective epsilon of the worst privacy leakage across all simulated attacks
    - `eps_low_90`, `eps_high_90` (across all MIAs only)

**Note**: 
- `eval_tapas.py`, `results/tapas`, `results/tapas_results.csv` are iteration 1 (naive 1 MIA, 1 AIA)
- `evaluation/eval_tapas/eff_eps`, `results/tapas_results` are iteration 2 (computing effective epsilon for PrivBayes, DPGAN across MIAs)
- `target_strategy/` are iteration 3 (final research direction to investigate effect of target selection strategy on effective epsilon computation)

### Computational Overhead
Measured in each synthetic method's notebook (across `.fit()` + `.generate()`). 
- *Wall clock time (s)* - via `time`
- *Peak memory (MB)* - via `tracemalloc` (Python-level memory allocations only)

---

## Key Concepts

- **DP is a guarantee, not a technique.** Noise mechanisms (Laplace, Gaussian, exponential) achieve it — noise is added to aggregate statistics, not raw rows.
- **Utility vs fidelity:** Utility = downstream task performance. Fidelity = statistical similarity (marginals, correlations).
- **ε_eff vs theoretical ε:** ε_eff = `ln(TP/FP)` (empirical lower bound on privacy leakage) should be ≤ theoretical ε. If it exceeds it, the implementation doesn't guarantee the theoretical ε.
- **TAPAS threat model:** attacker knowledge (no-box / black-box / white-box) × generator knowledge × target.

---

## Commit convention

`feat:` `fix:` `chore:` `docs:` `refactor:` `test:`
