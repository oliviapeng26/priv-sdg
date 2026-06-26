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

## Methods

---

## Evaluation

**Utility:**


**Privacy:**

---

## Key Concepts

- **DP is a guarantee, not a technique.** Noise mechanisms (Laplace, Gaussian, exponential) achieve it — noise is added to aggregate statistics, not raw rows.
- **Utility vs fidelity:** Utility = downstream task performance. Fidelity = statistical similarity (marginals, correlations).
- **ε_eff vs theoretical ε:** ε_eff = `ln(TP/FP)` (empirical lower bound on privacy leakage) should be ≤ theoretical ε. If it exceeds it, the implementation doesn't guarantee the theoretical ε.
- **TAPAS threat model:** attacker knowledge (no-box / black-box / white-box) × generator knowledge × target.

---

## Commit convention

`feat:` `fix:` `chore:` `docs:` `refactor:` `test:`
