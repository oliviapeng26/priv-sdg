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

AIM (`sdg/aim.py`) needs one extra step, because smartnoise-synth pins `opacus<0.15` while synthcity's DPGAN needs opacus 1.x:

```bash
pip install --no-deps smartnoise-synth==1.0.8
```

The pin is a packaging artefact, not a real constraint — AIM's actual dependency chain is mbi + jax + opendp + smartnoise-sql, and importing `snsynth.aim` pulls neither opacus nor torch. Everything it genuinely needs is already in `requirements.txt`, so `--no-deps` leaves nothing missing, and **AIM and DPGAN both run in the one `priv-sdg` env**. `pip check` will flag the unsatisfied opacus pin; that's expected. Don't let pip "fix" it by downgrading opacus — DPGAN breaks silently if you do.

---

## Dataset

**Adult Census** (UCI ML Repository) — ~32,500 rows, 15 columns, mixed categorical + continuous.

- Raw file: `data/adult.data` — no header, spaces after commas, missing values as `?`
- `data/` is gitignored

**Preprocessing** (see `data_preprocessing.ipynb`):
- Split seed comes from `seeds.py` (`DATA_SPLIT_SEED = 42`); the notebook's last cell re-derives the split from `adult_clean.csv` and asserts the CSVs on disk are that exact partition
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

---

## Reproducibility (`seeds.py`)

One source of truth for every fixed seed, at the repo root (not `config/` — `benchmark_tapas/` and `target_strategy/` already import their own `config.py` as the top-level name `config`, so a `config/` package would shadow-collide with them).

| Seed | Value | Controls |
|---|---|---|
| `DATA_SPLIT_SEED` | 42 | the 80/20 stratified split → 21,523 / 5,381 |
| `TAPAS_BG_SEED` | 42 | the 499-record background sample (`benchmark_tapas`) |
| `TAPAS_TARGET_SEED` | 43 | target `t` / alternate `t'` selection |
| `RUN_SEEDS` | `[100…104]` | per-run generator + classifier seed, for mean ± std |

The three fixed values are the ones the repo already ran with, hoisted rather than changed — a seed's numeric value is arbitrary, and changing them would invalidate `data/`, all five `synthetic_data/*.csv`, every `benchmark_tapas/cache/*/threat_model.pkl` and every committed table for no scientific gain. Treat them as frozen.

`set_all_seeds(seed)` seeds numpy, `random`, and torch **only if torch is already imported** — the statistical paths have no torch randomness, and importing torch into a process that later starts an xgboost thread pool is exactly the OpenMP collision documented below.

All four Synthcity plugins accept `random_state` (checked, not assumed): it is consumed in `Plugin.fit`, can be passed again to `Plugin.generate(random_state=…)`, and both route through `enable_reproducible_results`, which seeds numpy + torch + random. AIM takes no seed, and its Gaussian noise comes from opendp's unseedable CSPRNG, so AIM runs are not bit-reproducible at any seed.

**Deliberate exception — the TAPAS generator is left unseeded.** TAPAS re-fits the generator 2 × (`num_train` + `num_test`) times per attack on D⁺/D⁻ pairs differing in one record. A constant `random_state` would give those pairs common random numbers, collapsing the generator's sampling variance and inflating attack success — a change to the threat model, not a reproducibility fix. The privacy run is pinned where it should be (background + target draw); the per-simulation generator randomness stays free.

---

## Method

## Synthesis Methods

| | **Statistical** | **Neural** |
|---|---|---|
| **No DP** | Bayesian Network | CTGAN |
| **DP** (fixed ε = 1.0, vary later) | PrivBayes, AIM | DPGAN |

- **Bayesian Network** - learns conditional dependencies (as conditional probability tables CPTs) between attributes as a DAG, then samples new data points by traversing the graph and drawing from each attribute's conditional distribution given its parents. Handles mixed typed attributes naturally since each node of DAG has its own distribution.
- **PrivBayes** - Bayesian Network with DP guarantee by using the exponential mechanism to select edges/attribute dependencies to include in the network, and then adds Laplace noise to CPTs. 

**Correction (2026-08-15): BN and PrivBayes are NOT the same architecture in Synthcity, so a metric difference between them is not purely the DP cost.** Verified against the installed plugins:

| | `bayesian_network` | `privbayes` |
|---|---|---|
| continuous encoding | `TabularEncoder(max_clusters=10)` — GMM mode-specific normalisation, ≤10 clusters/column, **plus** `encoder_noise_scale=0.1` adding `N(0, 0.1)` to every `.value` component in `_encode_decode` | `n_bins=100` uniform bins, no noise beyond DP |
| structure learning | `tree_search` (Chow-Liu, in-degree 1; `struct_max_indegree=4` is ignored by this search method) | greedy degree-K network, `K=0` → auto |

Measured effect (5 seeds BN, 1 seed matched arm): the encoder accounts for the **fidelity** gap entirely — giving BN 100 clusters and no jitter moves KSComplement 0.727 → 0.972 — but for **utility** it accounts for none of it, TSTR holding at 0.759 against BN's 0.759 ± 0.010. So the ~0.10 AUC gap to PrivBayes survives the confound and is not an encoding artefact; the remaining suspect is the in-degree-1 tree, which can't represent the higher-order interactions XGBoost needs for `income`. Untested.

Treat BN↔PrivBayes as a *paradigm* comparison, not a controlled DP ablation.

- **CTGAN** (Conditional Tabular GAN) — handles mixed-type columns by using mode-specific normalisation for continuous columns (clusters values then normalises within each cluster) and conditional generation for categorical columns (specifically oversamples rare categories so the GAN doesn't ignore them).
- **DPGAN** — CTGAN with DP-SGD applied to the discriminator's training (noisy gradients). 

DPGAN has same architecture as CTGAN, so any metric difference is purely the DP cost.

- **AIM** (Adaptive and Iterative Mechanism) — marginal-based DP synthesiser. Iteratively picks the 2-way marginal currently worst-approximated by its model (exponential mechanism), measures it under the Gaussian mechanism, and refits a graphical model (Private-PGM) to all noisy measurements so far; samples from the final model. Unlike PrivBayes it spends budget *adaptively*, re-deciding what to measure each round instead of fixing a network up front.

AIM is the DP-statistical arm's second entry: same quadrant as PrivBayes, so the two isolate *how* a fixed ε = 1.0 is spent (fixed Bayesian network + Laplace vs adaptive marginal selection + Gaussian).

## Implementation

The first four methods use `Plugins().get(METHOD, ...)` with a `GenericDataLoader(df, target_column='income')` and generate N=21,523 rows (matching training set size). Only explicitly non-default parameters are listed — everything else is Synthcity defaults. **AIM is not a Synthcity plugin** and runs standalone (see below).

| Method | Synthcity plugin | Explicit params |
|---|---|---|
| Bayesian Network | `bayesian_network` | — |
| PrivBayes | `privbayes` | `epsilon=1.0` |
| CTGAN | `ctgan` | — |
| DPGAN | `dpgan` | `epsilon=1.0` |
| AIM | — (SmartNoise `snsynth.aim.AIMSynthesizer`) | `epsilon=1.0, delta=1e-9` |

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

**AIM** (`sdg/aim.py`, not a notebook — it needs a different env, see Notes)
- `snsynth.aim.AIMSynthesizer` — SmartNoise's port of the reference implementation (`ryan112358/private-pgm`, `mechanisms/aim.py`), running on `mbi` 1.1.0 (the jax Private-PGM inference engine) underneath. The pip `mbi` package ships **only** the inference engine; the AIM mechanism lives in the repo's `mechanisms/` folder, which pip does not install — hence SmartNoise.
- Defaults, stated explicitly in the script: `degree=2` (workload = all 2-way marginals), `max_cells=10000`, `max_model_size=80` MB, `rounds=16 × n_cols = 208`. These are exactly the reference implementation's `default_params()`, so the configuration is the published one rather than an ad-hoc pick.
- **Discretisation:** AIM needs an all-discrete domain, so the 5 continuous columns are binned before fitting and decoded after. Bin edges are fixed constants from the public Adult codebook (age 17–90 / 20 bins, education_num 1–16 / 16 bins, hours_per_week 1–99 / 20 bins), **never estimated from the training data**, so `preprocessor_eps=0` and the full ε = 1.0 goes to AIM. `capital_gain`/`capital_loss` (90.7% / 94.8% exact zeros) get a dedicated zero bin plus 19 log-spaced bins over the positive range — equal-width bins would smear the zero spike across a 5,000-wide bin. Decoding draws uniformly inside the chosen bin and floors, matching the integer support of the real columns and returning exact 0 for the zero bin. Round-trip error with no DP noise at all: mean |Δ| = 1.3 yrs (age), 0.0 (education_num), 2.3 hrs (hours_per_week); zero fractions preserved exactly.
- **Privacy accounting — δ is *not* matched across the three DP generators:**

  | | ε | δ | ρ (zCDP) |
  |---|---|---|---|
  | PrivBayes | 1.0 | 0 (pure DP) | — |
  | **AIM** | 1.0 | **1e-9** | 0.0150 |
  | DPGAN | 1.0 | 4.65e-5 (`= 1/n`, synthcity's silent default, `gan.py:591`) | 0.0367 |

  AIM cannot be made pure-DP: it composes Gaussian measurements + exponential-mechanism marginal selection under zCDP, and ρ-zCDP converts to (ε, δ)-DP only for δ > 0 (`delta=0` sets `rho=0`, i.e. infinite noise). The gap to PrivBayes is structural, not a tuning choice.

  δ = 1e-9 is deliberate, not merely inherited. It is the strictest δ in the grid, so it keeps AIM as close as the algorithm allows to its primary contrast — PrivBayes, same quadrant, δ = 0 — and no utility win can be blamed on a loose parameter. It also matches the AIM paper (McKenna et al., [arXiv:2201.12677](https://arxiv.org/abs/2201.12677): *"varying ϵ∈[0.01,100.0] and setting δ=10⁻⁹"*), the reference implementation's `default_params()`, and SmartNoise's default, so results stay comparable to published AIM numbers.

  The cost: AIM absorbs ~56% more noise than DPGAN's budget permits (initial σ 87.8 vs 56.1) — matching DPGAN's δ would be equivalent to giving AIM ε = 1.6. That confound runs **against** AIM, so a utility win is unattackable.

  **TODO if AIM underperforms the others on utility:** rerun at δ = 4.65e-5 and report it as a sensitivity line, separating the stricter budget from the algorithm itself.

## Synthetic Data Generation

**`sdg/generate_runs.py`** — one script, all five methods, fitted and sampled once per seed in `RUN_SEEDS`.

```bash
python sdg/generate_runs.py                     # all 5 methods x 5 seeds
python sdg/generate_runs.py ctgan dpgan         # the GPU half
python sdg/generate_runs.py privbayes --runs 2  # partial
```

- Reads `data/adult_train.csv`; writes `synthetic_data/runs/{method}_seed{seed}.csv`
- Records wall clock (`time.perf_counter`) + peak memory (`tracemalloc`) across fit + generate into `results/computational_cost.csv`
- Resumable — an existing run CSV is reused and the fit skipped; `--regenerate` forces a re-fit
- Hyperparameters unchanged from the legacy notebooks: plugin defaults plus ε = 1.0 for the DP methods

**Pipeline shape.** Generation and scoring are separate processes on purpose:

```
sdg/generate_runs.py  ──>  synthetic_data/runs/*.csv  ──┬──>  evaluation/eval_fidelity.py
   (GPU, expensive)                                     └──>  evaluation/eval_utility.py
                                                                  (CPU, cheap, re-runnable)
```

This also means generation never imports xgboost and the eval scripts never import torch — which sidesteps the macOS OpenMP collision below rather than working around it.

**Device.** All generators run on the same workstation, but not on the same device, and this is intended rather than a misconfiguration:

| method | device | why |
|---|---|---|
| Bayesian Network, PrivBayes | CPU | pgmpy/numpy — no GPU path exists in the plugin |
| CTGAN, DPGAN | CUDA | torch; GPU purely for the speedup |
| AIM | jax backend | reported as whatever `jax.default_backend()` returns |

The `device` column in `results/computational_cost.csv` records which was used per run, so the write-up can state it. **Peak memory is Python-level only** — `tracemalloc` cannot see torch CUDA buffers, jax buffers, or C-extension allocations, so it understates CTGAN/DPGAN/AIM badly. Wall clock is the trustworthy cross-method column.

**Legacy.** The per-method `sdg/*_LEGACY.ipynb` notebooks produced the single unseeded draws now at `synthetic_data/{method}_synthetic_LEGACY.csv`, and `sdg/computational_overhead.csv` holds their timings. Kept for provenance — the pre-2026-08 results trace to them — but they can't be reproduced (generated without a seed). `sdg/aim.py` is **not** legacy: `generate_runs.py` imports its bin edges and encode/decode, so the discretisation can't drift.


# Notes

Full sequence, from the repo root:

```bash
conda activate priv-sdg
python sdg/generate_runs.py bayesian_network privbayes ctgan dpgan   # generation, all seeds
python evaluation/eval_fidelity.py bayesian_network privbayes ctgan dpgan
python evaluation/eval_utility.py  bayesian_network privbayes ctgan dpgan
```

Name the methods explicitly: `aim` is in each script's default set but is **deferred** until the four grid methods are working end to end (generation → fidelity → utility → privacy → figures).

Privacy is gated separately — run `benchmark_tapas/neural_tuning/convergence_check.py` first, settle `n_iter` from its mean ± std, then the TAPAS audits. See `benchmark_tapas/README.md`.

Use `conda activate`, not `conda run -n ...`, for anything long — `conda run` buffers stdout, so a 30–90 min job shows nothing until it finishes. Activated, you get AIM's per-round `Selected (...) Budget Used 0.42` lines as a progress bar.

### macOS OpenMP fix

Now largely **moot by construction**: generation (`sdg/generate_runs.py`) never imports xgboost and the eval scripts never import torch, so the two runtimes no longer meet in one process. It still applies to anything that loads both — the legacy scripts in `evaluation/evaluation_LEGACY/`, and `benchmark_tapas/neural_tuning/convergence_check.py`, which fits GANs and then scores with xgboost.

The old `eval_synthcity.py` crashed on 2026-08-08 with `OMP: Error #179` / segfault during `performance.xgb` and `performance.feat_rank_distance`. Cause: **two conflicting OpenMP runtimes in one process.** `libxgboost.dylib` bundles no libomp — it has a single rpath, `/opt/homebrew/opt/libomp/lib`, so it uses Homebrew's (22.1.8, installed Jul 3). torch bundles its own at `torch/.dylibs/libomp.dylib`. The two builds cannot coexist: XGBoost dies the instant it starts a thread pool after torch is imported.

Isolated by bisection — xgboost alone at `nthread=8` is fine, and sklearn's bundled copy is fine; **only torch's conflicts.**

Fix — make torch share the one runtime:

```bash
cd $CONDA_PREFIX/lib/python3.10/site-packages/torch/.dylibs
cp -p libomp.dylib libomp.dylib.bak
ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib libomp.dylib
```

Verified after applying: torch training, xgboost `nthread=8` after torch, sklearn `n_jobs=4`, DPGAN (torch + opacus DP-SGD), CTGAN, and the full metric set multithreaded — scores bit-identical to single-threaded.

Two caveats: `pip install -U torch` restores torch's own copy and re-breaks it (redo the symlink), and this depends on Homebrew's libomp, which XGBoost already requires on macOS anyway. If you'd rather not touch site-packages, `OMP_NUM_THREADS=1` also works and costs ~0.7s (5.7s vs 5.0s for the full metric set).

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

Metrics and their definitions are unchanged (still synthetic vs **training** data — the question is how well the generator reproduced the distribution it was fitted on). Computed by **`evaluation/eval_fidelity.py`**, once per seeded run, into `results/fidelity_per_run.csv` + `results/fidelity_summary.csv`. (`evaluation/evaluation_LEGACY/eval_sdmetrics_LEGACY.py` is the superseded single-draw version, kept for provenance.)

### Utility

**`evaluation/eval_utility.py`** — TSTR / TRTR / retention, computed manually. This is the current source of utility numbers.

- *TSTR* — train XGBoost + logistic regression on synthetic data, score AUC on `data/adult_test.csv` (the 5,381 held-out records no generator has seen)
- *TRTR* — same two classifiers trained on the full 21,523-row training set, scored on the **same** 5,381 records
- *Retention* — `TSTR / TRTR`, a like-for-like ratio because both share a test set
    - `tstr_{xgboost,logistic_regression}_auc`, `trtr_…`, `retention_…`
- Features are one-hot encoded on a layout fitted on the **real** data only (train ∪ test), so a category maps to the same column in every method and run. One-hot rather than `LabelEncoder` because an ordinal code for `occupation` invents an ordering that LR reads as real; LR additionally gets a `StandardScaler` inside its pipeline (`capital_gain` spans 0–99,999 against 0/1 dummies, and unscaled LR doesn't converge in 1000 iterations).
- Scores every seeded run and reports mean ± std. Outputs `results/utility_per_run.csv` and `results/utility_summary.csv`.

```bash
python evaluation/eval_utility.py                        # all methods with run CSVs
python evaluation/eval_utility.py bayesian_network       # one method
python evaluation/eval_utility.py --runs 2               # first 2 seeds only
```

**Superseded — `evaluation/evaluation_LEGACY/eval_synthcity_LEGACY.py`**

`Metrics.evaluate` takes the real loader it's handed — `adult_train.csv`, the rows every generator was fitted on — and makes its **own** internal 80/20 split of it, training on synthetic and scoring on that internal holdout. Those "held-out" records were in the generator's training set, so memorisation is rewarded; `performance.*.gt` is scored on the same leaky split, so the `syn_id/gt` ratio doesn't correct for it either.

`results/synthcity_results.csv` and the three figures derived from it (`utility_metric_comparison.png`, `utility_privacy_tradeoff.png`, `utility_comparison_b32.png`) have been **deleted** — the numbers weren't salvageable. The script is kept only so the leak stays documented where it happened.

Everything that consumed it has been repointed: `benchmark_tapas/config.py` (`UTILITY_RESULTS`/`FIDELITY_RESULTS`), `aggregate.py`, `tradeoff.py` (whose X axis now carries across-seed error bars), and `evaluation/analysis.ipynb`. All of them degrade to blank/NaN utility columns with an actionable warning until `eval_utility.py` has run, rather than failing.

- *(old) eval_performance (Train on Synthetic, Test on Real)* - PerformanceEvaluatorLinear/XGB, FeatureImportanceRankDistance
    - `performance.linear_model.gt`, `performance.linear_model.syn_id`, `performance.linear_model.syn_ood`, `performance.xgb.gt`, `performance.xgb.syn_id`, `performance.xgb.syn_ood`, `performance.feat_rank_distance.corr`, `performance.feat_rank_distance.pvalue`

### Privacy
**TAPAs**: 
- *MIAttackReport, BinaryAIAttackReport* - membership- or attribute-inference attacks (modifiable via TM framework)
    - `mia_auc`, `mia_advantage`, `mia_privacy_gain`, `mia_eff_epsilon`, `mia_tp`, `mia_fp`
    - `aia_auc`, `aia_advantage`, `aia_privacy_gain`, `aia_eff_epsilon`
- *ROCReport* - aggregates summaries by plotting a ROC (receiver operating characteristic) curve for each attack
    - outputs plots saved to `results/tapas_LEGACY/tapas/{method}/`
- *EffectiveEpsilonReport* - effective epsilon of the worst privacy leakage across all simulated attacks
    - `eps_low_90`, `eps_high_90` (across all MIAs only)

**Note — iterations.** `benchmark_tapas/` is the current and only privacy result for the write-up. Everything below it is superseded and kept for traceability only:
- **iteration 1** (naive 1 MIA, 1 AIA): `evaluation/eval_tapas.py` → `results/tapas_LEGACY/tapas_results.csv`, `results/tapas_LEGACY/tapas/`, plus the three figures built from it (`epsilon_comparison.png`, `mia_effective_epsilon.png`, `privacy_attack_success.png`, all now in `results/tapas_LEGACY/`)
- **iteration 2** (effective epsilon, PrivBayes + DPGAN only): `evaluation/eval_tapas/eff_eps/` → `results/tapas_LEGACY/tapas_results/`
- **iteration 3** (effect of target-selection strategy on ε_eff): `target_strategy/`

See `results/tapas_LEGACY/README.md`. These are superseded, not leaky — there's no methodological error in them, they're just narrower.

### Computational Overhead
Measured in `sdg/generate_runs.py` across `.fit()` + `.generate()`, once per run, into `results/computational_cost.csv` (`method, seed, stage, wall_clock_s, peak_memory_mb, device`).
- *Wall clock time (s)* — via `time.perf_counter`
- *Peak memory (MB)* — via `tracemalloc` (Python-level allocations only; see the device note above)

Legacy single-draw timings are at `sdg/computational_overhead.csv`, measured in the `*_LEGACY.ipynb` notebooks on a Mac CPU, so they are not comparable to workstation numbers.

---

## Key Concepts

- **DP is a guarantee, not a technique.** Noise mechanisms (Laplace, Gaussian, exponential) achieve it — noise is added to aggregate statistics, not raw rows.
- **Utility vs fidelity:** Utility = downstream task performance. Fidelity = statistical similarity (marginals, correlations).
- **ε_eff vs theoretical ε:** ε_eff = `ln(TP/FP)` (empirical lower bound on privacy leakage) should be ≤ theoretical ε. If it exceeds it, the implementation doesn't guarantee the theoretical ε.
- **TAPAS threat model:** attacker knowledge (no-box / black-box / white-box) × generator knowledge × target.

---

## Known issues (deferred, not fixed)

**`sdg/aim_tuning/aim_tuning.py` still uses the leaky TSTR.** Its `xgb_syn_id` / `xgb_gt` / `linear_syn_id` columns and the TSTR panel of `aim_tuning_utility.png` come from synthcity's `Metrics.evaluate`, with the same internal-split leak fixed everywhere else. Its *fidelity* columns and marginal-selection counts are unaffected, and those are what the bins=32 conclusion rests on — so the choice probably stands, but the utility numbers shouldn't be quoted as they are. Fix = swap in `compute_tstr`/`compute_trtr` from `evaluation/eval_utility.py` and re-run the sweep. Full note in that script's header.

**`evaluation/analysis.ipynb` has been repointed but not yet executed.** It now reads `results/utility_summary.csv` + `results/fidelity_summary.csv`, which don't exist until the GPU run, so the notebook can't be run — or verified — until then. Two cells were converted to SUPERSEDED markdown: the utility-vs-privacy scatter (the correct version is `benchmark_tapas/analysis/tradeoff.py`) and the AIM b=32 utility comparison (blocked on the `aim_tuning` leak above). Its AIM b=32 fidelity cell now merges fidelity only, for the same reason.

---

## Commit convention

`feat:` `fix:` `chore:` `docs:` `refactor:` `test:`
