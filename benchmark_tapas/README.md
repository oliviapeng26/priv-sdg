# benchmark_tapas

TAPAS membership-inference audit of the 2x2 generator grid (statistical/neural x non-DP/DP)
on Adult Census. All four generators share one threat model: exact-knowledge data prior with a
fixed 499-record background, black-box generator knowledge, and the same target/alternate pair,
so any difference between them is the generator's.

| generator | family | DP | formal eps |
|---|---|---|---|
| bayesian_network | statistical | no | - |
| privbayes | statistical | yes | 1.0 |
| ctgan | neural | no | - |
| dpgan | neural | yes | 1.0 (reported 2.0, neighbouring-relation mismatch) |

Both GANs are capped at `n_iter=50`, set by `neural_tuning/convergence_check.py`. That makes
ctgan vs dpgan a clean DP ablation. bayesian_network vs privbayes is NOT one: different
encoders, and privbayes redraws its DAG every fit while bayesian_network's is deterministic.

## Layout

```
config.py                 experiment constants (counts, n_iter, paths, threat model)
common.py                 SynthcityGenerator, SwapMIALabeller, run_attack, run_method
privacy_analysis.ipynb    all analysis: tables, separation, decisiveness, heatmap, tradeoff
scripts/                  run_{bn,privbayes,ctgan,dpgan}.py, extract_scores.py
neural_tuning/            convergence_check.py (picks n_iter), probe_fit_time.py
results/                  per_method/ scores/ tables/ figures/ convergence/
cache/                    memoised threat models, gitignored
preseed_LEGACY/           everything invalidated by the seeding bug; see its README
```

## Running an audit

From the repo root, with the venv active. Each script runs the 5-attack battery against one
generator and writes both the effective-epsilon tables and the raw pre-threshold scores.

```
python benchmark_tapas/scripts/run_bn.py
python benchmark_tapas/scripts/run_privbayes.py
python benchmark_tapas/scripts/run_ctgan.py
python benchmark_tapas/scripts/run_dpgan.py
```

Roughly 3 h total on the GPU workstation, dominated by privbayes and dpgan. Resumable: finished
attacks and memoised simulations are skipped on re-run. `cache/` must be empty before a fresh
audit, or the cached threat model is reloaded instead of rebuilt.

Then open `privacy_analysis.ipynb` and run it top to bottom. It regenerates every table in
`results/tables/` and every figure in `results/figures/`. It needs pandas, numpy, matplotlib
and scipy only, so it runs outside the GPU venv.

`scripts/extract_scores.py` re-derives the raw scores from cached threat models without
re-generating data. Only needed if an audit was resumed from cache and the score CSVs are
missing, since `run_attack` short-circuits on its per-attack JSON cache, which stores aggregates
but not scores.

## Fit arithmetic

`fits = num_train + num_test`, not twice that. `num_train` counts synthetic datasets, one
generator fit each; the labeller halves it into pairs and emits both worlds. At 50/100 that is
150 fits, 75 D+ and 75 D- pairs.

## Generator seeding

Each fit runs at `seeds.TAPAS_GENERATOR_SEED_BASE + i`, so no two simulations and no D+/D- pair
share a draw. This is load-bearing, not cosmetic. Passing no `random_state` does not leave the
generator free-running: synthcity defaults it to 0 and reseeds numpy/torch/random globally on
every `fit()`. With a fixed background feeding every simulation, that made all D+ datasets
byte-identical and all D- datasets byte-identical, giving an effective sample size of 2 and
forcing TP=1 / FP=0 for every generator regardless of DP. Every result produced before
2026-08-23 is invalid for this reason and lives in `preseed_LEGACY/`.

## Reading the results

`results/tables/summary_combined.csv` is the headline table. The two axes are TSTR XGBoost AUC
(utility, from `evaluation/eval_utility.py`) and mean membership advantage across the 5 attacks
(leakage).

Interpret AUC against the null band, not against 0.5. With 50 members and 50 non-members an
attack with no signal still scatters with s.d. 0.058, so the 95% null band is [0.386, 0.614].
At the current counts no generator x attack cell falls outside it. Resolving an AUC of 0.55
would need roughly 260 per class, i.e. `num_test` around 520.
