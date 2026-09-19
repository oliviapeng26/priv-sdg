# benchmark_tapas

TAPAS membership-inference audits of six tabular generators on Adult Census: statistical vs
neural, DP vs non-DP. All of them share one threat model (exact-knowledge data prior with a
fixed 499-record background, black-box generator knowledge, the same target/alternate pair and
the same 5-attack battery), so any difference between them is the generator's.

| generator | family | DP | library | status |
|---|---|---|---|---|
| bayesian_network | statistical | no | Synthcity | main results |
| aim | statistical | ε = 1.0, δ = 1e-9 | SmartNoise | main results |
| ctgan | neural | no | Synthcity | main results |
| dpctgan | neural | ε = 1.0 (a stopping rule, σ = 5) | SmartNoise | main results |
| privbayes | statistical | ε = 1.0 | Synthcity | under investigation (paper App. A.3) |
| dpgan | neural | ε = 1.0 | Synthcity | under investigation, ε_eff spike at ε = 1.0 |

Both Synthcity GANs are capped at `n_iter=50` (`tuning/convergence_check.py`). BN vs PrivBayes is
not a clean DP ablation: different encoders, and PrivBayes redraws its DAG every fit.

## Layout

```
config.py  common.py     shared constants, and the TAPAS pipeline + SynthcityGenerator (imported by everything)
tapas_wrappers/          TAPAS Generator wrappers for the SmartNoise generators (aim_generator.py, dpctgan_generator.py)
audits/                  every script that runs a TAPAS audit (table below)
eps_sweep_pipeline/      DPGAN ε sweep: eps_sweep_{sigma_check,generate,evaluate,aggregate}.py
diagnostics/             DPGAN ε = 1.0 spike diagnosis (signal scan, recompute, ε nudge), disentangle run
tuning/                  convergence_check.py (picks n_iter, paper Table 4), probe_fit_time.py
privacy_analysis.ipynb   all tables and figures; see its first cell for the paper map
results/                 outputs, by experiment (below)
cache/                   memoised threat models, gitignored
```

Wrapper file names (`aim_generator`, `dpctgan_generator`) are kept as-is: threat-model pickles in
`cache/` refer to their classes by module name.

## What was run, and where the results are

| Experiment | Results | Script |
|---|---|---|
| Privacy at ε = 1.0, four audit sizes (50/100, 200/500, 500/1000, 1000/2500) | `results/sample_size_sweep/{generator}/{stage}/` | BN, PrivBayes, CTGAN, DPGAN: `audits/run_synthcity_sample_size_sweep.py`. AIM: `audits/run_aim_audit.py`. DP-CTGAN: `audits/run_dpctgan_audit.py` |
| Privacy at ε = 0.1, 1, 10, 100 at 1000/2500 | `results/eps_sweep/{dpgan,aim,dp_ctgan}/eps{e}/` | DPGAN: `audits/run_dpgan_eps_sweep.py`. AIM and DP-CTGAN: same scripts as above with `--epsilon` |
| DP-CTGAN epoch cap 300 / 500 / 750 / 1000 at ε = 100 | `results/extras/dp_ctgan_epoch_cap/` (cap 300 = `eps_sweep/dp_ctgan/eps100`) | `audits/run_dpctgan_audit.py --epoch-cap` |
| DPGAN ε = 1.0 spike diagnosis | `results/extras/dpgan_spike_diagnosis/` | `diagnostics/` |

ε = 1 is not duplicated in `eps_sweep/`: it is the 1000/2500 stage of `sample_size_sweep/`, and
`privacy_analysis.ipynb` stitches the two together. AIM ε = 100 was not run (about 12 h of
synthetic pool per arm). Neither `privbayes` nor the non-DP generators have an ε sweep.

Per-script detail: `audits/README_aim.md`, `audits/README_dpctgan.md`. DP-CTGAN needs the Opacus 0.x
API, so it runs in the workstation's separate environment.

## Running an audit

From the repo root with the venv active. Every script is resumable: finished attacks and memoised
simulations are skipped on re-run. `cache/` must be empty (or hold that arm's own pool) before a
fresh audit, or the cached threat model is reloaded instead of rebuilt.

```
python benchmark_tapas/audits/run_synthcity_sample_size_sweep.py               # all four Synthcity generators
python benchmark_tapas/audits/run_synthcity_sample_size_sweep.py --methods dpgan
python benchmark_tapas/audits/run_aim_audit.py --num-train 1000 --num-test 2500
python benchmark_tapas/audits/run_dpctgan_audit.py --probe 3                   # pre-flight, then the full audit
python benchmark_tapas/audits/run_dpgan_eps_sweep.py --epsilons 0.1 10 100
```

Then open `privacy_analysis.ipynb` and run it top to bottom. It regenerates the tables in
`results/tables/` and the figures in `results/figures/{paper,supplementary}/`. It needs pandas,
numpy, matplotlib and scipy only, so it runs outside the GPU venv.

## Fit arithmetic

`fits = num_train + num_test`, not twice that. `num_train` counts synthetic datasets, one
generator fit each; the labeller halves it into pairs and emits both worlds. At 50/100 that is
150 fits, 75 D+ and 75 D- pairs.

## Generator seeding

Each fit runs at `seeds.TAPAS_GENERATOR_SEED_BASE + i`, so no two simulations and no D+/D- pair
share a draw. This is load-bearing, not cosmetic. Passing no `random_state` does not leave the
generator free-running: Synthcity defaults it to 0 and reseeds numpy/torch/random globally on
every `fit()`. With a fixed background feeding every simulation, that made all D+ datasets
byte-identical and all D- datasets byte-identical, giving an effective sample size of 2 and forcing
TP=1 / FP=0 for every generator regardless of DP. Every result produced before 2026-08-23 was
invalid for this reason and has been deleted.

## Reading the results

Interpret AUC against the null band, not against 0.5: with `num_test` test datasets an attack with
no signal still scatters, and the 95% null half-width narrows from ±0.1146 at 100 test pairs to
±0.02264 at 2500. ε_eff is a Clopper-Pearson lower bound from the strongest attack, so read
`eps_low_95` against the formal ε (a correct implementation has `eps_low_95 <= ε`).

The two model-training attacks (Groundhog, ShadowModelling) vary between re-runs on identical
synthetic data (compare `archive/dp_ctgan_sep2_attack_rerun/` with the current ε = 1 and 100 arms).
