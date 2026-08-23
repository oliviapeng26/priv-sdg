# preseed_LEGACY

Everything here was produced before the per-fit generator seed was introduced on 2026-08-23.
None of it is usable. Kept for provenance only.

## The bug

`SynthcityGenerator.fit()` passed no `random_state`, on the assumption that this left the
generator free-running. It did not. Synthcity's `Plugin.__init__` defaults `random_state: int = 0`
and `Plugin.fit()` calls `enable_reproducible_results(self.random_state)`, which reseeds numpy,
torch and `random` globally.

The threat model uses `ExactDataKnowledge`, so every simulation trains on the identical
background. Identical input plus identical seed means identical output: every D+ simulation
produced one byte-identical synthetic dataset, and every D- simulation produced another. Two
distinct datasets in total, however many fits were run. The 1000/2500 DPGAN pilot's 3500 fits
yielded exactly 2.

## What that did to the numbers

With one dataset per class, any attack that separates them at all separates them perfectly. That
is the entire source of:

- TP = 1.0 and FP = 0.0 for every generator
- `eps_low_95 = 2.209964` identical to six decimals across all four generators
- every AUC landing on exactly 0.000, 0.500 or 1.000
- the count sweeps showing no improvement from 50/100 up to 1000/2500, since more fits only
  added copies

The saturation was read at the time as evidence that the threat model, not the sample size, was
the binding constraint. That conclusion is not supported. After the fix, sample size genuinely
is the constraint.

## The fix

Fit i runs at `seeds.TAPAS_GENERATOR_SEED_BASE + i`. See the `seeds.py` and
`benchmark_tapas/common.py` docstrings.

## Scope

Affected: everything in this folder, and `target_strategy/`, which has the same omission at
`target_strategy/common.py:141`.

Not affected: utility and fidelity (`sdg/generate_runs.py` passes a varying `random_state` per
run across `RUN_SEEDS`), and `neural_tuning/convergence_check.py`, which passes `random_state=seed`
explicitly. The n_iter = 50 decision therefore still stands.

## Contents

```
scripts/   run_pilot_counts.py, run_extra_counts.py   the count sweeps
           run_cdtgan_colab.ipynb                     Colab runner, superseded by the GPU workstation
analysis/  aggregate.py, tradeoff.py, attack_heatmap.py, plot_scores.py, read_tables.ipynb
           all merged into benchmark_tapas/privacy_analysis.ipynb
           eff_eps_vs_counts.py has no successor: its only input was the invalid count sweep
results/   pilot_counts/ and the pre-fix summary tables and figures
caches/    cache_LEGACY/ and cache_pilot_*/, gitignored, ~900 MB
           each holds 2 distinct synthetic datasets duplicated up to 3500 times
```

`results/summary_utility.csv` here is empty of data beyond labels. `aggregate.py` ran on
2026-08-15, a day before `results/utility_summary.csv` existed, and its join silently produced a
label-only table rather than failing. `privacy_analysis.ipynb` asserts against that case.
