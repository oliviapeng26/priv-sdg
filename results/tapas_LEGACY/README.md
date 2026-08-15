# TAPAS privacy results — iterations 1 and 2 (LEGACY)

**Not for the poster or the report.** The correct privacy iteration is
`benchmark_tapas/` — the full 5-attack MIA battery against all four generators
in the 2x2 grid, under one shared threat model. Everything in this folder is an
earlier, narrower iteration kept only so the progression stays traceable.

Nothing here is *wrong* in the way the deleted `synthcity_results.csv` was
wrong — there is no leak. These are simply superseded: fewer attacks, fewer
methods, and a threat model that later changed.

## Contents

| path | iteration | what it was |
|---|---|---|
| `tapas_results.csv`, `tapas/` | 1 | naive single MIA + single AIA, all four generators. Written by `evaluation/eval_tapas.py`. |
| `tapas_results/eff_eps_results/` | 2 | effective-epsilon across the MIA battery, **PrivBayes and DPGAN only**. Written by `evaluation/eval_tapas/eff_eps/`. |
| `epsilon_comparison.png`, `mia_effective_epsilon.png`, `privacy_attack_success.png` | 1 | figures built from `tapas_results.csv` by `evaluation/analysis.ipynb`. |

## Why the figures moved here too

All three plot iteration-1 privacy numbers. Since iteration 1 is not appearing
in the write-up, the figures derived from it belong with their source rather
than in `results/`, where they would read as current.

The scripts that write into this folder (`evaluation/eval_tapas.py`,
`evaluation/eval_tapas/eff_eps/common.py`) have been repointed here, so
re-running them lands in this folder rather than recreating the old top-level
paths.
