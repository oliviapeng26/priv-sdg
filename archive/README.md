# archive

Superseded or out-of-paper material, kept for reference. Not part of the pipeline; scripts here are
not runnable from this location without moving them back (their `sys.path` set-up assumes their old
folder).

| Folder | What | Why archived |
|---|---|---|
| `baseline_50_100/` | The four-generator 50/100 TAPAS baseline: `scripts/run_{bn,privbayes,ctgan,dpgan}.py`, `scripts/extract_scores.py`, `per_method/`, `scores/` | Post-seeding-fix and valid, but the sample-size sweep (`benchmark_tapas/results/sample_size_sweep/*/50_100`) covers the same stage and is the one the paper uses. |
| `dp_ctgan_sep2_attack_rerun/` | The first DP-CTGAN ε sweep script and its ε = 1, ε = 100 results | The same synthetic pools were later re-attacked by `audits/run_dpctgan_audit.py`. The two model-training attacks (Groundhog, ShadowModelling) differ between the runs, so this is a replicate of the attack step. ε = 0.1 and 10 exist only from the first run and live in `eps_sweep/dp_ctgan/`. |
| `target_strategy/` | Effect of target-selection strategy (random / outlier / artificial) on ε_eff, PrivBayes ε = 10 | Not in the paper. |
| `aim_tuning/` | AIM bin-count sweep with Synthcity's AIM, and the b = 32 fidelity comparison | Not in the paper; its utility columns use the leaky TSTR. |
| `evaluation_analysis_pruned_cells.ipynb` | Cells removed from `evaluation/analysis.ipynb` | Iteration-1/2 TAPAS tables, pilot counts and AIM tuning cells; not runnable. |
