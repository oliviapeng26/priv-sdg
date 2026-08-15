# benchmark_tapas — 2×2 generator-grid privacy–utility benchmark

The main deliverable experiment: run the **full 5-attack TAPAS MIA battery**
against all four generators in the 2×2 grid and place each on a single
privacy–utility tradeoff diagram.

|                | **Non-DP** | **DP (ε = 1.0)** |
|----------------|------------|------------------|
| **Statistical** | BayesNet  | PrivBayes        |
| **Neural**      | CTGAN     | DPGAN            |

## Research question

How do statistical vs neural synthetic-data generators compare on **empirical
privacy leakage** (adversarial MIA) and **data utility**, with and without a DP
guarantee? Non-DP methods are expected to sit at higher utility + higher
leakage; DP methods trade utility for privacy.

## Threat model (fixed across all four methods)

- **Exact-knowledge** data prior: a fixed background of `BACKGROUND_SIZE = 499`
  records, sampled once (seed 42) and reused for every method.
- **Black-box** generator knowledge.
- **Membership inference** only. Member world `D+ = background + target(t)`,
  non-member world `D- = background + alternate(t')`, both size 500,
  deterministic on every draw (custom `SwapMIALabeller` / `SwapTargetedMIA`,
  inherited from `target_strategy/`). Target/alternate chosen randomly (seed 43)
  — the *effect* of target selection is a separate study in `target_strategy/`.

The background, target, and alternate are **identical across all four
generators**, so any difference in the results is the generator's, not the draw's.

## The 5-attack battery (`common.build_attacks`)

1. `GroundhogAttack` — shadow-modelling on histogram/naive set features
2. `ShadowModelling(RandomQueries)` — shadow-modelling on random targeted queries
3. `LocalNeighbourhoodAttack` — L2 distance, radius = 5th-NN of the target
4. `ProbabilityEstimationAttack` — KernelDensity
5. `ClosestDistanceMIA` — L2 closest-distance

Per attack we record AUC, MIA advantage, privacy gain, TP/FP, and the
effective-epsilon report (Clopper–Pearson CI at 90/95/99%). The **worst-case
(max) AUC** across the five is the headline privacy number; the **worst-case
eff-epsilon** (max `eps_low_95`) is the headline leakage lower bound for the DP
methods.

## Layout

```
benchmark_tapas/
├── config.py                 # methods, DP flags, per-method counts + plugin_kwargs, paths
├── common.py                 # adapted from target_strategy/common.py (5-attack version)
├── scripts/                  # the audit entry points — one per generator
│   ├── run_bn.py             # BayesNet  (non-DP statistical)  — CPU
│   ├── run_privbayes.py      # PrivBayes (DP statistical, ε=1) — CPU
│   ├── run_ctgan.py          # CTGAN     (non-DP neural)        — CPU or Colab T4
│   ├── run_dpgan.py          # DPGAN     (DP neural, ε=1)       — CPU or Colab T4
│   └── run_cdtgan_colab.ipynb  # Colab GPU runner (runs both CTGAN + DPGAN)
├── neural_tuning/            # pick the neural (CTGAN/DPGAN) run settings BEFORE the audit
│   ├── convergence_check.py       # n_iter sweep vs TSTR utility (not through TAPAS)
│   ├── convergence_check_colab.ipynb  # its T4 runner
│   └── probe_fit_time.py          # measured 500-row per-fit time → counts + compute budget
├── analysis/                 # aggregate the results + build the figures
│   ├── aggregate.py          # privacy + (existing) utility/fidelity → summary_*.csv
│   ├── tradeoff.py           # key scatter (X=utility, Y=mean advantage) → privacy_utility_tradeoff.png
│   └── attack_heatmap.py     # AUC heatmap (attack × method) → attack_auc_heatmap.png
├── cache/                    # gitignored: threat_model.pkl + per-attack result_*.json
└── results/
    ├── per_method/           # one dir per generator: effeps_{m}.csv, effective_epsilon_*.csv, ROC PNG, log
    ├── tables/               # summary_{privacy,utility,combined}.csv + benchmark_privacy_per_attack.csv
    ├── figures/              # privacy_utility_tradeoff.png, attack_auc_heatmap.png
    └── convergence/          # convergence_check_tstr.csv + _LEGACY
```

`common.py` is a near-verbatim copy of `target_strategy/common.py` with three
changes only: (1) `epsilon` is passed to the plugin **only for DP methods**
(non-DP plugins reject the kwarg); (2) a `plugin_kwargs` passthrough carries the
GANs' `n_iter` cap and `device`; (3) `method`/`epsilon` are per-call parameters
rather than a hardcoded generator. Everything else — the swap labeller, threat-
model caching, per-attack JSON caching, ROC plotting — is unchanged.

## Compute plan (all four methods use the same 50/100)

Generator **fits ≈ (num_train + num_test) × 2** (each labelled pair retrains
`D+` and `D-`; shared across the 5 attacks via the threat model's memoisation).

| Method | counts (train/test) | where | Groundhog (first attack) | notes |
|--------|--------------------|-------|--------------------------|-------|
| BayesNet  | 50 / 100 | CPU | fast | ~1 s/fit |
| PrivBayes | 50 / 100 | CPU | ~92 min | ~28.6 s/fit |
| CTGAN     | 50 / 100 | Colab T4 | ~34 min | `n_iter=50`, ~6.8 s/fit on 500 rows |
| DPGAN     | 50 / 100 | Colab T4 | ~92 min | `n_iter=100`, ~18.3 s/fit; **keep tab active** |

The neural `n_iter` are set from `convergence_check.py` (Step 1): CTGAN's utility
is flat, so 50 already matches the converged model; DPGAN needs 100 (50
undertrains, >100 doesn't help under fixed ε=1). The counts were originally going
to be a reduced 10/20 for the neural methods, but `probe_fit_time.py` (Step 3)
measured the real 500-row per-fit time (~6.8 s CTGAN, ~18.3 s DPGAN) — fast enough
that **50/100 runs comfortably**, so all four methods share the same counts and the
count-asymmetry limitation is gone. Colab free-tier sessions run up to ~12 h (both
neural audits fit in one sitting); the binding limit is the **idle timeout
(~30–60 min of inactivity)**. The whole ~92-min DPGAN Groundhog must complete
before it checkpoints, so keep the browser tab open and the machine awake until
`[Groundhog] done` — don't walk away mid-run.

## How to run

From repo root, `conda activate priv-sdg` first.

**Statistical (CPU):**
```bash
python benchmark_tapas/scripts/run_bn.py          # ~4 min
python benchmark_tapas/scripts/run_privbayes.py   # ~2.4 hr
```

**Neural (Colab T4 — or local CPU):** the neural audits actually run *faster* on
a laptop CPU here (the 500-row TAPAS fits are tiny, ~5 s CTGAN / ~45 s DPGAN), so
they can just be run like the statistical ones: `python
benchmark_tapas/scripts/run_ctgan.py` then `run_dpgan.py`. The Colab path below is
kept for when the GPU is preferable.
1. **Step 1 — convergence/timing.** Upload `benchmark_tapas/{config.py,
   neural_tuning/convergence_check.py}` and `data/adult_{train,test}.csv` to
   `MyDrive/VRI/experimentation/`, then run `neural_tuning/convergence_check_colab.ipynb`.
   (Don't run this locally — it needs GPU timings, and a full-data GAN sweep on
   CPU takes hours.)
2. **Step 2 — decide (local edit).** From Step 1's output, set `CTGAN_N_ITER` /
   `DPGAN_N_ITER` in `config.py` (smallest `n_iter` where TSTR has plateaued —
   currently 50 / 100).
3. **Step 3 — the audits.** Upload `benchmark_tapas/{config.py, common.py,
   neural_tuning/probe_fit_time.py, scripts/run_*.py}` (+ data) to Drive, then run
   `scripts/run_cdtgan_colab.ipynb` (one notebook runs both CTGAN + DPGAN). It runs
   the `probe_fit_time.py` timing check (cell 3b) right after mount — read its
   FITS/OVER verdict before running the audit cells, and only adjust the neural
   `num_train`/`num_test` in `config.py` (re-upload) if it says OVER or you want
   tighter CIs.

All runs are **resumable**: each attack's result is cached to
`cache/<method>/result_<attack>.json` as it finishes, and the memoised threat
model to `threat_model.pkl` — re-running skips finished work. On Colab both live
on Drive, so a disconnected session resumes rather than restarts.

**Aggregate + plot** (after the four runs populate
`results/benchmark_privacy_per_attack.csv`). Run in this order — `tradeoff.py`
reads `summary_combined.csv` produced by `aggregate.py`:
```bash
python benchmark_tapas/analysis/aggregate.py       # → summary_privacy.csv, summary_utility.csv, summary_combined.csv
python benchmark_tapas/analysis/tradeoff.py        # → results/privacy_utility_tradeoff.png  (X=utility, Y=mean advantage)
python benchmark_tapas/analysis/attack_heatmap.py  # → results/attack_auc_heatmap.png        (AUC by attack × method)
```

Utility/fidelity are **not** recomputed here — `aggregate.py` reads
`results/utility_summary.csv` (in-house TSTR/TRTR/retention, from
`evaluation/eval_utility.py`) and `results/fidelity_summary.csv` (from
`evaluation/eval_fidelity.py`). Both are means ± std over `seeds.RUN_SEEDS`, so
the tradeoff scatter carries x error bars. Missing inputs degrade to NaN columns
rather than failing, so the privacy half can be aggregated on its own.

These replaced `results/synthcity_results.csv`, which was deleted: its TSTR was
scored on an internal split of the generators' own training data.

**On the privacy axis:** worst-case AUC saturates at 1.0 for all four generators
(Groundhog/ShadowModelling separate every generator, DP or not — ε-invariant), so
the tradeoff scatter uses **mean AUC across the 5 attacks** (breadth of
vulnerability) as its Y-axis, and `summary_privacy.csv` carries the worst-case
AUC, εeff (`eps_low_95`), formal ε, and the εeff−ε gap as a companion table.

## Tradeoff diagram

- **X** = `performance.xgb.syn_id` (TSTR in-distribution XGBoost; ↑ = more useful
  as a drop-in replacement for real data). Dashed vertical = real-data baseline
  `performance.xgb.gt`.
- **Y** = **mean membership advantage** (`tp − fp`) across the 5 attacks (↑ = more
  leakage / broader attack surface). Dashed horizontal at 0 = attacks no better
  than random. We use mean advantage rather than worst-case AUC (which saturates
  at 1.0 for all four methods) — see **Privacy metric** below for the full
  reasoning. **eff-epsilon** and worst-case AUC go in the companion table
  `summary_privacy.csv`, not on the axis.
- Marker **shape** = family (circle statistical, square neural); **fill** = DP
  status (filled DP, hollow non-DP).

## Privacy metric — what `AUC`/`tp`/`fp` mean, and how we chose the axis

For each (attack, generator) pair TAPAS reports `auc` (threshold-free, from the
score ranking) and `tp`/`fp` (true-/false-positive **rates** at TAPAS's chosen
threshold). `advantage = tp − fp` is the membership advantage.

### The per-attack outcomes (`auc`, `tp`, `fp`)

Reading one attack against one SDG method:

- **`auc = 1`** — **perfect separation** (tells members from non-members
  perfectly): the SDG method has **bad privacy** against this attack.
  `tp = 1`, `fp = 0`, so `advantage = 1`.
- **`auc = 0`, `tp = 0`, `fp = 1`** — **perfect separation the other way**
  (inverted): **the same as `auc = 1`**, because an adversary just flips the
  decision rule. `advantage = −1`. **NONE of our (attack, method) pairs have
  this** (verified: no pair has `tp = 0, fp = 1`).
- **`auc = 0`, `tp = 0`, `fp = 0`** — the **attack failed** (infinity/threshold
  issue: the threshold collapsed so the attack labels *everything* "non-member";
  the leftover `auc = 0` is incidental noise, not signal). **Every `auc = 0` in
  our results is this `tp = fp = 0` case** — so `auc = 0` always means "attack
  failed" for us, never the inverted case above. `advantage = 0`.
- **`auc = 0.5`** — the attack is **no better than random guessing**: the SDG
  method has **good privacy** against this attack. `tp = fp`, so `advantage = 0`.

So an attack only shows leakage when it *genuinely separates* (`auc = 1`). A
random (`auc = 0.5`) or failed (`auc = 0, tp = fp = 0`) attack means the method
**resisted** it → it must count as **zero** leakage. Our data only ever contains
**perfect separation, random guessing, or failed attacks** — no partial ones — so
**`advantage` is binary (0 or 1).**

### How we chose the tradeoff Y-axis (three steps)

1. **Worst-case (max) AUC — doesn't work.** It's **1.0 for all four generators**
   (Groundhog/ShadowModelling perfectly separate every generator, DP or not,
   ε-invariant). All four points collapse onto 1.0 → the axis can't differentiate
   the methods. (Real finding, reported in the table — useless as an axis.)

2. **Average AUC over attacks with raw `auc > 0.5`.** A *plain* mean of all 5 raw
   AUCs is wrong because a **random** attack (`auc = 0.5`, i.e. good privacy) would
   contribute **0.5** of leakage instead of 0. That flaw shows up as plain mean AUC
   = **0.40 / 0.50 / 0.70 / 0.70**, where PrivBayes (0.50) is wrongly placed above
   BN (0.40) only because PrivBayes's *failed* LocalNeighbourhood scored `auc = 0.5`
   while BN's scored `auc = 0.0` — both non-leaking. **Fix:** average only the
   attacks with `auc > 0.5`, over the fixed denominator of 5:

   | method | attacks with AUC > 0.5 | sum ÷ 5 |
   |---|---|---|
   | BN | Groundhog(1), ShadowModelling(1) | (1+1)/5 = **0.40** |
   | PrivBayes | Groundhog(1), ShadowModelling(1) | (1+1)/5 = **0.40** |
   | CTGAN | Groundhog(1), ShadowModelling(1), ProbabilityEstimation(1) | (1+1+1)/5 = **0.60** |
   | DPGAN | Groundhog(1), ShadowModelling(1), ProbabilityEstimation(1) | (1+1+1)/5 = **0.60** |

   Now BN = PrivBayes = 0.40 (correct — same 2 attacks succeed), neural = 0.60.

3. **Mean membership advantage (`tp − fp`) — what we report.** Advantage anchors
   "no leakage" at 0 (random attack: `tp = fp`) and "perfect separation" at 1
   (`tp = 1, fp = 0`), so it counts only genuine success, and it is a **standard,
   named** privacy metric. Mean advantage = **0.40 / 0.40 / 0.60 / 0.60**.

### Why "average AUC > 0.5" and "mean advantage" are the same here

Different formulas (one reads `auc`, one reads `tp − fp`), **same numbers**,
because every attack falls into one of the three cases above and both formulas
give it the same per-attack contribution:

| attack outcome | AUC | advantage | "AUC > 0.5" contributes | advantage contributes |
|---|---|---|---|---|
| perfect separation | 1.0 | 1.0 | 1.0 | 1.0 |
| random guessing | 0.5 | 0.0 | 0 (excluded, ≤ 0.5) | 0.0 |
| failed attack | 0.0 (`tp=fp=0`) | 0.0 | 0 (excluded, ≤ 0.5) | 0.0 |

Row by row the last two columns match (working → 1, random/failed → 0), so the
means match. This holds **only** because our data is binary (`advantage ∈ {0,1}`).
They would differ on a *partial* attack — e.g. `auc = 0.7`, `tp = 0.6`, `fp = 0.2`
→ "AUC > 0.5" counts 0.7 but advantage counts `tp − fp = 0.4` — which we don't
have. And AUC's threshold-freeness (its one edge over `tp`/`fp`) is **moot here**:
it only matters for partial attacks, and ours are all-or-nothing, so fixing one
threshold loses nothing.

**Net:** tradeoff Y-axis = **mean membership advantage** = 0.40 / 0.40 / 0.60 /
0.60, equivalently "# of 5 attacks that succeed" = 2 / 2 / 3 / 3.

## Limitations / caveats (carry into the report)

1. **Confidence intervals (symmetric counts — asymmetry resolved).** All four
   methods use 50/100, so their CIs are directly comparable; the earlier plan to
   run the neural methods at a reduced 10/20 was dropped once `probe_fit_time.py`
   showed 500-row fits are fast enough (~6.8 s CTGAN, ~18.3 s DPGAN) to afford
   50/100 in one T4 session. Note the CIs are still **wide in absolute terms** at
   num_test=100 (a perfectly-separating attack's eff-epsilon 95% lower bound tops
   out at ~2.21 — a sample-size ceiling, not a per-generator value), and the
   worst-case-over-5-attacks AUC is **biased upward** (max of noisy estimates).
   Report the CIs, not just point estimates.
2. **n_iter-deflation risk.** Capping the GANs' `n_iter` too aggressively
   undertrains them: a generator that learned little has little to leak, so a
   *low* MIA AUC would reflect a **weak generator, not genuine privacy**. Step 1
   picks the smallest `n_iter` where TSTR utility has plateaued to guard against
   this; **flag explicitly in the report if the chosen `n_iter` sits below the
   plateau** (i.e. if the time budget forced undertraining).
3. **500-row vs full-data convergence.** `n_iter` is tuned for quality on the full
   training set (TSTR needs the full set to be stable), but the TAPAS shadow
   models train on 500-row backgrounds and may converge/overfit differently —
   a methodological wrinkle to note.
4. **Utility vs privacy generation regimes differ.** Utility/fidelity are measured
   on the full-data synthetic sets (`synthetic_data/`), whereas privacy is
   measured on 500-row-background retrains inside TAPAS. This is inherent to the
   TAPAS methodology; the tradeoff plot pairs them per generator regardless.
