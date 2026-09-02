# DP-CTGAN audit

SmartNoise DP-CTGAN under the same TAPAS threat model, 5-attack battery and seeding
as the four Synthcity methods. Two experiments live here:

| experiment | what varies | fixed | results |
|---|---|---|---|
| counts sweep | `num_train`/`num_test` = 50/100 → 200/500 → 500/1000 → 1000/2500 | eps=1.0, epoch_cap=300 | `results/dp-ctgan/eps1/{nt}_{nte}/` |
| epoch_cap sweep | `epoch_cap` = 500, 750, 1000 | eps=100, 1000/2500 | `results/dp-ctgan/epoch_sweep/eps100_ep{cap}/1000_2500/` |

| file | what |
|---|---|
| `dpctgan_generator.py` | `DPCTGANGenerator` — DP-CTGAN wrapped as a TAPAS `Generator` |
| `run_dpctgan_audit.py` | both sweeps, and `--probe N` for epochs-run / timing |

`dpctgan_generator.py` moved here from `scripts/eps_sweep/spike_diagnosis/` when the
two sweeps started sharing it — that script now imports it from this folder. One
copy, deliberately: the epoch_cap sweep is read against the spike diagnosis's
epoch_cap=300 arm, so both must be fitting the same generator.

## Run order

```bash
# 0. pre-flight — DP-CTGAN needs the Opacus 0.x API (see Environment below)
python benchmark_tapas/dp-ctgan/run_dpctgan_audit.py --probe 3

# 1. counts sweep at eps=1.0 — walk upward, each stage banks a result
run_stage 50 100 && run_stage 200 500 && run_stage 500 1000 && run_stage 1000 2500

# 2. epoch_cap sweep at eps=100 — probe each cap first, then commit
python benchmark_tapas/dp-ctgan/run_dpctgan_audit.py --probe 3 --epsilon 100 --epoch-cap 1000
for cap in 500 750 1000; do run_stage 1000 2500 --epsilon 100 --epoch-cap $cap || break; done
```

`run_stage` is the restart loop in the module docstring — use it rather than calling
the script directly, since it exits 3 while the pools are unfinished.

## Cost

**The counts sweep should need close to zero new fits.** Its pool cache is
`cache/dpctgan_eps1`, the same directory the spike diagnosis grew to 3500 fits at
1000/2500, and TAPAS only ever grows a pool. All four stages draw from it, so the
cost is four attack passes (~18 min each at 1000/2500, much less below that). Do not
delete that cache to "start clean" — it is the expensive part.

**The epoch_cap sweep cannot share anything.** A different cap is a different
generator, so each of 500/750/1000 fits its own 3500-fit pool. Probe first: the
per-fit cost scales with epochs actually run, which is what `--probe` reports.

## The thing to check before trusting the epoch sweep

`epoch_cap` is a **cap, not a schedule**. SmartNoise treats epsilon as a stopping
rule — fixed sigma=5, break the first time the accountant says the budget is spent —
so if the accountant stops training before the cap, raising the cap changes nothing
and the arm silently duplicates a lower one.

That is why the sweep runs at **eps=100** rather than 1.0: a loose budget is what
makes the cap the binding constraint. `--probe` compares `epochs_run` against the cap
and warns when the accountant won. Check that warning before committing ~12 h to an
arm, and record `last_epochs_run` from each `meta.json` next to the results.

Also worth knowing: `batch_size` is 500 and the audit background is 500 rows, so the
sampling rate is 1.0 and there is **no privacy amplification from subsampling**.

## Environment

`snsynth` 1.0.8's DP-CTGAN is written against the **Opacus 0.x** API
(`PrivacyEngine(model, batch_size=…).attach(optimizer)`). Under Opacus 1.4.1 — which
synthcity's DPGAN requires, and which the `priv-sdg` conda env carries — every fit
dies with:

```
TypeError: PrivacyEngine.__init__() got an unexpected keyword argument 'batch_size'
```

Run this where the eps sweep ran (the workstation `venv/`), not in that conda env.
`--probe` fails fast and says so.

## Two things to carry into the writeup

**Epsilon means something different here.** synthcity solves for the noise multiplier
so exactly eps is spent over `n_iter` epochs — eps in, sigma out. SmartNoise inverts
it: sigma is fixed and eps is a stopping rule, and `delta` is not passed at all but
derived internally as 1/(n·√n) = 8.9e-5 on a 500-row background. "eps = 1.0" labels
two different mechanisms with two different deltas. Taken as the library ships, on
purpose — the question is whether an independent implementation of DP-SGD-on-CTGAN
behaves the same way, not whether a re-parameterised synthcity does.

**The stopping rule overshoots.** The accountant is queried at the *top* of each
epoch, before that epoch's steps, so when the break fires those epochs have already
been trained at a cost strictly greater than the target. `last_epsilon_spent` in
`meta.json` is what the released generator actually spent; the requested eps is a
target it steps past.
