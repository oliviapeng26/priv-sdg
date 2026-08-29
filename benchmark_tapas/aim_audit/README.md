# AIM audit

Adds AIM as a fifth method: generation, fidelity, utility, and a TAPAS MIA
privacy audit at 1000/2500. Self-contained — nothing outside this folder is
modified, and `config.METHOD_CONFIG` keeps the four-method shape every committed
result was produced under.

AIM is taken exactly as `sdg/aim.py` already runs it. `epsilon`, `delta`,
`degree`, `max_cells`, `max_model_size`, `rounds` and `BIN_EDGES` are imported
from that module, never restated, so the model audited for privacy is the same
configuration whose utility and fidelity are reported. There is no tuning knob
here on purpose.

| file | what |
|---|---|
| `aim_generator.py` | `AIMGenerator` — SmartNoise AIM wrapped as a TAPAS `Generator` |
| `run_aim_audit.py` | the audit, and `--probe N` for timing |

## Run order

Steps 1–3 need no code from this folder — AIM is already wired into the repo's
generation and evaluation scripts. Run them from the repo root with the env
active.

```bash
# 0. pre-flight: AIM has never run on the workstation
python -c "import snsynth.aim, mbi, jax; print(jax.default_backend())"

# 1. generation, 5 seeds on the full 21,523-row split          (~20 min)
python sdg/generate_runs.py aim

# 2. fidelity + utility, CPU only                              (~10 min)
python evaluation/eval_fidelity.py aim
python evaluation/eval_utility.py aim

# 3. confirm the per-fit cost on this machine                  (~2 min)
python benchmark_tapas/aim_audit/run_aim_audit.py --probe 6

# 4. privacy, overnight in a screen                            (~12 h)
python benchmark_tapas/aim_audit/run_aim_audit.py
```

Steps 1–2 fold AIM into `results/{utility,fidelity}_summary.csv` automatically —
`"aim"` is already in both scripts' `ALL_METHODS`. So the utility half is banked
before the privacy run starts and does not depend on it.

## Counts

1000/2500, matching every other method's headline privacy number. Measured cost
is 12.1 s/fit on a 500-row background (`--probe 6`, plus a 10/20 end-to-end run),
so 3500 fits is ~12 h.

An earlier plan said 200/500 on a projected 90 s/fit — that projection assumed
AIM's Private-PGM rounds dominate and the row count barely matters. Measurement
says otherwise: full-data fits are 82–153 s, the 500-row background is ~10x
cheaper.

Lower stages resolve less. In the counts sweep all four methods returned
`eps_low_95 = 0.000` at 50/100 with the max-AUC ordering inverted; at 200/500
only DPGAN separates; by 1000/2500 both GANs do.

Optional risk-managed sequence, near-free because TAPAS's memoisation only grows
the pools:

```bash
python benchmark_tapas/aim_audit/run_aim_audit.py --num-train 200 --num-test 500  # ~2.4 h
python benchmark_tapas/aim_audit/run_aim_audit.py                                 # extends it
```

The second run generates only the 2800 missing fits, and `run_attack`
self-invalidates its per-attack cache when the counts increase. Total fits stay
3500; the extra cost is one ~5 min attack pass.

## Two things to carry into the writeup

**AIM is not bit-reproducible.** Its Gaussian measurements come from opendp's
CSPRNG, which has no seeding hook. Every fit is an independent draw by
construction — the pre-2026-08-23 collapse bug cannot happen here, and
`AIMGenerator` needs none of `SynthcityGenerator`'s per-fit seeding. The cost is
that re-running the audit draws different simulations.

**δ is not matched across the DP methods at the audit.** AIM runs at δ=1e-9
regardless of dataset size; synthcity's DPGAN derives δ = 1/len(X) = 2e-3 on a
500-row background; PrivBayes is pure DP. This is already true of the committed
counts sweep — report it, don't pin it.

## The one fragile part

`load_adult_datasets` min-max scales continuous columns to [0,1], but
`BIN_EDGES` are in original units. Each fit runs unscale → encode → AIM → decode
→ rescale. Skip the unscale and `encode` puts every row in bin 0, producing a
constant table — verified: encoding scaled data uses 1–2 bins per column against
19/16/9/6/20 unscaled. The distinctness guard catches this before any attack
time is spent, and its failure message points here.
