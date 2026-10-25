#!/usr/bin/env python3
"""Synthetic checks for estimate_eff_eps.py, on cases whose answer is known.

  separated   every member scores above every non-member -> each bounded method sits
              at its sample-size cap (joint-beta: finite and large; its numerical
              integration is loose at the boundary), with zero false positives
  identical   members and non-members have the same scores -> Clopper-Pearson bounds are 0;
              joint-beta is a credible bound and the eps=0 region has no mass, so ~0.05
  overlapping members and non-members from the same N(0,1) -> bounds near 0
  flipped     members score LOWER. TAPAS's "inverse" branch and jax's one_sided=False both
              add log(TNR/FNR) for the SAME rule (score > tau = member), not the reverse
              rule, so both need higher = member: flipped scores must give ~0, and negating
              them back must give the original bound exactly
  zero_fp     a third of members above every non-member, the rest overlapping -> zero FP
              at the chosen threshold, finite lower bound, below the cap

Run from the repo root (a few minutes, mostly joint-beta):
  alt_eff_eps/.venv/bin/python alt_eff_eps/test_estimate_eff_eps.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_eff_eps import estimate_cell                        # noqa: E402

N = 2500                                       # the 1000/2500 test count
rng = np.random.default_rng(1)
labels = np.arange(N) % 2 == 0                 # alternating, as TAPAS writes them
BOUNDED = ["tapas_replica", "jax_cp_split", "jax_cp_split_tapas_tails", "jax_cp_bonferroni",
           "jax_cp_multisplit", "pe_joint_beta", "pe_clopper_pearson", "pe_jeffreys"]
failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def cells(scores, **kw):
    return pd.DataFrame(estimate_cell(scores, labels, **kw))


print("separated")
sep = np.where(labels, 1 + rng.uniform(0, 0.5, N), rng.uniform(0, 0.5, N))
df = cells(sep, gdp=True)
b = df[df.method.isin(BOUNDED + ["jax_gdp", "jax_raw_counts"]) & (df.method != "pe_joint_beta")]
check("all bounded methods at cap", b.capped.all(),
      b[~b.capped][["method", "confidence", "eps_lower", "cap"]].to_string() if not b.capped.all() else "")
jb = df[df.method == "pe_joint_beta"]
check("joint-beta finite and large (> 5)", (jb.eps_lower > 5).all(), str(jb.eps_lower.round(3).tolist()))
check("zero_fp flagged", df[df.method == "tapas_replica"].zero_fp.all())
check("upper bound inf", np.isinf(df[df.method == "pe_clopper_pearson"].eps_upper).all())
print(df[df.confidence == 0.95].groupby("method")[["eps_lower", "cap"]].first().round(3).to_string())

print("identical")
same = np.repeat(rng.normal(size=N // 2), 2)   # member i and non-member i share a score
df = cells(same, gdp=True)
b = df[df.method.isin(BOUNDED + ["jax_gdp"]) & (df.method != "pe_joint_beta")]
check("every CP-type bound is 0", (b.eps_lower.abs() < 0.02).all(),
      b[b.eps_lower.abs() >= 0.02][["method", "confidence", "eps_lower"]].to_string())
jb = df[df.method == "pe_joint_beta"]
check("joint-beta near 0 (< 0.1)", (jb.eps_lower < 0.1).all(), str(jb.eps_lower.round(3).tolist()))

print("overlapping")
df = cells(rng.normal(size=N))
b = df[df.method.isin(BOUNDED)]
check("bounds near 0 (< 0.15)", (b.eps_lower < 0.15).all(),
      f"max {b.eps_lower.max():.3f} ({b.loc[b.eps_lower.idxmax(), 'method']})")
raw = df[df.method == "jax_raw_counts"].eps_lower
check("raw counts small (< 0.5)", (raw < 0.5).all(), f"{raw.iloc[0]:.3f}")

print("flipped")
shifted = rng.normal(size=N) + 1.0 * labels
a = cells(shifted)
f = cells(-shifted)
back = cells(-(-shifted))
idx = ["method", "confidence", "delta"]
a_, f_, b_ = (d.set_index(idx).eps_lower for d in (a, f, back))
check("auc flags the flip (< 0.5)", f.auc.iloc[0] < 0.5 < a.auc.iloc[0],
      f"auc {a.auc.iloc[0]:.3f} -> {f.auc.iloc[0]:.3f}")
check("signal found unflipped (tapas_replica > 0.5)", a_["tapas_replica"].min() > 0.5)
for m in ["tapas_replica", "jax_cp_split", "jax_cp_bonferroni", "jax_cp_multisplit"]:
    check(f"{m} ~0 on flipped scores", f_[m].max() < 0.15, f"max {f_[m].max():.3f}")
    d = (a_[m] - b_[m]).abs().max()
    check(f"{m} restored by negating back", d < 1e-9, f"max |diff| {d:.2e}")

print("zero_fp")
zf = np.where(labels, np.where(rng.uniform(size=N) < 1 / 3, 10 + rng.uniform(size=N),
                               rng.normal(size=N)), rng.normal(size=N))
df = cells(zf)
r = df[(df.method == "tapas_replica")]
check("zero_fp at chosen threshold", r.zero_fp.all(), f"FP={r.FP.tolist()}")
check("finite lower bound below cap", (np.isfinite(r.eps_lower) & ~r.capped).all(),
      f"eps {r.eps_lower.round(3).tolist()} cap {r.cap.round(3).tolist()}")
jb = df[df.method == "pe_joint_beta"]
check("joint-beta handled (eps_lo finite, eps_hi inf)",
      (np.isfinite(jb.eps_lower) & np.isinf(jb.eps_upper)).all(), jb.note.iloc[0])

print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
sys.exit(1 if failures else 0)
