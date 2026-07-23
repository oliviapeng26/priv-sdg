#!/usr/bin/env python3
"""Pre-flight: measure the REAL per-fit time of a generator on a 500-row TAPAS
background, so we can project the audit budget before committing a full run.

Why this exists: convergence_check.py timed fits on the FULL 21,523 rows, but
TAPAS retrains each shadow generator on a 500-row background -- far faster. The
audit's first attack (Groundhog) generates ALL the shared shadow fits
((num_train+num_test)x2 of them) and only checkpoints them AFTER the entire Groundhog finishes,
so the first attack (Groundhog) must complete within one interactive session or it
restarts every time and never lands. Colab free-tier sessions cap at ~12 h, but the
binding limit is the idle timeout (~30-60 min of inactivity) -- so Groundhog's
duration is really "how long you must keep the tab open + machine awake". SESSION_BUDGET_MIN
below is that keep-alive window, not a hard quota.

The probe is purely about time/feasibility, not statistical adequacy.
It does 3 real fit+generate cycles on a 500-row background, takes the median s/fit, and multiplies out:
- Groundhog time = (num_train + num_test) × 2 × s/fit — the make-or-break first attack.
- FITS / OVER the keep-alive window (SESSION_BUDGET_MIN).
- The largest counts that would still fit the Google Colab T4 GPU time budget
  (in case fits are cheap enough to raise num_train/num_test above 10/20 and
  shrink the neural-vs-statistical CI asymmetry).
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/ (script now in a subfolder)
import common
from config import METHOD_CONFIG, FORMAL_EPSILON, NUM_SYNTHETIC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 3                     # fit+generate cycles to time (median reported)
SESSION_BUDGET_MIN = 120  # keep-alive window you'll babysit the 1st attack (not a hard cap;
                          # Colab sessions cap ~12h, idle timeout ~30-60 min of inactivity)
METHODS = sys.argv[1:] or ["ctgan", "dpgan"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("probe")


def build_member_dataset():
    """The 500-row world a shadow generator actually trains on: background+target."""
    train_dataset, _, description = common.load_adult_datasets()
    background_dataset, bidx = common.sample_background(train_dataset)
    target_record, _ = common.select_random_target(train_dataset, bidx)
    member = background_dataset.copy()
    member.add_records(target_record, in_place=True)
    return member, description


def probe(method, member, description):
    cfg = METHOD_CONFIG[method]
    plugin_kwargs = {**cfg["plugin_kwargs"], "device": DEVICE}
    log.info(f"[{method}] timing {K} fit+generate cycles on {len(member.data)} rows, "
             f"device={DEVICE}, plugin_kwargs={plugin_kwargs}")

    times = []
    for i in range(K):
        gen = common.SynthcityGenerator(method, description, FORMAL_EPSILON, plugin_kwargs)
        t0 = time.time()
        gen.fit(member)
        gen.generate(NUM_SYNTHETIC)
        dt = time.time() - t0
        times.append(dt)
        log.info(f"  [{method}] cycle {i+1}/{K}: {dt:.1f}s")

    s_per_fit = float(np.median(times))
    num_train, num_test = cfg["num_train"], cfg["num_test"]
    groundhog_fits = (num_train + num_test) * 2          # first attack does all shared fits
    groundhog_min = groundhog_fits * s_per_fit / 60.0
    fits_in_session = groundhog_min <= SESSION_BUDGET_MIN

    # Largest symmetric counts (num_train:num_test = 1:2) that keep the first
    # attack under ~75% of the session budget.
    max_pairs = int((0.75 * SESSION_BUDGET_MIN * 60) / (2 * s_per_fit)) if s_per_fit > 0 else 0
    max_train = max_pairs // 3
    max_test = max_pairs - max_train

    log.info(f"[{method}] median {s_per_fit:.1f}s/fit  ->  Groundhog "
             f"({num_train}/{num_test}) = {groundhog_fits} fits = {groundhog_min:.1f} min "
             f"[{'FITS' if fits_in_session else 'OVER'} the {SESSION_BUDGET_MIN}-min session]")
    if fits_in_session:
        log.info(f"[{method}] headroom: could raise counts to ~{max_train}/{max_test} "
                 f"and still fit ~75% of a session (tighter CIs, less asymmetry).")
    else:
        need = int((SESSION_BUDGET_MIN * 60) / (2 * s_per_fit) / 3)
        log.info(f"[{method}] to fit one session, drop counts to ~{need}/{need*2}, "
                 f"or lower n_iter / split across sessions (resumable per attack).")


def main():
    log.info(f"=== fit-time probe: methods={METHODS}, device={DEVICE}, K={K} ===")
    if DEVICE == "cpu":
        log.warning("Running on CPU -- timings won't match the Colab T4. Run this on the T4.")
    member, description = build_member_dataset()
    for method in METHODS:
        probe(method, member, description)


if __name__ == "__main__":
    main()
