#!/usr/bin/env python3
"""[4] Who are the target and the alternate, and is either of them unusual?

THE QUESTION
    Every DPGAN privacy number in the repo is measured against ONE target/alternate
    pair, fixed by TAPAS_TARGET_SEED. Before asking why the audit separates D+ from
    D-, it is worth knowing what actually differs between those two worlds.

    One hypothesis dies here if the answer is boring. Synthcity's TabularEncoder is
    fitted on the raw 500 rows (tabular_gan.py:184) and the categorical levels it
    finds determine the one-hot LAYOUT. If the target carried a category absent from
    the 499-record background and the alternate did not, D+ and D- would be encoded
    into different-width matrices -- a structural difference that no amount of
    DP-SGD noise on the discriminator could hide, because it is upstream of the
    discriminator entirely. That would be a clean, mechanical explanation.

    It requires a category with a background count of 0. So: print the counts.

NO NEW FITS, NO GPU
    Reads the same records the audit uses, via the same functions: common.sample_
    background under BACKGROUND_SEED and common.select_random_target under
    TAPAS_TARGET_SEED. Nothing is generated. Values are printed in ORIGINAL units
    (age in years) by locating the records in data/adult_train.csv -- the TAPAS
    representation min-max scales continuous columns to [0,1], which is unreadable.

RESULT WHEN THIS WAS RUN (2026-09-19)
    target = train row 12435: 36y, edu_num 6, Private, Married-civ-spouse,
             Craft-repair, Husband, White, Male, United-States, <=50K
    alternate = train row 1204: 47y, edu_num 9, Local-gov, Separated,
             Adm-clerical, Unmarried, Black, Female, United-States, <=50K

    No category of either record is absent from the background -- the rarest is
    Craft-repair at 63/499 for the target and Separated at 17/499 for the alternate.
    The encoder-layout hypothesis is RULED OUT for this pair. Eight of thirteen
    columns differ; the target's education_num (6) sits at the 5th percentile of the
    background, and the alternate is the more unusual record overall.

    Read together with [5]: the columns that carry membership signal at eps=1.0 are
    exactly the two numeric columns that DIFFER here (education_num, age), and the
    three that are identical in both records carry none. That is a prediction this
    script's output made in advance and [5] confirmed -- with the caveat that
    capital_gain/capital_loss are near-constant zero for most records anyway, so
    only a different pair (see [6]) can separate "follows this pair's values" from
    "those columns are uninformative for anyone".

OUTPUT
    results/extras/dpgan_spike_diagnosis/target_profile.csv

Run from the repo root, env active (CPU, seconds):
  python benchmark_tapas/diagnostics/dpgan_target_profile.py
  python benchmark_tapas/diagnostics/dpgan_target_profile.py --target-seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = next(p for p in Path(__file__).resolve().parents
                     if (p / "config.py").exists())
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(REPO_ROOT))

import common                                                     # noqa: E402
from config import (RESULTS_DIR, TRAIN_CSV,                       # noqa: E402
                    CONTINUOUS_COLS, CATEGORICAL_COLS)
from seeds import TAPAS_TARGET_SEED                               # noqa: E402

OUT_DIR = RESULTS_DIR / "extras" / "dpgan_spike_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def locate(train_dataset, record, background_idx) -> int:
    """Row position of a record in the training frame.

    select_random_target draws from the records NOT in the background, so any match
    inside the background is a coincidental duplicate row and is discarded.
    """
    match = (train_dataset.data == record.data.iloc[0]).all(axis=1).values
    hits = [int(h) for h in np.flatnonzero(match) if h not in background_idx]
    if not hits:
        raise RuntimeError("record not found outside the background -- has "
                           "data/adult_train.csv changed since the audit ran?")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-seed", type=int, default=TAPAS_TARGET_SEED,
                    help=f"seed picking the pair (default: {TAPAS_TARGET_SEED}, the "
                         f"audited pair). Other values profile a placebo pair, see [6]")
    args = ap.parse_args()

    raw = pd.read_csv(TRAIN_CSV)                    # original units
    train_dataset, _, _ = common.load_adult_datasets()   # scaled, same row order
    background, background_idx = common.sample_background(train_dataset)
    target, alternate = common.select_random_target(train_dataset, background_idx,
                                                    seed=args.target_seed)

    t_i, a_i = (locate(train_dataset, r, background_idx) for r in (target, alternate))
    t, a = raw.iloc[t_i], raw.iloc[a_i]
    bg = raw.iloc[sorted(background_idx)]

    rows = []
    for c in CONTINUOUS_COLS:
        rows.append({"column": c, "type": "numeric", "target": t[c], "alternate": a[c],
                     "target_rarity": f"{(bg[c] < t[c]).mean():.0%} of background below",
                     "alternate_rarity": f"{(bg[c] < a[c]).mean():.0%} of background below"})
    for c in CATEGORICAL_COLS:
        rows.append({"column": c, "type": "categorical", "target": t[c], "alternate": a[c],
                     "target_rarity": f"{(bg[c] == t[c]).sum()} of {len(bg)}",
                     "alternate_rarity": f"{(bg[c] == a[c]).sum()} of {len(bg)}"})

    out = pd.DataFrame(rows)
    out.insert(2, "differs", out.target.astype(str) != out.alternate.astype(str))
    out.insert(0, "target_seed", args.target_seed)
    out.insert(1, "target_row", t_i)
    out.insert(2, "alternate_row", a_i)

    tag = "" if args.target_seed == TAPAS_TARGET_SEED else f"_target{args.target_seed}"
    out_csv = OUT_DIR / f"target_profile{tag}.csv"
    out.to_csv(out_csv, index=False)

    print(f"\ntarget = train row {t_i}, alternate = train row {a_i}, "
          f"background = {len(bg)} rows (target_seed={args.target_seed})\n")
    print(out.drop(columns=["target_seed", "target_row", "alternate_row"])
             .to_string(index=False))
    print("\nRead the rarity columns as:")
    print("  categorical: '0 of 499' means the category NEVER occurs in the background,")
    print("               which would change the encoder's one-hot layout between D+ and D-")
    print("  numeric:     '99% of background below' means the record is at an extreme")
    print(f"\nwrote {out_csv.relative_to(BENCHMARK_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
