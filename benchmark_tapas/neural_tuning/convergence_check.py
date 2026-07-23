#!/usr/bin/env python3
"""Standalone n_iter convergence + timing check for CTGAN / DPGAN (Step 1).

NOT through TAPAS. Trains each GAN once on the FULL training set at several
n_iter caps, and records (a) wall-clock fit+generate time and (b) TSTR utility
(performance.xgb.syn_id -- the same metric the tradeoff plot uses). Two goals:

  1. Pick the smallest n_iter where TSTR utility has plateaued (so the capped
     generator still "learned enough" -- an undertrained GAN would deflate MIA
     AUC artificially, making it look private when it just leaked nothing; see
     README caveat).
  2. Measure seconds-per-fit so we can project the TAPAS shadow-model budget:
     fits = (num_train + num_test) x 2, and decide whether the neural counts
     fit inside one ~2 hr Colab T4 session.

Meant to run on the Colab T4 (device auto-detected). Writes incrementally to
results/convergence_check.csv and is resumable -- an already-recorded
(method, n_iter) row is skipped, so a disconnected session resumes.

Run:
  python benchmark_tapas/convergence_check.py                  # ctgan + dpgan, all n_iter
  python benchmark_tapas/convergence_check.py ctgan            # one method
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/ (script now in a subfolder)
from config import (
    TRAIN_CSV, CONTINUOUS_COLS, CATEGORICAL_COLS, TARGET_COL,
    CONVERGENCE_DIR, FORMAL_EPSILON, DP_METHODS, METHOD_CONFIG,
)

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ITER_CANDIDATES = [50, 100, 200, "default"]   # "default" -> plugin default (2000 max + early stop)
METHODS = sys.argv[1:] or ["ctgan", "dpgan"]
OUT_CSV = CONVERGENCE_DIR / "convergence_check.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(CONVERGENCE_DIR / "convergence_check_log.txt"
                                                  if CONVERGENCE_DIR.exists() else "convergence_check_log.txt"),
                              logging.StreamHandler()])
log = logging.getLogger("convergence")


def load_train_df():
    df = pd.read_csv(TRAIN_CSV)
    df[CONTINUOUS_COLS] = df[CONTINUOUS_COLS].astype(float)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")
    return df


def already_done(method, n_iter):
    if not OUT_CSV.exists():
        return False
    done = pd.read_csv(OUT_CSV)
    key = str(n_iter)
    return ((done["method"] == method) & (done["n_iter"].astype(str) == key)).any()


def append_row(row: dict):
    CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if OUT_CSV.exists():
        df = pd.concat([pd.read_csv(OUT_CSV), df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)


def run_one(method, n_iter, train_df, real_loader):
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    from synthcity.metrics.eval import Metrics

    kwargs = {"device": DEVICE}
    if n_iter != "default":
        kwargs["n_iter"] = int(n_iter)
    if method in DP_METHODS:
        kwargs["epsilon"] = FORMAL_EPSILON

    log.info(f"[{method} n_iter={n_iter}] fitting on {len(train_df)} rows, device={DEVICE}, kwargs={kwargs}")
    plugin = Plugins().get(method, **kwargs)

    t0 = time.time()
    plugin.fit(GenericDataLoader(train_df, target_column=TARGET_COL))
    fit_s = round(time.time() - t0, 2)

    t1 = time.time()
    syn = plugin.generate(count=len(train_df)).dataframe()
    gen_s = round(time.time() - t1, 2)

    # TSTR utility (same performance.xgb.* metrics as eval_synthcity.py)
    score = Metrics.evaluate(
        X_gt=real_loader, X_syn=syn, task_type="classification",
        metrics={"performance": ["xgb"]},
        workspace=CONVERGENCE_DIR / "workspace_convergence",
    )

    def pick(idx):
        try:
            return float(score.loc[idx, "mean"])
        except Exception:
            return np.nan

    row = {
        "method": method, "n_iter": n_iter, "device": DEVICE,
        "fit_time_s": fit_s, "generate_time_s": gen_s,
        "xgb_gt": pick("performance.xgb.gt"),
        "xgb_syn_id": pick("performance.xgb.syn_id"),
        "xgb_syn_ood": pick("performance.xgb.syn_ood"),
    }
    log.info(f"[{method} n_iter={n_iter}] fit={fit_s}s gen={gen_s}s "
             f"xgb_syn_id={row['xgb_syn_id']:.4f} (gt={row['xgb_gt']:.4f})")
    append_row(row)
    return row


def project_budget():
    """Print the TAPAS shadow-model budget projection from measured fit times."""
    if not OUT_CSV.exists():
        return
    df = pd.read_csv(OUT_CSV)
    log.info("=== TAPAS budget projection (fits = (num_train+num_test) x 2) ===")
    for method in df["method"].unique():
        cfg = METHOD_CONFIG.get(method, {"num_train": 10, "num_test": 20})
        fits = (cfg["num_train"] + cfg["num_test"]) * 2
        sub = df[df["method"] == method]
        for _, r in sub.iterrows():
            per_fit = r["fit_time_s"] + r["generate_time_s"]
            total_min = fits * per_fit / 60.0
            fits_2hr = "FITS" if total_min <= 120 else "OVER"
            log.info(f"  {method:16s} n_iter={str(r['n_iter']):>7s}  "
                     f"{per_fit:7.1f}s/fit x {fits} fits = {total_min:6.1f} min "
                     f"[{fits_2hr} 2h @ {cfg['num_train']}/{cfg['num_test']}]  "
                     f"xgb_syn_id={r['xgb_syn_id']:.4f}")


def main():
    log.info(f"=== convergence check: methods={METHODS}, device={DEVICE}, candidates={N_ITER_CANDIDATES} ===")
    if DEVICE == "cpu":
        log.warning("Running on CPU -- GAN fits will be slow. This is meant for the Colab T4.")

    from synthcity.plugins.core.dataloader import GenericDataLoader
    train_df = load_train_df()
    real_loader = GenericDataLoader(train_df, target_column=TARGET_COL)

    for method in METHODS:
        for n_iter in N_ITER_CANDIDATES:
            if already_done(method, n_iter):
                log.info(f"[{method} n_iter={n_iter}] already recorded, skipping")
                continue
            try:
                run_one(method, n_iter, train_df, real_loader)
            except Exception:
                import traceback
                log.error(f"[{method} n_iter={n_iter}] FAILED:\n{traceback.format_exc()}")

    project_budget()
    log.info(f"=== Done. Results at {OUT_CSV} ===")


if __name__ == "__main__":
    main()
