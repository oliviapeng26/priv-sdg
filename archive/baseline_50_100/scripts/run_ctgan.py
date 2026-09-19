#!/usr/bin/env python3
"""Benchmark privacy audit: CTGAN (non-DP neural baseline).

5-attack TAPAS MIA battery, exact-knowledge + black-box. GPU-constrained:
num_train=10/num_test=20 with the GAN's n_iter capped (see config.CTGAN_N_ITER),
targeting the Colab T4. Synthcity's ctgan plugin defaults to device='cpu' and
does NOT auto-detect CUDA, so we detect it here and pass it through.

Meant to run via run_ctgan_colab.ipynb on a GPU runtime; also runnable locally
on CPU (slow). Resumable (per-attack JSON + threat_model.pkl cache) -- safe to
re-run after a Colab disconnect.

Run from repo root:
  python benchmark_tapas/scripts/run_ctgan.py
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/
import common
from config import method_results_dir

METHOD = "ctgan"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(method_results_dir(METHOD) / f"{METHOD}_log.txt"),
              logging.StreamHandler()],
)
logging.getLogger("benchmark_tapas").info(
    f"CTGAN device: {DEVICE} (torch.cuda.is_available()={torch.cuda.is_available()})")

if __name__ == "__main__":
    common.run_method(METHOD, extra_plugin_kwargs={"device": DEVICE})
