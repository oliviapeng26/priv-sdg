#!/usr/bin/env python3
"""Benchmark privacy audit: Bayesian Network (non-DP statistical baseline).

5-attack TAPAS MIA battery, exact-knowledge + black-box, num_train=50/num_test=100.
Cheap on CPU -- run locally. Resumable (per-attack JSON + threat_model.pkl cache).

Run from repo root:
  python benchmark_tapas/scripts/run_bn.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/
import common
from config import method_results_dir

METHOD = "bayesian_network"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(method_results_dir(METHOD) / f"{METHOD}_log.txt"),
              logging.StreamHandler()],
)

if __name__ == "__main__":
    common.run_method(METHOD)
