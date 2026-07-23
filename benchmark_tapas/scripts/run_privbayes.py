#!/usr/bin/env python3
"""Benchmark privacy audit: PrivBayes (DP statistical, eps=1.0).

5-attack TAPAS MIA battery, exact-knowledge + black-box, num_train=50/num_test=100.
Fresh 5-attack run (the iteration-2 eff_eps run was 4 attacks) for an
apples-to-apples comparison with the other three methods. ~28.6 s/fit on CPU ->
~2.4 hr. Resumable (per-attack JSON + threat_model.pkl cache).

Run from repo root:
  python benchmark_tapas/scripts/run_privbayes.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # benchmark_tapas/
import common
from config import method_results_dir

METHOD = "privbayes"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(method_results_dir(METHOD) / f"{METHOD}_log.txt"),
              logging.StreamHandler()],
)

if __name__ == "__main__":
    common.run_method(METHOD)
