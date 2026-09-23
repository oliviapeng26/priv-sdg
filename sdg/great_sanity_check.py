#!/usr/bin/env python3
"""Phase 1 sanity check for GReaT (be-great) on a 500-row sample of adult_train.csv.

Not a scientific result -- just confirms the venv, GPU, and be-great API work
end to end before committing to the full 21,523-row x 5-seed overnight run in
sdg/generate_great.py. Trains the LLM set below for 10 epochs on the same
500-row sample (fixed SAMPLE_SEED), times fit and sample separately, and
writes the generated rows to synthetic_data/great_sanity/sample_500_{LLM}.csv
for manual inspection.

LLM is currently set to gpt2 (full fine-tuning, no LoRA) to compare against
distilgpt2's numbers -- see synthetic_data/great_sanity/sample_500.csv
(distilgpt2's run, from before OUT_CSV was made per-LLM) and the earlier
result: distilgpt2 fit=7.7s, sample(k=2000)=291.5s for 500 rows.

SAMPLE_K=2000 (not be_great's default of 100) is baked in here because the
default left the GPU at ~0% utilization during sample() -- per-token
generation overhead dominates wall clock at small batch sizes. Confirmed on
distilgpt2 (581s -> 291.5s for 500 rows); this run re-checks it holds for a
different (larger) model too.

DISK: no model checkpoints are written (save_strategy/logging_strategy="no",
report_to=[] disable HF Trainer's own checkpoint and tensorboard writes). The
scratch experiment_dir is removed after the run. The only persistent output
is one ~500-row CSV. Model weights (~350 MB for distilgpt2, ~550 MB for
gpt2) are downloaded once into ~/.cache/huggingface on first run and reused
after that.

Run from repo root:
  python sdg/great_sanity_check.py            # defaults to gpt2
  python sdg/great_sanity_check.py distilgpt2 # or override the LLM
"""

import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "adult_train.csv"
OUT_DIR = REPO_ROOT / "synthetic_data" / "great_sanity"
SCRATCH_DIR = REPO_ROOT / "sdg" / ".great_scratch_sanity"

SAMPLE_N = 500
SAMPLE_SEED = 42   # sanity check only -- not tied to seeds.RUN_SEEDS

# Optional CLI override so both models can be rerun back-to-back without
# editing this file, e.g. to check whether a timing difference between them
# is real or a first-call CUDA/cuDNN warm-up artifact:
#   python sdg/great_sanity_check.py distilgpt2
#   python sdg/great_sanity_check.py gpt2
LLM = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
BATCH_SIZE = 32
EPOCHS = 10
FP16 = True
DATALOADER_NUM_WORKERS = 4
# efficient_finetuning intentionally left unset below -- full fine-tuning,
# no LoRA, to match DP-2Stage's GPT-2 setup exactly.

OUT_CSV = OUT_DIR / f"sample_500_{LLM}.csv"

# GReaT.sample()'s own generation batch size (separate from fit's BATCH_SIZE
# above). Default in be_great is k=100; nvidia-smi showed ~0% GPU utilization
# during sampling at that default, consistent with per-token Python/CUDA
# dispatch overhead dominating wall clock for such a small batch on a tiny
# model. Bump this to test whether a larger generation batch amortizes that
# overhead and actually keeps the GPU busy.
SAMPLE_K = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("great_sanity")


def main() -> int:
    from be_great import GReaT

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    sample = df.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    log.info(f"Loaded {len(sample)}-row sample from {TRAIN_CSV.relative_to(REPO_ROOT)}")
    log.info(f"sample() generation batch size k={SAMPLE_K} (be_great default is 100)")

    model = GReaT(
        llm=LLM,
        experiment_dir=str(SCRATCH_DIR),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        fp16=FP16,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        save_strategy="no",       # no checkpoints -- disk is tight
        logging_strategy="no",
        report_to=[],
    )

    t0 = time.perf_counter()
    try:
        model.fit(sample)
        fit_s = time.perf_counter() - t0
        log.info(f"Fit done in {fit_s:.1f}s ({fit_s / 60:.1f} min)")

        t0 = time.perf_counter()
        synthetic = model.sample(n_samples=SAMPLE_N, k=SAMPLE_K)
        sample_s = time.perf_counter() - t0
        log.info(f"Sample done in {sample_s:.1f}s ({sample_s / 60:.1f} min)")
    finally:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)

    synthetic.to_csv(OUT_CSV, index=False)
    log.info(f"Wrote {len(synthetic)} synthetic rows -> {OUT_CSV.relative_to(REPO_ROOT)}")
    log.info(f"=== TOTAL: {fit_s + sample_s:.1f}s ({(fit_s + sample_s) / 60:.1f} min) ===")

    # Extrapolation only -- fit cost is not strictly linear in row count (fixed
    # per-epoch overhead, batch padding effects), but this is enough to gauge
    # whether epochs=10 is affordable on the full 21,523-row / 5-seed run.
    scale = 21_523 / SAMPLE_N
    log.info(f"Rough extrapolation to one full-data seed (21523 rows): "
             f"~{fit_s * scale / 60:.0f} min fit + ~{sample_s * scale / 60:.0f} min sample "
             f"(linear scaling assumption, likely an underestimate)")

    print("\n=== sample of generated rows ===")
    print(synthetic.head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
