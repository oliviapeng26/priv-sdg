#!/usr/bin/env python3
"""Probes for DP-2Stage, run on the GPU workstation in ~/venv-dp2stage.

Two subcommands, run in this order once Stage 1 exists (see sdg/dp2stage_driver.py):

  tokens   Rerun the max_new_tokens measurement under the workstation's
           transformers (4.30.2). The value MAX_NEW_TOKENS=115 was derived under
           transformers 5.12 on the Mac; this closes that assumption. No GPU, no
           Stage 1, no DP-2Stage checkout needed -- can run before anything else.

  fit      The real per-fit cost, on the 500-row scale every TAPAS audit fit runs
           at. Runs the driver's `fit-sample` REPEATS times (different seeds, same
           500 rows) and reports, per fit: fit and generation seconds, the SPENT
           epsilon and noise multiplier, acceptance after the public-vocabulary and
           integer filters, peak GPU memory, and an md5 of the output. On the first
           fit it also sweeps the sampling batch size (--probe-batches) on the
           already-fitted model: one DP fit serves the whole sweep.

Checks it prints PASS/FAIL for:
  - spent epsilon <= target (Opacus solves sigma to within 0.01 of the target)
  - the output CSVs differ between seeds (the seeding lesson from GReaT: identical
    md5s mean a seed is being silently overridden)
  - every fit returned the full row quota
  - integer columns are written as integers, not "47.0"
Whether generating from the in-memory fitted model works at all (their authors
always reload a saved checkpoint in a separate run) is answered simply by the
first fit finishing.

Run from ~/priv-sdg, one GPU pinned, in a named screen:
  source ~/venv-dp2stage/bin/activate
  export CUDA_VISIBLE_DEVICES=0
  nvidia-smi                                   # is czha4500's job on this GPU?
  python -u sdg/dp2stage_probe.py tokens
  python -u sdg/dp2stage_probe.py fit

What to send back: the summary tables at the end of each command (also appended
to sdg/dp2stage_probe_log.txt), plus the tail of sdg/.dp2stage_probe/fit_seed*.log
if a fit failed. Nothing here is deleted afterwards: sdg/.dp2stage_probe/ holds
two 500-row CSVs, two info JSONs and two logs (all tiny, all gitignored).

The 500 private rows are seeded with TAPAS_BG_SEED, so they are a stand-in of the
same size as the audit's 499-row background + target -- not the same rows.
"""

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SDG_DIR = REPO_ROOT / "sdg"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SDG_DIR))

from seeds import TAPAS_BG_SEED                                  # noqa: E402
from dp2stage_driver import INTEGER_COLS, MAX_NEW_TOKENS        # noqa: E402

TRAIN_CSV = REPO_ROOT / "data" / "adult_train.csv"
PROBE_DIR = SDG_DIR / ".dp2stage_probe"
DRIVER = SDG_DIR / "dp2stage_driver.py"

# Measured on the Mac under transformers 5.12.1 over all 21,523 training rows in
# DP-2Stage's serialised form (income first, rest alphabetical, integers,
# "<key> is <value>," joined by ","), prompt "income is" removed.
LOCAL_TOKEN_STATS = {"min": 68, "median": 77, "p99": 85, "max": 94, "worst_case": 110}

# Opacus make_private_with_epsilon solves for sigma with epsilon_tolerance=0.01.
EPS_TOLERANCE = 0.01
TARGET_EPSILON = 1.0

log = logging.getLogger("dp2stage_probe")


# -- tokens ---------------------------------------------------------------------

def cmd_tokens(a):
    import transformers
    from transformers import AutoTokenizer

    df = pd.read_csv(TRAIN_CSV)
    df[INTEGER_COLS] = df[INTEGER_COLS].astype(int)
    cols = ["income"] + sorted(set(df.columns) - {"income"})    # ft_opacus.get_dataset, shuffle off
    df = df[cols]

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.add_tokens(",")                                          # LLMtgDataset.set_tokenizer
    log.info(f"transformers {transformers.__version__}, tokenizers "
             f"{__import__('tokenizers').__version__}, fast={tok.is_fast}")

    prompt_len = len(tok("income is")["input_ids"])
    rows = [",".join(f" {c} is {str(v).strip()}" for c, v in zip(cols, r))
            for r in df.itertuples(index=False)]
    generated = [r[len(" income is"):] for r in rows]
    n = np.array([len(tok(g, add_special_tokens=False)["input_ids"]) for g in generated])

    per_col = {c: max(len(tok(f" {c} is {v}", add_special_tokens=False)["input_ids"])
                      for v in df[c].astype(str).unique()) for c in cols}
    # every column at its longest, +12 commas, minus the prompt, +1 for the EOS
    worst = sum(per_col.values()) + (len(cols) - 1) - prompt_len + 1

    here = {"min": int(n.min()), "median": int(np.median(n)),
            "p99": int(np.percentile(n, 99)), "max": int(n.max()), "worst_case": worst}
    log.info(f"{len(rows)} rows; prompt 'income is' = {prompt_len} tokens; "
             f"per-column longest value: {per_col}")
    log.info(f"{'stat':<12}{'Mac (5.12)':>12}{'here':>8}")
    for k, v in LOCAL_TOKEN_STATS.items():
        log.info(f"{k:<12}{v:>12}{here[k]:>8}{'' if here[k] == v else '   <-- differs'}")

    # Informational only: tokenizer versions can shift a single row by a token (the
    # first workstation run had max 93 vs the Mac's 94). What decides the verdict is
    # whether the cap still covers the worst case measured HERE.
    same = here == LOCAL_TOKEN_STATS
    log.info(f"counts identical to the Mac measurement: "
             f"{'yes' if same else 'no (informational -- see the differing rows above)'}")
    cap_ok = MAX_NEW_TOKENS >= worst
    log.info(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} >= worst case {worst}: "
             f"{'PASS' if cap_ok else 'FAIL -- cap would truncate valid rows; re-derive it'}")
    return 0 if cap_ok else 1


# -- fit ------------------------------------------------------------------------

def _gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        return out.strip()
    except Exception as e:                                       # noqa: BLE001
        return f"(nvidia-smi unavailable: {e})"


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def summarise(runs, n_samples):
    """runs: list of dicts with 'seed', 'info' (driver JSON) and 'out_csv'. Returns ok."""
    log.info("")
    log.info(f"{'seed':>5} {'fit_s':>7} {'gen_s':>6} {'eps_spent':>9} {'sigma':>7} {'steps':>6} "
             f"{'accept':>7} {'rows':>5} {'GBfit':>6} {'GBgen':>6} {'finalperp':>10}  md5")
    ok = True
    md5s = []
    for r in runs:
        i = r["info"]
        md5s.append(_md5(r["out_csv"]))
        log.info(f"{r['seed']:>5} {i['fit_s']:>7} {i['gen_s']:>6} {i['epsilon_spent']:>9.4f} "
                 f"{i['noise_multiplier']:>7.3f} {i['dp_steps']:>6} {str(i['acceptance']):>7} "
                 f"{i['n_returned']:>5} {i['peak_gpu_gb_fit']:>6} {i['peak_gpu_gb_gen']:>6} "
                 f"{str(r.get('train_perp')):>10}  {md5s[-1][:10]}")

    eps_ok = all(r["info"]["epsilon_spent"] <= TARGET_EPSILON + EPS_TOLERANCE for r in runs)
    rows_ok = all(r["info"]["n_returned"] == n_samples for r in runs)
    distinct = len(set(md5s)) == len(md5s)
    fmt_ok = all(all(str(t) == "int64" for t in
                     pd.read_csv(r["out_csv"])[INTEGER_COLS].dtypes) for r in runs)
    log.info("")
    log.info(f"spent epsilon <= {TARGET_EPSILON}+{EPS_TOLERANCE}:     {'PASS' if eps_ok else 'FAIL'}")
    log.info(f"full row quota on every fit:          {'PASS' if rows_ok else 'FAIL'}")
    log.info(f"outputs differ between seeds (md5):   "
             f"{'PASS' if distinct else 'FAIL -- a seed is being overridden'}"
             if len(runs) > 1 else "outputs differ between seeds: n/a (one fit)")
    log.info(f"integer columns written as integers:  {'PASS' if fmt_ok else 'FAIL'}")
    ok = eps_ok and rows_ok and fmt_ok and (distinct or len(runs) == 1)

    sweep = runs[0]["info"].get("batch_probe")
    if sweep:
        log.info("")
        log.info(f"sample-batch sweep, one fitted model (seed {runs[0]['seed']}); "
                 f"peak = torch ALLOCATED GB (nvidia-smi reads higher)")
        log.info(f"{'k':>6} {'ok':>5} {'rows':>6} {'drawn':>6} {'rounds':>6} {'sec':>7} "
                 f"{'rows/s':>7} {'peakGB':>7}  note")
        for s in sweep:
            log.info(f"{s['k']:>6} {str(s['ok']):>5} {s['rows']:>6} {s.get('rows_drawn', '-'):>6} "
                     f"{s.get('gen_rounds', '-'):>6} {s['seconds']:>7} {s['rows_per_s']:>7} "
                     f"{s['peak_gpu_gb']:>7}  {s.get('note', '')}")
    log.info("")
    log.info(f"OVERALL: {'PASS' if ok else 'FAIL -- see above'}")
    return ok


def cmd_fit(a):
    PROBE_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(TRAIN_CSV)
    idx = np.random.RandomState(TAPAS_BG_SEED).choice(len(train), a.n_train, replace=False)
    private_csv = PROBE_DIR / f"private{a.n_train}.csv"
    train.iloc[idx].to_csv(private_csv, index=False)

    log.info(f"GPU before the run (is another job on this card?):\n{_gpu_snapshot()}")
    log.info(f"private sample: {a.n_train} rows of adult_train (TAPAS_BG_SEED={TAPAS_BG_SEED}); "
             f"{a.repeats} fits, seeds {a.seed}..{a.seed + a.repeats - 1}, "
             f"n_samples={a.n_samples}, sample_batch={a.sample_batch}, "
             f"batch sweep={a.probe_batches or 'off'}, lr={a.lr if a.lr is not None else 'driver default'}")

    runs = []
    for r in range(a.repeats):
        seed = a.seed + r
        sfx = f"_lr{a.lr:g}" if a.lr is not None else ""
        out_csv = PROBE_DIR / f"synth_seed{seed}{sfx}.csv"
        info_json = PROBE_DIR / f"info_seed{seed}{sfx}.json"
        fit_log = PROBE_DIR / f"fit_seed{seed}{sfx}.log"
        cmd =[sys.executable, "-u", str(DRIVER), "fit-sample",
               "--train-csv", str(private_csv), "--out-csv", str(out_csv),
               "--info-json", str(info_json), "--n-samples", str(a.n_samples),
               "--seed", str(seed), "--gen-seed", str(seed),
               "--sample-batch", str(a.sample_batch), "--scratch-root", str(PROBE_DIR)]
        if a.stage1_dir:
            cmd += ["--stage1-dir", a.stage1_dir]
        if a.lr is not None:
            cmd += ["--lr", str(a.lr)]
        if r == 0 and a.probe_batches:
            cmd += ["--probe-batches", *map(str, a.probe_batches),
                    "--probe-rows", str(a.probe_rows)]
        log.info(f"--- fit {r + 1}/{a.repeats} (seed {seed}); live output: tail -f {fit_log}")
        t0 = time.perf_counter()
        with open(fit_log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        log.info(f"    exit {rc} after {time.perf_counter() - t0:.0f}s wall")
        perps = re.findall(r"Train perp = ([0-9.eE+\-]+)", fit_log.read_text(errors="replace"))
        train_perp = round(float(perps[-1]), 2) if perps else None
        if rc != 0 or not info_json.exists():
            if info_json.exists():       # the driver writes it even when generation fails
                fi = json.loads(info_json.read_text())
                log.info(f"    the fit itself ran: lr={fi.get('lr')} fit_s={fi.get('fit_s')} "
                         f"eps_spent={fi.get('epsilon_spent')} sigma={fi.get('noise_multiplier')} "
                         f"steps={fi.get('dp_steps')} final Train perp={train_perp} "
                         f"(Stage 1 ended near 1.2; ~50000 is random guessing)")
            meaning = {3: "hard failure (CUDA OOM or a zero-row generation round) -- "
                          "check nvidia-smi for another job", 4: "quota not filled"}.get(rc, "")
            log.info(f"FIT FAILED (exit {rc}) {meaning}\n--- tail of {fit_log} ---\n"
                     + "".join(fit_log.read_text().splitlines(keepends=True)[-25:]))
            return 1
        runs.append({"seed": seed, "info": json.loads(info_json.read_text()),
                     "out_csv": out_csv, "train_perp": train_perp})

    return 0 if summarise(runs, a.n_samples) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tokens", help="max_new_tokens under this env's transformers")

    f = sub.add_parser("fit", help="500-row fit-sample x REPEATS + sample-batch sweep")
    f.add_argument("--n-train", type=int, default=500)
    f.add_argument("--n-samples", type=int, default=500)
    f.add_argument("--seed", type=int, default=1000)
    f.add_argument("--repeats", type=int, default=2, help="fits, seeds SEED..SEED+REPEATS-1")
    f.add_argument("--sample-batch", type=int, default=100, help="k for the main generation")
    f.add_argument("--probe-batches", type=int, nargs="*", default=[100, 500, 2000],
                   help="batch sizes to sweep on the first fit; pass none to skip")
    f.add_argument("--probe-rows", type=int, default=2000)
    f.add_argument("--stage1-dir", default=None, help="default: the driver's STAGE1_DIR")
    f.add_argument("--lr", type=float, default=None,
                   help="Stage 2 learning rate, PROBING only (default: the driver's LR). Outputs get "
                        "an _lr<value> suffix so runs with different rates never overwrite each other.")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(SDG_DIR / "dp2stage_probe_log.txt", mode="a"),
                                  logging.StreamHandler()])
    if a.cmd == "fit" and a.probe_batches and a.probe_rows < max(a.probe_batches):
        p.error("--probe-rows must be >= the largest --probe-batches, or that batch never fills")
    return cmd_tokens(a) if a.cmd == "tokens" else cmd_fit(a)


if __name__ == "__main__":
    sys.exit(main())
