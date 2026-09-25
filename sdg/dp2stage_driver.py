#!/usr/bin/env python3
"""DP-2Stage (Afonja et al., TMLR 2025), variant O, driven from priv-sdg.

RUNS IN ~/venv-dp2stage, NOT ~/priv-sdg/venv
    DP-2Stage pins python 3.9, torch 2.5.1, transformers 4.30.2, opacus 1.4.1 and
    numpy 1.21.6 -- incompatible with the shared venv (torch 2.2.2, transformers<5),
    the same split as DP-CTGAN's ~/venv-smartnoise. This script therefore imports
    nothing from priv-sdg and must stay python-3.9 compatible. sdg/generate_dp2stage.py
    and the TAPAS wrapper call it as a subprocess with ~/venv-dp2stage/bin/python.

    The DP-2Stage checkout is found at $DP2STAGE_ROOT (default ~/DP-2Stage), pinned
    to commit 8234a7a151fe58bde67f4b38f228c52f92bb1969 -- every line reference
    below is against that commit.

WHY THIS CALLS THEIR main() RATHER THAN RE-IMPLEMENTING IT
    Training is ft_opacus.main(), unmodified, driven through sys.argv -- so the
    DP-SGD loop, the weighted (disentangled) loss, Opacus's make_private_with_epsilon
    and BatchMemoryManager are exactly the published method. Generation is their
    generation.generate(), also unmodified. Three things are patched from outside,
    none of which touches the mechanism:

      ft_opacus.save_model   -> captures the trained model instead of writing
                                ~1.5 GB (weights + optimizer state + RNG) per fit.
                                Stage 1 writes weights only.
      ft_opacus.PrivacyEngine -> a subclass that remembers itself, so the SPENT
                                epsilon can be read back. Their code logs the target
                                epsilon only; the get_epsilon line is commented out.
      generation.get_metadata -> the public-vocabulary fix, below.

    Generation runs in the same process on the captured model (model._module, the
    unwrapped GPT-2 their own final save_model receives), in eval mode under
    torch.no_grad -- the same weights their separate generate.sh run would reload.

SUBCOMMANDS
    stage1      non-DP fine-tune on the Airline pseudo data. Run ONCE; every Stage 2
                fit (5 seeds, every TAPAS simulation) starts from this checkpoint.
                Stage 1 never sees Adult, which is what makes variant O's epsilon
                the whole story.
    fit-sample  DP Stage 2 on one private CSV, then sample N rows. Writes the rows
                to --out-csv and privacy/timing diagnostics to --info-json.

DELIBERATE DEVIATIONS FROM THE REPO'S OWN SCRIPTS -- each one is a privacy or
correctness fix, not tuning, and belongs in the write-up
  1. --shuffle_dataset False, passed explicitly. ft_opacus.py defaults it to True
     and 2Stage_train.sh / generate.sh both set True; only the paper's config files
     (ood/stage1/full_nodp.sh, ood/stage2/full_dp-eps1.sh), sourced afterwards,
     flip it to False. Relying on that chain would silently give GReaT-style column
     permutation. DP-2Stage fixes the order (its own finding: shuffling hurts under
     DP): income first, the rest alphabetical (ft_opacus.get_dataset). GReaT
     shuffles by design -- the two are SUPPOSED to differ here.
  2. --start_prompt default, NOT GPT2.sh's "categorical". categorical_start
     (utils/dataset.py:640) draws each prompt's income value from the PRIVATE
     training set's own income frequencies, with no noise -- the private income
     marginal copied into the output outside the DP accounting. "default" prompts
     just "income is" and the DP-trained model generates the value.
  3. Public vocabulary for rejection. generate() validates categories against
     get_metadata(dataset.to_pandas()) -- the PRIVATE training set's observed
     categories (generation.py:98, utils/dataset.py:871). Any category the private
     data happens not to contain is rejected, so the output support reveals which
     values were present: a deterministic membership signal outside epsilon. On a
     500-row TAPAS dataset 15 of 41 native_country values are absent, so a rare-
     country target would be accepted under D+ and filtered under D-. The patched
     get_metadata returns the public schema instead -- the union of adult_train and
     adult_test categories, the same codebook TAPAS's build_description gives every
     generator -- with no private statistics at all (no weights, no min/max, no
     value lists: anything that reached for them would KeyError, not leak).
     Same patch also fixes a case bug: their get_word_case normaliser turns
     'Married-AF-spouse' and 'Outlying-US(Guam-USVI-etc)' into spellings not in the
     vocabulary, so those two values could never be generated. Here a generated
     value is matched case-insensitively and replaced by its canonical spelling.
  4. Integer serialisation, both directions. DP-2Stage has no format setting: its
     serialiser writes str(value) of whatever dtype the CSV loaded, and its own
     Adult download is int64 ("age is 47"), so the float question never arose for
     them. adult_train.csv stores the 5 numeric columns as floats ("47.0"); all are
     integer-valued, so they are cast to int before training -- the form the token
     budget below was measured on. On the way out, their postprocess_data truncates
     a generated "47.5" to 47 and can leave a column float after a swallowed cast
     error, so integer columns are read as float and rows with non-integer values
     are rejected and refilled (_integer_rows_only). Stage 1's Airline data is NOT
     cast: its float columns (arrival-delay, which also has NaNs) serialise as
     "is 0.0"/"is nan". That only shapes Stage 1's template; Stage 2 retrains on
     integers. Check the Airline CSV after download.

max_new_tokens
    Measured, not inherited: every one of the 21,523 training rows serialised in
    this exact format (fixed order, "<key> is <value>," joined by ",", tokenizer
    with "," registered as an added token as set_tokenizer does), the prompt
    "income is" stripped, the remainder tokenised with GPT-2's tokenizer:
        min 68, median 77, p99 85, max 94 tokens.
    Theoretical worst case, every column at its longest value: 99 (per-column
    maxima) + 12 commas - 2 (the "income is" prompt) = 109, +1 for the EOS the model
    learns from eos-padding = 110. MAX_NEW_TOKENS = 115: the exact bound plus a
    small buffer (5), chosen by hand -- not a derived margin. It matters for more than truncation: training rows carry no EOS of
    their own, and the deserialiser lets a later "<key> is" overwrite an earlier one
    (utils/dataset.py convert_great), so a model that runs on past the row would
    splice a second partial row into the first. A tight cap bounds that.
    Sampling-time only: it does not enter the DP accounting.

Usage (from ~/priv-sdg, one GPU pinned):
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0
  ~/venv-dp2stage/bin/python sdg/dp2stage_driver.py stage1 \\
      --train-csv ~/DP-2Stage/data/airline/k1000/train.csv
  ~/venv-dp2stage/bin/python sdg/dp2stage_driver.py fit-sample \\
      --train-csv <private.csv> --out-csv <synth.csv> --info-json <info.json> \\
      --n-samples 500 --seed 1000 --gen-seed 1000

EXIT CODES (fit-sample)
  0  full quota written
  3  hard failure: CUDA OOM in training, or a generation round that produced NO TEXT AT ALL
     (generate() swallows every exception, OOM included, and returns what it has).
     Transient under GPU contention -- callers map it to HardSampleFailure.
  4  quota not filled: either MAX_GEN_ROUNDS rounds that each produced SOME valid rows, or a
     round that wrote text but zero VALID rows (NoValidRows) -- a model problem, not
     contention; retrying the same config fails the same way.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DP2STAGE_ROOT = Path(os.environ.get("DP2STAGE_ROOT", "~/DP-2Stage")).expanduser()
DP2STAGE_COMMIT = "8234a7a151fe58bde67f4b38f228c52f92bb1969"

# Public schema: the same train+test union TAPAS's build_description uses.
PUBLIC_VOCAB_CSVS = [REPO_ROOT / "data" / "adult_train.csv",
                     REPO_ROOT / "data" / "adult_test.csv"]
INTEGER_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

STAGE1_DIR = DP2STAGE_ROOT / "runs" / "priv-sdg_stage1_airline"   # outside our repo: ~0.5 GB

LLM = "gpt2"

# Stage 1: ood/stage1/full_nodp.sh + stage1_master.sh + airline_general.sh.
STAGE1_EPOCHS = 5
STAGE1_BATCH_SIZE = 32
STAGE1_LR = 5e-4
STAGE1_START_COL = "satisfaction"
STAGE1_WEIGHTED_LOSS = -1          # -1 -> None in main(): plain LM loss

# Stage 2: ood/stage2/full_dp-eps1.sh + master.sh + adult_general.sh.
EPOCHS = 10
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 16              # BatchMemoryManager physical batch; not a privacy param
LR = 5e-4                          # released scripts (5e-4), NOT the paper text (5e-5, §5.1).
                                   # Chosen because the scripts are more likely what produced
                                   # the published tables (their run folders are named
                                   # LR0.0005) -- an inference, not confirmed. Not a privacy
                                   # parameter: epsilon is unaffected, utility is not.
WEIGHTED_LOSS = 0.65
EPSILON = 1.0
DELTA = 1e-5                       # ft_opacus.py default; no config overrides it.
                                   # Differs from AIM (1e-9) / DP-CTGAN (1/(n sqrt n)) /
                                   # DPGAN (1/n) -- a stated limitation, not an oversight.
CLIP = 1.0

# Generation: GPT2.sh (temperature, top_p), adult_general.sh (sample batch),
# ft_opacus.py argparse defaults (retries). top_k is never set anywhere in
# DP-2Stage, so HF's default of 50 applies -- same as GReaT.
START_COL = "income"
START_PROMPT = "default"           # deviation 2 -- see docstring
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_NEW_TOKENS = 115               # measured bound 110 + 5 -- see docstring
SAMPLE_BATCH = 100                 # DP-2Stage's own default (GPT2.sh / generate.sh); override
                                   # with --sample-batch. Sampling-time only, no effect on
                                   # epsilon. generate() draws min(k, rows still needed) per
                                   # call, so k above the requested row count changes nothing.
SAMPLING_MAX_RETRIES = 15
MAX_GEN_ROUNDS = 20
GEN_ROUND_SEED_STRIDE = 100_000    # round r of generation runs at gen_seed + r * stride

EXIT_HARD_FAILURE = 3
EXIT_SHORTFALL = 4


def _import_dp2stage():
    if not (DP2STAGE_ROOT / "ft_opacus.py").exists():
        sys.exit(f"DP-2Stage checkout not found at {DP2STAGE_ROOT} (set DP2STAGE_ROOT)")
    sys.path.insert(0, str(DP2STAGE_ROOT))   # their code imports `utils.*`, `generation`
    import ft_opacus
    import generation
    return ft_opacus, generation


def _run_main(ft_opacus, argv):
    """ft_opacus.main() with our argv; returns (model, args) from its final save_model."""
    captured = {}

    def capture_save_model(model, savedir, args):
        captured["model"], captured["args"] = model, args

    class RecordingPrivacyEngine(ft_opacus.PrivacyEngine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            captured["engine"] = self

    ft_opacus.save_model = capture_save_model
    ft_opacus.PrivacyEngine = RecordingPrivacyEngine
    sys.argv = ["ft_opacus.py"] + [str(x) for x in argv]
    ft_opacus.main()
    assert "model" in captured, "ft_opacus.main() returned without reaching its final save_model"
    return captured


def _common_argv(train_csv, output_dir, seed, model_path, checkpoint_path, epochs,
                 batch_size, lr, weighted_loss, start_col):
    return [
        "--train_mode", "True", "--evaluation_mode", "False", "--generation_mode", "False",
        "--train_file", train_csv,                 # no --validation_file: eval is off
        "--output_dir", output_dir,                # always a fresh temp dir -- see below
        "--resume_from_checkpoint", "False",       # find_latest_checkpoint would otherwise
                                                   # restore a previous fit's model + RNG
                                                   # from any resume dict in output_dir
        "--seed", seed,                            # set_seed() once at the top of main();
                                                   # there is no HF Trainer to reset it
        "--model_name_or_path", model_path, "--config_name", model_path,
        "--model_type", LLM, "--tokenizer_name", LLM,
        "--checkpoint_path", checkpoint_path,
        "--finetune_type", "entire", "--loading_4_bit", "False",
        "--num_train_epochs", epochs, "--max_train_steps", 0,
        "--save_every_epoch", 0, "--save_every_step", 0,   # 0 -> None: no mid-run saves
        "--train_batch_size", batch_size, "--learning_rate", lr,
        "--lr_scheduler_type", "linear", "--weighted_loss", weighted_loss,
        "--shuffle_dataset", "False",              # deviation 1 -- see docstring
        "--start_col", start_col,
        "--cache_dir", DP2STAGE_ROOT / "cache", "--device", "cuda",
    ]


# -- stage 1 ------------------------------------------------------------------

def cmd_stage1(a):
    ft_opacus, _ = _import_dp2stage()
    import safetensors.torch

    out_dir = Path(a.out_dir)
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists -- Stage 1 is trained once. Move it aside "
                 f"first if you really mean to retrain.")
    scratch = Path(tempfile.mkdtemp(prefix=".dp2stage_stage1_", dir=REPO_ROOT / "sdg"))
    t0 = time.perf_counter()
    try:
        cap = _run_main(ft_opacus, _common_argv(
            a.train_csv, scratch, a.seed, LLM, "None", STAGE1_EPOCHS, STAGE1_BATCH_SIZE,
            STAGE1_LR, STAGE1_WEIGHTED_LOSS, STAGE1_START_COL) + ["--enable_privacy", "False"])
        # Their save_model minus the ~1 GB optimizer/RNG resume dict.
        model = cap["model"]
        out_dir.mkdir(parents=True)
        model.config.save_pretrained(out_dir)
        model.generation_config.save_pretrained(out_dir)
        safetensors.torch.save_model(model, str(out_dir / "model.safetensors"),
                                     metadata={"format": "pt"})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    elapsed = time.perf_counter() - t0
    info = {"train_csv": str(a.train_csv), "rows": len(pd.read_csv(a.train_csv)),
            "seed": a.seed, "epochs": STAGE1_EPOCHS, "batch_size": STAGE1_BATCH_SIZE,
            "lr": STAGE1_LR, "dp2stage_commit": DP2STAGE_COMMIT,
            "elapsed_s": round(elapsed, 1)}
    (out_dir / "priv-sdg_stage1_info.json").write_text(json.dumps(info, indent=2))
    print(f"Stage 1 done in {elapsed / 3600:.2f} h -> {out_dir}")
    return 0


# -- stage 2 + sampling -----------------------------------------------------------

def _public_metadata_factory(dataset_utils):
    """Replacement for generation.get_metadata: public schema, no private statistics."""
    public = pd.concat([pd.read_csv(p) for p in PUBLIC_VOCAB_CSVS], ignore_index=True)

    def get_metadata(data):
        metadata = {}
        for col in data.columns:
            col = col.lower()
            entry = {"case": dataset_utils.get_word_case(col)}   # column-name case, as theirs
            if col in INTEGER_COLS:
                # NOT "int64": postprocess_data does astype(dtype) after float(), which
                # silently truncates a generated "47.5" to 47, and when any row in the
                # batch is NaN the cast raises and is swallowed, leaving the column
                # float ("47.0" in the CSV). Read as float here; _integer_rows_only
                # rejects non-integers and casts explicitly.
                entry["dtype"] = "float64"
            else:
                vocab = sorted(public[col].astype(str).unique())
                entry["dtype"] = "object"
                entry["categories"] = {
                    "unique": vocab,
                    # exact canonical spelling for any case variant -- deviation 3
                    "case": {v.lower(): (lambda x, v=v: v) for v in vocab},
                }
            metadata[col] = entry
        return metadata

    return get_metadata


def _integer_rows_only(df):
    """Keep rows whose integer columns are all integer-valued; cast them to int64.

    The generation-side twin of deviation 4: a model that drifts to "age is 47.5"
    would otherwise be truncated silently, i.e. a row the model never produced.
    """
    ok = pd.Series(True, index=df.index)
    for c in INTEGER_COLS:
        ok &= (pd.to_numeric(df[c], errors="coerce") % 1 == 0)
    df = df[ok].copy()
    df[INTEGER_COLS] = df[INTEGER_COLS].astype("float64").astype("int64")
    return df


class NoValidRows(RuntimeError):
    """A generation round produced text but ZERO valid rows: the model ran, and what it
    wrote is not Adult rows (seen at n=500: Stage 2 gets 160 DP steps and keeps writing
    Stage 1's Airline columns). NOT transient -- retrying the same config fails the same
    way -- so it exits EXIT_SHORTFALL, not EXIT_HARD_FAILURE, and the audit's restart loop
    does not retry it forever."""


class HardGenerationFailure(RuntimeError):
    """A generation round returned zero rows. generate() catches every exception
    (CUDA OOM included) and returns what it has, so zero rows is how OOM or GPU
    contention shows up -- transient, unlike a model that steadily under-fills."""


def _sample_rows(generation, model, dataset, torch, scratch, tag, n, k, gen_seed):
    """Up to n rows via generation.generate(), refilling the shortfall each round.

    Returns (rows, rows_drawn, rounds); rows may be < n after MAX_GEN_ROUNDS.
    Raises HardGenerationFailure on a zero-row round.
    """
    collected, have, drawn, rounds = [], 0, 0, 0
    with torch.no_grad():
        while have < n and rounds < MAX_GEN_ROUNDS:
            name = f"{tag}_round{rounds}"   # fresh name -> no saved RNG state to reload
            batch = generation.generate(
                n_samples=n - have, model=model, dataset=dataset,
                start_prompt=START_PROMPT, start_col=START_COL,
                temperature=TEMPERATURE, top_p=TOP_P, k=k,
                max_length=MAX_NEW_TOKENS, drop_nan=True, do_impute=False,
                prompt_template=None, device="cuda",
                max_retries=SAMPLING_MAX_RETRIES, max_allowed_time=None,
                save_folder=str(scratch / "synth"), save_name=name,
                seed=gen_seed + rounds * GEN_ROUND_SEED_STRIDE)
            raw_txt = scratch / "synth" / "raw_texts" / f"{name}.txt"
            drawn += sum(1 for _ in open(raw_txt)) if raw_txt.exists() else 0
            rounds += 1
            if len(batch) == 0:
                # Zero valid rows is either contention (nothing was generated) or a model
                # that writes well-formed nonsense (e.g. still emitting Stage 1's Airline
                # keys). The raw text tells them apart, and it lives in a scratch dir that
                # is deleted on exit -- so show a few rows in the message itself.
                lines = []
                if raw_txt.exists():
                    lines = [l.rstrip("\n") for l in open(raw_txt) if l.strip()][:3]
                if lines:       # the model ran and wrote text, none of it a valid row
                    raise NoValidRows(
                        f"generation round {rounds} ({tag}, k={k}) wrote text but 0 VALID rows -- "
                        f"a model problem, not contention. Raw model output, first {len(lines)} "
                        f"row(s) of this round (prompt included):\n"
                        + "\n".join("    " + repr(l[:400]) for l in lines))
                raise HardGenerationFailure(
                    f"generation round {rounds} ({tag}, k={k}) produced no text at all -- "
                    f"check nvidia-smi for OOM/contention")
            batch = _integer_rows_only(batch)   # after the zero-row check: an empty
                                                # result here is a format problem, not OOM
            collected.append(batch)
            have += len(batch)
    return pd.concat(collected, ignore_index=True).head(n), drawn, rounds


def cmd_fit_sample(a):
    ft_opacus, generation = _import_dp2stage()
    import torch
    import utils.dataset as dataset_utils

    stage1_dir = Path(a.stage1_dir)
    if not (stage1_dir / "model.safetensors").exists():
        sys.exit(f"no Stage 1 checkpoint at {stage1_dir} -- run `stage1` first")
    lr = a.lr if a.lr is not None else LR    # --lr is for PROBING; the final value belongs in LR

    private = pd.read_csv(a.train_csv)
    for c in INTEGER_COLS:                                          # deviation 4
        vals = pd.to_numeric(private[c])
        assert (vals % 1 == 0).all(), f"{c} has non-integer values; refusing to truncate"
        private[c] = vals.astype("int64")
    out_cols = list(private.columns)

    scratch = Path(tempfile.mkdtemp(prefix=".dp2stage_fit_", dir=a.scratch_root))
    info = {"n_train": len(private), "n_requested": a.n_samples, "seed": a.seed,
            "gen_seed": a.gen_seed, "target_epsilon": EPSILON, "delta": DELTA,
            "clip": CLIP, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": lr,
            "max_new_tokens": MAX_NEW_TOKENS, "sample_batch": a.sample_batch,
            "stage1_dir": str(stage1_dir),
            "dp2stage_commit": DP2STAGE_COMMIT}
    try:
        train_csv = scratch / "private.csv"
        private.to_csv(train_csv, index=False)

        t0 = time.perf_counter()
        try:
            cap = _run_main(ft_opacus, _common_argv(
                train_csv, scratch / "run", a.seed, stage1_dir,
                stage1_dir / "model.safetensors", EPOCHS, BATCH_SIZE, lr,
                WEIGHTED_LOSS, START_COL) + [
                "--enable_privacy", "True", "--micro_batch_size", MICRO_BATCH_SIZE,
                "--target_epsilon", EPSILON, "--target_delta", DELTA,
                "--max_grad_norm", CLIP])
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM during Stage 2 training: {e}", file=sys.stderr)
            return EXIT_HARD_FAILURE
        info["fit_s"] = round(time.perf_counter() - t0, 1)

        # RDP accountant history: one (noise_multiplier, sample_rate, steps) entry
        # per distinct setting -- exactly what the spent epsilon was computed from.
        engine = cap["engine"]
        (sigma, q, steps), = engine.accountant.history
        info.update(epsilon_spent=float(engine.get_epsilon(DELTA)),
                    noise_multiplier=float(sigma), sample_rate=float(q), dp_steps=int(steps))

        # Rebuild their dataset object (tokenizer, serializer, fixed column order)
        # exactly as main() did, for the deserialiser.
        tokenizer = ft_opacus.get_tokenizer(LLM, cache_dir=str(DP2STAGE_ROOT / "cache"))
        dataset, _ = ft_opacus.get_dataset(cap["args"], tokenizer)
        generation.get_metadata = _public_metadata_factory(dataset_utils)   # deviation 3

        model = cap["model"]
        model.eval()
        # torch's ALLOCATED peak, not nvidia-smi's (which adds the caching allocator's
        # reserve and the CUDA context) -- expect nvidia-smi to read higher.
        info["peak_gpu_gb_fit"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        gen_kw = dict(generation=generation, model=model, dataset=dataset,
                      torch=torch, scratch=scratch)

        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        try:
            out, drawn, rounds = _sample_rows(tag="main", n=a.n_samples, k=a.sample_batch,
                                              gen_seed=a.gen_seed, **gen_kw)
        except NoValidRows as e:
            print(e, file=sys.stderr)
            return EXIT_SHORTFALL
        except HardGenerationFailure as e:
            print(e, file=sys.stderr)
            return EXIT_HARD_FAILURE
        info.update(gen_s=round(time.perf_counter() - t1, 1), gen_rounds=rounds,
                    rows_drawn=drawn, acceptance=round(len(out) / drawn, 4) if drawn else None,
                    peak_gpu_gb_gen=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    n_returned=len(out))
        out[out_cols].to_csv(a.out_csv, index=False)

        # Sample-batch probe: same fitted model, generate --probe-rows rows at each k.
        # One DP fit serves the whole sweep. A zero-row round (generate() swallows an
        # OOM and returns nothing) is recorded and the sweep continues.
        if a.probe_batches:
            info["batch_probe"] = []
            for k in a.probe_batches:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t2 = time.perf_counter()
                rec = {"k": k, "rows_requested": a.probe_rows}
                try:
                    rows, drawn_k, rounds_k = _sample_rows(
                        tag=f"probe{k}", n=a.probe_rows, k=k, gen_seed=a.gen_seed, **gen_kw)
                    rec.update(ok=True, rows=len(rows), rows_drawn=drawn_k, gen_rounds=rounds_k)
                except (HardGenerationFailure, NoValidRows) as e:
                    rows = ()
                    rec.update(ok=False, rows=0, note=str(e)[:200])
                secs = time.perf_counter() - t2
                rec.update(seconds=round(secs, 1), peak_gpu_gb=round(
                    torch.cuda.max_memory_allocated() / 2**30, 2),
                    rows_per_s=round(len(rows) / secs, 2))
                info["batch_probe"].append(rec)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if a.info_json:
            Path(a.info_json).write_text(json.dumps(info, indent=2))

    if info["n_returned"] < a.n_samples:
        print(f"only {info['n_returned']}/{a.n_samples} rows after {MAX_GEN_ROUNDS} rounds",
              file=sys.stderr)
        return EXIT_SHORTFALL
    print(json.dumps(info))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("stage1", help="non-DP fine-tune on Airline (once)")
    s1.add_argument("--train-csv", required=True)
    s1.add_argument("--out-dir", default=str(STAGE1_DIR))
    s1.add_argument("--seed", type=int, default=1000)   # GPT2.sh's SEED

    s2 = sub.add_parser("fit-sample", help="DP Stage 2 on a private CSV, then sample")
    s2.add_argument("--train-csv", required=True)
    s2.add_argument("--out-csv", required=True)
    s2.add_argument("--info-json", default=None)
    s2.add_argument("--n-samples", type=int, required=True)
    s2.add_argument("--seed", type=int, required=True, help="training seed")
    s2.add_argument("--gen-seed", type=int, required=True, help="generation seed")
    s2.add_argument("--stage1-dir", default=str(STAGE1_DIR))
    s2.add_argument("--lr", type=float, default=None,
                    help="Stage 2 learning rate, for PROBING only (default: the LR constant). "
                         "generate_dp2stage.py and the TAPAS wrapper never pass it, so whatever "
                         "wins must be written into LR or the runs will not match the probe.")
    s2.add_argument("--sample-batch", type=int, default=SAMPLE_BATCH,
                    help="rows drawn per model.generate() call (sampling only)")
    s2.add_argument("--probe-batches", type=int, nargs="+", default=None,
                    help="after the main run, generate --probe-rows rows at each of these "
                         "batch sizes on the same fitted model (timing + peak GPU memory)")
    s2.add_argument("--probe-rows", type=int, default=2000,
                    help="rows per --probe-batches setting; must be >= the largest batch "
                         "size or that batch never fills")
    s2.add_argument("--scratch-root", default=str(REPO_ROOT / "sdg"),
                    help="parent dir for the per-call temp dir (deleted on exit)")

    a = p.parse_args()
    return cmd_stage1(a) if a.cmd == "stage1" else cmd_fit_sample(a)


if __name__ == "__main__":
    sys.exit(main())
