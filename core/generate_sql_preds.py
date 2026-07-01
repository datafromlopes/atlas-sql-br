# coding=utf-8
# Copyright (C) 2026  Diego Lopes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#     https://www.gnu.org/licenses/gpl-3.0.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# =============================================================================
#  Generate SQL predictions on the holdout (train == 0) for ONE experiment,
#  with the BASE model (no training) and the FINE-TUNED model, for comparison.
#
#  Run like the training entrypoint:
#     uv run python core/generate_sql_preds.py --experiment_version 1
#
#  It reads exp-v{N}.yaml to resolve the base model (and generation params),
#  derives the fine-tuned artifact path from the experiment, and writes one
#  file per experiment to {PREDS_DIR}/predictions_v{N}.json.
#
#  Key points (aligned with finetuning_strategies.py):
#   - Decoder-only models ONLY (Llama / Qwen); encoder-decoder is rejected.
#   - Prompt and generation read the SAME dataset the SAME way as training, and
#     the prompt is IDENTICAL to finetuning_strategies._build_prompt — otherwise
#     base vs fine-tuned predictions are not comparable.
#   - Filters train == 0 (gold `sql_code` goes along for scoring on Postgres).
#   - Baseline experiments (train_method=none) produced NO artifact, so the model
#     is loaded from the Hugging Face hub (base_model) and NEVER from artifacts;
#     the fine-tuned predictions equal the base ones (delta = 0 by construction).
#
#  Output: list of {sql_validation_id, nivel, question, sql_code_gold,
#  sql_code_base, sql_code_finetuned} — ready for EXECUTION scoring on Postgres.
# =============================================================================
from __future__ import annotations

import gc
import re
import sys
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import yaml
import torch
import polars as pl
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# =============================================================================
# PATH SETUP
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import (
    PROJECT_PATH,
    PROJECT_NAME,
    PREDS_DIR,
    DATASET_PATH,
    DATASET_VALIDATION_NAME
)
from utils.utils import Logger  # noqa: E402

logger = Logger("predict").setup_logging()

# Predictions output dir comes from utils (single source of truth); one file
# per experiment lives here as predictions_v{N}.json.
PREDS_DIR = Path(PREDS_DIR)

# ── Dataset column names (same as the training pipeline) ─────────────────────
ID_COL, Q_COL, SQL_COL, LEVEL_COL, TRAIN_COL = "id", "question", "sql_code", "level", "train"


@dataclass
class GenContext:
    """Everything resolved at runtime from the experiment YAML and the device.

    Generation params are read from the experiment config so they stay identical
    to the values used during training.
    """
    base_model: str
    artifact_path: Path
    device: torch.device
    dtype: torch.dtype
    max_new_tokens: int
    batch_size: int
    input_max_length: int

# ═══════════════════════════════════════════════════════════════════════════
# Config / device
# ═══════════════════════════════════════════════════════════════════════════
def load_config(experiment_version: int):
    path = Path(PROJECT_PATH) / "experiments" / f"exp-v{experiment_version}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), path


def setup_device():
    if torch.cuda.is_available():
        logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        logger.info("Device: MPS (Apple Silicon) -> float32")
        return torch.device("mps"), torch.float32
    logger.info("Device: CPU -> float32")
    return torch.device("cpu"), torch.float32

# ═══════════════════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════════════════
def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def unload_model(model) -> None:
    del model
    clear_memory()
    logger.info("Model unloaded, memory freed.")

# ═══════════════════════════════════════════════════════════════════════════
# Tokenizer / models  (decoder-only: Llama / Qwen)
# ═══════════════════════════════════════════════════════════════════════════
def load_tokenizer(ctx: GenContext):
    logger.info(f"Loading tokenizer: {ctx.base_model}")
    tok = AutoTokenizer.from_pretrained(ctx.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_base_model(ctx: GenContext):
    # Always loaded from the Hugging Face hub (no local artifact involved).
    clear_memory()
    logger.info(f"Loading base model from Hugging Face hub: {ctx.base_model}")
    model = AutoModelForCausalLM.from_pretrained(ctx.base_model, torch_dtype=ctx.dtype)
    return model.to(ctx.device).eval()


def load_finetuned_model(ctx: GenContext):
    """Load the trained model on top of a fresh base instance.
    Supports a PEFT adapter (LoRA/IA3) and a fully saved full fine-tuned model."""
    clear_memory()
    art = Path(ctx.artifact_path)
    if not art.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {art}")

    base = AutoModelForCausalLM.from_pretrained(ctx.base_model, torch_dtype=ctx.dtype)
    if (art / "adapter_config.json").exists():
        logger.info(f"Loading PEFT adapter from: {art}")
        model = PeftModel.from_pretrained(base, str(art))
    else:
        logger.info(f"Loading full fine-tuned weights from: {art}")
        del base
        model = AutoModelForCausalLM.from_pretrained(str(art), torch_dtype=ctx.dtype)
    return model.to(ctx.device).eval()

# ═══════════════════════════════════════════════════════════════════════════
# Prompt + generation (IDENTICAL to finetuning_strategies._build_prompt)
# ═══════════════════════════════════════════════════════════════════════════
def build_input(question: str) -> str:
    # Must match finetuning_strategies._build_prompt exactly (train/inference parity).
    return f"Pergunta: {question}\nSQL:"


def _clean_sql(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


@torch.no_grad()
def predict_batch(tokenizer, model, questions: list[str], ctx: GenContext) -> list[str]:
    texts = [build_input(q) for q in questions]

    side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # causal generation requires left padding

    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=ctx.input_max_length).to(ctx.device)

    out = model.generate(
        **enc,
        max_new_tokens=ctx.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen = out[:, enc["input_ids"].shape[1]:]   # only the generated continuation
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

    tokenizer.padding_side = side
    return [_clean_sql(d) for d in decoded]


def run_pass(tokenizer, model, validation_data: list[dict], label: str, ctx: GenContext) -> dict:
    total = len(validation_data)
    predictions: dict = {}

    logger.banner(f"{label}  ({total} questions)", width=60)
    for start in range(0, total, ctx.batch_size):
        batch = validation_data[start:start + ctx.batch_size]
        ids   = [it[ID_COL] for it in batch]
        qs    = [it[Q_COL] for it in batch]
        try:
            for vid, sql in zip(ids, predict_batch(tokenizer, model, qs, ctx)):
                predictions[vid] = sql
            logger.info(f"[{min(start + ctx.batch_size, total):>4}/{total}] OK")
        except Exception as exc:
            for vid in ids:
                predictions[vid] = ""
            logger.error(f"[{start:>4}/{total}] ERROR {exc}", exc_info=True)
    return predictions

# ═══════════════════════════════════════════════════════════════════════════
# Load and normalize the validation set (train == 0)  — same read as training
# ═══════════════════════════════════════════════════════════════════════════
def load_validation_data() -> list[dict]:
    logger.info(f"Reading dataset: {DATASET_VALIDATION_NAME}")
    df = pl.scan_parquet(f"{DATASET_PATH}/{DATASET_VALIDATION_NAME}").collect()

    # normalize the question column name coming from the original CSV
    if Q_COL not in df.columns and "pergunta" in df.columns:
        df = df.rename({"pergunta": Q_COL})

    # filter the holdout
    if TRAIN_COL in df.columns:
        df = df.filter(pl.col(TRAIN_COL) == 0)
        logger.info(f"Filtered train == 0: {len(df)} validation examples.")
    else:
        logger.warning("Column 'train' missing — using all rows as validation.")

    cols = [c for c in (ID_COL, Q_COL, SQL_COL, LEVEL_COL) if c in df.columns]
    data = df.select(cols).to_dicts()
    # ensure expected keys
    for d in data:
        d.setdefault(SQL_COL, "")
        d.setdefault(LEVEL_COL, "")
    return data

# ═══════════════════════════════════════════════════════════════════════════
# Quality preview (normalized exact-match) — proxy; gold is execution
# ═══════════════════════════════════════════════════════════════════════════
def _norm_sql(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s[:-1].strip() if s.endswith(";") else s


def print_em_summary(results: list[dict]):
    def em(key):
        hits = [_norm_sql(r[key]) == _norm_sql(r["sql_code_gold"]) for r in results]
        return sum(hits) / len(hits) if hits else 0.0

    logger.banner("Exact-match preview (proxy — gold metric is execution)", width=60)
    logger.info(f"  base      : {em('sql_code_base'):.4f}")
    logger.info(f"  fine-tuned: {em('sql_code_finetuned'):.4f}")
    logger.info(f"  delta     : {em('sql_code_finetuned') - em('sql_code_base'):+.4f}")

    levels = sorted({r.get("nivel", "") for r in results})
    for lvl in levels:
        sub = [r for r in results if r.get("nivel", "") == lvl]
        if not sub:
            continue
        b = sum(_norm_sql(r["sql_code_base"]) == _norm_sql(r["sql_code_gold"]) for r in sub) / len(sub)
        f = sum(_norm_sql(r["sql_code_finetuned"]) == _norm_sql(r["sql_code_gold"]) for r in sub) / len(sub)
        logger.info(f"    [{lvl or 'NA':<14}] base={b:.3f}  ft={f:.3f}  Δ={f - b:+.3f}  (n={len(sub)})")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate base vs fine-tuned SQL predictions for one experiment.")
    parser.add_argument("--experiment_version", type=int, default=0)
    exp = parser.parse_args().experiment_version

    cfg, cfg_path = load_config(exp)
    logger.banner(f"PREDICT | exp=v{exp} | {PROJECT_NAME}", width=60)
    logger.info(f"config: {cfg_path}")

    base_model = cfg["base_model"]
    train_method = str(cfg.get("train_method", "none")).lower()
    device, dtype = setup_device()

    # Decoder-only only (Llama / Qwen); reject encoder-decoder to match training.
    config = AutoConfig.from_pretrained(base_model)
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError(
            f"{base_model} is encoder-decoder; this pipeline supports only "
            f"decoder-only models (Llama / Qwen).")
    logger.info(f"Model: {base_model} | decoder-only | method={train_method}")

    # Artifact path mirrors finetuning_strategies.py: <PROJECT_PATH>/models/experiments/v{N}/artifacts
    artifact_path = Path(PROJECT_PATH) / "models" / "experiments" / f"v{exp}" / "artifacts"

    ctx = GenContext(
        base_model=base_model,
        artifact_path=artifact_path,
        device=device,
        dtype=dtype,
        max_new_tokens=int(cfg.get("gen_max_new_tokens", 512)),
        batch_size=max(1, int(cfg.get("gen_batch_size", cfg.get("batch_size", 8)))),
        input_max_length=int(cfg.get("max_length", 1024)),
    )

    validation_data = load_validation_data()
    logger.info(f"{len(validation_data)} questions loaded.")

    tokenizer = load_tokenizer(ctx)

    # ── Pass 1: base model (no training) ─────────────────────────────────────
    base_model_obj = load_base_model(ctx)
    base_preds = run_pass(tokenizer, base_model_obj, validation_data, "Pass 1 - base model", ctx)
    unload_model(base_model_obj)

    # ── Pass 2: fine-tuned model ─────────────────────────────────────────────
    # train_method=none means nothing was trained/saved: reuse the Hugging Face
    # hub base model (already produced in Pass 1) and do NOT touch any artifact.
    if train_method == "none":
        logger.info("Baseline experiment (train_method=none): no artifact was produced; "
                    "using the Hugging Face hub base model. fine-tuned = base (delta = 0).")
        finetuned_preds = dict(base_preds)
    else:
        finetuned_model = load_finetuned_model(ctx)
        finetuned_preds = run_pass(tokenizer, finetuned_model, validation_data,
                                   "Pass 2 - fine-tuned model", ctx)
        unload_model(finetuned_model)

    # ── Merge + save (gold included, ready for execution scoring) ────────────
    results = [
        {
            "sql_validation_id":  it[ID_COL],
            "nivel":              it.get(LEVEL_COL, ""),
            "question":           it[Q_COL],
            "sql_code_gold":      it.get(SQL_COL, ""),
            "sql_code_base":      base_preds.get(it[ID_COL], ""),
            "sql_code_finetuned": finetuned_preds.get(it[ID_COL], ""),
        }
        for it in validation_data
    ]

    # One predictions file per experiment: predictions_v{N}.json inside PREDS_DIR.
    out_path = PREDS_DIR / Path(f"predictions_v{exp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. {len(results)} predictions saved -> {out_path.resolve()}")
    print_em_summary(results)


if __name__ == "__main__":
    main()