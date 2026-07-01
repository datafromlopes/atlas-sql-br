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
#  Text2SQL fine-tuning pipeline.
#
#  Study goal: measure whether the DATASET adds value to the model (whether it
#  learns), comparing performance BEFORE (baseline, no training) vs AFTER
#  (trained) on the curated test set `train == 0` (holdout with unseen
#  entities/columns).
#
#  Supports (decoder-only models only — Llama / Qwen):
#   - architecture: "llama" | "qwen"  (decoder-only / causal LM)
#   - train_method: "none" (baseline, eval only) | "lora" | "ia3" | "full" | "scratch"
#
#  This script ONLY trains: it is driven by eval_loss on an internal validation
#  split (early stopping + best-checkpoint selection) and saves the best model.
#  Generation and execution scoring on the holdout (train == 0) are done by the
#  separate harness: generate_sql_preds.py + score_predictions.py.
# =============================================================================
import os
import gc
import json
import sys
import hashlib
import subprocess
import argparse
from urllib.parse import quote_plus

import yaml
import torch
import polars as pl
import psutil

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from peft import get_peft_model, LoraConfig, IA3Config, TaskType
from huggingface_hub import login
import mlflow

# =============================================================================
# PATH SETUP
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import PROJECT_PATH, PROJECT_NAME, DATASET_PATH, DATASET_NAME  # noqa: E402
from utils.utils import Logger  # noqa: E402

# Dataset column names.
Q_COL, SQL_COL, LEVEL_COL, TRAIN_COL = "question", "sql_code", "level", "train"

# =============================================================================
# LOGGING
# =============================================================================
logger = Logger("finetuning").setup_logging()

# =============================================================================
# DEVICE  (cheap to compute; no network — fine to run at import time)
# =============================================================================
def setup_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("Device: MPS (Apple Silicon) — unified memory")
        return torch.device("mps")
    logger.warning("Device: CPU (training will be slow)")
    return torch.device("cpu")

DEVICE = setup_device()
IS_MPS = DEVICE.type == "mps"
IS_CUDA = DEVICE.type == "cuda"


def resolve_precision(fp16: bool, bf16: bool, tf32: bool):
    """Align precision flags with the actual device (fixes config inconsistencies)."""
    if fp16 and bf16:
        raise ValueError("fp16 and bf16 cannot both be True.")
    if not IS_CUDA:
        # tf32 is CUDA-only (Ampere+); fp16/bf16 via Trainer on MPS/CPU is
        # unstable/ignored — disabling and training in fp32 is the safe choice.
        if tf32 or fp16 or bf16:
            logger.warning("Non-CUDA device: forcing fp16/bf16/tf32 = False (fp32).")
        return False, False, False
    return fp16, bf16, tf32

# =============================================================================
# MLFLOW HELPERS
# =============================================================================
def build_mlflow_uri() -> str:
    """MLflow URI with local fallback and PASSWORD redacted in the log."""
    pg_user = os.environ.get("PG_USER")
    pg_pass_raw = os.environ.get("PG_PASS", "")
    if pg_user and pg_pass_raw:
        uri = f"postgresql://{pg_user}:{quote_plus(pg_pass_raw)}@localhost:5432/mlflow"
        redacted = uri.replace(quote_plus(pg_pass_raw), "****")
    else:
        uri = f"file://{PROJECT_PATH}/mlruns"
        redacted = uri
        logger.warning("PG_USER/PG_PASS not set — using local MLflow at ./mlruns")
    logger.info(f"MLflow tracking URI: {redacted}")
    return uri


def get_git_info() -> dict:
    info = {}
    for key, cmd in (("git_commit", ["git", "rev-parse", "--short", "HEAD"]),
                     ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            info[key] = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            info[key] = "unavailable"
    return info


def get_dataset_fingerprint(df: pl.DataFrame, n_samples: int = 5) -> dict:
    md5 = hashlib.md5(df.write_csv().encode()).hexdigest()
    sample = df.sample(n=min(n_samples, len(df)), seed=42).to_dicts()
    return {"dataset_md5": md5, "dataset_rows": len(df), "dataset_cols": len(df.columns),
            "dataset_sample": json.dumps(sample, ensure_ascii=False, default=str)}


def get_model_summary(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable,
            "params_frozen": total - trainable,
            "params_trainable_pct": round(100 * trainable / total, 2) if total else 0}


def log_system_metrics(step=None):
    try:
        m = {"system/cpu_percent": psutil.cpu_percent(interval=0.1),
             "system/ram_used_gb": psutil.virtual_memory().used / 1e9,
             "system/ram_percent": psutil.virtual_memory().percent}
        if IS_CUDA:
            m["system/gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        if IS_MPS:
            m["system/mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        mlflow.log_metrics(m, step=step)
    except Exception as e:
        logger.warning(f"Failed to log system metrics: {e}")


class MLflowRealtimeCallback(transformers.TrainerCallback):
    """Log training metrics to MLflow. Visual progress is shown by the Trainer's
    native tqdm bar (kept via disable_tqdm=False)."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        m = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if m:
            mlflow.log_metrics(m, step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        log_system_metrics(step=state.global_step)


def mlflow_log_params(params: dict):
    safe = {}
    for k, v in params.items():
        try:
            json.dumps(v); safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    mlflow.log_params(safe)


def mlflow_end_run(status="FINISHED"):
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except Exception as e:
        logger.error(f"MLflow end_run error: {e}")

# =============================================================================
# DATASETS
# =============================================================================
def _build_prompt(q: str) -> str:
    # Single prompt format, identical in training and generation (and it must
    # match generate_sql_preds.py). Ends with "SQL:" so a decoder-only model
    # knows the continuation is the query.
    return f"Pergunta: {q}\nSQL:"


class CausalDataset(torch.utils.data.Dataset):
    """Decoder-only (Llama / Qwen): prompt+SQL concatenated; loss ONLY on the SQL span."""
    def __init__(self, df: pl.DataFrame, tokenizer, max_length: int):
        self.samples = []
        eos = tokenizer.eos_token or ""
        for q, s in zip(df[Q_COL].to_list(), df[SQL_COL].to_list()):
            prompt = _build_prompt(q)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + " " + s + eos,
                                 add_special_tokens=False,
                                 truncation=True, max_length=max_length)["input_ids"]
            labels = list(full_ids)
            # FIX (bug #1): mask the prompt so loss is not computed on the question.
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            self.samples.append((full_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": torch.tensor(labels, dtype=torch.long)}


class CausalCollator:
    """Dynamic right padding for causal LM; pad_to_multiple_of=8 (Tensor Cores)."""
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = ((max_len + self.mult - 1) // self.mult) * self.mult
        pad_id = self.tok.pad_token_id or 0
        ids, mask, lbl = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            ids.append(torch.cat([f["input_ids"], torch.full((n,), pad_id, dtype=torch.long)]))
            mask.append(torch.cat([f["attention_mask"], torch.zeros(n, dtype=torch.long)]))
            lbl.append(torch.cat([f["labels"], torch.full((n,), -100, dtype=torch.long)]))
        return {"input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "labels": torch.stack(lbl)}


def stratified_val_split(df: pl.DataFrame, val_fraction: float, seed: int):
    """Split an internal validation set (early stopping) from training, stratified
    by level. Does NOT touch the train==0 holdout — that stays untouched for the
    final test."""
    train_parts, val_parts = [], []
    for _, sub in df.partition_by(LEVEL_COL, as_dict=True).items():
        sub = sub.sample(fraction=1.0, shuffle=True, seed=seed)
        n_val = max(1, int(len(sub) * val_fraction))
        val_parts.append(sub[:n_val])
        train_parts.append(sub[n_val:])
    return pl.concat(train_parts), pl.concat(val_parts)

# =============================================================================
# MODEL FACTORIES
# =============================================================================
def load_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"  # training; generation switches to 'left' locally
    return tok


def _peft_targets():
    """target_modules for decoder-only (Llama / Qwen)."""
    return {"lora": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "ia3": ["k_proj", "v_proj", "down_proj"], "ia3_ff": ["down_proj"]}


def build_model(base_model_name: str, train_method: str, cfg: dict, dtype):
    config = AutoConfig.from_pretrained(base_model_name)
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError(
            f"{base_model_name} is encoder-decoder; this pipeline supports only "
            f"decoder-only models (Llama / Qwen).")
    ModelCls = AutoModelForCausalLM
    tgt = _peft_targets()

    if train_method == "scratch":
        # FIX (bug #2/#3): same architecture/vocab as base, random weights.
        model = ModelCls.from_config(config)
        logger.info("Model initialized FROM SCRATCH (same architecture as base, random weights).")
    else:
        model = ModelCls.from_pretrained(base_model_name, torch_dtype=dtype)
        if train_method == "lora":
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=tgt["lora"],
                r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                inference_mode=False))
            model.print_trainable_parameters()
        elif train_method == "ia3":
            model = get_peft_model(model, IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=tgt["ia3"], feedforward_modules=tgt["ia3_ff"],
                inference_mode=False))
            model.print_trainable_parameters()
        # "full" and "none": model as-is (full trains everything; none does not train).

    # FIX (bug #6): gradient checkpointing + PEFT needs this so grads can flow.
    if cfg.get("gradient_checkpoints") and train_method != "none":
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    return model.to(DEVICE), config

# =============================================================================
# TRAIN
# =============================================================================
def train_model(model, train_ds, val_ds, tokenizer, collator, ta: dict):
    args = TrainingArguments(
        output_dir=ta["checkpoints_dir"],
        num_train_epochs=ta["epochs"],
        per_device_train_batch_size=ta["batch_size"],
        per_device_eval_batch_size=ta["batch_size"],
        gradient_accumulation_steps=ta["grad_accum"],
        learning_rate=ta["learning_rate"],          # FIX (bug #5)
        lr_scheduler_type=ta["lr_scheduler_type"],  # FIX (bug #5)
        warmup_ratio=ta["warmup_ratio"],
        weight_decay=ta["weight_decay"],
        logging_dir=ta["logs_dir"],
        logging_steps=ta["logging_steps"],
        logging_strategy=ta["logging_strategy"],
        eval_strategy=ta["evaluation_strategy"],
        save_strategy=ta["save_strategy"],
        save_total_limit=2,
        fp16=ta["fp16"], bf16=ta["bf16"], tf32=ta["tf32"],
        dataloader_num_workers=ta["dataloader_num_workers"],
        dataloader_pin_memory=ta["dataloader_pin_memory"],
        optim=ta["optim"],
        gradient_checkpointing=ta["gradient_checkpoints"],
        load_best_model_at_end=ta["load_best_model"],
        metric_for_best_model=ta["metric_for_best_model"],
        greater_is_better=ta["greater_is_better"],
        report_to="none",
        disable_tqdm=False,  # keep the native tqdm progress bar
        run_name=ta["run_name"],
        seed=ta["seed"],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(ta["early_stopping_patience"],
                                         ta["early_stopping_threshold"]),
                   MLflowRealtimeCallback()],
    )
    logger.info(f"Active device: {next(trainer.model.parameters()).device}")
    trainer.train()
    return trainer

# =============================================================================
# CONFIG / CLI
# =============================================================================
def load_config(experiment_version: int):
    path = PROJECT_PATH / "experiments" / f"exp-v{experiment_version}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f), path


def load_dataset() -> pl.DataFrame:
    df = pl.scan_parquet(f"{DATASET_PATH}/{DATASET_NAME}").collect()
    # Ensure the required columns are present.
    missing = [c for c in (Q_COL, SQL_COL, TRAIN_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}. Available: {df.columns}")
    return df

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Text2SQL training")
    parser.add_argument("--experiment_version", type=int, default=0)
    exp = parser.parse_args().experiment_version

    cfg, cfg_path = load_config(exp)
    EXP = f"v{exp}"

    # Baseline experiments (train_method == "none") have nothing to train. The
    # base model's SQL generation is produced by generate_sql_preds.py (which
    # loads the base model itself). So skip here BEFORE any HF download or model
    # load — no point spending that time/memory in the training entrypoint.
    if str(cfg.get("train_method", "none")).lower() == "none":
        logger.banner(f"START | exp={EXP}", width=80)
        logger.info(f"model={cfg.get('base_model')} | method=none | device={DEVICE}")
        logger.info("Baseline (train_method=none): nothing to train; base-model "
                    "generation/eval is handled by generate_sql_preds.py. "
                    "Skipping model load to save time.")
        logger.banner(f"SKIPPED (baseline) | exp={EXP}", width=80)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in the environment (export HF_TOKEN='...').")
    login(token=token)

    set_seed(cfg.get("seed", 42))

    mlflow.set_tracking_uri(build_mlflow_uri())
    mlflow.set_experiment(PROJECT_NAME)

    BASE_MODEL = cfg["base_model"]
    METHOD = cfg["train_method"]
    fp16, bf16, tf32 = resolve_precision(cfg["fp16"], cfg["bf16"], cfg["tf32"])
    dtype = torch.float32 if IS_MPS else (torch.bfloat16 if bf16 else torch.float32)
    run_id = f"{cfg['model_name']}_{METHOD}_{EXP}"

    git = get_git_info()
    logger.banner(f"START | exp={EXP}", width=80)
    logger.info(f"model={BASE_MODEL} | method={METHOD} | device={DEVICE}")
    logger.info(f"git: {git['git_branch']} @ {git['git_commit']}")

    # Silence transformers' INFO noise (config dumps, "loading/saving ..."),
    # while keeping the native tqdm progress bar (disable_tqdm=False).
    transformers.utils.logging.set_verbosity_error()

    tokenizer = load_tokenizer(BASE_MODEL)
    model = trainer = None
    train_ds = val_ds = None

    mlflow_end_run()
    try:
        # ── Data: train = train==1 (with internal val); test = train==0 ──────
        df = load_dataset()
        train_pool = df.filter(pl.col(TRAIN_COL) == 1)
        test_df = df.filter(pl.col(TRAIN_COL) == 0)
        logger.info(f"Data: {len(train_pool)} train(+val) | {len(test_df)} test (holdout)")

        mlflow.start_run(run_name=run_id, tags={
            "experiment_version": EXP, "model_name": cfg["model_name"],
            "train_method": METHOD, "architecture": cfg["architecture"],
            "device": str(DEVICE), **git})
        mlflow_log_params(get_dataset_fingerprint(df))
        mlflow.log_artifact(str(cfg_path), artifact_path="config")
        mlflow_log_params({
            "base_model": BASE_MODEL, "train_method": METHOD, "architecture": cfg["architecture"],
            "device": str(DEVICE),
            "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
            "grad_accum": cfg["grad_accum"], "learning_rate": cfg.get("learning_rate"),
            "lr_scheduler": cfg.get("lr_scheduler_type"), "warmup_ratio": cfg.get("warmup_ratio"),
            "lora_r": cfg.get("lora_r"), "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"), "seed": cfg.get("seed", 42),
            "fp16": fp16, "bf16": bf16, "tf32": tf32, "max_length": cfg["max_length"],
            "n_train_pool": len(train_pool), "n_test_holdout": len(test_df),
            "transformers": transformers.__version__, "torch": torch.__version__,
        })

        # ── Model ─────────────────────────────────────────────────────────────
        model, _ = build_model(BASE_MODEL, METHOD, cfg, dtype)
        mlflow_log_params(get_model_summary(model))

        # train_method == "none" exits early in main(); only trained methods reach
        # here. Evaluation during training is the Trainer's eval_loss on the
        # internal validation split — it drives early stopping and best-checkpoint
        # selection. Generation/execution scoring on the holdout is done separately
        # by generate_sql_preds.py + score_predictions.py.
        if METHOD != "none":
            # ── Internal val split (does NOT touch the holdout) ───────────────
            tr_df, vl_df = stratified_val_split(train_pool, cfg.get("val_fraction", 0.1),
                                                cfg.get("seed", 42))
            DS = CausalDataset
            train_ds = DS(tr_df, tokenizer, cfg["max_length"])
            val_ds = DS(vl_df, tokenizer, cfg["max_length"])
            collator = CausalCollator(tokenizer)

            out_dir = f"{PROJECT_PATH}/models/experiments/{EXP}"
            logs_dir = f"{out_dir}/logs"
            os.makedirs(logs_dir, exist_ok=True)

            # Guard: load_best_model_at_end requires compatible eval/save.
            if cfg["load_best_model"] and cfg["evaluation_strategy"] != cfg["save_strategy"]:
                raise ValueError("load_best_model=True requires evaluation_strategy == save_strategy.")

            ta = {"checkpoints_dir": f"{out_dir}/checkpoints", "logs_dir": logs_dir,
                  "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
                  "grad_accum": cfg["grad_accum"], "learning_rate": cfg["learning_rate"],
                  "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
                  "warmup_ratio": cfg.get("warmup_ratio", 0.06),
                  "weight_decay": cfg["weight_decay"], "logging_steps": cfg["logging_steps"],
                  "logging_strategy": cfg["logging_strategy"],
                  "evaluation_strategy": cfg["evaluation_strategy"],
                  "save_strategy": cfg["save_strategy"], "fp16": fp16, "bf16": bf16, "tf32": tf32,
                  "dataloader_num_workers": 0 if not IS_CUDA else cfg["dataloader_num_workers"],
                  "dataloader_pin_memory": False if not IS_CUDA else cfg["dataloader_pin_memory"],
                  "optim": cfg["optim"], "gradient_checkpoints": cfg["gradient_checkpoints"],
                  "load_best_model": cfg["load_best_model"],
                  "metric_for_best_model": cfg["metric_for_best_model"],
                  "greater_is_better": cfg["greater_is_better"],
                  "early_stopping_patience": cfg["early_stopping_patience"],
                  "early_stopping_threshold": cfg["early_stopping_threshold"],
                  "run_name": run_id, "seed": cfg.get("seed", 42)}

            logger.info("Starting training…")
            trainer = train_model(model, train_ds, val_ds, tokenizer, collator, ta)

            # ── Artifacts (best checkpoint, restored by load_best_model_at_end) ──
            art = f"{out_dir}/artifacts"
            os.makedirs(art, exist_ok=True)
            trainer.model.save_pretrained(art)
            tokenizer.save_pretrained(art)
            mlflow.log_artifacts(art, artifact_path="model")
            logger.info(f"✔ {run_id} complete. Artifacts at: {art}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        mlflow_end_run(status="KILLED")
        raise
    except Exception as e:
        logger.error(f"Error in {run_id}: {e}", exc_info=True)
        mlflow_end_run(status="FAILED")
        raise  # FIX (bug #12): do NOT swallow the failure — propagate to exit code != 0
    finally:
        try:
            if trainer is not None:
                for attr in ("model", "train_dataset", "eval_dataset", "data_collator",
                             "optimizer", "lr_scheduler", "callback_handler"):
                    setattr(trainer, attr, None)
                del trainer
            if model is not None:
                del model
            if train_ds is not None:
                del train_ds
            if val_ds is not None:
                del val_ds
            gc.collect()
            if IS_CUDA:
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            if IS_MPS:
                torch.mps.empty_cache()
        except Exception as ce:
            logger.warning(f"Cleanup error: {ce}")
        mlflow_end_run()

    logger.banner(f"COMPLETE | exp={EXP}", width=80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        sys.exit(1)