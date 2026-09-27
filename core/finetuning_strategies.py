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

from utils import PROJECT_PATH, PROJECT_NAME  # noqa: E402
from utils.utils import Logger  # noqa: E402
from utils.experiment import (  # noqa: E402
    Q_COL, SQL_COL, LEVEL_COL, TRAIN_COL,
    load_experiment_config, experiment_dir, build_prompt, prompt_template,
    load_train_pool, load_holdout, dataset_files, dataset_version, file_md5,
    build_mlflow_uri as _shared_mlflow_uri, save_run_id,
)

# The experiment config of the current run (set in main; read by the prompt
# builder so training and generation share one template).
CFG: dict = {}

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
    """MLFLOW_TRACKING_URI > Postgres (PG_USER/PG_PASS) > file store ./mlruns."""
    return _shared_mlflow_uri(logger)


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
    # Single prompt format, identical in training and generation: both stages
    # call utils.experiment.build_prompt with the experiment's `prompt_template`
    # (default "Pergunta: {question}\nSQL:"; the paper uses
    # "Traduza para SQL: {question}"). No schema is injected.
    return build_prompt(CFG, q)


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


def _peft_targets(cfg: dict | None = None):
    """target_modules for decoder-only (Llama / Qwen). `lora_target_modules` in
    the experiment YAML overrides the LoRA default (the paper uses q_proj and
    v_proj only)."""
    cfg = cfg or {}
    lora = list(cfg.get("lora_target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"])
    return {"lora": lora,
            "ia3": ["k_proj", "v_proj", "down_proj"], "ia3_ff": ["down_proj"]}


def build_model(base_model_name: str, train_method: str, cfg: dict, dtype):
    config = AutoConfig.from_pretrained(base_model_name)
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError(
            f"{base_model_name} is encoder-decoder; this pipeline supports only "
            f"decoder-only models (Llama / Qwen).")
    ModelCls = AutoModelForCausalLM
    tgt = _peft_targets(cfg)

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
        # warmup_steps (absolute, as in the paper: 612) takes precedence over
        # warmup_ratio when it is set to a positive value.
        warmup_steps=int(ta.get("warmup_steps") or 0),
        warmup_ratio=(0.0 if int(ta.get("warmup_steps") or 0) > 0 else ta["warmup_ratio"]),
        weight_decay=ta["weight_decay"],
        logging_dir=ta["logs_dir"],
        logging_steps=ta["logging_steps"],
        logging_strategy=ta["logging_strategy"],
        eval_strategy=ta["evaluation_strategy"],
        save_strategy=ta["save_strategy"],
        save_total_limit=None,
        fp16=ta["fp16"], bf16=ta["bf16"], tf32=ta["tf32"],
        dataloader_num_workers=ta["dataloader_num_workers"],
        dataloader_pin_memory=ta["dataloader_pin_memory"],
        optim=ta["optim"],
        gradient_checkpointing=ta["gradient_checkpoints"],
        load_best_model_at_end=ta["load_best_model"],
        metric_for_best_model=ta["metric_for_best_model"],
        greater_is_better=ta["greater_is_better"],
        report_to="none",
        # Evaluate before the first optimization step, so the validation loss of
        # the untrained (base) model is logged at step 0 and the training curve
        # can be drawn against that reference.
        eval_on_start=bool(ta.get("eval_on_start", True)),
        disable_tqdm=False,  # keep the native tqdm progress bar
        run_name=ta["run_name"],
        seed=ta["seed"],
    )
    callbacks = [MLflowRealtimeCallback()]
    # early_stopping_patience <= 0 disables early stopping: the run then goes
    # through every epoch and load_best_model_at_end keeps the best checkpoint.
    if int(ta.get("early_stopping_patience") or 0) > 0:
        callbacks.insert(0, EarlyStoppingCallback(int(ta["early_stopping_patience"]),
                                                  float(ta["early_stopping_threshold"])))
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, callbacks=callbacks,
    )
    logger.info(f"Active device: {next(trainer.model.parameters()).device}")
    trainer.train()
    return trainer

# =============================================================================
# CONFIG / CLI
# =============================================================================
def load_config(experiment_version: int):
    return load_experiment_config(experiment_version)


# =============================================================================
# HUGGING FACE HUB
# =============================================================================
def push_to_hub(cfg: dict, art_dir: str, run_id: str, metrics: dict, token: str) -> str | None:
    """Upload the trained adapter (or full weights), the tokenizer and a model
    card to the Hugging Face Hub. Returns the repository URL."""
    from huggingface_hub import HfApi
    repo_id = cfg.get("hf_repo_id")
    if not cfg.get("hf_push") or not repo_id:
        logger.info("hf_push disabled or hf_repo_id missing — skipping Hub upload.")
        return None
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=bool(cfg.get("hf_private", False)),
                    exist_ok=True)
    card = f"""---
base_model: {cfg["base_model"]}
library_name: peft
language: [pt]
license: mit
tags: [text-to-sql, geospatial, postgis, lora, brazilian-portuguese, atlas-sql-br]
datasets: [datafromlopes/atlas-sql-br]
---

# {repo_id}

{cfg.get("train_method", "").upper()} adapter of `{cfg["base_model"]}` trained on the
curated split of [AtlasSQL-BR](https://huggingface.co/datasets/datafromlopes/atlas-sql-br)
(Brazilian Portuguese geospatial Text-to-SQL over PostGIS).

- Run id (MLflow): `{run_id}`
- Prompt: `{prompt_template(cfg)}` (no schema injection)
- LoRA: r={cfg.get("lora_r")}, alpha={cfg.get("lora_alpha")}, dropout={cfg.get("lora_dropout")},
  targets={cfg.get("lora_target_modules")}
- Training: {cfg.get("epochs")} epochs, batch {cfg.get("batch_size")} x {cfg.get("grad_accum")},
  optim={cfg.get("optim")}, lr={cfg.get("learning_rate")}, warmup_steps={cfg.get("warmup_steps")},
  weight_decay={cfg.get("weight_decay")}, max_length={cfg.get("max_length")}
- Best validation loss: {metrics.get("best_eval_loss")}

Training and evaluation code: https://github.com/datafromlopes/atlas-sql-br
"""
    card_path = os.path.join(art_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=art_dir,
                      commit_message=f"Upload {run_id}")
    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"✔ Pushed to the Hugging Face Hub: {url}")
    return url

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Text2SQL training")
    parser.add_argument("--experiment_version", type=int, default=0)
    exp = parser.parse_args().experiment_version

    cfg, cfg_path = load_config(exp)
    CFG.clear(); CFG.update(cfg)
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
    if token:
        login(token=token)
    else:
        logger.warning("HF_TOKEN not set: gated weights must already be in the local cache; "
                       "the Hub upload (hf_push) will be skipped.")

    set_seed(cfg.get("seed", 42))

    mlflow.set_tracking_uri(build_mlflow_uri())
    mlflow.set_experiment(PROJECT_NAME)

    BASE_MODEL = cfg["base_model"]
    METHOD = cfg["train_method"]
    fp16, bf16, tf32 = resolve_precision(cfg["fp16"], cfg["bf16"], cfg["tf32"])
    # Weight dtype. On CUDA the Trainer's bf16 autocast is used; on MPS autocast
    # is not available, so the base weights themselves are loaded in bf16 when
    # the config asks for bf16 (PEFT keeps the trainable adapter in fp32).
    if IS_MPS and cfg["bf16"]:
        dtype = torch.bfloat16
        logger.info("MPS: base weights loaded in bfloat16 (adapter parameters stay fp32).")
    else:
        dtype = torch.bfloat16 if bf16 else torch.float32
    # Optimizer: the 8-bit AdamW of bitsandbytes exists only for CUDA.
    optim = str(cfg["optim"])
    if optim.startswith("adamw_bnb") and not IS_CUDA:
        logger.warning(f"{optim} requires CUDA (bitsandbytes); falling back to adamw_torch.")
        optim = "adamw_torch"
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
        # ── Data: training pool (internal val carved below); holdout untouched ──
        train_pool = load_train_pool(cfg)
        test_df = load_holdout(cfg)
        logger.info(f"Data [{dataset_version(cfg)}]: {len(train_pool)} train(+val) | "
                    f"{len(test_df)} test (holdout)")

        active = mlflow.start_run(run_name=run_id, tags={
            "experiment_version": EXP, "model_name": cfg["model_name"],
            "train_method": METHOD, "architecture": cfg["architecture"],
            "dataset_version": dataset_version(cfg), "stage": "train",
            "device": str(DEVICE), **git})
        save_run_id(exp, active.info.run_id)
        logger.info(f"MLflow run id: {active.info.run_id} (saved for the generate/evaluate stages)")

        mlflow_log_params(get_dataset_fingerprint(train_pool))
        for fpath in dataset_files(cfg):
            mlflow.log_artifact(str(fpath), artifact_path="dataset")
            mlflow_log_params({"dataset_md5_" + fpath.name.replace(".", "_"): file_md5(fpath)})
        mlflow.log_artifact(str(cfg_path), artifact_path="config")
        mlflow_log_params({
            "base_model": BASE_MODEL, "train_method": METHOD, "architecture": cfg["architecture"],
            "device": str(DEVICE), "dataset_version": dataset_version(cfg),
            "dataset_partition": cfg.get("dataset_partition"),
            "prompt_template": prompt_template(cfg),
            "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
            "grad_accum": cfg["grad_accum"], "learning_rate": cfg.get("learning_rate"),
            "lr_scheduler": cfg.get("lr_scheduler_type"), "warmup_ratio": cfg.get("warmup_ratio"),
            "warmup_steps": cfg.get("warmup_steps"), "weight_decay": cfg.get("weight_decay"),
            "optim": cfg.get("optim"), "optim_effective": optim,
            "gradient_checkpointing": cfg.get("gradient_checkpoints"),
            "eval_on_holdout": bool(cfg.get("eval_on_holdout", False)),
            "early_stopping_patience": cfg.get("early_stopping_patience"),
            "lora_r": cfg.get("lora_r"), "lora_alpha": cfg.get("lora_alpha"),
            "lora_dropout": cfg.get("lora_dropout"),
            "lora_target_modules": cfg.get("lora_target_modules"), "seed": cfg.get("seed", 42),
            "fp16": fp16, "bf16": bf16, "tf32": tf32, "max_length": cfg["max_length"],
            "val_fraction": (None if cfg.get("eval_on_holdout") else cfg.get("val_fraction", 0.1)),
            "n_train_pool": len(train_pool), "n_test_holdout": len(test_df),
            "transformers": transformers.__version__, "torch": torch.__version__,
            "peft": __import__("peft").__version__,
            "hf_repo_id": cfg.get("hf_repo_id"),
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
            if cfg.get("eval_on_holdout"):
                # Paper protocol: the validation split (196) is the eval set that
                # drives checkpoint selection; every training pair is used.
                tr_df, vl_df = train_pool, test_df
                logger.info(f"Split: {len(tr_df)} train | {len(vl_df)} validation "
                            f"(the held-out split; eval_on_holdout=True)")
            else:
                tr_df, vl_df = stratified_val_split(train_pool, cfg.get("val_fraction", 0.1),
                                                    cfg.get("seed", 42))
                logger.info(f"Split: {len(tr_df)} train | {len(vl_df)} internal val "
                            f"(stratified by tier, seed {cfg.get('seed', 42)})")
            mlflow_log_params({"n_train": len(tr_df), "n_val": len(vl_df)})
            DS = CausalDataset
            train_ds = DS(tr_df, tokenizer, cfg["max_length"])
            val_ds = DS(vl_df, tokenizer, cfg["max_length"])
            collator = CausalCollator(tokenizer)

            out_dir = str(experiment_dir(exp))
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
                  "warmup_steps": cfg.get("warmup_steps", 0),
                  "weight_decay": cfg["weight_decay"], "logging_steps": cfg["logging_steps"],
                  "logging_strategy": cfg["logging_strategy"],
                  "evaluation_strategy": cfg["evaluation_strategy"],
                  "save_strategy": cfg["save_strategy"], "fp16": fp16, "bf16": bf16, "tf32": tf32,
                  "dataloader_num_workers": 0 if not IS_CUDA else cfg["dataloader_num_workers"],
                  "dataloader_pin_memory": False if not IS_CUDA else cfg["dataloader_pin_memory"],
                  "optim": optim, "gradient_checkpoints": cfg["gradient_checkpoints"],
                  "load_best_model": cfg["load_best_model"],
                  "metric_for_best_model": cfg["metric_for_best_model"],
                  "greater_is_better": cfg["greater_is_better"],
                  "early_stopping_patience": cfg["early_stopping_patience"],
                  "early_stopping_threshold": cfg["early_stopping_threshold"],
                  "run_name": run_id, "seed": cfg.get("seed", 42),
                  "eval_on_start": cfg.get("eval_on_start", True)}

            logger.info("Starting training…")
            trainer = train_model(model, train_ds, val_ds, tokenizer, collator, ta)

            # ── Artifacts (best checkpoint, restored by load_best_model_at_end) ──
            art = f"{out_dir}/artifacts"
            os.makedirs(art, exist_ok=True)
            trainer.model.save_pretrained(art)
            tokenizer.save_pretrained(art)

            # Training history, trainer state and args go with the artifacts so a
            # run can be audited without the tracking server.
            with open(os.path.join(art, "training_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
            trainer.state.save_to_json(os.path.join(art, "trainer_state.json"))
            torch.save(trainer.args, os.path.join(art, "training_args.bin"))
            with open(os.path.join(art, "experiment_config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            evals = [h for h in trainer.state.log_history if "eval_loss" in h]
            best = min(evals, key=lambda h: h["eval_loss"]) if evals else {}
            final_metrics = {
                "best_eval_loss": best.get("eval_loss"),
                "best_epoch": best.get("epoch"),
                "best_step": best.get("step"),
                "epochs_run": trainer.state.epoch,
                "global_steps": trainer.state.global_step,
            }
            mlflow.log_metrics({k: float(v) for k, v in final_metrics.items() if v is not None})
            mlflow_log_params({"best_checkpoint": trainer.state.best_model_checkpoint})
            mlflow.log_artifacts(art, artifact_path="model")
            logger.info(f"✔ {run_id} complete. Artifacts at: {art} | best eval_loss="
                        f"{final_metrics['best_eval_loss']} (epoch {final_metrics['best_epoch']})")

            # ── Hugging Face Hub ─────────────────────────────────────────────
            try:
                url = push_to_hub(cfg, art, run_id, final_metrics, token)
                if url:
                    mlflow.set_tags({"hf_repo_url": url, "hf_repo_id": cfg.get("hf_repo_id")})
            except Exception as he:
                logger.error(f"Hub upload failed (artifacts are safe locally at {art}): {he}",
                             exc_info=True)
                mlflow.set_tag("hf_push_error", str(he)[:250])

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