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
# =============================================================================================
# IMPORTS - STANDARD LIBRARY
# =============================================================================================
import os
import gc
import json
import logging
import sys
from pathlib import Path
import argparse
import yaml
# =============================================================================================
# IMPORTS - THIRD PARTY
# =============================================================================================
import torch
from torch.utils.data import Dataset, DataLoader

import transformers
import peft
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    T5Config,
    T5ForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM
)

from peft import (
    get_peft_model,
    LoraConfig,
    IA3Config,
    TaskType,
    PeftModel,
)

import polars as pl
from huggingface_hub import login
import mlflow
import mlflow.pytorch
import numpy as np
# =============================================================================================
# PATH SETUP
# =============================================================================================
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_raiz  = os.path.abspath(os.path.join(diretorio_atual, ".."))
if diretorio_raiz not in sys.path:
    sys.path.append(diretorio_raiz)

from utils import (
    PROJECT_NAME,
    PROJECT_PATH,
    DATASET_FULL_NAME
)
# =============================================================================================
# GLOBAL VARIABLES SETUP
# =============================================================================================






# =============================================================================================
# ENVIRONMENT & SECURITY SETUP
# =============================================================================================
os.environ["TOKENIZERS_PARALLELISM"]           = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]      = "1"

# Busca o token de forma segura da variável de ambiente do sistema operacional
HUGGINGFACE_HUB_TOKEN = os.environ.get("HF_TOKEN")

if not HUGGINGFACE_HUB_TOKEN:
    raise ValueError(
        "Token do Hugging Face não encontrado! "
        "Certifique-se de exportar a variável no seu terminal do Mac antes de rodar o script:\n"
        "export HF_TOKEN='seu_token_aqui'"
    )

os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_HUB_TOKEN
login(token=HUGGINGFACE_HUB_TOKEN)

# =============================================================================================
# LOGGING SETUP
# =============================================================================================
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
COLORS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[41m",
}
ICONS = {
    "DEBUG":    "·",
    "INFO":     "✔",
    "WARNING":  "⚠",
    "ERROR":    "✖",
    "CRITICAL": "☠",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, "")
        icon  = ICONS.get(record.levelname, "•")
        ts    = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return (
            f"{DIM}{ts}{RESET} "
            f"{color}{BOLD}{icon} [{record.levelname:<8}]{RESET} "
            f"{DIM}[{record.name}]{RESET} "
            f"{record.getMessage()}"
        )


logging.basicConfig(
    level=logging.INFO,
    force=True,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logging.getLogger().handlers[0].setFormatter(ColorFormatter())
logging.getLogger().handlers[1].setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logger = logging.getLogger("Fine-Tuning-Pipeline")

# =============================================================================================
# DEVICE CONFIGURATION
# =============================================================================================
def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device      = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: CUDA ({device_name})")
        logger.info(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
        logger.info(
            "  Unified memory: shared between CPU and GPU — "
            "avoid pin_memory and multi-worker DataLoader"
        )
    else:
        device = torch.device("cpu")
        logger.warning("Device: CPU (training will be slow)")

    return device


DEVICE = setup_device()

IS_MPS  = DEVICE.type == "mps"
IS_CUDA = DEVICE.type == "cuda"

# =============================================================================================
# MPS OVERRIDES
# =============================================================================================
# if IS_MPS:
#     FP16 = False
#     BF16 = False
#     TF32 = False
#     DATALOADER_NUM_WORKERS = 0
#     DATALOADER_PIN_MEMORY  = False
#     GRADIENT_CHECKPOINTING = False
#     OPTIM                  = "adamw_torch"
#     BATCH_SIZE             = 4
#     GRAD_ACCUM             = 4

#     logger.info(
#         "MPS overrides applied: fp16/bf16/tf32=False, "
#         "num_workers=0, pin_memory=False, "
#         f"batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}"
#     )

# =============================================================================================
# MLFLOW SETUP
# =============================================================================================
MLFLOW_TRACKING_URI = f"file://{PROJECT_PATH}/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(PROJECT_NAME)
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


def mlflow_start_run(run_name: str, tags: dict) -> mlflow.ActiveRun:
    """Start an MLflow run, ending any currently active one first."""
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass
    return mlflow.start_run(run_name=run_name, tags=tags)


def mlflow_log_params(params: dict):
    """Log a flat dict of params, safely serializing non-primitive values."""
    safe = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    mlflow.log_params(safe)


def mlflow_log_training_history(log_history: list):
    """Log each entry of trainer.state.log_history as MLflow metrics."""
    for entry in log_history:
        step = entry.get("step", 0)
        metrics = {
            k: v for k, v in entry.items()
            if k != "step" and isinstance(v, (int, float))
        }
        if metrics:
            mlflow.log_metrics(metrics, step=step)


def mlflow_end_run(status: str = "FINISHED"):
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except Exception as e:
        logger.error(f"MLflow end_run error: {e}")


# =============================================================================================
# UTILITY FUNCTIONS
# =============================================================================================
def get_versions() -> dict:
    return {
        "python":       sys.version.split()[0],
        "torch":        torch.__version__,
        "transformers": transformers.__version__,
        "cuda":         torch.version.cuda if IS_CUDA else None,
        "mps":          str(IS_MPS),
    }


def save_training_artifacts(
    trainer,
    tokenizer,
    output_dir: str,
    metadata: dict | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if trainer.state.log_history:
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        mlflow.log_artifact(metrics_path)

    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
    mlflow.log_artifact(os.path.join(output_dir, "trainer_state.json"))

    torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))

    if metadata:
        meta_path = os.path.join(output_dir, "metadata.json")
        safe_meta = {}
        for k, v in metadata.items():
            try:
                json.dumps(v)
                safe_meta[k] = v
            except (TypeError, ValueError):
                safe_meta[k] = str(v)
        with open(meta_path, "w") as f:
            json.dump(safe_meta, f, indent=2)
        mlflow.log_artifact(meta_path)

    mlflow.log_artifacts(output_dir, artifact_path="model")
    logger.info(f"Artifacts logged to MLflow run and saved at: {output_dir}")

    return output_dir


def safe_config_to_dict(config) -> dict:
    try:
        raw = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    except Exception:
        return {}

    result = {}
    for k, v in raw.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result


# =============================================================================================
# DATASET
# =============================================================================================
class Text2SQLDataset(Dataset):
    def __init__(self, data: pl.DataFrame, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        questions = data["question"].to_list()
        divisions = data["territorial_division"].to_list()
        sqls      = data["sql_code"].to_list()

        eos   = tokenizer.eos_token
        texts = [
            f"Pergunta: {q}" + (f" [Divisão: {d}]" if d else "") + f"\nSQL: {s}{eos}"
            for q, d, s in zip(questions, divisions, sqls)
        ]

        encodings = tokenizer(
            texts,
            max_length  = max_length,
            truncation  = True,
            padding     = False,
        )

        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids  = torch.tensor(self.input_ids[idx],      dtype=torch.long)
        mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        return {
            "input_ids":      ids,
            "attention_mask": mask,
            "labels":         ids.clone(),
        }

class OptimizedDataCollator:
    """Dynamic padding collator — pad_to_multiple_of=8 for Tensor Core alignment."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer          = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _round_up(self, n: int) -> int:
        m = self.pad_to_multiple_of
        return ((n + m - 1) // m) * m

    def __call__(self, features):
        input_ids      = [f["input_ids"]     for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels         = [f["labels"]         for f in features]

        max_len = self._round_up(max(len(ids) for ids in input_ids))
        pad_id  = self.tokenizer.pad_token_id or 0

        padded_ids, padded_mask, padded_labels = [], [], []
        for ids, mask, lbl in zip(input_ids, attention_mask, labels):
            pad = max_len - len(ids)
            padded_ids.append(
                torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)])
            )
            padded_mask.append(
                torch.cat([mask, torch.zeros(pad, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([lbl, torch.full((pad,), -100, dtype=torch.long)])
            )

        return {
            "input_ids":      torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_mask),
            "labels":         torch.stack(padded_labels),
        }


def prepare_data(
    df: pl.DataFrame,
    tokenizer,
    train_split: float,
    max_length: int,
):
    n_train       = int(len(df) * train_split)
    train_dataset = Text2SQLDataset(df[:n_train], tokenizer, max_length=max_length)
    val_dataset   = Text2SQLDataset(df[n_train:], tokenizer, max_length=max_length)
    return train_dataset, val_dataset


# =============================================================================================
# TRAIN CONFIG BUILDERS
# =============================================================================================
_ATTENTION_MODULES = ["q_proj", "v_proj"]
_FFN_MODULES       = ["gate_proj", "up_proj", "down_proj"]
_ALL_MODULES       = _ATTENTION_MODULES + _FFN_MODULES

def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        target_modules = _ATTENTION_MODULES,
        inference_mode = False,
    )

def get_ia3_config() -> IA3Config:
    return IA3Config(
        task_type           = TaskType.CAUSAL_LM,
        target_modules      = _ATTENTION_MODULES + ["down_proj"],
        feedforward_modules = ["down_proj"],
        inference_mode      = False,
    )

def get_base_config(model_architecture: str):
    config = None
    match model_architecture.upper():
        case "T5":
            config = T5Config(
                vocab_size=32128,          # Tamanho do vocabulário padrão do T5
                d_model=256,               # Dimensão dos vetores de cada palavra
                d_kv=64,                   # Dimensão das matrizes de Atenção (Key/Value)
                d_ff=1024,                 # Tamanho da rede Feed-Forward (memória interna)
                num_layers=4,              # Quantidade de blocos no Encoder e no Decoder
                num_heads=4,               # Número de cabeças de atenção
                pad_token_id=0,            # ID usado para preencher espaços vazios
                eos_token_id=1,            # ID que avisa o fim da frase
                decoder_start_token_id=0,  # ID que avisa o Decoder para começar a gerar o SQL
            )
        case "LLAMA":
            config = LlamaConfig(
                vocab_size=32000,              # Tamanho do dicionário de palavras
                hidden_size=512,               # d_model (Tamanho do vetor de cada token)
                intermediate_size=1024,        # Tamanho da camada Feed-Forward
                num_hidden_layers=6,           # Quantidade de blocos Transformer
                num_attention_heads=8,         # Número de cabeças de atenção
                max_position_embeddings=1024,  # Contexto máximo (quantidade de tokens)
            )
    return config

# =============================================================================================
# MODEL FACTORIES
# =============================================================================================
def _load_tokenizer(base_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer

def _build_model(config_type, model_architecture, base_model_name: str, config):
    model = None
    
    if config_type.upper() == "BASE":
        config = get_base_config(model_architecture=model_architecture)

        match model_architecture.upper():
            case "T5":
                model = T5ForConditionalGeneration(config)
            case "LLAMA":
                model = LlamaForCausalLM(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype = torch.bfloat16
        )
        model = model.to(DEVICE)
        model = get_peft_model(model, config)

    model.print_trainable_parameters()
    return model, safe_config_to_dict(config)


def create_lora_model(model_architecture, base_model_name: str, **kwargs):
    config_type="LORA"
    return _build_model(config_type, model_architecture, base_model_name, get_lora_config())

def create_ia3_model(model_architecture, base_model_name: str, **kwargs):
    config_type="IA3"
    return _build_model(config_type, model_architecture, base_model_name, get_ia3_config())

def create_from_scratch(model_architecture, base_model_name: str, **kwargs):
    config_type="BASE"
    return _build_model(config_type, model_architecture, base_model_name, get_base_config())

def save_merged_standalone(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    tokenizer,
):
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter path not found: '{adapter_path}'. "
            "Point to the artifacts directory of a completed training run."
        )

    logger.info(f"[save_merged] Loading base: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype         = torch.float32 if IS_MPS else torch.bfloat16,
        device_map          = None,
        attn_implementation = "eager",
    )

    logger.info(f"[save_merged] Loading adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable = False,
    )

    logger.info("[save_merged] Merging…")
    merged = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[save_merged] Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "source_base_model":  base_model_name,
        "source_adapter_path": adapter_path,
        "merged_model_path":  output_dir,
        "torch_dtype":        "float32" if IS_MPS else "bfloat16",
        "merge_strategy":     "peft.merge_and_unload",
    }
    with open(os.path.join(output_dir, "merge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[save_merged] ✔ Standalone model saved at: {output_dir}")
    return output_dir

# =============================================================================================
# TRAINER WRAPPER
# =============================================================================================
def train_model(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    custom_training_args: dict,
):
    training_args = TrainingArguments(
        output_dir                  = custom_training_args["checkpoints_dir"],
        num_train_epochs            = custom_training_args["epochs"],
        per_device_train_batch_size = custom_training_args["batch_size"],
        per_device_eval_batch_size  = custom_training_args["batch_size"],
        gradient_accumulation_steps = custom_training_args["grad_accum"],
        warmup_steps                = custom_training_args["warmup_steps"],
        weight_decay                = custom_training_args["weight_decay"],
        logging_dir                 = custom_training_args["logs_dir"],
        logging_steps               = custom_training_args["logging_steps"],
        eval_strategy               = custom_training_args["eval_strategy"],
        save_strategy               = custom_training_args["save_strategy"],
        fp16                        = custom_training_args["fp16"],
        bf16                        = custom_training_args["bf16"],
        tf32                        = custom_training_args["tf32"],
        dataloader_num_workers      = custom_training_args["dataloader_num_workers"],
        dataloader_pin_memory       = custom_training_args["dataloader_pin_memory"],
        optim                       = custom_training_args["optim"],
        gradient_checkpointing      = custom_training_args["gradient_checkpointing"],
        load_best_model_at_end      = custom_training_args["load_best_model_at_end"],
        report_to                   = "none",  
        logging_strategy            = custom_training_args["logging_strategy"],
        run_name                    = custom_training_args["run_name"],
        metric_for_best_model       = custom_training_args["metric_for_best_model"],
        greater_is_better           = custom_training_args["greater_is_better"],
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = OptimizedDataCollator(tokenizer),
        callbacks     = [
            EarlyStoppingCallback(
                early_stopping_patience  = 2,
                early_stopping_threshold = 0.0001,
            )
        ],
    )

    logger.info(f"Device used: {next(trainer.model.parameters()).device}")
    trainer.train()
    return trainer

# =============================================================================================
# EXPERIMENT PARAMETERS
# =============================================================================================
parser = argparse.ArgumentParser(description="Training Script")

parser.add_argument("--experiment_version", type=int, default=0, help="The experiment version")
args = parser.parse_args()

experiment_version = args.experiment_version
CONFIG_PATH = PROJECT_PATH / "experiments" / f"exp-v{experiment_version}.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

TRAIN_FACTORY = {
    "lora":    create_lora_model,
    "ia3":     create_ia3_model,
    "scratch": create_from_scratch
}

TRAIN_METHOD = config['train_method']
TRAIN_ARCHITECTURE = config['architecture']
EXPERIMENT_VERSION = config['experiment_version']
DATASET_PARTITION = config['dataset_partition']
DATASET_ARTEFACT_NAME = config['dataset_artefact_name']
DATASET_ARTEFACT_VERSION = config['dataset_artefact_version']
BASE_MODEL = config['base_model']
MODEL_NAME = config['model_name']
EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
GRAD_ACCUM = config['grad_accum']
LOGGING_STEPS = config['logging_steps']
WEIGHT_DECAY = config['weight_decay']
WARMUP_STEPS = config['warmup_steps']
GRADIENT_CHECKPOINTING = config['gradient_checkpoints']
LOGGING_STRATEGY = config['logging_strategy']
EVALUATION_STRATEGY = config['evaluation_strategy']
SAVE_STRATEGY = config['save_strategy']
FP16 = config['fp16']
BF16 = config['bf16']
TF32 = config['tf32']
DATALOADER_NUM_WORKERS = config['dataloader_num_workers']
DATALOADER_PIN_MEMORY = config['dataloader_pin_memory']
OPTIM = config['optim']
LOAD_BEST_MODEL = config['load_best_model']
GREATER_IS_BETTER = config['greater_is_better']
METRIC_FOR_BEST_MODEL = config['metric_for_best_model']
MAX_LENGTH = config['max_length']
TRAIN_SPLIT = config['train_split']

def _run_name(model_name: str, peft_type: str, version: str) -> str:
    return f"{model_name}_{peft_type}_{version}"

# =============================================================================================
# MAIN
# =============================================================================================
def main():
    mlflow_end_run()  

    dataset_partition = DATASET_PARTITION
    libs_versions     = get_versions()

    logger.info("=" * 80)
    logger.info(
        f"  PIPELINE START  |  experiment={EXPERIMENT_VERSION}"
        f"  |  partition={dataset_partition}  |  device={DEVICE}"
    )
    logger.info(f"  Model  : {BASE_MODEL}")
    logger.info(f"  Method: {TRAIN_METHOD}")
    logger.info("=" * 80)

    tokenizer = _load_tokenizer(BASE_MODEL)

    run_id  = _run_name(MODEL_NAME, TRAIN_METHOD, EXPERIMENT_VERSION)
    model   = None
    trainer = None

    logger.info(f"\n{'─' * 80}")
    logger.info(f"  ▶  {run_id}")
    logger.info(f"{'─' * 80}\n")

    try:
        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------
        logger.info("Loading dataset from local parquet…")
        dataset_path = f"{DATASET_FULL_NAME}"
        df = pl.scan_parquet(dataset_path, hive_partitioning=True).collect()
        if dataset_partition != "all":
            df = df.filter(pl.col("source") == dataset_partition)

        df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
        train_dataset, val_dataset = prepare_data(
            df_shuffled, tokenizer, TRAIN_SPLIT, MAX_LENGTH
        )
        total_steps = (len(train_dataset) // BATCH_SIZE) * EPOCHS // GRAD_ACCUM

        logger.info(
            f"Dataset: {len(train_dataset)} train | "
            f"{len(val_dataset)} val | {total_steps} steps"
        )

        # ------------------------------------------------------------------
        # MLflow run
        # ------------------------------------------------------------------
        run_tags = {
            "experiment_version": EXPERIMENT_VERSION,
            "model_name":         MODEL_NAME,
            "train_method":       TRAIN_METHOD,
            "dataset_partition":  DATASET_PARTITION,
            "architecture":       TRAIN_ARCHITECTURE,
            "device":             str(DEVICE)
        }
        mlflow_start_run(run_name=run_id, tags=run_tags)

        run_params = {
            "base_model":             BASE_MODEL,
            "model_name":             MODEL_NAME,
            "train_method":           TRAIN_METHOD,
            "experiment_version":     EXPERIMENT_VERSION,
            "dataset_partition":      DATASET_PARTITION,
            "device":                 str(DEVICE),
            "epochs":                 EPOCHS,
            "batch_size":             BATCH_SIZE,
            "grad_accum":             GRAD_ACCUM,
            "warmup_steps":           WARMUP_STEPS,
            "weight_decay":           WEIGHT_DECAY,
            "optim":                  OPTIM,
            "bf16":                   BF16,
            "fp16":                   FP16,
            "tf32":                   TF32,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "transformers_version":   libs_versions.get("transformers"),
            "torch_version":          libs_versions.get("torch"),
            "cuda_version":           libs_versions.get("cuda"),
            "mps":                    libs_versions.get("mps"),
            "python_version":         libs_versions.get("python"),
            "train_samples":          len(train_dataset),
            "val_samples":            len(val_dataset),
            "total_steps":            total_steps,
        }

        mlflow_log_params(run_params)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        factory = TRAIN_FACTORY[TRAIN_METHOD]
        model, model_config_dict = factory(TRAIN_ARCHITECTURE, BASE_MODEL, total_steps=total_steps)
        mlflow_log_params({f"peft_cfg_{k}": v for k, v in model_config_dict.items()})

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        output_dir = (
            f"{PROJECT_PATH}/models/experiments/{EXPERIMENT_VERSION}"
        )
        os.makedirs(output_dir, exist_ok=True)

        custom_training_args = {
            "checkpoints_dir":        f"{output_dir}/checkpoints",
            "epochs":                 EPOCHS,
            "batch_size":             BATCH_SIZE,
            "grad_accum":             GRAD_ACCUM,
            "warmup_steps":           WARMUP_STEPS,
            "weight_decay":           WEIGHT_DECAY,
            "logs_dir":               f"{output_dir}/logs",
            "logging_steps":          LOGGING_STEPS,
            "eval_strategy":          EVALUATION_STRATEGY,
            "save_strategy":          SAVE_STRATEGY,
            "fp16":                   FP16,
            "bf16":                   BF16,
            "tf32":                   TF32,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "dataloader_pin_memory":  DATALOADER_PIN_MEMORY,
            "optim":                  OPTIM,
            "load_best_model_at_end": LOAD_BEST_MODEL,
            "report_to":              "none",
            "logging_strategy":       LOGGING_STRATEGY,
            "run_name":               run_id,
            "metric_for_best_model":  METRIC_FOR_BEST_MODEL,
            "greater_is_better":      GREATER_IS_BETTER,
        }

        logger.info("Starting training…")
        trainer = train_model(
            model,
            train_dataset,
            val_dataset,
            tokenizer,
            custom_training_args,
        )

        if trainer.state.log_history:
            mlflow_log_training_history(trainer.state.log_history)

        # ------------------------------------------------------------------
        # Artifacts
        # ------------------------------------------------------------------
        artifact_metadata = {
            "run_id":               run_id,
            "base_model":           BASE_MODEL,
            "model_name":           MODEL_NAME,
            "train_method":         TRAIN_METHOD,
            "experiment_version":   EXPERIMENT_VERSION,
            "dataset_partition":    DATASET_PARTITION,
            "train_samples":        len(train_dataset),
            "val_samples":          len(val_dataset),
            "total_steps":          total_steps,
            "transformers_version": libs_versions.get("transformers"),
            "torch_version":        libs_versions.get("torch"),
            "cuda_version":         libs_versions.get("cuda"),
            "python_version":       libs_versions.get("python"),
        }
        artifact_metadata.update(custom_training_args)

        logger.info(f"Saving artifacts as '{run_id}'…")
        artifact_dir = f"{output_dir}/artifacts"
        save_training_artifacts(
            trainer    = trainer,
            tokenizer  = tokenizer,
            output_dir = artifact_dir,
            metadata   = artifact_metadata,
        )

        logger.info(f"✔ {run_id} done. Artifacts at: {artifact_dir}\n")

    except KeyboardInterrupt:
        logger.warning(f"Pipeline interrupted by user at {run_id}")
        mlflow_end_run(status="KILLED")
        raise

    except Exception as e:
        logger.error(f"Error in {run_id}: {e}", exc_info=True)
        mlflow_end_run(status="FAILED")
        logger.info("Continuing to next run…\n")

    finally:
        # ------------------------------------------------------------------
        # Memory cleanup
        # ------------------------------------------------------------------
        try:
            if trainer is not None:
                for attr in (
                    "model", "train_dataset", "eval_dataset",
                    "data_collator", "optimizer", "lr_scheduler",
                    "callback_handler",
                ):
                    setattr(trainer, attr, None)
                del trainer

            if model is not None:
                del model

            del train_dataset, val_dataset

            gc.collect()

            if IS_CUDA:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                logger.info(
                    f"CUDA allocated after cleanup: "
                    f"{torch.cuda.memory_allocated() / 1e9:.3f} GB"
                )

            if IS_MPS:
                torch.mps.empty_cache()
                torch.mps.synchronize()
                logger.info("MPS cache cleared.")

        except Exception as cleanup_error:
            logger.warning(f"Cleanup error in {run_id}: {cleanup_error}")

        mlflow_end_run()

    logger.info("=" * 80)
    logger.info(f"  PIPELINE COMPLETE  |  experiment={EXPERIMENT_VERSION}")
    logger.info("=" * 80)

# =============================================================================================
# ENTRY POINT
# =============================================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)