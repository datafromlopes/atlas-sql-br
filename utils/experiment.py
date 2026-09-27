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
"""
experiment.py — helpers shared by the three pipeline stages (train, generate,
evaluate) so that they agree on: the experiment config, the prompt, the dataset
version, the SQL extraction procedure and the MLflow run they write to.

Dataset versions
----------------
  working        data/atlas_sql_br.parquet (+ _validation): the internal
                 working copy with a `train` flag (train==0 is the holdout).
  paper_released data/paper_released/{train,validation}.parquet: the exact
                 snapshot published with the SBBD 2026 paper (mirror of the
                 Hugging Face tag `paper-released`). Training uses the curated
                 `base_dataset` partition of train.parquet; validation.parquet is
                 the held-out evaluation split (196 records).
  paper_split    data/paper_split/{train,validation}.parquet: the protocol as
                 DESCRIBED in the paper — the 980 curated pairs partitioned into
                 784 training / 196 validation pairs, stratified by complexity
                 tier and spatial function (seed 42). Produced once by
                 core/make_paper_split.py and versioned. This is the version the
                 reproduction experiment (exp-v6) uses.

MLflow
------
  The tracking URI comes from MLFLOW_TRACKING_URI when set (e.g. on a remote
  GPU box), else from PG_USER/PG_PASS (the local Postgres backend), else a file
  store in ./mlruns. The training stage writes its run id to
  models/experiments/v{N}/mlflow_run_id.txt so generation and evaluation attach
  their artifacts and metrics to the SAME run.
"""
from __future__ import annotations

import os
import re
import hashlib
from pathlib import Path
from urllib.parse import quote_plus

import yaml
import polars as pl

from .global_variables import (
    PROJECT_PATH, DATASET_PATH, DATASET_NAME, DATASET_VALIDATION_NAME,
)

# ── Dataset column names (identical across the pipeline) ─────────────────────
ID_COL, Q_COL, SQL_COL, LEVEL_COL, TRAIN_COL = "id", "question", "sql_code", "level", "train"
DIV_COL, FUNC_COL, SOURCE_COL = "territorial_division", "geospatial_functions", "source"

PAPER_DIR = Path(DATASET_PATH) / "paper_released"
PAPER_TRAIN = PAPER_DIR / "train.parquet"
PAPER_VALIDATION = PAPER_DIR / "validation.parquet"
SPLIT_DIR = Path(DATASET_PATH) / "paper_split"
SPLIT_TRAIN = SPLIT_DIR / "train.parquet"
SPLIT_VALIDATION = SPLIT_DIR / "validation.parquet"

DEFAULT_PROMPT = "Pergunta: {question}\nSQL:"

# Known spelling variants of the complexity tiers in the published files.
_LEVEL_FIX = {"Facíl": "Fácil", "Facil": "Fácil", "Medio": "Médio", "Dificil": "Difícil",
              "Muito Dificil": "Muito Difícil"}


# =============================================================================
# CONFIG
# =============================================================================
def load_experiment_config(experiment_version: int) -> tuple[dict, Path]:
    path = Path(PROJECT_PATH) / "experiments" / f"exp-v{experiment_version}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), path


def experiment_dir(experiment_version: int) -> Path:
    return Path(PROJECT_PATH) / "models" / "experiments" / f"v{experiment_version}"


# =============================================================================
# PROMPT
# =============================================================================
def prompt_template(cfg: dict) -> str:
    return str(cfg.get("prompt_template") or DEFAULT_PROMPT)


def build_prompt(cfg: dict, question: str) -> str:
    """The prompt is the ONLY thing the model sees (no schema injection). It must
    be identical in training and generation, so both stages call this."""
    return prompt_template(cfg).format(question=question)


# =============================================================================
# DATASET
# =============================================================================
def normalize_levels(df: pl.DataFrame) -> pl.DataFrame:
    if LEVEL_COL in df.columns:
        df = df.with_columns(pl.col(LEVEL_COL).replace(_LEVEL_FIX))
    return df


def dataset_version(cfg: dict) -> str:
    return str(cfg.get("dataset_version") or "working")


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python core/make_paper_split.py` first "
            f"to build the 784/196 split described in the paper.")
    return path


def _limit(cfg: dict, df: pl.DataFrame) -> pl.DataFrame:
    """`smoke_max_samples` in the YAML truncates the data for pipeline smoke tests."""
    n = int(cfg.get("smoke_max_samples") or 0)
    return df.head(n) if n > 0 else df


def load_train_pool(cfg: dict) -> pl.DataFrame:
    """Rows available for training. For `paper_split` this is exactly the 784
    training pairs; for the other versions an internal validation slice may be
    carved out by the training script. The holdout is never touched."""
    if dataset_version(cfg) == "paper_split":
        return _limit(cfg, normalize_levels(pl.read_parquet(_require(SPLIT_TRAIN))))
    if dataset_version(cfg) == "paper_released":
        df = pl.read_parquet(PAPER_TRAIN)
        partition = str(cfg.get("dataset_partition") or "base_dataset")
        if SOURCE_COL in df.columns and partition != "all":
            df = df.filter(pl.col(SOURCE_COL) == partition)
        return normalize_levels(df)
    df = pl.read_parquet(Path(DATASET_PATH) / DATASET_NAME)
    missing = [c for c in (Q_COL, SQL_COL, TRAIN_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}. Available: {df.columns}")
    return normalize_levels(df.filter(pl.col(TRAIN_COL) == 1))


def load_holdout(cfg: dict) -> pl.DataFrame:
    """The evaluation split: the published validation.parquet for the paper
    version, or the train==0 rows of the working validation file."""
    if dataset_version(cfg) == "paper_split":
        return _limit(cfg, normalize_levels(pl.read_parquet(_require(SPLIT_VALIDATION))))
    if dataset_version(cfg) == "paper_released":
        return normalize_levels(pl.read_parquet(PAPER_VALIDATION))
    df = pl.read_parquet(Path(DATASET_PATH) / DATASET_VALIDATION_NAME)
    if Q_COL not in df.columns and "pergunta" in df.columns:
        df = df.rename({"pergunta": Q_COL})
    if TRAIN_COL in df.columns:
        df = df.filter(pl.col(TRAIN_COL) == 0)
    return normalize_levels(df)


def dataset_files(cfg: dict) -> list[Path]:
    """The data files an experiment depends on (logged to MLflow as artifacts)."""
    if dataset_version(cfg) == "paper_split":
        return [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_DIR / "split_manifest.json"]
    if dataset_version(cfg) == "paper_released":
        return [PAPER_TRAIN, PAPER_VALIDATION]
    return [Path(DATASET_PATH) / DATASET_NAME, Path(DATASET_PATH) / DATASET_VALIDATION_NAME]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# SQL EXTRACTION  (the fixed, mechanical procedure applied to BOTH models)
# =============================================================================
_FENCE_RE = re.compile(r"```(?:sql|SQL|postgresql)?\s*(.*?)```", re.DOTALL)
_TAG_RE = re.compile(r"^\s*(?:\[[^\]\n]*\]\s*)+", re.MULTILINE)
_START_RE = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)
_TRAIL_RE = re.compile(r";\s*\S.*$", re.DOTALL)


def extract_sql(text: str) -> str:
    """Four steps, as documented in the paper and the dissertation:
      (i)   strip Markdown code fences when the answer is wrapped in them;
      (ii)  strip leading annotation tags such as `[Divisão: UF]`;
      (iii) take the first substring that begins with WITH or SELECT;
      (iv)  discard trailing text after the query (anything past the first `;`).
    Returns "" when no SQL token is found (the record is then counted as
    'no recognizable SQL')."""
    text = (text or "").strip()
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = _TAG_RE.sub("", text, count=1).strip()
    m = _START_RE.search(text)
    if not m:
        return ""
    text = text[m.start():]
    text = _TRAIL_RE.sub(";", text)
    return text.strip()


# =============================================================================
# MLFLOW
# =============================================================================
def build_mlflow_uri(logger=None) -> str:
    """MLFLOW_TRACKING_URI > Postgres (PG_USER/PG_PASS) > file store ./mlruns.
    The password never reaches the log."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        if logger: logger.info(f"MLflow tracking URI: {env_uri} (from MLFLOW_TRACKING_URI)")
        return env_uri
    pg_user = os.environ.get("PG_USER")
    pg_pass = os.environ.get("PG_PASS", "")
    pg_host = os.environ.get("PG_HOST", "localhost")
    pg_port = os.environ.get("PG_PORT", "5432")
    if pg_user and pg_pass:
        uri = f"postgresql://{pg_user}:{quote_plus(pg_pass)}@{pg_host}:{pg_port}/mlflow"
        if logger: logger.info(f"MLflow tracking URI: {uri.replace(quote_plus(pg_pass), '****')}")
        return uri
    uri = f"file://{PROJECT_PATH}/mlruns"
    if logger: logger.warning(f"MLFLOW_TRACKING_URI / PG_USER / PG_PASS not set — using {uri}")
    return uri


def run_id_file(experiment_version: int) -> Path:
    return experiment_dir(experiment_version) / "mlflow_run_id.txt"


def save_run_id(experiment_version: int, run_id: str) -> None:
    p = run_id_file(experiment_version)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run_id.strip() + "\n", encoding="utf-8")


def load_run_id(experiment_version: int) -> str | None:
    p = run_id_file(experiment_version)
    if p.exists():
        rid = p.read_text(encoding="utf-8").strip()
        return rid or None
    return None
