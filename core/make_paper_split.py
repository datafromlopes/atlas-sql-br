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
#  Build the train/validation split described in the SBBD 2026 paper:
#
#    "The 980 pairs were partitioned into training (80%, 784 samples) and
#     validation (20%, 196 samples) using stratified sampling by complexity
#     tier and spatial operator type, ensuring each tier is proportionally
#     represented in both splits."
#
#  Input : data/paper_released/train.parquet, partition source == base_dataset
#          (the 980 curated pairs; mirror of the Hugging Face tag paper-released)
#  Output: data/paper_split/train.parquet       (784 rows)
#          data/paper_split/validation.parquet  (196 rows)
#          data/paper_split/split_manifest.json (counts, seed, md5 of inputs/outputs)
#
#  Stratum = (level, geospatial_functions, territorial_division). The curated
#  corpus is perfectly balanced: 4 tiers x 7 functions x 7 divisions = 196 cells
#  of 5 pairs each. Exactly one pair per cell goes to the validation split, so
#  the 196 validation pairs are balanced on all three axes (49 per tier, 28 per
#  function, 28 per division) and the 784 training pairs keep 4 per cell.
#  Each cell is shuffled with its own random stream derived from the seed, so
#  the choice is independent across cells (a single seed applied to equal-sized
#  groups would repeat the same permutation in every cell).
#
#     uv run python core/make_paper_split.py            # writes the files
#     uv run python core/make_paper_split.py --dry-run  # only prints the counts
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.utils import Logger  # noqa: E402
from utils.experiment import (  # noqa: E402
    ID_COL, LEVEL_COL, FUNC_COL, DIV_COL, SOURCE_COL, PAPER_TRAIN, SPLIT_DIR, SPLIT_TRAIN,
    SPLIT_VALIDATION, normalize_levels, file_md5,
)

logger = Logger("split").setup_logging()

VAL_FRACTION = 0.20
TIER_ORDER = ["Fácil", "Médio", "Difícil", "Muito Difícil"]


def stratified_split(df: pl.DataFrame, val_fraction: float, seed: int):
    """One validation pair per (tier, function, division) cell, chosen with an
    independent random stream per cell. Falls back to proportional sampling if
    a cell does not have the expected size."""
    rng = np.random.default_rng(seed)
    cells = df.partition_by([LEVEL_COL, FUNC_COL, DIV_COL], as_dict=True, maintain_order=True)
    val_parts, train_parts = [], []
    for key in sorted(cells):                      # deterministic cell order
        g = cells[key].sort(ID_COL)
        n_val = max(1, int(round(len(g) * val_fraction)))
        perm = rng.permutation(len(g))             # independent draw per cell
        idx_val = sorted(perm[:n_val].tolist())
        idx_train = sorted(perm[n_val:].tolist())
        val_parts.append(g[idx_val]); train_parts.append(g[idx_train])
    val = pl.concat(val_parts).sort(ID_COL)
    train = pl.concat(train_parts).sort(ID_COL)
    return train, val


def describe(name: str, df: pl.DataFrame) -> dict:
    tiers = Counter(df[LEVEL_COL].to_list())
    funcs = Counter(df[FUNC_COL].to_list())
    divs = Counter(df[DIV_COL].to_list())
    logger.info(f"{name}: {len(df)} rows | tiers {dict(sorted(tiers.items()))} | "
                f"divisions {dict(sorted(divs.items()))}")
    return {"rows": len(df), "by_tier": dict(sorted(tiers.items())),
            "by_function": dict(sorted(funcs.items())), "by_division": dict(sorted(divs.items()))}


def main():
    ap = argparse.ArgumentParser(description="Build the 784/196 split described in the paper.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Print the counts; write nothing.")
    args = ap.parse_args()

    src = pl.read_parquet(PAPER_TRAIN)
    if SOURCE_COL in src.columns:
        src = src.filter(pl.col(SOURCE_COL) == "base_dataset")
    src = normalize_levels(src)
    logger.info(f"Curated corpus: {len(src)} pairs from {PAPER_TRAIN}")
    if len(src) != 980:
        logger.warning(f"Expected 980 curated pairs, found {len(src)}.")

    train, val = stratified_split(src, VAL_FRACTION, args.seed)
    assert len(train) + len(val) == len(src)
    assert not set(train[ID_COL].to_list()) & set(val[ID_COL].to_list())
    if len(src) == 980:
        assert len(val) == 196 and len(train) == 784, (len(train), len(val))
        for col, expect in ((LEVEL_COL, 49), (FUNC_COL, 28), (DIV_COL, 28)):
            counts = Counter(val[col].to_list())
            assert all(v == expect for v in counts.values()), (col, counts)
        logger.info("Balance check passed: 49 per tier, 28 per function, 28 per division in validation.")

    manifest = {
        "source_file": str(PAPER_TRAIN), "source_md5": file_md5(PAPER_TRAIN),
        "source_partition": "base_dataset", "seed": args.seed, "val_fraction": VAL_FRACTION,
        "stratified_by": [LEVEL_COL, FUNC_COL, DIV_COL],
        "train": describe("train", train), "validation": describe("validation", val),
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k in ("train", "validation")},
                         ensure_ascii=False, indent=1))
        return

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.write_parquet(SPLIT_TRAIN)
    val.write_parquet(SPLIT_VALIDATION)
    manifest["train_md5"] = file_md5(SPLIT_TRAIN)
    manifest["validation_md5"] = file_md5(SPLIT_VALIDATION)
    (SPLIT_DIR / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    logger.info(f"✔ Written: {SPLIT_TRAIN} ({len(train)}), {SPLIT_VALIDATION} ({len(val)}), "
                f"{SPLIT_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
