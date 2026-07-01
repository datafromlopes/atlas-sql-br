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
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

PROJECT_PATH = PROJECT_ROOT
PROJECT_NAME = config['project_name']
DATASET_PATH = f"{PROJECT_PATH}/data"
DATASET_VALIDATION_NAME = config['dataset_validation_name']
DATASET_NAME = config['dataset_name']
PREDS_DIR = f"{PROJECT_PATH}/data/predictions"