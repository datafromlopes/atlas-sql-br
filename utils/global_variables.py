# coding=utf-8
# Copyright (C) 2025  Diego Lopes
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

DATASET_TYPE =  config["dataset"]["type"]
DATASET_FULL_NAME = f"{PROJECT_ROOT}/data/{config['dataset']['full']}"
BASE_DATASET_FULL_NAME = f"{PROJECT_ROOT}/data/{config['dataset']['base']}.{config['dataset']['type']}"

TF_IDF_MATRIX_NAME = f"{PROJECT_ROOT}/data/{config['tf_idf_matrix']['name']}.{config['tf_idf_matrix']['type']}"
TF_IDF_FEATURES_NAME = f"{PROJECT_ROOT}/data/{config['tf_idf_features']['name']}.{config['tf_idf_features']['type']}"

TRANSLATOR_MODEL = config['data_aug_params']['translator_model']
UNMASKER_MODEL = config['data_aug_params']['unmasker_model']
NLP_VOCAB = config['data_aug_params']['nlp_vocab']
NLP_LANGUAGE = config['data_aug_params']['nlp_language']
