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

#----------------------------------------------------------------------------------------------------------------
# GENERAL PARAMETERS
#----------------------------------------------------------------------------------------------------------------
PROJECT_NAME = config['general']['project_name']
PROJECT_PATH = PROJECT_ROOT

#----------------------------------------------------------------------------------------------------------------
# DATASET PARAMETERS
#----------------------------------------------------------------------------------------------------------------
DATASET_TYPE =  config["dataset"]["type"]
DATASET_FULL_NAME = f"{PROJECT_ROOT}/data/{config['dataset']['full']}"

#----------------------------------------------------------------------------------------------------------------
# TF-IDF PARAMETERS
#----------------------------------------------------------------------------------------------------------------
TF_IDF_MATRIX_NAME = f"{PROJECT_ROOT}/data/{config['tf_idf_matrix']['name']}.{config['tf_idf_matrix']['type']}"
TF_IDF_FEATURES_NAME = f"{PROJECT_ROOT}/data/{config['tf_idf_features']['name']}.{config['tf_idf_features']['type']}"

#----------------------------------------------------------------------------------------------------------------
# DATA AUGMENTATION PARAMETERS
#----------------------------------------------------------------------------------------------------------------
TRANSLATOR_MODEL = config['data_aug_params']['translator_model']
UNMASKER_MODEL = config['data_aug_params']['unmasker_model']
NLP_VOCAB = config['data_aug_params']['nlp_vocab']
NLP_LANGUAGE = config['data_aug_params']['nlp_language']
SRC_LANG = config['data_aug_params']['src_lang']
TGT_LANG = config['data_aug_params']['tgt_lang']

#----------------------------------------------------------------------------------------------------------------
# SQL VALIDATION PARAMETERS
#----------------------------------------------------------------------------------------------------------------
INPUT_FILE  = f"{PROJECT_ROOT}/{config['sql_validation']['input_file']}"
OUTPUT_FILE = f"{PROJECT_ROOT}/{config['sql_validation']['output_file']}"