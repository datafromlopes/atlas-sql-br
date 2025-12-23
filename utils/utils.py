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
from scipy.sparse import load_npz, csr_matrix, save_npz
import polars as pl
from .global_variables import (
    DATASET_FULL_NAME,
    BASE_DATASET_FULL_NAME,
    TF_IDF_MATRIX_NAME,
    TF_IDF_FEATURES_NAME
)


class Dataset:

    @staticmethod
    def get_features() -> pl.LazyFrame:
        return pl.scan_parquet(TF_IDF_FEATURES_NAME)

    @staticmethod
    def get_dataset(base_dataset=False, partition=None) -> pl.LazyFrame:
        """Get the dataset.

        Arguments:
            base_dataset (bool, optional): if True, returns the base dataset. Default is False.
            partition (str, optional): if not None, returns the partition. Default is None.

        Returns:
           pl.LazyFrame: Polars LazyFrame
        """
        if base_dataset:
            return pl.scan_parquet(BASE_DATASET_FULL_NAME)

        if partition:
            file = f"{DATASET_FULL_NAME}/source={partition}"
            return pl.scan_parquet(file)

        return pl.scan_parquet(DATASET_FULL_NAME)


class TfIdfVectorizer:
    @staticmethod
    def get_tfidf_matrix() -> csr_matrix:
        """Get the TF-IDF matrix.

        Returns:
            scipy.csr_matrix: Sparse Matrix
        """
        return load_npz(TF_IDF_MATRIX_NAME)

    @staticmethod
    def save_tfidf_matrix(sparse_matrix: csr_matrix, features: pl.DataFrame) -> None:
        """Save the TF-IDF matrix and features."""
        save_npz(TF_IDF_MATRIX_NAME, sparse_matrix)
        features.write_parquet(TF_IDF_FEATURES_NAME)