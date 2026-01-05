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
from scipy.sparse import load_npz, csr_matrix, save_npz
import polars as pl
from .global_variables import (
    DATASET_FULL_NAME,
    BASE_DATASET_FULL_NAME,
    TF_IDF_MATRIX_NAME,
    TF_IDF_FEATURES_NAME
)

# SYSTEM
import logging
import sys

class GeoDataset:
    """
    A utility class for accessing and retrieving dataset partitions and features.

    This class provides static methods to lazily load datasets and feature sets
    from Parquet files, handling file paths and partition logic abstractly.
    """
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
    """
    A utility class for managing the persistence of TF-IDF matrices and features.

    This class provides static methods to save and load TF-IDF sparse matrices
    and their corresponding feature dataframes to/from disk using predefined
    file paths.
    """
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

class Logger(logging.Logger):
    """
    A wrapper class to configure and manage application logging.

    This class initializes a logger instance with a specific name namespace
    ('app.{pid_name}') and configures output formatting to include the
    process identifier (pid_name). It ensures that logs are printed to
    stdout with a standardized timestamp and log level.

    Attributes:
        pid_name (str): The process identifier name used in log formatting.
    """
    def __init__(self, pid_name: str = "augmentation"):
        super().__init__(name=f"app.{pid_name}")
        self.__pid_name = pid_name

    def setup_logging(self):
        """
        Configures and retrieves the logger instance.

        This method sets the logging level to INFO, clears existing root handlers
        to prevent duplicate logs, and attaches a formatted StreamHandler to
        standard output (sys.stdout). It also disables propagation to prevent
        logs from bubbling up to the root logger.

        Returns:
            logging.Logger: The fully configured logger instance ready for use.

        Raises:
            Exception: If an error occurs during the configuration of handlers
                or formatters.
        """
        if not self.handlers:
            logging.getLogger().handlers.clear()
            self.setLevel(logging.INFO)

            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                f'%(asctime)s - %(levelname)s - {self.__pid_name} - %(message)s'
            )
            handler.setFormatter(formatter)

            self.handlers.clear()
            self.addHandler(handler)

        self.propagate = False
        return self

    def banner(self, title: str, width: int = 60):
        line = "=" * width
        self.info("")
        self.info(line)
        self.info(title.center(width))
        self.info(line)
        self.info("")

    def section(self, title: str):
        self.info("")
        self.info(f"[ {title.upper()} ]")