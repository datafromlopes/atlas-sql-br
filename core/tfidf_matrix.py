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
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import GeoDataset
from polars import LazyFrame, DataFrame
from scipy.sparse import save_npz
from utils import PROJECT_ROOT
from absl import app


def get_tfidf_matrix(dataframe: DataFrame) -> tuple:
    corpus = dataframe['question'].to_list()

    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    return sparse_matrix, features

def main(argv):
    dataset_loader = GeoDataset()
    dataframe = dataset_loader.get_dataset(base_dataset=True)
    sparse_matrix, features = get_tfidf_matrix(dataframe)
    df_features = DataFrame({'features': features})

    df_features.write_parquet(f"{PROJECT_ROOT}/data/features.parquet")
    save_npz(f"{PROJECT_ROOT}/data/sparse_matrix.npz", sparse_matrix)


if __name__ == '__main__':
  app.run(main)