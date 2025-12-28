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

# UTILS
from rich.progress import Progress, MofNCompleteColumn, BarColumn, TextColumn, TimeRemainingColumn
from utils import Dataset, TfIdfVectorizer
from utils import (
    DATASET_FULL_NAME,
    TRANSLATOR_MODEL,
    SYNONYMS_MODEL,
    NLP_VOCAB,
    NLP_LANGUAGE
)
from utils import Logger
from absl import app, flags
import random
import copy
import re

# TRANSFORMERS
from transformers import pipeline
from transformers.utils import logging as hf_logging

# NLP
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import nltk
import mlconjug3
import spacy

# DATA MANIPULATION & TYPES
from collections import defaultdict
import pyarrow.dataset as ds
import polars as pl

# SYSTEM
import logging
import sys
import warnings

logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

_translator = None
_unmasker = None
_nlp = None
_conjugator = None

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "ops",
    ["all"],
    "Augmentation operations: swap, delete, insert, synonym, all"
)
logger = Logger(pid_name="Data Augmentation").setup_logging()

def initialize_models(device: int = 0) -> None:
    nltk.download('wordnet')
    nltk.download('omw')

    get_translator(device)
    get_unmasker()
    get_nlp()
    get_conjugator()

def get_translator(device: int = 0):
    global _translator

    if _translator is None:
        _translator = pipeline(
            "translation",
            model=TRANSLATOR_MODEL,
            device=device,
        )
    return _translator

def get_unmasker():
    global _unmasker

    if _unmasker is None:
        _unmasker = pipeline(
            "fill-mask",
            model=SYNONYMS_MODEL,
            top_k=10
        )
    return _unmasker

def get_nlp():
    global _nlp

    if _nlp is None:
        _nlp = spacy.load(NLP_VOCAB)
        _nlp.add_pipe("spacy_wordnet")
    return _nlp

def get_conjugator():
    global _conjugator

    if _conjugator is None:
        _conjugator = mlconjug3.Conjugator(language=NLP_LANGUAGE)
    return _conjugator

def get_scores(tfidf_matrix, feature_names, index):
    scores = []
    for idx, score in zip(tfidf_matrix[index].indices, tfidf_matrix[index].data):
        word = feature_names[idx]
        scores.append((score, word))
    scores.sort(key=lambda x: x[0], reverse=False)

    return scores

def get_allowed_token_idxs(question, question_score):
    nlp = get_nlp()
    doc = nlp(question)

    protected_idxs = set()

    for ent in doc.ents:
        if ent.label_ in nlp.pipe_labels['ner']:
            protected_idxs.update(range(ent.start, ent.end))

    allowed_token_idxs = [
        tok.i
        for tok in doc
        if tok.is_alpha and tok.i not in protected_idxs
    ]

    return allowed_token_idxs, doc

def rank_allowed_tokens(doc, allowed_idxs, question_score):
    score_map = {w: s for s, w in question_score}

    ranked = [
        (score_map.get(doc[i].lemma_.lower(), float("inf")), i)
        for i in allowed_idxs
    ]

    ranked.sort(key=lambda x: x[0])
    return ranked

def get_synonyms_pt(feature_names, target_similarity=0.25):
    nlp = get_nlp()
    conjugator = get_conjugator()
    synonyms = defaultdict(set)

    for word in feature_names:
        token = nlp(word)[0]
        synsets = token._.wordnet.synsets()

        is_gerund = "VerbForm=Ger" in token.morph

        for synset in synsets:
            for lemma in synset.lemmas(lang='por'):
                synonym_doc = nlp(lemma.name())
                if not synonym_doc.vector_norm:
                    continue

                similarity = token.similarity(synonym_doc)
                if similarity < target_similarity:
                    continue

                synonym = lemma.name().lower()
                if is_gerund:
                    try:
                        conjugation = conjugator.conjugate(synonym)
                        gerund = conjugation.conjug_info['Gerúndio']['Gerúndio Gerúndio'][0]
                        synonyms[word].add(gerund.lower())
                    except Exception:
                        synonyms[word].add(synonym)
                else:
                    synonyms[word].add(synonym)

        synonyms[word].discard(word)

    return synonyms

def tfidf_safe_swap(args):
    logger.info("TF-IDF Safe Swap: In Progress...")

    dataframe = args['data']
    tfidf_matrix = args['tfidf_matrix']
    feature_names = args['feature_names']

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[green]Processing...", total=dataframe.height)

        augmented_rows = []
        for idx, row in enumerate(dataframe.iter_rows(named=True)):
            question = row['question']
            scores = get_scores(tfidf_matrix, feature_names, idx)
            allowed_idxs, doc = get_allowed_token_idxs(question, scores)

            ranked = rank_allowed_tokens(doc, allowed_idxs, scores)
            k = min(5, len(ranked))
            candidates = [i for _, i in ranked[:k]]

            idx1, idx2 = random.sample(candidates, 2)

            tokens = [t.text_with_ws for t in doc]
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]

            new_question = "".join(tokens)

            augmented_rows.append({
                "id": row['id'],
                "question": new_question,
                "territorial_division": row['territorial_division'],
                "level": row['level'],
                "geospatial_functions": row['geospatial_functions'],
                "sql_code": row['sql_code'],
                "source": "swap"
            })
            progress.update(task_id, advance=1)

    if augmented_rows:
        logger.info("TF-IDF Safe Swap: Saving...")
        df_augmented = pl.DataFrame(augmented_rows)
        arrow_table = df_augmented.to_arrow()
        ds.write_dataset(
            arrow_table,
            base_dir=DATASET_FULL_NAME,
            format="parquet",
            partitioning=["source"],
            existing_data_behavior="delete_matching"
        )
        logger.info("TF-IDF Safe Swap: Saved.")

    logger.info("TF-IDF Safe Swap: Done!")

def tfidf_safe_delete(args):
    logger.info("TF-IDF Safe Delete: In Progress...")

    dataframe = args['data']
    tfidf_matrix = args['tfidf_matrix']
    feature_names = args['feature_names']

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[green]Processing...", total=dataframe.height)

        augmented_rows = []
        for idx, row in enumerate(dataframe.iter_rows(named=True)):
            question = row['question']
            scores = get_scores(tfidf_matrix, feature_names, idx)
            allowed_idxs, doc = get_allowed_token_idxs(question, scores)

            ranked = rank_allowed_tokens(doc, allowed_idxs, scores)
            idx1 = ranked[0][1]
            tokens = [t.text_with_ws for t in doc]
            del tokens[idx1]

            new_question = "".join(tokens)

            augmented_rows.append({
                "id": row['id'],
                "question": new_question,
                "territorial_division": row['territorial_division'],
                "level": row['level'],
                "geospatial_functions": row['geospatial_functions'],
                "sql_code": row['sql_code'],
                "source": "delete"
            })
            progress.update(task_id, advance=1)

    if augmented_rows:
        logger.info("TF-IDF Safe Delete: Saving...")

        df_augmented = pl.DataFrame(augmented_rows)
        arrow_table = df_augmented.to_arrow()
        ds.write_dataset(
            arrow_table,
            base_dir=DATASET_FULL_NAME,
            format="parquet",
            partitioning=["source"],
            existing_data_behavior="delete_matching"
        )
        logger.info("TF-IDF Safe Delete: Saved.")

    logger.info("TF-IDF Safe Delete: Done!")

def random_insertion(args):
    unmasker = get_unmasker()
    clean_words = args['clean_words']
    words = args['words']
    allowed_words_scores = args['allowed_words_scores']

    target_word = allowed_words_scores[-1][1]
    target_idx = clean_words.index(target_word)
    masked_words = clean_words.copy()

    position = random.randint(0, 1)
    masked_words.insert(target_idx + position, '[MASK]')
    masked_sentence = " ".join(masked_words)

    predictions = unmasker(masked_sentence)

    valid_predictions = [
        pred for pred in predictions
        if pred['token_str'].strip() and pred['token_str'].isalpha() and pred['token_str'] not in clean_words
    ]
    valid_predictions.sort(key=lambda x: x['score'], reverse=False)
    new_word = valid_predictions[-1]['token_str']
    words.insert(target_idx + position, new_word)

    return " ".join(words).capitalize()

def synonym_replacement(args):
    clean_words = args['clean_words']
    words = args['words']
    allowed_words_scores = args['allowed_words_scores']
    synonyms = args['synonyms']

    target_word = allowed_words_scores[-1][1]
    target_idx = clean_words.index(target_word)
    word_clean = target_word.lower().strip(".,?!")

    if synonyms[word_clean]:
        synonym_list = list(synonyms[word_clean])
        words[target_idx] = random.choice(synonym_list)

    return " ".join(words).capitalize()

def back_translation(args):
    translator = get_translator()

    question = args['question']

    src_lang = 'por_Latn'
    tgt_lang = 'jpn_Jpan'

    pivot = translator(
        question,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=256,
    )[0]["translation_text"]

    back = translator(
        pivot,
        src_lang=tgt_lang,
        tgt_lang=src_lang,
        max_length=256,
    )[0]["translation_text"]

    return back

def main(argv):
    del argv

    ops = set(FLAGS.ops)

    logger.info("Initializing Data Augmentation.")
    logger.info("Initializing Models.")

    initialize_models()

    logger.info("Models Initialized.")
    logger.info("Collecting Base Dataset, TFIDF Matrix and Features.")

    dataset_loader = Dataset()
    tfidf_vectorizer = TfIdfVectorizer()

    lf_data = dataset_loader.get_dataset(base_dataset=True)
    tfidf_matrix = tfidf_vectorizer.get_tfidf_matrix()
    lf_features = dataset_loader.get_features()

    dataframe = lf_data.collect()
    feature_names = lf_features.collect()["features"].to_list()
    synonyms = get_synonyms_pt(feature_names=feature_names)

    logger.info("Base Dataset, TFIDF Matrix and Features collected!")
    logger.info(f"Base Dataset: {len(dataframe)} questions.")

    args_dict = {
        'data': dataframe,
        'feature_names': feature_names,
        'synonyms': synonyms,
        'tfidf_matrix': tfidf_matrix

    }

    if "all" in ops:
        ops = {"swap", "delete", "insert", "synonym"}
    if "swap" in ops:
        tfidf_safe_swap(args_dict)
    if "delete" in ops:
        tfidf_safe_delete(args_dict)


if __name__ == '__main__':
  app.run(main)