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
    DATASET_TYPE,
    TRANSLATOR_MODEL,
    UNMASKER_MODEL,
    NLP_VOCAB,
    NLP_LANGUAGE,
    SRC_LANG,
    TGT_LANG
)
from utils import Logger
from absl import app, flags
import random
import copy

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
import pyarrow as pa

# SYSTEM
import logging
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
            model=UNMASKER_MODEL,
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

def get_allowed_token_idxs(question, ignore_stopwords=False):
    nlp = get_nlp()
    doc = nlp(question)

    protected_idxs = set()

    for ent in doc.ents:
        if ent.label_ in nlp.pipe_labels['ner']:
            protected_idxs.update(range(ent.start, ent.end))

    allowed_token_idxs = []

    for token in doc:
        if token.is_alpha and token.i not in protected_idxs:
            if ignore_stopwords:
                if not token.is_stop and not token.is_punct:
                    allowed_token_idxs.append(token.i)
            else:
                allowed_token_idxs.append(token.i)

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

def save_data(augmented_rows):
    arrow_table = pa.Table.from_pylist(augmented_rows)
    part = ds.partitioning(
        pa.schema([("source", pa.string())]),
        flavor="hive"
    )
    ds.write_dataset(
        arrow_table,
        base_dir=DATASET_FULL_NAME,
        format=DATASET_TYPE,
        partitioning=part,
        existing_data_behavior="delete_matching"
    )

def tfidf_safe_swap(args):
    logger.info("TF-IDF Safe Swap: In Progress...")

    dataframe = args['data']
    tfidf_matrix = args['tfidf_matrix']
    feature_names = args['feature_names']
    top_lowest = args['top_lowest']

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
            allowed_idxs, doc = get_allowed_token_idxs(question)

            ranked = rank_allowed_tokens(doc, allowed_idxs, scores)
            candidates = [idx for _, idx in ranked[:top_lowest]]

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
        save_data(augmented_rows)
        logger.info("TF-IDF Safe Swap: Saved.")

    logger.info("TF-IDF Safe Swap: Done!")

def tfidf_safe_delete(args):
    logger.info("TF-IDF Safe Delete: In Progress...")

    dataframe = args['data']
    tfidf_matrix = args['tfidf_matrix']
    feature_names = args['feature_names']
    top_lowest = args['top_lowest']

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
            allowed_idxs, doc = get_allowed_token_idxs(question)

            ranked = rank_allowed_tokens(doc, allowed_idxs, scores)
            candidates = [idx for _, idx in ranked[:top_lowest]]
            idx1 = random.sample(candidates, 1)[0]

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
        save_data(augmented_rows)
        logger.info("TF-IDF Safe Delete: Saved.")

    logger.info("TF-IDF Safe Delete: Done!")

def tfidf_safe_insert(args):
    logger.info("TF-IDF Safe Insert: In Progress...")

    unmasker = get_unmasker()

    dataframe = args['data']
    tfidf_matrix = args['tfidf_matrix']
    feature_names = args['feature_names']
    top_highest = args['top_highest']
    top_lowest = args['top_lowest']

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
            allowed_idxs, doc = get_allowed_token_idxs(question)
            ranked = rank_allowed_tokens(doc, allowed_idxs, scores)

            candidates = [i for _, i in ranked[:top_lowest]]
            idx1 = random.sample(candidates, 1)[0]

            tokens = [t.text_with_ws for t in doc]
            position = random.randint(0, 1)

            masked_tokens = tokens.copy()
            masked_tokens.insert(idx1 + position, '[MASK]')
            masked_sentence = "".join(masked_tokens)

            predictions = unmasker(masked_sentence)

            valid_predictions = [
                pred for pred in predictions if pred['token_str'] not in tokens
            ]
            valid_predictions.sort(key=lambda x: x['score'], reverse=True)

            preds = [pred['token_str'] for pred in valid_predictions[:top_highest]]
            new_word = random.sample(preds, 1)[0]

            tokens.insert(idx1 + position, f"{new_word} ")
            new_question = "".join(tokens)

            augmented_rows.append({
                "id": row['id'],
                "question": new_question,
                "territorial_division": row['territorial_division'],
                "level": row['level'],
                "geospatial_functions": row['geospatial_functions'],
                "sql_code": row['sql_code'],
                "source": "insert"
            })
            progress.update(task_id, advance=1)

    if augmented_rows:
        logger.info("TF-IDF Safe Insert: Saving...")
        save_data(augmented_rows)
        logger.info("TF-IDF Safe Insert: Saved.")

    logger.info("TF-IDF Safe Insert: Done!")

def tfidf_safe_synonym_replacement(args):
    logger.info("TF-IDF Safe Synonym Replacement: In Progress...")

    dataframe = args['data']
    tfidf_matrix = args['tfidf_matrix']
    feature_names = args['feature_names']
    list_synonyms = args['synonyms']
    top_highest = args['top_highest']

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
            allowed_idxs, doc = get_allowed_token_idxs(question, ignore_stopwords=True)
            ranked = rank_allowed_tokens(doc, allowed_idxs, scores)
            tokens = [t.text_with_ws for t in doc]

            candidates = [i for _, i in ranked[-top_highest:]]
            if not candidates:
                raise Exception(f"There is no candidates for replacement.")

            synonyms = []
            idx = None
            while not synonyms and candidates:
                idx = random.sample(candidates, 1)[0]
                word = tokens[idx].strip()
                synonyms = list_synonyms[word]

                if not synonyms:
                    candidates.remove(idx)

            if synonyms:
                new_word = random.sample(synonyms, 1)[0]
                tokens[idx] = f"{new_word} "
                new_question = "".join(tokens)
            else:
                new_question = question

            augmented_rows.append({
                "id": row['id'],
                "question": new_question,
                "territorial_division": row['territorial_division'],
                "level": row['level'],
                "geospatial_functions": row['geospatial_functions'],
                "sql_code": row['sql_code'],
                "source": "synonym"
            })
            progress.update(task_id, advance=1)

        if augmented_rows:
            logger.info("TF-IDF Safe Synonym Replacement: Saving...")
            save_data(augmented_rows)
            logger.info("TF-IDF Safe Synonym Replacement: Saved.")

        logger.info("TF-IDF Safe Synonym Replacement: Done!")

def back_translation(args):
    logger.info("Back Translation: In Progress...")

    translator = get_translator()
    dataframe = args['data']

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[green]Processing...", total=dataframe.height)

        augmented_rows = []
        batch_size = 5
        for i in range(0, dataframe.height, batch_size):
            batch_df = dataframe[i: i + batch_size]
            batch_questions = batch_df['question'].to_list()

            pivot_results = translator(
                batch_questions,
                src_lang=SRC_LANG,
                tgt_lang=TGT_LANG,
                max_length=256,
                batch_size=batch_size
            )
            pivot_texts = [res['translation_text'] for res in pivot_results]

            back_results = translator(
                pivot_texts,
                src_lang=TGT_LANG,
                tgt_lang=SRC_LANG,
                max_length=256,
                batch_size=batch_size
            )
            back_texts = [res['translation_text'] for res in back_results]

            for idx, row in enumerate(batch_df.iter_rows(named=True)):
                augmented_rows.append({
                    "id": row['id'],
                    "question": back_texts[idx],
                    "territorial_division": row['territorial_division'],
                    "level": row['level'],
                    "geospatial_functions": row['geospatial_functions'],
                    "sql_code": row['sql_code'],
                    "source": "translate"
                })
            progress.update(task_id, advance=len(batch_df))

        if augmented_rows:
            logger.info("Back Translation: Saving...")
            save_data(augmented_rows)
            logger.info("Back Translation: Saved.")

        logger.info("Back Translation: Done!")


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
        'tfidf_matrix': tfidf_matrix,
        'top_lowest': 10,
        'top_highest': 10
    }

    if "all" in ops:
        ops = {"swap", "delete", "insert", "synonym", "translate"}
    if "swap" in ops:
        tfidf_safe_swap(copy.deepcopy(args_dict))
    if "delete" in ops:
        tfidf_safe_delete(copy.deepcopy(args_dict))
    if "insert" in ops:
        tfidf_safe_insert(copy.deepcopy(args_dict))
    if "synonym" in ops:
        tfidf_safe_synonym_replacement(copy.deepcopy(args_dict))
    if "translate" in ops:
        back_translation(copy.deepcopy(args_dict))


if __name__ == '__main__':
  app.run(main)