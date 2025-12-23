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
from utils import Dataset, TfIdfVectorizer
from utils import (
    DATASET_FULL_NAME,
    TRANSLATOR_MODEL,
    SYNONYMS_MODEL,
    NLP_VOCAB,
    NLP_LANGUAGE
)
from absl import app
import random

# TRANSFORMERS
from transformers import pipeline
from transformers.utils import logging as hf_logging

# NLP
import nltk
import mlconjug3
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

# DATA MANIPULATION
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

def setup_logging(pid_name: str) -> logging.Logger:
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL)

    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        f'%(asctime)s - [{pid_name}] - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)

    logger.propagate = False

    return logger

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

def get_data():
    dataset_loader = Dataset()
    tfidf_vectorizer = TfIdfVectorizer()

    lf_data = dataset_loader.get_dataset(base_dataset=True)
    tfidf_matrix = tfidf_vectorizer.get_tfidf_matrix()
    lf_features = dataset_loader.get_features()

    return lf_data, tfidf_matrix, lf_features

def get_scores(tfidf_matrix, features, index):
    scores = {}
    feature_names = features["features"].to_list()
    for idx, score in zip(tfidf_matrix[index].indices, tfidf_matrix[index].data):
        word = feature_names[idx]
        scores[word] = score

    return scores

def get_question_scores(scores, question):
    words = question.lower().strip(".,?!").split()
    question_score = [(scores[word],word) for word in words if word in scores]

    return question_score

def get_synonyms_pt(word, top_n=10, target_similarity=0.25):
    nlp = get_nlp()
    conjugator = get_conjugator()

    token = nlp(word)[0]
    synsets = token._.wordnet.synsets()
    synonyms = set()

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
                    synonyms.add(gerund.lower())
                except Exception:
                    synonyms.add(synonym)
            else:
                synonyms.add(synonym)

    synonyms.discard(word)
    synonyms = list(synonyms)
    synonyms.sort(key=lambda x: x[0], reverse=True)

    return synonyms[:top_n]

def get_allowed_words_scores(question, question_score):
    nlp = get_nlp()

    doc = nlp(question)
    words = question.lower().strip(".,?!").split()
    prohibited_words = []

    for ent in doc.ents:
        ent_words = ent.text.lower().strip(".,?!").split()
        start_idx = words.index(ent_words[0])
        end_idx = words.index(ent_words[-1])

        prohibited_words.extend(words[start_idx:end_idx + 1])

    allowed_words = list(set(words) - set(prohibited_words))
    allowed_words_scores = [item for item in question_score if item[1] in allowed_words]
    allowed_words_scores.sort(key=lambda x: x[0], reverse=False)

    return allowed_words_scores

def get_words(question):
    clean_words = question.lower().strip(".,?!").split()
    words = question.lower().split()

    return words, clean_words

def random_swap(question, question_score):
    words, clean_words = get_words(question)
    allowed_words_scores = get_allowed_words_scores(question, question_score)

    target_word1 = allowed_words_scores[0][1]
    target_word2 = allowed_words_scores[1][1]

    target_idx1 = clean_words.index(target_word1)
    target_idx2 = clean_words.index(target_word2)

    clean_words[target_idx1], clean_words[target_idx2] = clean_words[target_idx2], clean_words[target_idx1]

    return " ".join(clean_words).capitalize()

def random_delete(question, question_score):
    words, clean_words = get_words(question)
    allowed_words_scores = get_allowed_words_scores(question, question_score)

    target_word = allowed_words_scores[0][1]
    target_idx = clean_words.index(target_word)
    del clean_words[target_idx]

    return " ".join(clean_words).capitalize()

def random_insertion(question, question_score):
    unmasker = get_unmasker()
    words, clean_words = get_words(question)
    allowed_words_scores = get_allowed_words_scores(question, question_score)

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
    clean_words.insert(target_idx + position, new_word)

    return " ".join(clean_words).capitalize()

def synonym_replacement(question, question_score):
    words, clean_words = get_words(question)
    allowed_words_scores = get_allowed_words_scores(question, question_score)

    target_word = allowed_words_scores[-1][1]
    target_idx = clean_words.index(target_word)

    word_clean = target_word.lower().strip(".,?!")
    synonyms = get_synonyms_pt(word_clean)

    if synonyms:
        clean_words[target_idx] = random.choice(synonyms)

    return " ".join(clean_words).capitalize()

def back_translation(question):
    translator = get_translator()

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
    logger = setup_logging("augmentation")
    logger.info("Initializing Data Augmentation.")
    logger.info("Initializing Models...")

    initialize_models()

    logger.info("Models Initialized.")
    logger.info("Collecting Base Dataset, TFIDF Matrix and Features...")

    lf_data, tfidf_matrix, lf_features = get_data()

    dataframe = lf_data.collect()
    features = lf_features.collect()

    logger.info("Base Dataset, TFIDF Matrix and Features collected!")
    logger.info(f"Base Dataset: {len(dataframe)} questions.")
    logger.info("Data Augmentation: In Progress...")

    augmented_rows = []
    for idx, row in enumerate(dataframe.iter_rows(named=True)):
        question = row['question']

        scores = get_scores(tfidf_matrix, features, idx)
        question_score = get_question_scores(scores, question)
        question_score.sort(key=lambda x: x[0], reverse=False)

        swap_question = random_swap(question=question, question_score=question_score.copy())
        delete_question = random_delete(question=question, question_score=question_score.copy())
        insertion_question = random_insertion(question=question, question_score=question_score.copy())
        synonym_question = synonym_replacement(question=question, question_score=question_score.copy())
        translated_question = back_translation(question=question)

        questions_map = {
            'base_dataset': row['question'],
            'random_swap': swap_question,
            'random_delete': delete_question,
            'random_insertion': insertion_question,
            'synonym_replacement': synonym_question,
            'back_translation': translated_question
        }

        augmented_rows = []
        for source, new_question in questions_map.items():
            augmented_rows.append({
                "id": row['id'],
                "question": new_question,
                "territorial_division": row['territorial_division'],
                "level": row['level'],
                "geospatial_functions": row['geospatial_functions'],
                "sql_code": row['sql_code'],
                "source": source
            })
        logger.info(f"Processed Row: {row['id']}")
    logger.info("Data Augmentation: Completed.")

    df_augmented = pl.DataFrame(augmented_rows)

    logger.info("Data Augmentation: Saving...")

    df_augmented.write_parquet(DATASET_FULL_NAME, partition_by="source")

    logger.info("Data Augmentation: Saved!!!")

if __name__ == '__main__':
  app.run(main)