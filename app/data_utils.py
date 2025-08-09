"""
Utilities for loading and preprocessing the medical question‑answer dataset.

The dataset consists of pairs of questions and answers.  This module provides
functions for reading the dataset from a CSV file, performing basic text
cleaning, and optionally grouping answers for identical questions.

These helper functions are used by the training and evaluation scripts to
prepare the data for model fitting and testing.
"""

from __future__ import annotations

import pandas as pd
import re
from typing import List

def clean_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces in a string.

    Args:
        text: Arbitrary text to clean.

    Returns:
        The cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)
    # collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the question–answer dataset and perform basic cleaning.

    The CSV is expected to have at least two columns named ``question`` and
    ``answer``.  Two additional columns ``question_clean`` and ``answer_clean``
    are added to the returned data frame containing normalized versions of
    the text.  Duplicate rows are preserved so that the model sees all
    available answers during training.

    Args:
        csv_path: Path to the CSV file on disk.

    Returns:
        A pandas DataFrame with cleaned columns.
    """
    df = pd.read_csv(csv_path)
    # ensure required columns exist
    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'question' and 'answer' columns")
    # clean text columns
    df = df.copy()
    df["question_clean"] = df["question"].apply(clean_text)
    df["answer_clean"] = df["answer"].apply(clean_text)
    return df


def group_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Group multiple answers for identical cleaned questions into lists.

    Many questions appear multiple times in the dataset with slightly
    different answer phrasings.  Grouping answers together simplifies
    evaluation: a retrieved answer is considered correct if it matches
    any of the known answers for a question.  The original ``question``
    and ``answer`` columns are not retained in the grouped frame to
    prevent downstream confusion.

    Args:
        df: A DataFrame returned from :func:`load_dataset`.

    Returns:
        A DataFrame with columns ``question_clean`` and ``answers``.  The
        ``answers`` column contains lists of strings.
    """
    if "question_clean" not in df or "answer_clean" not in df:
        raise ValueError("DataFrame must contain 'question_clean' and 'answer_clean' columns")
    grouped = df.groupby("question_clean")
    result = grouped["answer_clean"].apply(list).reset_index()
    result = result.rename(columns={"answer_clean": "answers"})
    return result