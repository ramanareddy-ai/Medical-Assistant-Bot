"""
Simple TF‑IDF based retrieval model for question answering.

The model uses a TF‑IDF vectorizer to represent questions from the training
set and a nearest neighbour search to retrieve the most similar question
for a given user query.  The answer(s) associated with the retrieved
question are then returned.  This approach is scalable to relatively
large datasets and does not require any external AI services or large
pretrained models.

Usage:
    from medical_assistant_bot.model import TFIDFRetrievalQA
    from medical_assistant_bot.data_utils import load_dataset, group_answers

    df = load_dataset('path/to/csv')
    grouped = group_answers(df)
    model = TFIDFRetrievalQA()
    model.train(grouped['question_clean'].tolist(), grouped['answers'].tolist())
    answer = model.answer("What causes glaucoma?")
    model.save('model.joblib')
    # later...
    model2 = TFIDFRetrievalQA.load('model.joblib')
    answer = model2.answer("How do you prevent glaucoma?")

The underlying TF‑IDF matrix and vectorizer are saved to disk using
joblib, allowing fast reloading without retraining.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import List, Sequence, Optional


class TFIDFRetrievalQA:
    """A TF‑IDF based question answering model.

    Attributes:
        vectorizer: The fitted :class:`TfidfVectorizer` used to convert
            questions into sparse vectors.
        nn: A fitted :class:`NearestNeighbors` instance for performing
            nearest neighbour search.
        questions: The list of training questions used to build the TF‑IDF
            matrix.  This is used only for debugging or inspection.
        answers: A list of answer lists corresponding to each training
            question.
    """

    def __init__(self, *, n_neighbors: int = 1) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nn: Optional[NearestNeighbors] = None
        self.questions: List[str] = []
        self.answers: List[List[str]] = []
        self.n_neighbors = n_neighbors

    def train(self, questions: Sequence[str], answers: Sequence[Sequence[str]]) -> None:
        """Fit the TF‑IDF vectorizer and nearest neighbour model on the data.

        Args:
            questions: A sequence of cleaned question strings.
            answers: A sequence of answer lists.  ``answers[i]`` contains one
                or more strings corresponding to the answers for
                ``questions[i]``.
        """
        if len(questions) != len(answers):
            raise ValueError("questions and answers must have the same length")
        # store copies of data
        self.questions = list(questions)
        self.answers = [list(ans_list) for ans_list in answers]
        # fit vectorizer on all questions
        self.vectorizer = TfidfVectorizer(stop_words="english")
        X = self.vectorizer.fit_transform(self.questions)
        # fit nearest neighbor model
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine")
        self.nn.fit(X)

    def answer(self, query: str, *, top_k: int = 1) -> str:
        """Return the most relevant answer to a user query.

        Args:
            query: The user input string.
            top_k: Number of candidate answers to return.  If ``top_k > 1``,
                the answers from the ``top_k`` most similar questions are
                concatenated (separated by blank lines) in the result.

        Returns:
            A single answer string composed of one or more answers from
            the retrieved questions.
        """
        if self.vectorizer is None or self.nn is None:
            raise RuntimeError("Model must be trained or loaded before calling answer().")
        # transform query into vector
        q_vec = self.vectorizer.transform([query])
        # retrieve nearest neighbours
        distances, indices = self.nn.kneighbors(q_vec, n_neighbors=top_k)
        # distances shape: (1, k)
        # indices shape: (1, k)
        results: List[str] = []
        for idx in indices[0]:
            ans_list = self.answers[idx]
            # pick the first answer for simplicity; multiple answers could be
            # concatenated or randomly selected.  Here we join them with a
            # blank line if there are multiple.
            results.append("\n".join(ans_list))
        # join results if top_k > 1
        return "\n\n".join(results)

    def save(self, path: str) -> None:
        """Persist the model to disk.

        The vectorizer, nearest neighbour model, questions and answers are
        serialized using joblib.  The saved file can be reloaded using
        :meth:`load`.

        Args:
            path: Destination filename.  Should end with ``.joblib``.
        """
        if self.vectorizer is None or self.nn is None:
            raise RuntimeError("Model must be trained before saving.")
        joblib.dump({
            'vectorizer': self.vectorizer,
            'nn': self.nn,
            'questions': self.questions,
            'answers': self.answers,
            'n_neighbors': self.n_neighbors,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'TFIDFRetrievalQA':
        """Load a previously saved model from disk.

        Args:
            path: Path to the ``.joblib`` file created by :meth:`save`.

        Returns:
            An instance of :class:`TFIDFRetrievalQA` with the loaded state.
        """
        data = joblib.load(path)
        model = cls(n_neighbors=data.get('n_neighbors', 1))
        model.vectorizer = data['vectorizer']
        model.nn = data['nn']
        model.questions = data['questions']
        model.answers = data['answers']
        return model