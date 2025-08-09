"""
Script to train and evaluate the TF‑IDF retrieval model.

This script loads the provided CSV dataset, performs a random train/validation
split, trains the :class:`TFIDFRetrievalQA` model on the training portion,
evaluates it on the validation set and saves the fitted model to disk.

Running this file directly will produce a report of the validation accuracy
and mean reciprocal rank (MRR), and write the trained model to
``./tfidf_model.joblib`` in the working directory.

Example:
    python train_model.py --data-path ../mle_screening_dataset.csv --model-out tfidf_model.joblib
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from app.data_utils import load_dataset, group_answers
from app.model import TFIDFRetrievalQA


def compute_accuracy_and_mrr(model: TFIDFRetrievalQA, questions: List[str], true_answers: List[List[str]]) -> tuple[float, float]:
    """Compute exact match accuracy and mean reciprocal rank on a dataset.

    For each question in the dataset, the model retrieves the most similar
    training question.  If the first answer of that retrieved question
    matches any of the known true answers exactly, it is counted as a
    correct prediction.  Additionally, the mean reciprocal rank (MRR) is
    computed over the full ranked list returned by the nearest neighbour
    search with ``top_k=len(model.questions)``.

    Args:
        model: A trained TFIDFRetrievalQA instance.
        questions: List of query strings from the validation set.
        true_answers: List of lists of acceptable answers corresponding
            to each query.  Each ``true_answers[i]`` may contain multiple
            correct answers for a given question.

    Returns:
        A tuple ``(accuracy, mrr)``.
    """
    if not questions:
        return 0.0, 0.0
    correct = 0
    reciprocal_ranks = []
    # Prepare the full neighbour search once to reuse within loop.
    # We request all neighbours to compute MRR.
    # Precompute the matrix of training answer strings for comparison.
    for q, answers in zip(questions, true_answers):
        # compute neighbours for query
        if model.vectorizer is None or model.nn is None:
            raise RuntimeError("Model must be trained before evaluation.")
        q_vec = model.vectorizer.transform([q])
        distances, indices = model.nn.kneighbors(q_vec, n_neighbors=len(model.questions))
        # distances shape: (1, n), indices shape: (1, n)
        # Flatten arrays
        idxs = indices[0]
        found_correct = False
        # Check top prediction for accuracy
        top_idx = idxs[0]
        pred_ans = model.answers[top_idx][0]  # use first answer
        if any(pred_ans == ans for ans in answers):
            correct += 1
            found_correct = True
        # Compute reciprocal rank
        # Find the rank of the first occurrence where any answer matches
        rr = 0.0
        for rank, idx in enumerate(idxs, start=1):
            candidate_answers = model.answers[idx]
            # if any candidate answer matches any true answer
            if any(ans == candidate for ans in answers for candidate in candidate_answers):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    accuracy = correct / len(questions)
    mrr = float(np.mean(reciprocal_ranks))
    return accuracy, mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TF‑IDF retrieval model for medical QA")
    parser.add_argument("--data-path", type=str, default="../mle_screening_dataset.csv",
                        help="Path to the CSV dataset file")
    parser.add_argument("--model-out", type=str, default="tfidf_model.joblib",
                        help="Filename where the trained model will be saved")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Proportion of the data to use as the validation set")
    args = parser.parse_args()

    # load and preprocess data
    df = load_dataset(args.data_path)
    grouped = group_answers(df)
    questions = grouped["question_clean"].tolist()
    answers = grouped["answers"].tolist()

    # split into train and validation
    train_questions, val_questions, train_answers, val_answers = train_test_split(
        questions, answers, test_size=args.test_size, random_state=42, shuffle=True
    )

    # train model
    model = TFIDFRetrievalQA()
    model.train(train_questions, train_answers)

    # evaluate on validation set
    acc, mrr = compute_accuracy_and_mrr(model, val_questions, val_answers)
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation MRR: {mrr:.4f}")

    # save model
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()