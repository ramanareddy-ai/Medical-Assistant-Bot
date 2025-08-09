"""
Evaluate a previously trained TF‑IDF retrieval model on a held‑out test set.

This script loads a saved model, splits the dataset into train and test
partitions using the same random seed as the training script and reports
accuracy and mean reciprocal rank on the test portion.  By default, the
test size is 10 % of the full dataset.

Note that the metrics reported here are indicative of the model's ability to
recover exactly matching answers from the dataset.  Due to the diversity of
answer phrasing in the dataset, exact match accuracy can be very low even
for reasonable retrieval systems.
"""

from __future__ import annotations

import argparse
from typing import List

from sklearn.model_selection import train_test_split

from app.data_utils import load_dataset, group_answers
from app.model import TFIDFRetrievalQA
from app.train_model import compute_accuracy_and_mrr 


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TF‑IDF medical QA model")
    parser.add_argument("--data-path", type=str, default="../mle_screening_dataset.csv",
                        help="Path to the CSV dataset file")
    parser.add_argument("--model-path", type=str, default="tfidf_model.joblib",
                        help="Path to the trained model (.joblib) file")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Proportion of the data to use as the test set")
    args = parser.parse_args()
    # load dataset
    df = load_dataset(args.data_path)
    grouped = group_answers(df)
    questions = grouped["question_clean"].tolist()
    answers = grouped["answers"].tolist()
    # split into train and test (discard training portion)
    _, test_questions, _, test_answers = train_test_split(
        questions, answers, test_size=args.test_size, random_state=42, shuffle=True
    )
    # load model
    model = TFIDFRetrievalQA.load(args.model_path)
    # compute metrics
    acc, mrr = compute_accuracy_and_mrr(model, test_questions, test_answers)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()