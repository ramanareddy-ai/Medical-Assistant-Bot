# Architecture

This document describes the high‑level architecture of the Medical Assistant QA bot.

## Components

1. **Data loader (`data_utils.py`)** – provides helper functions to load and clean the CSV dataset.  It normalises whitespace and groups duplicate questions with their answers for evaluation.

2. **Model (`model.py`)** – implements a TF‑IDF based retrieval model using scikit‑learn’s `TfidfVectorizer` and `NearestNeighbors`.  It exposes methods to train on a set of questions and answers, answer new queries, and save/load the trained model.

3. **Training script (`train_model.py`)** – loads the dataset, splits it into training and validation sets, trains the model, computes metrics (accuracy and mean reciprocal rank) on the validation set, and saves the resulting model to a `.joblib` file.

4. **Evaluation script (`evaluate_model.py`)** – loads a saved model and evaluates it on a held‑out test set.

5. **Interactive CLI (`main.py`)** – loads a trained model and starts a command line interface where the user can type medical questions and receive answers.

## Data flow

1. **Dataset ingestion** – the `load_dataset` function reads the `mle_screening_dataset.csv` file into a pandas DataFrame, cleans text and adds `question_clean` and `answer_clean` columns.

2. **Grouping** – `group_answers` collects multiple answers for identical cleaned questions into a list, reducing noise during evaluation.

3. **Training** – the `TFIDFRetrievalQA` model fits a TF‑IDF vectoriser on the training questions and builds a cosine‑similarity nearest neighbour index.  The original questions and their lists of answers are stored alongside the model.

4. **Inference** – for a new query, the CLI uses the loaded model to transform the query into a TF‑IDF vector, finds the nearest training question, and returns the associated answer(s).

## Files

- `ARCHITECTURE.md` – this file.
- `SAMPLE_OUTPUT.md` – examples of the model’s responses to sample queries.
- `mle_screening_dataset.csv` – the dataset of medical questions and answers used for training and evaluation.