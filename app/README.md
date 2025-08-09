# Medical Assistant QA Bot

This repository contains a simple medical question–answering system
developed for the MLE screening assignment.  The goal of the project is to
build a model capable of answering user queries about medical diseases
using a provided dataset of question–answer pairs.  The challenge
explicitly forbids the use of external AI services; the model here is
implemented entirely with open‑source Python libraries.

## Problem statement

The original assignment asks for a medical question–answering system that
can respond to user queries about diseases.  A CSV dataset with
16 406 rows of questions and answers is provided.  Each row in the file
contains a question and an answer text; many questions are repeated
multiple times with slightly different answers.  The tasks include:

* **Data preprocessing** – prepare the dataset for training, validation and
  testing.
* **Model training** – choose a suitable architecture and train a
  model to answer medical questions.
* **Model evaluation** – report metrics to quantify the model’s
  performance.
* **Example interactions** – demonstrate the model answering user
  questions.
* **Documentation** – describe assumptions, results and potential
  improvements.

## Dataset

The provided file `mle_screening_dataset.csv` contains two columns,
`question` and `answer`.  After cleaning and normalising white space there
are **14 981 unique questions** and a total of **16 406 question–answer
pairs**.  Most questions appear only once; on average there are **≈1.1
answers per unique question**.

## Approach

Given the constraints of the assignment (no large language models or
external AI services) and the nature of the dataset (largely
non‑duplicated question–answer pairs), I chose a **retrieval‑based
baseline** using a **Term Frequency–Inverse Document Frequency (TF‑IDF)
vectoriser** coupled with a **nearest neighbour search**.  The overall
workflow is:

1. **Loading and cleaning the data** – The questions and answers are
   stripped of extraneous whitespace.  Duplicate rows are preserved so
   that all available answers are considered during training.  A helper
   function groups duplicate questions and collects their answers into
   lists for evaluation.
2. **Data splitting** – Using scikit‑learn’s `train_test_split`, the
   unique questions are divided into training and validation sets (by
   default 90 %/10 %).  The associated answers are split in the same
   proportion.
3. **Model training** – A `TfidfVectorizer` with English stop‑words is
   fitted on the training questions.  The resulting sparse matrix is
   given to a `NearestNeighbors` model using cosine similarity.  For a
   new query, the model computes its TF‑IDF vector, retrieves the most
   similar training question and returns its answer(s).
4. **Evaluation** – On the validation set the model’s predictions are
   compared to the set of acceptable answers for each question.  Two
   metrics are reported:

   * **Accuracy** – the proportion of queries where the top retrieved
     answer matches one of the known answers exactly.
   * **Mean reciprocal rank (MRR)** – computed over the full ranked list
     returned by the nearest neighbour search.  This measures how high
     in the ranking a correct answer appears, even if it is not the very
     first.

   Because the dataset contains many unique questions with slightly
   different answer phrasings, exact match accuracy is low; the MRR
   metric gives additional insight into how close the retrieved answers
   are to a correct one.

5. **Interactive interface** – A simple CLI application (`main.py`)
   loads the trained model and allows users to ask questions.  The
   system retrieves the most similar question in the dataset and prints
   its answer.

The code is modular: helper functions for data loading and grouping are in
`data_utils.py`, the retrieval model is implemented in `model.py`, and
scripts for training and evaluation are in `train_model.py` and
`evaluate_model.py` respectively.  A dependency list is provided in
`requirements.txt`.

## How to run

Ensure you have Python 3.10+ installed.  Install the required
dependencies:

```bash
pip install -r requirements.txt
```

Train the model (this will also report validation metrics):

```bash
python -m medical_assistant_bot.train_model \
    --data-path path/to/mle_screening_dataset.csv \
    --model-out tfidf_model.joblib \
    --test-size 0.1
```

Evaluate an existing model on a held‑out test set:

```bash
python -m medical_assistant_bot.evaluate_model \
    --data-path path/to/mle_screening_dataset.csv \
    --model-path tfidf_model.joblib \
    --test-size 0.1
```

Start an interactive session with the trained bot:

```bash
python -m medical_assistant_bot.main --model-path tfidf_model.joblib
```

## Results

On a random 90 %/10 % train/validation split of the 14 981 unique
questions, the TF‑IDF retrieval model achieved the following scores:

| Metric    | Score |
|---------:|------:|
| Validation accuracy | **0.27 %** |
| Validation mean reciprocal rank (MRR) | **0.007** |

These numbers reflect the difficulty of the task when judged by exact
string matching.  Many questions have very similar wording but slightly
different answer phrasings; retrieving the exact original answer is
therefore challenging.  Nevertheless, qualitative inspection of the
retrieved answers shows that the model often returns relevant medical
information.

## Example interactions

Below are three example conversations with the trained bot.  The model
returns the answer associated with the most similar question in the
dataset.

1. **Glaucoma symptoms**

   *You:* `What are the symptoms of glaucoma?`

   *Bot:* “At first, open‑angle glaucoma has no symptoms. It causes no
   pain. Vision seems normal. Without treatment, people with glaucoma will
   slowly lose their peripheral, or side vision. They seem to be looking
   through a tunnel. Over time, straight‑ahead (central) vision may
   decrease until no vision remains.”

2. **Asthma prevention**

   *You:* `How can we prevent asthma?`

   *Bot:* “You can’t prevent asthma. However, you can take steps to
   control the disease and prevent its symptoms. For example: Learn about
   your asthma and ways to control it. Follow your written asthma action
   plan...”

3. **Migraine causes**

   *You:* `What causes migraine headaches?`

   *Bot:* “If you suffer from migraine headaches, you're not alone. About
   12 percent of the U.S. population gets them. Migraines are recurring
   attacks of moderate to severe pain. The pain is throbbing or pulsing...”

These examples illustrate that even without sophisticated generative
models the system can retrieve informative passages from the medical
dataset.

## Assumptions

* **Exact answer match as correctness criterion:**  For evaluation the
  top retrieved answer is considered correct only if it matches one of
  the known answers verbatim.  This is a very strict criterion and
  penalises legitimate paraphrases.
* **TF‑IDF features:**  No attempt was made to use more advanced
  embeddings or deep learning models, both because of the assignment
  restrictions and to keep the solution lightweight.
* **English stop‑word removal:**  The TF‑IDF vectoriser removes common
  English words.  This helps emphasise domain‑specific terms.
* **Single retrieved answer:**  The model returns the first answer
  associated with the most similar question.  In cases where multiple
  answers exist, all are concatenated with blank lines.

## Strengths and weaknesses

**Strengths:**

* Simple, transparent approach using widely available libraries (pandas,
  scikit‑learn).  No proprietary AI services required.
* Fast training and inference; the model can be trained in seconds on a
  laptop and answers queries in milliseconds.
* Easy to extend – for example by adjusting the number of neighbours
  considered or by switching to a more sophisticated text representation.

**Weaknesses:**

* Exact match evaluation results are very low; the model often retrieves
  relevant information but with slightly different wording, which is
  counted as incorrect.  More tolerant metrics (e.g. semantic
  similarity) would better reflect performance.
* TF‑IDF cannot capture synonyms or deep semantic relationships, so
  queries that use different vocabulary from the dataset questions may
  yield less relevant answers.
* The model does not generate answers; it merely retrieves existing
  answers from the dataset.  This limits its ability to combine
  information from multiple sources or provide concise summaries.

## Potential improvements

Several enhancements could improve the system’s effectiveness:

* **Use of word embeddings or sentence encoders:**  Replacing TF‑IDF
  vectors with embeddings from models like Word2Vec, GloVe or
  Sentence‑BERT (if permitted) would allow better matching of synonyms
  and semantic similarity.
* **Re‑ranking of candidates:**  Retrieve the top *k* nearest
  neighbours and use a secondary scoring function (e.g. BM25 or a
  logistic regression classifier) to re‑rank candidate answers.
* **Answer aggregation:**  For questions with multiple correct answers,
  aggregate the answers into a single response or pick the most
  comprehensive one.
* **User feedback loop:**  Allow users to indicate whether an answer was
  helpful and use this feedback to refine the retrieval process.

## A note on AI assistance

In accordance with the assignment requirements, this solution was
developed **without the assistance of any third‑party AI systems** (such
as OpenAI GPT models, Anthropic Claude, etc.).  Only open‑source
libraries and the provided dataset were used.