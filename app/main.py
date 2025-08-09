"""
Entry point for interacting with the trained medical QA model.

This script loads a previously trained TF‑IDF retrieval model from disk
and enters a simple interactive loop where the user can type medical
questions and receive answers.  To exit the loop, press Ctrl+C or
Ctrl+D, or type ``exit``.

Usage:
    python main.py --model-path tfidf_model.joblib
"""

from __future__ import annotations

import argparse
import sys 

from app.model import TFIDFRetrievalQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical question answering interactive bot")
    parser.add_argument("--model-path", type=str, default="tfidf_model.joblib",
                        help="Path to the trained model file (.joblib)")
    args = parser.parse_args()
    try:
        model = TFIDFRetrievalQA.load(args.model_path)
    except FileNotFoundError:
        print(f"Error: model file '{args.model_path}' not found. Run train_model.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    print("Medical QA bot loaded. Ask me a question! (type 'exit' to quit)")
    try:
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            answer = model.answer(query)
            print(f"Bot: {answer}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()