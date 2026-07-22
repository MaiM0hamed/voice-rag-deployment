"""Interactive CLI for the Electro Pi RAG pipeline.

Run:
    python app.py

Loads (or builds) the Chroma vector store once, then answers free-form
questions in a loop until the user types ``exit`` (or ``quit``). Each answer is
shown with only the citations the answer actually referenced; out-of-scope or
unverifiable questions are refused without fabricating an answer.

Prerequisites (same as the rest of the section):
    pip install -r requirements.txt
    ollama serve && ollama pull qwen2.5:1.5b
"""

from __future__ import annotations

import logging
import os
import sys

from src.config import settings
from src.ingest import build_vectorstore, chunk_documents, load_documents, load_vectorstore
from src.rag_chain import answer_question

logger = logging.getLogger("app")

_EXIT_WORDS = {"exit", "quit", ":q", "q"}
_BANNER = "-" * 32


def _print_response(response) -> None:
    """Render one answer and its citations to stdout."""
    print()
    print(f"Answer: {response.answer}")
    if response.citations:
        print("\nCitations:")
        for citation in response.citations:
            print(f"  {citation}")
    print(_BANNER)


def main() -> int:
    """Start the interactive question/answer loop."""
    # Quiet by default so the prompt stays readable; set LOG_LEVEL=INFO to trace.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING").upper(),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print(_BANNER)
    print("Electro Pi RAG — interactive CLI")
    print("Type a question, or 'exit' to quit.")
    print(_BANNER)

    # Load the corpus and vector store once so the BM25 index and embeddings are
    # built a single time and reused across every question in the session.
    docs = load_documents(settings.data_dir)
    chunks = chunk_documents(docs)
    if settings.persist_dir.exists():
        store = load_vectorstore()
    else:
        print("No vector store found; building it now...")
        store = build_vectorstore(chunks)

    while True:
        print("\nAsk a question:")
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not question:
            continue
        if question.lower() in _EXIT_WORDS:
            print("Bye.")
            return 0

        try:
            response = answer_question(store, question, all_chunks=chunks)
        except RuntimeError as exc:
            print(f"\nError: {exc}")
            continue

        _print_response(response)


if __name__ == "__main__":
    sys.exit(main())
