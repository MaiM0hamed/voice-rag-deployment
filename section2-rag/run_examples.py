"""Run the example questions end-to-end and write `outputs/example_runs.md`.

This is the script that populates the deliverable required by the assessment
("provide 3 example questions and the actual answers your pipeline produced").
It ingests the corpus if needed, runs each question, and records the answer,
citations and retrieved context.

Question 4 is deliberately out-of-scope to demonstrate the refusal path.

Run:
    ollama serve && ollama pull qwen2.5:1.5b
    python -m pip install -r requirements.txt
    python run_examples.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from src.config import settings
from src.ingest import build_vectorstore, chunk_documents, load_documents, load_vectorstore
from src.rag_chain import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_examples")

QUESTIONS: list[str] = [
    # In-scope, single-document answers.
    "How long do I have to return a development board, and when will I get my refund?",
    "What are the shipping fees for an order under 1,500 EGP delivered in Cairo?",
    "What happens if I exceed the API rate limit on the free tier?",
    # Out-of-scope: must trigger the refusal guard.
    "What is Electro Pi's employee stock option vesting schedule?",
]


def main() -> int:
    """Execute all example questions and write the markdown report."""
    parser = argparse.ArgumentParser(description="Run RAG example questions.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Force re-ingestion of the corpus."
    )
    args = parser.parse_args()

    docs = load_documents(settings.data_dir)
    chunks = chunk_documents(docs)

    if args.rebuild or not settings.persist_dir.exists():
        logger.info("Building vector store...")
        store = build_vectorstore(chunks)
    else:
        logger.info("Loading existing vector store from %s", settings.persist_dir)
        store = load_vectorstore()

    responses = []
    for question in QUESTIONS:
        logger.info("Q: %s", question)
        try:
            response = answer_question(store, question, all_chunks=chunks)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("A: %s", response.answer)
        responses.append(response)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.output_dir / "example_runs.md"

    header = [
        "# Section 2 — Example RAG Runs",
        "",
        f"_Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `run_examples.py`._",
        "",
        f"- Embedding backend: `{settings.embedding_backend}`",
        f"- Generation model: `{settings.ollama_model}` via Ollama",
        f"- Chunking: size {settings.chunk_size}, overlap {settings.chunk_overlap}",
        f"- Retrieval: top-{settings.top_k}, "
        f"hybrid={'on' if settings.use_hybrid_search else 'off'}, "
        f"relevance threshold {settings.relevance_threshold}",
        "",
        "---",
        "",
    ]
    body = "\n\n---\n\n".join(r.to_markdown() for r in responses)
    out_path.write_text("\n".join(header) + body + "\n", encoding="utf-8")

    logger.info("Wrote %s", out_path)
    refused = sum(1 for r in responses if r.refused)
    logger.info("%d answered, %d refused", len(responses) - refused, refused)
    return 0


if __name__ == "__main__":
    sys.exit(main())
