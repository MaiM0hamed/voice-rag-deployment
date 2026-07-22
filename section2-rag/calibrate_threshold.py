"""Calibrate ``RELEVANCE_THRESHOLD`` for the configured embedding backend.

Cosine scores are not comparable across embedding models, so the refusal
threshold must be tuned per backend. This script scores a labelled set of
in-scope and out-of-scope questions and reports the separating gap, then
suggests a threshold at its midpoint.

Run:
    python calibrate_threshold.py
"""

from __future__ import annotations

import logging
import sys

from src.config import settings
from src.ingest import chunk_documents, load_documents
from src.retrieval import retrieve_with_scores

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("calibrate")

IN_SCOPE = [
    "How long do I have to return a development board?",
    "What are the shipping fees for an order under 1,500 EGP in Cairo?",
    "What happens if I exceed the API rate limit on the free tier?",
    "How do I upgrade to Tier 2 verification?",
    "How long do card refunds take to settle?",
]

OUT_OF_SCOPE = [
    "What is Electro Pi's employee stock option vesting schedule?",
    "Who won the football match last night?",
    "What is the capital of Norway?",
    "How do I bake sourdough bread?",
]


def main() -> int:
    """Score labelled questions and suggest a threshold."""
    from langchain_chroma import Chroma

    from src.embeddings import build_embeddings

    chunks = chunk_documents(load_documents(settings.data_dir))
    store = Chroma.from_documents(
        documents=chunks,
        embedding=build_embeddings(settings),
        collection_name="calibration_run",
        collection_configuration={"hnsw": {"space": "cosine"}},
    )

    def best(question: str) -> float:
        scored = retrieve_with_scores(store, question, settings)
        return max((s for _, s in scored), default=0.0)

    print(f"\nBackend: {settings.embedding_backend}")
    print(f"Current threshold: {settings.relevance_threshold}\n")

    print("IN-SCOPE (should score high):")
    in_scores = []
    for q in IN_SCOPE:
        s = best(q)
        in_scores.append(s)
        print(f"  {s:6.3f}  {q}")

    print("\nOUT-OF-SCOPE (should score low):")
    out_scores = []
    for q in OUT_OF_SCOPE:
        s = best(q)
        out_scores.append(s)
        print(f"  {s:6.3f}  {q}")

    lowest_in = min(in_scores)
    highest_out = max(out_scores)
    print(f"\nLowest in-scope : {lowest_in:.3f}")
    print(f"Highest out-scope: {highest_out:.3f}")

    if lowest_in > highest_out:
        suggested = (lowest_in + highest_out) / 2
        print(f"Clean separation. Suggested RELEVANCE_THRESHOLD={suggested:.2f}")
        return 0

    print(
        "WARNING: overlap between in-scope and out-of-scope scores. "
        "No threshold separates them cleanly -- consider a better embedding "
        "model, a reranker, or richer source documents."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
