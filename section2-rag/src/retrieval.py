"""Retrieval layer: dense search, optional hybrid (BM25 + dense), and gating.

The "no relevant context" guard lives here rather than in the prompt. Relying on
the LLM alone to say "I don't know" is unreliable for a 1.5B model, so we make
the refusal a *retrieval* decision: if the best chunk's similarity is below
`RELEVANCE_THRESHOLD`, the chain short-circuits and never calls the LLM. That
makes the guardrail deterministic and testable.
"""

from __future__ import annotations

import logging
import math

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Settings, settings

logger = logging.getLogger(__name__)

# Cache built hybrid retrievers so the BM25 index is constructed once per chunk
# set rather than rebuilt on every query. Keyed by the chunk list's identity,
# which is stable for the lifetime of a loaded corpus (CLI session, script run).
_HYBRID_RETRIEVERS: dict[int, object] = {}


def retrieve_with_scores(
    store: Chroma, question: str, cfg: Settings = settings
) -> list[tuple[Document, float]]:
    """Dense-retrieve chunks with normalised relevance scores.

    Args:
        store: The Chroma vector store.
        question: The user's question.
        cfg: Settings controlling ``top_k``.

    Returns:
        ``(document, relevance)`` pairs, where relevance is in [0, 1] and
        higher is more similar. Sorted most-relevant first.
    """
    # LangChain normalises distance -> relevance for us.
    results = store.similarity_search_with_relevance_scores(question, k=cfg.top_k)
    for doc, score in results:
        logger.debug(
            "candidate score=%.3f source=%s section=%s",
            score,
            doc.metadata.get("source"),
            doc.metadata.get("section"),
        )
    return results


def get_hybrid_retriever(
    store: Chroma, all_chunks: list[Document], cfg: Settings = settings
):
    """Build (once) and cache the BM25 + dense ensemble retriever.

    Building the BM25 index tokenises the whole corpus, so doing it per query is
    wasteful. We construct the ensemble once per chunk set and reuse it.

    Args:
        store: The Chroma vector store.
        all_chunks: The full chunk list, needed to build the BM25 index.
        cfg: Settings controlling ``top_k``.

    Returns:
        A cached `EnsembleRetriever` fusing BM25 and dense retrieval.
    """
    cached = _HYBRID_RETRIEVERS.get(id(all_chunks))
    if cached is not None:
        return cached

    # EnsembleRetriever moved to `langchain_classic` in LangChain 1.x; support both.
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:  # LangChain 0.x
        from langchain.retrievers import EnsembleRetriever

    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = cfg.top_k

    dense = store.as_retriever(search_kwargs={"k": cfg.top_k})

    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
    _HYBRID_RETRIEVERS[id(all_chunks)] = ensemble
    logger.info("Built hybrid retriever; BM25 index cached for reuse")
    return ensemble


def hybrid_retrieve(
    store: Chroma, all_chunks: list[Document], question: str, cfg: Settings = settings
) -> list[Document]:
    """Combine BM25 keyword search with dense retrieval.

    Dense embeddings capture paraphrase; BM25 captures exact identifiers and
    rare terms (order codes, ``sk_test_`` prefixes, "429") that embeddings often
    wash out. `EnsembleRetriever` fuses both with reciprocal rank fusion. The
    retriever (and its BM25 index) is built once and cached.

    Args:
        store: The Chroma vector store.
        all_chunks: The full chunk list, needed to build the BM25 index.
        question: The user's question.
        cfg: Settings controlling ``top_k``.

    Returns:
        Fused, de-duplicated documents ordered by combined rank.
    """
    ensemble = get_hybrid_retriever(store, all_chunks, cfg)
    docs = ensemble.invoke(question)
    logger.info("Hybrid retrieval returned %d chunks", len(docs))
    return docs[: cfg.top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, robust to unnormalised vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def score_documents(
    store: Chroma, question: str, docs: list[Document], cfg: Settings = settings
) -> list[tuple[Document, float]]:
    """Score an arbitrary set of retrieved documents against the question.

    Used to run the relevance gate on the *final* (hybrid) retrieval set rather
    than on a separate dense-only query, so the gate and the answer see the same
    evidence. Scores are cosine similarities on the same scale as
    :func:`retrieve_with_scores`, so ``relevance_threshold`` stays comparable.

    Args:
        store: The Chroma vector store (source of the embedding function).
        question: The user's question.
        docs: The documents to score (typically the hybrid results).
        cfg: Settings.

    Returns:
        ``(document, relevance)`` pairs in the input order.
    """
    if not docs:
        return []

    embeddings = getattr(store, "embeddings", None)
    if embeddings is None:  # pragma: no cover - defensive fallback
        dense = {
            doc.page_content: score
            for doc, score in retrieve_with_scores(store, question, cfg)
        }
        return [(doc, dense.get(doc.page_content, 0.0)) for doc in docs]

    query_vec = embeddings.embed_query(question)
    doc_vecs = embeddings.embed_documents([doc.page_content for doc in docs])
    return [(doc, _cosine(query_vec, vec)) for doc, vec in zip(docs, doc_vecs)]


def is_relevant(
    scored: list[tuple[Document, float]], cfg: Settings = settings
) -> bool:
    """Decide whether retrieval found anything worth answering from.

    Args:
        scored: Output of :func:`retrieve_with_scores`.
        cfg: Settings carrying ``relevance_threshold``.

    Returns:
        True if the best chunk clears the threshold.
    """
    if not scored:
        logger.info("Relevance gate: no chunks retrieved -> refuse")
        return False

    best = max(score for _, score in scored)
    passed = best >= cfg.relevance_threshold
    logger.info(
        "Relevance gate: best=%.3f threshold=%.3f -> %s",
        best,
        cfg.relevance_threshold,
        "answer" if passed else "refuse",
    )
    return passed


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a numbered, citable context block.

    Each chunk is labelled ``[n]`` so the LLM can cite it, and the label maps
    back to ``source``/``section`` metadata for the caller's citation list.

    Args:
        docs: Retrieved chunks.

    Returns:
        A formatted context string for the prompt.
    """
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        header = f"[{index}] {source}" + (f" — {section}" if section else "")
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def format_citations(
    docs: list[Document], referenced: set[int] | None = None
) -> list[str]:
    """Build human-readable citation strings matching the ``[n]`` labels.

    Args:
        docs: The same retrieved chunks passed to :func:`format_context`.
        referenced: If given, only chunks whose 1-based label is in this set are
            returned -- i.e. the sources the answer actually cited. When
            ``None`` every chunk is returned (full label mapping).

    Returns:
        Citation strings, in label order.
    """
    citations: list[str] = []
    for index, doc in enumerate(docs, start=1):
        if referenced is not None and index not in referenced:
            continue
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        chunk_id = doc.metadata.get("chunk_id", "?")
        label = f"[{index}] {source}"
        if section:
            label += f" — {section}"
        label += f" (chunk {chunk_id})"
        citations.append(label)
    return citations
