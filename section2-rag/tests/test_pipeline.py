"""Offline tests for the RAG pipeline.

These run without downloading any model: embeddings are replaced with a
deterministic hashing stub and generation with a canned response. They cover
the parts that most often break silently -- chunk metadata, the relevance gate,
citation formatting, and the refusal path.

Run:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import math

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src import rag_chain
from src.config import Settings
from src.ingest import chunk_documents, load_documents
from src.retrieval import format_citations, format_context, is_relevant, retrieve_with_scores


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings; no network required."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(scope="module")
def cfg() -> Settings:
    # Threshold lowered to suit the weak stub embeddings.
    return Settings(embedding_backend="huggingface")


@pytest.fixture(scope="module")
def chunks(cfg: Settings):
    return chunk_documents(load_documents(cfg.data_dir), cfg)


@pytest.fixture(scope="module")
def store(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=HashEmbeddings(),
        collection_name="pytest_collection",
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def test_documents_load(cfg: Settings) -> None:
    docs = load_documents(cfg.data_dir)
    assert len(docs) == 4
    assert all(d.metadata.get("source", "").endswith(".md") for d in docs)


def test_chunks_have_citation_metadata(chunks) -> None:
    assert len(chunks) > len(load_documents(Settings().data_dir))
    for chunk in chunks:
        assert chunk.metadata["source"]
        assert "chunk_id" in chunk.metadata
        assert "section" in chunk.metadata


def test_chunks_respect_size_budget(chunks, cfg: Settings) -> None:
    assert all(len(c.page_content) <= cfg.chunk_size * 1.2 for c in chunks)


def test_scores_are_normalised(store, cfg: Settings) -> None:
    """Cosine config must keep relevance in [0, 1] (regression test).

    With Chroma's default L2 space these scores go negative, which silently
    breaks any fixed threshold.
    """
    scored = retrieve_with_scores(store, "refund settle card", cfg)
    assert scored
    assert all(0.0 <= s <= 1.0 for _, s in scored)


def test_in_scope_outranks_out_of_scope(store, cfg: Settings) -> None:
    in_scope = max(s for _, s in retrieve_with_scores(store, "return window refund", cfg))
    out_scope = max(
        s for _, s in retrieve_with_scores(store, "sourdough bread recipe", cfg)
    )
    assert in_scope > out_scope


def test_relevance_gate_refuses_out_of_scope(store) -> None:
    tight = Settings(embedding_backend="huggingface")
    scored = retrieve_with_scores(store, "capital city of Norway", tight)
    assert is_relevant(scored, tight) is False


def test_citations_align_with_context(store, cfg: Settings) -> None:
    docs = [d for d, _ in retrieve_with_scores(store, "shipping fees Cairo", cfg)]
    context = format_context(docs)
    citations = format_citations(docs)
    assert len(citations) == len(docs)
    for index in range(1, len(docs) + 1):
        assert f"[{index}]" in context
        assert citations[index - 1].startswith(f"[{index}]")


def test_refusal_short_circuits_llm(store, chunks, monkeypatch) -> None:
    """An out-of-scope question must not reach the LLM at all."""
    called = {"n": 0}

    def _boom(cfg=None):
        called["n"] += 1
        raise AssertionError("LLM must not be called on refusal")

    monkeypatch.setattr(rag_chain, "build_llm", _boom)
    response = rag_chain.answer_question(
        store, "What is the capital of Norway?", all_chunks=chunks
    )
    assert response.refused is True
    assert response.answer == rag_chain.NO_CONTEXT_MESSAGE
    assert response.citations == []
    assert called["n"] == 0


def test_answer_includes_citations(store, chunks, monkeypatch) -> None:
    monkeypatch.setattr(
        rag_chain,
        "build_llm",
        lambda cfg=None: FakeListChatModel(
            responses=["Development boards have a 30 day window [1]."]
        ),
    )
    loose = Settings(embedding_backend="huggingface")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.20")
    response = rag_chain.answer_question(
        store, "return window development board refund", all_chunks=chunks, cfg=loose
    )
    assert response.refused is False
    assert response.grounded is True
    assert "[1]" in response.answer
    assert response.citations


def test_stable_chunk_ids_survive_reingestion(cfg: Settings) -> None:
    """Chunk ids must be content-addressed, not a running index."""
    first = chunk_documents(load_documents(cfg.data_dir), cfg)
    second = chunk_documents(load_documents(cfg.data_dir), cfg)
    ids_a = [c.metadata["chunk_id"] for c in first]
    ids_b = [c.metadata["chunk_id"] for c in second]
    assert ids_a == ids_b
    # Not a plain 0..n index, and unique per distinct chunk.
    assert all(isinstance(i, str) for i in ids_a)
    assert len(set(ids_a)) == len(ids_a)


def test_only_referenced_citations_returned(store, chunks, monkeypatch) -> None:
    """Only the chunks the answer actually cites are returned."""
    monkeypatch.setattr(
        rag_chain,
        "build_llm",
        lambda cfg=None: FakeListChatModel(
            responses=["Development boards have a 30 day window [2]."]
        ),
    )
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.20")
    response = rag_chain.answer_question(
        store, "return window development board refund", all_chunks=chunks
    )
    assert response.grounded is True
    assert len(response.citations) == 1
    assert response.citations[0].startswith("[2]")


def test_hallucinated_citation_is_rejected(store, chunks, monkeypatch) -> None:
    """An answer citing a non-existent chunk fails groundedness."""
    monkeypatch.setattr(
        rag_chain,
        "build_llm",
        lambda cfg=None: FakeListChatModel(
            responses=["Boards can be returned in 30 days [99]."]
        ),
    )
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.20")
    response = rag_chain.answer_question(
        store, "return window development board refund", all_chunks=chunks
    )
    assert response.grounded is False
    assert response.answer == rag_chain.UNVERIFIED_MESSAGE
    assert response.citations == []
