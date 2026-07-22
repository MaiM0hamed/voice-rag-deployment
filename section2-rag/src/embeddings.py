"""Embedding backend factory.

**Primary backend: HuggingFace** — `sentence-transformers/all-MiniLM-L6-v2`,
run locally via `langchain-huggingface`. This is the default and the
recommended configuration for normal development machines. (`bge-small-en-v1.5`
was benchmarked as an upgrade but rejected — it collapsed the refusal gate's
in/out-of-scope separation on this corpus; see NOTES.md.)

**Optional fallback: Ollama** — `nomic-embed-text` served by the same local
Ollama instance used for generation. Useful when HuggingFace is unreachable
(corporate proxy, air-gapped CI) or when you want to avoid the torch dependency.

Select with the ``EMBEDDING_BACKEND`` environment variable. Both run entirely
locally and neither requires an API key.

Imports are deliberately lazy: `langchain_huggingface` pulls in torch, which we
don't want to require when the user selected the Ollama fallback (and vice
versa). Both backends return a LangChain `Embeddings` object, so the rest of
the pipeline is backend-agnostic.

Note on first run: the HuggingFace backend downloads ~90 MB of model weights
from huggingface.co and caches them under ``~/.cache/huggingface``. This
requires network access the first time only; subsequent runs are offline.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from .config import Settings, settings

logger = logging.getLogger(__name__)


def build_embeddings(cfg: Settings = settings) -> Embeddings:
    """Construct the embedding model selected by ``EMBEDDING_BACKEND``.

    Args:
        cfg: Settings instance (defaults to the module-level singleton).

    Returns:
        A LangChain ``Embeddings`` implementation.

    Raises:
        ValueError: If the configured backend name is not recognised.
        RuntimeError: If the backend's optional dependency is missing.
    """
    backend = cfg.embedding_backend.strip().lower()

    if backend == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "EMBEDDING_BACKEND=huggingface requires `langchain-huggingface` "
                "and `sentence-transformers` (which pulls in torch). Install them, "
                "or set EMBEDDING_BACKEND=ollama to use nomic-embed-text instead."
            ) from exc

        logger.info("Embeddings: HuggingFace %s (local)", cfg.hf_embedding_model)
        # normalize_embeddings=True -> cosine similarity == dot product, which
        # makes the relevance threshold in retrieval.py interpretable.
        return HuggingFaceEmbeddings(
            model_name=cfg.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if backend == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "EMBEDDING_BACKEND=ollama requires `langchain-ollama`. "
                "Install it and run: ollama pull nomic-embed-text"
            ) from exc

        logger.info(
            "Embeddings: Ollama %s at %s",
            cfg.ollama_embedding_model,
            cfg.ollama_url,
        )
        return OllamaEmbeddings(
            model=cfg.ollama_embedding_model, base_url=cfg.ollama_url
        )

    raise ValueError(
        f"Unknown EMBEDDING_BACKEND {backend!r}; expected 'huggingface' or 'ollama'."
    )
