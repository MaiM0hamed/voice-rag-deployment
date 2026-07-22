"""Environment-driven configuration for the RAG pipeline.

Two embedding backends are supported and selected with `EMBEDDING_BACKEND`:

* ``huggingface`` (default) -- `sentence-transformers/all-MiniLM-L6-v2` run
  locally via `langchain-huggingface`. Best quality/robustness, needs ~90 MB of
  model weights downloaded once from HuggingFace.
* ``ollama`` -- `nomic-embed-text` served by the same local Ollama instance used
  for generation. Useful when HuggingFace is unreachable (corporate proxy,
  air-gapped CI) or when you want a single model server for the whole stack.

Both are free and run entirely locally; neither requires an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SECTION_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for ingestion and querying."""

    # --- paths ---
    data_dir: Path = field(default_factory=lambda: SECTION_DIR / "data")
    persist_dir: Path = field(default_factory=lambda: SECTION_DIR / "chroma_db")
    output_dir: Path = field(default_factory=lambda: SECTION_DIR / "outputs")

    # --- embeddings ---
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "huggingface")
    # all-MiniLM-L6-v2 is the default. bge-small-en-v1.5 was benchmarked as a
    # drop-in upgrade but rejected: it collapsed the refusal gate's in/out-of-
    # scope separation (see NOTES.md). Override with HF_EMBEDDING_MODEL.
    hf_embedding_model: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # --- generation ---
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # --- chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # --- retrieval ---
    top_k: int = int(os.getenv("TOP_K", "4"))
    collection_name: str = os.getenv("COLLECTION_NAME", "electro_pi_docs")
    use_hybrid_search: bool = _env_bool("USE_HYBRID_SEARCH", True)

    @property
    def relevance_threshold(self) -> float:
        """Minimum best-match cosine similarity required to answer.

        Below this, the question is treated as out-of-scope and refused without
        calling the LLM. The right value is embedding-model dependent -- cosine
        scores are not comparable across models -- so the default is chosen per
        backend and can always be overridden with ``RELEVANCE_THRESHOLD``.

        Calibrate on your own corpus: run a handful of in-scope and out-of-scope
        questions, log the best score for each (``retrieval`` logs it at INFO),
        and pick a value in the gap between the two clusters.
        """
        override = os.getenv("RELEVANCE_THRESHOLD")
        if override is not None:
            return float(override)
        # Calibrated on this corpus (calibrate_threshold.py): all-MiniLM-L6-v2
        # separates in-scope (>=0.61) from out-of-scope (<=0.45) with a clean gap;
        # 0.50 sits in it so the gate refuses out-of-scope deterministically while
        # keeping headroom for in-scope. nomic-embed-text baselines run higher.
        return 0.50 if self.embedding_backend == "huggingface" else 0.55


settings = Settings()
