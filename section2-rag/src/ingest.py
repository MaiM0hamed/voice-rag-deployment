"""Document loading, chunking and vector-store construction.

Pipeline: load markdown/PDF -> split into chunks -> embed -> persist to Chroma.

Chunk metadata carries `source` (file name) and `chunk_id`, which the answer
chain uses to produce citations back to the originating chunk.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import Settings, settings
from .embeddings import build_embeddings

logger = logging.getLogger(__name__)

# Split on markdown headers first so a chunk never straddles two policy
# sections; the recursive splitter then enforces the size budget.
_HEADERS = [("#", "doc_title"), ("##", "section")]


def _stable_chunk_id(source: str, section: str, page: object, content: str) -> str:
    """Derive a content-addressed chunk id that survives re-ingestion.

    A running index changes whenever chunk boundaries shift, silently
    invalidating any citation stored against it. Hashing the identifying
    fields instead keeps the id stable as long as the underlying text is.
    """
    digest = hashlib.sha1(
        f"{source}|{section}|{page}|{content}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def load_documents(data_dir: Path) -> list[Document]:
    """Load every ``.md`` and ``.pdf`` file in ``data_dir``.

    Args:
        data_dir: Directory containing the corpus.

    Returns:
        One Document per source file, tagged with a ``source`` metadata key.

    Raises:
        FileNotFoundError: If the directory is missing or contains no documents.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: list[Document] = []

    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
        logger.info("Loaded %s (%d chars)", path.name, len(text))

    for path in sorted(data_dir.glob("*.pdf")):
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:  # pragma: no cover
            logger.warning("pypdf not installed; skipping %s", path.name)
            continue
        for page in PyPDFLoader(str(path)).load():
            page.metadata["source"] = path.name
            docs.append(page)
        logger.info("Loaded %s", path.name)

    if not docs:
        raise FileNotFoundError(f"No .md or .pdf documents found in {data_dir}")

    return docs


def chunk_documents(docs: list[Document], cfg: Settings = settings) -> list[Document]:
    """Split documents into retrieval-sized chunks with citation metadata.

    Uses a two-stage strategy: header-aware splitting keeps each policy section
    intact, then a recursive character splitter caps the chunk size while
    preferring paragraph and sentence boundaries.

    Args:
        docs: Loaded source documents.
        cfg: Settings controlling chunk size and overlap.

    Returns:
        Chunks carrying ``source``, ``section`` and ``chunk_id`` metadata.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS, strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    sectioned: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source.endswith(".md"):
            for piece in header_splitter.split_text(doc.page_content):
                piece.metadata["source"] = source
                sectioned.append(piece)
        else:
            sectioned.append(doc)

    chunks = char_splitter.split_documents(sectioned)

    for chunk in chunks:
        chunk.metadata.setdefault("section", "")
        chunk.metadata["chunk_id"] = _stable_chunk_id(
            chunk.metadata.get("source", "unknown"),
            chunk.metadata.get("section", ""),
            chunk.metadata.get("page", ""),
            chunk.page_content,
        )

    logger.info("Split %d documents into %d chunks", len(docs), len(chunks))
    return chunks


def build_vectorstore(
    chunks: list[Document], cfg: Settings = settings
) -> Chroma:
    """Embed chunks and persist them to a Chroma collection.

    Args:
        chunks: Chunked documents to index.
        cfg: Settings controlling persistence path and collection name.

    Returns:
        The populated Chroma vector store.
    """
    cfg.persist_dir.mkdir(parents=True, exist_ok=True)

    # Chroma defaults to L2 distance, whose LangChain relevance conversion
    # (1 - d/sqrt(2)) can go negative and makes a fixed 0..1 threshold
    # meaningless. We pin cosine so `relevance_threshold` is interpretable.
    store = Chroma.from_documents(
        documents=chunks,
        embedding=build_embeddings(cfg),
        collection_name=cfg.collection_name,
        persist_directory=str(cfg.persist_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    logger.info(
        "Indexed %d chunks into Chroma collection %r at %s",
        len(chunks),
        cfg.collection_name,
        cfg.persist_dir,
    )
    return store


def load_vectorstore(cfg: Settings = settings) -> Chroma:
    """Open the previously persisted Chroma collection.

    Args:
        cfg: Settings pointing at the persisted store.

    Returns:
        The Chroma vector store ready for querying.

    Raises:
        FileNotFoundError: If the store has not been built yet.
    """
    if not cfg.persist_dir.exists():
        raise FileNotFoundError(
            f"No vector store at {cfg.persist_dir}. Run `python -m src.ingest` first."
        )
    return Chroma(
        collection_name=cfg.collection_name,
        embedding_function=build_embeddings(cfg),
        persist_directory=str(cfg.persist_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def main() -> None:
    """CLI entrypoint: build the vector store from ``data/``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    docs = load_documents(settings.data_dir)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
