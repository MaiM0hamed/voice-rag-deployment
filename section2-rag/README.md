# Section 2 — LangChain RAG

A retrieval-augmented question-answering pipeline over Electro Pi domain
documents. It chunks and embeds the corpus into a persistent Chroma vector
store, retrieves with a BM25 + dense hybrid, and answers **only** from retrieved
context — refusing before the LLM is ever called when nothing relevant is found,
and rejecting any answer whose citations cannot be verified.

Everything runs locally and free: `all-MiniLM-L6-v2` embeddings and
`qwen2.5:1.5b` generation via [Ollama](https://ollama.com). No paid API.

---

## Overview

- **Loader** — reads every `.md` / `.pdf` in `data/`.
- **Chunking** — header-aware split (keeps policy sections intact) then a
  recursive character splitter enforcing a size budget. Each chunk carries
  `source`, `section`, `doc_title` and a **stable content-hash `chunk_id`**.
- **Embeddings** — `all-MiniLM-L6-v2`, normalised for cosine. (`bge-small-en-v1.5`
  was benchmarked but rejected — it collapsed the refusal gate's separation; see
  [`NOTES.md`](NOTES.md).)
- **Vector store** — Chroma, persisted to `chroma_db/`, cosine space pinned.
- **Retrieval** — hybrid `EnsembleRetriever` (BM25 0.4 / dense 0.6); the BM25
  index is built once per session and reused.
- **Hallucination guard** — a deterministic relevance gate on the *final*
  retrieval set; below threshold the pipeline refuses and **never calls the LLM**.
- **Prompt** — instructs the model to use only the context, cite every claim,
  never fabricate citations, and refuse when uncertain.
- **Groundedness check** — after generation, citations are validated against the
  retrieved chunks; unverifiable answers are rejected.
- **Citations** — only the chunks the answer actually referenced are returned.

---

## Architecture Diagram

```
        Documents (data/*.md, *.pdf)
                 │
              Loader                     load_documents()
                 │
             Chunking                    header-aware + recursive splitter
                 │                        → stable content-hash chunk_id
             Embeddings                  all-MiniLM-L6-v2 (normalised)
                 │
              ChromaDB                   persistent, cosine space
                 │
          Hybrid Retrieval               BM25 (0.4) + dense (0.6), RRF fusion
                 │                        BM25 index built once, cached
        Hallucination Guard  ── refuse ▶ "I don't have enough information."
                 │  (score final set)     (LLM never called)
              Prompt                     use-only-context, cite, never fabricate
                 │
               LLM                       qwen2.5:1.5b via Ollama
                 │
        Groundedness Check   ── fail ──▶ "I couldn't verify the answer using
                 │                         the retrieved documents."
        Answer + Citations              only the chunks actually referenced
```

---

## Installation

```bash
cd section2-rag
pip install -r requirements.txt

# Generation backend (shared with sections 1 and 4):
ollama serve
ollama pull qwen2.5:1.5b
```

First run downloads ~90 MB of embedding weights from huggingface.co (cached
under `~/.cache/huggingface`); later runs are offline. To avoid the torch
dependency entirely, set `EMBEDDING_BACKEND=ollama` and
`ollama pull nomic-embed-text`.

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed. All have sensible defaults.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_BACKEND` | `huggingface` | `huggingface` (all-MiniLM-L6-v2) or `ollama` (nomic-embed-text) |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model when backend is `ollama` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Generation model |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `CHUNK_SIZE` | `600` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Chunk overlap |
| `TOP_K` | `4` | Chunks retrieved per query |
| `COLLECTION_NAME` | `electro_pi_docs` | Chroma collection name |
| `USE_HYBRID_SEARCH` | `true` | Toggle BM25 + dense hybrid |
| `RELEVANCE_THRESHOLD` | backend-aware (0.50 HF / 0.55 Ollama) | Refusal gate; leave unset to use the default, or tune with `calibrate_threshold.py` |
| `LOG_LEVEL` | `WARNING` | Log verbosity for the interactive CLI (`app.py`); set `INFO` to trace retrieval/gate decisions |

---

## Ingest Documents

Build the persistent vector store from `data/`:

```bash
python -m src.ingest
```

This loads the corpus, chunks it, embeds it and writes `chroma_db/`. Re-running
is safe; content-hash chunk ids stay stable across re-ingestion.

---

## Run Commands

```bash
python -m pytest tests/ -q          # offline test suite (no model downloads)
python calibrate_threshold.py       # tune the refusal gate for your backend
python run_examples.py --rebuild    # runs the example questions → outputs/example_runs.md
python app.py                       # interactive CLI (see below)
```

---

## Interactive CLI

```bash
python app.py
```

```
--------------------------------
Electro Pi RAG — interactive CLI
Type a question, or 'exit' to quit.
--------------------------------

Ask a question:
> How long do I have to return a development board?

Answer: Development boards and kits have a 30 calendar day return window [1].

Citations:
  [1] 02_returns_policy.md — Return window (chunk a1b2c3d4e5f6)
--------------------------------
```

The loop accepts any question and continues until you type `exit` (or `quit`).
Out-of-scope questions are refused without calling the LLM; unverifiable answers
are rejected by the groundedness check.

---

## Example Output

Three complete, pipeline-generated executions (question → retrieved chunks →
answer → citations), plus the out-of-scope refusal path, are recorded in
[`outputs/example_runs.md`](outputs/example_runs.md), produced by
`python run_examples.py --rebuild`. See [`outputs/README.md`](outputs/README.md).

---

## Testing

```bash
python -m pytest tests/ -q
```

The suite runs fully offline — embeddings are a deterministic hash stub and
generation is a fake chat model — so it needs no downloads. It covers document
loading, chunk metadata, stable chunk ids, cosine score normalisation,
in/out-of-scope ordering, the refusal short-circuit (LLM never called), citation
alignment, referenced-only citations, and rejection of hallucinated citations.

---

## Limitations

- The corpus is **synthetic**, written for this assessment; facts are internally
  consistent but fictional.
- Generation quality is bounded by a 1.5B local model. The relevance gate and
  groundedness check contain its failure modes, but nuance can suffer.
- Relevance thresholds are reasoned defaults, not tuned on your embeddings — run
  `calibrate_threshold.py` before trusting them.
- No cross-encoder reranker yet (see below).
- Single-node embedded Chroma; the BM25 cache is per-process.

---

## Future Improvements

See [`NOTES.md`](NOTES.md) for the full write-up. Highest-value next steps:

1. **Cross-encoder reranking** — retrieve top-20, rerank, keep top 3–4.
2. **Retrieval evaluation** (RAGAS / labelled hit-rate@k, MRR, faithfulness).
3. **Parent-document / semantic chunking** for longer documents.
4. **Context compression** and answer/embedding **caching**.
5. **Streaming + async retrieval**, and a client/server Chroma deployment for
   horizontal scale.
