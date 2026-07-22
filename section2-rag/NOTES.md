# Section 2 — Write-up

## What this builds

A RAG pipeline over four Electro Pi domain documents (`data/*.md` — customer
onboarding, returns policy, shipping, developer API). Corpus choice is stated
per the brief: **own domain docs**, written to contain precise, checkable facts
(thresholds, day counts, rate limits) so citation accuracy is easy to audit.

```
docs → header-aware + recursive chunking → embeddings → Chroma (cosine)
     → hybrid retrieve (BM25 + dense) → relevance gate → LCEL chain
     → groundedness check → cited answer
```

**Everything is free and local.** Generation is `qwen2.5:1.5b` via Ollama.
Embeddings are switchable (`EMBEDDING_BACKEND`):

| Backend | Model | When to use |
|---|---|---|
| `huggingface` (default) | `all-MiniLM-L6-v2` | Best quality. Needs torch (~2 GB). |
| `ollama` | `nomic-embed-text` | HF unreachable, or to avoid the torch dependency. |

### Embedding choice: bge-small was benchmarked and rejected

`BAAI/bge-small-en-v1.5` is the obvious "upgrade" (same 384-dim/~130 MB class,
higher MTEB retrieval score) and I evaluated it as a drop-in replacement. On this
corpus it **regressed the thing that matters most here — the refusal gate**.
Measured best-match cosine (via `calibrate_threshold.py`):

| | MiniLM | bge-small |
|---|---|---|
| Lowest in-scope | 0.607 | 0.729 |
| Highest out-of-scope | 0.447 ("stock-option vesting") | **0.736** ("stock-option vesting") |
| Clean separating threshold? | **yes (~0.50)** | **no — clusters overlap** |

bge compresses everything into a high, narrow similarity band, so a
company-policy-*adjacent* out-of-scope question ("stock-option vesting schedule")
scored **above** a genuine in-scope question. No single threshold could admit the
in-scope set while refusing that out-of-scope one. Since the deterministic
refusal gate is the headline guarantee of this system, I kept MiniLM, whose wider
score spread separates cleanly. The lesson: pick embeddings for *separation on
your gate*, not for a leaderboard number. bge would be the right call **with** a
cross-encoder reranker driving the gate instead of raw cosine (see below).

## Hallucination guardrail: gate at retrieval, not in the prompt

The prompt does instruct the model to say *"I don't have enough information."*,
but a 1.5B model follows that instruction unreliably. So the real guard is a
**deterministic retrieval gate**: if the best chunk's cosine similarity falls
below `RELEVANCE_THRESHOLD`, `answer_question` returns the refusal string and
**never calls the LLM**. A model that is never invoked cannot hallucinate.

The gate scores the **final (hybrid) retrieval set**, not a separate dense-only
query, so the evidence the gate judges is exactly the evidence the answer sees —
a keyword-only match (e.g. an exact `429` or `sk_test_`) that BM25 surfaces is no
longer wrongly refused because dense similarity was low.

**A second guard runs after generation.** The `[n]` citations in the answer are
parsed and validated against the retrieved chunks; if the model cites a chunk
that does not exist — or cites nothing at all — the answer is discarded and
`answer_question` returns *"I couldn't verify the answer using the retrieved
documents."* Only the chunks the answer actually referenced are returned as
citations. This is covered by `test_hallucinated_citation_is_rejected` and
`test_only_referenced_citations_returned`.

This is verified by `test_refusal_short_circuits_llm`, which patches `build_llm`
to raise if called and asserts an out-of-scope question still returns cleanly.

Thresholds are **backend-aware** (0.50 HF / 0.55 Ollama), because cosine scores
are not comparable across embedding models. `calibrate_threshold.py` scores
labelled in/out-of-scope questions and suggests a threshold at the midpoint of
the separating gap — a hardcoded constant would be a latent bug on any new corpus.

### A real bug this surfaced

Chroma defaults to **L2 distance**, and LangChain's L2→relevance conversion
(`1 - d/√2`) returns **negative** values for unnormalised vectors. A fixed 0–1
threshold silently rejected *everything*. Fixed by pinning
`collection_configuration={"hnsw": {"space": "cosine"}}` at both creation and
load. `test_scores_are_normalised` is the regression test.

## Chunking strategy

Two-stage, in `src/ingest.py`:

1. **`MarkdownHeaderTextSplitter`** splits on `#`/`##` first, so a chunk never
   straddles two policy sections. Header text is retained (`strip_headers=False`)
   because "## Refund timing" is a strong retrieval signal, and the header lands
   in metadata for citations.
2. **`RecursiveCharacterTextSplitter`** (600/100) enforces the size budget,
   preferring paragraph → line → sentence boundaries.

Verified: 4 docs → 15 chunks, every one under budget, each carrying
`source`, `section`, `doc_title`, `chunk_id`.

## Citations

`format_context` numbers each chunk `[n]`; the prompt requires bracketed
citations after each claim; `format_citations` maps `[n]` back to
`source — section (chunk id)`. `test_citations_align_with_context` asserts the
labels stay in sync — the failure mode where the model cites `[3]` but only two
chunks were retrieved.

## If answer quality on longer documents were poor

**Chunking.** Fixed 600-char windows are a compromise. For long documents I'd
move to (a) **semantic chunking** — split where embedding similarity between
adjacent sentences drops, so chunks follow topic boundaries rather than
character counts; and (b) **parent-document retrieval** — embed small precise
chunks but feed the LLM the enclosing parent section, decoupling retrieval
granularity from generation context. Also **contextual retrieval**: prefix each
chunk with a one-line LLM-generated summary of its place in the document, which
substantially helps when chunks contain ambiguous pronouns.

**Hybrid search (implemented).** `EnsembleRetriever` fuses BM25 with dense
retrieval via reciprocal rank fusion (0.4/0.6). Embeddings capture paraphrase;
BM25 catches exact identifiers embeddings wash out — `sk_test_`, `429`,
`ORD-1001`. In testing, BM25 pulled the exactly-correct "Rate limits" and
"Refund timing" chunks that weak dense retrieval alone had missed. Weights
should be tuned per corpus; identifier-heavy corpora want more BM25.

**Reranking.** The highest-leverage next step. Retrieve top-20 cheaply, then
rerank with a cross-encoder (`bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`,
both free) and keep the top 3–4. Bi-encoders embed query and document
independently; a cross-encoder attends over both jointly and is far more
accurate at fine-grained ordering. Costs ~50–100 ms on CPU for 20 candidates —
usually worth it. It also sharpens the refusal gate, since reranker scores
separate relevant from irrelevant better than raw cosine.

**Metadata filtering.** Chunks already carry `source`/`section`, so Chroma's
`where` clause can pre-filter (e.g. `{"source": "04_api_platform.md"}` for API
questions). At scale I'd add `doc_type`, `effective_date` and `version`, then
filter to current documents — critical for policy corpora where a superseded
2019 refund policy is worse than no answer. A small classifier or the LLM itself
can infer the filter from the question.

**Retrieval evaluation.** I'd add RAGAS or a hand-labelled question→chunk set
measuring hit-rate@k, MRR, and faithfulness. Without measurement, "improving"
retrieval is guesswork — and this is the piece I'd build first on a real project.

## Taking this to production

The current design is a single-process library + CLI. The pieces I'd add for a
real deployment, roughly in priority order:

**Hybrid search** — already implemented (`EnsembleRetriever`, BM25 0.4 / dense
0.6). The BM25 index is now built once and cached rather than per query
(`get_hybrid_retriever`). At scale BM25 moves into the store itself (OpenSearch /
Elasticsearch, or a `bm25s` sidecar) so it is persistent and shardable.

**Cross-encoder re-ranking** — retrieve top-20, rerank with `bge-reranker-base`,
keep 3–4. Highest-leverage quality change; also the right way to drive the
refusal gate (reranker scores separate relevant/irrelevant far better than raw
cosine — see the bge embedding note above).

**Parent-document retrieval** — embed small precise chunks for retrieval but feed
the LLM the enclosing parent section, decoupling retrieval granularity from
generation context. Pairs well with semantic chunking for long documents.

**Context compression** — before generation, filter or summarise retrieved
chunks down to the sentences that actually answer the question
(`LLMChainExtractor` / `EmbeddingsFilter`). Cuts prompt tokens and latency and
reduces the chance the model anchors on an irrelevant passage.

**Caching** — three layers: (1) an embedding cache (`CacheBackedEmbeddings`) so
unchanged chunks are never re-embedded on re-ingest; (2) a semantic query cache
keyed by query embedding for repeated/near-duplicate questions; (3) an answer
cache with TTL keyed on `(question, corpus_version)`, invalidated on re-ingest.

**Streaming** — swap the blocking `chain.invoke` for `chain.astream` and surface
tokens over SSE (mirrors Section 4) so time-to-first-token is low. The
groundedness check then runs on the assembled final answer before citations are
committed, or the answer is streamed with citations validated at the end.

**Async retrieval** — make the pipeline `async`: run BM25 and dense retrieval
concurrently (`asyncio.gather`), then the reranker; use async embedding/LLM
clients so a server handles many concurrent requests without thread-per-request.

**Scaling** — move Chroma from embedded mode to a Chroma server (or a managed
vector DB) so multiple stateless API workers share one index; shard by tenant or
`doc_type`; put ingestion behind a queue so re-indexing never blocks serving.
Stateless workers autoscale horizontally behind the Section 4 FastAPI service.

**Monitoring & observability** — structured logs already exist; add request
tracing (LangSmith or OpenTelemetry) capturing retrieval scores, gate/groundedness
decisions, token counts and latency per stage. Emit metrics: refusal rate,
groundedness-rejection rate, hit-rate@k, p50/p95 latency, TTFT. Alert on refusal
or rejection-rate spikes (a corpus or embedding regression) and on retrieval
latency. Log every refusal with its best score to feed threshold re-calibration.

**Security** — treat documents as untrusted once they are user-uploaded: fence
chunk content and add prompt-injection defenses so a chunk cannot override the
system instructions (the groundedness check already blocks answers that cite
nothing real). Bound question length and rate-limit at the API edge; add
authn/authz and per-tenant corpus isolation. Validate `OLLAMA_URL` against an
allowlist (env-controlled URL is a mild SSRF surface). Never log full document
contents or PII; scrub before tracing. Pin model and dependency versions
(already done) so a supply-chain swap cannot change behaviour silently.

## Assumptions, shortcuts, limitations (stated honestly)

- **Corpus is synthetic**, written for this assessment. Facts are internally
  consistent but fictional.
- **Verified offline**: chunking, stable chunk ids, metadata, cosine
  normalisation, gate-on-final-results ordering, hybrid retrieval, citation
  alignment, referenced-only citations, hallucinated-citation rejection, and the
  refusal short-circuit that never calls the LLM (12/12 pytest, no downloads).
- **Verified on a real run**: `outputs/example_runs.md` was generated by
  `run_examples.py` against real `all-MiniLM-L6-v2` embeddings and real
  `qwen2.5:1.5b` generation via Ollama. The offline suite still uses deterministic
  hash embeddings so it needs no downloads.
- **Threshold is calibrated for this corpus**: `calibrate_threshold.py` gives a
  clean in/out-of-scope gap under MiniLM (in-scope >=0.61, out-of-scope <=0.45);
  the 0.50 default sits in it. Re-run it on any new corpus before trusting it.
- **No reranker implemented** — discussed above, deliberately left out to avoid
  a second model download for a small corpus. This is the first thing I'd add.
- **Groundedness validates citation *existence*, not *entailment*.** The check
  rejects hallucinated/missing citations, and a corrective retry recovers answers
  where a small model simply forgot the brackets. It does **not** yet verify the
  cited chunk actually supports the claim — a 1.5B model can pair a valid citation
  with an inaccurate sentence. Closing that gap needs an NLI/faithfulness check
  (or an LLM judge) over each claim→chunk pair; noted under retrieval evaluation.
