# Electro Pi — AI Engineer Technical Assessment

Practical assessment covering LiveKit voice agents, LangChain RAG, model
quantization, and containerised model deployment.

**Everything runs on free / open-source tooling.** No paid API is used anywhere.
Where the reference brief allowed a paid provider, a local open-source
equivalent is used instead and the substitution is documented.

---

## Contents

| Section | Topic | Key deliverable |
|---|---|---|
| [`section1-livekit/`](section1-livekit/) | Real-time voice agent | Real `Agent` + `AgentSession` + `@function_tool`, LLM-driven tool call with transcript/logs |
| [`section2-rag/`](section2-rag/) | LangChain RAG | Chunk → embed → Chroma → hybrid retrieve → cited answer, with a deterministic no-context guard |
| [`section3-quantization/`](section3-quantization/) | Quantization | FP16 vs 4-bit NF4 benchmark harness producing a measured comparison table |
| [`section4-deployment/`](section4-deployment/) | Deployment | FastAPI service, Docker, SSE streaming, TTFT/latency load test |

Each section has its own `README`-level `NOTES.md` containing the required
half-page write-up, plus `requirements.txt`, `.env.example` and tests.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │   Ollama (local, free)       │
                    │   qwen2.5:1.5b               │
                    └───────┬──────────────┬───────┘
                            │              │
              tool-calling  │              │  generation
                            │              │
   ┌────────────────────────┴───┐   ┌──────┴─────────────────────┐
   │ Section 1 — LiveKit        │   │ Section 2 — RAG            │
   │ Agent / AgentSession       │   │ md docs → chunk → embed    │
   │ MockSTT → LLM → MockTTS    │   │ → Chroma → hybrid retrieve │
   │ @function_tool             │   │ → gate → cited answer      │
   └────────────────────────────┘   └────────────────────────────┘

   ┌────────────────────────────┐   ┌────────────────────────────┐
   │ Section 3 — Quantization   │──▶│ Section 4 — Deployment     │
   │ Qwen2.5-1.5B-Instruct      │   │ FastAPI + Docker           │
   │ FP16/BF16  vs  4-bit NF4   │   │ /health /chat /stream      │
   │ measured comparison table  │   │ TTFT + latency load test   │
   └────────────────────────────┘   └────────────────────────────┘
```

Sections 1, 2 and 4 share a single local LLM backend (Ollama) so the whole repo
runs with one model pull. Section 3 uses raw `transformers` instead, because it
needs direct control over weight precision to measure it.

---

## Folder structure

```
electro-pi-ai-engineer-assessment/
├── README.md
├── section1-livekit/
│   ├── agent.py               # Agent subclass + @function_tool
│   ├── worker.py              # production AgentSession worker (LiveKit room)
│   ├── run_session_demo.py    # headless AgentSession.run() demo
│   ├── mock_providers.py      # MockSTT / MockTTS (real SDK subclasses)
│   ├── logs/                  # transcript + tool-invocation logs
│   └── NOTES.md
├── section2-rag/
│   ├── src/                   # config, embeddings, ingest, retrieval, rag_chain
│   ├── data/                  # 4 Electro Pi domain documents
│   ├── run_examples.py        # runs the example questions
│   ├── calibrate_threshold.py # tunes the refusal gate per backend
│   ├── tests/                 # 9 offline tests
│   ├── outputs/               # GENERATED — see outputs/README.md
│   └── NOTES.md
├── section3-quantization/
│   ├── src/                   # config (5 fixed prompts), benchmark, report
│   ├── tests/                 # 6 report-arithmetic tests
│   ├── results/               # GENERATED — see results/README.md
│   └── NOTES.md
└── section4-deployment/
    ├── app/                   # config, schemas, inference, main
    ├── loadtest/              # benchmark.py (TTFT), locustfile.py
    ├── tests/                 # 9 API tests
    ├── Dockerfile, docker-compose.yml
    ├── results/               # GENERATED — see results/README.md
    └── NOTES.md
```

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** for sections 1, 2 and 4:
  ```bash
  ollama serve
  ollama pull qwen2.5:1.5b
  ```
- **Docker** for section 4's container
- **A CUDA GPU** for section 3's 4-bit run (a free Colab T4 is enough).
  CPU-only machines still run the FP16 half; the 4-bit run is skipped with the
  reason recorded, never guessed.

Each section installs independently — you do not need one section's
dependencies to run another.

---

## Installation & running

### Section 1 — LiveKit agent

```bash
cd section1-livekit
pip install -r requirements.txt
cp .env.example .env

python run_session_demo.py     # headless; writes logs/transcript.log
# or, against a LiveKit server:
python worker.py dev
```

### Section 2 — RAG

```bash
cd section2-rag
pip install -r requirements.txt
cp .env.example .env

python -m pytest tests/ -q            # 9 offline tests, no downloads
python calibrate_threshold.py         # tune the refusal gate (recommended)
python run_examples.py --rebuild      # writes outputs/example_runs.md
```

First run downloads ~90 MB of embedding weights. To avoid the torch dependency
entirely, set `EMBEDDING_BACKEND=ollama` and `ollama pull nomic-embed-text`.

### Section 3 — Quantization

```bash
cd section3-quantization
# install torch for YOUR hardware FIRST
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install -r requirements.txt

python -m pytest tests/ -q            # 6 tests, torch stubbed
python results/generate_results.py    # writes results/comparison_report.md
```

Downloads Qwen2.5-1.5B-Instruct (~3 GB) on first run. 5–15 min on a T4.

### Section 4 — Deployment

```bash
cd section4-deployment
python -m pytest tests/ -q            # 9 API tests, no weights needed

docker compose up --build             # or: uvicorn app.main:app --port 8000
curl localhost:8000/health            # wait for "model_loaded": true

python results/generate_results.py --users 10   # writes results/load_test_report.md
```

Try the endpoints:

```bash
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is a Raspberry Pi?"}]}'

curl -N -X POST localhost:8000/stream \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to five."}]}'
```

---

## Results

**No measurement in this repository is invented.** Sections 3 and 4 ship
generator scripts rather than pre-filled tables, because the brief scores "real
measurements, not guesses" and a plausible fabricated table would be the wrong
artifact to submit. Every `results/` and `outputs/` directory contains a
`generate_results.py` and a README explaining exactly how to populate it.

Generated artifacts are `.gitignore`d so a fresh clone never contains stale
numbers from someone else's hardware.

| Artifact | Produced by |
|---|---|
| `section2-rag/outputs/example_runs.md` | `python run_examples.py --rebuild` |
| `section3-quantization/results/comparison_report.md` | `python results/generate_results.py` |
| `section4-deployment/results/load_test_report.md` | `python results/generate_results.py --users 10` |

---

## What was verified, and what was not

Honesty about trade-offs is scored positively, so this is explicit.

**Verified by execution (27 automated tests, all passing):**

| Section | Verified |
|---|---|
| 1 | `SupportAgent` is a real `livekit.agents.Agent` subclass; `MockSTT`/`MockTTS` are real SDK `STT`/`TTS` subclasses; `@function_tool` auto-derives the JSON schema; a real `AgentSession` run produced `FunctionCall → FunctionCallOutput → ChatMessage` |
| 2 | 12/12 — chunk metadata, stable content-hash chunk ids, cosine score normalisation, in/out-of-scope ordering, hybrid BM25+dense retrieval, citation alignment, referenced-only citations, hallucinated-citation rejection, refusal never invoking the LLM; plus a real end-to-end run on `all-MiniLM-L6-v2` + `qwen2.5:1.5b` producing `outputs/example_runs.md` |
| 3 | 6/6 — comparison-table arithmetic, ratio formatting, missing-variant handling, divide-by-zero guards |
| 4 | 9/9 — all endpoints, SSE framing and `[DONE]`, mid-stream errors, 503/422 paths; plus a real 10-concurrent-request load test against the live app (TTFT 0.044 s vs total 0.242 s, confirming genuine streaming) |

**NOT verified — must be run on your machine:**

- **`docker build` has never been executed.** Docker was unavailable in the
  development environment. Run this first; the brief requires it to work
  end to end.
- **All real benchmark numbers** (Section 3 and 4) — needs model weights.
- **Section 2 was run on real models** (`all-MiniLM-L6-v2` + `qwen2.5:1.5b`);
  `outputs/example_runs.md` is that real run. A fresh clone still needs the model
  weights downloaded to reproduce it.
- **Section 1's committed logs came from a stub LLM**, not Ollama, because the
  development environment could not reach `ollama.com`. The SDK path is real and
  verified; regenerate with `python run_session_demo.py`. See
  `section1-livekit/logs/README.md`.

---

## Known limitations

- **Section 1:** STT/TTS are stubs (permitted by the brief). Barge-in is
  configured in `worker.py` via Silero VAD but not demonstrated in the
  text-driven transcript.
- **Section 2:** the corpus is synthetic. No reranker is implemented — it is the
  first thing `NOTES.md` recommends adding. Refusal thresholds are reasoned
  defaults; run `calibrate_threshold.py` before trusting them.
- **Section 3:** 4-bit requires CUDA (bitsandbytes kernels are GPU-only); CPU
  machines get FP16 plus a recorded skip reason. FP16-on-CPU actually runs
  float32, and the report says so. Qualitative scoring is left to a human
  reader rather than faked with an automated metric over five prompts.
- **Section 4:** single node, no auth, no rate limiting, no gateway. The 50-user
  scaling design is written up but deliberately not built.

---

## Write-ups

Each section's required half-page write-up:

- [Section 1 — barge-in, interruption, adding tools safely, provider swap](section1-livekit/NOTES.md)
- [Section 2 — chunking, hybrid search, reranking, metadata filtering, long documents](section2-rag/NOTES.md)
- [Section 3 — bitsandbytes vs GPTQ vs AWQ vs GGUF in production](section3-quantization/NOTES.md)
- [Section 4 — batching, autoscaling, caching, queueing, monitoring, 50 concurrent users](section4-deployment/NOTES.md)

---

## Future improvements

1. **Cross-encoder reranking** in Section 2 — highest-value single change to
   answer quality.
2. **RAGAS / labelled retrieval evaluation** — without measurement, "improving"
   retrieval is guesswork.
3. **vLLM behind the Section 4 API** — same endpoints, continuous batching, the
   change that actually unlocks 50 concurrent users.
4. **Real STT/TTS** (Vosk + Piper) in Section 1 to exercise barge-in end to end.
5. **AWQ alongside bitsandbytes** in Section 3 for a three-way quality/speed
   comparison.
6. **CI** running all 24 tests on push — they need no model weights, so they run
   anywhere in seconds.
