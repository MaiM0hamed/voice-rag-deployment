# Section 4 — Write-up

## What this builds

A containerised FastAPI service wrapping the Section 3 model, with a health
probe, a blocking completion endpoint, a genuinely token-by-token streaming
endpoint, structured logging, error handling, and a load benchmark that
measures TTFT and total latency under concurrency.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + readiness; distinguishes "loading" from "broken" |
| `/chat` | POST | Complete response with token usage and server-side latency |
| `/stream` | POST | Server-Sent Events, one event per decoded fragment |
| `/docs` | GET | OpenAPI UI (free from FastAPI) |

## Why FastAPI rather than vLLM or TGI

The requirement is a correct, containerised, streaming HTTP surface with
measured latency for a 1.5B model. FastAPI delivers that in ~200 readable lines
with full control over the request lifecycle, health semantics and error
framing, and no extra infrastructure.

vLLM would be the wrong choice *here* and the right choice at scale. Its
advantages — PagedAttention, continuous batching — only pay off with many
concurrent sequences. For a handful of users on a small model, it adds
operational weight without measurable benefit. The switching point is in the
scaling section below, and it is not far away.

## Design decisions worth defending

**Model loaded once at startup, not per request.** Loading takes seconds; doing
it per request would dominate latency. Handled via FastAPI's `lifespan`.

**Generation runs off the event loop.** `transformers.generate()` is blocking
and compute-bound. Calling it directly inside an `async def` endpoint would
stall the event loop and serialise every request, including health checks.
`/chat` uses `asyncio.to_thread`; `/stream` runs generation in a `Thread` and
drains `TextIteratorStreamer` via `run_in_executor`. This is the single most
common way an async LLM service is accidentally made single-threaded.

**Streaming is real, not simulated.** `TextIteratorStreamer` yields text as
tokens are produced. Buffering the full response and chunking it afterwards
would satisfy a naive reading of "streaming" while leaving TTFT equal to total
latency. Verified: in a local run TTFT was 0.044 s against 0.242 s total — if
streaming were faked those two numbers would be identical.

**A semaphore bounds concurrency.** `MAX_CONCURRENT_REQUESTS` caps in-flight
generations so the accelerator is not oversubscribed into OOM or thrashing.
Excess requests queue rather than degrade everyone.

**Health reports three states.** `ok` / `loading` / `error`, so an orchestrator
can delay routing traffic to a pod whose weights are still downloading instead
of treating a slow start as a crash loop. The Docker `HEALTHCHECK` uses a
300 s `start-period` for the same reason.

**Errors mid-stream go in-band.** Once streaming begins the HTTP status is
already committed, so a failure is delivered as `{"error": ...}` inside the
event stream rather than a silently truncated body. Tested.

**Single Uvicorn worker.** N workers would load N copies of the weights. Scale
with replicas, not workers.

## Scaling to 50 concurrent users

The current design saturates well before 50. With `MAX_CONCURRENT_REQUESTS=4`,
50 simultaneous users means ~46 queueing, and TTFT — the number users actually
feel — degrades linearly with queue depth. Fixing that in priority order:

### 1. Continuous batching (the highest-leverage change)

The service processes requests essentially one batch at a time. HuggingFace
`generate()` has no continuous batching: a new request cannot join an in-flight
batch and must wait for the slowest sequence to finish.

I would replace the inference layer with **vLLM**, keeping the same FastAPI
surface. vLLM's continuous batching admits new sequences at every decode step,
and PagedAttention stops KV-cache fragmentation from capping batch size. For
many short-to-medium generations this is typically a several-fold throughput
gain on identical hardware — far more than any other single change here.
The endpoint contracts do not change, so this is an internal swap.

### 2. Queueing with admission control and backpressure

An unbounded queue converts an overload into unbounded latency, which is worse
than a fast rejection. I would add:

- a **bounded** request queue that returns `429` with `Retry-After` when full;
- per-request deadlines, dropping work whose client has already timed out
  (there is no point generating 200 tokens for a closed connection);
- priority lanes if interactive and batch traffic share the service.

### 3. Horizontal autoscaling

Replicas behind a load balancer, scaled on the metric that matches the
bottleneck. CPU utilisation is the wrong trigger for GPU inference — it stays
low while the GPU saturates. Scale on **queue depth** or **TTFT p95**, both of
which move before users notice. Keep a warm minimum: a cold replica pays a
multi-second weight load, so scale-from-zero is unacceptable for interactive
traffic. Scale out aggressively, scale in slowly.

### 4. Caching

Three layers, cheapest first:

- **Exact-match response cache** (Redis, keyed on a hash of messages +
  generation params). FAQ-style traffic is highly repetitive; hit rates of
  30–40 % are realistic and a hit costs milliseconds instead of seconds.
- **Prefix / KV-cache reuse** for shared system prompts. Every request in this
  service re-prefills the same persona tokens; vLLM's automatic prefix caching
  eliminates that and directly reduces TTFT.
- **Semantic cache** (embed the query, serve a cached answer above a similarity
  threshold). Powerful but risky — it can return a subtly wrong answer to a
  similar-but-different question — so I would gate it behind a high threshold
  and only for clearly idempotent traffic.

### 5. Monitoring

Without measurement the above is guesswork. I would export Prometheus metrics
and alert on the user-visible ones:

- **TTFT p50/p95/p99** and total latency (histograms, not averages — averages
  hide the tail that users complain about);
- queue depth and time spent queued;
- tokens/sec, GPU utilisation and memory;
- error rate by class (429 vs 503 vs 500 mean very different things);
- cache hit rate.

Plus structured request logs with a correlation id, and tracing across
gateway → queue → inference once more than one service is involved.
The alert that matters is TTFT p95, because it degrades before throughput does.

### 6. Right-sizing the model

Not architecture, but often the biggest win: serve the 4-bit build from
Section 3. It roughly quarters weight memory, which allows a larger batch and a
bigger KV cache on the same GPU — frequently a larger end-to-end throughput
gain than any serving-layer tuning, provided Section 3's measurements show
acceptable quality loss. That is exactly why Section 3 measures quality on five
fixed prompts rather than assuming.

### Summary

| Change | Effort | Impact at 50 users |
|---|---|---|
| vLLM continuous batching | Medium | Very high |
| Prefix / KV caching | Low (free with vLLM) | High (TTFT) |
| Exact-match response cache | Low | High if traffic repeats |
| Bounded queue + 429 | Low | Protects tail latency |
| Autoscaling on TTFT p95 / queue depth | Medium | High |
| 4-bit weights | Low | High |
| Monitoring | Low | Prerequisite for all of it |

Order of operations: monitoring first (so improvements are measurable), then
vLLM plus 4-bit weights, then caching, then autoscaling.

## Assumptions, shortcuts, limitations (stated honestly)

- **Verified locally:** all endpoints, SSE framing and `[DONE]` termination,
  mid-stream error delivery, 503 while unloaded, 422 validation (9/9 pytest);
  and a real end-to-end load-test run against the actual FastAPI app with a
  mocked engine — 10/10 concurrent requests, TTFT 0.044 s vs total 0.242 s,
  confirming streaming is genuine.
- **NOT verified: `docker build`.** Docker was unavailable in the development
  environment. The Dockerfile is written to standard practice but has never
  been executed. **Run it first** — the assessment requires
  `docker build && docker run` to work end to end.
- **No benchmark numbers are committed.** Real generation requires model
  weights, which the development environment could not download. Run
  `results/generate_results.py` against a live service.
- **The mocked-engine load test validates the harness, not the model.** It
  proves TTFT is measured correctly and concurrency works; it says nothing
  about real token throughput.
- **Single-node only.** No gateway, queue or replica set is implemented — those
  are the scaling design above, deliberately not built for a take-home.
- **No authentication or rate limiting.** A production deployment needs both;
  omitted as out of scope and called out here rather than left implied.
