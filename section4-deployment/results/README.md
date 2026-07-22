# results/ — GENERATED ARTIFACTS

Everything here except `generate_results.py` and this README is **produced by
running the load test against a live service**. No latency or throughput
figures in this repo are estimated or copied from elsewhere.

## How to generate

**1. Start the service** (either way works):

```bash
# Docker (recommended — this is also the docker build && docker run check)
docker compose up --build

# or locally
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or /cu121
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait until `/health` reports `"model_loaded": true` — the first start downloads
~3 GB of weights into the `hf-cache` volume. Subsequent starts are fast.

```bash
curl localhost:8000/health
```

**2. Run the benchmark:**

```bash
python results/generate_results.py                  # 10 concurrent users
python results/generate_results.py --users 1 10 25  # concurrency sweep
```

## What gets produced

| File | Contents |
|---|---|
| `load_test_report.md` | TTFT and total-latency tables (mean/median/p95/min/max), throughput, errors |
| `load_test_results.json` | Every per-request timing and token count |
| `concurrency_sweep.md` / `.json` | Only when `--users` has more than one level |
| `locust_report.html` | Only if you run the Locust profile (below) |

## Optional: Locust

`benchmark.py` fires one precise burst of N simultaneous requests. Locust models
sustained traffic with think-time, ramp-up and live percentile charts:

```bash
locust -f loadtest/locustfile.py --headless \
       -u 10 -r 10 -t 60s --host http://localhost:8000 \
       --html results/locust_report.html
```

## How the numbers are measured

- **TTFT** is measured against `/stream`, timing from request start to the first
  SSE event. Measuring it on `/chat` would be meaningless — the first byte
  arrives only when generation finishes, so TTFT would equal total latency by
  construction.
- **Total latency** runs to the terminating `data: [DONE]` event.
- **Per-request throughput** is computed over the decode phase only
  (`total − ttft`), which is what governs perceived streaming speed.
- **p95** is a nearest-rank percentile. With only 10 samples it is indicative,
  not statistically robust; raise `--users` or use Locust over a longer window
  for tail latency you can rely on.

## Expected shape of the results

On a single-GPU or CPU host with `MAX_CONCURRENT_REQUESTS=4`, 10 simultaneous
requests will queue: roughly four run concurrently while six wait on the
semaphore. Expect median TTFT to rise noticeably with concurrency — that
queueing is the exact bottleneck `NOTES.md` addresses for the 50-user case.
