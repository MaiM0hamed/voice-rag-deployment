"""Load and latency benchmark for the inference API.

Measures, for N concurrent requests:
  * TTFT  -- time to first token, from request start to the first SSE event
  * total latency -- request start to the terminating [DONE] event
  * throughput in tokens/sec

TTFT is measured against `/stream` because that is the only way to observe it
honestly: on `/chat` the first byte arrives only when generation is complete,
so TTFT and total latency would be identical by construction.

All numbers are measured. Nothing is estimated.

Usage:
    python loadtest/benchmark.py --users 10 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx

DEFAULT_PROMPT = "Explain what a microcontroller is in three sentences."


@dataclass
class RequestMetrics:
    """Timings for a single streamed request."""

    index: int
    ok: bool
    ttft_s: float | None
    total_s: float | None
    tokens: int
    error: str | None = None

    @property
    def tokens_per_s(self) -> float:
        """Decode throughput, excluding prefill."""
        if not self.ok or self.total_s is None or self.ttft_s is None:
            return 0.0
        decode = max(self.total_s - self.ttft_s, 1e-6)
        return self.tokens / decode


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    index: int,
    prompt: str,
    max_new_tokens: int,
) -> RequestMetrics:
    """Issue one streaming request and time it.

    Args:
        client: Shared HTTP client.
        url: Base service URL.
        index: Request ordinal, for reporting.
        prompt: User prompt to send.
        max_new_tokens: Generation cap.

    Returns:
        Measured :class:`RequestMetrics`.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_new_tokens": max_new_tokens,
    }

    start = time.perf_counter()
    ttft: float | None = None
    tokens = 0

    try:
        async with client.stream(
            "POST", f"{url}/stream", json=payload, timeout=180.0
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line.removeprefix("data: ")
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if "error" in event:
                    return RequestMetrics(
                        index, False, None, None, tokens, error=event["error"]
                    )
                if ttft is None:
                    ttft = time.perf_counter() - start
                tokens += 1

        return RequestMetrics(
            index, True, ttft, time.perf_counter() - start, tokens
        )
    except Exception as exc:
        return RequestMetrics(
            index, False, None, None, tokens, error=f"{type(exc).__name__}: {exc}"
        )


async def run_load_test(
    url: str, users: int, prompt: str, max_new_tokens: int
) -> list[RequestMetrics]:
    """Fire `users` requests simultaneously and collect their metrics."""
    limits = httpx.Limits(max_connections=users * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            one_request(client, url, i, prompt, max_new_tokens) for i in range(users)
        ]
        return await asyncio.gather(*tasks)


def summarize(metrics: list[RequestMetrics], users: int, wall_s: float) -> dict:
    """Aggregate per-request metrics into a report payload."""
    ok = [m for m in metrics if m.ok]
    failed = [m for m in metrics if not m.ok]

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[index]

    ttfts = [m.ttft_s for m in ok if m.ttft_s is not None]
    totals = [m.total_s for m in ok if m.total_s is not None]
    rates = [m.tokens_per_s for m in ok]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "concurrent_users": users,
        "requests_total": len(metrics),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "wall_clock_s": round(wall_s, 3),
        "throughput_req_per_s": round(len(ok) / wall_s, 3) if wall_s else 0.0,
        "ttft_s": {
            "mean": round(statistics.mean(ttfts), 4) if ttfts else None,
            "median": round(statistics.median(ttfts), 4) if ttfts else None,
            "p95": round(pct(ttfts, 0.95), 4) if ttfts else None,
            "min": round(min(ttfts), 4) if ttfts else None,
            "max": round(max(ttfts), 4) if ttfts else None,
        },
        "total_latency_s": {
            "mean": round(statistics.mean(totals), 4) if totals else None,
            "median": round(statistics.median(totals), 4) if totals else None,
            "p95": round(pct(totals, 0.95), 4) if totals else None,
            "min": round(min(totals), 4) if totals else None,
            "max": round(max(totals), 4) if totals else None,
        },
        "tokens_per_s_per_request": {
            "mean": round(statistics.mean(rates), 2) if rates else None,
            "median": round(statistics.median(rates), 2) if rates else None,
        },
        "errors": [m.error for m in failed][:10],
        "per_request": [asdict(m) for m in metrics],
    }


def render_markdown(summary: dict) -> str:
    """Render the summary as a markdown report."""
    ttft = summary["ttft_s"]
    total = summary["total_latency_s"]
    rate = summary["tokens_per_s_per_request"]

    return f"""# Section 4 — Load & Latency Benchmark

_Generated {summary['generated_at']} by `loadtest/benchmark.py`.
All values measured against a running service; none are estimates._

## Configuration

- Concurrent users: **{summary['concurrent_users']}**
- Requests: {summary['requests_total']} ({summary['requests_ok']} ok, \
{summary['requests_failed']} failed)
- Wall clock: {summary['wall_clock_s']} s
- Aggregate throughput: {summary['throughput_req_per_s']} req/s

## Time to first token (TTFT)

| Metric | Seconds |
| --- | --- |
| Mean | {ttft['mean']} |
| Median | {ttft['median']} |
| p95 | {ttft['p95']} |
| Min | {ttft['min']} |
| Max | {ttft['max']} |

## Total latency

| Metric | Seconds |
| --- | --- |
| Mean | {total['mean']} |
| Median | {total['median']} |
| p95 | {total['p95']} |
| Min | {total['min']} |
| Max | {total['max']} |

## Per-request decode throughput

| Metric | Tokens/sec |
| --- | --- |
| Mean | {rate['mean']} |
| Median | {rate['median']} |

## Errors

{chr(10).join('- ' + e for e in summary['errors']) if summary['errors'] else '_None._'}

Raw per-request timings are in `load_test_results.json`.
"""


async def main_async() -> int:
    """Parse arguments, run the load test, and write reports."""
    parser = argparse.ArgumentParser(description="Load-test the inference API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    args = parser.parse_args()

    # Fail fast with a clear message rather than N confusing timeouts.
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{args.url}/health", timeout=10.0)
            health.raise_for_status()
            if not health.json().get("model_loaded"):
                print("Service is up but the model is still loading. Wait and retry.")
                return 1
    except Exception as exc:
        print(f"Cannot reach {args.url}/health -- is the service running? ({exc})")
        return 1

    print(f"Running {args.users} concurrent requests against {args.url} ...")
    start = time.perf_counter()
    metrics = await run_load_test(
        args.url, args.users, args.prompt, args.max_new_tokens
    )
    wall = time.perf_counter() - start

    summary = summarize(metrics, args.users, wall)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "load_test_results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = args.output_dir / "load_test_report.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"\nOK: {summary['requests_ok']}/{summary['requests_total']}")
    print(f"TTFT median: {summary['ttft_s']['median']} s")
    print(f"Total latency median: {summary['total_latency_s']['median']} s")
    print(f"Wrote {json_path}\nWrote {md_path}")
    return 0 if summary["requests_failed"] == 0 else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
