"""Generate the Section 4 benchmark artifacts.

Wraps `loadtest/benchmark.py` so every results directory in this repo exposes
the same `generate_results.py` entrypoint. It runs the load test at one or more
concurrency levels and writes the reports here.

The service must already be running. Start it with either:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
    # or
    docker compose up --build

Then:

    python results/generate_results.py                  # 10 concurrent users
    python results/generate_results.py --users 1 10 25  # a sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

SECTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SECTION_ROOT))

from loadtest.benchmark import (  # noqa: E402
    DEFAULT_PROMPT,
    render_markdown,
    run_load_test,
    summarize,
)

RESULTS_DIR = Path(__file__).resolve().parent


async def _check_service(url: str) -> bool:
    """Verify the service is up and the model is loaded."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/health", timeout=10.0)
            response.raise_for_status()
            body = response.json()
    except Exception as exc:
        print(f"ERROR: cannot reach {url}/health ({exc})")
        print("Start the service first:  uvicorn app.main:app --port 8000")
        return False

    if not body.get("model_loaded"):
        print(f"ERROR: service is up but model_loaded=false (status={body.get('status')})")
        print("Weights may still be downloading; wait and retry.")
        return False

    print(f"Service ready: model={body['model']} device={body['device']} "
          f"quantized={body['quantized']}")
    return True


async def main_async() -> int:
    """Run the load test at each requested concurrency level."""
    parser = argparse.ArgumentParser(description="Generate Section 4 benchmarks.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        default=[10],
        help="Concurrency level(s) to test. The assessment requires 10.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if not await _check_service(args.url):
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep: list[dict] = []

    for users in args.users:
        print(f"\n--- {users} concurrent user(s) ---")
        start = time.perf_counter()
        metrics = await run_load_test(
            args.url, users, args.prompt, args.max_new_tokens
        )
        wall = time.perf_counter() - start
        summary = summarize(metrics, users, wall)
        sweep.append(summary)

        print(f"  ok={summary['requests_ok']}/{summary['requests_total']}  "
              f"TTFT median={summary['ttft_s']['median']}s  "
              f"total median={summary['total_latency_s']['median']}s")

        # The headline artifacts use the primary (first) concurrency level.
        if users == args.users[0]:
            (RESULTS_DIR / "load_test_results.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            (RESULTS_DIR / "load_test_report.md").write_text(
                render_markdown(summary), encoding="utf-8"
            )

    if len(sweep) > 1:
        lines = [
            "# Concurrency sweep",
            "",
            "| Users | OK | TTFT median (s) | Total median (s) | Throughput (req/s) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for summary in sweep:
            lines.append(
                f"| {summary['concurrent_users']} "
                f"| {summary['requests_ok']}/{summary['requests_total']} "
                f"| {summary['ttft_s']['median']} "
                f"| {summary['total_latency_s']['median']} "
                f"| {summary['throughput_req_per_s']} |"
            )
        (RESULTS_DIR / "concurrency_sweep.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        (RESULTS_DIR / "concurrency_sweep.json").write_text(
            json.dumps(sweep, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {RESULTS_DIR / 'concurrency_sweep.md'}")

    print(f"Wrote {RESULTS_DIR / 'load_test_report.md'}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
