"""Locust load profile for the inference API.

Complements `benchmark.py`: that script measures a precise burst of N
simultaneous requests, while Locust models sustained traffic with think-time
and gives a live web UI, ramp-up control and percentile charts.

Headless (matches the 10-concurrent-user requirement):
    locust -f loadtest/locustfile.py --headless \
           -u 10 -r 10 -t 60s --host http://localhost:8000 \
           --html results/locust_report.html

Web UI:
    locust -f loadtest/locustfile.py --host http://localhost:8000
    # then open http://localhost:8089
"""

from __future__ import annotations

import json
import time

from locust import HttpUser, between, events, task

PROMPTS = [
    "Explain what a microcontroller is in three sentences.",
    "What is the difference between a Raspberry Pi and an Arduino?",
    "Write a one-sentence summary of what an ADC does.",
    "List three common uses for a GPIO pin.",
]


class InferenceUser(HttpUser):
    """Simulates a user alternating between streaming and blocking calls."""

    # Think time between requests; without this Locust measures a hammer test
    # rather than realistic user behaviour.
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Fail fast if the model is not ready, instead of timing 503s."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"health returned {response.status_code}")
                return
            if not response.json().get("model_loaded"):
                response.failure("model not loaded")

    @task(3)
    def stream_chat(self) -> None:
        """Streaming request; reports time-to-first-token explicitly."""
        payload = {
            "messages": [{"role": "user", "content": PROMPTS[0]}],
            "max_new_tokens": 64,
        }
        start = time.perf_counter()
        ttft: float | None = None
        tokens = 0

        with self.client.post(
            "/stream",
            json=payload,
            stream=True,
            catch_response=True,
            name="/stream",
        ) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
                return

            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                body = line.removeprefix("data: ")
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if "error" in event:
                    response.failure(event["error"])
                    return
                if ttft is None:
                    ttft = time.perf_counter() - start
                tokens += 1

            if tokens == 0:
                response.failure("no tokens received")
            else:
                response.success()

        if ttft is not None:
            # Surface TTFT as its own Locust metric.
            events.request.fire(
                request_type="TTFT",
                name="/stream first-token",
                response_time=ttft * 1000,
                response_length=0,
                exception=None,
                context={},
            )

    @task(1)
    def blocking_chat(self) -> None:
        """Non-streaming request, for comparison against the streaming path."""
        payload = {
            "messages": [{"role": "user", "content": PROMPTS[1]}],
            "max_new_tokens": 64,
        }
        with self.client.post("/chat", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}")
            elif not response.json().get("content"):
                response.failure("empty content")
            else:
                response.success()
