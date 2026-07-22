"""API tests using a mocked inference engine.

These verify routing, schema validation, SSE framing and error handling
without loading model weights, so they run in CI on any machine.

Real generation quality is exercised by actually running the service; see
`results/README.md`.

Run:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


class MockEngine:
    """Stand-in for `InferenceEngine` with deterministic output."""

    def __init__(self, loaded: bool = True) -> None:
        self.is_loaded = loaded
        self.device = "cpu"
        self.quantized = False

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, int, int, float]:
        return "Hello from the mock model.", 12, 6, 0.1234

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        for token in ["Hello", " from", " the", " mock", "."]:
            await asyncio.sleep(0)
            yield token


class BrokenEngine(MockEngine):
    """Engine that fails mid-stream, to exercise error framing."""

    async def stream(self, *args, **kwargs) -> AsyncIterator[str]:
        yield "partial"
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(main_module, "engine", MockEngine())
    return TestClient(main_module.app)


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_reports_loading(monkeypatch) -> None:
    """A not-yet-loaded model must be distinguishable from a broken one."""
    monkeypatch.setattr(main_module, "engine", MockEngine(loaded=False))
    response = TestClient(main_module.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "loading"


def test_chat_returns_usage(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "Hello from the mock model."
    assert body["usage"]["total_tokens"] == 18
    assert body["latency_s"] > 0


def test_chat_rejects_empty_messages(client: TestClient) -> None:
    assert client.post("/chat", json={"messages": []}).status_code == 422


def test_chat_rejects_out_of_range_temperature(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "x"}], "temperature": 9.0},
    )
    assert response.status_code == 422


def test_chat_503_when_model_unloaded(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "engine", MockEngine(loaded=False))
    response = TestClient(main_module.app).post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 503


def test_stream_emits_sse_tokens(client: TestClient) -> None:
    """Tokens must arrive as separate SSE events terminated by [DONE]."""
    with client.stream(
        "POST", "/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        lines = [
            line for line in response.iter_lines() if line.startswith("data: ")
        ]

    assert lines[-1] == "data: [DONE]"
    tokens = [
        json.loads(line.removeprefix("data: "))["token"]
        for line in lines
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert "".join(tokens) == "Hello from the mock."


def test_stream_reports_midstream_error(monkeypatch) -> None:
    """Once streaming starts the status is committed, so errors go in-band."""
    monkeypatch.setattr(main_module, "engine", BrokenEngine())
    client = TestClient(main_module.app)
    with client.stream(
        "POST", "/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    ) as response:
        lines = [
            line for line in response.iter_lines() if line.startswith("data: ")
        ]

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in lines
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert any("error" in p for p in payloads)


def test_streaming_disables_proxy_buffering(client: TestClient) -> None:
    """X-Accel-Buffering:no keeps nginx from defeating the stream."""
    with client.stream(
        "POST", "/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    ) as response:
        assert response.headers.get("x-accel-buffering") == "no"
