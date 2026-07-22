"""FastAPI service exposing the Section 3 model.

Endpoints:
    GET  /health   liveness + readiness (reports whether weights are loaded)
    POST /chat     complete response in one JSON body
    POST /stream   Server-Sent Events, token-by-token

Why FastAPI rather than vLLM/TGI: this assessment serves a 1.5B model where the
requirement is a correct, containerised, streaming HTTP surface with measured
latency. FastAPI gives full control over the request lifecycle, health
semantics and error handling in ~200 readable lines, with no additional
infrastructure. vLLM would be the right answer at higher scale -- its
PagedAttention and continuous batching dominate for many concurrent users --
and `NOTES.md` sets out exactly when I would switch.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .inference import InferenceEngine
from .schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse, Usage

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

engine = InferenceEngine(settings)
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup and release it on shutdown."""
    logger.info("Starting up; loading model...")
    try:
        engine.load()
    except Exception:
        # Log and continue: /health then reports not-ready, which is more
        # debuggable for an orchestrator than a container that vanishes.
        logger.exception("Startup model load failed; service will report unhealthy")
    yield
    engine.unload()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Electro Pi Inference API",
    description="Serves Qwen2.5-1.5B-Instruct with streaming support.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status and duration for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info(
        "%s %s -> %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    response.headers["X-Response-Time"] = f"{duration:.4f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a structured error instead of leaking a stack trace."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "error_type": type(exc).__name__},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Report service and model status.

    Returns 200 with ``status="loading"`` when the process is up but weights
    are not resident, so a load balancer can distinguish "starting" from
    "broken" and delay routing traffic accordingly.
    """
    return HealthResponse(
        status="ok" if engine.is_loaded else "loading",
        model=settings.model_id,
        model_loaded=engine.is_loaded,
        device=engine.device,
        quantized=engine.quantized,
        uptime_s=round(time.time() - _start_time, 2),
    )


def _require_model() -> None:
    """Reject requests while the model is unavailable."""
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Check /health.",
        )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["inference"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not loaded yet"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate a complete response.

    Args:
        request: Conversation plus optional generation overrides.

    Returns:
        The generated text with token usage and server-side latency.
    """
    _require_model()
    messages = [m.model_dump() for m in request.messages]

    try:
        text, prompt_tokens, completion_tokens, latency = await engine.generate(
            messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return ChatResponse(
        content=text,
        model=settings.model_id,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        latency_s=round(latency, 4),
    )


@app.post(
    "/stream",
    tags=["inference"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not loaded yet"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def stream(request: ChatRequest) -> StreamingResponse:
    """Stream the response token by token as Server-Sent Events.

    Each event carries a JSON payload ``{"token": "..."}``; the stream ends
    with ``data: [DONE]``. Errors mid-stream are delivered as an ``error``
    field rather than a truncated body, since HTTP status is already committed
    once streaming begins.
    """
    _require_model()
    messages = [m.model_dump() for m in request.messages]

    async def event_source() -> AsyncIterator[str]:
        try:
            async for chunk in engine.stream(
                messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            ):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Stream failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering, which would otherwise defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )
