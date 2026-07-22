"""Request and response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single conversation turn."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Payload for `/chat` and `/stream`."""

    messages: list[ChatMessage] = Field(
        min_length=1, description="Conversation history, oldest first."
    )
    max_new_tokens: int | None = Field(
        default=None, ge=1, le=2048, description="Overrides the server default."
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [{"role": "user", "content": "What is a Raspberry Pi?"}],
                "max_new_tokens": 128,
                "temperature": 0.7,
            }
        }
    }


class Usage(BaseModel):
    """Token accounting for one completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Non-streaming completion result."""

    content: str
    model: str
    usage: Usage
    latency_s: float = Field(description="Server-side wall-clock generation time.")


class HealthResponse(BaseModel):
    """Health/readiness probe result."""

    status: Literal["ok", "loading", "error"]
    model: str
    model_loaded: bool
    device: str
    quantized: bool
    uptime_s: float


class ErrorResponse(BaseModel):
    """Structured error body."""

    detail: str
    error_type: str
