"""Application configuration, driven entirely by environment variables.

Defaults are chosen so `docker run` works with no configuration, while every
value can be overridden for a real deployment.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings for the inference service."""

    # --- model ---
    model_id: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model id to serve.",
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Serve the bitsandbytes NF4 build from Section 3 (needs CUDA).",
    )
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # --- server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # --- capacity ---
    # Bounds concurrent generations so the GPU is not oversubscribed. Requests
    # beyond this queue on the semaphore rather than thrashing memory.
    max_concurrent_requests: int = Field(default=4, ge=1)
    request_timeout_s: float = Field(default=120.0, gt=0)

    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "extra": "ignore",
        "protected_namespaces": (),
    }


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
