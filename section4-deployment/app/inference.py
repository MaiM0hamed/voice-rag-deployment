"""Model loading and inference, including true token-by-token streaming.

Design notes:

* The model is loaded once at application startup (FastAPI lifespan) rather
  than per request -- loading Qwen2.5-1.5B takes seconds and would dominate
  latency otherwise.
* `transformers.generate()` is synchronous and CPU/GPU-bound. Calling it
  directly inside an async endpoint would block the event loop and serialise
  every request. Instead generation runs in a worker thread and results are
  handed back to the event loop.
* Streaming uses `TextIteratorStreamer`, which yields decoded text as tokens
  are produced. This gives a genuinely low time-to-first-token rather than
  buffering the full response and chunking it afterwards (which would fake
  streaming while keeping TTFT equal to total latency).
* A semaphore bounds concurrent generations so the accelerator is not
  oversubscribed under load.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from threading import Thread
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps a causal LM and exposes blocking and streaming generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "unloaded"
        self.quantized: bool = False
        self._loaded = False
        # Bounds in-flight generations; see module docstring.
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    @property
    def is_loaded(self) -> bool:
        """Whether the model is ready to serve."""
        return self._loaded

    def load(self) -> None:
        """Load tokenizer and model. Called once during startup.

        Raises:
            RuntimeError: If loading fails, so the container exits loudly
                rather than serving 500s indefinitely.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        start = time.perf_counter()
        model_id = self.settings.model_id
        logger.info("Loading model %s ...", model_id)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            has_cuda = torch.cuda.is_available()
            kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}

            if self.settings.load_in_4bit:
                if not has_cuda:
                    raise RuntimeError(
                        "LOAD_IN_4BIT=true requires a CUDA GPU (bitsandbytes "
                        "kernels are GPU-only). Set LOAD_IN_4BIT=false to serve "
                        "at full precision on CPU."
                    )
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
                self.quantized = True
            else:
                kwargs["torch_dtype"] = torch.bfloat16 if has_cuda else torch.float32
                if has_cuda:
                    kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if not has_cuda and not self.settings.load_in_4bit:
                self.model = self.model.to("cpu")

            self.model.eval()
            self.device = str(next(self.model.parameters()).device)
            self._loaded = True

            logger.info(
                "Model ready in %.1fs (device=%s, quantized=%s)",
                time.perf_counter() - start,
                self.device,
                self.quantized,
            )
        except Exception as exc:
            logger.exception("Model loading failed")
            raise RuntimeError(f"Failed to load {model_id}: {exc}") from exc

    def unload(self) -> None:
        """Release the model on shutdown."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
        logger.info("Model unloaded")

    def _build_inputs(self, messages: list[dict[str, str]]) -> Any:
        """Apply the chat template and tokenize."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def _generation_kwargs(
        self, max_new_tokens: int | None, temperature: float | None
    ) -> dict[str, Any]:
        """Merge per-request overrides with server defaults."""
        temp = (
            temperature
            if temperature is not None
            else self.settings.default_temperature
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.settings.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        # temperature=0 means greedy; passing temperature=0 to sampling is invalid.
        if temp > 0:
            kwargs.update({"do_sample": True, "temperature": temp})
        else:
            kwargs["do_sample"] = False
        return kwargs

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, int, int, float]:
        """Generate a complete response.

        Args:
            messages: Conversation history.
            max_new_tokens: Optional per-request cap.
            temperature: Optional per-request sampling temperature.

        Returns:
            ``(text, prompt_tokens, completion_tokens, latency_seconds)``.
        """
        import torch

        if not self._loaded:
            raise RuntimeError("Model is not loaded")

        async with self._semaphore:
            start = time.perf_counter()
            inputs = self._build_inputs(messages)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            kwargs = self._generation_kwargs(max_new_tokens, temperature)

            def _run() -> Any:
                with torch.inference_mode():
                    return self.model.generate(**inputs, **kwargs)

            # Off the event loop: generate() is blocking and CPU/GPU-bound.
            output_ids = await asyncio.to_thread(_run)

            generated = output_ids[0][prompt_tokens:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            latency = time.perf_counter() - start

            return text, prompt_tokens, int(generated.shape[-1]), latency

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield decoded text incrementally as tokens are generated.

        Args:
            messages: Conversation history.
            max_new_tokens: Optional per-request cap.
            temperature: Optional per-request sampling temperature.

        Yields:
            Text fragments, in generation order.
        """
        import torch
        from transformers import TextIteratorStreamer

        if not self._loaded:
            raise RuntimeError("Model is not loaded")

        async with self._semaphore:
            inputs = self._build_inputs(messages)
            kwargs = self._generation_kwargs(max_new_tokens, temperature)

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            def _run() -> None:
                try:
                    with torch.inference_mode():
                        self.model.generate(**inputs, streamer=streamer, **kwargs)
                except Exception:
                    logger.exception("Streaming generation failed")
                    # Ending the stream unblocks the consumer; the endpoint
                    # surfaces the error to the client.
                    streamer.end()

            thread = Thread(target=_run, daemon=True)
            thread.start()

            loop = asyncio.get_running_loop()
            iterator = iter(streamer)
            sentinel = object()

            while True:
                # next() on the streamer blocks until the next token is ready,
                # so it must not run on the event loop.
                chunk = await loop.run_in_executor(
                    None, lambda: next(iterator, sentinel)
                )
                if chunk is sentinel:
                    break
                if chunk:
                    yield chunk
