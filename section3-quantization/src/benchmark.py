"""Benchmark harness: load a model at a given precision and measure it.

Every number this module reports is measured at runtime. Nothing is estimated,
interpolated, or hard-coded.

Measured per variant:
  * model size on disk / in memory (sum of parameter tensor bytes)
  * peak VRAM (torch.cuda.max_memory_allocated) or process RSS on CPU
  * prefill latency == time to first token (TTFT)
  * decode throughput in tokens/sec
  * total wall-clock latency
  * the generated text, for qualitative comparison
"""

from __future__ import annotations

import gc
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Literal

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import PROMPTS, BenchmarkConfig, config

logger = logging.getLogger(__name__)

Precision = Literal["fp16", "int4"]


@dataclass
class PromptResult:
    """Measurements for a single prompt at a single precision."""

    prompt_id: str
    category: str
    prompt: str
    output: str
    ttft_s: float
    total_s: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_s: float


@dataclass
class VariantResult:
    """Aggregate measurements for one precision variant."""

    precision: Precision
    model_id: str
    load_time_s: float
    param_bytes: int
    peak_memory_bytes: int
    device: str
    dtype: str
    prompts: list[PromptResult] = field(default_factory=list)

    @property
    def param_mb(self) -> float:
        """Parameter footprint in megabytes."""
        return self.param_bytes / 1024**2

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory during generation in megabytes."""
        return self.peak_memory_bytes / 1024**2

    @property
    def mean_tokens_per_s(self) -> float:
        """Mean decode throughput across prompts."""
        values = [p.tokens_per_s for p in self.prompts if p.tokens_per_s > 0]
        return sum(values) / len(values) if values else 0.0

    @property
    def median_ttft_s(self) -> float:
        """Median time to first token across prompts."""
        values = [p.ttft_s for p in self.prompts]
        return median(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON report."""
        payload = asdict(self)
        payload["param_mb"] = round(self.param_mb, 2)
        payload["peak_memory_mb"] = round(self.peak_memory_mb, 2)
        payload["mean_tokens_per_s"] = round(self.mean_tokens_per_s, 2)
        payload["median_ttft_s"] = round(self.median_ttft_s, 4)
        return payload


def describe_environment() -> dict[str, Any]:
    """Capture the hardware/software context the numbers were produced on.

    Benchmark numbers are meaningless without this: 4-bit is often *slower*
    than FP16 on CPU and *faster* on GPU, so the reader must know which.
    """
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
        info["cuda_version"] = torch.version.cuda
    return info


def _reset_memory_stats() -> None:
    """Clear caches and memory counters between variants."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_bytes() -> int:
    """Peak VRAM if on GPU, else current process RSS."""
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return int(psutil.Process().memory_info().rss)


def _param_bytes(model: torch.nn.Module) -> int:
    """Sum the on-device bytes of all parameters and buffers.

    For a bitsandbytes 4-bit model the quantized weights are stored as packed
    uint8, so this correctly reports roughly a quarter of the FP16 footprint
    rather than the notional element count.
    """
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()
    return total


def load_model(
    precision: Precision, cfg: BenchmarkConfig = config
) -> tuple[Any, Any, float]:
    """Load the model at the requested precision.

    Args:
        precision: ``"fp16"`` for half precision, ``"int4"`` for bitsandbytes NF4.
        cfg: Benchmark configuration.

    Returns:
        ``(model, tokenizer, load_time_seconds)``.

    Raises:
        RuntimeError: If 4-bit is requested without a CUDA device, since
            bitsandbytes NF4 kernels require a GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_cuda = torch.cuda.is_available()
    start = time.perf_counter()

    if precision == "fp16":
        # bfloat16 on GPU (wider exponent range, no loss scaling needed);
        # float32 on CPU because float16 matmul is not well supported there.
        dtype = torch.bfloat16 if has_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            device_map="auto" if has_cuda else None,
            low_cpu_mem_usage=True,
        )
        if not has_cuda:
            model = model.to("cpu")

    elif precision == "int4":
        if not has_cuda:
            raise RuntimeError(
                "4-bit bitsandbytes quantization requires a CUDA GPU. "
                "No GPU was detected. Run this section on a machine with an "
                "NVIDIA GPU (a free Colab T4 is sufficient), or use the GGUF/"
                "llama.cpp route described in NOTES.md for CPU-only quantization."
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # NF4: information-theoretically optimal for normally-distributed
            # weights, which is what trained LLM weights approximately are.
            bnb_4bit_quant_type="nf4",
            # Compute in bf16: weights are stored 4-bit but dequantized to bf16
            # for the actual matmul, which preserves most of the quality.
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Quantize the quantization constants too (~0.4 bits/param saved).
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"Unknown precision: {precision}")

    model.eval()
    load_time = time.perf_counter() - start
    logger.info(
        "Loaded %s at %s in %.1fs (device=%s)",
        cfg.model_id,
        precision,
        load_time,
        next(model.parameters()).device,
    )
    return model, tokenizer, load_time


@torch.inference_mode()
def _measure_ttft(model: Any, inputs: dict[str, torch.Tensor]) -> float:
    """Time the prefill pass, i.e. time to first token.

    Generating exactly one token isolates prefill cost from decode cost.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(**inputs, max_new_tokens=1, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


@torch.inference_mode()
def run_prompt(
    model: Any,
    tokenizer: Any,
    prompt_spec: dict[str, str],
    cfg: BenchmarkConfig = config,
) -> PromptResult:
    """Run one prompt and measure latency, TTFT and throughput.

    Args:
        model: The loaded model.
        tokenizer: Matching tokenizer.
        prompt_spec: One entry from :data:`PROMPTS`.
        cfg: Benchmark configuration.

    Returns:
        A :class:`PromptResult` with measured timings and the generated text.
    """
    messages = [{"role": "user", "content": prompt_spec["text"]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    ttft = _measure_ttft(model, inputs)

    # Repeat the full generation and take the median wall-clock time.
    durations: list[float] = []
    output_ids = None
    for _ in range(cfg.repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    total_s = median(durations)
    generated_ids = output_ids[0][prompt_tokens:]
    generated_tokens = int(generated_ids.shape[-1])
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Throughput is measured over the decode phase only (total minus prefill),
    # which is the number that matters for streaming latency.
    decode_s = max(total_s - ttft, 1e-6)
    tokens_per_s = generated_tokens / decode_s

    logger.info(
        "%s: %d tok in %.2fs (%.1f tok/s, TTFT %.3fs)",
        prompt_spec["id"],
        generated_tokens,
        total_s,
        tokens_per_s,
        ttft,
    )

    return PromptResult(
        prompt_id=prompt_spec["id"],
        category=prompt_spec["category"],
        prompt=prompt_spec["text"],
        output=output_text,
        ttft_s=round(ttft, 4),
        total_s=round(total_s, 4),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        tokens_per_s=round(tokens_per_s, 2),
    )


def benchmark_variant(
    precision: Precision, cfg: BenchmarkConfig = config
) -> VariantResult:
    """Load one precision variant and run all five prompts against it.

    Args:
        precision: ``"fp16"`` or ``"int4"``.
        cfg: Benchmark configuration.

    Returns:
        A :class:`VariantResult` holding every measurement for this variant.
    """
    _reset_memory_stats()
    model, tokenizer, load_time = load_model(precision, cfg)

    # Warm-up: the first generate() pays kernel autotune / CUDA graph costs
    # that would otherwise be charged to whichever variant ran first.
    warm_inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **warm_inputs, max_new_tokens=cfg.warmup_tokens, do_sample=False
        )

    result = VariantResult(
        precision=precision,
        model_id=cfg.model_id,
        load_time_s=round(load_time, 2),
        param_bytes=_param_bytes(model),
        peak_memory_bytes=0,
        device=str(next(model.parameters()).device),
        dtype=str(next(model.parameters()).dtype),
    )

    for prompt_spec in PROMPTS:
        result.prompts.append(run_prompt(model, tokenizer, prompt_spec, cfg))

    result.peak_memory_bytes = _peak_memory_bytes()

    del model, tokenizer
    _reset_memory_stats()
    return result
