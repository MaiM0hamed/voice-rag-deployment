"""Configuration for the quantization benchmark.

All settings are environment-overridable so the same code runs on a laptop CPU,
a Colab T4, or a workstation GPU without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SECTION_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime settings for the FP16 vs 4-bit comparison."""

    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    results_dir: Path = field(default_factory=lambda: SECTION_DIR / "results")

    # Generation settings. Greedy decoding (do_sample=False) so that any output
    # difference between FP16 and 4-bit is attributable to quantization rather
    # than sampling noise.
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
    do_sample: bool = os.getenv("DO_SAMPLE", "false").lower() == "true"
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Number of timed repetitions per prompt; the median is reported to reduce
    # the effect of one-off scheduler hiccups.
    repeats: int = int(os.getenv("REPEATS", "3"))

    # Tokens generated during warm-up (excluded from measurements). The first
    # forward pass pays CUDA-graph/kernel-autotune costs that would otherwise
    # be misattributed to the quantized model.
    warmup_tokens: int = int(os.getenv("WARMUP_TOKENS", "16"))


config = BenchmarkConfig()


# The five fixed prompts. IDENTICAL for both precisions -- this is the core
# requirement of the task, since qualitative comparison is only meaningful on
# matched inputs. They are chosen to probe different failure modes of
# aggressive quantization:
#   1. factual recall        -- quantization can erase rare facts
#   2. arithmetic reasoning  -- sensitive to small logit perturbations
#   3. code generation       -- syntax is brittle under precision loss
#   4. summarization         -- tests instruction following
#   5. structured output     -- JSON validity degrades noticeably at low bits
PROMPTS: list[dict[str, str]] = [
    {
        "id": "P1_factual",
        "category": "Factual recall",
        "text": "In two sentences, explain what a Raspberry Pi is and what it is typically used for.",
    },
    {
        "id": "P2_reasoning",
        "category": "Arithmetic reasoning",
        "text": (
            "A shop sells resistor packs for 45 EGP each. A customer buys 7 packs "
            "and pays with a 500 EGP note. How much change do they receive? "
            "Show your working."
        ),
    },
    {
        "id": "P3_code",
        "category": "Code generation",
        "text": (
            "Write a Python function called `fibonacci` that returns the first n "
            "Fibonacci numbers as a list. Include a docstring."
        ),
    },
    {
        "id": "P4_summarize",
        "category": "Summarization",
        "text": (
            "Summarize the following in one sentence: 'Quantization reduces the "
            "numerical precision of model weights, lowering memory use and often "
            "increasing throughput, at the cost of some accuracy. Different "
            "methods trade off calibration effort against runtime performance.'"
        ),
    },
    {
        "id": "P5_structured",
        "category": "Structured output",
        "text": (
            "Return ONLY valid JSON (no prose) describing a product with these "
            'keys: "name" (string), "price_egp" (number), "in_stock" (boolean). '
            "Use a microcontroller board as the product."
        ),
    },
]
