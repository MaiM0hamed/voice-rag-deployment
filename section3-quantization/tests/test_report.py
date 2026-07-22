"""Unit tests for the reporting layer.

These verify the *arithmetic and formatting* of the comparison table using
synthetic `VariantResult` objects. They deliberately do NOT produce or validate
benchmark numbers -- real measurements come from running
`results/generate_results.py` on hardware.

torch/transformers are stubbed so these tests run anywhere.

Run:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import types

import pytest


def _stub_heavy_imports() -> None:
    """Insert minimal stubs so `src.benchmark` imports without torch."""
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        nn_stub = types.ModuleType("nn")
        nn_stub.Module = object  # type: ignore[attr-defined]
        torch_stub.nn = nn_stub  # type: ignore[attr-defined]
        torch_stub.Tensor = object  # type: ignore[attr-defined]
        torch_stub.inference_mode = lambda f=None: (f if f else (lambda g: g))  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_stub
    if "transformers" not in sys.modules:
        tf_stub = types.ModuleType("transformers")
        tf_stub.AutoModelForCausalLM = object  # type: ignore[attr-defined]
        tf_stub.AutoTokenizer = object  # type: ignore[attr-defined]
        tf_stub.BitsAndBytesConfig = object  # type: ignore[attr-defined]
        sys.modules["transformers"] = tf_stub
    if "psutil" not in sys.modules:
        sys.modules["psutil"] = types.ModuleType("psutil")


_stub_heavy_imports()

from src.benchmark import PromptResult, VariantResult  # noqa: E402
from src.report import build_comparison_table, build_quality_section  # noqa: E402


def _make_variant(
    precision: str,
    param_bytes: int,
    peak_bytes: int,
    tokens_per_s: float,
    ttft: float,
) -> VariantResult:
    """Build a synthetic variant result for formatting tests."""
    variant = VariantResult(
        precision=precision,  # type: ignore[arg-type]
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        load_time_s=10.0,
        param_bytes=param_bytes,
        peak_memory_bytes=peak_bytes,
        device="cuda:0",
        dtype="torch.bfloat16" if precision == "fp16" else "torch.uint8",
    )
    for index in range(5):
        variant.prompts.append(
            PromptResult(
                prompt_id=f"P{index + 1}",
                category="test",
                prompt="prompt",
                output="output",
                ttft_s=ttft,
                total_s=2.0,
                prompt_tokens=20,
                generated_tokens=128,
                tokens_per_s=tokens_per_s,
            )
        )
    return variant


@pytest.fixture
def fp16() -> VariantResult:
    return _make_variant("fp16", 3_090_000_000, 3_600_000_000, 42.0, 0.120)


@pytest.fixture
def int4() -> VariantResult:
    return _make_variant("int4", 980_000_000, 1_250_000_000, 55.0, 0.095)


def test_param_mb_conversion(fp16: VariantResult) -> None:
    assert fp16.param_mb == pytest.approx(3_090_000_000 / 1024**2, rel=1e-6)


def test_mean_throughput(fp16: VariantResult) -> None:
    assert fp16.mean_tokens_per_s == pytest.approx(42.0)


def test_median_ttft(int4: VariantResult) -> None:
    assert int4.median_ttft_s == pytest.approx(0.095)


def test_table_reports_memory_reduction(
    fp16: VariantResult, int4: VariantResult
) -> None:
    table = build_comparison_table({"fp16": fp16, "int4": int4})
    assert "Parameter memory (MB)" in table
    # 3.09 GB / 0.98 GB ~= 3.15x
    assert "3.15x smaller" in table


def test_table_handles_missing_variant(fp16: VariantResult) -> None:
    """CPU-only machines skip 4-bit; the table must say so, not invent numbers."""
    table = build_comparison_table({"fp16": fp16})
    assert "NOT RUN" in table
    assert "Only one variant ran" in table


def test_table_never_divides_by_zero() -> None:
    """A zero-throughput variant must not crash report generation."""
    zero = _make_variant("int4", 1, 1, 0.0, 0.0)
    other = _make_variant("fp16", 1, 1, 0.0, 0.0)
    table = build_comparison_table({"fp16": other, "int4": zero})
    assert "n/a" in table


def _variant_with_outputs(precision: str, outputs: list[str]) -> VariantResult:
    """Build a variant carrying the real prompt ids and given outputs."""
    variant = VariantResult(
        precision=precision,  # type: ignore[arg-type]
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        load_time_s=1.0,
        param_bytes=1,
        peak_memory_bytes=1,
        device="cuda:0",
        dtype="torch.bfloat16",
    )
    ids = ["P1_factual", "P2_reasoning", "P3_code", "P4_summarize", "P5_structured"]
    for pid, out in zip(ids, outputs):
        variant.prompts.append(
            PromptResult(pid, "cat", "prompt", out, 0.1, 1.0, 10, 20, 20.0)
        )
    return variant


def test_quality_section_has_objective_signals_no_placeholder() -> None:
    """Quality section must be self-contained: real signals, no fill-in TODO."""
    fp16 = _variant_with_outputs(
        "fp16",
        [
            "A Raspberry Pi is a small computer.",
            "7 * 45 = 315, so change is 185 EGP.",
            "def fibonacci(n):\n    ...",
            "A one-sentence summary.",
            '{"name": "Pico", "price_egp": 350, "in_stock": true}',
        ],
    )
    int4 = _variant_with_outputs(
        "int4",
        [
            "A Raspberry Pi is a small computer.",  # identical -> divergence identical
            "The change is 200 EGP.",  # wrong -> INT4 fail on the 185 check
            "def fib(n): pass",  # no `def fibonacci` -> INT4 fail
            "A slightly different summary.",
            "not json at all",  # -> INT4 fail on JSON check
        ],
    )
    section = build_quality_section({"fp16": fp16, "int4": int4})

    assert "fill in" not in section.lower()
    assert "TODO" not in section
    assert "Quality signals" in section
    assert "Output divergence (INT4 vs FP16): **identical**" in section  # P1
    assert "Objective check" in section
    assert "FP16=pass, INT4=fail" in section  # holds for P2, P3 and P5
