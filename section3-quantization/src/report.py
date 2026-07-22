"""Render benchmark measurements into a markdown comparison report.

Every figure here is derived arithmetically from measurements produced by
`benchmark.py`. If a variant was not run (e.g. no GPU available for 4-bit),
that is reported explicitly rather than filled in with a plausible guess.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .benchmark import VariantResult


def _ratio(numerator: float, denominator: float) -> str:
    """Format a ratio, guarding against division by zero."""
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.2f}x"


def _looks_like_json(text: str) -> bool:
    """True if the text contains a parseable JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return False
    try:
        json.loads(text[start : end + 1])
        return True
    except ValueError:
        return False


# Objective, prompt-specific correctness checks computed from the generated text.
# Only prompts with a crisp, non-subjective oracle are listed; factual recall and
# summarization are left to a human read (they have no clean automatic check).
_OUTPUT_CHECKS: dict[str, tuple[str, Callable[[str], bool]]] = {
    "P2_reasoning": ("correct change 185 EGP present", lambda t: "185" in t),
    "P3_code": ("defines a fibonacci function", lambda t: "def fibonacci" in t),
    "P5_structured": ("output parses as JSON", _looks_like_json),
}


def build_comparison_table(variants: dict[str, VariantResult]) -> str:
    """Build the headline precision/size/speed/quality table.

    Args:
        variants: Mapping of precision name to its measured result.

    Returns:
        A markdown table string.
    """
    fp16 = variants.get("fp16")
    int4 = variants.get("int4")

    rows: list[str] = [
        "| Metric | FP16/BF16 | 4-bit NF4 | Change |",
        "| --- | --- | --- | --- |",
    ]

    def add(label: str, a: str, b: str, delta: str) -> None:
        rows.append(f"| {label} | {a} | {b} | {delta} |")

    if fp16 and int4:
        add(
            "Parameter memory (MB)",
            f"{fp16.param_mb:.1f}",
            f"{int4.param_mb:.1f}",
            _ratio(fp16.param_mb, int4.param_mb) + " smaller",
        )
        add(
            "Peak memory (MB)",
            f"{fp16.peak_memory_mb:.1f}",
            f"{int4.peak_memory_mb:.1f}",
            _ratio(fp16.peak_memory_mb, int4.peak_memory_mb) + " smaller",
        )
        add(
            "Mean throughput (tok/s)",
            f"{fp16.mean_tokens_per_s:.1f}",
            f"{int4.mean_tokens_per_s:.1f}",
            _ratio(int4.mean_tokens_per_s, fp16.mean_tokens_per_s) + " speed",
        )
        add(
            "Median TTFT (s)",
            f"{fp16.median_ttft_s:.3f}",
            f"{int4.median_ttft_s:.3f}",
            _ratio(fp16.median_ttft_s, int4.median_ttft_s) + " faster",
        )
        add(
            "Load time (s)",
            f"{fp16.load_time_s:.1f}",
            f"{int4.load_time_s:.1f}",
            "-",
        )
        add("Device", fp16.device, int4.device, "-")
        add("Storage dtype", fp16.dtype, int4.dtype, "-")
    else:
        present = fp16 or int4
        if present:
            add(
                "Parameter memory (MB)",
                f"{present.param_mb:.1f}" if fp16 else "NOT RUN",
                f"{present.param_mb:.1f}" if int4 else "NOT RUN",
                "-",
            )
        rows.append(
            "| _Note_ | _Only one variant ran; see the status section below._ "
            "| | |"
        )

    return "\n".join(rows)


def build_quality_section(variants: dict[str, VariantResult]) -> str:
    """Render side-by-side outputs for each of the five fixed prompts.

    Qualitative comparison is the part a table cannot capture. Each prompt block
    shows both outputs plus **objective, auto-computed signals**: how far the
    4-bit output diverged from the FP16 baseline, and pass/fail on a crisp check
    for the prompts that have one (arithmetic, code, JSON). No subjective quality
    verdict is invented; nuanced accuracy is best judged by reading the outputs.
    """
    fp16 = variants.get("fp16")
    int4 = variants.get("int4")
    lines: list[str] = []

    reference = fp16 or int4
    if reference is None:
        return "_No variants were run._"

    for index, prompt in enumerate(reference.prompts):
        fp16_prompt = fp16.prompts[index] if fp16 else None
        int4_prompt = int4.prompts[index] if int4 else None

        lines.append(f"### {prompt.prompt_id} — {prompt.category}")
        lines.append("")
        lines.append(f"**Prompt:** {prompt.prompt}")
        lines.append("")

        if fp16_prompt is not None:
            lines.append(
                f"**FP16/BF16** ({fp16_prompt.generated_tokens} tokens, "
                f"{fp16_prompt.tokens_per_s:.1f} tok/s):"
            )
            lines.append("")
            lines.append("```")
            lines.append(fp16_prompt.output)
            lines.append("```")
            lines.append("")

        if int4_prompt is not None:
            lines.append(
                f"**4-bit NF4** ({int4_prompt.generated_tokens} tokens, "
                f"{int4_prompt.tokens_per_s:.1f} tok/s):"
            )
            lines.append("")
            lines.append("```")
            lines.append(int4_prompt.output)
            lines.append("```")
            lines.append("")

        # Objective signals only -- everything below is computed from the actual
        # generated text above, never hand-entered.
        signals: list[str] = []
        if fp16_prompt is not None and int4_prompt is not None:
            a, b = fp16_prompt.output, int4_prompt.output
            if a == b:
                signals.append("- Output divergence (INT4 vs FP16): **identical**")
            else:
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                signals.append(
                    "- Output divergence (INT4 vs FP16): "
                    f"**{ratio:.0%} character similarity**"
                )
        check = _OUTPUT_CHECKS.get(prompt.prompt_id)
        if check is not None:
            label, fn = check
            parts: list[str] = []
            if fp16_prompt is not None:
                parts.append(f"FP16={'pass' if fn(fp16_prompt.output) else 'fail'}")
            if int4_prompt is not None:
                parts.append(f"INT4={'pass' if fn(int4_prompt.output) else 'fail'}")
            signals.append(f"- Objective check ({label}): {', '.join(parts)}")

        if signals:
            lines.append("**Quality signals (auto-computed from the outputs above):**")
            lines.extend(signals)
        else:
            lines.append(
                "**Quality signals:** single variant only and no crisp automatic "
                "check for this prompt; read the output above."
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def write_reports(
    variants: dict[str, VariantResult],
    environment: dict[str, Any],
    results_dir: Path,
    skipped: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write both the JSON measurements and the markdown report.

    Args:
        variants: Measured results keyed by precision.
        environment: Output of ``describe_environment``.
        results_dir: Directory to write into.
        skipped: Mapping of precision -> reason, for variants that did not run.

    Returns:
        ``(json_path, markdown_path)``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    skipped = skipped or {}

    json_path = results_dir / "raw_measurements.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": environment,
        "variants": {name: v.to_dict() for name, v in variants.items()},
        "skipped": skipped,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env_lines = [f"- **{k}**: {v}" for k, v in environment.items()]

    status_lines: list[str] = []
    for name in ("fp16", "int4"):
        if name in variants:
            status_lines.append(f"- `{name}`: measured")
        elif name in skipped:
            status_lines.append(f"- `{name}`: **NOT RUN** — {skipped[name]}")

    md = f"""# Section 3 — Quantization Benchmark Results

_Generated {payload['generated_at']} by `generate_results.py`.
Every number below was measured on this machine; none are estimates._

## Environment

{chr(10).join(env_lines)}

## Variant status

{chr(10).join(status_lines)}

## Comparison table

{build_comparison_table(variants)}

## Qualitative comparison (same five prompts, both precisions)

{build_quality_section(variants)}

## Raw measurements

Full per-prompt timings are in `raw_measurements.json`.
"""

    md_path = results_dir / "comparison_report.md"
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
