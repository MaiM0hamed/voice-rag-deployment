"""Run the full FP16 vs 4-bit benchmark and write the comparison report.

This is the single entrypoint for Section 3. It:
  1. detects the hardware,
  2. benchmarks the model at FP16/BF16,
  3. benchmarks it at 4-bit NF4 (bitsandbytes),
  4. writes `comparison_report.md` and `raw_measurements.json` here.

If no CUDA GPU is present, the 4-bit variant is skipped with an explicit reason
recorded in the report -- bitsandbytes NF4 kernels are GPU-only. The FP16 run
still completes so the section produces useful output on CPU-only machines.

Prerequisites:
    pip install -r ../requirements.txt

Run (from the section root or this directory):
    python results/generate_results.py

Runtime: roughly 5-15 minutes on a free Colab T4, longer on CPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SECTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SECTION_ROOT))

from src.benchmark import VariantResult, benchmark_variant, describe_environment  # noqa: E402
from src.config import config  # noqa: E402
from src.report import write_reports  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("generate_results")


def main() -> int:
    """Benchmark both precisions and write the reports."""
    parser = argparse.ArgumentParser(description="Run the quantization benchmark.")
    parser.add_argument(
        "--skip-fp16", action="store_true", help="Skip the full-precision run."
    )
    parser.add_argument(
        "--skip-int4", action="store_true", help="Skip the 4-bit run."
    )
    args = parser.parse_args()

    environment = describe_environment()
    logger.info("Environment: %s", environment)

    if not environment["cuda_available"]:
        logger.warning(
            "No CUDA GPU detected. FP16 will run on CPU (slowly) and 4-bit "
            "bitsandbytes will be skipped -- its kernels require a GPU."
        )

    variants: dict[str, VariantResult] = {}
    skipped: dict[str, str] = {}

    if args.skip_fp16:
        skipped["fp16"] = "skipped via --skip-fp16"
    else:
        logger.info("=== Benchmarking FP16/BF16 ===")
        try:
            variants["fp16"] = benchmark_variant("fp16", config)
        except Exception as exc:
            logger.exception("FP16 benchmark failed")
            skipped["fp16"] = f"failed: {exc}"

    if args.skip_int4:
        skipped["int4"] = "skipped via --skip-int4"
    else:
        logger.info("=== Benchmarking 4-bit NF4 ===")
        try:
            variants["int4"] = benchmark_variant("int4", config)
        except RuntimeError as exc:
            # Expected on CPU-only machines; recorded, not fabricated.
            logger.warning("4-bit benchmark skipped: %s", exc)
            skipped["int4"] = str(exc)
        except Exception as exc:
            logger.exception("4-bit benchmark failed")
            skipped["int4"] = f"failed: {exc}"

    if not variants:
        logger.error("No variants completed; nothing to report.")
        return 1

    json_path, md_path = write_reports(
        variants, environment, config.results_dir, skipped
    )
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)

    for name, variant in variants.items():
        logger.info(
            "%s: %.1f MB params, %.1f tok/s mean, %.3fs median TTFT",
            name,
            variant.param_mb,
            variant.mean_tokens_per_s,
            variant.median_ttft_s,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
