"""Generate `outputs/example_runs.md` using the REAL pipeline.

This is a thin wrapper around `run_examples.py` kept here so every results
directory in the repo exposes the same `generate_results.py` entrypoint.

It uses real embeddings (HuggingFace `all-MiniLM-L6-v2` by default) and real
generation (`qwen2.5:1.5b` via Ollama). Nothing is stubbed or simulated.

Prerequisites:
    pip install -r ../requirements.txt
    ollama serve
    ollama pull qwen2.5:1.5b
    # If EMBEDDING_BACKEND=ollama:  ollama pull nomic-embed-text

Run:
    python outputs/generate_results.py
    # or, equivalently, from the section root:
    python run_examples.py --rebuild
"""

from __future__ import annotations

import sys
from pathlib import Path

SECTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SECTION_ROOT))


def main() -> int:
    """Rebuild the index and regenerate the example-runs report."""
    # Imported after sys.path setup so the `src` package resolves.
    from run_examples import main as run_examples_main

    # `--rebuild` guarantees the vector store matches the current corpus and
    # the currently selected embedding backend.
    sys.argv = ["run_examples.py", "--rebuild"]
    return run_examples_main()


if __name__ == "__main__":
    sys.exit(main())
