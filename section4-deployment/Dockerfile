# syntax=docker/dockerfile:1
#
# Multi-stage build for the Electro Pi inference API.
#
# Stage 1 installs dependencies into a virtualenv; stage 2 copies only that
# venv and the application code. This keeps build toolchains out of the final
# image and makes rebuilds fast when only source changes.
#
# Build:  docker build -t electropi-inference .
# Run:    docker run -p 8000:8000 electropi-inference
#
# GPU:    docker run --gpus all -e LOAD_IN_4BIT=true -p 8000:8000 electropi-inference
#         (requires the NVIDIA Container Toolkit on the host)

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed by some wheels; it stays in this stage only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# CPU-only torch by default: the default PyPI wheel drags in ~4 GB of CUDA
# libraries that are dead weight in a CPU container. For a GPU deployment,
# build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX}" torch \
    && pip install -r requirements.txt

# --------------------------------------------------------------- runtime ---
FROM python:3.11-slim AS runtime

# curl is used by HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/

USER appuser

EXPOSE 8000

# start-period is generous: the first run downloads ~3 GB of weights, and we
# do not want the orchestrator killing the container mid-download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single worker: the model occupies most of the accelerator, and N workers
# would load N copies of the weights. Scale with replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
