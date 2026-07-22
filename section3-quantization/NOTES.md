# Section 3 — Write-up

## What this builds

A benchmark harness comparing **Qwen2.5-1.5B-Instruct** at full precision
(FP16/BF16) against **4-bit NF4 via bitsandbytes**, measuring memory,
throughput, latency and output quality on five fixed prompts.

Run `python results/generate_results.py` to produce
`results/comparison_report.md` and `results/raw_measurements.json`.

**No numbers are committed to this repo.** The assessment scores "real
measurements, not guesses", and a plausible-looking table I invented would be
exactly the wrong artifact. Everything is generated on your hardware.

## Measurement methodology

Naive benchmarking produces misleading results, so:

- **Warm-up before timing.** The first `generate()` pays kernel autotune and
  CUDA graph capture costs. Without a warm-up these land on whichever variant
  runs first, typically making FP16 look artificially slow.
- **TTFT isolated from decode.** Time-to-first-token is measured with a
  `max_new_tokens=1` call, so prefill cost is separated from per-token decode.
  Throughput is then computed over `total − prefill`, which is the number that
  actually governs streaming latency.
- **Median of N repeats** (default 3) rather than a single sample, to blunt
  scheduler noise.
- **`torch.cuda.synchronize()` around every timed region.** CUDA kernels are
  asynchronous; timing without synchronisation measures launch overhead, not
  execution.
- **Greedy decoding** (`do_sample=False`) so any output difference between
  precisions is attributable to quantization, not sampling randomness.
- **Real parameter bytes** via `numel() * element_size()` summed over
  parameters and buffers, so a packed 4-bit tensor reports its true footprint
  rather than a notional element count.
- **Environment captured** in the report (GPU model, VRAM, torch/CUDA version).
  Quantization results are meaningless without it: 4-bit is usually *faster*
  on GPU and *slower* on CPU.

## The five prompts

Chosen to probe different failure modes rather than five variations of the same
task: factual recall (quantization erases rare facts first), arithmetic
reasoning (sensitive to small logit perturbations), code generation (syntax is
brittle), summarization (instruction following), and structured JSON output
(validity degrades visibly at low bit-widths). Identical for both variants.

## bitsandbytes vs GPTQ vs AWQ vs GGUF

### bitsandbytes (used here)

Zero-shot, **calibration-free** quantization applied at load time. NF4 assumes
weights are roughly normally distributed — true for trained transformers — and
uses an information-theoretically optimal 4-bit grid for that distribution.
Double quantization compresses the quantization constants themselves for
another ~0.4 bits/param.

- **Strengths:** one config object, no calibration dataset, no offline step,
  works with any HuggingFace checkpoint immediately. Supports QLoRA training on
  top of the quantized weights.
- **Weaknesses:** inference is not the fastest, because weights are dequantized
  to bf16 on the fly for each matmul. GPU-only.
- **Pick it when:** iterating quickly, fine-tuning with QLoRA, or serving a
  model whose weights change often. It is the right default for development and
  the right answer for "I need this to fit on this GPU by this afternoon."

### GPTQ

**Post-training, calibration-based.** Uses second-order (approximate Hessian)
information from a calibration set to choose quantization points that minimise
layer-wise output error, quantizing weights column-by-column with error
compensation.

- **Strengths:** better quality than naive round-to-nearest at the same bit
  width, and fast fused inference kernels (ExLlama v2 and similar).
- **Weaknesses:** requires a calibration pass (minutes to hours) and a
  calibration set representative of production traffic. Quality is
  calibration-sensitive: an off-distribution set produces a model that looks
  fine on benchmarks and degrades on real inputs.
- **Pick it when:** a stable model is served at high volume on GPU and the
  one-off calibration cost amortises across millions of requests. Classic
  high-throughput GPU serving choice.

### AWQ (Activation-aware Weight Quantization)

Also calibration-based, but the insight differs: rather than treating all
weights equally, AWQ identifies the **salient weight channels** — those
multiplied by large-magnitude activations — and protects them by scaling before
quantization. A small fraction of weights disproportionately affects output.

- **Strengths:** typically better quality retention than GPTQ at 4-bit,
  especially for instruction-tuned models, and very fast kernels. Less
  sensitive to calibration set choice than GPTQ, because activation magnitude
  is a more stable signal than Hessian estimates.
- **Weaknesses:** still an offline step; narrower architecture support.
- **Pick it when:** serving 4-bit on GPU in production and quality matters more
  than the last few percent of throughput. In practice my default recommendation
  over GPTQ for instruction-tuned chat models.

### GGUF (llama.cpp)

A **file format plus runtime**, not just a quantization algorithm. Supports many
schemes (Q4_K_M, Q5_K_M, Q8_0, …) with k-quants that vary bit-width per tensor
according to sensitivity.

- **Strengths:** the only serious option for **CPU and Apple Silicon**
  inference. Memory-maps weights, so a model can start serving before it is
  fully resident. Runs on consumer hardware with no CUDA. Excellent
  quality/size curve at Q4_K_M and above. Single-file distribution.
- **Weaknesses:** not integrated with the HuggingFace training stack; a separate
  conversion step; less suited to high-concurrency GPU batch serving, where
  vLLM-style continuous batching over AWQ/GPTQ wins.
- **Pick it when:** deploying on-device, on CPU-only servers, on Apple Silicon,
  or shipping to end users who cannot be assumed to have a GPU. Also the
  pragmatic choice for edge deployments where a single binary matters.

### Summary heuristic

| Situation | Choice |
|---|---|
| Rapid iteration, QLoRA fine-tuning | bitsandbytes |
| High-throughput GPU serving, stable model | AWQ (GPTQ if AWQ lacks support) |
| Maximum quality retention at 4-bit on GPU | AWQ |
| CPU, Apple Silicon, on-device, edge | GGUF |
| Model weights change frequently | bitsandbytes (no calibration step) |

The deciding questions in practice: *is there a GPU?* (no → GGUF), *do the
weights change often?* (yes → bitsandbytes), *is this high-volume steady-state
serving?* (yes → AWQ/GPTQ and pay the calibration cost once).

## Assumptions, shortcuts, limitations (stated honestly)

- **Verified locally:** the reporting arithmetic, ratio formatting, missing-
  variant handling and divide-by-zero guards (6/6 pytest in `tests/`, which stub
  torch so they run anywhere).
- **Requires your hardware:** all actual measurements. The development sandbox
  had neither a GPU nor network access to model weights, so **no benchmark
  numbers are included** — by design, not omission.
- **4-bit requires CUDA.** bitsandbytes NF4 kernels are GPU-only. On a CPU-only
  machine the harness runs FP16 and records an explicit skip reason for 4-bit.
  For CPU quantization use the GGUF route above.
- **FP16 on CPU actually runs float32**, because CPU float16 matmul support is
  poor and would misrepresent full-precision speed. Recorded in the report's
  `dtype` field.
- **Qualitative assessment mixes objective signals with a human read.** The
  report emits both outputs side by side plus **auto-computed objective signals**:
  the INT4-vs-FP16 output divergence (character similarity) and pass/fail on a
  crisp check for the prompts that have one (arithmetic = 185 EGP, code defines
  `fibonacci`, structured output parses as JSON). Factual recall and
  summarization have no clean automatic oracle, so nuanced accuracy there is left
  to a two-minute human read rather than a fabricated quality score.
- **Single model, single size.** Quantization damage grows as models shrink; a
  1.5B model degrades more visibly at 4-bit than a 70B one. Conclusions here
  should not be extrapolated upward without re-measuring.
