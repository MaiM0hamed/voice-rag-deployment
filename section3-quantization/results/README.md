# results/ — GENERATED ARTIFACTS

Everything in this directory except `generate_results.py` and this README is
**produced by running the benchmark on your hardware**. No numbers in this repo
are estimated, interpolated, or copied from documentation.

## How to generate

```bash
cd section3-quantization

# 1. Install torch for YOUR hardware first
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only

# 2. Install the rest
pip install -r requirements.txt

# 3. Run the benchmark
python results/generate_results.py
```

First run downloads Qwen2.5-1.5B-Instruct (~3 GB) from HuggingFace into
`~/.cache/huggingface`. Later runs are offline.

Runtime: ~5–15 min on a free Colab T4; considerably longer on CPU.

## Reproducing on Google Colab (free T4 GPU)

The 4-bit NF4 variant requires a CUDA GPU (bitsandbytes kernels are GPU-only). A
free Colab T4 runs **both** variants and produces the full comparison. In a Colab
notebook (set *Runtime → Change runtime type → T4 GPU*):

```python
!nvidia-smi                                   # confirm a GPU is attached

# Get the code (clone your repo, or upload + unzip the section3-quantization folder)
!git clone <YOUR_REPO_URL>
%cd <repo>/section3-quantization

# Install a CUDA build of torch, then the rest of the deps
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r requirements.txt

# Run BOTH variants (FP16/BF16 and 4-bit NF4) on the same GPU
!python results/generate_results.py

# Inspect and download the real artifacts
from IPython.display import Markdown, display
display(Markdown(open('results/comparison_report.md').read()))
from google.colab import files
files.download('results/comparison_report.md')
files.download('results/raw_measurements.json')
```

Everything is measured on the T4; nothing is estimated. On a CPU-only machine the
FP16 run still completes and the 4-bit run is skipped with the reason recorded.

## What gets produced

| File | Contents |
|---|---|
| `comparison_report.md` | Environment, variant status, the precision/size/speed/quality table, and side-by-side outputs for all 5 prompts |
| `raw_measurements.json` | Every per-prompt timing, token count and generated string |

## GPU vs CPU

`bitsandbytes` NF4 kernels **require a CUDA GPU**. Behaviour by machine:

- **With a GPU:** both FP16/BF16 and 4-bit run; the table is fully populated.
- **CPU only:** FP16 runs (in float32, since CPU float16 matmul is poorly
  supported); the 4-bit run is **skipped with the reason recorded** in the
  report's "Variant status" section. It is never filled in with a guess.

For CPU-only quantization, use the GGUF/llama.cpp route described in `NOTES.md`.

## Reading the numbers

- **TTFT** is measured with a `max_new_tokens=1` generate call, isolating
  prefill from decode.
- **Throughput** is computed over the decode phase only (total − prefill).
- Each prompt is run `REPEATS` times (default 3) and the **median** reported.
- A warm-up generation runs before timing so kernel-autotune costs are not
  charged to whichever variant loads first.
- Expect 4-bit to be **smaller always**, but **faster only on GPU** — on CPU the
  dequantization overhead usually makes it slower. The report captures whichever
  is true on your machine.

## Optional: run only one variant

```bash
python results/generate_results.py --skip-int4   # FP16 only
python results/generate_results.py --skip-fp16   # 4-bit only
```
