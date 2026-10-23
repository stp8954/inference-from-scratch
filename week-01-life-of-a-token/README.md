# Blog 1 — The Life of a Token

`generate.py` is the exact Python listing from the October 21 revised article candidate, “The Life of a Token.” It makes the autoregressive loop visible: score the sequence, select the highest-scoring next token, append it, and repeat.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Device and precision: CPU by default, float32. Use `--device cuda` for an explicit CUDA run.
- One chat-formatted prompt; greedy decoding; at most 32 new tokens, stopping earlier on a configured end token.
- `use_cache=False`: every call processes the entire growing sequence, and the loop reads only the last position's logits. This illustrates token dependencies; it is not efficient cached decode.
- Only generated text is printed. There is no timing or benchmark result.

## Setup and run

Use Python 3.10 or newer in a fresh environment. The first execution downloads model files from Hugging Face and caches them locally; internet access, disk space and RAM are required. CPU mode needs no GPU or paid service. CUDA mode requires a working PyTorch CUDA installation and an available GPU.

From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade -r week-01-life-of-a-token/requirements.txt
.\.venv\Scripts\python.exe week-01-life-of-a-token/generate.py
```

For an explicit CUDA run:

```powershell
.\.venv\Scripts\python.exe week-01-life-of-a-token/generate.py --device cuda
```

CUDA mode checks that CUDA is available and that both model parameters and input tensors are on CUDA; it does not silently fall back to CPU.

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe` after creating the environment.

The requirements list direct dependencies without claiming a complete environment lock. The listing uses the Transformers `dtype` argument. [Transformers installation guidance](https://huggingface.co/docs/transformers/installation) and [model loading documentation](https://huggingface.co/docs/transformers/main_classes/model) describe installation and loading options.

## Smoke test — October 22, 2026

- Syntax parsing passed; source matches the revised article's Python block exactly after newline normalization.
- The exact listing completed a real model load with exit code 0 and returned `The capital of France is Paris.` This is an observed example output, not a benchmark or a promise of identical output under every package/model revision.
- CPU smoke test environment: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 5.20.2, NVIDIA RTX 2000 Ada Generation. The downloaded model snapshot was `7ae557604adf67be50417f59c2c2f167def9a775`.
- A separate CUDA smoke test is recorded in the Chief of Staff review; it is a device validation, not a performance measurement.
- The script's output does not report generated-token count or whether EOS ended the loop. A clean exit confirms generation completed, but does not establish which stopping condition ended the loop. Do not claim EOS stopping was demonstrated.
- This is a smoke test of loading and generation, not a performance measurement.

## Observed CUDA token counts — October 23, 2026

One explicit CUDA run on an NVIDIA RTX 2000 Ada Generation completed on CUDA (`cuda:0`) with float32. The chat-formatted prompt contained 36 tokens. Generation produced 8 tokens including the EOS token (ID `151645`) and stopped because EOS was selected, before the 32-token cap. The displayed text was `The capital of France is Paris.` The input sequence lengths before the eight forward calls were 36 through 43 tokens. Raw machine-readable evidence, including package versions, model snapshot revision, source SHA-256 and UTC run time, is in [results/2026-10-23-cuda.json](results/2026-10-23-cuda.json).

These are token-count observations for this one prompt, model snapshot and software environment. They are not timing, throughput or benchmark results. In the JSON, `generated_token_count_includes_eos: true` is a counting convention: count every generated ID, including EOS if emitted. It does not say whether EOS occurred in a particular run; `stop_reason` records whether that run stopped on EOS or the token cap. Decoded display text skips special tokens.

To reproduce, install the listed direct requirements in an environment with CUDA-enabled PyTorch, then run:

```powershell
python week-01-life-of-a-token/generate.py --device cuda --report week-01-life-of-a-token/results/run.json
```

The captured run used Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 5.20.2, and model revision `7ae557604adf67be50417f59c2c2f167def9a775`.

The article candidate is currently a local draft in the Chief of Staff workspace (`media/drafts/week-01-life-of-a-token-v2.md`); its public publication link is pending. The older draft in Inference-engineer has a different listing and is not the source for this file.

## Scope retained for later

The previous implementation plan is preserved in [legacy-plan.md](legacy-plan.md). Its napkin calculator, model-selection flags, sampling, timing and H100 targets are historical/future plans, not features of this script. Cached generation and performance comparisons are separate follow-up work.
