<!-- Historical plan preserved on 2026-10-22. Superseded for Blog 1 by README.md; this is not an implementation or run record. -->

# Week 1 — Life of a token

Post: [drafts/week-01-life-of-a-token.md](https://github.com/stp8954/Inference-engineer/blob/main/drafts/week-01-life-of-a-token.md)

This folder is the code that post points at. Week 1 does **not** claim measured decode speed on the H100. It ships a loop you can run anywhere and a calculator for the ceiling the post derives.

## Pins

| Name | Value |
|---|---|
| Series model | `meta-llama/Llama-3.1-8B-Instruct` |
| Dtype | BF16 |
| Lab GPU (later weeks) | H100 SXM 80GB, 3.35 TB/s advertised HBM |
| Teaching model | `Qwen/Qwen2.5-0.5B-Instruct` — so the loop runs without 16 GB VRAM |
| Ceiling this week | `3350 GB/s / 16 GB ≈ 210 tok/s` for batch-1 dense BF16 decode on the H100 pin |

HF gated: accept the Llama 3.1 license and pass a token if you load the 8B.

## What to implement

Three small files. No server, no CUDA graphs, no `use_cache=True` in the teaching loop.

### 1. `generate.py`

The post's naive loop, copy-paste close.

- Default `MODEL` = Qwen 0.5B.
- `--model meta-llama/Llama-3.1-8B-Instruct` for the series checkpoint.
- `use_cache=False` on every step. Comment in the file: Hugging Face still forwards **all** positions; we only read the last logit.
- Chat template + optional greedy / nucleus.
- Print `n tokens in t s -> n/t tok/s` and make it obvious this number is **not** a series result (no warmup, no `torch.cuda.synchronize`, includes sampling + `torch.cat`).

Do not treat this print as the Week 4 number.

### 2. `napkin.py`

Pure arithmetic. No GPU required.

```
weight_bytes = n_params * bytes_per_param
ceiling_tok_s = bandwidth_gb_s / (weight_bytes / 1e9)
compute_s = (2 * n_params) / flop_s
weight_read_s = weight_bytes / bandwidth_B_s
ops_per_byte = flop_s / bandwidth_B_s
```

Defaults:

```
--params 8e9 --dtype bf16 --bandwidth-tb-s 3.35 --dense-tflops 989
```

Print, labeled as ceilings:

- weight GiB
- H100 batch-1 decode ceiling (~210 tok/s)
- weight-read lower bound (~4.8 ms)
- dense-compute lower bound (~16 µs)
- ridge (~295 FLOP/byte)
- intensity of this workload (~1 FLOP/byte for dense batch-1 decode)

Optional `--compare` table for other advertised bandwidths (4090 ~1.0 TB/s, M5 153 GB/s, M5 Max 614 GB/s). Label those **formula examples**, not lab pins. Note a 16 GB laptop cannot hold the 8B BF16 weights.

No measured tok/s in this file.

### 3. `requirements.txt`

Pin enough to run `generate.py`:

```
torch
transformers
accelerate
```

Do not invent CUDA/torch versions until the Week 4 image exists. When that image is pinned, add it here as a comment: `WEEK4_IMAGE=...`.

## What not to put here yet

- `bench_decode.py --cache` — that is Week 3/4 (KV + sync + warmup + H100 log).
- vLLM / SGLang / TensorRT.
- Quant, MTP, speculative decode.
- Rust.
- A Mac-only path presented as the series measurement.

If you want a placeholder:

```
# bench_decode.py  — Week 4. Same 8B, H100 SXM, warmup, synchronize, raw JSON log.
```

## How to run

```bash
python napkin.py
python generate.py
# needs ~16 GB VRAM + Llama access
python generate.py --model meta-llama/Llama-3.1-8B-Instruct
```

## Done when

- [ ] `napkin.py` reproduces 16 GB, ~210 tok/s, ~4.8 ms, ~16 µs, ~295 ops/byte without a GPU
- [ ] `generate.py` runs the 0.5B loop on some device
- [ ] README states the 8B + H100 pin and that printed tok/s is not the lab result
- [ ] no file in this folder claims an H100 measurement

Week 4 is when this repo grows a harness and a results JSON from the rented H100.
