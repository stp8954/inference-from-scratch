# inference-from-scratch

Companion code for the series in [stp8954/Inference-engineer](https://github.com/stp8954/Inference-engineer).

| Blog | Folder and current scope |
| --- | --- |
| 1: The Life of a Token | [week-01-life-of-a-token/](week-01-life-of-a-token/) — CPU generation loop; prefill/decode explanation |

Blog 1 uses Qwen2.5-0.5B-Instruct with float32, CPU by default or explicitly selected CUDA, and caching disabled. It makes no speed or hardware benchmark claims. See its README for setup and validation status.

## Later measurement contract

The existing series plan names `meta-llama/Llama-3.1-8B-Instruct` at BF16 on one H100 SXM 80GB. Those are later measurement targets, not Blog 1 dependencies or completed results. A measured claim requires the exact checkpoint, hardware, pinned environment and checked-in harness with raw evidence. No H100 result is supplied here.
