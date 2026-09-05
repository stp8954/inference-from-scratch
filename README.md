# inference-from-scratch

Companion code for the series in [stp8954/Inference-engineer](https://github.com/stp8954/Inference-engineer).

**Locked contract**

- Model: `meta-llama/Llama-3.1-8B-Instruct` at BF16
- Lab GPU: one cloud H100 SXM 80GB
- "I measured" = that checkpoint + that SKU + a pinned image + a checked-in harness

Week 1 is first principles and spec-sheet ceilings. Do not publish H100 tok/s until the Week 4 harness exists.

| Week | Folder |
|---|---|
| 1 | [`week-01-life-of-a-token/`](week-01-life-of-a-token/) — naive loop + napkin ceilings |
