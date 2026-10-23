# Qwen3 reranker adaptation notice

`adapters/qwen3_reranker.py` adapts the formatting, prompt wrapper and yes/no
scoring example published by the Qwen Team in the official model card:

https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/fd9fb1d26c07223ced488065909faf522e29cc7d/README.md

The fixed model card identifies the code/model distribution as Apache-2.0.
The fixed model snapshot contains no separate LICENSE or NOTICE file. A copy of
the Apache License 2.0 is provided in `APACHE-2.0.txt` in this directory.

Changes in ContextKV: explicit local-only loading, device and precision;
batched scoring; exact pre-truncation length diagnostics; final-position logits
only via the Transformers `logits_to_keep=1` API; no returned KV cache; frozen
candidate-pool replay and post-hoc evaluation. The pretrained weights are not
redistributed here. The model card's benchmark results are not project results.
