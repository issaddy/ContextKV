# ContextPilot source integration

Upstream: [EfficientContext/ContextPilot](https://github.com/EfficientContext/ContextPilot).
Authors: Yinsicheng Jiang, Yeqi Huang, Liang Cheng, Cheng Deng, Xuan Sun, Luo Mai.
Pinned commit: `b4c80327f6cfae2286839413731fcb6f53578b2f` (package version 0.5.0).
License: Apache License 2.0; see [license text](APACHE-2.0.txt) and the
[upstream license at the pinned commit](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/LICENSE).
Paper: [ContextPilot: Fast Long-Context Inference via Context Reuse](https://arxiv.org/abs/2511.03475).

`adapters/contextpilot.py` retains the independent committed-history ordering
baseline. Its optional `SourceOrderIndex` loads the author's unchanged CPU
context-index, ordering, metadata and live-index modules from an external
checkout prepared by `scripts/prepare_contextpilot.py`. Preparation and first
index load verify the fixed git commit and unchanged tracked source. No upstream
source is vendored here.

The loader creates package namespaces instead of running the upstream package
initializer, which imports unrelated retrieval/model pipelines. It never runs
setup, installs the upstream `.pth` inference hook, or invokes the author's
prompt construction, conversation deduplication, server, or model pipeline.
The CPU index path requires NumPy and SciPy. The upstream distance module also
attempts its optional CuPy import; the adapter always constructs `use_gpu=False`.
No ContextPilot dependency or inference hook is installed by this repository.

Ordering proposals call the author's read-only search and traversal operations;
search/traversal diagnostic counters may change, but requests, source membership
and LRU access timestamps do not. Successful hash-bound execution is registered
separately through initial single-context construction or insertion. A completed
execution that reaches its output length limit is still eligible: its actual
input already produced source KV, irrespective of answer quality. Index
matches suggest source permutations only; native engine quotes remain the
source of truth for physical prefix reuse. This integration adds no TTL policy.
