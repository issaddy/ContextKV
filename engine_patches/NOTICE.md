# SGLang source attribution

The patch context derives
from [sgl-project/sglang](https://github.com/sgl-project/sglang), commit
`297e54fd105e974cdf401e5b4d20253a22c16f90`. The upstream license is reproduced in
`SGLANG_LICENSE`. Source identity and target paths are recorded in
`integrations/sglang-0.5.23/manifest.json`.

The InfoFlow implementation follows [InfoFlow KV](https://arxiv.org/abs/2603.05353):
independent chunk KV, inference-consistent global RoPE, query attention-mass
selection and sparse causal replay. The author's [Qwen implementation](https://github.com/tx467/InfoFlow-KV/tree/main/llm/models/qwen)
provided the algorithm reference. ContextKV implements these operations over
SGLang's Qwen weights and shared native GPU/Host pages.

Sparse attention split-K partials use SGLang's `_fwd_kernel_stage2` log-sum-exp merger
from `sglang/kernels/ops/attention/decode_attention.py`. Independent block
prefill uses PyTorch's optimized `causal_lower_right` SDPA dispatch.

Segmented decoding calls SGLang's `_decode_grouped_att_m_fwd` directly. It applies
inverse block RoPE to queries, and merges native segment/split outputs with the
same log-sum-exp identity. The packing, query rotation and segmented merger are
ContextKV kernels; shared KV pages remain immutable. Sparse replay continues to
use the global-position attention path and the upstream split-K merger.
