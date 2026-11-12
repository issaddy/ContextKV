# SGLang source attribution

The patch context derives
from [sgl-project/sglang](https://github.com/sgl-project/sglang), commit
`71de97b264b04dcd514cf904003028aefe9775c8`. The upstream license is reproduced in
`SGLANG_LICENSE`. Source identity and target paths are recorded in
`integrations/sglang-0.5.18/manifest.json`.

The InfoFlow implementation follows [InfoFlow KV](https://arxiv.org/abs/2603.05353):
independent chunk KV, inference-consistent global RoPE, query attention-mass
selection and sparse causal replay. The author's [Qwen implementation](https://github.com/tx467/InfoFlow-KV/tree/main/llm/models/qwen)
provided the algorithm reference. ContextKV implements these operations over
SGLang's Qwen weights and shared native GPU/Host pages.
