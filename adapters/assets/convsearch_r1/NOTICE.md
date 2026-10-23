# ConvSearch-R1 attribution and adaptation

This directory includes the runtime `INSTRUCTION` string from ConvSearch-R1,
without changing its wording or whitespace, and the complete upstream Apache
License, Version 2.0. The prompt is extracted from the Python string literal;
source-code continuation backslashes are not literal prompt characters.

Source repository: https://github.com/BeastyZ/ConvSearch-R1

Fixed revision: `e7854c89a79c904f331b306d36edde6e9ed12484`

Source file: `src/infer/infer.py`, instruction lines 18–28, tagged extraction
lines 80–90, and Q/A serialization lines 150–163.

Source: https://github.com/BeastyZ/ConvSearch-R1/blob/e7854c89a79c904f331b306d36edde6e9ed12484/src/infer/infer.py

License: https://github.com/BeastyZ/ConvSearch-R1/blob/e7854c89a79c904f331b306d36edde6e9ed12484/LICENSE

Authors credited by the repository: Changtai Zhu, Siyin Wang, Ruijun Feng,
Kai Song, and Xipeng Qiu. The upstream inference file itself is marked
`SPDX-License-Identifier: Apache-2.0` and credits adaptation of a vLLM example.
This adapter does not copy that vLLM execution framework.

ContextKV modifications: a pure function accepts only caller-supplied complete
prior actual user/assistant pairs, with no dataset/reference-answer loader.
Another pure function retains the official `<think>` and `<rewrite>` extraction
and adds explicit completion, single-pair, tag-order and nonempty-output checks.
Raw output and failures are retained; failed parsing never synthesizes a rewrite.
These runtime checks are not a source-grounding mechanism. The rewrite is an
`unverified_rewrite` baseline and may repeat false statements from actual history.

The model weights, training pipeline, GRPO framework and retrieval evaluation are
not bundled or reproduced. A candidate author's checkpoint and fixed revision are
listed in `protocol.json` for future explicit execution. No model is downloaded
or called by this module; code licensing here does not assert a separate weight
license or claim measured retrieval/answer-quality improvement.
