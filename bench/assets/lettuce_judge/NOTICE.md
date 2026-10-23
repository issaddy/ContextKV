# LettuceDetect adaptation

The English QA prompt is copied verbatim from KRLabsOrg/LettuceDetect,
commit `fbce1991f26e210cc49ace3913562ab4f2f31a60`,
`lettucedetect/prompts/qa_prompt_en.txt`. The formatting, pair-tokenization,
two-label argmax and consecutive-token span logic in `bench/lettuce_judge.py`
are small adaptations of the same revision's `detectors/prompt_utils.py`,
`detectors/transformer.py` and `datasets/hallucination_dataset.py`.

Copyright (c) 2025 KR Labs. The upstream MIT license is reproduced in LICENSE.

ContextKV changes: local-only pinned loader; complete single input with no
truncation/chunk aggregation; explicit omissions/failures; answer-side
sequence IDs and direct original-character offsets, preserving leading
whitespace; audit binding and citation-overlap diagnostics. This is not a full
reproduction of upstream evaluation or a replacement for quality acceptance.

The separately obtained `KRLabsOrg/lettucedect-v2-mmbert-base` model at revision
`8831ce063dff760b01efe4519b4188f5ee2456f2` declares Apache-2.0. Model weights are
not bundled. Loading requires an operator-provided local snapshot; the
adapter does not download or repeatedly hash weights.
