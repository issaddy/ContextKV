"""Opt-in verification against the pinned Qwen tokenizer, without loading weights."""
import os
from dataclasses import replace

import pytest

from controller.candidates import build_candidates
from controller.evidence import EvidencePolicy, make_evidence
from controller.schema import ContextEpoch, SessionSnapshot
from controller.templates import HuggingFaceTemplate, longest_common_prefix


@pytest.mark.skipif(os.environ.get("CONTEXTKV_TEST_HF_TOKENIZER") != "1",
                    reason="opt-in pinned tokenizer check; run with tokenizer extra and cached HF files")
def test_pinned_qwen_template_and_epoch_independent_input():
    template = HuggingFaceTemplate("Qwen/Qwen2.5-7B-Instruct",
                                  "a09a35458c702b33eeacc393d103063234e8bc28", local_files_only=True)
    snapshot = SessionSnapshot("first")
    evidence = (make_evidence("请求超时为30秒；连接超时为5秒。", "timeouts"),)
    first = build_candidates(snapshot, "配置超时是多少？", evidence, template=template)[0]
    second = build_candidates(replace(snapshot, session_id="second",
                         current_epoch=ContextEpoch("second", 99, "random-engine-id", 7)),
                         "配置超时是多少？", evidence, template=template)[0]
    assert first.feasible and first.required_blocks <= first.visible_blocks
    assert any(token > 255 for token in first.input_ids)
    assert first.input_ids == second.input_ids and first.input_hash == second.input_hash
    native_ids = template.tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in first.messages],
        tokenize=True, return_dict=False, add_generation_prompt=True,
        enable_thinking=template.enable_thinking)
    assert first.input_ids == tuple(native_ids)
    assert "random-engine-id" not in template.render(first.messages).text
    measured = build_candidates(snapshot, "配置超时是多少？", evidence, template=template,
                                measure_evidence_redundancy=True)[0]
    assert measured.input_ids == first.input_ids and measured.input_hash == first.input_hash
    assert measured.evidence_body_tokens == len(template.tokenizer.encode(
        evidence[0].text, add_special_tokens=False))
    assert measured.nonrequired_evidence_body_tokens == 0
    for layout in ("stable_key", "contextpilot"):
        first = build_candidates(snapshot, "配置超时是多少？", evidence,
                                 EvidencePolicy(layout=layout), template)[0]
        second = build_candidates(snapshot, "如何修改连接超时？", evidence,
                                  EvidencePolicy(layout=layout), template)[0]
        prefix = longest_common_prefix(first.input_ids, second.input_ids)
        decoded_prefix = template.tokenizer.decode(first.input_ids[:prefix])
        assert evidence[0].text in decoded_prefix
        assert "Question:" in decoded_prefix
