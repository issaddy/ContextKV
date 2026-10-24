"""Batch rendering must preserve complete template semantics and native token IDs."""
from dataclasses import replace
import random

import pytest

from controller.candidates import build_candidates
from controller.evidence import make_evidence
from controller.schema import ExecutionLedger, Message, SessionSnapshot, TurnRecord
from controller.templates import ByteTemplate, HuggingFaceTemplate


def snapshot_with_history():
    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    evidence = (make_evidence("Required full body", "doc"),)
    first = build_candidates(snapshot, "q1", evidence, template=template)[0]
    output = template.tokenizer.encode("answer")
    ledger = ExecutionLedger(1, "epoch", 1, first.messages+(Message("assistant", "answer"),),
        first.input_ids, output, first.visible_blocks, template.fingerprint.template_hash,
        template.fingerprint.model_revision, snapshot.corpus_snapshot_id, first.evidence_spans, first.reference_map)
    return replace(snapshot, version=1, raw_history_version=1, execution=ledger, archive=evidence,
        raw_history=(TurnRecord(1, "q1", "answer", output),), reference_map=first.reference_map), evidence


def test_candidate_batch_matches_scalar_only_adapter_without_dropping_candidates():
    reference = ByteTemplate()

    class ScalarOnly:
        fingerprint = reference.fingerprint
        render = reference.render

    class Batched(ScalarOnly):
        calls = []
        def render_many(self, messages):
            self.calls.append(messages)
            return tuple(self.render(m) for m in messages)

    snapshot, evidence = snapshot_with_history()
    scalar = build_candidates(snapshot, "q2", evidence, template=ScalarOnly())
    batcher = Batched()
    batched = build_candidates(snapshot, "q2", evidence, template=batcher)
    assert len(batcher.calls) == 1 and len(batcher.calls[0]) == 2
    assert [c.action for c in batched] == ["APPEND", "REBUILD"]
    assert tuple(replace(c, build_ms=0.) for c in scalar) == tuple(replace(c, build_ms=0.) for c in batched)


@pytest.fixture
def qwen(qwen3_tokenizer_template):
    return qwen3_tokenizer_template

def test_special_boundary_batches_match_whole_native_encoding_for_unicode_and_literals(qwen):
    pieces = ("e\u0301", "é", "中文", "👩🏽‍💻", "\r\n", "\t  ", "\u200d", "\u00a0",
              "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>",
              "<|im_sta", "rt|>", "<|fim_prefix|>", "'re", "12345", "\n\n", "<>|/", "a")
    rng = random.Random(42)
    conversations = []
    for index in range(50):
        payload = "".join(rng.choices(pieces, k=30))
        # Include the same long committed assistant in both candidates, literal
        # registered tokens inside payloads, and a missing explicit system role.
        common = (Message("system", "共同配置 e\u0301"), Message("user", payload),
                  Message("assistant", "Immutable reply " + payload))
        conversations.append(common + (Message("user", payload[::-1]),))
        if index % 5 == 0:
            conversations.append(common[1:])
    expected = tuple(qwen.render(messages) for messages in conversations)
    actual = qwen.render_many(tuple(conversations))
    assert actual == expected  # exact rendered bytes, IDs and token hashes
    for messages, result in zip(conversations, actual):
        ids = qwen.tokenizer.apply_chat_template([{"role": m.role, "content": m.content} for m in messages],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        assert result.input_ids == tuple(ids)
    assert qwen.render_many(()) == ()


def test_whole_prompt_batch_and_render_override_keep_scalar_semantics(qwen):
    conversations = ((Message("user", "e\u0301 <|im_start|> 中文"),),
                     (Message("system", "system"), Message("user", "second question")))
    expected = tuple(qwen.render(messages) for messages in conversations)
    assert qwen.render_many(conversations) == expected

    class CustomRender(HuggingFaceTemplate):
        def render(self, messages):
            return "custom", messages

    # A preexisting custom adapter that overrides only render still owns its
    # rendering semantics; no tokenizer is needed for this compatibility path.
    custom = object.__new__(CustomRender)
    assert custom.render_many(conversations) == tuple(custom.render(messages) for messages in conversations)


def test_changed_added_tokens_keep_whole_prompt_semantics(qwen, monkeypatch):
    from tokenizers import AddedToken
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: qwen.tokenizer)
    conversations = ((Message("user", "X<|im_start|>Y"),),)
    qwen.tokenizer.split_special_tokens = True
    split = HuggingFaceTemplate("local", "fixed-revision", local_files_only=True)
    assert split.fingerprint.tokenizer_hash == qwen.fingerprint.tokenizer_hash
    assert split.render_many(conversations) == tuple(split.render(m) for m in conversations)

    # Literal special-token text can belong to a longer AddedToken. Whole-prompt
    # batching must preserve the same extraction as scalar encoding.
    qwen.tokenizer.split_special_tokens = False
    qwen.tokenizer.backend_tokenizer.add_tokens([AddedToken("X<|im_start|>Y", normalized=False)])
    changed = HuggingFaceTemplate("local", "fixed-revision", local_files_only=True)
    assert changed.render_many(conversations) == tuple(changed.render(m) for m in conversations)
