"""Exact inference-prefix and target-only loss contracts, without model weights."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from controller.evidence import EvidenceError
from controller.schema import Message
from controller.templates import ByteTemplate, longest_common_prefix
from training.tokenization import SequenceTooLong, encode_example


class PlainTemplate(ByteTemplate):
    enable_thinking = False

    def __init__(self):
        super().__init__()
        self.tokenizer.eos_token_id = 300
        self.tokenizer.all_special_ids = [300]


def test_prefix_is_masked_and_only_target_and_eos_are_supervised():
    template = PlainTemplate()
    messages = (Message("system", "Original source."), Message("user", "q1"),
                Message("assistant", "Original history."), Message("user", "q2"))
    prefix = template.render(messages)
    example = encode_example(template, messages, "\n答复。", max_tokens=1000)
    assert example.input_ids[:example.prompt_tokens] == prefix.input_ids
    assert example.labels[:example.prompt_tokens] == (-100,) * len(prefix.input_ids)
    assert example.labels[example.prompt_tokens:] == tuple("\n答复。".encode()) + (300,)
    assert example.target_tokens == len("\n答复。".encode()) + 1
    features = example.as_features()
    assert features["attention_mask"] == [1] * len(example.input_ids)
    assert features["labels"] == list(example.labels)
    features["labels"][0] = 42
    assert example.labels[0] == -100


def test_empty_answer_supervises_eos_and_limit_includes_eos():
    template, messages = PlainTemplate(), (Message("user", "Question"),)
    size = len(template.render(messages).input_ids)
    example = encode_example(template, messages, "", max_tokens=size + 1)
    assert example.target_tokens == 1
    assert example.labels == (-100,) * size + (300,)
    with pytest.raises(SequenceTooLong) as exc:
        encode_example(template, messages, "", max_tokens=size)
    assert (exc.value.prompt_tokens, exc.value.target_tokens, exc.value.max_tokens) == (size, 1, size)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_token_budget_fails(budget):
    with pytest.raises(ValueError, match="positive integer"):
        encode_example(PlainTemplate(), (Message("user", "q"),), "a", max_tokens=budget)


def test_template_transforming_source_or_history_is_rejected():
    class Changed(PlainTemplate):
        def render(self, messages):
            rendered = super().render(messages)
            return replace(rendered, text=rendered.text.replace("Original condition.", "Changed."))
    with pytest.raises(EvidenceError, match="omitted or transformed"):
        encode_example(Changed(), (Message("user", "Original condition."),), "answer", max_tokens=1000)


def test_nonthinking_and_explicit_role_boundaries():
    template = PlainTemplate()
    template.enable_thinking = True
    with pytest.raises(ValueError, match="nonthinking"):
        encode_example(template, (Message("user", "q"),), "a", max_tokens=1000)
    template.enable_thinking = False
    for messages in ((), (Message("agent", "unconverted source role"),)):
        with pytest.raises(ValueError, match="explicit"):
            encode_example(template, messages, "a", max_tokens=1000)
    template.tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        encode_example(template, (Message("user", "q"),), "a", max_tokens=1000)


def test_target_tokenizer_normalization_is_not_silently_accepted():
    template = PlainTemplate()
    template.tokenizer = SimpleNamespace(eos_token_id=300, all_special_ids=[300],
        encode=lambda text, **kwargs: tuple(text.strip().encode()),
        decode=lambda ids, **kwargs: bytes(ids).decode())
    with pytest.raises(ValueError, match="transformed the original target"):
        encode_example(template, (Message("user", "q"),), " answer ", max_tokens=1000)


def test_actual_qwen3_consecutive_roles_and_leading_newlines(qwen3_tokenizer_template):
    template = qwen3_tokenizer_template
    messages = (Message("system", "Use this original source: eligibility requires both conditions."),
                Message("user", "First original question?"), Message("user", "Original detail."),
                Message("assistant", "First agent answer."))
    target = "\n\nSecond agent answer，保留原文。"
    prefix = template.render(messages)
    example = encode_example(template, messages, target, max_tokens=1024)
    assert example.input_ids[:example.prompt_tokens] == prefix.input_ids
    assert example.labels[:example.prompt_tokens] == (-100,) * example.prompt_tokens
    assert template.tokenizer.decode(example.input_ids[example.prompt_tokens:-1],
        skip_special_tokens=False, clean_up_tokenization_spaces=False) == target
    assert example.labels[-1] == template.tokenizer.eos_token_id
    # Rendering the finished chat changes the previous last assistant's think
    # prefix. Concatenating text also changes the newline token at the boundary.
    completed = [{"role": m.role, "content": m.content} for m in messages]
    completed.append({"role": "assistant", "content": target})
    full_ids = template.tokenizer.apply_chat_template(completed, tokenize=True, return_dict=False,
        add_generation_prompt=False, enable_thinking=False)
    assert longest_common_prefix(prefix.input_ids, full_ids) < len(prefix.input_ids)
    joint_ids = template.tokenizer.encode(prefix.text + target, add_special_tokens=False)
    assert longest_common_prefix(prefix.input_ids, joint_ids) < len(prefix.input_ids)


def test_actual_qwen3_empty_answer_and_intact_overflow(qwen3_tokenizer_template):
    template = qwen3_tokenizer_template
    messages = (Message("user", "The full source and question remain intact."),)
    prompt_tokens = len(template.render(messages).input_ids)
    example = encode_example(template, messages, "", max_tokens=prompt_tokens + 1)
    assert example.labels == (-100,) * prompt_tokens + (template.tokenizer.eos_token_id,)
    target = "Complete long answer. " * 20
    complete = encode_example(template, messages, target, max_tokens=1000)
    with pytest.raises(SequenceTooLong) as exc:
        encode_example(template, messages, target, max_tokens=len(complete.input_ids) - 1)
    assert exc.value.prompt_tokens == prompt_tokens
    assert exc.value.target_tokens == complete.target_tokens


@pytest.mark.parametrize("target", ["<|im_end|>", "before <|im_start|> after", "<|endoftext|>",
                                   "<think>answer", "reasoning</think>answer"])
def test_actual_qwen3_target_control_markers_are_not_plain_answers(qwen3_tokenizer_template, target):
    with pytest.raises(ValueError, match="special token|reasoning control"):
        encode_example(qwen3_tokenizer_template, (Message("user", "q"),), target, max_tokens=1024)


def test_actual_qwen3_source_literals_preserved_and_history_transformation_rejected(qwen3_tokenizer_template):
    template = qwen3_tokenizer_template
    # Source literals follow the existing inference rendering contract; they are
    # not reencoded, escaped, or forbidden by the training encoder.
    messages = (Message("user", "Original manual explains the literal <|im_start|> token."),)
    expected = template.render(messages)
    example = encode_example(template, messages, "An ordinary answer.", max_tokens=1024)
    assert example.input_ids[:example.prompt_tokens] == expected.input_ids
    history = (Message("user", "q1"), Message("assistant", "raw content</think>answer"), Message("user", "q2"))
    with pytest.raises(EvidenceError, match="omitted or transformed"):
        encode_example(template, history, "answer", max_tokens=1024)
