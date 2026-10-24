"""Target-only SFT encoding with the exact inference prefix.

Rendering a completed chat can change earlier Qwen3 assistant messages, and
tokenizing prompt + target text can merge the boundary newline tokens. Keep the
already rendered inference IDs and encode the original completion separately.
"""
from __future__ import annotations

from dataclasses import dataclass

from controller.schema import Message
from controller.templates import HuggingFaceTemplate, verify_rendered_evidence


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int

    @property
    def target_tokens(self) -> int:
        """Supervised token count, including the final EOS."""
        return len(self.input_ids) - self.prompt_tokens

    def as_features(self) -> dict[str, list[int]]:
        """Supply explicit labels to a collator that preserves supplied labels."""
        return {"input_ids": list(self.input_ids), "labels": list(self.labels),
                "attention_mask": [1] * len(self.input_ids)}


class SequenceTooLong(ValueError):
    """An intact example exceeds the caller's budget; no tokens were removed."""

    def __init__(self, prompt_tokens: int, target_tokens: int, max_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        super().__init__(f"complete sequence needs {prompt_tokens + target_tokens} tokens "
                         f"(prompt={prompt_tokens}, target_with_eos={target_tokens}), "
                         f"exceeding max_tokens={max_tokens}")


def encode_example(template: HuggingFaceTemplate, messages: tuple[Message, ...],
                   target: str, *, max_tokens: int) -> EncodedExample:
    """Mask the full source/history/prompt; supervise only the target and EOS.

    No roles are merged or invented and no text is stripped. An empty target
    explicitly trains an immediate EOS; dataset selection owns whether to use it.
    The caller supplies the already selected, causally valid message prefix.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if template.enable_thinking:
        raise ValueError("training encoding requires the nonthinking inference template")
    if not messages or any(m.role not in {"system", "user", "assistant"} for m in messages):
        raise ValueError("a nonempty prefix with explicit system/user/assistant roles is required")
    tokenizer = template.tokenizer
    if tokenizer.eos_token_id is None:
        raise ValueError("the inference tokenizer must define EOS")
    prefix = template.render(messages)
    verify_rendered_evidence(messages, (), prefix.text)
    target_ids = tuple(tokenizer.encode(target, add_special_tokens=False))
    if set(target_ids).intersection(tokenizer.all_special_ids):
        raise ValueError("target contains a reserved special token instead of plain answer text")
    if "<think>" in target or "</think>" in target:
        raise ValueError("target contains reasoning control markers in a nonthinking task")
    if tokenizer.decode(target_ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != target:
        raise ValueError("tokenizer transformed the original target text")
    completion = target_ids + (tokenizer.eos_token_id,)
    prompt_tokens = len(prefix.input_ids)
    if prompt_tokens + len(completion) > max_tokens:
        raise SequenceTooLong(prompt_tokens, len(completion), max_tokens)
    return EncodedExample(prefix.input_ids + completion,
                          (-100,) * prompt_tokens + completion, prompt_tokens)
