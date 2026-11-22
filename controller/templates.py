"""Chat rendering and exact original-text token offsets."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


class ThreadTokenizer:
    """Share one vocabulary while serializing calls that use its mutable backend."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._lock = Lock()

    def __getattr__(self, name):
        value = getattr(self._tokenizer, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            with self._lock:
                return value(*args, **kwargs)

        return call

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._tokenizer(*args, **kwargs)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...] | None = None


class HuggingFaceTemplate:
    def __init__(self, model, revision=None, *, enable_thinking=False, local_files_only=True):
        from transformers import AutoTokenizer
        self.tokenizer = ThreadTokenizer(AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=local_files_only))
        self.enable_thinking = enable_thinking

    def render(self, messages, *, with_offsets=False):
        text = self.tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False,
                                 return_token_type_ids=False, return_offsets_mapping=with_offsets)
        offsets = tuple(encoded["offset_mapping"]) if with_offsets else None
        return RenderedPrompt(text, tuple(encoded["input_ids"]), offsets)
