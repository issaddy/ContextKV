"""Chat rendering and exact original-text token offsets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...] | None = None


class ByteTokenizer:
    def encode(self, text, **kwargs):
        return tuple(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")

    def __call__(self, texts, *, return_offsets_mapping=False, **kwargs):
        single = isinstance(texts, str)
        values = (texts,) if single else texts
        ids = [self.encode(text) for text in values]
        result = {"input_ids": ids[0] if single else ids}
        if return_offsets_mapping:
            offsets = [tuple((index, index+1) for index, char in enumerate(text)
                             for _ in char.encode("utf-8")) for text in values]
            result["offset_mapping"] = offsets[0] if single else offsets
        return result


class ByteTemplate:
    def __init__(self):
        self.tokenizer = ByteTokenizer()

    def render(self, messages, *, with_offsets=False):
        text = "".join(f"<|{message.role}|>\n{message.content}\n<|end|>\n" for message in messages)
        text += "<|assistant|>\n"
        encoded = self.tokenizer(text, return_offsets_mapping=with_offsets)
        return RenderedPrompt(text, tuple(encoded["input_ids"]), encoded.get("offset_mapping"))


class HuggingFaceTemplate:
    def __init__(self, model, revision=None, *, enable_thinking=False, local_files_only=True):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        self.enable_thinking = enable_thinking

    def render(self, messages, *, with_offsets=False):
        text = self.tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False,
                                 return_token_type_ids=False, return_offsets_mapping=with_offsets)
        offsets = tuple(encoded["offset_mapping"]) if with_offsets else None
        return RenderedPrompt(text, tuple(encoded["input_ids"]), offsets)
