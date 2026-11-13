"""OpenAI-compatible transport for the official offline benchmark judges."""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import httpx


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    model: str
    revision: str
    tokenizer_path: str | None = None
    tokenizer_revision: str | None = None
    context_window: int = 32768
    max_new_tokens: int = 256
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    top_p: float | None = None
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})


@dataclass(frozen=True)
class CompletionResult:
    status: str
    text: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    usage: dict | None = None
    elapsed_ms: float = 0.0
    error: str | None = None


class OpenAICompletionClient:
    def __init__(self, config, *, tokenizer=None, transport=None):
        self.config = config
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path or config.model,
                revision=config.tokenizer_revision or config.revision, local_files_only=True)
        self.tokenizer = tokenizer
        self.client = httpx.Client(timeout=config.timeout_seconds, transport=transport)

    @property
    def manifest(self):
        return {"model": self.config.model, "revision": self.config.revision,
                "base_url": self.config.base_url, "temperature": self.config.temperature,
                "max_tokens": self.config.max_new_tokens,
                "enable_thinking": self.config.chat_template_kwargs["enable_thinking"]}

    def complete(self, messages, *, response_format=None):
        started = time.perf_counter()
        count = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                    return_dict=False, **self.config.chat_template_kwargs))
        if count + self.config.max_new_tokens > self.config.context_window:
            return CompletionResult("input_over_budget", input_tokens=count,
                                    elapsed_ms=(time.perf_counter()-started)*1000)
        body = {"model": self.config.model, "messages": messages, "temperature": self.config.temperature,
                "max_tokens": self.config.max_new_tokens, "stream": False,
                "chat_template_kwargs": self.config.chat_template_kwargs}
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        if response_format is not None:
            body["response_format"] = response_format
        base = self.config.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        try:
            response = self.client.post(url, json=body)
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            return CompletionResult("completed", choice["message"]["content"], choice["finish_reason"],
                count, raw.get("usage"), (time.perf_counter()-started)*1000)
        except httpx.HTTPError as error:
            return CompletionResult("failed", input_tokens=count,
                                    elapsed_ms=(time.perf_counter()-started)*1000, error=str(error))

    def close(self):
        self.client.close()
