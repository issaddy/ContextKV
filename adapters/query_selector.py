"""Small synchronous OpenAI-compatible completion transport for query adapters.

Runs inside the retriever worker. It counts the complete fixed-tokenizer input
before HTTP and never truncates history, source windows, or model output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import httpx

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


@dataclass(frozen=True)
class SelectorConfig:
    base_url: str
    model: str
    revision: str
    tokenizer_path: str | None = None
    tokenizer_revision: str | None = None
    context_window: int = 32768
    max_input_tokens: int = 8192
    max_new_tokens: int = 256
    timeout_seconds: float = 60.0

    def __post_init__(self):
        if not self.base_url or not self.model or not self.revision:
            raise ValueError("selector needs explicit base_url, model and revision")
        if min(self.context_window, self.max_input_tokens, self.max_new_tokens, self.timeout_seconds) <= 0:
            raise ValueError("selector budgets and timeout must be positive")
        if self.max_input_tokens + self.max_new_tokens > self.context_window:
            raise ValueError("selector input/output budgets exceed context_window")


@dataclass(frozen=True)
class CompletionResult:
    status: str
    text: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    usage: dict | None = None
    raw_response: object = None
    elapsed_ms: float = 0.0
    error: str | None = None


class OpenAICompletionClient:
    def __init__(self, config: SelectorConfig, *, tokenizer=None, transport=None):
        self.config = config
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path or config.model,
                revision=config.tokenizer_revision or config.revision, local_files_only=True, trust_remote_code=False)
        self.tokenizer = tokenizer
        if not tokenizer.chat_template:
            raise ValueError("selector tokenizer must have a chat template")
        self.client = httpx.Client(timeout=config.timeout_seconds, transport=transport)
        self.calls = 0

    @property
    def manifest(self):
        return {**asdict(self.config), "temperature": 0, "local_files_only": True,
                "model_identity": "configured_revision_not_remote_attestation",
                "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
                "chat_template": self.tokenizer.chat_template, "http_calls": self.calls,
                "input_policy": "count complete messages; no truncation"}

    def complete(self, messages, *, response_format=None):
        started = time.perf_counter()
        count = self.count_input_tokens(messages)
        if count > self.config.max_input_tokens or count + self.config.max_new_tokens > self.config.context_window:
            return CompletionResult("input_over_budget", input_tokens=count,
                                    elapsed_ms=(time.perf_counter()-started)*1000)
        body = {"model": self.config.model, "messages": messages, "temperature": 0,
                "max_tokens": self.config.max_new_tokens, "stream": False,
                "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)}
        if response_format is not None:
            body["response_format"] = response_format
        base = self.config.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        response = None
        self.calls += 1
        try:
            response = self.client.post(url, json=body)
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            text, finish = choice["message"]["content"], choice.get("finish_reason")
            if not isinstance(text, str):
                raise ValueError("completion content must be text")
            return CompletionResult("completed", text, finish, count, raw.get("usage"), raw,
                                    (time.perf_counter()-started)*1000)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            return CompletionResult("failed", input_tokens=count,
                raw_response=response.text if response is not None else None,
                elapsed_ms=(time.perf_counter()-started)*1000, error=f"{type(exc).__name__}: {exc}")

    def count_input_tokens(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, **CHAT_TEMPLATE_KWARGS)
        return len(ids)

    def close(self):
        self.client.close()
