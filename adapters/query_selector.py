"""Small synchronous OpenAI-compatible completion transport for query adapters.

Runs inside the retriever worker. It counts the complete fixed-tokenizer input
before HTTP and never truncates history, source windows, or model output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from threading import Lock

import httpx

from controller.auxiliary_spend import auxiliary_call

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
    temperature: float = 0
    top_p: float | None = None
    chat_template_kwargs: dict = field(default_factory=lambda: dict(CHAT_TEMPLATE_KWARGS))

    def __post_init__(self):
        if not self.base_url or not self.model or not self.revision:
            raise ValueError("selector needs explicit base_url, model and revision")
        if min(self.context_window, self.max_input_tokens, self.max_new_tokens, self.timeout_seconds) <= 0:
            raise ValueError("selector budgets and timeout must be positive")
        if self.max_input_tokens + self.max_new_tokens > self.context_window:
            raise ValueError("selector input/output budgets exceed context_window")
        if (not math.isfinite(self.temperature) or self.temperature < 0
                or (self.top_p is not None and (not math.isfinite(self.top_p) or not 0 < self.top_p <= 1))):
            raise ValueError("invalid completion sampling parameters")
        if not isinstance(self.chat_template_kwargs, dict) or set(self.chat_template_kwargs) & {
                "tokenize", "return_dict", "add_generation_prompt", "truncation", "max_length", "padding"}:
            raise ValueError("chat_template_kwargs cannot override exact input token counting")


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
        self._input_lock = Lock()

    @property
    def manifest(self):
        return {**asdict(self.config), "local_files_only": True,
                "model_identity": "configured_revision_not_remote_attestation",
                "chat_template_kwargs": dict(self.config.chat_template_kwargs),
                "chat_template": self.tokenizer.chat_template, "http_calls": self.calls,
                "input_policy": "count complete messages; no truncation"}

    def complete(self, messages, *, response_format=None, max_new_tokens=None, chat_template_kwargs=None):
        started = time.perf_counter()
        output_budget = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        template_kwargs = self.config.chat_template_kwargs if chat_template_kwargs is None else chat_template_kwargs
        # Tokenizer state/counters are local shared resources; the thread-safe
        # HTTP client need not serialize independent sessions while awaiting LLMs.
        with self._input_lock:
            count = self.count_input_tokens(messages, chat_template_kwargs=template_kwargs)
            if count > self.config.max_input_tokens or count + output_budget > self.config.context_window:
                return CompletionResult("input_over_budget", input_tokens=count,
                                        elapsed_ms=(time.perf_counter()-started)*1000)
            self.calls += 1
        body = {"model": self.config.model, "messages": messages, "temperature": self.config.temperature,
                "max_tokens": output_budget, "stream": False,
                "chat_template_kwargs": dict(template_kwargs)}
        if self.config.top_p is not None:
            body["top_p"] = self.config.top_p
        if response_format is not None:
            body["response_format"] = response_format
        base = self.config.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        response, usage = None, None
        with auxiliary_call(self.config.model) as receipt:
            try:
                response = self.client.post(url, json=body)
                raw = response.json()
                usage = raw.get("usage")
                if usage is not None:
                    receipt.update(input_tokens=usage.get("prompt_tokens"),
                                   output_tokens=usage.get("completion_tokens"))
                response.raise_for_status()
                choice = raw["choices"][0]
                text, finish = choice["message"]["content"], choice.get("finish_reason")
                if not isinstance(text, str):
                    raise ValueError("completion content must be text")
                receipt.update(status="RETURNED", finish_reason=finish)
                if receipt["input_tokens"] is None:
                    receipt["input_tokens"] = count
                return CompletionResult("completed", text, finish, count, usage, raw,
                                        (time.perf_counter()-started)*1000)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                if response is not None:
                    receipt.update(status="RESPONSE_ERROR", http_status=response.status_code)
                return CompletionResult("failed", input_tokens=count, usage=usage,
                    raw_response=response.text if response is not None else None,
                    elapsed_ms=(time.perf_counter()-started)*1000, error=f"{type(exc).__name__}: {exc}")

    def count_input_tokens(self, messages, *, chat_template_kwargs=None):
        template_kwargs = self.config.chat_template_kwargs if chat_template_kwargs is None else chat_template_kwargs
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=False, **template_kwargs)
        return len(ids)

    def catalog_token_lengths(self, messages, documents):
        """Pack whole auxiliary records with the same tokenizer and shared lock."""
        with self._input_lock:
            base = self.count_input_tokens(messages)
            values = self.tokenizer(documents, add_special_tokens=False, return_attention_mask=False)["input_ids"] if documents else ()
            return base, [len(tokens) for tokens in values]

    def close(self):
        self.client.close()
