import json

import httpx
import pytest

from adapters.query_selector import OpenAICompletionClient, SelectorConfig


class Tokenizer:
    chat_template = "fixed test template"
    def __init__(self, count=12):
        self.count, self.calls = count, []
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        # Regression: current Transformers returns BatchEncoding by default.
        assert kwargs["return_dict"] is False and "truncation" not in kwargs
        return list(range(self.count))


def config(**overrides):
    return SelectorConfig("http://local.test/v1", "test-model", "fixed-revision", **overrides)


def test_complete_preserves_raw_usage_and_counts_actual_token_ids():
    calls = []
    raw = {"model": "served-alias", "choices": [{"message": {"content": '{"status":"unchanged"}'}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 12, "completion_tokens": 7}}
    def respond(request):
        calls.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=raw)
    client = OpenAICompletionClient(config(), tokenizer=Tokenizer(), transport=httpx.MockTransport(respond))
    result = client.complete([{"role": "user", "content": "literal"}], response_format={"type": "json_object"})
    assert result.status == "completed" and result.input_tokens == 12
    assert result.raw_response == raw and result.usage == raw["usage"]
    assert calls[0]["temperature"] == 0 and calls[0]["max_tokens"] == 256
    assert "top_p" not in calls[0] and client.manifest["top_p"] is None
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["chat_template_kwargs"] == client.manifest["chat_template_kwargs"] == {"enable_thinking": False}
    assert client.tokenizer.calls[0][1]["enable_thinking"] is False
    assert client.manifest["http_calls"] == 1
    client.close()


def test_over_budget_does_not_call_http_or_truncate():
    def forbidden(_):
        raise AssertionError("must not call model")
    tokenizer = Tokenizer(count=30)
    client = OpenAICompletionClient(config(max_input_tokens=20), tokenizer=tokenizer,
                                    transport=httpx.MockTransport(forbidden))
    result = client.complete([{"role": "user", "content": "complete input"}])
    assert result.status == "input_over_budget" and result.input_tokens == 30
    assert client.calls == 0 and tokenizer.calls[0][0][0]["content"] == "complete input"
    client.close()


@pytest.mark.parametrize("response", [httpx.Response(400, text="unsupported grammar"),
    httpx.Response(200, json={"choices": []}), httpx.Response(200, text="not-json")])
def test_failed_http_or_schema_retains_response_without_retry(response):
    client = OpenAICompletionClient(config(), tokenizer=Tokenizer(), transport=httpx.MockTransport(lambda _: response))
    result = client.complete([])
    assert result.status == "failed" and result.raw_response == response.text
    assert client.calls == 1
    client.close()


def test_timeout_is_recorded_and_not_retried():
    def timeout(_):
        raise httpx.ReadTimeout("deliberate local timeout")
    client = OpenAICompletionClient(config(timeout_seconds=2), tokenizer=Tokenizer(), transport=httpx.MockTransport(timeout))
    result = client.complete([])
    assert result.status == "failed" and "ReadTimeout" in result.error and client.calls == 1
    client.close()


def test_author_sampling_and_empty_template_kwargs_match_count_http_manifest():
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "literal"}, "finish_reason": "stop"}]})
    tokenizer = Tokenizer()
    client = OpenAICompletionClient(config(temperature=.7, top_p=1., max_new_tokens=4096,
        chat_template_kwargs={}), tokenizer=tokenizer, transport=httpx.MockTransport(respond))
    assert client.complete([{"role": "user", "content": "full QA"}]).status == "completed"
    assert sent[0]["temperature"] == .7 and sent[0]["top_p"] == 1 and sent[0]["max_tokens"] == 4096
    assert sent[0]["chat_template_kwargs"] == client.manifest["chat_template_kwargs"] == {}
    assert "enable_thinking" not in tokenizer.calls[0][1]
    assert "response_format" not in sent[0] and "stop" not in sent[0]
    client.close()


def test_legacy_config_missing_sampling_fields_preserves_selector_defaults():
    legacy = SelectorConfig(**{"base_url": "http://local", "model": "model", "revision": "rev"})
    assert legacy.temperature == 0 and legacy.top_p is None
    assert legacy.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize("settings", [{"revision": ""}, {"max_input_tokens": 32768}, {"timeout_seconds": 0},
    {"temperature": -1}, {"temperature": float("nan")}, {"top_p": 0}, {"top_p": 1.1},
    {"chat_template_kwargs": {"truncation": True}}, {"chat_template_kwargs": {"return_dict": True}}])
def test_invalid_selector_contract_is_rejected_at_config_boundary(settings):
    values = {"base_url": "http://local", "model": "model", "revision": "rev", **settings}
    with pytest.raises(ValueError):
        SelectorConfig(**values)
