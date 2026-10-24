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


@pytest.mark.parametrize("settings", [{"revision": ""}, {"max_input_tokens": 32768}, {"timeout_seconds": 0}])
def test_invalid_selector_contract_is_rejected_at_config_boundary(settings):
    values = {"base_url": "http://local", "model": "model", "revision": "rev", **settings}
    with pytest.raises(ValueError):
        SelectorConfig(**values)
