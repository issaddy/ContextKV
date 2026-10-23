from __future__ import annotations

from dataclasses import asdict, replace
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from bench.judge import DIMENSIONS
from bench.native_judge import (
    DATA_POLICY, NativeJudgeClient, NativeJudgeConfig, evaluate_file, main, parse_native_result,
)
from controller.templates import token_hash


def config(**overrides):
    return NativeJudgeConfig(**({"provider": "local_openai_compatible", "model": "selene-local",
        "revision": "427792f1c3e2073cb7da216924fd884b1ba496e0", "base_url": "http://localhost/v1",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct", "base_family": "llama"} | overrides))


def audit():
    return {"task_id": "s::1", "conversation_id": "s", "query": "What is the limit?",
        "answer": "10 MB [E_abc]", "actual_history": [{"user": "Earlier?", "assistant": "A prior claim"}],
        "retrieved_evidence": [{"document_id": "now", "text": "Current source"}],
        "visible_evidence_manifest": [{"key": {"document_id": str(i), "version": "v1"},
                                      "reference_id": "E_abc" if i == 0 else f"E_{i:x}"} for i in range(12)],
        "audit_evidence": [{"key": {"document_id": str(i), "version": "v1"},
                            "text": f"Full visible body {i}"} for i in range(12)],
        "input_ids": [11, 22], "measurement_kind": "simulation", "truncated": False,
        "targets": [{"text": "FORBIDDEN_ONLINE_TARGET"}], "enrichments": {"gold": "FORBIDDEN_ENRICHMENT"}}


def response(text="**Reasoning:** Supported.\n\n**Result:** 4", *, finish="stop"):
    return httpx.Response(200, json={"id": "response-id", "model": "selene-local",
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12}})


@pytest.mark.parametrize("protocol,text,score", [
    ("selene_absolute_v1", "**Reasoning:** Correct.\n\n**Result:** 5", 1),
    ("prometheus_absolute_v1", "Incorrect. [RESULT] 1", 0),
    ("prometheus_absolute_v1", "Mixed support.\n[RESULT] 3\n", .5),
])
def test_native_parsers_use_model_specific_complete_result(protocol, text, score):
    result = parse_native_result(text, protocol)
    assert result["status"] == "scored" and result["score"] == score
    assert result["reason"]


@pytest.mark.parametrize("text", [
    "**Reasoning:** Good. **Result:** 6", "**Reasoning:** Good. **Result:** 4.5",
    "**Reasoning:** Good. **Result:** 4 more text", "**Reasoning:** **Result:** 4",
    "**Reasoning:** **Result:** 1 actually **Result:** 5", "{\"score\": 5}",
    "Feedback [RESULT] 5", "**Reasoning:** Cannot assess. **Result:** NA",
])
def test_invalid_or_ambiguous_response_is_not_coerced_to_score(text):
    with pytest.raises(ValueError):
        parse_native_result(text, "selene_absolute_v1")


def test_na_is_explicit_extension_not_invalid_text_or_zero():
    result = parse_native_result("**Reasoning:** Necessary basis unavailable. **Result:** NA",
                                 "selene_absolute_v1", na_policy="explicit_na_v1")
    assert result == {"status": "not_applicable", "native_score": None, "score": None,
                      "reason": "Necessary basis unavailable."}
    assert parse_native_result("No basis. [RESULT] NA", "prometheus_absolute_v1",
                               na_policy="explicit_na_v1")["score"] is None


@pytest.mark.asyncio
async def test_full_evidence_history_binding_and_native_metadata_survive():
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        return response()

    client = NativeJudgeClient(config(), transport=httpx.MockTransport(serve))
    try:
        row = await client.evaluate(audit())
    finally:
        await client.aclose()
    assert len(requests) == 5
    for request in requests:
        text = request["messages"][-1]["content"]
        assert "Full visible body 11" in text and "A prior claim" in text
        assert "Current source" in text and "10 MB [E_abc]" in text
        assert "FORBIDDEN" not in text and "Reference answer:" not in text
        assert "response_format" not in request and request["stream"] is False
    assert row["scores"] == dict.fromkeys(DIMENSIONS, .75)
    assert row["status"] == "success" and row["source"] == "offline_judge"
    assert row["generation_measurement_kind"] == "simulation"
    assert row["input_hash"] == token_hash([11, 22]) and row["query"] == audit()["query"]
    assert row["script_history_mismatch"] is None
    grade = row["native_results"]["citations"]
    assert grade["native_score"] == 4 and grade["raw_text"].endswith("4")
    assert grade["finish_reason"] == "stop" and grade["usage"]["completion_tokens"] == 12
    meta = row["judge"]
    assert meta["base_family"] == "llama" and meta["base_model"].startswith("meta-llama/")
    assert set(meta["protocol_manifest"]["rubrics"]) == set(DIMENSIONS)
    assert meta["protocol_manifest"]["native_templates"]["revision"] == "418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de"
    assert meta["protocol_manifest"]["data_policy"] == DATA_POLICY


@pytest.mark.asyncio
async def test_reference_argument_really_selects_reference_template_for_both_models():
    for protocol, marker in [("selene_absolute_v1", "Reference answer:"),
                             ("prometheus_absolute_v1", "###Reference Answer (Score 5):")]:
        c = config(protocol=protocol)
        client = NativeJudgeClient(c, transport=httpx.MockTransport(lambda _: response()))
        try:
            plain = client.messages(audit(), "correctness")
            referenced = client.messages(audit(), "correctness", {"targets": [{"text": "REFERENCE_ONLY"}]})
        finally:
            await client.aclose()
        assert marker not in plain[-1]["content"]
        assert marker in referenced[-1]["content"] and "REFERENCE_ONLY" in referenced[-1]["content"]
        assert "FORBIDDEN" not in referenced[-1]["content"]
        assert len(plain) == (2 if protocol.startswith("prometheus") else 1)


@pytest.mark.asyncio
async def test_partial_failure_na_and_truncation_remain_distinct():
    responses = iter([
        response("**Reasoning:** No assessment basis. **Result:** NA"),
        response(finish="length"), response("Unknown format"), response(), response(),
    ])
    client = NativeJudgeClient(config(na_policy="explicit_na_v1"),
                               transport=httpx.MockTransport(lambda _: next(responses)))
    try:
        row = await client.evaluate(audit())
    finally:
        await client.aclose()
    assert row["status"] == "failed"
    assert row["scores"] == dict(zip(DIMENSIONS, (None, None, None, .75, .75)))
    assert row["native_results"]["correctness"]["status"] == "not_applicable"
    for dimension in ("groundedness", "answerability"):
        assert row["native_results"][dimension]["status"] == "failed"
    assert row["native_results"]["groundedness"]["finish_reason"] == "length"
    assert row["native_results"]["groundedness"]["raw_text"].endswith("4")
    assert row["judge"]["protocol_manifest"]["applicability_extension"]


@pytest.mark.asyncio
async def test_valid_na_is_completed_but_missing_dimension_not_a_filled_rating():
    client = NativeJudgeClient(config(na_policy="explicit_na_v1"), transport=httpx.MockTransport(
        lambda _: response("**Reasoning:** No assessment basis. **Result:** NA")))
    try:
        row = await client.evaluate(audit())
    finally:
        await client.aclose()
    assert row["status"] == "success" and row["scores"] == dict.fromkeys(DIMENSIONS)
    assert all(g["status"] == "not_applicable" for g in row["native_results"].values())


@pytest.mark.asyncio
async def test_context_preflight_never_truncates_or_calls_http():
    def unexpected(_):
        pytest.fail("over-budget prompt must not reach the server")
    cfg = config(context_window=2048, max_tokens=128)
    with pytest.raises(ValueError, match="token_counter"):
        NativeJudgeClient(cfg)
    client = NativeJudgeClient(cfg, token_counter=lambda messages: 2000,
                               transport=httpx.MockTransport(unexpected))
    try:
        row = await client.evaluate(audit())
    finally:
        await client.aclose()
    assert row["status"] == "failed" and all(v is None for v in row["scores"].values())
    assert row["native_results"]["correctness"]["input_tokens"] == 2000
    assert "not truncated" in row["native_results"]["correctness"]["error"]


@pytest.mark.asyncio
async def test_missing_visible_body_is_input_failure_not_model_na():
    def unexpected(_):
        pytest.fail("incomplete audit must not be graded")
    record = audit()
    record["audit_evidence"] = record["audit_evidence"][:-1]
    client = NativeJudgeClient(config(), transport=httpx.MockTransport(unexpected))
    try:
        row = await client.evaluate(record)
    finally:
        await client.aclose()
    assert row["status"] == "failed"
    assert all(g["status"] == "failed" for g in row["native_results"].values())


@pytest.mark.asyncio
async def test_file_transport_errors_preserve_all_tasks_and_refuse_overwrite(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    rows = [audit(), audit() | {"task_id": "s::2"}]
    audit_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / "scores.jsonl"
    transport = httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable"))
    await evaluate_file(audit_path, config(), output, transport=transport)
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(results) == 2 and all(r["status"] == "failed" for r in results)
    assert all(set(r["scores"]) == set(DIMENSIONS) for r in results)
    assert all(r["input_hash"] == token_hash([11, 22]) for r in results)
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        await evaluate_file(audit_path, config(), output, transport=transport)
    assert output.read_bytes() == previous


@pytest.mark.asyncio
async def test_missing_reference_coverage_rejected_before_scoring(tmp_path):
    a, r, o = (tmp_path / name for name in ("audit.jsonl", "refs.jsonl", "out.jsonl"))
    a.write_text(json.dumps(audit()) + "\n")
    r.write_text(json.dumps({"task_id": "another", "targets": []}) + "\n")
    with pytest.raises(ValueError, match="cover"):
        await evaluate_file(a, config(), o, references_path=r)
    assert not o.exists()


def test_cli_runs_saved_audit_through_fake_http(tmp_path, monkeypatch):
    a, c, o = (tmp_path / name for name in ("audit.jsonl", "config.json", "out.jsonl"))
    a.write_text(json.dumps(audit()) + "\n")
    c.write_text(json.dumps(asdict(config())))
    original_init = NativeJudgeClient.__init__

    def fake_init(self, cfg, **kwargs):
        original_init(self, cfg, **(kwargs | {"transport": httpx.MockTransport(lambda _: response())}))
    monkeypatch.setattr(NativeJudgeClient, "__init__", fake_init)
    main(["--audit", str(a), "--config", str(c), "--output", str(o)])
    assert json.loads(o.read_text())["scores"] == dict.fromkeys(DIMENSIONS, .75)


def test_cli_counts_token_ids_not_batchencoding_fields(tmp_path, monkeypatch):
    a, c, o = (tmp_path / name for name in ("audit.jsonl", "config.json", "out.jsonl"))
    a.write_text(json.dumps(audit()) + "\n")
    c.write_text(json.dumps(asdict(config(context_window=2048, max_tokens=128))))
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            ids = [1] * 2000
            # Transformers 5 returns BatchEncoding by default. Counting its two
            # fields would incorrectly pass this over-budget prompt through.
            return ids if kwargs.get("return_dict") is False else {"input_ids": ids, "attention_mask": ids}

    def from_pretrained(path, **kwargs):
        assert path == "pinned-local-tokenizer" and kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False and kwargs["revision"] == config().revision
        return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    original_init = NativeJudgeClient.__init__

    def unexpected(_):
        pytest.fail("the full token count must block an over-budget request")

    def fake_init(self, cfg, **kwargs):
        original_init(self, cfg, **(kwargs | {"transport": httpx.MockTransport(unexpected)}))
    monkeypatch.setattr(NativeJudgeClient, "__init__", fake_init)
    main(["--audit", str(a), "--config", str(c), "--output", str(o),
          "--tokenizer-path", "pinned-local-tokenizer"])
    row = json.loads(o.read_text())
    assert row["status"] == "failed" and len(calls) == 5
    assert all(g["input_tokens"] == 2000 for g in row["native_results"].values())
    assert all(call["return_dict"] is False for call in calls)


@pytest.mark.asyncio
async def test_protocol_metadata_changes_when_policy_changes():
    original = config()
    changed = replace(original, prompt="A different data policy.")
    first, second = NativeJudgeClient(original), NativeJudgeClient(changed)
    try:
        assert first.metadata["protocol_manifest"] != second.metadata["protocol_manifest"]
        assert second.metadata["protocol_manifest"]["data_policy"] == changed.prompt
    finally:
        await first.aclose()
        await second.aclose()
    with pytest.raises(ValueError, match="base_model"):
        replace(original, base_family="")
