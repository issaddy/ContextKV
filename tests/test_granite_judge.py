import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from bench.granite_judge import (CRITERION_PROFILES, GraniteJudge, GraniteJudgeConfig,
                                guardian_block, native_input, parse_score)
from bench.native_baseline import NativeSGLangClient


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        # Transformers 5 defaults to BatchEncoding; the adapter must explicitly
        # request token IDs rather than iterating the mapping's string keys.
        ids = [101, 202, 303, 404]
        return ids if kwargs.get("return_dict") is False else {"input_ids": ids}


def record():
    def key(name):
        return {"corpus_id": "test", "document_id": name, "passage_id": name, "version": "v1"}
    return {"task_id": "case", "conversation_id": "session", "query": "Which limit applies?",
        "answer": "The limit is 12 MiB [E_a].", "input_ids": [11, 12], "measurement_kind": "simulation",
        "actual_history": [{"speaker": "user", "text": "Earlier question"},
                           {"speaker": "assistant", "text": "A prior untrusted statement"}],
        "retrieved_evidence": [{"document_id": "a", "text": "Current top-one only"}],
        "visible_evidence_manifest": [{"reference_id": "E_a", "key": key("a")},
                                      {"reference_id": "E_b", "key": key("b")}],
        "audit_evidence": [{"key": key("a"), "text": "Exact source a\nwith code: x = 12"},
                           {"key": key("b"), "text": "Earlier still-visible source b"},
                           {"key": key("hidden"), "text": "Archived but not visible; must not be used"}],
        "targets": [{"text": "must not cross the scoring boundary"}]}


def test_native_input_preserves_roles_full_visible_source_and_last_answer():
    value = record()
    before = deepcopy(value)
    messages, documents = native_input(value)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-2]["content"] == value["answer"]
    assert messages[-1]["content"].startswith("<guardian><no-think>")
    assert messages[:2] == [{"role": r["speaker"], "content": r["text"]} for r in value["actual_history"]]
    assert documents == [{"doc_id": "E_a", "text": value["audit_evidence"][0]["text"]},
                         {"doc_id": "E_b", "text": value["audit_evidence"][1]["text"]}]
    assert "prior untrusted" not in json.dumps(documents)
    assert "must not cross" not in json.dumps([messages, documents])
    assert value == before
    value["audit_evidence"] = value["audit_evidence"][:1]
    with pytest.raises(ValueError, match="complete original body"):
        native_input(value)


@pytest.mark.parametrize("text,expected", [("<think>\n</think><score>yes</score>", True),
    ("<score>no</score>", False), (" <think>source conflicts</think> <score>YES</score> ", True)])
def test_native_binary_risk_polarity(text, expected):
    assert parse_score(text)["risk_detected"] is expected
    assert guardian_block(think=True).startswith("<guardian><think>")


@pytest.mark.parametrize("text", ["yes", '{"score":"yes"}', "[RESULT] 5",
    "<score>yes</score><score>no</score>", "<score>unknown</score>", "<think>unfinished<score>no</score>"])
def test_invalid_native_output_is_not_salvaged(text):
    with pytest.raises(ValueError, match="native"):
        parse_score(text)


@pytest.mark.parametrize("response_kind", ["stop", "length", "abort", "http_error", "wrong_prompt", "invalid_score"])
def test_native_ids_exact_wire_without_double_template_and_failures_retained(response_kind):
    requests = []

    def respond(request):
        data = json.loads(request.content)
        requests.append(data)
        assert request.url.path == "/generate"
        assert data["input_ids"] == [101, 202, 303, 404]
        assert not {"messages", "text", "session_id"}.intersection(data)
        if response_kind == "http_error":
            return httpx.Response(503, json={"error": "fixture_unavailable"})
        return httpx.Response(200, json={"text": "bad" if response_kind == "invalid_score" else "<score>yes</score>",
            "output_ids": [8], "prompt_token_ids": [999] if response_kind == "wrong_prompt" else data["input_ids"],
            "meta_info": {"finish_reason": {"type": response_kind if response_kind in {"length", "abort"} else "stop"},
                          "completion_tokens": 1, "prompt_tokens": 4}})

    async def run():
        async with httpx.AsyncClient(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as http:
            native = NativeSGLangClient("http://fixture.test", http_client=http, audit_prompt_ids=True)
            tokenizer = Tokenizer()
            judge = GraniteJudge(GraniteJudgeConfig("http://fixture.test", 128), tokenizer, native)
            result = await judge.evaluate(record())
            assert len(tokenizer.calls) == 1 and tokenizer.calls[0][1]["tokenize"] is True
            assert tokenizer.calls[0][1]["add_generation_prompt"] is True
            assert tokenizer.calls[0][1]["return_dict"] is False
            assert result["judge_input_ids"] == requests[0]["input_ids"]
            assert result["raw_response"] == result["native_observation"]["response"]
            assert "scores" not in result and not result["quality_noninferiority_established"]
            assert result["generation_measurement_kind"] == "simulation"
            if response_kind == "stop":
                assert result["status"] == "scored" and result["risk_detected"] is True
                assert result["output_ids"] == [8] and result["finish_reason"] == "stop"
            else:
                assert result["status"] == "failed" and result["risk_detected"] is None
                assert result["error"]
                if response_kind == "length":
                    assert result["truncated"] and result["failure_kind"] == "truncated_output"
                if response_kind == "abort":
                    assert result["output_ids"] == [8] and result["finish_reason"] == "abort"
            await judge.aclose()

    asyncio.run(run())


def test_exact_input_overflow_sends_no_request_and_does_not_trim():
    async def run():
        async with httpx.AsyncClient(base_url="http://fixture.test",
            transport=httpx.MockTransport(lambda request: pytest.fail("overflow must not generate"))) as http:
            client = NativeSGLangClient("http://fixture.test", http_client=http)
            judge = GraniteJudge(GraniteJudgeConfig("http://fixture.test", 67), Tokenizer(), client)
            result = await judge.evaluate(record())
            assert result["failure_kind"] == "input_overflow"
            assert result["judge_input_ids"] == [101, 202, 303, 404]
            assert len(result["documents"]) == 2 and not client.observations
    asyncio.run(run())
    with pytest.raises(ValueError, match="model context"):
        GraniteJudge(GraniteJudgeConfig("http://fixture.test", 200), Tokenizer(), object(), model_context_limit=128)


def test_file_retains_failed_case_and_continues_without_retry(tmp_path, monkeypatch):
    import transformers
    import bench.granite_judge as module
    values = [record(), {**record(), "task_id": "second"}]
    audit = tmp_path/"audit.jsonl"
    audit.write_text("".join(json.dumps(r)+"\n" for r in values))
    tokenizer, calls = Tokenizer(), []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(max_position_embeddings=131072))

    def respond(request):
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"context_length": 128})
        data = json.loads(request.content)
        calls.append(data)
        return httpx.Response(200, json={"text": "<score>no</score>", "output_ids": [8],
            "prompt_token_ids": data["input_ids"], "meta_info": {"completion_tokens": 1,
                "finish_reason": {"type": "abort" if len(calls) == 1 else "stop"}}})

    async def run():
        async with httpx.AsyncClient(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as http:
            monkeypatch.setattr(module, "NativeSGLangClient", lambda *a, **k:
                NativeSGLangClient("http://fixture.test", http_client=http, audit_prompt_ids=True))
            target = tmp_path/"ledger.jsonl"
            summary = await module.evaluate_file(audit, GraniteJudgeConfig("http://fixture.test", 128),
                                                 target, tokenizer_path="fixture-only")
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            assert [r["status"] for r in rows] == ["failed", "scored"]
            assert rows[0]["finish_reason"] == "abort" and rows[1]["risk_detected"] is False
            assert summary["rows"] == 2 and summary["scored"] == 1 and summary["failed_task_ids"] == ["case"]
            assert target.with_suffix(".summary.json").exists()
            assert len(calls) == len(tokenizer.calls) == 2
    asyncio.run(run())


FIXTURES = Path(__file__).parent/"fixtures"
VARIANTS = json.loads((FIXTURES/"granite_byoc_variants.json").read_text())


def test_frozen_profile_texts_and_original_default_are_exact():
    assert guardian_block() == (FIXTURES/"granite_ungroundedness_native_v1.txt").read_text()
    assert GraniteJudgeConfig("http://fixture", 128).criterion_profile == "ungroundedness"
    for profile, criterion in VARIANTS["criteria"].items():
        assert CRITERION_PROFILES[profile] == criterion
    with pytest.raises(ValueError, match="criterion_profile"):
        GraniteJudgeConfig("http://fixture", 128, criterion_profile="arbitrary_prompt")


@pytest.mark.parametrize("profile", ["relation_scope_v1", "citation_binding_v1"])
@pytest.mark.parametrize("variant", VARIANTS["cases"], ids=lambda v: v["record"]["task_id"])
def test_structure_variants_are_payload_contracts_not_model_accuracy(variant, profile):
    value = deepcopy(variant["record"])
    # Deliberately add private sidecar material to prove it is not submitted.
    value.update(offline_expected=variant["offline_expected"], label_source=variant["label_source"])
    before, requests = deepcopy(value), []

    def respond(request):
        data = json.loads(request.content)
        requests.append(data)
        assert set(data) == {"input_ids", "stream", "sampling_params", "return_prompt_token_ids"}
        # This constant mock answer intentionally does not follow the labels.
        return httpx.Response(200, json={"text": "<score>no</score>", "output_ids": [8],
            "prompt_token_ids": data["input_ids"], "meta_info": {"completion_tokens": 1,
                "finish_reason": {"type": "stop"}}})

    async def run():
        async with httpx.AsyncClient(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as http:
            tokenizer = Tokenizer()
            client = NativeSGLangClient("http://fixture.test", http_client=http, audit_prompt_ids=True)
            judge = GraniteJudge(GraniteJudgeConfig("http://fixture.test", 128, criterion_profile=profile), tokenizer, client)
            result = await judge.evaluate(value)
            assert result["judge"]["criterion_profile"] == profile
            assert result["judge"]["criterion_profile_version"] == 1
            assert result["judge"]["criterion"] == VARIANTS["criteria"][profile]
            assert "yes=criterion_risk_detected" in result["judge"]["risk_polarity"]
            assert "scores" not in result and result["quality_noninferiority_established"] is False
            assert "synthetic_definition_not_human_rating" not in json.dumps([result["messages"], result["documents"]])
            assert result["messages"][:-3] == [{"role": m["speaker"], "content": m["text"]}
                                                for m in value["actual_history"]]
            assert result["messages"][-2]["content"] == value["answer"]
            assert result["messages"][-1]["content"] == guardian_block(criterion_profile=profile)
            assert [d["text"] for d in result["documents"]] == [b["text"] for b in value["audit_evidence"]]
            if profile == "citation_binding_v1" and variant["offline_expected"][profile] is None:
                assert result["status"] == "not_covered" and result["risk_detected"] is None
                assert result["coverage_reason"] == "no_supported_citation_syntax"
                assert not requests and not tokenizer.calls and "judge_input_ids" not in result
            else:
                assert result["status"] == "scored" and result["risk_detected"] is False
                assert len(requests) == len(tokenizer.calls) == 1
                assert tokenizer.calls[0][0] == result["messages"]
                assert tokenizer.calls[0][1]["documents"] == result["documents"]
                assert requests[0]["input_ids"] == result["judge_input_ids"]
                assert requests[0]["sampling_params"] == {"temperature": 0, "max_new_tokens": 64}
            if profile == "citation_binding_v1" and value["task_id"] == "unknown-in-group":
                assert result["rule_audit"]["unknown_citations"] == ["E_f00d"]
                assert "unknown_citations" in result["rule_audit"]["flags"]
                assert result["status"] == "scored"  # Unknown ID is not skipped or auto-scored.
            assert value == before
    asyncio.run(run())


def test_file_counts_not_covered_separately_from_failures(tmp_path, monkeypatch):
    import transformers
    import bench.granite_judge as module
    values = [record(), {**record(), "task_id": "uncited", "answer": "The answer has no citation.",
        "actual_history": [{"speaker": "assistant", "text": "Prior citation [E_a] is not current."}]}]
    audit = tmp_path/"audit.jsonl"
    audit.write_text("".join(json.dumps(r)+"\n" for r in values))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(max_position_embeddings=131072))

    def respond(request):
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"context_length": 128})
        data = json.loads(request.content)
        return httpx.Response(200, json={"text": "<score>yes</score>", "output_ids": [8],
            "prompt_token_ids": data["input_ids"], "meta_info": {"completion_tokens": 1, "finish_reason": {"type": "stop"}}})

    async def run():
        async with httpx.AsyncClient(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as http:
            monkeypatch.setattr(module, "NativeSGLangClient", lambda *a, **k:
                NativeSGLangClient("http://fixture.test", http_client=http, audit_prompt_ids=True))
            target = tmp_path/"ledger.jsonl"
            summary = await module.evaluate_file(audit,
                GraniteJudgeConfig("http://fixture.test", 128, criterion_profile="citation_binding_v1"),
                target, tokenizer_path="fixture-only")
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            assert [r["status"] for r in rows] == ["scored", "not_covered"]
            assert summary["rows"] == 2 and summary["scored"] == 1
            assert summary["failed_task_ids"] == [] and summary["not_covered_task_ids"] == ["uncited"]
            assert rows[1]["native_score"] is None and rows[1]["risk_detected"] is None
    asyncio.run(run())
