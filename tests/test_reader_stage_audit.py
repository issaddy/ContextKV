"""Offline reader stage audit, immutable review and native cohort boundaries."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from training.reader_data import history_digest, reader_review_body, review_reader_trace


def trace(identity="answer", phase="ANSWER"):
    return {"example_id": identity, "task_id": identity, "attempt_id": identity,
        "conversation_id": identity, "turn": 1, "split": "train", "phase": phase,
        "step_id": identity, "question": "Read the mass.", "messages": [
            {"role": "user", "content": "Observed mass: 655 kg."}],
        "actual_history": [], "actual_history_sha256": history_digest([]),
        "prompt_token_hash": identity, "reader_identity": "R0", "task_state": None,
        "read_protocol_version": 2, "read_task": {"output_budget": 1024},
        "output_schema": {"type": "object"}, "actual_output": "655 kg",
        "original_sources": [{"source_key": {"document_id": identity}, "start": 0,
            "end": 31, "text": "Observed mass: 655 kg. Total: 666 kg."}]}


@dataclass
class Completion:
    text: str
    input_tokens: int = 10
    usage: dict = None
    status: str = "completed"
    finish_reason: str = "stop"
    elapsed_ms: float = 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["READ", "REDUCE", "ANSWER"])
async def test_fixed_old_targets_only_get_independent_review_with_complete_original(phase):
    item = trace(phase, phase)
    target = "655 kg" if phase == "ANSWER" else json.dumps({"claims": [{"text": "655 kg"}]})
    calls, ledger = [], []

    class Client:
        config = SimpleNamespace(max_new_tokens=1024)
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))

        def complete(self, messages, *, response_format, max_new_tokens, chat_template_kwargs):
            calls.append(response_format["json_schema"]["name"])
            body = json.loads(messages[-1]["content"])
            assert body["target_output"] == (target if phase == "ANSWER" else json.loads(target))
            assert "666 kg" in body["original_sources"][0]["text"]
            assert "target" not in response_format["json_schema"]["schema"]["properties"]
            return Completion(json.dumps({"quality": .2, "source_support_verified": False,
                "correct_task_binding": True, "visible_prompt_support": True, "upstream_errors": [], "reason": "incorrect total"}),
                usage={"completion_tokens": 4})

    result = await review_reader_trace(item, Client(), minimum_quality=.85,
        reviewer_identity="teacher", review_input_budget=4096, record_call=ledger.append,
        fixed_target_text=target)
    assert calls == ["reader_review"]
    assert result["target_text"] == target and result["status"] == "rejected"
    assert result["review_protocol"] == "visible_target_independent_review_v6"
    assert ledger[0]["stage"] == "review"
    assert "666 kg" in reader_review_body(item)["original_sources"][0]["text"]


def test_candidate_diagnosis_keeps_ties_histories_and_declared_denominator():
    from training.train_evidence_policy import candidate_quality_diagnostics
    common = dict(task_id="observed", actual_history=[], body_budget=100, output_budget=20,
        reader_checkpoint="fixed", reader_profile_id="protocol", scorer_identity={"version": 1},
        query_encoder={"id": "fixed"}, evidence_set={"quality_prediction": .8})
    rewards = [{**common, "input_hash": "good", "quality_components": {"q": 1}},
               {**common, "input_hash": "bad", "quality_components": {"q": .2}}]
    report = candidate_quality_diagnostics(rewards, weights={"q": 1},
                                           declared_task_ids=["observed", "missing"])
    assert report["declared_tasks"] == 2 and report["observed_tasks"] == 1
    assert report["missing_task_ids"] == ["missing"]
    assert report["available_group_oracle_mean"] == 1
    assert report["available_group_prediction_regret_interval"] == [0, .8]
    assert candidate_quality_diagnostics(list(reversed(rewards)), weights={"q": 1},
        declared_task_ids=["observed", "missing"]) == report
    changed_history = {**rewards[1], "actual_history": [{"assistant": "a different actual answer"}]}
    split = candidate_quality_diagnostics([rewards[0], changed_history], weights={"q": 1},
                                          declared_task_ids=["observed"])
    assert split["comparison_groups"] == 2
    assert split["available_group_oracle_mean"] == .6


@pytest.mark.asyncio
async def test_batch_audit_includes_current_numeric_read_and_selector_protocols(tmp_path, monkeypatch):
    from training import reader_data

    items = [dict(trace("read", "READ"), read_protocol_version=5),
             dict(trace("select", "REDUCE"), read_protocol_version=7),
             dict(trace(), read_protocol_version=None)]
    reader_data.write_rows(tmp_path / "traces.jsonl", items)
    scope = SimpleNamespace(model_path="specified-fixture", identity="fixed",
                            manifest=lambda: {"identity": "fixed"})
    monkeypatch.setattr("engine_patches.contextkv_model_scope.load_model_scope", lambda _: scope)
    monkeypatch.setattr("adapters.query_selector.OpenAICompletionClient",
                        lambda _: SimpleNamespace(close=lambda: None))
    seen = []

    async def review(item, client, **kwargs):
        seen.append((item["phase"], item["read_protocol_version"]))
        return {"example_id": item["example_id"], "review_protocol": reader_data.REVIEW_PROTOCOL,
                "status": "accepted"}

    monkeypatch.setattr(reader_data, "review_reader_trace", review)
    manifest = await reader_data.review_reader_traces({"reader_audit": {
        "base_scope": "specified-fixture", "data": {"output_dir": str(tmp_path), "minimum_quality": .85},
        "review": {"base_urls": ["http://fixture.invalid"], "max_input_tokens": 4096,
                   "max_output_tokens": 1024, "context_window": 8192, "timeout_seconds": 1}}})
    assert set(seen) == {("READ", 5), ("REDUCE", 7), ("ANSWER", None)}
    assert manifest["requested_examples"] == manifest["reviewed_examples"] == 3
