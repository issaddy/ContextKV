"""Offline reader stage audit, immutable review and native cohort boundaries."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from controller.context_schema import EvidenceSet
from training.reader_curriculum import prepare_stage_cohorts
from training.reader_data import (audit_reader_targets, history_digest, prepare_reader_data,
    reader_review_body, review_reader_trace, rows, write_rows)
from training.preference_rollout import real_candidate_family


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


def review(item, **updates):
    return {"example_id": item["example_id"], "status": "accepted", "target_text": "655 kg",
        "target_origin": "old_teacher", "quality": 1, "source_support_verified": True,
        "correct_task_binding": True, "actual_history_sha256": item["actual_history_sha256"],
        "prompt_token_hash": item["prompt_token_hash"], "reader_identity": "R0", **updates}


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
    calls, ledger = [], []

    class Client:
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))

        def complete(self, messages, *, response_format):
            calls.append(response_format["json_schema"]["name"])
            body = json.loads(messages[-1]["content"])
            assert body["target_text"] == "655 kg"
            assert "666 kg" in body["original_sources"][0]["text"]
            assert "target" not in response_format["json_schema"]["schema"]["properties"]
            return Completion(json.dumps({"quality": .2, "source_support_verified": False,
                "correct_task_binding": True, "visible_prompt_support": True, "upstream_errors": []}),
                usage={"completion_tokens": 4})

    result = await review_reader_trace(item, Client(), minimum_quality=.85,
        reviewer_identity="teacher", review_input_budget=4096, record_call=ledger.append,
        fixed_target_text="655 kg")
    assert calls == ["reader_review"]
    assert result["target_text"] == "655 kg" and result["status"] == "rejected"
    assert result["review_protocol"] == "visible_target_independent_review_v2"
    assert ledger[0]["stage"] == "review"
    assert "666 kg" in reader_review_body(item)["original_sources"][0]["text"]


def test_old_target_inventory_preserves_text_and_unknown_visibility(tmp_path):
    old, output = tmp_path/"old", tmp_path/"audit"
    first, second = trace(), trace("read", "READ")
    first["reading_steps"] = [{"step_id": "read", "phase": "READ"}]
    write_rows(old/"traces.jsonl", [first, second])
    write_rows(old/"reviews.jsonl", [review(first), review(second, visible_prompt_support=False,
        source_support_verified=False, upstream_errors=[{"phase": "READ", "step_id": "read",
            "reason": "quantity extraction"}])])
    before = (old/"reviews.jsonl").read_bytes()
    manifest = audit_reader_targets({"reader_training": {"data": {"output_dir": str(output),
        "audit_sources": [{"traces_path": str(old/"traces.jsonl"), "reviews_path": str(old/"reviews.jsonl")}]}}})
    assert (old/"reviews.jsonl").read_bytes() == before
    assert rows(output/"fixed_targets.jsonl")[0]["target_text"] == "655 kg"
    assert manifest["model_requests"] == 0
    assert manifest["phases"]["ANSWER"]["visibility_unverified"] == 1
    assert manifest["phases"]["READ"]["hidden_information_target"] == 1
    assert manifest["phases"]["READ"]["source_unsupported_target"] == 1
    assert rows(output/"read_upstream_errors.jsonl")[0]["observed_step"]


def test_cohorts_respect_declared_sessions_reader_exposure_and_report_shortages(tmp_path):
    native = [{"task_id": f"t{i}", "conversation_id": f"s{i//2}", "turn": i%2+1,
        "split": "train", "actual_history": []} for i in range(10)]
    source = tmp_path/"native.jsonl"
    write_rows(source, native)
    heldout, exposed = tmp_path/"dev.jsonl", tmp_path/"old_train.jsonl"
    write_rows(heldout, [{"conversation_id": "s0"}, {"conversation_id": "s1"}])
    write_rows(exposed, [{"conversation_id": "s1"}])
    result = prepare_stage_cohorts({"reader_stage_cohort": {"output_dir": str(tmp_path/"cohorts"),
        "sources": [{"split": "train", "turns_path": str(source)}], "validation_fraction": 0,
        "dev_session_paths": [str(heldout)], "reader_train_paths": [str(exposed)],
        "full": {"train": 6, "dev": 4}, "pilot": {"train": 2, "dev": 1}}})
    full = json.loads(open(result["cohorts"]["full"]).read())
    pilot = json.loads(open(result["cohorts"]["pilot"]).read())
    train = {member["conversation_id"] for member in full["members"] if member["split"] == "train"}
    dev = {member["conversation_id"] for member in full["members"] if member["split"] == "dev"}
    assert not train & dev and dev == {"s0"}
    assert full["splits"]["dev"]["shortfall"] == 2
    assert full["heldout_training_exposure_exclusions"] == 2
    assert {member["task_id"] for member in pilot["members"]} <= {member["task_id"] for member in full["members"]}
    assert full["candidate_status"] == "pending_real_source_pool_construction_and_reader_rollout"


def test_real_candidate_family_deduplicates_and_rejects_missing_dependencies():
    a = SimpleNamespace(unit_id="a", dependencies=())
    b = SimpleNamespace(unit_id="b", dependencies=("a",))
    pool = SimpleNamespace(units=(a, b))
    # Empty sets need no serialization of fixture units; they still count once.
    empty = EvidenceSet("empty", ())
    selected, audit = real_candidate_family(SimpleNamespace(task_id="task"), pool,
        {"current": empty, "redundant": empty, "wrong_condition_version": None})
    assert list(selected) == ["current"] and not audit["ready"]
    assert [row["reason"] for row in audit["exclusions"]] == ["duplicate_evidence_set", "no_matching_source_intervention"]
    with pytest.raises(ValueError, match="unresolved source dependency"):
        real_candidate_family(SimpleNamespace(task_id="task"), pool, {"deleted": EvidenceSet("broken", (b,))})


def test_prepare_requires_new_independent_review_when_stage_requests_it(tmp_path):
    item = trace()
    write_rows(tmp_path/"traces.jsonl", [item])
    write_rows(tmp_path/"reviews.jsonl", [review(item)])
    result = prepare_reader_data({"reader_training": {"data": {"output_dir": str(tmp_path),
        "minimum_quality": .85, "read_protocol_versions": [2], "split_seed": 42,
        "validation_fraction": 0, "require_independent_review": True}}})
    assert result["selection_exclusions"] == {"independent_review_required": 1}
    assert rows(tmp_path/"train.jsonl") == []


@pytest.mark.asyncio
async def test_audit_namespaces_same_local_ids_across_assets_and_reverifies_each_fixed_target(tmp_path, monkeypatch):
    from training.reader_data import review_reader_traces

    sources = []
    for asset, amount in (("r1", "655"), ("curriculum", "666")):
        item = trace("shared-local-example")
        item.update(attempt_id="shared-local-attempt", prompt_token_hash=f"prompt-{asset}",
            messages=[{"role": "user", "content": f"Observed mass: {amount} kg."}])
        directory = tmp_path/asset
        write_rows(directory/"traces.jsonl", [item])
        write_rows(directory/"reviews.jsonl", [review(item, target_text=f"{amount} kg")])
        sources.append({"asset_id": asset, "traces_path": str(directory/"traces.jsonl"),
                        "reviews_path": str(directory/"reviews.jsonl")})
    output = tmp_path/"audit"
    manifest = audit_reader_targets({"reader_training": {"data": {
        "output_dir": str(output), "audit_sources": sources}}})
    audited = rows(output/"traces.jsonl")
    targets = {row["example_id"]: row for row in rows(output/"fixed_targets.jsonl")}
    assert len(audited) == len(targets) == 2
    original_reviews = rows(manifest["original_reviews_path"])
    assert {item["example_id"] for item in original_reviews} == set(targets)
    assert {item["target_text"] for item in original_reviews} == {"655 kg", "666 kg"}
    assert {item["attempt_id"] for item in audited} == {
        "r1:shared-local-attempt", "curriculum:shared-local-attempt"}
    assert all(item["original_example_id"] == "shared-local-example" for item in audited)
    assert all(item["original_attempt_id"] == "shared-local-attempt" for item in audited)
    assert manifest["phases"]["ANSWER"]["saved_targets"] == 2
    calls = []

    class Client:
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))

        def __init__(self, *args, **kwargs):
            pass

        def complete(self, messages, *, response_format):
            assert response_format["json_schema"]["name"] == "reader_review"
            body = json.loads(messages[-1]["content"])
            assert body["target_text"] in body["messages"][-1]["content"]
            calls.append(body["target_text"])
            return Completion(json.dumps({"quality": 1, "source_support_verified": True,
                "correct_task_binding": True, "visible_prompt_support": True, "upstream_errors": []}),
                usage={"completion_tokens": 4})

        def close(self):
            pass

    monkeypatch.setattr("adapters.query_selector.OpenAICompletionClient", Client)
    monkeypatch.setattr("engine_patches.contextkv_model_scope.load_model_scope", lambda path:
        SimpleNamespace(model_path="/fixture", identity="teacher", manifest=lambda: {"fixture": True}))
    config = {"reader_training": {"base_scope": "fixture", "data": {
        "output_dir": str(output), "fixed_targets_path": manifest["fixed_targets_path"],
        "minimum_quality": .85}, "review": {"base_urls": ["http://fixture"],
        "max_input_tokens": 4096, "max_output_tokens": 1024, "context_window": 8192, "timeout_seconds": 30}}}
    await review_reader_traces(config, concurrency=2)
    saved = rows(output/"reviews.jsonl")
    assert sorted(calls) == ["655 kg", "666 kg"]
    assert {item["example_id"] for item in saved} == set(targets)
    for item in saved:
        assert item["target_text"] == targets[item["example_id"]]["target_text"]
        assert item["source_asset_id"] == targets[item["example_id"]]["source_asset_id"]
        assert item["original_example_id"] == "shared-local-example"
        assert item["source_reviews"] == targets[item["example_id"]]["source_reviews"]


@pytest.mark.asyncio
async def test_cohort_label_preparation_uses_actual_sqlite_sources_and_rejects_mismatched_old_labels(tmp_path):
    from bench.mock_engine import MockEngine
    from controller.evidence import make_evidence
    from controller.planner import PlannerKnobs
    from controller.service import Controller
    from controller.session_store import SessionStore
    from controller.templates import ByteTemplate
    from controller.serialization import to_jsonable
    from training.task_data import prepare_cohort_task_labels

    database = tmp_path/"sessions.sqlite3"
    template = ByteTemplate()
    store = SessionStore(database)
    controller = Controller(store, MockEngine(fingerprint=template.fingerprint, response_text="Actual generated answer"),
        template=template, knobs=PlannerKnobs(strategy="always_rebuild"))
    source = make_evidence("The policy allowance is 12 credits.", "policy", corpus_id="train", version="v1")
    await controller.serve_turn("session", "Read the policy.", (source,), request_id="first", corpus_snapshot_id="snapshot")
    history = to_jsonable(store.get_snapshot("session").raw_history)
    response = await controller.serve_turn("session", "What is the allowance?", (source,), request_id="second", corpus_snapshot_id="snapshot")
    await controller.aclose()
    store.close()
    row = {"task_id": "task", "conversation_id": "session", "turn": 2, "question": "What is the allowance?",
        "split": "train", "actual_history": history, "snapshot_id": "snapshot", "grounding": [], "response": to_jsonable(response)}
    grounding = [{"passage_id": "policy", "start": 0, "end": len(source.text), "quote": source.text}]
    actual_path, original_path = tmp_path/"actual.jsonl", tmp_path/"author.jsonl"
    source_path, prior_path = tmp_path/"sources.jsonl", tmp_path/"old_labels.jsonl"
    write_rows(actual_path, [row])
    write_rows(original_path, [{"task_id": "task", "grounding": grounding,
        "reference_history": [{"role": "assistant", "content": "Authored answer differs"}], "reference_answer": "12"}])
    write_rows(source_path, [{"source_id": "policy", "text": source.text, "version": "v1", "source_uri": source.source_uri}])
    write_rows(prior_path, [{**row, "actual_history": [], "script_history_review_status": "reviewed"}])
    cohort_path = tmp_path/"cohort.json"
    cohort_path.write_text(json.dumps({"members": [{"task_id": "task", "conversation_id": "session",
        "turn": 2, "split": "dev", "actual_history_sha256": history_digest(history)}]}))
    result = prepare_cohort_task_labels({"task_cohort_labeling": {"output_dir": str(tmp_path/"labels"),
        "cohort_manifest": str(cohort_path), "actual_turns_path": str(actual_path), "state_db": str(database),
        "original_turns_path": str(original_path), "sources_path": str(source_path), "existing_labels_path": str(prior_path)},
        "training": {"teacher_urls": ["http://127.0.0.1:30000"], "task_history_encoder_device": "cuda:1"}})
    selected = rows(result["turns_path"])[0]
    assert selected["actual_history"] == history
    assert selected["actual_history"][0]["assistant_text"] == "Actual generated answer"
    assert selected["cohort_split"] == "dev" and selected["split"] == "train"
    assert selected["original_sources"][0]["text"] == source.text
    assert selected["grounding_original_sources"][0]["text"] == source.text
    assert selected["source_attempt"]["attempt_id"] == response.attempt_id
    assert "source_scope" not in selected  # Observed retrieval does not impose a task constraint.
    assert result["registered_states"] == result["pending_labels"] == 1 and result["reused_labels"] == 0
    assert rows(tmp_path/"labels"/"label_reuse_exclusions.jsonl")[0]["reason"] == "existing_label_not_bound_to_registered_actual_state"


def mock_task_label_teacher(monkeypatch, *, truncated=False, source_valid=True):
    import numpy as np
    from adapters.query_selector import CompletionResult
    from test_task_state_sources import Characters
    calls = []

    class Encoder:
        tokenizer = Characters()
        manifest = {"fixture": True}

        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"
            self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0}

        def encode(self, texts):
            self.usage["calls"] += 1
            self.usage["input_tokens"] += sum(len(text) for text in texts)
            return np.ones((len(texts), 2))

        def close(self):
            pass

    class Client:
        manifest = {"fixture": True}

        def __init__(self, config):
            self.config = config

        def complete(self, messages, *, response_format):
            name = response_format["json_schema"]["name"]
            body = json.loads(messages[-1]["content"])
            calls.append((name, body))
            if name == "contextkv_task_spans":
                if truncated:
                    return CompletionResult("completed", "unfinished", "length", 20, {"completion_tokens": 30}, elapsed_ms=5)
                ref = {"turn_id": body["turn_id"], "speaker": "user", "boundary_chars": [0, len(body["question"])-1]}
                value = {"relation": "standalone", "resume_task_id": "", "history_spans": [], "conditions": {},
                    "condition_removals": [], "needs": {"n0": {"anchor_spans": [ref], "subject_refs": [], "operation": "lookup",
                        "condition_ids": [], "predecessor_need_ids": [], "scope": "selected_sources"}},
                    "version_intent": {"mode": "current", "temporal_relation": "at", "document_families": [],
                        "version_ids": [], "resolved_version_ids": [], "time_refs": [], "source_turns": []}}
            else:
                assert name == "actual_history_grounding_review"
                assert body["task_state"]["raw_question"] == body["question"]
                assert body["grounding_original_sources"][0]["text"] == "Allowance: 12 credits."
                value = {"script_history_mismatch": False, "applicable_grounding_indices": [0],
                    "correct_task_binding": True, "source_support_verified": source_valid}
            return CompletionResult("completed", json.dumps(value), "stop", 20, {"completion_tokens": 30}, elapsed_ms=5)

        def close(self):
            pass

    monkeypatch.setattr("training.task_labeling.LocalBGE", Encoder)
    monkeypatch.setattr("training.task_labeling.OpenAICompletionClient", Client)
    return calls


def task_label_run_config(tmp_path):
    item = {"task_id": "task", "conversation_id": "session", "turn": 1, "question": "What is the allowance?",
        "actual_history": [], "snapshot_id": "snapshot", "split": "train", "cohort_split": "dev",
        "cohort_manifest_sha256": "bound", "reference_history": [{"role": "assistant", "content": "Authored history"}],
        "grounding": [{"passage_id": "policy", "quote": "Allowance: 12 credits."}],
        "original_sources": [{"text": "Allowance: 12 credits."}],
        "grounding_original_sources": [{"text": "Allowance: 12 credits."}]}
    write_rows(tmp_path/"turns.jsonl", [item])
    cohort = tmp_path/"cohort.json"
    cohort.write_text(json.dumps({"members": [{"task_id": "task", "conversation_id": "session", "turn": 1,
        "split": "dev", "actual_history_sha256": history_digest([])}]}))
    from training.reader_data import file_digest
    item["cohort_manifest_sha256"] = file_digest(cohort)
    write_rows(tmp_path/"turns.jsonl", [item])
    return {"training": {"data_dir": str(tmp_path), "teacher_urls": ["http://fixture"],
        "task_history_encoder_device": "cpu", "task_history_encoder_dtype": "float32", "task_label_cohort_manifest": str(cohort)}}


@pytest.mark.asyncio
async def test_cohort_task_labels_use_existing_teacher_independent_review_and_separate_dev_costs(tmp_path, monkeypatch):
    from training.task_labeling import label_rollouts
    calls = mock_task_label_teacher(monkeypatch)
    config = task_label_run_config(tmp_path)
    manifest = await label_rollouts(config, concurrency=1)
    assert [name for name, _ in calls] == ["contextkv_task_spans", "actual_history_grounding_review"]
    assert "Authored history" not in json.dumps(calls[0][1])
    assert manifest["status"] == "completed" and manifest["labels"] == 1
    assert manifest["cohort_splits"] == {"train": 0, "dev": 1}
    assert rows(manifest["split_labels"]["train"]) == []
    assert len(rows(manifest["split_labels"]["dev"])) == 1
    assert manifest["teacher_input_tokens"] == 40 and manifest["teacher_output_tokens"] == 60
    assert [call["stage"] for call in rows(manifest["cost_ledger"])] == ["task", "grounding"]
    reused = await label_rollouts(config, concurrency=1)
    assert reused["teacher_calls"] == 0 and reused["generated_labels"] == 0
    assert len(rows(manifest["cost_ledger"])) == 2


@pytest.mark.asyncio
async def test_truncated_task_label_call_keeps_paid_cost_and_incomplete_cohort_manifest(tmp_path, monkeypatch):
    from training.task_labeling import label_rollouts
    mock_task_label_teacher(monkeypatch, truncated=True)
    with pytest.raises(ExceptionGroup, match="unhandled errors"):
        await label_rollouts(task_label_run_config(tmp_path), concurrency=1)
    manifest = json.loads((tmp_path/"task_labels_manifest.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["pending_labels"] == 1
    assert manifest["teacher_input_tokens"] == 20 and manifest["teacher_output_tokens"] == 30
    calls = rows(manifest["cost_ledger"])
    assert calls[0]["finish_reason"] == "length" and calls[0]["model_request_executed"]
    assert rows(tmp_path/"task_labels.jsonl") == []


@pytest.mark.asyncio
async def test_source_rejected_task_state_does_not_become_a_cohort_label(tmp_path, monkeypatch):
    from training.task_labeling import label_rollouts
    mock_task_label_teacher(monkeypatch, source_valid=False)
    manifest = await label_rollouts(task_label_run_config(tmp_path), concurrency=1)
    assert manifest["status"] == "incomplete" and manifest["pending_labels"] == 1
    assert manifest["teacher_calls"] == 2 and manifest["teacher_output_tokens"] == 60
    assert rows(tmp_path/"task_labels.jsonl") == []
    assert rows(tmp_path/"task_label_rejections.jsonl")[0]["source_support_verified"] is False
