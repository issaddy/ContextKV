from dataclasses import replace
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from bench.mock_engine import MockEngine
from controller.evidence import make_evidence
from controller.planner import PlannerKnobs
from controller.serialization import to_jsonable
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from training.reader_data import (augment_retrieved_reads, export_reader_traces, history_digest, prepare_reader_data,
                                  reader_review_body, review_reader_traces, rows, write_rows)
from training.train_reader import encode_example, measured_command, physical_device_ids


class TrainingTokenizer:
    eos_token_id = 256
    pad_token_id = 256

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<|{message['role']}|>\n{message['content']}\n<|end|>\n" for message in messages)+"<|assistant|>\n"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode()), "offset_mapping": [
            (index, index+1) for index, char in enumerate(text) for _ in char.encode()]}


@pytest.mark.asyncio
async def test_reader_export_uses_exact_runtime_prompt_actual_history_and_source_coordinates(tmp_path):
    database = tmp_path/"sessions.sqlite3"
    store = SessionStore(database)
    template = ByteTemplate()
    engine = MockEngine(fingerprint=template.fingerprint, response_text="Linux allowance is 12 credits.")
    service = Controller(store, engine, template=template, knobs=PlannerKnobs(strategy="always_rebuild"))
    source = make_evidence("Linux allowance is 12 credits.", "policy", corpus_id="train", version="v2")
    source = replace(source, span=(80, 80+len(source.text)))
    turns = []
    for number, question in enumerate(("What is the Linux allowance?", "Give that amount again."), 1):
        before = store.read_snapshot("train-session", "train-snapshot")
        response = await service.serve_turn("train-session", question, (source,), request_id=f"task-{number}",
                                            corpus_snapshot_id="train-snapshot")
        turns.append({"task_id": f"task-{number}", "conversation_id": "conversation", "turn": number,
            "split": "train", "question": question, "actual_history": to_jsonable(before.raw_history),
            "snapshot_id": "train-snapshot", "response": to_jsonable(response),
            "reference_history": [{"speaker": "assistant", "text": "AUTHOR_HISTORY_MUST_NOT_REPLACE_ACTUAL"}]})
    await service.aclose()
    store.close()
    # The original Dense5 rollout database predates execution_steps.
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE execution_steps")
    connection.commit()
    connection.close()
    path = tmp_path/"turns.jsonl"
    write_rows(path, turns)
    config = {"reader_training": {"data": {"output_dir": str(tmp_path/"data"), "rollouts": [{
        "turns_path": str(path), "state_db": str(database), "source_partition": "native_train",
        "split": "train", "reader_identity": "R0"}]}}}
    result = export_reader_traces(config)
    traces = rows(result["traces_path"])
    assert result["turns"] == 2 and result["phases"] == {"ANSWER": 2}
    assert result["sources"][0]["has_execution_steps"] is False
    assert traces[1]["actual_history"][0]["assistant_text"] == "Linux allowance is 12 credits."
    assert "AUTHOR_HISTORY" not in json.dumps(traces)
    assert traces[0]["original_sources"][0]["start"] == 80
    encoded, rejected = encode_example({**traces[0], "target_text": "12 credits."}, TrainingTokenizer(), max_length=10000)
    assert rejected is None
    assert encoded["labels"][:traces[0]["prompt_tokens"]] == [-100]*traces[0]["prompt_tokens"]
    assert encoded["labels"][-1] == TrainingTokenizer.eos_token_id
    with pytest.raises(ValueError, match="tokenizer/template"):
        encode_example({**traces[0], "target_text": "12", "prompt_token_hash": "changed"}, TrainingTokenizer(), max_length=10000)
    assert encode_example({**traces[0], "target_text": "12 credits."}, TrainingTokenizer(), max_length=10)[0] is None
    mutation = {"task_id": traces[0]["task_id"], "mutation_id": "allowance-12-to-19",
        "source_key": traces[0]["original_sources"][0]["source_key"],
        "start": 80+source.text.index("12"), "end": 80+source.text.index("12")+2,
        "original_text": "12", "replacement": "19"}
    mutation_path = tmp_path/"mutations.jsonl"
    write_rows(mutation_path, [mutation])
    config["reader_training"]["data"].update(retrieved_read_turns=1, paired_mutations_path=str(mutation_path))
    template.tokenizer = TrainingTokenizer()
    augmented = augment_retrieved_reads(config, template=template)
    assert augmented["read_examples"] == 2 and augmented["read_protocol_version"] == 2
    new_inputs = [row for row in rows(result["traces_path"]) if row["phase"] == "READ"]
    assert all(len(row["messages"]) == 3 and row["actual_output"] == "" for row in new_inputs)
    changed = next(row for row in new_inputs if row["source_mutations"])
    assert changed["pair_id"] == traces[0]["example_id"]
    assert "19 credits" in changed["original_sources"][0]["text"]
    assert changed["prompt_token_hash"] != new_inputs[0]["prompt_token_hash"]
    assert changed["original_sources"][0]["source_key"]["content_sha256"] != source.key.content_sha256
    assert changed["trace_origin"]["target_origin"] == "pending_real_teacher_generation"


def test_reader_selection_requires_quality_source_support_actual_history_and_current_read_contract(tmp_path):
    data = tmp_path/"data"
    data.mkdir()
    history = [{"turn_id": 1, "user_text": "Use Windows", "assistant_text": "Actual Windows answer"}]
    base = {"task_id": "task", "conversation_id": "conversation", "split": "train", "phase": "ANSWER",
        "actual_history_sha256": history_digest(history), "prompt_token_hash": "prompt", "reader_identity": "R0",
        "read_protocol_version": None, "actual_output": "A raw wrong answer is not a target"}
    traces = [{**base, "example_id": name} for name in ("accepted", "mismatch", "unsupported", "unscored")]
    traces.append({**base, "example_id": "legacy", "phase": "READ", "read_protocol_version": 1})
    reviews = [{"example_id": trace["example_id"], "status": "accepted", "quality": .95,
        "source_support_verified": True, "correct_task_binding": True, "target_text": "Correct sourced answer",
        "target_origin": "teacher_corrected_actual_runtime", "actual_history_sha256": trace["actual_history_sha256"],
        "prompt_token_hash": "prompt", "reader_identity": "R0"} for trace in traces if trace["example_id"] != "unscored"]
    reviews[1]["actual_history_sha256"] = history_digest([{**history[0], "assistant_text": "Authored hypothetical history"}])
    reviews[2]["source_support_verified"] = False
    write_rows(data/"traces.jsonl", traces)
    write_rows(data/"reviews.jsonl", reviews)
    result = prepare_reader_data({"reader_training": {"data": {"output_dir": str(data),
        "read_protocol_versions": [2], "minimum_quality": .85, "validation_fraction": 0,
        "split_seed": 42}}})
    selected = rows(data/"train.jsonl")
    assert len(selected) == 1 and selected[0]["example_id"] == "accepted"
    assert selected[0]["target_text"] != base["actual_output"]
    assert result["selection_exclusions"] == {"actual_history_binding_mismatch": 1,
        "unsupported_target": 1, "missing_review": 1, "old_read_protocol": 1}
    assert result["training_turns"] == 1


def test_reader_export_rejects_non_training_partition_before_reading_any_assets(tmp_path):
    with pytest.raises(ValueError, match="native training partition"):
        export_reader_traces({"reader_training": {"data": {"output_dir": str(tmp_path),
            "rollouts": [{"split": "dev"}]}}})


@pytest.mark.asyncio
async def test_reader_cost_maps_visible_devices_and_preserves_each_command_measurement(tmp_path, monkeypatch):
    assert physical_device_ids("1,3,4,5") == [1, 3, 4, 5]
    assert physical_device_ids("") == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,4,5")
    class FixtureSampler:
        def __init__(self, devices, processes, *, interval):
            assert devices == []
            self.samples = []
        async def start(self):
            self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 0, "host_gib": 20})
        async def stop(self):
            self.samples.append({"timestamp": time.monotonic(), "gpu_gib": 0, "host_gib": 20})
        def observation(self):
            return {"samples": self.samples, "measurement_kind": "unit_fixture"}
    monkeypatch.setattr("bench.study_resources.StudyResourceSampler", FixtureSampler)
    config = {"reader_training": {"data": {"output_dir": str(tmp_path)}}}
    for _ in range(2):
        await measured_command(config, "prepare", lambda: {"status": "prepared"})
    saved = list(tmp_path.glob("prepare_cost.*.json"))
    assert len(saved) == 2
    assert all(json.loads(path.read_text())["physical"]["complete_window_coverage"] for path in saved)


def test_review_deduplicates_exact_prompt_views_without_dropping_unrepresented_actual_qa():
    history = [{"turn_id": 1, "user_text": "Old question", "assistant_text": "Actual prior answer"}]
    messages = [{"role": "user", "content": "SOURCE original. Old question"},
                {"role": "assistant", "content": "Actual prior answer"}]
    trace = {"phase": "ANSWER", "question": "Current?", "actual_history": history,
        "actual_history_sha256": history_digest(history), "messages": messages,
        "task_state": None, "original_sources": [{"text": "SOURCE original.", "start": 80, "end": 96}],
        "actual_output": "Original actual output"}
    body = reader_review_body(trace)
    assert body["messages"] == messages and body["additional_actual_history"] == []
    assert "text" not in body["original_sources"][0]
    trace["actual_history"] = [{**history[0], "assistant_text": "Unrepresented complete actual answer"}]
    body = reader_review_body(trace)
    assert body["additional_actual_history"] == trace["actual_history"]


@pytest.mark.asyncio
async def test_review_input_over_budget_is_a_costed_exclusion_and_other_samples_continue(tmp_path, monkeypatch):
    from adapters.query_selector import CompletionResult
    class Client:
        tokenizer = TrainingTokenizer()
        def __init__(self, *args, **kwargs):
            pass
        def complete(self, messages, *, response_format):
            body = json.loads(messages[-1]["content"])
            if body["question"] == "Over budget":
                return CompletionResult("input_over_budget", input_tokens=40000, elapsed_ms=3)
            if body["question"] == "Truncated":
                return CompletionResult("completed", '{"target":"unfinished', "length", 200,
                                        {"completion_tokens": 2000}, elapsed_ms=9)
            return CompletionResult("completed", json.dumps({"target": "Correct sourced answer", "quality": .95,
                "source_support_verified": True, "correct_task_binding": True}), "stop", 100,
                {"completion_tokens": 20}, elapsed_ms=4)
        def close(self):
            pass
    monkeypatch.setattr("adapters.query_selector.OpenAICompletionClient", Client)
    monkeypatch.setattr("engine_patches.contextkv_model_scope.load_model_scope", lambda path:
        SimpleNamespace(model_path="/fixture", identity="fixture-r0", manifest=lambda: {"fixture": True}))
    traces = [{"example_id": identity, "task_id": identity, "step_id": "ANSWER", "phase": "ANSWER",
        "question": question, "actual_history": [], "actual_history_sha256": history_digest([]),
        "prompt_token_hash": identity, "messages": [{"role": "user", "content": question}],
        "task_state": None, "original_sources": [], "actual_output": "Old output", "reader_identity": "R0",
        "read_protocol_version": None, "split": "train", "conversation_id": identity}
        for identity, question in (("over", "Over budget"), ("length", "Truncated"), ("good", "Normal"))]
    write_rows(tmp_path/"traces.jsonl", traces)
    result = await review_reader_traces({"reader_training": {"base_scope": "fixture",
        "data": {"output_dir": str(tmp_path), "minimum_quality": .85, "read_protocol_versions": [2]},
        "review": {"base_urls": ["http://fixture"], "max_input_tokens": 30000,
            "max_output_tokens": 2000, "context_window": 32768, "timeout_seconds": 30}}}, concurrency=2)
    assert result["status_counts"] == {"input_over_budget": 1, "teacher_target_truncated": 1, "accepted": 1}
    assert result["reviewed_examples"] == 1
    ledger = {row["example_id"]: row for row in rows(tmp_path/"review_calls.jsonl")}
    assert ledger["over"]["model_input_tokens"] == ledger["over"]["model_output_tokens"] == 0
    assert ledger["over"]["input_tokens"] == 40000
    assert ledger["length"]["model_input_tokens"] == 200 and ledger["length"]["model_output_tokens"] == 2000
    saved = {row["example_id"]: row for row in rows(tmp_path/"reviews.jsonl")}
    assert saved["length"]["raw_output"] == '{"target":"unfinished'
    assert "target_text" not in saved["length"]
    prepared = prepare_reader_data({"reader_training": {"data": {"output_dir": str(tmp_path),
        "minimum_quality": .85, "read_protocol_versions": [2], "validation_fraction": 0, "split_seed": 42}}})
    assert prepared["selection_exclusions"] == {"review_input_over_budget": 1, "teacher_target_truncated": 1}
    assert prepared["training_turns"] == 1
