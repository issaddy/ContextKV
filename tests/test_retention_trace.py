"""Trace fidelity, workload windows and measured cost accounting (no GPU claims)."""
from dataclasses import replace
import json
import sqlite3

import pytest

from bench.mock_engine import MockEngine
from controller.context_schema import ContextBudget, EvidenceSet, EvidenceUnit, Need, SourceSpan, TaskState
from controller.evidence import make_evidence
from controller.planner import PlannerKnobs
from controller.prefix_retention import EmpiricalRetentionModel
from controller.reading import build_read_plan
from controller.reading import step_messages
from controller.schema import CompletionReceipt, ContextEpoch, Message
from controller.serialization import to_jsonable
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, token_hash
from profiling.retention_trace import (SourceLedger, choose_windows, event_id, export_attempt,
    integrate_prefix_residency, marginal_page_counts, reuse_pages, source_token_ranges, verify_trace, window_trace)


def simple_trace(turns=10):
    values = []
    for index in range(turns):
        events = []
        for step, kind, offset in (("read:0", "READ", 0), ("ANSWER", "ANSWER", 1)):
            ids = [index+1, offset, 7]
            hashed = token_hash(ids)
            events.append(dict(step_id=step, kind=kind, input_ids=ids, input_hash=hashed,
                prefix_input_ids=ids if kind == "READ" else [],
                request={"input_ids": ids, "max_new_tokens": 2048}, original_receipt={"input_hash": hashed},
                predecessor_step_ids=[] if kind == "READ" else ["read:0"],
                source_started_s=index*4+offset, source_completed_s=index*4+offset+1))
        values.append(dict(attempt_id="turn-"+str(index), session_id="session", events=events))
    return dict(format="contextkv_source_first_trace_v2", layout="source_first", turns=values,
        input_hashes=[event["input_hash"] for turn in values for event in turn["events"]], input_dag_digest="original")


def test_windows_preserve_contiguous_pressure_and_bound_total_work():
    trace = simple_trace()
    selected, omitted = choose_windows(trace, horizon_seconds=5, max_candidates=2,
        history_steps=2, max_replay_steps=10, seed=42)
    assert len(selected) == 2 and selected == choose_windows(trace, horizon_seconds=5,
        max_candidates=2, history_steps=2, max_replay_steps=10, seed=42)[0]
    for window in selected:
        expected = trace["turns"][window["first_turn"]:window["last_turn"]+1]
        assert window["turns"] == expected and window["steps"] <= 10
        local = window_trace(trace, window)
        assert local["input_hashes"] == [event["input_hash"] for turn in expected for event in turn["events"]]
        assert window["sampling_probability"] == .25 and window["sampling_weight"] == 4
        verify_trace(local)
    assert any(row["reason"] == "candidate_budget" for row in omitted)
    assert any(row["reason"] == "no_observed_post_horizon_turn" for row in omitted)
    none, rejected = choose_windows(trace, horizon_seconds=5, max_candidates=2,
        history_steps=2, max_replay_steps=2, seed=42)
    assert not none and any(row["reason"] == "window_exceeds_replay_step_budget" for row in rejected)


def test_fixed_dag_hash_proof_rejects_token_changes_and_duplicate_model_steps():
    trace = simple_trace()
    verify_trace(trace)
    changed = json.loads(json.dumps(trace))
    changed["turns"][0]["events"][0]["input_ids"][0] = 99
    with pytest.raises(ValueError, match="token input"):
        verify_trace(changed)
    duplicate = simple_trace()
    duplicate["turns"].append(duplicate["turns"][0])
    with pytest.raises(ValueError, match="duplicated"):
        verify_trace(duplicate)


def test_partial_causal_evidence_reuse_and_system_only_are_distinct():
    target = dict(prefix_input_ids=list(range(1, 13)), evidence_token_ranges=[[4, 10]], fixed_system_prefix_tokens=2)
    future = {**target, "prefix_input_ids": list(range(1, 9))+[99, 100, 101, 102]}
    receipt = dict(actual_device_hit_tokens=6, actual_host_loaded_tokens=2)
    result = reuse_pages(target, future, receipt, page_size=2, scope="scope")
    assert result["demand_category"] == result["actual_use_category"] == "partial_evidence"
    assert len(result["shared_evidence_page_ids"]) == 2 and result["actual_evidence_token_hits"] == 4
    assert len(result["actual_device_evidence_page_ids"]) == len(result["actual_host_evidence_page_ids"]) == 1
    miss = reuse_pages(target, future, dict(actual_device_hit_tokens=0, actual_host_loaded_tokens=0), page_size=2, scope="scope")
    assert miss["demand_category"] == "partial_evidence" and miss["actual_use_category"] == "none"
    common = reuse_pages(target, {**target, "prefix_input_ids": [1, 2]+list(range(50, 60))},
        receipt, page_size=2, scope="scope")
    assert common["demand_category"] == "system_or_metadata_only" and not common["shared_evidence_page_ids"]
    full = reuse_pages(target, target, dict(actual_device_hit_tokens=12, actual_host_loaded_tokens=0), page_size=2, scope="scope")
    assert full["actual_use_category"] == "full_retained_evidence"


def test_source_body_token_ranges_preserve_unicode_and_json_escapes():
    template = ByteTemplate()
    block = make_evidence('原文 📚 "quoted"\nsecond line', "unicode")
    unit = EvidenceUnit("unit", (SourceSpan("span", block.key, 0, len(block.text)),), block.text, need_ids=("n",))
    plan = build_read_plan(TaskState("t", "Question", needs=(Need("n"),)),
        EvidenceSet("set", (unit,), source_blocks=(block,)), "selected_sources", ContextBudget(),
        template=template, read_layout="source_first")
    messages = step_messages(plan, plan.steps[0], (unit,))
    rendered = template.render(messages)
    ranges, system = source_token_ranges(template, rendered, messages, (unit,))
    assert len(ranges) == 1 and ranges[0][0] > system
    start, end = ranges[0]
    assert template.tokenizer.decode(rendered.input_ids[start:end]) == json.dumps(block.text, ensure_ascii=False)[1:-1]


def prefix_view(gpu, host, pending=0):
    return dict(gpu_page_ids=gpu, host_page_ids=host,
        residency={"inflight_h2d_tokens": 0, "inflight_d2h_tokens": pending})


def test_shared_causal_pages_are_integrated_once_with_frozen_order_and_real_boundaries():
    first = {"a": prefix_view(["root", "a"], ["root"]),
             "b": prefix_view(["root", "b"], ["root", "b"])}
    second = {"a": prefix_view(["root"], ["root"]),
              "b": prefix_view(["root", "b"], ["root", "b"])}
    counts, union = marginal_page_counts(first, ["a", "b"], "gpu")
    assert counts == {"a": 2, "b": 1} and sum(counts.values()) == union == 3
    observation = dict(marginal_order=["a", "b"], samples=[
        {"timestamp": 0, "prefixes": first}, {"timestamp": 1, "prefixes": second},
        {"timestamp": 2, "prefixes": second}])
    a = integrate_prefix_residency(observation, "a", 0, 2, 64)
    b = integrate_prefix_residency(observation, "b", 0, 2, 64, protected_prefix_ids=("a",))
    assert a["gpu_token_seconds"] == 192 and b["gpu_token_seconds"] == 128
    assert a["gpu_token_seconds"]+b["gpu_token_seconds"] == a["gpu_observed_union_token_seconds"] == 320
    assert a["host_token_seconds"]+b["host_token_seconds"] == a["host_observed_union_token_seconds"] == 256
    standalone = integrate_prefix_residency(observation, "b", 0, 2, 64)
    assert standalone["gpu_token_seconds"] == 256  # A separately held path is not implied by observation alone.
    assert standalone["gpu_first_access_attributed_token_seconds"] == 128
    assert not integrate_prefix_residency(observation, "a", 0, 3, 64)["complete_window_coverage"]
    first["a"]["residency"]["inflight_d2h_tokens"] = 64
    assert integrate_prefix_residency(observation, "a", 0, 2, 64)["transfer_pending_observed"]


def test_forecast_control_is_paid_once_and_unweighted_by_reuse_probability():
    row = dict(features={"revisit": True}, cache_scope_hash="scope", horizon_seconds=30.0,
        reused=True, native_input_hashes=["same"], retained_input_hashes=["same"],
        native_reuse_cost_ms=100, retained_reuse_cost_ms=20, retention_control_ms=5,
        native_window_cost_ms=100, retained_window_cost_ms=25,
        gpu_page_count=1, host_page_count=0, page_size=64,
        native_gpu_token_seconds=0, retained_gpu_token_seconds=0,
        native_host_token_seconds=0, retained_host_token_seconds=0, sampling_weight=1)
    absent = {**row, "reused": False, "native_reuse_cost_ms": None,
              "retained_reuse_cost_ms": None, "sampling_weight": 3,
              "native_window_cost_ms": 0, "retained_window_cost_ms": 5}
    forecast = EmpiricalRetentionModel.fit([row, absent]).predict({"revisit": True}, "scope", 30)
    assert forecast.reuse_probability == .25 and forecast.retention_control_ms == 5
    assert forecast.saved_ms == 15
    never = EmpiricalRetentionModel.fit([absent]).predict({"revisit": True}, "scope", 30)
    assert never.native_reuse_cost_ms is None and never.saved_ms == -5
    frequency = EmpiricalRetentionModel.fit([row, absent], reuse_rows=[row, absent,
        {**absent, "sampling_weight": 4}]).predict({"revisit": True}, "scope", 30)
    assert frequency.saved_ms == 5
    collateral = EmpiricalRetentionModel.fit([{**row, "native_window_cost_ms": 200,
        "retained_window_cost_ms": 250}]).predict({"revisit": True}, "scope", 30)
    assert collateral.native_reuse_cost_ms > collateral.retained_reuse_cost_ms
    assert collateral.expected_collateral_saved_ms < 0 and collateral.saved_ms == -50


class ReadingEngine(MockEngine):
    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema is not None:
            self.response_text = json.dumps({"claims": [{"need_id": "n", "text": "Use the form.",
                "entity": "policy", "relation": "procedure", "value": None, "unit": "", "conditions": [],
                "version": "v1", "span_ids": ["span"], "parent_claim_ids": []}],
                "support": ["n"], "conflicts": [], "open_needs": []})
        else:
            self.response_text = "Use the form."
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_export_hydrates_single_steps_preserves_budgets_and_rejects_task_first(tmp_path):
    template = ByteTemplate()
    block = make_evidence("Use the form.", "policy")
    unit = EvidenceUnit("unit", (SourceSpan("span", block.key, 0, len(block.text)),), block.text, need_ids=("n",))
    state = TaskState("task", "How do I apply?", needs=(Need("n"),), current_turn_id=1)
    plan = build_read_plan(state, EvidenceSet("set", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(), template=template, read_layout="source_first")
    path = tmp_path/"state.sqlite"
    store = SessionStore(path)
    controller = Controller(store, ReadingEngine(), template=template,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    response = await controller.serve_turn("session", state.raw_question, context_plans=(plan,), request_id="one")
    attempt = store.get_attempt(response.attempt_id)
    meta = dict(response=to_jsonable(response), max_new_tokens=controller.policy.max_new_tokens,
                dataset="unit_test", split="dev")
    source = SourceLedger(path)
    raw = source.execution_steps(attempt.attempt_id)[0]
    stage, typed, _ = source.execution_step(attempt.attempt_id, raw.step_id)
    assert isinstance(typed.engine_epoch, ContextEpoch)
    assert isinstance(typed.input_messages[0], Message) and isinstance(typed.receipt, CompletionReceipt)
    exported = export_attempt(source, attempt, template, {"temperature": 0}, meta)
    read, answer = exported["events"]
    assert read["input_hash"] == typed.receipt.input_hash
    assert read["request"]["max_new_tokens"] == typed.task.output_budget
    assert read["original_output_ids"] == list(typed.output_ids)
    assert answer["kind"] == "ANSWER" and answer["input_hash"] == attempt.result.receipt.input_hash
    assert read["sampling_evidence"] == "declared_original_run_config"
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        source.db.execute("DELETE FROM attempts")
    old = replace(attempt, reading_outcome=replace(attempt.reading_outcome,
        plan=replace(attempt.reading_outcome.plan, read_layout="task_first")))
    with pytest.raises(ValueError, match="task_first"):
        export_attempt(source, old, template, {"temperature": 0}, meta)
    assert event_id(exported, read) == attempt.attempt_id+":"+typed.step_id
    source.close()
    await controller.aclose()
    store.close()
