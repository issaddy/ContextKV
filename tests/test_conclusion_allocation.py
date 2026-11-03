"""Final-envelope accounting and aggregate source-shard/forest targets."""
import json
from dataclasses import replace

import pytest

from bench.mock_engine import MockEngine
from controller.candidates import build_plan_candidates
from controller.claim_graph import ClaimGraph
from controller.context_budget import allocate_answer_window, bind_answer_window
from controller.context_schema import (
    AnswerPlan, AnswerPoint, ContextBudget, ContextPlan, EvidenceSet, EvidenceUnit,
    Need, SourceSpan, TaskState, TextRef,
)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor
from controller.schema import SessionSnapshot, TurnRecord
from controller.session_store import SessionStore
from controller.templates import ByteTemplate


def sources(count, width=0):
    blocks = tuple(make_evidence(f"R{i} qualifies. " + "Source detail. " * width, f"source-{i}")
                   for i in range(count))
    units = tuple(EvidenceUnit(f"R{i}", (SourceSpan(f"row-{i}", block.key, 0, len(block.text)),),
                               block.text) for i, block in enumerate(blocks))
    return EvidenceSet("sources", units, source_blocks=blocks)


def test_actual_history_custom_system_and_all_source_metadata_precede_content_shares():
    state = TaskState("t", "Infer the qualifying records.", needs=(Need("n", operation="infer"),),
        working_history=(TextRef(0, "user", 0, 100),), current_turn_id=1)
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(0, "H" * 100, "ok"),))
    plan = ContextPlan("p", state, sources(3), "all_supplied_documents",
        budget=ContextBudget(window_tokens=8000, output_tokens=200), answer_system_prompt="Custom " * 200)
    allocation = allocate_answer_window(plan, ByteTemplate(), snapshot)
    empty = replace(plan, selected_set=replace(plan.selected_set, selected_units=()),
                    coverage_scope="selected_sources", answer_plan=AnswerPlan((AnswerPoint("n"),)))
    actual = build_plan_candidates(snapshot, state.raw_question, empty, ByteTemplate(), layouts=("existing",))[0]
    assert actual.messages[0].content == plan.answer_system_prompt
    assert actual.messages[1].content.endswith("H" * 100)
    assert allocation.fixed_input_tokens == len(actual.input_ids)
    assert allocation.source_metadata_tokens > 0
    assert (allocation.fixed_input_tokens + allocation.source_metadata_tokens
            + allocation.conclusion_tokens + plan.budget.output_tokens) == plan.budget.window_tokens
    assert dict(allocation.need_tokens)["n"] == allocation.conclusion_tokens
    short = allocate_answer_window(replace(plan, answer_system_prompt="Custom"), ByteTemplate(), snapshot)
    assert short.conclusion_tokens > allocation.conclusion_tokens


def test_rebinding_refreshes_changed_history_and_union_sources_preserving_paid_read():
    from controller.reading import build_read_plan
    selected = sources(1)
    state = TaskState("t", "Infer all qualifying records.", needs=(Need("n", operation="infer"),),
        working_history=(TextRef(0, "user", 0, 50),), current_turn_id=1)
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(0, "H" * 100, "ok"),))
    plan = build_read_plan(state, selected, "selected_sources", ContextBudget(),
        template=ByteTemplate(), snapshot=snapshot)
    paid = plan.steps[0]
    combined = sources(2)
    successor = replace(paid, step_id="gap:read:0", input_unit_ids=("R1",), output_schema={})
    changed = replace(plan, selected_set=combined, steps=(paid, successor),
        task_state=replace(state, working_history=(TextRef(0, "user", 0, 100),)))
    rebound = bind_answer_window(changed, ByteTemplate(), snapshot, preserve_step_ids=(paid.step_id,))
    assert rebound.answer_allocation.fixed_input_tokens > plan.answer_allocation.fixed_input_tokens
    assert rebound.answer_allocation.source_metadata_tokens > plan.answer_allocation.source_metadata_tokens
    assert rebound.steps[0] == paid
    assert dict(rebound.steps[1].retained_conclusion_tokens)["n"] == max(0,
        dict(rebound.answer_allocation.need_tokens)["n"]-dict(paid.retained_conclusion_tokens)["n"])


def test_read_source_shards_share_one_retained_target_and_keep_extraction_cap():
    from controller.reading import build_read_plan, step_messages
    selected = sources(4)
    state = TaskState("t", "Infer all qualifying records.", needs=(Need("n", operation="infer"),))
    plan = build_read_plan(state, selected, "all_supplied_documents",
        ContextBudget(window_tokens=7000, read_input_tokens=5500, read_output_tokens=1800,
                      output_tokens=200), template=ByteTemplate(), read_packing="source_windows",
        answer_system_prompt="Custom " * 200)
    assert len(plan.steps) == 4
    target = dict(plan.answer_allocation.need_tokens)["n"]
    assert sum(dict(task.retained_conclusion_tokens)["n"] for task in plan.steps) == target
    assert all(task.output_budget == 1800 and task.requires_reduction for task in plan.steps)
    units = {unit.unit_id: unit for unit in selected.selected_units}
    for task in plan.steps:
        payload = json.loads(step_messages(plan, task,
            tuple(units[identity] for identity in task.input_unit_ids))[1].content)
        assert payload["conclusion_budget_tokens"] == dict(task.retained_conclusion_tokens)
        assert payload["extraction_output_budget_tokens"] == 1800
        assert payload["requires_reduction"] is True
    rebound = bind_answer_window(plan, ByteTemplate(), SessionSnapshot("s"), "Larger " * 300)
    assert [task.step_id for task in rebound.steps] == [task.step_id for task in plan.steps]
    assert [task.input_unit_ids for task in rebound.steps] == [task.input_unit_ids for task in plan.steps]
    assert rebound.answer_allocation.conclusion_tokens < plan.answer_allocation.conclusion_tokens
    assert sum(dict(task.retained_conclusion_tokens)["n"] for task in rebound.steps) == \
        rebound.answer_allocation.conclusion_tokens


class PartitionEngine(MockEngine):
    def __init__(self, compress):
        super().__init__()
        self.compress = compress

    async def generate(self, *, response_schema=None, **kwargs):
        rendered = bytes(kwargs["input_ids"]).decode()
        payload = json.loads(rendered.split("<|user|>\n")[-1].split("\n<|end|>")[0])
        if payload["units"]:
            facts = [{"need_id": "n", "text": unit["text"], "entity": unit["unit_id"],
                "relation": "qualifies", "value": None, "unit": "", "conditions": [],
                "span_ids": [span["span_id"] for span in unit["source_spans"]],
                "parent_claim_ids": [], "followups": []} for unit in payload["units"]]
        else:
            inputs = payload["claims"]
            if self.compress:
                facts = [{"need_id": "n", "text": "Qualifying records: " + ", ".join(
                    claim["entity"] for claim in inputs), "entity": "combined", "relation": "qualifies",
                    "value": None, "unit": "", "conditions": [], "span_ids": [],
                    "parent_claim_ids": [claim["claim_id"] for claim in inputs], "followups": []}]
            else:
                facts = [{"need_id": "n", "text": claim["text"], "entity": claim["entity"],
                    "relation": claim["relation"], "value": claim["value"], "unit": claim["unit"],
                    "conditions": claim["conditions"], "span_ids": [],
                    "parent_claim_ids": [claim["claim_id"]], "followups": []} for claim in inputs]
        self.response_text = json.dumps({"format_version": 2, "claims": facts,
            "support": ["n"], "conflicts": [], "open_needs": []})
        return await super().generate(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("compress", [True, False])
async def test_partition_forest_and_merges_spend_one_logical_target_with_complete_proof(tmp_path, compress):
    from controller.reading import build_read_plan
    selected = sources(6, width=55)
    state = TaskState("t", "Infer all qualifying records.", needs=(Need("n", operation="infer"),))
    plan = build_read_plan(state, selected, "all_supplied_documents",
        ContextBudget(window_tokens=6500, read_input_tokens=5500, read_output_tokens=2600,
                      output_tokens=200, body_tokens=0, max_steps=64), template=ByteTemplate(),
        read_packing="source_windows", answer_system_prompt="Custom " * 200)
    store = SessionStore(tmp_path / "state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, PartitionEngine(compress), ByteTemplate()).execute_reads(
        attempt.attempt_id, snapshot, plan)
    records = store.execution_steps(attempt.attempt_id)
    parts = tuple(record for record in records if ":part:0:" in record.step_id)
    assert len(parts) > 1
    logical_target = dict(plan.answer_allocation.need_tokens)["n"]
    assert sum(dict(record.task.retained_conclusion_tokens)["n"] for record in parts) == logical_target
    by_id = {record.step_id: record for record in records}
    for record in records:
        if ":merge:" in record.step_id or (record.task.kind == "REDUCE" and ":part:" not in record.step_id):
            assert dict(record.task.retained_conclusion_tokens)["n"] == sum(
                dict(by_id[identity].task.retained_conclusion_tokens)["n"]
                for identity in record.task.predecessor_step_ids)
    graph = ClaimGraph(outcome.claims)
    leaves = {span.source_key for point in outcome.plan.answer_plan.points
              for identity in point.claim_ids for span in graph.source_spans(identity)}
    assert leaves == {block.key for block in selected.source_blocks}
    candidate = build_plan_candidates(snapshot, state.raw_question, outcome.plan,
        ByteTemplate(), outcome=outcome, layouts=("existing",))[0]
    if compress:
        assert candidate.feasible
    else:
        assert not candidate.feasible and candidate.reason == "context_window_budget"
    assert len(tuple(record for record in records if record.task.kind == "REDUCE")) <= 2 * len(parts) - 1
