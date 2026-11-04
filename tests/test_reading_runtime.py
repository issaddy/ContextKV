"""Reading protocol, provenance and one outer commit with scripted engine outputs."""
from reading_helpers import ScriptedReadingEngine, fact, read_output, material
import json
from dataclasses import replace

import pytest

from bench.mock_engine import MockEngine
from controller.claim_graph import ClaimGraph
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceClaim, EvidenceSet,
                                       Need, ReadingOutcome, TaskState)
from controller.execution import ReadingExecutor
from controller.planner import PlannerKnobs
from controller.reading import build_read_plan
from controller.reduction import reduce_claims
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, verify_evidence_spans


@pytest.mark.asyncio
async def test_read_answer_commit_once_and_replay_does_no_work(tmp_path):
    block, unit = material("Eligible applicants use the online form.")
    state = TaskState("t", "How do I apply?", needs=(Need("n", operation="procedure"),), current_turn_id=1)
    selected = EvidenceSet("s", (unit,), quality_prediction=.9, source_blocks=(block,))
    plan = build_read_plan(state, selected, "selected_sources", ContextBudget(), template=ByteTemplate())
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    response = await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one")
    snapshot = store.get_snapshot("s")
    assert snapshot.version == 1 and len(snapshot.raw_history) == 1
    assert response.read_count == 1 and response.reduce_count == 0
    assert response.total_input_tokens > response.input_tokens
    assert snapshot.raw_history[0].evidence_set.set_id == "s"
    assert snapshot.raw_history[0].answer_plan.points[0].claim_ids == ("read:0:claim:0",)
    assert snapshot.raw_history[0].answer_plan.points[0].citation_ids == response.citation_scope
    assert engine.constraints[0][0] is not None and engine.constraints[0][1] is None
    assert engine.constraints[1][0] is None and engine.constraints[1][1] is not None
    steps = store.execution_steps(response.attempt_id)
    assert steps[0].result.claims[0].source_spans == unit.source_spans
    assert steps[0].metrics["quote_ms"] >= 0
    assert steps[0].metrics["cache_quote"]["quotes"][0]["candidate_id"] == response.attempt_id+":read:0"
    assert steps[0].metrics["cache_config_hash"] == (await engine.info()).cache_config_hash
    calls = len(engine.constraints)
    replay = await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one")
    assert replay.idempotent_replay and len(engine.constraints) == calls
    assert replay.total_input_tokens == response.total_input_tokens
    await controller.drain()
    assert not store.outbox()


@pytest.mark.asyncio
async def test_source_prefix_observer_finishes_before_read_epoch_retirement_and_qa_commit(tmp_path):
    block, unit = material("Eligible applicants use the online form.")
    state = TaskState("t", "How do I apply?", needs=(Need("n", operation="procedure"),), current_turn_id=1)
    plan = build_read_plan(state, EvidenceSet("set", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(), template=ByteTemplate(), read_layout="source_first")
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    seen = []

    async def observe(current_plan, task, units, rendered, record):
        assert current_plan.read_layout == "source_first"
        assert engine._epochs[record.engine_epoch.engine_session_id].state == "COMMITTED"
        assert not store.get_snapshot("session").raw_history
        assert record.input_ids == rendered.input_ids
        assert units == (unit,)
        seen.append(record.receipt.attempt_id)
        return {"owner_id": "independent-owner", "state": "RETAINED"}

    controller.observe_prefix = observe
    response = await controller.serve_turn("session", state.raw_question, context_plans=(plan,), request_id="one")
    _, record, _ = store.execution_step(response.attempt_id, "read:0")
    assert record.metrics["evidence_prefix"]["owner_id"] == "independent-owner"
    assert record.metrics["prefix_control_ms"] >= 0
    assert engine._epochs[record.engine_epoch.engine_session_id].state == "RETIRED"
    assert len(store.get_snapshot("session").raw_history) == 1
    await controller.serve_turn("session", state.raw_question, context_plans=(plan,), request_id="one")
    assert seen == [record.receipt.attempt_id]
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
async def test_derived_answer_keeps_all_leaf_spans_and_only_final_legal_citations(tmp_path):
    from controller.evidence_planner import render_selection

    first, a = material("A is approved.", "a")
    second, b = material("B is approved.", "b")
    template = ByteTemplate()
    body_budget = max(len(template.tokenizer.encode(render_selection((unit,)))) for unit in (a, b))+1
    state = TaskState("t", "Count approved entries.", needs=(Need("n", operation="count"),))
    plan = build_read_plan(state, EvidenceSet("set", (a, b), source_blocks=(first, second)),
        "all_supplied_documents", ContextBudget(body_tokens=body_budget), template=template)
    engine = ScriptedReadingEngine([read_output(fact("a", "A"), fact("b", "B"))])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    response = await controller.serve_turn("s", state.raw_question, request_id="one", context_plans=(plan,))
    point = controller.store.get_snapshot("s").raw_history[0].answer_plan.points[0]
    assert {span.span_id for span in point.source_spans} == {"a", "b"}
    assert len(point.citation_ids) == 2 and point.citation_ids == response.citation_scope
    final = controller.store.get_attempt(response.attempt_id).plan.candidate
    assert {dict(final.reference_map)[ref] for ref in point.citation_ids} == final.required_blocks
    assert final.required_blocks == {span.source_key for span in point.source_spans}
    assert len({span.key for span in final.evidence_spans}) < len(final.required_blocks)
    await controller.drain()


@pytest.mark.asyncio
async def test_hierarchical_reduce_keeps_all_source_leaves_within_each_window(tmp_path):
    class CombiningEngine(MockEngine):
        async def generate(self, *, response_schema=None, **kwargs):
            rendered = bytes(kwargs["input_ids"]).decode()
            payload = json.loads(rendered.split("<|user|>\n")[-1].split("\n<|end|>")[0])
            if payload["units"]:
                claims = [fact(unit["source_spans"][0]["span_id"], unit["unit_id"])
                          for unit in payload["units"]]
            else:
                claim = fact("", "combined")
                claim["span_ids"] = []
                claim["parent_claim_ids"] = [claim["claim_id"] for claim in payload["claims"]]
                claims = [claim]
            self.response_text = json.dumps(read_output(*claims))
            return await super().generate(**kwargs)

    materialized = [material(f"Entity {index} participates.", str(index)) for index in range(100)]
    blocks, units = zip(*materialized)
    state = TaskState("t", "Infer the combined relationship.", needs=(Need("n", operation="infer"),))
    budget = ContextBudget(window_tokens=10000, body_tokens=2048,
                           read_input_tokens=6000, read_output_tokens=3000)
    plan = build_read_plan(state, EvidenceSet("s", units, source_blocks=blocks),
                           "all_supplied_documents", budget, template=ByteTemplate())
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, CombiningEngine(), ByteTemplate()).execute_reads(
        attempt.attempt_id, snapshot, plan)
    reductions = [step for step in outcome.steps if step.task.kind == "REDUCE"]
    assert len(reductions) > 1
    assert all(len(step.input_ids)+step.task.output_budget <= budget.window_tokens for step in outcome.steps)
    point = outcome.plan.answer_plan.points[0]
    assert len(point.claim_ids) == 1 and len(point.source_spans) == 100
    assert len(ClaimGraph(outcome.claims).leaf_claims(point.claim_ids[0])) == 100
    from controller.candidates import build_plan_candidates
    candidates = build_plan_candidates(snapshot, state.raw_question, outcome.plan,
                                        ByteTemplate(), outcome=outcome)
    for candidate in candidates:
        assert candidate.feasible
        assert len(candidate.input_ids)+candidate.output_budget <= budget.window_tokens
        assert len(candidate.required_blocks) == 100
        assert {span.key for span in candidate.evidence_spans} < candidate.required_blocks
        verify_evidence_spans(candidate.messages, candidate.evidence_spans, blocks,
                              candidate.reference_map)
        content = candidate.messages[-1].content
        assert '"claim_ids":["c0"]' in content
        assert "parent_claim_ids" not in content
    assert len(outcome.plan.answer_plan.points[0].source_spans) == 100


@pytest.mark.parametrize("operation", ["union", "group"])
def test_enumeration_and_group_roots_retain_values_in_final_answer_window(tmp_path, operation):
    from controller.candidates import build_plan_candidates
    from controller.schema import SessionSnapshot

    materialized = [material(text+" Background notes."*15, name) for name, text in (
        ("a", "Alpha has the role Archivist."),
        ("b", "Alpha was founded in 1997."),
        ("c", "Beta has the role Curator."))]
    blocks, units = zip(*materialized)
    claims = tuple(EvidenceClaim(f"read:0:claim:{index}", "n", "extract", text=text,
        value=value, source_spans=units[source].source_spans, entity=entity, relation=relation)
        for index, (entity, relation, value, text, source) in enumerate((
            ("Alpha", "role", "Archivist", "Alpha has the role Archivist.", 0),
            ("Alpha", "founded", "1997", "Alpha was founded in 1997.", 1),
            ("Beta", "role", "Curator", "Beta has the role Curator.", 2),
            ("Alpha", "role", "Archivist", "The role of Alpha is Archivist.", 0))))
    reduced = reduce_claims(operation, claims, claim_id="reduce:n:1:claim:0", need_id="n")
    if operation == "union":
        assert len(reduced.value) == 3
    else:
        alpha = next(group for group in reduced.value if group["entity"] == "Alpha")
        assert {fact["relation"] for fact in alpha["facts"]} == {"role", "founded"}
    assert len(reduced.parent_claim_ids) == 4
    state = TaskState("t", "List all roles and founding years.",
                      needs=(Need("n", operation="enumerate"),))
    selected = EvidenceSet("s", units, source_blocks=blocks)
    plan = ContextPlan("p", state, selected, "all_supplied_documents",
        answer_plan=ClaimGraph((*claims, reduced)).answer_plan(state),
        budget=ContextBudget(body_tokens=750))
    outcome = ReadingOutcome(plan, claims=(*claims, reduced))
    candidates = build_plan_candidates(SessionSnapshot("s"), state.raw_question, plan,
                                        ByteTemplate(), outcome=outcome)
    for candidate in candidates:
        assert candidate.feasible
        originals = {block.key: block for block in blocks}
        assert candidate.required_blocks <= originals.keys()
        assert any(span.source_end-span.source_start < len(originals[span.key].text)
                   for span in candidate.evidence_spans)
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
            blocks, candidate.reference_map) == candidate.required_blocks
        prompt = candidate.messages[-1].content
        assert all(value in prompt for value in ("Archivist", "1997", "Curator"))
        assert "read:0:claim:" not in prompt


@pytest.mark.asyncio
async def test_every_table_read_carries_header_and_coverage_counts_header_once(tmp_path):
    class TableEngine(MockEngine):
        async def generate(self, *, response_schema=None, **kwargs):
            text = bytes(kwargs["input_ids"]).decode()
            payload = json.loads(text.split("<|user|>\n")[-1].split("\n<|end|>")[0])
            if response_schema["properties"]["format_version"]["const"] == 7:
                self.response_text = json.dumps({"format_version": 7, "views": {
                    alias: {parent: True for parent in view["properties"]}
                    for alias, view in response_schema["properties"]["views"]["properties"].items()}})
                return await super().generate(**kwargs)
            assert any(unit["unit_id"] == "header" for unit in payload["units"])
            self.response_text = json.dumps(read_output(*(fact(unit["source_spans"][0]["span_id"],
                unit["unit_id"]) for unit in payload["units"] if unit["unit_id"] != "header")))
            return await super().generate(**kwargs)

    header, header_unit = material("Shipment identifier | Approved status", "header")
    pairs = [material(f"S{index:02d} | Approved", f"row{index}") for index in range(18)]
    blocks, rows = zip(*pairs)
    rows = tuple(replace(unit, dependencies=(header_unit.unit_id,), table_id="shipments") for unit in rows)
    state = TaskState("t", "Count approved shipments.", needs=(Need("n", operation="count"),))
    plan = build_read_plan(state, EvidenceSet("s", (header_unit, *rows), source_blocks=(header, *blocks)),
        "all_supplied_documents", ContextBudget(read_output_tokens=1024), template=ByteTemplate())
    assert len(plan.steps) > 1
    assert all("header" in task.input_unit_ids for task in plan.steps)
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, TableEngine(), ByteTemplate()).execute_reads(attempt.attempt_id, snapshot, plan)
    assert outcome.claims[-1].value == 18
    assert len(outcome.coverage) == 19
    assert sum(row[2] == "header" for row in outcome.coverage) == 1
    assert all(any(span.span_id == "header" for span in claim.source_spans)
               for claim in outcome.claims if claim.kind == "extract")
