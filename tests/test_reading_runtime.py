"""Reading protocol, provenance and one outer commit with scripted engine outputs."""
import json
import re
from dataclasses import replace

import pytest

from bench.mock_engine import MockEngine
from controller.claim_graph import ClaimGraph
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceClaim, EvidencePool, EvidenceSet,
                                       EvidenceUnit, Need, ReadingOutcome, SourceSpan, TaskState)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor
from controller.engine_client import UnknownOutcomeError
from controller.planner import PlannerKnobs
from controller.reading import build_read_plan
from controller.reduction import reduce_claims
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, verify_evidence_spans


class ScriptedReadingEngine(MockEngine):
    def __init__(self, readings):
        super().__init__()
        self.readings = iter(readings)
        self.constraints = []

    async def generate(self, *, response_schema=None, **kwargs):
        self.constraints.append((response_schema, kwargs.get("citation_ids")))
        self.response_text = json.dumps(next(self.readings)) if response_schema else "Supported final answer."
        return await super().generate(**kwargs)


def fact(span, entity="item", value="1", unit="", need_id="n"):
    return {"need_id": need_id, "text": entity, "entity": entity, "relation": "quantity",
            "value": value, "unit": unit, "conditions": [], "version": "v1",
            "span_ids": [span], "parent_claim_ids": []}


def read_output(*facts, support=("n",), open_needs=()):
    return {"claims": list(facts), "support": list(support), "conflicts": [],
            "open_needs": list(open_needs)}


def material(text, name="a"):
    block = make_evidence(text, name)
    span = SourceSpan(name, block.key, 0, len(text))
    unit = EvidenceUnit(name, (span,), text, body_tokens=len(text), need_ids=("n",))
    return block, unit


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


def test_reduction_aligns_entities_units_and_preserves_all_leaves():
    block, unit = material("A=1kg; A=1000g; B=500g")
    claims = tuple(EvidenceClaim(str(i), "n", "extract", entity=entity, value=value,
                    unit=measure, source_spans=unit.source_spans)
                   for i, (entity, value, measure) in enumerate(
                       (("A", "1", "kg"), ("a", "1000", "g"), ("B", "500", "g"))))
    total = reduce_claims("sum", claims, claim_id="total", need_id="n")
    count = reduce_claims("count", claims, claim_id="count", need_id="n")
    assert total.value == "1500" and total.unit == "g" and count.value == 2
    graph = ClaimGraph((*claims, total, count))
    assert len(graph.leaf_claims("total")) == 3
    assert graph.source_spans("total") == unit.source_spans


@pytest.mark.asyncio
async def test_full_coverage_reads_every_pack_and_count_runs_on_cpu(tmp_path):
    first, unit_a = material("Alpha\n"*120, "a")
    second, unit_b = material("Beta\n"*120, "b")
    state = TaskState("t", "Count all entities.", needs=(Need("n", operation="count"),))
    selected = EvidenceSet("s", (unit_a, unit_b), source_blocks=(first, second))
    budget = ContextBudget(read_input_tokens=7000, read_output_tokens=1024)
    plan = build_read_plan(state, selected, "all_supplied_documents", budget, template=ByteTemplate(),
                           read_packing="source_windows")
    assert len(plan.steps) == 2
    engine = ScriptedReadingEngine([read_output(fact("a", "Alpha")), read_output(fact("b", "Beta"))])
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, engine, ByteTemplate()).execute_reads(attempt.attempt_id, snapshot, plan)
    assert {row[2] for row in outcome.coverage} == {"a", "b"}
    assert outcome.claims[-1].value == 2
    assert len(outcome.steps) == 3 and len(engine.constraints) == 2
    assert outcome.plan.answer_plan.points[0].claim_ids == (outcome.claims[-1].claim_id,)
    assert store.get_snapshot("s").version == 0


@pytest.mark.asyncio
async def test_open_need_adds_another_read_and_claim_parent_chain(tmp_path):
    first, unit_a = material("A belongs to B.", "a")
    second, unit_b = material("B started in 2001.", "b")
    state = TaskState("t", "When did A's parent start?", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit_a,), source_blocks=(first, second)),
                           "selected_sources", ContextBudget(), template=ByteTemplate())
    gap = {"need_id": "n", "missing_relation": "start date", "bound_entities": ["B"],
           "temporal_scope": "current", "parent_claim_ids": ["read:0:claim:0"]}
    successor = fact("b", "B", "2001")
    successor["parent_claim_ids"] = ["read:0:claim:0"]
    engine = ScriptedReadingEngine([read_output(fact("a", "B"), support=(), open_needs=(gap,)),
                                   read_output(successor)])
    requested = []

    async def extend(plan, open_needs, claims):
        requested.extend(open_needs)
        return EvidencePool((unit_b,), auxiliary_usage={"input_tokens": 0,
            "output_tokens": 0, "compute_ms": 0.0, "models": []})

    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, engine, ByteTemplate(), extend_open_needs=extend).execute_reads(
        attempt.attempt_id, snapshot, plan)
    assert len(requested) == 1 and len(outcome.results) == 2
    assert outcome.results[1].covered_units == ("b",)
    assert outcome.steps[1].task.predecessor_step_ids == ("read:0",)
    constraint = outcome.steps[1].task.output_schema["properties"]["open_needs"]["items"]["properties"]["parent_claim_ids"]["items"]["pattern"]
    assert re.fullmatch(constraint, "read:0:claim:0")
    assert not re.fullmatch(constraint, "hop:1:read:0:claim:1")
    assert not re.fullmatch(constraint, "read:0:claim:1")
    assert not outcome.open_needs
    assert outcome.plan.answer_plan.points[0].claim_ids == (outcome.claims[-1].claim_id,)
    assert {span.span_id for span in outcome.plan.answer_plan.points[0].source_spans} == {"a", "b"}


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
async def test_online_residual_observation_includes_the_same_sunk_work_as_prediction(tmp_path):
    from controller.cost_model import CostModel
    from controller.schema import CostEstimate

    class NativeTimingEngine(ScriptedReadingEngine):
        async def info(self):
            return replace(await super().info(), engine_kind="sglang")
        async def generate(self, **kwargs):
            return replace(await super().generate(**kwargs), server_elapsed_ms=10.0)

    class RecordingCost(CostModel):
        observed = None
        def predict_native(self, candidate, quote, session=None, *, batch=None, concurrency=1,
                           control_ms=0.0, queue_ms=0.0):
            return CostEstimate(candidate.candidate_id,
                10+control_ms+candidate.predicted_read_ms+candidate.auxiliary_ms, calibrated=True)
        def observe(self, predicted, actual_elapsed_ms):
            self.observed = actual_elapsed_ms

    block, unit = material("A fact.")
    state = TaskState("t", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("set", (unit,), source_blocks=(block,)),
                           "selected_sources", ContextBudget(), template=ByteTemplate())
    plan = replace(plan, auxiliary_ms=(("task", 7.0),))
    model = RecordingCost()
    engine = NativeTimingEngine([read_output(fact("a"))])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine, cost_model=model,
        knobs=PlannerKnobs(strategy="cost"), online_calibration=True)
    response = await controller.serve_turn("s", state.raw_question, request_id="one", context_plans=(plan,),
        auxiliary_usage={"input_tokens": 2, "output_tokens": 3, "compute_ms": 7.0, "models": []})
    assert model.observed is not None
    assert model.observed-response.control_ms == pytest.approx(10+response.reading_ms+7)
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
                           read_input_tokens=4000, read_output_tokens=3000)
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


@pytest.mark.asyncio
async def test_lmap_uses_plain_summaries_and_model_merge_for_count_task(tmp_path):
    first, unit_a = material("Alpha is registered.", "a")
    second, unit_b = material("Beta is registered.", "b")
    state = TaskState("t", "Count all entities.", needs=(Need("n", operation="count"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit_a, unit_b), source_blocks=(first, second)),
                           "all_supplied_documents", ContextBudget(), template=ByteTemplate())
    plan = replace(plan, steps=tuple(replace(step, operation="summarize") for step in plan.steps))
    summary = fact("", "summary", "2")
    summary["span_ids"] = []
    summary["parent_claim_ids"] = ["read:0:claim:0", "read:0:claim:1"]
    engine = ScriptedReadingEngine([read_output(fact("a", "Alpha"), fact("b", "Beta")),
                                   read_output(summary)])
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, engine, ByteTemplate()).execute_reads(attempt.attempt_id, snapshot, plan)
    assert len(engine.constraints) == 2
    assert [step.task.operation for step in outcome.steps] == ["summarize", "summarize"]
    assert all("cpu_reduce_ms" not in step.metrics for step in outcome.steps)
    assert "Summarize the supplied document window" in outcome.steps[0].input_messages[0].content
    assert "Merge the supplied window summaries" in outcome.steps[1].input_messages[0].content
    assert len(outcome.plan.answer_plan.points[0].source_spans) == 2


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
@pytest.mark.parametrize("auxiliary_input,lost_read", [(27, False), (None, True)])
async def test_total_online_tokens_include_persisted_auxiliary_usage_once(tmp_path, auxiliary_input, lost_read):
    block, unit = material("A supported fact.")
    state = TaskState("t", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(max_total_tokens=1048576 if auxiliary_input is not None else None),
        template=ByteTemplate())
    usage = {"input_tokens": auxiliary_input, "output_tokens": 11, "compute_ms": 12.5,
             "models": [{"roles": ["task", "support"], "compute_ms": 12.5}]}
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    calls = 0

    async def provide(snapshot):
        nonlocal calls
        calls += 1
        return EvidenceSelection((block,), (block.key,), context_plans=(plan,), auxiliary_usage=usage)

    if lost_read:
        engine.drop_next_generate_response = True
        with pytest.raises(UnknownOutcomeError):
            await controller.serve_turn("s", state.raw_question, request_id="r", evidence_provider=provide)
        assert store.find_request("s", "r").auxiliary_usage == usage
    response = await controller.serve_turn("s", state.raw_question, request_id="r", evidence_provider=provide)
    assert calls == 1
    assert response.auxiliary_input_tokens == auxiliary_input
    assert response.auxiliary_output_tokens == 11 and response.auxiliary_model_ms == 12.5
    assert response.total_output_tokens == response.reader_output_tokens+11
    assert response.total_input_tokens == (response.reader_input_tokens+auxiliary_input
                                           if auxiliary_input is not None else None)
    replay = await controller.serve_turn("s", state.raw_question, request_id="r", evidence_provider=provide)
    assert replay.auxiliary_usage == response.auxiliary_usage and calls == 1
    assert store.get_snapshot("s").version == 1
    await controller.drain()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,model_calls", [("coverage_only", 1), ("claims", 2), ("typed_reduce", 1)])
async def test_reading_ablation_modes_execute_distinct_count_paths(tmp_path, mode, model_calls):
    first, unit_a = material("Alpha is approved.", "a")
    second, unit_b = material("Beta is approved.", "b")
    state = TaskState("t", "Count approved entities.", needs=(Need("n", operation="count"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit_a, unit_b), source_blocks=(first, second)),
        "all_supplied_documents", ContextBudget(), template=ByteTemplate(), reading_mode=mode)
    outputs = [read_output() if mode == "coverage_only" else read_output(fact("a", "Alpha"), fact("b", "Beta"))]
    if mode == "claims":
        count = fact("", "count", "2")
        count["span_ids"] = []
        count["parent_claim_ids"] = ["read:0:claim:0", "read:0:claim:1"]
        outputs.append(read_output(count))
    engine = ScriptedReadingEngine(outputs)
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, engine, ByteTemplate()).execute_reads(attempt.attempt_id, snapshot, plan)
    assert len(engine.constraints) == model_calls
    assert {row[2] for row in outcome.coverage} == {"a", "b"}
    if mode == "coverage_only":
        assert not outcome.claims and not outcome.plan.answer_plan.points[0].claim_ids
        assert engine.constraints[0][0]["properties"]["claims"]["maxItems"] == 0
    elif mode == "claims":
        assert outcome.claims[-1].value == "2"
        assert "cpu_reduce_ms" not in outcome.steps[-1].metrics
    else:
        assert outcome.claims[-1].value == 2
        assert "cpu_reduce_ms" in outcome.steps[-1].metrics


@pytest.mark.asyncio
async def test_every_table_read_carries_header_and_coverage_counts_header_once(tmp_path):
    class TableEngine(MockEngine):
        async def generate(self, *, response_schema=None, **kwargs):
            text = bytes(kwargs["input_ids"]).decode()
            payload = json.loads(text.split("<|user|>\n")[-1].split("\n<|end|>")[0])
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
