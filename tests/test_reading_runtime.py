"""Reading protocol, provenance and one outer commit with scripted engine outputs."""
import json
import re
from dataclasses import asdict, replace

import httpx
import pytest

from bench.mock_engine import MockEngine
from controller.claim_graph import ClaimGraph
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceClaim, EvidencePool, EvidenceSet,
                                       EvidenceUnit, Need, ReadingOutcome, SourceSpan, TaskState)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor, ReadingBudgetExceeded, OnlineTokenBudgetUnmeasured
from controller.engine_client import SGLangEngineClient, SGLANG_COMMIT, UnknownOutcomeError
from controller.engine_client import EngineError
from controller.planner import PlannerKnobs
from controller.reading import build_read_plan
from controller.reduction import reduce_claims
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, verify_evidence_spans
from controller.templates import token_hash


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


@pytest.mark.asyncio
async def test_provider_without_auxiliary_measurements_keeps_online_totals_unknown(tmp_path):
    block, unit = material("A fact.")
    state = TaskState("t", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
                           "selected_sources", ContextBudget(max_total_tokens=None), template=ByteTemplate())
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))

    async def provider(snapshot):
        return EvidenceSelection((block,), (block.key,), context_plans=(plan,))

    response = await controller.serve_turn("s", state.raw_question, request_id="one",
                                           evidence_provider=provider)
    assert response.reader_input_tokens > 0 and response.reader_output_tokens > 0
    assert response.auxiliary_input_tokens is None and response.auxiliary_output_tokens is None
    assert response.total_input_tokens is None and response.total_output_tokens is None
    assert response.auxiliary_model_ms is None
    await controller.drain()


@pytest.mark.asyncio
@pytest.mark.parametrize("auxiliary_input", [20000, None])
async def test_finite_online_budget_accounts_for_auxiliary_before_first_read(tmp_path, auxiliary_input):
    block, unit = material("A supported fact.")
    state = TaskState("t", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(max_total_tokens=10000), template=ByteTemplate())
    engine = ScriptedReadingEngine([])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    usage = {"input_tokens": auxiliary_input, "output_tokens": 0, "compute_ms": 1.0, "models": []}
    error = OnlineTokenBudgetUnmeasured if auxiliary_input is None else ReadingBudgetExceeded
    with pytest.raises(error):
        await controller.serve_turn("s", state.raw_question, request_id="one",
            context_plans=(plan,), auxiliary_usage=usage)
    assert engine.constraints == []
    assert controller.store.get_snapshot("s").raw_history == ()
    assert controller.store.find_request("s", "one").auxiliary_usage == usage


@pytest.mark.asyncio
async def test_gap_auxiliary_spend_stops_followup_generation_and_retains_cost(tmp_path):
    block, unit = material("Alpha is governed by Beta.")
    next_block, next_unit = material("Beta belongs to Gamma.", "b")
    state = TaskState("t", "Which parent governs Alpha?", needs=(Need("n", operation="infer"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(max_total_tokens=20000), template=ByteTemplate())
    reading = read_output(fact("a", entity="Alpha"), support=(), open_needs=({"need_id": "n",
        "missing_relation": "Beta parent", "bound_entities": ["Beta"], "temporal_scope": [],
        "parent_claim_ids": ["read:0:claim:0"]},))
    engine = ScriptedReadingEngine([reading])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    usage = {"input_tokens": 100000, "output_tokens": 200, "compute_ms": 7.0, "models": []}
    async def gap(plan, needs, claims):
        return EvidencePool((next_unit,), (next_block,), auxiliary_usage=usage)
    with pytest.raises(ReadingBudgetExceeded):
        await controller.serve_turn("s", state.raw_question, request_id="one",
            context_plans=(plan,), open_need_provider=gap)
    attempt = store.find_request("s", "one")
    assert len(engine.constraints) == 1 and store.get_snapshot("s").raw_history == ()
    assert store.reading_progress(attempt.attempt_id).auxiliary_usage["input_tokens"] == 100000
    assert store.context_plan(attempt.attempt_id).online_tokens_spent > 100000


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


def test_numeric_need_isolation_uses_disjoint_steps_and_literal_quantity_values():
    from controller.reading import step_messages
    block, unit = material("Shipment R31 has mass 32000 g. Shipment R32 has mass 33 kg.")
    state = TaskState("t", "Count shipments and sum mass.", needs=(Need("count", operation="count"),
                      Need("mass", operation="sum")))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
        "all_supplied_documents", ContextBudget(), template=ByteTemplate(), separate_needs=True)
    assert len(plan.steps) == 2 and len({step.step_id for step in plan.steps}) == 2
    assert len({item.unit_id for item in plan.selected_set.selected_units}) == 2
    by_id = {item.unit_id: item for item in plan.selected_set.selected_units}
    for task in plan.steps:
        units = tuple(by_id[uid] for uid in task.input_unit_ids)
        payload = json.loads(step_messages(plan, task, units)[1].content)
        assert tuple(need["need_id"] for need in payload["task"]["needs"]) == task.need_ids
        values = task.output_schema["properties"]["claims"]["items"]["properties"]["value"]["enum"]
        if task.operation == "sum":
            assert "32000" in values and "320000" not in values and "33" in values
            assert set(task.output_schema["properties"]["claims"]["items"]["properties"]["unit"]["enum"]) == {"g", "kg"}
        else:
            assert task.operation == "count" and values == ["1"]
        assert units[0].source_spans[0].start == 0
        assert units[0].source_spans[0].end == len(block.text)


def test_structured_numeric_ids_come_from_original_body_and_versions_are_controller_owned():
    from controller.context_schema import ReadTask
    from controller.reading import output_schema, parse_read_result
    block = make_evidence("Shipment R31 has mass 32000 g.", "register-10", version="v4")
    unit = EvidenceUnit("register-10", (SourceSpan("row", block.key, 0, len(block.text)),), block.text)
    task = ReadTask("need:mass:read:0", "READ", ("n",), operation="sum")
    schema = output_schema(task, (unit,), numeric_identity="source_ids")
    properties = schema["properties"]["claims"]["items"]["properties"]
    assert properties["entity"]["enum"] == ["R31"]
    assert "version" not in properties
    assert schema["properties"]["open_needs"]["maxItems"] == 0
    value = fact("row", "R31", "32000", "g")
    del value["version"]
    value["followups"] = [{"need_id": "n", "missing_relation": "destination",
                            "bound_entities": ["R31"], "temporal_scope": ""}]
    data = {"format_version": 2, **read_output(value)}
    result = parse_read_result(task, json.dumps(data), (unit,))
    assert result.claims[0].version == "v4"
    assert result.open_needs[0].parent_claim_ids == ("need:mass:read:0:claim:0",)


def test_numeric_literals_keep_chinese_units_and_normalize_thousands_separators():
    from controller.context_schema import ReadTask
    from controller.reading import output_schema
    _, unit = material("甲记录金额为1,200万元。乙记录金额为30元。")
    schema = output_schema(ReadTask("read:0", "READ", ("n",), operation="sum"), (unit,))
    properties = schema["properties"]["claims"]["items"]["properties"]
    assert "1,200" in properties["value"]["enum"]
    assert set(properties["unit"]["enum"]) == {"万元", "元"}
    values = (EvidenceClaim("a", "n", "extract", entity="甲", value="1,200", unit="万元",
                           source_spans=unit.source_spans),
              EvidenceClaim("b", "n", "extract", entity="乙", value="30", unit="元",
                           source_spans=unit.source_spans))
    total = reduce_claims("sum", values, claim_id="total", need_id="n")
    assert total.value == "12000030" and total.unit == "CNY"


@pytest.mark.asyncio
async def test_full_coverage_reads_every_pack_and_count_runs_on_cpu(tmp_path):
    first, unit_a = material("Alpha\n"*120, "a")
    second, unit_b = material("Beta\n"*120, "b")
    state = TaskState("t", "Count all entities.", needs=(Need("n", operation="count"),))
    selected = EvidenceSet("s", (unit_a, unit_b), source_blocks=(first, second))
    budget = ContextBudget(read_input_tokens=3500, read_output_tokens=1024)
    plan = build_read_plan(state, selected, "all_supplied_documents", budget, template=ByteTemplate())
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


def test_oversized_source_is_split_with_exact_unicode_coordinates():
    block, unit = material("一二三四五\n"*400)
    state = TaskState("t", "List all", needs=(Need("n"),))
    selected = EvidenceSet("s", (unit,), source_blocks=(block,))
    plan = build_read_plan(state, selected, "all_supplied_documents",
        ContextBudget(read_input_tokens=3200), template=ByteTemplate())
    pieces = plan.selected_set.selected_units
    assert len(pieces) > 1
    assert "".join(piece.text for piece in pieces) == block.text
    assert len({span.span_id for piece in pieces for span in piece.source_spans}) == len(pieces)
    for piece in pieces:
        span = piece.source_spans[0]
        assert block.text[span.start:span.end] == piece.text


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
async def test_lost_read_response_recovers_native_result_then_commits_once(tmp_path):
    block, unit = material("A fact.")
    state = TaskState("t", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
                           "selected_sources", ContextBudget(), template=ByteTemplate())
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    engine.drop_next_generate_response = True
    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one")
    assert store.get_snapshot("s").version == 0
    assert store.pending_attempt("s").stage == "STARTED"
    response = await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one")
    assert len(engine.constraints) == 2
    assert response.read_count == 1 and store.get_snapshot("s").version == 1
    await controller.drain()


@pytest.mark.asyncio
async def test_gap_work_and_usage_survive_lost_successor_read(tmp_path):
    first, unit_a = material("A belongs to B.", "a")
    second, unit_b = material("B started in 2001.", "b")
    state = TaskState("t", "When did A's parent start?", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit_a,), source_blocks=(first,)),
                           "selected_sources", ContextBudget(), template=ByteTemplate())
    gap = {"need_id": "n", "missing_relation": "start date", "bound_entities": ["B"],
           "temporal_scope": "current", "parent_claim_ids": ["read:0:claim:0"]}
    successor = fact("b", "B", "2001")
    successor["parent_claim_ids"] = ["read:0:claim:0"]

    class LostSuccessor(ScriptedReadingEngine):
        lost = False
        async def generate(self, **kwargs):
            if ":hop:" in kwargs["attempt_id"] and not self.lost:
                self.lost = True
                self.drop_next_generate_response = True
            return await super().generate(**kwargs)

    engine = LostSuccessor([read_output(fact("a", "B"), support=(), open_needs=(gap,)),
                            read_output(successor)])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    calls = []

    async def extend(plan, needs, claims):
        calls.append(needs)
        return EvidencePool((unit_b,), (second,), auxiliary_ms=(("support", 7.0),),
            auxiliary_usage={"input_tokens": 3, "output_tokens": 5, "compute_ms": 7.0, "models": []})

    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one",
                                    open_need_provider=extend)
    pending = store.pending_attempt("s")
    assert store.reading_progress(pending.attempt_id).auxiliary_usage["input_tokens"] == 3
    initial_reading_ms = pending.reading_elapsed_ms
    assert initial_reading_ms > 0
    response = await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one",
                                           open_need_provider=extend)
    assert len(calls) == 1
    assert response.read_count == 2
    assert response.auxiliary_input_tokens == 3 and response.auxiliary_output_tokens == 5
    assert response.reading_ms > initial_reading_ms
    assert store.get_attempt(response.attempt_id).reading_elapsed_ms == response.reading_ms
    assert response.control_ms >= 0
    assert len(store.get_snapshot("s").raw_history) == 1
    assert store.get_snapshot("s").version == 1
    await controller.drain()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["no_steps", "coverage_only"])
async def test_direct_source_answer_plan_keeps_spans_and_binds_actual_citations(tmp_path, mode):
    from controller.context_schema import AnswerPlan, AnswerPoint, SupportBundle

    block, unit = material("Eligible applicants use Form L-9.")
    state = TaskState("t", "How do applicants apply?", needs=(Need("n"),))
    support = SupportBundle("direct", "n", (unit.unit_id,), 1, 1, unit.source_spans, "complete")
    selected = EvidenceSet("set", (unit,), need_support=(support,), source_blocks=(block,))
    points = AnswerPlan((AnswerPoint("n", source_spans=unit.source_spans),))
    if mode == "coverage_only":
        plan = build_read_plan(state, selected, "selected_sources", ContextBudget(),
                               template=ByteTemplate(), reading_mode="coverage_only")
        plan = replace(plan, answer_plan=points)
    else:
        plan = ContextPlan("p", state, selected, answer_plan=points)
    engine = ScriptedReadingEngine([read_output()] if mode == "coverage_only" else [])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    response = await controller.serve_turn("s", state.raw_question, request_id="one", context_plans=(plan,))
    point = controller.store.get_snapshot("s").raw_history[0].answer_plan.points[0]
    assert point.source_spans == unit.source_spans and not point.claim_ids
    assert point.citation_ids == response.citation_scope and point.citation_ids
    await controller.drain()


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
    assert len(point.citation_ids) == 1 and point.citation_ids == response.citation_scope
    final = controller.store.get_attempt(response.attempt_id).plan.candidate
    assert {dict(final.reference_map)[ref] for ref in point.citation_ids} == final.required_blocks
    assert len(final.required_blocks) < len({span.source_key for span in point.source_spans})
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
async def test_native_json_schema_and_citation_constraints_are_separate():
    fingerprint = ByteTemplate().fingerprint
    bodies = []

    def handler(request):
        if request.url.path == "/context/engine/info":
            return httpx.Response(200, json={"engine_restart_id": "r", "protocol_version": "contextkv-v1",
                "guarded_generate_supported": True, "scheduler_owned": True,
                "completion_status_supported": True, "sglang_commit": SGLANG_COMMIT,
                "fingerprint": asdict(fingerprint)})
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"text": "{}", "output_ids": [1], "meta_info": {
            "contextkv_receipt": {"rid": body["rid"], "attempt_id": body["rid"],
                "engine_session_id": "s", "session_generation": 1, "engine_restart_id": "r",
                "input_hash": token_hash((1, 2)), "finish_reason": "stop", "generation_complete": True}}})

    engine = SGLangEngineClient("http://engine", fingerprint,
        transport=httpx.MockTransport(handler), stream=False)
    kwargs = {"input_ids": (1, 2), "session_id": "s", "expected_generation": 1,
              "expected_restart_id": "r", "attempt_id": "step", "fingerprint": fingerprint}
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    await engine.generate(**kwargs, response_schema=schema)
    assert json.loads(bodies[0]["sampling_params"]["json_schema"]) == schema
    assert "regex" not in bodies[0]["sampling_params"]
    await engine.generate(**{**kwargs, "attempt_id": "answer"}, citation_ids=("E_one",))
    assert "regex" in bodies[1]["sampling_params"] and "json_schema" not in bodies[1]["sampling_params"]
    with pytest.raises(ValueError, match="one output constraint"):
        await engine.generate(**kwargs, response_schema=schema, citation_ids=())
    await engine.aclose()


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
        assert 0 < len(candidate.required_blocks) < 100
        assert {span.key for span in candidate.evidence_spans} == candidate.required_blocks
        verify_evidence_spans(candidate.messages, candidate.evidence_spans, blocks,
                              candidate.reference_map)
        content = candidate.messages[-1].content
        assert point.claim_ids[0] in content
        assert "parent_claim_ids" not in content
    assert len(outcome.plan.answer_plan.points[0].source_spans) == 100


@pytest.mark.asyncio
async def test_truncated_read_keeps_raw_result_without_coverage_or_user_commit(tmp_path):
    block, unit = material("Complete source text.")
    state = TaskState("t", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit,), source_blocks=(block,)),
                           "all_supplied_documents", ContextBudget(read_output_tokens=64),
                           template=ByteTemplate())
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    with pytest.raises(EngineError, match="READ_STEP_OUTPUT_TRUNCATED"):
        await controller.serve_turn("s", state.raw_question, context_plans=(plan,), request_id="one")
    attempt = store.find_request("s", "one")
    assert attempt.stage == "FAILED" and store.get_snapshot("s").version == 0
    step = store.execution_steps(attempt.attempt_id)[0]
    assert step.result.raw_output and not step.result.covered_units
    assert step.metrics["finish_reason"] == "length"
    await controller.drain()
    assert not store.outbox()


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
        assert candidate.feasible and len(candidate.required_blocks) < len(blocks)
        prompt = candidate.messages[-1].content
        assert all(value in prompt for value in ("Archivist", "1997", "Curator"))
        assert "read:0:claim:" not in prompt


@pytest.mark.asyncio
async def test_dense_rows_are_packed_for_output_capacity_before_reading(tmp_path):
    class DenseEngine(MockEngine):
        async def generate(self, *, response_schema=None, **kwargs):
            prompt = bytes(kwargs["input_ids"]).decode()
            payload = json.loads(prompt.split("<|user|>\n")[-1].split("\n<|end|>")[0])
            self.response_text = json.dumps(read_output(*(fact(unit["source_spans"][0]["span_id"],
                unit["unit_id"]) for unit in payload["units"])))
            return await super().generate(**kwargs)

    materialized = [material(f"Approved shipment S{index:02d}.", f"row{index}") for index in range(24)]
    blocks, units = zip(*materialized)
    units = tuple(replace(unit, table_id="shipments") for unit in units)
    state = TaskState("t", "Count all shipments.", needs=(Need("n", operation="count"),))
    selected = EvidenceSet("s", units, source_blocks=blocks)
    budget = ContextBudget(read_input_tokens=12288, read_output_tokens=1024)
    dense = build_read_plan(state, selected, "all_supplied_documents", budget, template=ByteTemplate())
    plain = build_read_plan(state, selected, "all_supplied_documents", budget,
                            template=ByteTemplate(), output_tokens_per_fact=None)
    assert len(dense.steps) > len(plain.steps)
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, DenseEngine(), ByteTemplate()).execute_reads(
        attempt.attempt_id, snapshot, dense)
    assert outcome.claims[-1].value == 24
    assert len(outcome.coverage) == 24
    assert all(step.metrics["finish_reason"] == "stop" for step in outcome.steps if step.task.kind == "READ")


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
@pytest.mark.parametrize("cap,expected_reads", [(2, 1), (None, 2)])
async def test_gap_expansion_counts_dependencies_against_cumulative_passage_cap(tmp_path, cap, expected_reads):
    first, unit_a = material("A belongs to B.", "a")
    second, unit_b = material("B is supported by C.", "b")
    third, unit_c = material("C started in 1997.", "c")
    unit_c = replace(unit_c, dependencies=(unit_b.unit_id,))
    state = TaskState("t", "When did A's sponsor start?", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("s", (unit_a,), source_blocks=(first,)),
        "selected_sources", ContextBudget(max_passages=cap), template=ByteTemplate())
    gap = {"need_id": "n", "missing_relation": "sponsor start", "bound_entities": ["B"],
           "temporal_scope": "current", "parent_claim_ids": ["read:0:claim:0"]}
    successor_b, successor_c = fact("b", "B"), fact("c", "C", "1997")
    successor_b["parent_claim_ids"] = successor_c["parent_claim_ids"] = ["read:0:claim:0"]
    outputs = [read_output(fact("a", "A"), support=(), open_needs=(gap,))]
    if cap is None:
        outputs.append(read_output(successor_b, successor_c))
    engine = ScriptedReadingEngine(outputs)
    usage = {"input_tokens": 10, "output_tokens": 3, "compute_ms": 5.0, "models": []}

    async def extend(plan, open_needs, claims):
        return EvidencePool((unit_c, unit_b), (second, third),
                            auxiliary_ms=(("retrieval", 5.0),), auxiliary_usage=usage)

    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, engine, ByteTemplate(), extend_open_needs=extend).execute_reads(
        attempt.attempt_id, snapshot, plan)
    assert len(engine.constraints) == expected_reads
    assert outcome.auxiliary_usage["input_tokens"] == 10
    assert sum(value for name, value in outcome.plan.auxiliary_ms if name.startswith("gap:")) == 5.0
    if cap is not None:
        assert len(outcome.claims) == 1 and len(outcome.plan.selected_set.source_blocks) == 1
        assert outcome.open_needs and outcome.plan.answer_plan.disposition == "partial"
    else:
        assert len(outcome.claims) == 3 and len(outcome.plan.selected_set.source_blocks) == 3
        assert not outcome.open_needs


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


@pytest.mark.asyncio
async def test_gap_read_repeats_old_dependency_and_updates_support_map(tmp_path):
    from controller.context_schema import SupportBundle
    header, header_unit = material("Entity | Foundation year", "header")
    first, unit_a = material("A names B as its parent.", "a")
    next_source, unit_b = material("B | 1997", "b")
    unit_b = replace(unit_b, dependencies=(header_unit.unit_id,))
    state = TaskState("t", "When was A's parent founded?", needs=(Need("n"),))
    initial_support = SupportBundle("initial", "n", ("a",), 1, .3, disposition="partial")
    plan = build_read_plan(state, EvidenceSet("s", (header_unit, unit_a),
        need_support=(initial_support,), source_blocks=(header, first)),
        "selected_sources", ContextBudget(), template=ByteTemplate())
    gap = {"need_id": "n", "missing_relation": "foundation year", "bound_entities": ["B"],
           "temporal_scope": "current", "parent_claim_ids": ["read:0:claim:0"]}
    successor = fact("b", "B", "1997")
    successor["parent_claim_ids"] = ["read:0:claim:0"]
    engine = ScriptedReadingEngine([read_output(fact("a", "A"), support=(), open_needs=(gap,)), read_output(successor)])
    complete = SupportBundle("expanded", "n", ("a", "b", "header"), 1, .95,
                             (*unit_a.source_spans, *unit_b.source_spans, *header_unit.source_spans), "complete")

    async def extend(plan, open_needs, claims):
        selected = EvidenceSet("expanded-set", (unit_b, header_unit), need_support=(complete,), quality_prediction=.95)
        return EvidencePool((unit_b,), (next_source,), selected_set=selected,
                            auxiliary_usage={"input_tokens": 0, "output_tokens": 0, "compute_ms": 0, "models": []})

    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", state.raw_question, 0)
    outcome = await ReadingExecutor(store, engine, ByteTemplate(), extend_open_needs=extend).execute_reads(
        attempt.attempt_id, snapshot, plan)
    assert "header" in outcome.steps[1].task.input_unit_ids
    assert outcome.plan.selected_set.need_support == (complete,)
    assert outcome.plan.selected_set.quality_prediction == .95
    assert len(outcome.coverage) == 3
    assert {span.span_id for span in outcome.plan.answer_plan.points[0].source_spans} == {"a", "b", "header"}
