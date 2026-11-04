"""Shared source facts feed independent numeric views through the real DAG."""
import json
from dataclasses import replace

import pytest

from bench.mock_engine import MockEngine
from controller.claim_graph import ClaimGraph
from controller.context_schema import (
    ContextBudget, EvidenceClaim, EvidenceSet, EvidenceUnit, Need, ReadResult, ReductionState,
    SourceSpan, TaskState,
)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor
from controller.reading import (
    advance_read_plan, build_read_plan, parse_read_result,
    step_messages,
)
from controller.reduction import reduce_claims
from controller.rolling_policy import _reduction_tasks
from controller.session_store import SessionStore
from controller.templates import ByteTemplate


def material(text, name="register", version="v1"):
    block = make_evidence(text, name, version=version)
    return block, EvidenceUnit(name, (SourceSpan(name, block.key, 0, len(text)),), text)


def numeric_plan(pairs, needs=None, **kwargs):
    blocks, units = zip(*pairs)
    state = TaskState("numeric", "Count shipments and sum their stated mass.", needs=needs or (
        Need("shipment-count", operation="count"), Need("shipment-mass", operation="sum")))
    return build_read_plan(state, EvidenceSet("selected", units, source_blocks=blocks),
        "all_supplied_documents", ContextBudget(window_tokens=32000, read_input_tokens=16000,
            read_output_tokens=8000), template=ByteTemplate(), **kwargs)


def row(entity, value, unit="", needs=("n0", "n1"), scope="g0", source="s0"):
    excerpt = f"{entity} mass {value} {unit}." if value is not None else f"{entity} mass unstated."
    return {"t": excerpt, "g": scope, "n": list(needs), "e": entity, "r": "mass", "v": value,
            "u": unit, "s": [source], "p": []}


def response(*facts):
    return {"format_version": 5, "facts": list(facts), "claims": [],
            "support": ["n0", "n1"], "conflicts": [], "open_needs": []}


class NumericEngine(MockEngine):
    def __init__(self, output):
        super().__init__()
        self.output, self.reads, self.selections = output, 0, 0

    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema["properties"]["format_version"]["const"] == 7:
            self.selections += 1
            # Each view keeps its eligible source facts, including missing/zero mass.
            views = response_schema["properties"]["views"]["properties"]
            rendered = bytes(kwargs["input_ids"]).decode()
            payload = json.loads(rendered.split("<|user|>\n")[-1].split("\n<|end|>")[0])
            self.response_text = json.dumps({"format_version": 7,
                "views": {alias: {parent: payload["needs"][alias]["operation"] == "count"
                    or payload["parents"][parent]["value"] is not None
                    for parent in value["properties"]} for alias, value in views.items()}})
        else:
            self.reads += 1
            self.response_text = json.dumps(self.output)
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_one_read_two_cpu_views_with_absent_and_zero_mass_and_durable_replay(tmp_path):
    pair = material("R1 mass 1 kg. R2 mass unstated. R3 mass 0 g.")
    plan = numeric_plan((pair,))
    engine = NumericEngine(response(row("R1", "1", "kg"),
        row("R2", None, needs=("n0",)), row("R3", "0", "g")))
    store = SessionStore(tmp_path / "state.db")
    snapshot = store.read_snapshot("session")
    attempt = store.begin_attempt("session", "request", "hash", plan.task_state.raw_question, 0)
    executor = ReadingExecutor(store, engine, ByteTemplate())
    outcome = await executor.execute_reads(attempt.attempt_id, snapshot, plan)
    extracts = tuple(claim for claim in outcome.claims if claim.kind == "extract")
    derived = {claim.operation: claim for claim in outcome.claims if claim.kind == "derived" and claim.operation in {"count", "sum"}}
    assert engine.reads == 1 and engine.selections == 2 and len(extracts) == 3
    assert derived["count"].value == 3
    assert (derived["sum"].value, derived["sum"].unit) == ("1000", "g")
    by_id = {claim.claim_id: claim for claim in outcome.claims}
    assert tuple(by_id[identity].parent_claim_ids[0] for identity in derived["count"].parent_claim_ids) == \
        tuple(claim.claim_id for claim in extracts)
    assert tuple(by_id[identity].parent_claim_ids[0] for identity in derived["sum"].parent_claim_ids) == \
        (extracts[0].claim_id, extracts[2].claim_id)
    assert extracts[0].shared_need_ids == ("shipment-mass",)
    graph = ClaimGraph(outcome.claims)
    assert graph.source_spans(derived["sum"].claim_id) == pair[1].source_spans
    assert [point.claim_ids for point in outcome.plan.answer_plan.points] == [
        (derived["count"].claim_id,), (derived["sum"].claim_id,)]
    replay = await executor.execute_reads(attempt.attempt_id, snapshot, outcome.plan)
    assert replay.claims == outcome.claims and engine.reads == 1 and engine.selections == 2
    assert len(store.execution_steps(attempt.attempt_id)) == 5


def test_count_reduction_does_not_consume_unfinished_sum_view_or_rolling_basis():
    plan = numeric_plan((material("R1 mass 500 g."),))
    task = plan.steps[0]
    result = parse_read_result(task, json.dumps(response(row("R1", "500", "g"))),
                               plan.selected_set.selected_units)
    tasks = advance_read_plan(plan, (result,))
    assert len(tasks) == 2 and tasks[0].input_claim_ids == tasks[1].input_claim_ids
    count = reduce_claims("count", result.claims, claim_id="count", need_id="shipment-count")
    done = ReadResult("count-step", (count,), reduction=ReductionState("count-step",
        ("shipment-count",), "count", tasks[0].input_claim_ids, ("count",), "merged"))
    answer = ClaimGraph((*result.claims, count)).answer_plan(plan.task_state, results=(result, done))
    assert answer.points[0].claim_ids == ("count",)
    assert answer.points[1].claim_ids == (result.claims[0].claim_id,)
    from controller.context_schema import ReadingOutcome
    observed = ReadingOutcome(plan, claims=result.claims, results=(result,))
    rolling = _reduction_tasks(plan, observed)
    assert len(rolling) == 2 and all(task.input_claim_ids == tasks[0].input_claim_ids for task in rolling)


def test_condition_scopes_and_source_versions_remain_separate():
    needs = tuple(Need(f"{condition}-{operation}", operation=operation, condition_ids=(condition,))
                  for condition in ("approved", "east") for operation in ("count", "sum"))
    plan = numeric_plan((material("R1 approved mass 1 kg.", "old", "v1"),
                         material("R1 east mass 2 kg.", "new", "v2")), needs)
    task = plan.steps[0]
    output = response(row("R1", "1", "kg"),
                      row("R1", "2", "kg", needs=("n2", "n3"), scope="g1", source="s1"))
    output["support"] = [f"n{index}" for index in range(len(task.need_ids))]
    result = parse_read_result(task, json.dumps(output), plan.selected_set.selected_units)
    assert [(fact.conditions, fact.version) for fact in result.claims] == [
        (("approved",), "v1"), (("east",), "v2")]
    reduced = {need.need_id: reduce_claims(need.operation, result.claims,
        claim_id=need.need_id, need_id=need.need_id) for need in needs}
    assert [reduced[need.need_id].value for need in needs] == [1, "1000", 1, "2000"]
    assert reduced["approved-sum"].source_spans == plan.selected_set.selected_units[0].source_spans
    assert reduced["east-sum"].source_spans == plan.selected_set.selected_units[1].source_spans
    output["facts"][0]["n"] = ["n0", "n3"]
    with pytest.raises(ValueError, match="population scope"):
        parse_read_result(task, json.dumps(output), plan.selected_set.selected_units)


def test_shared_fact_preserves_dependency_sources_and_predecessor_parent_proof():
    eligibility, prerequisite = material("Approved shipments qualify.", "eligibility", "v1")
    register, quantity = material("R1 mass 500 g.", "register", "v2")
    quantity = replace(quantity, dependencies=(prerequisite.unit_id,))
    parent = EvidenceClaim("eligible", "eligibility", "extract", entity="R1",
                           source_spans=prerequisite.source_spans)
    needs = tuple(Need(f"shipment-{operation}", operation=operation,
                       condition_ids=("approved",), predecessor_need_ids=("eligibility",))
                  for operation in ("count", "sum"))
    plan = numeric_plan(((register, quantity), (eligibility, prerequisite)), needs,
                        input_claims=(parent,))
    units = plan.selected_set.selected_units
    payload = json.loads(step_messages(plan, plan.steps[0], units, (parent,))[1].content)
    source_alias = next(alias for alias, identity in payload["fact_protocol"]["sources"].items()
                        if identity == quantity.source_spans[0].span_id)
    output = response(row("R1", "500", "g", source=source_alias))
    output["facts"][0]["p"] = ["p0"]
    result = parse_read_result(plan.steps[0], json.dumps(output), units, (parent,))
    assert result.claims[0].version == "v2"
    total = reduce_claims("sum", result.claims, claim_id="sum", need_id="shipment-sum")
    graph = ClaimGraph((parent, *result.claims, total))
    assert set(graph.source_spans("sum")) == {*quantity.source_spans, *prerequisite.source_spans}
    assert {claim.claim_id for claim in graph.leaf_claims("sum")} == {"eligible", result.claims[0].claim_id}


def test_numeric_selection_requires_last_parent_decision_and_preserves_rejected_background():
    from controller.reading import output_schema

    plan = numeric_plan((material('Inventory-3-035: 35 credits. Registration costs 24 credits.'),))
    task = replace(plan.steps[0], step_id='select', kind='REDUCE', operation='numeric_select')
    claims = (
        EvidenceClaim('background', 'shipment-count', 'extract',
            text='Registration costs 24 credits.', entity='Registration', value='24', unit='credits',
            shared_need_ids=('shipment-mass',)),
        EvidenceClaim('last-valid', 'shipment-count', 'extract',
            text='Inventory-3-035: 35 credits.', entity='Inventory-3-035', value='35', unit='credits',
            source_spans=plan.selected_set.selected_units[0].source_spans, shared_need_ids=()),
        EvidenceClaim('missing-quantity', 'shipment-count', 'extract',
            text='Inventory-3-036 quantity unstated.', entity='Inventory-3-036', value=None, unit=''),
    )
    schema = output_schema(task, claims=claims)
    assert schema['properties']['format_version']['const'] == 7
    for view in schema['properties']['views']['properties'].values():
        assert view['required'] == ['p0', 'p1', 'p2']
        assert view['properties']['p1'] == {'type': 'boolean'}
    assert schema['properties']['views']['properties']['n0']['properties']['p2'] == {'type': 'boolean'}
    assert schema['properties']['views']['properties']['n1']['properties']['p2'] == {
        'type': 'boolean', 'const': False}
    output = {'format_version': 7,
              'views': {'n0': {'p0': False, 'p1': True, 'p2': False},
                        'n1': {'p0': False, 'p1': True, 'p2': False}}}
    selected = parse_read_result(task, json.dumps(output), claims=claims)
    assert [claim.parent_claim_ids for claim in selected.claims] == [('last-valid',), ('last-valid',)]
    assert all(claim.text == claims[1].text for claim in selected.claims)
    assert not claims[1].serves('shipment-mass')
    assert selected.claims[1].need_id == 'shipment-mass'
    assert (selected.claims[1].value, selected.claims[1].unit, selected.claims[1].source_spans) == \
        (claims[1].value, claims[1].unit, claims[1].source_spans)
    del output['views']['n1']['p1']
    with pytest.raises(ValueError, match='every parent'):
        parse_read_result(task, json.dumps(output), claims=claims)
    historical = {'format_version': 6, 'views': {'n0': ['p1'], 'n1': ['p1']}}
    with pytest.raises(ValueError, match='protocol 7'):
        parse_read_result(task, json.dumps(historical), claims=claims)
