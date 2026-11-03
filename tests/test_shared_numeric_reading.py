"""Shared source facts feed independent numeric views through the real DAG."""
import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator

from bench.mock_engine import MockEngine
from controller.claim_graph import ClaimGraph
from controller.context_schema import (
    ContextBudget, EvidenceClaim, EvidenceSet, EvidenceUnit, Need, ReadResult, ReductionState,
    SourceSpan, TaskState, TextRef,
)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor
from controller.reading import (
    advance_read_plan, build_read_plan, numeric_read_scopes, parse_read_result,
    read_output_reservation, stable_prefix_messages, step_messages, visible_output_schema,
)
from controller.reduction import reduce_claims
from controller.rolling_policy import _reduction_tasks, namespace_plan
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
    return {"g": scope, "n": list(needs), "e": entity, "r": "mass", "v": value,
            "u": unit, "s": [source], "p": []}


def response(*facts):
    return {"format_version": 3, "facts": list(facts), "claims": [],
            "support": ["shipment-count", "shipment-mass"], "conflicts": [], "open_needs": []}


def test_reader_sees_the_field_types_and_legal_values_used_by_its_decoder():
    plan = numeric_plan((material("R1 mass 1 kg. R2 mass unstated."),))
    task = plan.steps[0]
    message = step_messages(plan, task, plan.selected_set.selected_units)[-1]
    payload = json.loads(message.content)
    assert payload["output_schema"] == visible_output_schema(task.output_schema)
    branches = payload["output_schema"]["properties"]["facts"]["items"]["anyOf"]
    assert any(branch["properties"]["v"].get("enum") == ["1"] for branch in branches)
    assert task.output_schema["properties"]["facts"]["items"]["anyOf"][0]["required"]
    assert "required" not in branches[0]


class NumericEngine(MockEngine):
    def __init__(self, output):
        super().__init__()
        self.output, self.reads = output, 0

    async def generate(self, *, response_schema=None, **kwargs):
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
    derived = {claim.operation: claim for claim in outcome.claims if claim.kind == "derived"}
    assert engine.reads == 1 and len(extracts) == 3
    assert derived["count"].value == 3
    assert (derived["sum"].value, derived["sum"].unit) == ("1000", "g")
    assert derived["count"].parent_claim_ids == tuple(claim.claim_id for claim in extracts)
    assert derived["sum"].parent_claim_ids == (extracts[0].claim_id, extracts[2].claim_id)
    assert extracts[0].shared_need_ids == ("shipment-mass",)
    graph = ClaimGraph(outcome.claims)
    assert graph.source_spans(derived["sum"].claim_id) == pair[1].source_spans
    assert [point.claim_ids for point in outcome.plan.answer_plan.points] == [
        (derived["count"].claim_id,), (derived["sum"].claim_id,)]
    replay = await executor.execute_reads(attempt.attempt_id, snapshot, outcome.plan)
    assert replay.claims == outcome.claims and engine.reads == 1
    assert len(store.execution_steps(attempt.attempt_id)) == 3


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
    output["support"] = list(task.need_ids)
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


@pytest.mark.parametrize("change", [
    {"subject_refs": (TextRef(1, "user", 0, 5),)},
    {"condition_ids": ("approved",)},
    {"predecessor_need_ids": ("eligible",)},
    {"scope": "all_supplied_documents"},
])
def test_different_populations_do_not_share_numeric_scope(change):
    pair = material("R1 mass 1 kg.")
    count = Need("count", operation="count")
    total = replace(Need("sum", operation="sum"), **change)
    state = TaskState("task", "Count all shipments; sum eligible masses.", needs=(count, total))
    assert numeric_read_scopes(state, (pair[1],)) == ()


def test_different_source_assignments_do_not_share_numeric_scope():
    _, unit = material("R1 mass 1 kg.")
    state = TaskState("task", "Count and sum.", needs=(Need("count", operation="count"),
                                                     Need("sum", operation="sum")))
    assert numeric_read_scopes(state, (replace(unit, need_ids=("count",)),)) == ()


def test_same_entity_in_distinct_versions_keeps_two_source_proofs():
    plan = numeric_plan((material("R1 mass 1 kg.", "old", "v1"),
                         material("R1 mass 2 kg.", "new", "v2")))
    result = parse_read_result(plan.steps[0], json.dumps(response(row("R1", "1", "kg"),
        row("R1", "2", "kg", source="s1"))), plan.selected_set.selected_units)
    total = reduce_claims("sum", result.claims, claim_id="sum", need_id="shipment-mass")
    assert total.value == "3000" and len(total.source_spans) == 2
    count = reduce_claims("count", result.claims, claim_id="count", need_id="shipment-count")
    assert count.value == 2


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


@pytest.mark.parametrize("fact,valid", [
    (row("R1", "1", "kg"), True),
    (row("R3", "0", "g"), True),
    (row("R2", None, needs=("n0",)), True),
    (row("R2", None, needs=("n1",)), False),
    (row("R2", None), False),
    (row("R2", None, "kg", needs=("n0",)), False),
    (row("R1", "99", "kg"), False),
    (row("R1", 1, "kg"), False),
])
def test_schema_types_quantity_views_and_entity_only_count_views(fact, valid):
    plan = numeric_plan((material("R1 mass 1 kg. R2 mass unstated. R3 mass 0 g."),))
    schema = plan.steps[0].output_schema
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(response(fact)) is valid
    scope = plan.steps[0].shared_numeric_scopes[0]
    assert scope.operations == ("count", "sum")


def test_schema_binds_scope_and_operation_memberships_together():
    needs = tuple(Need(f"{condition}-{operation}", operation=operation, condition_ids=(condition,))
                  for condition in ("approved", "east") for operation in ("count", "sum"))
    plan = numeric_plan((material("R1 approved mass 1 kg. R2 east mass unstated."),), needs)
    schema = plan.steps[0].output_schema
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    def valid(fact):
        data = response(fact)
        data["support"] = [need.need_id for need in needs]
        return validator.is_valid(data)

    assert valid(row("R1", "1", "kg"))
    assert valid(row("R2", None, scope="g1", needs=("n2",)))
    assert not valid(row("R1", "1", "kg", needs=("n0", "n3")))
    assert not valid(row("R1", "1", "kg", scope="g1", needs=("n0", "n1")))
    assert not valid(row("R2", None, scope="g1", needs=("n3",)))


def test_schema_without_quantity_literals_allows_only_entity_count_rows():
    plan = numeric_plan((material("R1 qualifies. R2 has unstated mass."),))
    schema = plan.steps[0].output_schema
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert validator.is_valid(response(row("R1", None, needs=("n0",))))
    assert not validator.is_valid(response(row("R1", None, needs=("n1",))))
    assert not validator.is_valid(response(row("R1", "1", "kg")))


def test_shared_read_measures_less_actual_prompt_and_output_than_isolation():
    text = "\n".join(f"Shipment R{i} is approved for the northern route and has mass {i} kg."
                     for i in range(1, 51))
    pair = material(text)
    shared = numeric_plan((pair,), read_layout="source_first", output_tokens_per_fact=None)
    isolated = numeric_plan((pair,), read_layout="source_first", output_tokens_per_fact=None,
                            separate_needs=True)
    template = ByteTemplate()

    def measure(plan):
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        source_tokens = prompt_tokens = 0
        for task in plan.steps:
            selected = tuple(units[identity] for identity in task.input_unit_ids)
            prompt_tokens += len(template.render(step_messages(plan, task, selected)).input_ids)
            source_tokens += sum(len(template.tokenizer.encode(unit.text)) for unit in selected)
        return source_tokens, prompt_tokens

    shared_source, shared_prompt = measure(shared)
    isolated_source, isolated_prompt = measure(isolated)
    assert isolated_source == 2 * shared_source and shared_prompt < isolated_prompt
    shared_rows = response(*(row(f"R{i}", str(i), "kg") for i in range(1, 51)))
    shared_output = len(template.tokenizer.encode(json.dumps(shared_rows)))
    old_rows = [{"need_id": need.need_id, "text": f"R{i}", "entity": f"R{i}",
                 "relation": "mass", "value": str(i) if need.operation == "sum" else "1",
                 "unit": "kg" if need.operation == "sum" else "", "conditions": [],
                 "span_ids": ["need:0:register"], "parent_claim_ids": [], "followups": []}
                for need in shared.task_state.needs for i in range(1, 51)]
    isolated_output = len(template.tokenizer.encode(json.dumps({"format_version": 2,
        "claims": old_rows, "support": list(shared.steps[0].need_ids), "conflicts": [], "open_needs": []})))
    assert shared_output < isolated_output
    reservation = read_output_reservation(shared.task_state, shared.steps[0],
        shared.selected_set.selected_units, template.tokenizer)
    isolated_reservation = sum(read_output_reservation(isolated.task_state, task,
        tuple(unit for unit in isolated.selected_set.selected_units if unit.unit_id in task.input_unit_ids),
        template.tokenizer) for task in isolated.steps)
    assert reservation < isolated_reservation
    assert shared.steps[0].output_schema["properties"]["format_version"]["const"] == 3
    print({"source_tokens": [isolated_source, shared_source],
           "prompt_tokens": [isolated_prompt, shared_prompt],
           "fact_output_tokens": [isolated_output, shared_output],
           "output_reservation": [isolated_reservation, reservation]})
    # Source-first causal content remains independent of the operation view aliases.
    renamed = replace(shared.steps[0], need_ids=("renamed-count", "renamed-sum"))
    assert stable_prefix_messages(shared, shared.steps[0], shared.selected_set.selected_units) == \
        stable_prefix_messages(shared, renamed, shared.selected_set.selected_units)
    assert namespace_plan(shared).steps[0].shared_numeric_scopes == shared.steps[0].shared_numeric_scopes
