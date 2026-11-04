"""CPU protocol fixtures exercise action collection; these are not GPU reward measurements."""
from action_helpers import Resources, material, frozen_fixture
from dataclasses import replace
import json

import pytest

from controller.context_schema import (
    EvidenceClaim, EvidencePool, EvidenceUnit, Need, ReadingOutcome,
    ReadResult, SourceSpan, StepRecord,
)
from controller.evidence import make_evidence
from controller.execution import ReadingBudgetExceeded, ReadingExecutor
from controller.plan_policy import CompleteCost
from controller.serialization import to_jsonable
from training.collect_plan_gains import collect_pair
from training.plan_gain_protocol import (
    GapRetrievalRequest, ZERO_AUX, build_observed_action_plans, observed_state_binding,
)
from training.plan_gain_rollout import GenerationBudget, execute_branch


@pytest.mark.asyncio
async def test_frozen_nonempty_state_runs_each_single_action_without_replaying_or_unselected_gap(tmp_path):
    service, native, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    assert {plan.semantic_kind for plan in plans} == {"direct_answer", "read_once", "gap_retrieval"}
    calls, runs = [], {}

    async def retrieve(plan, registered, claims):
        before = json.loads((tmp_path / "gap_retrieval" / "before.json").read_text())
        assert before["gap_request"] == to_jsonable(registered)
        assert registered == request and claims == frozen.claims
        assert {unit.unit_id for unit in plan.selected_set.selected_units} == {"R1"}
        assert {unit.unit_id for unit in frozen.plan.selected_set.selected_units} == {"R1", "R2"}
        assert plan.budget.max_passages == 2  # one completed passage plus one new passage
        calls.append(registered)
        block, unit = material("R3")
        return EvidencePool((unit,), (block,), auxiliary_usage={"input_tokens": 3,
            "output_tokens": 2, "compute_ms": 1, "models": []})

    for plan in plans:
        run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path / plan.semantic_kind,
            binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
            generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
        assert run["status"] == "completed" and run["qa_increment"] == 1
        assert frozen.steps[0].step_id not in run["new_step_ids"]
        assert run["cost"]["complete_ms"] == pytest.approx(500+run["actual_branch_execution_cost"]["complete_ms"])
        assert run["actual_branch_execution_cost"]["window"]["end"] <= run["cleanup_cost"]["window"]["end"]
        assert run["observed_state_binding"] == observed
        assert all(step["step_id"] in run["new_step_ids"] for step in run["steps"])
        expected = {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1}
        if plan.semantic_kind != "direct_answer":
            expected[{"read_once": "READ", "reduce_once": "REDUCE", "gap_retrieval": "GAP_RETRIEVAL"}[plan.semantic_kind]] = 1
        assert run["new_action_counts"] == expected
        runs[plan.semantic_kind] = run
    assert len(calls) == 1
    assert runs["direct_answer"]["response"]["read_count"] == 1
    assert runs["direct_answer"]["new_action_counts"]["READ"] == 0
    answers = [prompt for structured, prompt in native.inputs if not structured]
    assert len(answers) == 3 and all("Use the frozen condition" in prompt and "R1" in prompt for prompt in answers)
    assert "R3" in answers[-1]
    assert sum(structured for structured, _ in native.inputs) == 2  # one capture READ, one new READ
    direct = runs["direct_answer"]
    candidate = runs["read_once"]
    for run, quality in ((direct, .4), (candidate, .7)):
        run["scoring"] = {"status": "scored", "identity": binding["scorer_identity"]}
        run["metrics"] = {"Q": quality}
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    packet = {"plan_candidate": to_jsonable(plan), "working_state": to_jsonable(frozen),
        "snapshot": to_jsonable(snapshot), "metric": "Q", "normalization": {"scale": 1,
        "offset": 0, "protocol_source": "specified protocol fixture"},
        "scorer_identity": binding["scorer_identity"], "comparison_budget": binding["comparison_budget"]}
    record = collect_pair(packet, direct, candidate)
    assert record["delta_q"] == pytest.approx(.3)
    assert record["incremental_complete_cost"]["complete_ms"] == pytest.approx(
        candidate["actual_branch_execution_cost"]["complete_ms"]-direct["actual_branch_execution_cost"]["complete_ms"])
    assert record["direct_run"]["read_count"] == 0
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_empty_new_read_still_gets_a_final_answer_and_prior_paid_budget_is_enforced(tmp_path):
    service, native, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    native.empty_reads = True
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path / "empty-read",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request,
        gap_provider=None)
    assert run["status"] == "completed" and run["new_action_counts"]["READ"] == 1
    assert run["steps"][0]["result"]["claims"] == []
    assert run["new_action_counts"]["ANSWER"] == 1
    from training.plan_gain_protocol import FixedCounterfactualRuntime
    runtime = FixedCounterfactualRuntime(plan, initial_state=frozen,
        paid_cost=CompleteCost(500, 0, 0, 1000000, 1000000))
    service.store.read_snapshot("over-budget")
    attempt = service.store.begin_attempt("over-budget", "r", "hash", row["question"], 0)
    with pytest.raises(ReadingBudgetExceeded, match="online token allocation"):
        await runtime.run(attempt.attempt_id, snapshot, (plan.context_plan,),
            ReadingExecutor(service.store, service.engine, service.template), service.engine,
            service.template, None, None, arrived=0, auxiliary_usage=dict(ZERO_AUX),
            events=service.events)
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_material,existing_terminal", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("hard_steps", [2, 20])
async def test_bounded_infer_bridge_reads_terminal_once_and_empty_gap_stops(tmp_path, terminal_material, existing_terminal, hard_steps):
    from training.plan_gain_protocol import BOUNDED_CONTINUATION
    from controller.reading import build_read_plan
    service, native, budget, captured, snapshot, old, _, _, binding, row, prep = await frozen_fixture(tmp_path)
    task_state = replace(captured.plan.task_state, needs=(Need("n", operation="infer"),))
    selected = captured.plan.selected_set if existing_terminal else replace(captured.plan.selected_set,
        selected_units=captured.plan.selected_set.selected_units[:1])
    base = build_read_plan(task_state, selected, "selected_sources", captured.plan.budget,
        template=service.template, snapshot=snapshot, read_packing="source_windows")
    base = replace(base, budget=replace(base.budget, max_steps=hard_steps))
    frozen = ReadingOutcome(base, auxiliary_usage=dict(ZERO_AUX))
    observed = observed_state_binding(frozen, snapshot, paid_cost=CompleteCost(500, 0, 0, 7, 0),
        native_condition=old["native_condition"], reader_checkpoint=old["reader_checkpoint"],
        reader_profile_id=old["reader_profile_id"])
    plans, _ = build_observed_action_plans(frozen, snapshot, service.template,
        reader_checkpoint=old["reader_checkpoint"])
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    request = GapRetrievalRequest((), "", {"implementation": "specified-fixture"},
        {"max_units": 1, "max_passages": 1})
    calls = []

    async def retrieve(plan, request, claims):
        calls.append(request)
        assert request.open_needs and request.open_needs[0].missing_relation == "qualifying record"
        block = make_evidence("R3 qualifies. "*220, "R3")
        unit = EvidenceUnit("R3", (SourceSpan("R3", block.key, 0, len(block.text)),), block.text)
        return EvidencePool((unit,) if terminal_material else (), (block,) if terminal_material else (),
            auxiliary_usage=dict(ZERO_AUX))

    if existing_terminal:
        tickets = len(budget.tickets)
        unavailable = await execute_branch(service, plan, frozen, snapshot, row,
            case_dir=tmp_path/"insufficient-continuation-reserve", binding=binding, preparation=prep,
            resources=Resources(), cache_start="observed", generation_budget=GenerationBudget(3, tmp_path/"reserve.jsonl"),
            observed=observed, gap_request=request, gap_provider=retrieve)
        assert unavailable is None and len(budget.tickets) == tickets

    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path/"bridge",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
    assert run["status"] == "completed", run["failure"]
    assert len(calls) == (0 if existing_terminal or hard_steps == 2 else 1)
    assert run["new_action_counts"] == {"READ": 2 if terminal_material and hard_steps > 2 else 1,
        "REDUCE": 0, "GAP_RETRIEVAL": 0 if existing_terminal or hard_steps == 2 else 1, "ANSWER": 1}
    if hard_steps == 2:
        assert len(run["steps"]) == 1
        assert {claim["entity"] for claim in run["steps"][0]["result"]["claims"]} == {"R1"}
        assert not any(step["step_id"].startswith("bounded-terminal:") for step in run["steps"])
    if terminal_material and hard_steps > 2:
        assert run["steps"][-1]["task"]["input_claim_ids"]
        from training.plan_gain_completion import completion_record
        before = json.loads((tmp_path/"bridge"/"before.json").read_text())
        labelled = completion_record(before, run)
        assert 0 <= labelled["read_source_char_fraction"] <= 1
        assert labelled["read_original_characters"] == labelled["required_original_characters"]
    assert run["continuation_protocol"] == BOUNDED_CONTINUATION
    assert run["pre_action_quote"]["candidate_id"] is not None
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_gap_for_numeric_need_does_not_continue_an_unrelated_infer_need(tmp_path):
    from training.plan_gain_protocol import gap_query
    service, native, budget, frozen, snapshot, old, _, _, binding, row, prep = await frozen_fixture(tmp_path)
    task_state = replace(frozen.plan.task_state, needs=(*frozen.plan.task_state.needs,
                                                     Need("already-supported", operation="infer")))
    frozen = replace(frozen, plan=replace(frozen.plan, task_state=task_state),
        results=tuple(replace(result, support=("already-supported",)) for result in frozen.results))
    observed = observed_state_binding(frozen, snapshot, paid_cost=CompleteCost(500, 0, 0, 7, 0),
        native_condition=old["native_condition"], reader_checkpoint=old["reader_checkpoint"],
        reader_profile_id=old["reader_profile_id"])
    # Retain the already paid reader tokens in this branch's observed binding.
    observed["paid_cost"]["input_tokens"] += sum(len(step.input_ids) + len(step.output_ids)
                                                  for step in frozen.steps)
    request = GapRetrievalRequest(frozen.open_needs, gap_query(task_state, frozen.open_needs),
        {"implementation": "specified-fixture"}, {"max_units": 1, "max_passages": 1})
    plans, _ = build_observed_action_plans(frozen, snapshot, service.template,
        reader_checkpoint=old["reader_checkpoint"], gap_request=request)
    plan = next(plan for plan in plans if plan.semantic_kind == "gap_retrieval")

    async def retrieve(*args):
        block, unit = material("R3")
        return EvidencePool((unit,), (block,), auxiliary_usage=dict(ZERO_AUX))

    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path/"numeric-gap",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
    assert run["status"] == "completed", run["failure"]
    assert run["new_action_counts"] == {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 1, "ANSWER": 1}
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['numeric_select', 'sum'])
async def test_observed_numeric_actions_keep_factory_bound_shared_and_selected_basis(tmp_path, stage):
    from controller.action_proposals import observed_reduction_task
    from controller.plan_policy import complete_plan
    from controller.reading import build_read_plan, numeric_selection_task, parse_read_result
    from training.plan_gain_protocol import FixedCounterfactualRuntime

    service, native, _, captured, snapshot, observed, _, _, _, _, _ = await frozen_fixture(tmp_path)
    needs = (Need('sum', operation='sum'), Need('count', operation='count'))
    base = build_read_plan(replace(captured.plan.task_state, needs=needs),
        replace(captured.plan.selected_set, selected_units=captured.plan.selected_set.selected_units[:1]),
        'selected_sources', captured.plan.budget, template=service.template, snapshot=snapshot)
    read = base.steps[0]
    original = EvidenceClaim(read.step_id+':claim:0', 'count', 'extract',
        text='Inventory-13-029: 29 credits.', entity='Inventory-13-029', value='29', unit='credits',
        source_spans=base.selected_set.selected_units[0].source_spans)
    result = ReadResult(read.step_id, (original,), covered_units=read.input_unit_ids)
    frozen = ReadingOutcome(base, claims=(original,), results=(result,),
        steps=(StepRecord('capture', read.step_id, read, result=result),), auxiliary_usage=dict(ZERO_AUX))
    if stage == 'sum':
        # Both selectors have completed; CPU sum must consume selector-derived facts.
        for need_id in ('sum', 'count'):
            selector = numeric_selection_task(base, frozen.results, frozen.claims, need_id=need_id)
            selected = parse_read_result(selector, json.dumps({'format_version': 7,
                'views': {'n0': {'p0': True}}}), claims=(original,))
            frozen = replace(frozen, claims=(*frozen.claims, *selected.claims),
                results=(*frozen.results, selected),
                steps=(*frozen.steps, StepRecord('capture', selector.step_id, selector, result=selected)))
    task = observed_reduction_task(frozen)
    assert task.operation == stage
    expected = tuple(claim for claim in frozen.claims if claim.claim_id in task.input_claim_ids)
    assert expected and (expected[0].kind == 'derived' if stage == 'sum' else not original.serves('sum'))
    plan = complete_plan(replace(base, steps=(*(record.task for record in frozen.steps), task)), (),
        semantic_kind='reduce_once', reader_checkpoint=observed['reader_checkpoint'])

    class BasisObserved(Exception):
        pass

    class Executor:
        async def execute_one(self, attempt_id, snapshot, plan, actual, state, **kwargs):
            assert actual.input_claim_ids == task.input_claim_ids
            assert tuple(claim for claim in state.claims if claim.claim_id in actual.input_claim_ids) == expected
            raise BasisObserved

    runtime = FixedCounterfactualRuntime(plan, initial_state=frozen, paid_cost=CompleteCost(0, 0, 0, 0, 0))
    with pytest.raises(BasisObserved):
        await runtime.run('basis', snapshot, (plan.context_plan,), Executor(), native, service.template,
            None, None, arrived=0, auxiliary_usage=dict(ZERO_AUX), events=service.events)
    await service.aclose()
    service.store.close()
