from dataclasses import replace

import pytest

from controller.context_schema import ReadResult, ReadTask, ReadingOutcome
from controller.cost_model import CostModel
from controller.plan_policy import (CompleteCost, NativePlanningState, PlanPolicy,
    PolicyLimits, ResourceForecast, StepForecast, complete_plan)
from controller.schema import CacheQuote, CacheQuoteBatch, CostEstimate, Message, PromptCandidate, SessionSnapshot
from test_context_planning import context_plan


class SpecifiedGainFixture:
    """Explicit unit-test gains, not a deployed or trained quality model."""
    reader_checkpoint = "reader-fixture"
    reader_profile_id = "profile-fixture"
    asset_id = "test-only-specified-gains"
    def predict(self, features):
        return .6 if features["read_steps"] else .1


class QuotedCostFixture(CostModel):
    def __init__(self, costs):
        super().__init__()
        self.costs = costs
        self.calls = []
    def predict_native(self, candidate, quote, *args, **kwargs):
        self.calls.append(candidate)
        return CostEstimate(candidate.candidate_id, self.costs[candidate.candidate_id], calibrated=True)


def prompt(name, *, action="REBUILD", output=64, kind="ANSWER", tokens=16):
    return PromptCandidate(name, action, 0, (Message("system", kind+" fixture system"),), (1,)*tokens, "hash:"+name,
        frozenset(), frozenset(), output_budget=output, generation_kind=kind)


def bind_fixture_timing(model, candidates):
    """Explicit protocol bindings for specified test costs, not real profiles."""
    for candidate in candidates:
        kind = candidate.generation_kind
        child = model if kind == "ANSWER" else model.step_models.setdefault(kind, CostModel())
        child.calibration_metadata = {"generation_kind": kind,
            "output_prior_system_prompt": candidate.messages[0].content,
            "step_control_scope": ["rolling_proposals_compilation_start"]}
        child.step_output_priors[f"{kind}:{candidate.output_budget}"] = {
            "mean_tokens": 1, "p90_tokens": 1, "mean_control_ms": 0,
            "mean_remaining_control_ms": 0}


def make_plans(*, two_reads=False, scope=False):
    base, _ = context_plan()
    base = replace(base, budget=replace(base.budget, lambda_time=.001, max_total_tokens=100000,
        output_tokens=64), coverage_scope="all_supplied_documents" if scope else "selected_sources",
        coverage_units=("u",) if scope else ())
    direct = complete_plan(replace(base, plan_id="direct", steps=()), (prompt("direct-answer"),),
        semantic_kind="direct_answer", reader_checkpoint="reader-fixture")
    read = ReadTask("read:0", "READ", ("need",), ("u",), input_budget=1000, output_budget=64)
    tasks = (read,)
    if two_reads:
        first_unit = base.selected_set.selected_units[0]
        second_unit = replace(first_unit, unit_id="second", source_spans=tuple(
            replace(span, span_id="span:second") for span in first_unit.source_spans))
        base = replace(base, selected_set=replace(base.selected_set,
            selected_units=(*base.selected_set.selected_units, second_unit)))
        tasks += (ReadTask("read:1", "READ", ("need",), ("second",), predecessor_step_ids=("read:0",),
                           input_budget=1000, output_budget=64),)
    plan = complete_plan(replace(base, plan_id="read-plan", steps=tasks), (),
        semantic_kind="scope_scan" if scope else "multihop" if two_reads else "complementary_evidence",
        reader_checkpoint="reader-fixture", step_variants={"read:0": (prompt("read0", kind="READ"),),
            "read:1": (prompt("read1", kind="READ"),)}, unmet_answer_points=("need",))
    return base, direct, plan


def native(plans, *, latency=None, future_latency=10, paid=CompleteCost(5, 10, 2, 2, 3), prefixes=None):
    candidates = [candidate for plan in plans for step in plan.steps for candidate in step.variants]
    batch = CacheQuoteBatch("q", 9, 1, "engine", "cache", 100000, 100000,
        tuple(CacheQuote(c.candidate_id, c.input_hash, len(c.input_ids), len(c.input_ids)-1) for c in candidates),
        fingerprint=None, active_requests=0)
    resources = {c.candidate_id: ResourceForecast(100, 20, 2, 1, "resource-fixture", 9,
        tuple((prefixes or {}).get(c.candidate_id, ())), True) for c in candidates}
    future = {(plan.plan_id, step.task.step_id): (StepForecast("REBUILD", 64,
        CompleteCost(future_latency, 100, 20, 16, 64, 2, 1), "future-fixture", True),)
        for plan in plans for step in plan.steps}
    model = QuotedCostFixture({c.candidate_id: (latency or {}).get(c.candidate_id, 10) for c in candidates})
    bind_fixture_timing(model, candidates)
    return NativePlanningState("reader-fixture", "profile-fixture", SessionSnapshot("s"),
        batch, model, resources, future, paid)


def test_joint_complete_plan_gain_selects_complementary_read_and_ignores_support_quality():
    base, direct, complementary = make_plans()
    direct = replace(direct, context_plan=replace(direct.context_plan, selected_set=replace(
        direct.context_plan.selected_set, quality_prediction=.99)))
    complementary = replace(complementary, context_plan=replace(complementary.context_plan, selected_set=replace(
        complementary.context_plan.selected_set, quality_prediction=.01)))
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((direct, complementary), ReadingOutcome(base), native((direct, complementary)), {})
    step, action, output = decision
    assert step.plan_id == complementary.plan_id and step.task.kind == "READ"
    assert action == "REBUILD" and output == 64
    selected = [row for row in policy.last_evaluations if row.plan_id == complementary.plan_id][0]
    assert selected.predicted_delta_q == .6
    assert selected.cost == CompleteCost(25, 210, 42, 34, 131, 2, 1)
    assert len(selected.lookahead) == 2


def test_complete_tail_cost_changes_choice_even_when_next_read_is_cheap():
    base, direct, multihop = make_plans(two_reads=True)
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((direct, multihop), ReadingOutcome(base),
        native((direct, multihop), latency={"read0": 1}, future_latency=400), {})
    assert decision.next_step.plan_id == direct.plan_id
    evaluated = next(row for row in policy.last_evaluations if row.plan_id == multihop.plan_id)
    assert evaluated.cost.latency_ms == 806
    assert len(evaluated.lookahead) == 2 and len(evaluated.forecast_sources) == 3


def test_two_step_lookahead_compares_joint_output_budget_choices():
    class BudgetGain(SpecifiedGainFixture):
        def predict(self, features):
            return features["output_budget_tokens"]*.002
    base, direct, read = make_plans(two_reads=True)
    first = read.steps[0]
    read = replace(read, steps=(replace(first, variants=(prompt("small", output=32, kind="READ"),
        prompt("large", output=64, kind="READ"))), replace(read.steps[1],
        task=replace(read.steps[1].task, output_budget=128)), read.steps[2]))
    machine = native((direct, read), latency={"small": 5, "large": 10})
    machine.future[read.plan_id, read.steps[1].task.step_id] = (
        StepForecast("REBUILD", 32, CompleteCost(10, 100, 20, 16, 32), "small-answer", True),
        StepForecast("REBUILD", 128, CompleteCost(20, 100, 20, 16, 128), "large-answer", True))
    policy = PlanPolicy(BudgetGain())
    decision = policy.choose((direct, read), ReadingOutcome(base), machine, {})
    assert decision.output_budget == 64
    choices = [row for row in policy.last_evaluations if row.plan_id == read.plan_id]
    assert len(choices) == 4
    assert max(choices, key=lambda row: row.objective).lookahead[-1][-1] == 128


def test_full_scope_is_a_hard_constraint_and_read_completion_allows_direct_answer():
    base, direct, scan = make_plans(scope=True)
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((direct, scan), ReadingOutcome(base), native((direct, scan)), {})
    assert decision.next_step.plan_id == scan.plan_id
    assert next(row for row in policy.last_evaluations if row.plan_id == direct.plan_id).reason == "required_scope_uncovered"
    done = ReadingOutcome(scan.context_plan, results=(ReadResult("read:0", covered_units=("u",)),))
    next_direct = policy.choose((direct,), done, native((direct,)), {})
    assert next_direct.next_step.task.kind == "ANSWER"


def test_rolls_forward_after_real_read_result_and_requotes_the_next_step():
    base, direct, plan = make_plans(two_reads=True)
    policy = PlanPolicy(SpecifiedGainFixture())
    first = policy.choose((plan,), ReadingOutcome(base), native((plan,)), {})
    assert first.next_step.task.step_id == "read:0"
    progress = ReadingOutcome(plan.context_plan, results=(ReadResult("read:0", covered_units=("u",)),))
    second = policy.choose((plan,), progress, native((plan,)), {})
    assert second.next_step.task.step_id == "read:1"
    assert second.next_step.candidate.candidate_id == "read1"
    finished = replace(progress, results=(*progress.results, ReadResult("read:1", covered_units=("second",))))
    ready_answer = replace(plan, steps=(*plan.steps[:-1], replace(plan.steps[-1], variants=(prompt("final-answer"),))))
    answer = policy.choose((ready_answer,), finished, native((ready_answer,)), {})
    assert answer.next_step.task.kind == "ANSWER"


def test_repeated_read_id_does_not_skip_different_original_source_content():
    base, _, plan = make_plans()
    old_unit = plan.context_plan.selected_set.selected_units[0]
    changed_unit = replace(old_unit, text="a different original source", source_spans=tuple(
        replace(span, source_key=replace(span.source_key, version="different-version"))
        for span in old_unit.source_spans))
    new_plan = replace(plan, context_plan=replace(plan.context_plan, selected_set=replace(
        plan.context_plan.selected_set, selected_units=(changed_unit,))))
    progress = ReadingOutcome(plan.context_plan, results=(ReadResult("read:0", covered_units=("u",)),))
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((new_plan,), progress, native((new_plan,)), {})
    assert decision.next_step.task.step_id == "read:0"


@pytest.mark.parametrize("changed", ({"read_layout": "source_first"}, {"read_packing": "source_windows"}))
def test_completed_read_requires_the_same_layout_and_packing(changed):
    _, _, plan = make_plans()
    progress = ReadingOutcome(plan.context_plan, results=(ReadResult("read:0", covered_units=("u",)),))
    changed_plan = replace(plan, context_plan=replace(plan.context_plan, **changed))
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((changed_plan,), progress, native((changed_plan,)), {})
    assert decision.next_step.task.kind == "READ"


@pytest.mark.parametrize("paid,limits,reason", (
    (CompleteCost(0, 999, 0, 0, 0), PolicyLimits(max_gpu_gib_ms=1000), "gpu_integral_budget"),
    (CompleteCost(0, 0, 999, 0, 0), PolicyLimits(max_host_gib_ms=1000), "host_integral_budget"),
    (CompleteCost(0, 0, 0, 99999, 0), PolicyLimits(), "cumulative_token_budget"),
))
def test_paid_cost_and_complete_plan_reservations_obey_hard_budgets(paid, limits, reason):
    base, direct, read = make_plans()
    policy = PlanPolicy(SpecifiedGainFixture(), limits=limits)
    with pytest.raises(ValueError, match="no complete plan"):
        policy.choose((read,), ReadingOutcome(base), native((read,), paid=paid), {})
    assert policy.last_evaluations[0].reason == reason


def test_native_prefixes_choose_context_action_without_a_second_saving_credit():
    base, direct, read = make_plans()
    first = replace(read.steps[0], variants=(prompt("cold", kind="READ"), prompt("warm", kind="READ", action="APPEND")))
    read = replace(read, steps=(first, read.steps[1]))
    prefixes = {"warm": ("stable-source-owner",)}
    machine = native((read,), latency={"cold": 30, "warm": 5}, prefixes=prefixes)
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((read,), ReadingOutcome(base), machine, prefixes)
    assert decision.context_action == "APPEND"
    assert decision.next_step.reusable_prefix_ids == ("stable-source-owner",)
    assert min(row.cost.latency_ms for row in policy.last_evaluations) == 20
    assert all(candidate.predicted_read_ms == candidate.auxiliary_ms == 0 for candidate in machine.cost_model.calls)


def test_policy_requires_reader_bound_gain_asset_and_rejects_stale_native_quotes():
    with pytest.raises(ValueError, match="not enabled"):
        PlanPolicy(None)
    base, direct, _ = make_plans()
    policy = PlanPolicy(SpecifiedGainFixture())
    with pytest.raises(ValueError, match="different reader"):
        policy.choose((direct,), ReadingOutcome(base), replace(native((direct,)), reader_checkpoint="reader-updated"), {})
    machine = native((direct,))
    machine.resources["direct-answer"] = replace(machine.resources["direct-answer"], quote_seq=8)
    with pytest.raises(ValueError, match="different observations"):
        policy.choose((direct,), ReadingOutcome(base), machine, {})


def test_later_ready_step_uses_future_profile_not_current_quote_cache_savings():
    base, direct, plan = make_plans(two_reads=True)
    second = replace(plan.steps[1], task=replace(plan.steps[1].task, predecessor_step_ids=()))
    plan = replace(plan, steps=(plan.steps[0], second, plan.steps[-1]))
    machine = native((direct, plan), latency={"read0": 1, "read1": 1}, future_latency=400)
    policy = PlanPolicy(SpecifiedGainFixture())
    selected = policy.choose((direct, plan), ReadingOutcome(base), machine, {})
    assert selected.next_step.plan_id == direct.plan_id
    row = next(row for row in policy.last_evaluations if row.plan_id == plan.plan_id)
    assert row.cost.latency_ms == 806
    assert all(call.candidate_id != "read1" for call in machine.cost_model.calls)
