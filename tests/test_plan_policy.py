from policy_helpers import SpecifiedGainFixture, bind_fixture_timing
from dataclasses import replace


from controller.claim_graph import ClaimGraph
from controller.context_schema import EvidenceClaim, ReadResult, ReadTask, ReadingOutcome
from controller.cost_model import CostModel
from controller.plan_policy import (CompleteCost, NativePlanningState, PlanPolicy,
    ResourceForecast, StepForecast, complete_plan)
from controller.schema import CacheQuote, CacheQuoteBatch, CostEstimate, EvidenceSpan, Message, PromptCandidate, SessionSnapshot
from context_helpers import context_plan


class QuotedCostFixture(CostModel):
    def __init__(self, costs):
        super().__init__()
        self.costs = costs
        self.calls = []
    def predict_native(self, candidate, quote, *args, **kwargs):
        self.calls.append(candidate)
        return CostEstimate(candidate.candidate_id, self.costs[candidate.candidate_id], calibrated=True)


def prompt(name, *, action="REBUILD", output=64, kind="ANSWER", tokens=16):
    base, block = context_plan()
    source = base.selected_set.selected_units[0].source_spans[0]
    text = block.text[source.start:source.end]
    spans = (EvidenceSpan(source.source_key, 1, 0, len(text), block.source_uri, block.provenance,
                          source.span_id, source_start=source.start,
                          source_end=source.end),) if kind == "ANSWER" else ()
    return PromptCandidate(name, action, 0, (Message("system", kind+" fixture system"), Message("user", text)),
        (1,)*tokens, "hash:"+name, frozenset(), frozenset(), evidence_spans=spans,
        output_budget=output, generation_kind=kind)


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


def test_full_scope_completion_allows_a_budgeted_answer_from_a_large_read_archive():
    base, direct, scan = make_plans(scope=True)
    archive = replace(base.selected_set, token_cost=base.budget.body_tokens*4)
    base = replace(base, selected_set=archive)
    direct = replace(direct, context_plan=replace(direct.context_plan, selected_set=archive))
    direct = replace(direct, steps=tuple(replace(step, variants=tuple(
        replace(candidate, evidence_spans=()) for candidate in step.variants)) for step in direct.steps))
    scan = replace(scan, context_plan=replace(scan.context_plan, selected_set=archive))
    policy = PlanPolicy(SpecifiedGainFixture())
    decision = policy.choose((direct, scan), ReadingOutcome(base), native((direct, scan)), {})
    assert decision.next_step.plan_id == scan.plan_id
    assert next(row for row in policy.last_evaluations if row.plan_id == direct.plan_id).reason == "answer_evidence_incomplete"
    claim = EvidenceClaim("read:0:claim:0", "need", "extract", text="Complete supported conclusion",
        source_spans=base.selected_set.selected_units[0].source_spans)
    result = ReadResult("read:0", (claim,), covered_units=("u",), support=("need",))
    done_plan = replace(scan.context_plan, answer_plan=ClaimGraph((claim,)).answer_plan(
        base.task_state, results=(result,)))
    done = ReadingOutcome(done_plan, results=(result,), claims=(claim,))
    next_direct = policy.choose((direct,), done, native((direct,)), {})
    assert next_direct.next_step.task.kind == "ANSWER"
    assert policy.last_evaluations[0].cost.input_tokens == 2+len(next_direct.next_step.candidate.input_ids)


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
