"""Actual CPU Controller execution and Native quotes, not model/performance evidence."""
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from adapters.contextpilot import SourceOrderIndex
from bench.mock_engine import MockEngine
from controller.action_policy import route_completion_reserve
from controller.action_proposals import BOUNDED_CONTINUATION
from controller.context_budget import COMPLETION_RESERVE_SCOPE
from controller.context_schema import ContextBudget, EvidenceSet, Need, ReadingOutcome, TaskState
from controller.cost_model import CostModel
from controller.demand import DemandState, JointOption, joint_options
from controller.events import EventLog
from controller.plan_policy import (CompleteCost, NATIVE_COST_SCHEMA, OBSERVED_FEATURE_NAMES,
    OBSERVED_FEATURE_SCHEMA, PlanPolicy, PlanStep, complete_plan)
from controller.plan_profiles import ObservedNativeObserver
from controller.reading import build_read_plan
from controller.request_compiler import RequestCompiler
from controller.rolling_policy import RollingPlanRuntime, compile_route_steps, compile_step
from controller.schema import SessionSnapshot
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from reading_helpers import fact, material, read_output


class RecordedPaid:
    def __init__(self):
        self.states = []
        self.last_observation = {}

    async def start(self):
        pass

    async def aclose(self):
        pass

    async def observe(self, arrived, state, usage):
        self.states.append(state)
        self.last_observation = {'end': time.monotonic(), 'basis': 'CPU test ledger'}
        return CompleteCost((time.monotonic()-arrived)*1000, 0, 0,
            sum(len(record.input_ids) for record in state.steps),
            sum(len(record.output_ids) for record in state.steps))


class DemandProjectionProbe:
    settings = {'mode': 'demand'}
    observe_reads = False

    def __init__(self):
        self.options = []
        self.finished = None

    async def demand_options(self, plans, state, session_id, batch, **kwargs):
        options = joint_options(plans, state)
        self.options.append((plans, state, options))
        return options

    async def finish_demand(self, outcome, *args):
        self.finished = outcome

    async def flush_demand_releases(self, *args, **kwargs):
        pass


class SourceReader(MockEngine):
    def __init__(self, store):
        super().__init__(page_size=16)
        self.store, self.catalogs, self.read_sources = store, [], []

    async def quote_batch(self, candidates, fingerprint):
        batch = await super().quote_batch(candidates, fingerprint)
        return replace(batch, native_max_new_tokens_clip=100000, active_requests=0, queued_requests=0)

    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema is not None:
            payload = json.loads(bytes(kwargs['input_ids']).decode().split('\nTask:\n')[-1].split('\n<|end|>')[0])
            sources = tuple(payload['fact_protocol']['sources'].values())
            self.read_sources.append(sources)
            attempt = self.store.find_request('session', 'run')
            self.catalogs.append(self.store.context_plan(attempt.attempt_id))
            self.response_text = json.dumps(read_output(*(fact(source, source) for source in sources)))
        else:
            self.response_text = 'Supported answer.'
        return await super().generate(**kwargs)


def source_plan(template, *, scope='all_supplied_documents'):
    blocks, units = zip(*(material('Material '+name+'. '*150, name) for name in ('a', 'c', 'b', 'd')))
    return build_read_plan(TaskState('t', 'Describe all sources', needs=(Need('n', operation='procedure', scope=scope),)),
        EvidenceSet('s', units, source_blocks=blocks), scope,
        ContextBudget(body_tokens=200, read_input_tokens=6600, read_output_tokens=1000),
        template=template, read_layout='source_first', read_packing='source_windows')


def author_index():
    checkout = Path(__file__).resolve().parents[1]/'vendor/contextpilot'
    if not (checkout/'contextpilot/server/live_index.py').exists():
        pytest.skip('prepared author ContextPilot checkout unavailable')
    return SourceOrderIndex(checkout)


async def warm_path(engine, compiler, plan, order):
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    task = replace(plan.steps[0], input_unit_ids=order)
    info = await engine.info()
    candidate = compiler.compile_read(plan, task, tuple(units[key] for key in order), (), SessionSnapshot('warm'),
        outputs=(1000,), cache_scope=info)[0]
    ack = await engine.prepare('warm', 'prepare-warm', info.engine_restart_id)
    result = await engine.generate(input_ids=candidate.input_ids, session_id='warm',
        expected_generation=ack.session_generation, expected_restart_id=info.engine_restart_id,
        attempt_id='warm', fingerprint=compiler.template.fingerprint, max_new_tokens=1)
    compiler.observe_executed(candidate, result.receipt,
        cache_config_hash=info.cache_config_hash, engine_restart_id=info.engine_restart_id)
    await engine.retire('warm', ack.session_generation, 'retire-warm', info.engine_restart_id, 'completed fixture')
    return candidate


@pytest.mark.asyncio
async def test_selected_cached_organization_persists_and_drives_execution_and_demand(tmp_path):
    template, store = ByteTemplate(), SessionStore(tmp_path/'state.db')
    engine = SourceReader(store)
    compiler = RequestCompiler(template, author_index())
    plan = source_plan(template)
    assert [task.input_unit_ids for task in plan.steps] == [('a', 'c'), ('b', 'd')]
    warmed = await warm_path(engine, compiler, plan, ('a', 'b'))
    paid = RecordedPaid()
    native = ObservedNativeObserver(paid, template.fingerprint, reader_checkpoint='reader', reader_profile_id='profile')
    runtime = RollingPlanRuntime(None, native, reader_checkpoint='reader', reader_profile_id='profile',
        output_budgets=(1000,), mode='observed_actions', joint_context=True)
    controller = Controller(store, engine, template=template, rolling_policy=runtime, request_compiler=compiler,
        events=EventLog(tmp_path/'events.jsonl'))
    demand = DemandProjectionProbe()
    controller.observe_prefix = demand
    response = await controller.serve_turn('session', plan.task_state.raw_question, context_plans=(plan,), request_id='run')
    assert response.read_count == 2
    assert engine.read_sources == [('a', 'b'), ('c', 'd')]
    assert all(':observed:' not in catalog.plan_id for catalog in engine.catalogs)
    assert all([task.input_unit_ids for task in catalog.steps if task.kind == 'READ'] == [('a', 'b'), ('c', 'd')]
               for catalog in engine.catalogs)
    outcome = store.get_attempt(response.attempt_id).reading_outcome
    assert {span.span_id for claim in outcome.claims for span in claim.source_spans} == {'a', 'b', 'c', 'd'}
    assert {identity for result in outcome.results for identity in result.covered_units} == {'a', 'b', 'c', 'd'}
    first, second = outcome.steps
    assert second.task.step_id in DemandState.from_outcome(paid.states[1]).remaining_read_ids
    assert first.task.step_id not in DemandState.from_outcome(paid.states[1]).remaining_read_ids
    assert len(paid.states[1].claims) == 2
    assert demand.finished.claims == outcome.claims
    assert demand.finished.coverage == outcome.coverage
    assert demand.finished.plan.plan_id == outcome.plan.plan_id
    selected_options = [(proposal, options[candidate.candidate_id])
        for plans, state, options in demand.options for proposal in plans for step in proposal.steps
        for candidate in step.variants if candidate.read_unit_order == ('a', 'b')]
    assert selected_options
    assert any(second.task.step_id in option.demand.remaining_read_ids for _, option in selected_options)
    bound = first.metrics['compiled_input']
    boundary = min(warmed.source_prefix_tokens, bound['source_prefix_tokens'])
    assert first.input_ids[:boundary] == warmed.input_ids[:boundary]
    assert first.receipt.actual_prefill_tokens < len(first.input_ids)
    await controller.aclose()


@pytest.mark.asyncio
async def test_cheaper_first_read_loses_when_its_complete_route_costs_more(tmp_path):
    from controller.admission import quoted_native_work

    template, snapshot = ByteTemplate(), SessionSnapshot('cost')
    store = SessionStore(tmp_path/'cost.db')
    engine = SourceReader(store)
    compiler = RequestCompiler(template)
    original = source_plan(template, scope='selected_sources')
    # Deliberately supply a dominated three-package route to the selector. The
    # path proposer itself rejects this gratuitous split; this regression guards
    # full-route pricing independently of that proposer guard.
    prototype = original.steps[0]
    extra = replace(original, plan_id='more-calls', steps=tuple(replace(prototype,
        step_id='extra:'+str(index), input_unit_ids=units) for index, units in enumerate((('a',), ('b',), ('c', 'd')))))
    original = replace(original, plan_id='two-calls')
    state = ReadingOutcome(original)
    info = await engine.info()
    proposals = []
    for organization in (original, extra):
        task = organization.steps[0]
        branch = complete_plan(replace(organization, steps=(task,)), (),
            semantic_kind='read_once', reader_checkpoint='reader')
        variants = compile_step(organization, task, state, snapshot, template, outputs=(1000,),
                                request_compiler=compiler, cache_scope=info)
        branch = replace(branch, context_plan=organization,
                         steps=(PlanStep(task, variants), branch.steps[-1]))
        proposals.append(replace(branch, route_steps=compile_route_steps(branch, state, snapshot, template,
            output_budgets=(1000,), request_compiler=compiler, cache_scope=info)))
    paid = RecordedPaid()
    observer = ObservedNativeObserver(paid, template.fingerprint, reader_checkpoint='reader', reader_profile_id='profile')
    native, reuse = await observer.observe(proposals, state, snapshot, engine, CostModel(), time.monotonic(),
        {}, selection_started=time.monotonic())
    quote = {row.candidate_id: row for row in native.quotes.quotes}
    first = [sum(quoted_native_work(quote[proposal.steps[0].variants[0].candidate_id])) for proposal in proposals]
    assert first[1] < first[0]
    reserves = [route_completion_reserve(proposal, state, proposal.steps[0].task,
        proposal.steps[0].variants[0], native.quotes) for proposal in proposals]
    assert [len(future) for _, future in reserves] == [1, 2]
    assert reserves[1][0].input_tokens > reserves[0][0].input_tokens
    assert reserves[1][0].steps > reserves[0][0].steps
    policy = PlanPolicy(None, rule_baseline=True)
    decision = policy.choose_observed_actions(proposals, state, native, reuse,
                                              joint_options=joint_options(proposals, state))
    assert decision.next_step.plan_id == 'two-calls'
    costs = [policy.last_option_diagnostics[proposal.steps[0].variants[0].candidate_id]['quoted_work_proxy']
             for proposal in proposals]
    assert costs[1] > costs[0]
    store.close()




class SpecifiedNativeBranch:
    reader_checkpoint, reader_profile_id, asset_id = "reader", "profile", "specified-test-cost"
    feature_names = OBSERVED_FEATURE_NAMES
    provenance = {"continuation_protocol": BOUNDED_CONTINUATION,
                  "observed_feature_schema": OBSERVED_FEATURE_SCHEMA}
    cost_provenance = {"feature_schema": NATIVE_COST_SCHEMA,
        "completion_reserve_scope": COMPLETION_RESERVE_SCOPE,
        "observed_feature_schema": OBSERVED_FEATURE_SCHEMA, "prefix_control_scope": "native_excludes_prefix_control"}

    def supports_semantic_kind(self, kind, *, costs=False):
        return True

    def predict(self, features):
        return .5

    def predict_costs(self, features):
        cost = dict(complete_ms=100, gpu_gib_seconds=0, host_gib_seconds=0,
                    total_input_tokens=100, total_output_tokens=10)
        return {"branch_complete_cost": cost, "incremental_complete_cost": dict(cost)}


@pytest.mark.asyncio
async def test_fixed_observer_quotes_future_sources_without_promoting_them_to_current_answers(tmp_path):
    from controller.execution import ReadingExecutor
    from controller.rolling_policy import FixedReadingObserver

    template, store = ByteTemplate(), SessionStore(tmp_path/"state.db")
    engine = SourceReader(store)
    compiler = RequestCompiler(template, author_index())
    block, unit = material("Material A supports a partial procedure.", "a")
    plan = build_read_plan(TaskState("t", "How?", needs=(Need("n", operation="procedure"),)),
        EvidenceSet("s", (unit,), disposition="partial", source_blocks=(block,)), "selected_sources",
        ContextBudget(read_output_tokens=512), template=template, read_layout="source_first")
    cohort, output = tmp_path/"cohort.json", tmp_path/"observations.jsonl"
    cohort.write_text(json.dumps({"members": []}))
    observer = FixedReadingObserver({"reader_checkpoint": template.fingerprint.model_revision,
        "reader_profile_id": "profile", "observation_path": str(output), "output_budgets": [512],
        "gap_limits": {"max_units": 2, "max_passages": 2}, "cohort_split": "train",
        "cohort_manifest": str(cohort), "joint_context": True}, RecordedPaid(), template.fingerprint)
    snapshot = store.read_snapshot("session")
    attempt = store.begin_attempt("session", "run", "hash", plan.task_state.raw_question, 0)
    state = ReadingOutcome(plan)
    cache_scope = await engine.info()
    direct = replace(complete_plan(replace(plan, steps=()), (), semantic_kind="direct_answer",
        reader_checkpoint=template.fingerprint.model_revision), context_plan=plan)
    assert compile_route_steps(direct, state, snapshot, template, output_budgets=(512,),
        request_compiler=compiler, cache_scope=cache_scope) == ()
    quoted = []
    quote_batch = engine.quote_batch

    async def capture(candidates, fingerprint):
        quoted.extend(candidates)
        return await quote_batch(candidates, fingerprint)

    engine.quote_batch = capture

    def answers(snap, plans, observed, **options):
        assert observed.claims == state.claims and observed.results == state.results
        return compiler.compile_answers(snap, plan.task_state.raw_question, plans[0], outcome=observed,
                                         cache_scope=cache_scope, **options)

    executor = ReadingExecutor(store, engine, template, request_compiler=compiler)
    await observer.observe(attempt.attempt_id, executor, state, snapshot, engine, CostModel(), answers,
        arrived=time.monotonic(), auxiliary_usage={}, expanded=set(), startup_ms=0)
    packet = json.loads(output.read_text())
    assert packet["legal_answer_representations"] == ["hybrid"]
    assert "read_once" in packet["legal_actions"]
    assert any(candidate.representation == "raw" and candidate.source_body_token_ranges for candidate in quoted)
    assert not engine._attempts
    store.close()


@pytest.mark.asyncio
async def test_owner_control_added_once_without_rebating_native_hits(tmp_path):
    template, snapshot = ByteTemplate(), SessionSnapshot("cost")
    store = SessionStore(tmp_path/"cost.db")
    engine, compiler = SourceReader(store), RequestCompiler(template)
    plan = source_plan(template, scope="selected_sources")
    plan = replace(plan, budget=replace(plan.budget, lambda_time=.01))
    state = ReadingOutcome(plan)
    info = await engine.info()
    task = plan.steps[0]
    branch = complete_plan(replace(plan, steps=(task,)), (), semantic_kind="read_once", reader_checkpoint="reader")
    variants = compile_step(plan, task, state, snapshot, template, outputs=(1000,),
                            request_compiler=compiler, cache_scope=info)
    branch = replace(branch, context_plan=plan, steps=(PlanStep(task, variants), branch.steps[-1]))
    branch = replace(branch, route_steps=compile_route_steps(branch, state, snapshot, template,
        output_budgets=(1000,), request_compiler=compiler, cache_scope=info))
    observer = ObservedNativeObserver(RecordedPaid(), template.fingerprint,
                                      reader_checkpoint="reader", reader_profile_id="profile")
    native, reuse = await observer.observe((branch,), state, snapshot, engine, CostModel(), time.monotonic(),
        {}, selection_started=time.monotonic())
    policy = PlanPolicy(SpecifiedNativeBranch())
    key = variants[0].candidate_id
    ordinary = JointOption(DemandState.from_outcome(state))
    control = {"predicted_ms": 5, "source": "specified test inspect callback"}
    policy.choose_observed_actions((branch,), state, native, reuse, read_control=control,
                                  joint_options={key: ordinary})
    before = policy.last_option_diagnostics[key]
    before_objective = policy.last_evaluations[0].objective
    # Allocation value does not discount a quote that already includes hits.
    retained = replace(ordinary, retention_allocation_value_ms=500,
                       retention_control_ms=7)
    policy.choose_observed_actions((branch,), state, native, reuse, read_control=control,
                                  joint_options={key: retained})
    after = policy.last_option_diagnostics[key]
    assert before["rollout_expected_cost"].latency_ms == 105  # One known action, not every quoted pack.
    assert after["rollout_expected_cost"].latency_ms == 112  # One selected owner call.
    assert after["retention_owner_control_ms"] == 7
    assert policy.last_evaluations[0].objective == pytest.approx(before_objective-.07)
    policy.gain_model.cost_provenance = {**policy.gain_model.cost_provenance,
        "cost_target_scope": BOUNDED_CONTINUATION,
        "read_count_profile": {"groups": {"read_once": {
            "without_infer": {"mean": 1.25, "samples": 4}}}}}
    policy.choose_observed_actions((branch,), state, native, reuse, read_control=control,
                                  joint_options={key: retained})
    measured = policy.last_option_diagnostics[key]
    assert measured["rollout_expected_cost"].latency_ms == 113.25
    assert measured["read_control_count_samples"] == 4
    assert measured["cost_target_scope"] == BOUNDED_CONTINUATION
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_ms", (7, 10001, None))
async def test_mandatory_scan_includes_owner_control_in_budget_and_trace(tmp_path, monkeypatch, owner_ms):
    import json
    from types import SimpleNamespace
    from controller.events import EventLog
    from controller.plan_policy import NoCompletePlan
    from controller.rolling_policy import RollingPlanRuntime
    from controller.service import Controller

    template, store = ByteTemplate(), SessionStore(tmp_path/"state.db")
    engine = SourceReader(store)
    plan = source_plan(template)
    plan = replace(plan, budget=replace(plan.budget, max_complete_ms=10000))
    native = ObservedNativeObserver(RecordedPaid(), template.fingerprint,
                                    reader_checkpoint="reader", reader_profile_id="profile")
    runtime = RollingPlanRuntime(None, native, reader_checkpoint="reader", reader_profile_id="profile",
        output_budgets=(1000,), mode="observed_actions", joint_context=True,
        completion_timing=SimpleNamespace(reserve_ms=lambda point: 100))
    # Explicit CPU timing fixture; no calibration or measured latency claim.
    monkeypatch.setattr("controller.rolling_policy.timing_point", lambda *args, **kwargs: None)
    monkeypatch.setattr("controller.action_policy.timing_point", lambda *args, **kwargs: None)

    class OwnerCost(DemandProjectionProbe):
        async def demand_options(self, *args, **kwargs):
            options = await super().demand_options(*args, **kwargs)
            return {key: replace(option, retention_control_ms=owner_ms) for key, option in options.items()}

    controller = Controller(store, engine, template=template, rolling_policy=runtime,
        request_compiler=RequestCompiler(template), events=EventLog(tmp_path/"events.jsonl"))
    controller.observe_prefix = OwnerCost()
    if owner_ms != 7:
        reason = "unmeasured_retention_control" if owner_ms is None else "complete_latency_budget"
        with pytest.raises(NoCompletePlan, match=reason):
            await controller.serve_turn("session", plan.task_state.raw_question, context_plans=(plan,), request_id="run")
        assert engine.read_sources == []
    else:
        await controller.serve_turn("session", plan.task_state.raw_question, context_plans=(plan,), request_id="run")
        events = [json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()]
        scanned = [event for event in events if event["event"] == "rolling_required_scope_step"]
        assert scanned
        assert all(event["retention_owner_control_ms"] == 7 for event in scanned)
        assert all(event["completion_latency_ms"] == event["completion_cost"]["latency_ms"] == 107 for event in scanned)
    await controller.aclose()
    store.close()
