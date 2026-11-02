"""CPU protocol fixtures exercise the real Controller transaction and loop."""
from dataclasses import replace
import time

import pytest

from controller.context_schema import ContextBudget, EvidenceSet, Need, ReadTask, TaskState
from controller.cost_model import CostModel
from controller.events import EventLog
from controller.plan_policy import CompleteCost, NativePlanningState, ResourceForecast, StepForecast
from controller.reading import build_read_plan
from controller.rolling_policy import RollingPlanRuntime, namespace_plan
from controller.schema import CostEstimate
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from test_plan_policy import SpecifiedGainFixture, bind_fixture_timing
from test_reading_runtime import ScriptedReadingEngine, fact, material, read_output


class SpecifiedCosts(CostModel):
    def predict_native(self, candidate, quote, *args, **kwargs):
        return CostEstimate(candidate.candidate_id, 1, calibrated=True)


class NativeObservationFixture:
    """Specified unit-test physical forecasts, never hardware measurements."""
    def __init__(self):
        self.states, self.paid, self.selection_starts = [], [], []
    async def start(self):
        pass
    async def aclose(self):
        pass
    async def observe(self, plans, state, snapshot, engine, cost_model, arrived, usage, *, selection_started):
        self.states.append(state)
        self.selection_starts.append(selection_started)
        candidates = tuple(candidate for plan in plans for step in plan.steps for candidate in step.variants
                           if candidate.feasible)
        quotes = await engine.quote_batch(candidates, ByteTemplate().fingerprint)
        quotes = replace(quotes, active_requests=0)
        resource = {candidate.candidate_id: ResourceForecast(2, 3, 1, 1,
            "explicit-test-resource", quotes.observed_seq, (), True) for candidate in candidates}
        future = {}
        for plan in plans:
            for step in plan.steps:
                task = step.task
                cpu = task.kind == "REDUCE" and task.operation in {"count", "sum", "group", "union"}
                future[plan.plan_id, task.step_id] = (StepForecast("CPU" if cpu else "REBUILD",
                    0 if cpu else task.output_budget, CompleteCost(1, 2, 3, 0 if cpu else 10,
                        0 if cpu else task.output_budget, 1, 1), "explicit-test-future", True,
                    "cpu_profile" if cpu else "future-test-profile"),)
        paid = CompleteCost((time.monotonic()-arrived)*1000, 7*len(state.steps), 11*len(state.steps),
            sum(len(row.input_ids) for row in state.steps), sum(len(row.output_ids) for row in state.steps))
        self.paid.append(paid)
        bind_fixture_timing(cost_model, candidates)
        return NativePlanningState("reader-fixture", "profile-fixture", snapshot, quotes,
            cost_model, resource, future, paid), {}


def setup_controller(tmp_path, *, operation="procedure", scope="selected_sources", reads=1, gain=None):
    blocks, units = zip(*(material("Original evidence "+str(index), str(index)) for index in range(reads)))
    selected = EvidenceSet("s", units, source_blocks=blocks)
    state = TaskState("t", "How?", needs=(Need("n", operation=operation),), current_turn_id=1)
    plan = build_read_plan(state, selected, scope, ContextBudget(output_tokens=64, read_output_tokens=512),
                           template=ByteTemplate())
    # Separate original windows exercise stepwise scheduling, not a fake READ result.
    plan = replace(plan, steps=tuple(ReadTask("read:"+str(i), "READ", ("n",), (unit.unit_id,),
        input_budget=12000, output_budget=512) for i, unit in enumerate(units)),
        coverage_units=tuple(unit.unit_id for unit in units))
    observer = NativeObservationFixture()
    runtime = RollingPlanRuntime(gain or SpecifiedGainFixture(), observer,
        reader_checkpoint="reader-fixture", reader_profile_id="profile-fixture", output_budgets=(256, 512))
    engine = ScriptedReadingEngine([read_output(fact(str(i), "entity "+str(i))) for i in range(reads)])
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, cost_model=SpecifiedCosts(), rolling_policy=runtime,
        events=EventLog(tmp_path/"events.jsonl"))
    return controller, store, engine, observer, plan


@pytest.mark.asyncio
async def test_controller_reselects_after_each_read_and_commits_one_qa(tmp_path):
    controller, store, engine, observed, plan = setup_controller(tmp_path, reads=2)
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    assert response.read_count == 2 and response.reduce_count == 0
    assert len(observed.states) == 3
    assert [len(state.steps) for state in observed.states] == [0, 1, 2]
    assert observed.paid[-1].input_tokens > observed.paid[1].input_tokens > 0
    assert response.total_input_tokens == response.reader_input_tokens
    assert len(store.get_snapshot("s").raw_history) == store.get_snapshot("s").version == 1
    window = response.answer_phase_window
    assert window["start"] == observed.selection_starts[-1]
    assert window["start"] < window["commit_at"] <= window["end"]
    import json
    committed = next(json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()
                     if json.loads(line)["event"] == "ledger_committed")
    assert committed["answer_phase_window"] == window
    attempt = store.find_request("s", "r")
    assert attempt.plan.reason == "rolling_plan_gain"
    assert len({row.step_id for row in attempt.reading_outcome.steps}) == 2
    assert all(row.step_id.startswith("policy:") for row in attempt.reading_outcome.steps)
    calls = len(engine.constraints)
    replay = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    assert replay.idempotent_replay and len(engine.constraints) == calls
    assert replay.answer_phase_window is None
    await controller.aclose()


@pytest.mark.asyncio
async def test_cpu_reduce_is_one_observed_chosen_step_with_no_model_call(tmp_path):
    controller, store, engine, observed, plan = setup_controller(tmp_path, operation="count")
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    assert response.read_count == response.reduce_count == 1
    assert len(engine.constraints) == 2  # one READ, one terminal ANSWER
    assert [len(state.steps) for state in observed.states] == [0, 1, 2]
    outcome = store.find_request("s", "r").reading_outcome
    assert outcome.steps[-1].task.operation == "count"
    assert outcome.steps[-1].result.claims[0].value == 1
    assert outcome.steps[-1].metrics["phase_window"]["start"] > outcome.steps[0].metrics["phase_window"]["end"]
    await controller.aclose()


@pytest.mark.asyncio
async def test_full_scope_forbids_cheap_direct_answer_before_all_originals_are_read(tmp_path):
    class NegativeGain(SpecifiedGainFixture):
        def predict(self, features):
            return -.5
    controller, _, _, observed, plan = setup_controller(tmp_path, scope="all_supplied_documents", reads=2,
                                                       gain=NegativeGain())
    response = await controller.serve_turn("s", "How?", context_plans=(plan,))
    assert response.read_count == 2
    assert len(observed.states[-1].results) == 2
    assert {unit for result in observed.states[-1].results for unit in result.covered_units} == {"0", "1"}
    await controller.aclose()


def test_content_namespace_separates_different_read_zero_inputs_and_shares_identical_tasks():
    from test_context_planning import context_plan
    plan, _ = context_plan()
    first = replace(plan, steps=(ReadTask("read:0", "READ", ("need",), ("u",)),))
    other = replace(first, plan_id="another-set")
    assert namespace_plan(first).steps[0].step_id == namespace_plan(other).steps[0].step_id
    changed = replace(first, read_packing="source_windows")
    assert namespace_plan(first).steps[0].step_id != namespace_plan(changed).steps[0].step_id
    changed = replace(first, selected_set=replace(first.selected_set, selected_units=(
        replace(first.selected_set.selected_units[0], text="different"),)))
    assert namespace_plan(first).steps[0].step_id != namespace_plan(changed).steps[0].step_id


@pytest.mark.asyncio
async def test_fixed_reader_reports_fresh_answer_tail_and_keeps_replay_unobserved(tmp_path):
    from controller.planner import PlannerKnobs
    controller, store, _, _, plan = setup_controller(tmp_path)
    controller.rolling_policy = None
    controller.knobs = PlannerKnobs(strategy="always_rebuild")
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="fixed")
    window = response.answer_phase_window
    step = store.find_request("s", "fixed").reading_outcome.steps[0]
    assert step.metrics["phase_window"]["end"] <= window["start"]
    assert window["start"] < window["commit_at"] <= window["end"]
    assert window["start_definition"] == "final_answer_compilation_start"
    replay = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="fixed")
    assert replay.answer_phase_window is None
    await controller.aclose()


@pytest.mark.asyncio
async def test_observed_open_need_proposes_priced_multihop_with_real_parent_and_auxiliary_cost(tmp_path):
    import json
    from bench.mock_engine import MockEngine
    from controller.context_schema import EvidencePool
    controller, store, _, observed, plan = setup_controller(tmp_path)
    controller.rolling_policy.output_budgets = (512,)
    block, unit = material("The prerequisite is the current form.", "new")

    class FollowupEngine(MockEngine):
        def __init__(self):
            super().__init__()
            self.model_reads = 0
        async def generate(self, *, response_schema=None, **kwargs):
            if response_schema:
                self.model_reads += 1
                properties = response_schema["properties"]["claims"]["items"]["properties"]
                parents = ([] if self.model_reads == 1 else
                           properties["parent_claim_ids"]["items"]["enum"])
                value = fact(properties["span_ids"]["items"]["enum"][0])
                del value["version"]
                value["parent_claim_ids"] = parents
                value["followups"] = ([{"need_id": "n", "missing_relation": "prerequisite",
                    "bound_entities": ["item"], "temporal_scope": "current"}]
                    if self.model_reads == 1 else [])
                self.response_text = json.dumps({"format_version": 2, **read_output(value)})
            else:
                self.response_text = "The prerequisite form is required."
            return await super().generate(**kwargs)

    controller.engine = FollowupEngine()
    calls = []
    async def gap_provider(current, needs, claims):
        calls.append((current, needs, claims))
        return EvidencePool((unit,), source_blocks=(block,),
            auxiliary_usage={"input_tokens": 7, "output_tokens": 2, "compute_ms": 3.0, "models": []})
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), open_need_provider=gap_provider)
    assert response.read_count == 2 and len(calls) == 1
    outcome = store.get_attempt(response.attempt_id).reading_outcome
    assert outcome.steps[-1].task.input_claim_ids == (outcome.claims[0].claim_id,)
    assert outcome.claims[-1].parent_claim_ids == (outcome.claims[0].claim_id,)
    assert response.auxiliary_input_tokens == 7 and response.auxiliary_output_tokens == 2
    assert outcome.open_needs == ()
    assert len(observed.states) == 3
    await controller.aclose()


@pytest.mark.asyncio
async def test_rolling_resolves_lost_read_receipt_without_generating_the_read_twice(tmp_path):
    from controller.engine_client import UnknownOutcomeError
    controller, store, engine, observed, plan = setup_controller(tmp_path)
    engine.drop_next_generate_response = True
    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="lost")
    pending = store.find_request("s", "lost")
    assert pending.stage == "STARTED"
    assert store.get_snapshot("s").raw_history == ()
    first = store.execution_step(pending.attempt_id, store.execution_steps(pending.attempt_id)[0].step_id)[1]
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="lost")
    assert response.read_count == 1
    assert len(engine.constraints) == 2  # native receipt replay, then one ANSWER
    assert len(store.get_snapshot("s").raw_history) == 1
    final = store.execution_steps(response.attempt_id)[0]
    assert final.metrics["phase_window"]["start"] == first.metrics["phase_started_monotonic"]
    assert observed.states[-1].steps[0].step_id == first.step_id
    assert observed.paid[-1].input_tokens == len(final.input_ids)
    await controller.aclose()
