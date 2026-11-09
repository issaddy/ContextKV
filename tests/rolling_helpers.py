"""Shared CPU protocol fixtures for core property tests."""
from dataclasses import replace
import time
from controller.context_schema import ContextBudget, EvidenceSet, Need, ReadTask, TaskState
from controller.cost_model import CostModel
from controller.events import EventLog
from controller.plan_policy import CompleteCost, NativePlanningState, ResourceForecast, StepForecast
from controller.reading import build_read_plan
from controller.rolling_policy import RollingPlanRuntime
from controller.schema import CostEstimate
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from policy_helpers import SpecifiedGainFixture, bind_fixture_timing
from reading_helpers import ScriptedReadingEngine, fact, material, read_output


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
        candidates = tuple({candidate.candidate_id: candidate for plan in plans
            for step in (*plan.steps, *plan.route_steps) for candidate in step.variants
            if candidate.feasible}.values())
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
