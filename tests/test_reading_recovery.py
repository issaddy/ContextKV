"""Paid logical steps survive interruption without another model/tool call."""
import json
import time
from dataclasses import replace

import pytest

from bench.mock_engine import MockEngine
from controller.action_proposals import GapRetrievalRequest
from controller.claim_graph import ClaimGraph
from controller.context_schema import EvidencePool, Need, OpenNeed, ReadingOutcome, ReadTask
from controller.execution import ReadingExecutor
from controller.reading import advance_read_plan
from controller.rolling_policy import RollingPlanRuntime, ZERO_AUX, compile_step, namespace_plan
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from policy_helpers import SpecifiedGainFixture
from rolling_helpers import NativeObservationFixture, SpecifiedCosts, setup_controller
from test_shared_numeric_reading import NumericEngine, material, numeric_plan, response, row


class Interrupted(RuntimeError):
    pass


class EmptySelectionEngine(NumericEngine):
    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema["properties"]["format_version"]["const"] != 7:
            return await super().generate(response_schema=response_schema, **kwargs)
        self.selections += 1
        views = response_schema["properties"]["views"]["properties"]
        self.response_text = json.dumps({"format_version": 7, "views": {
            alias: dict.fromkeys(view["properties"], False) for alias, view in views.items()}})
        return await MockEngine.generate(self, **kwargs)


def interrupt_after_save(store, predicate):
    save = store.save_execution_step

    def interrupted(record, stage, generation=None):
        save(record, stage, generation)
        if predicate(record, stage):
            store.save_execution_step = save
            raise Interrupted()

    store.save_execution_step = interrupted


@pytest.mark.asyncio
@pytest.mark.parametrize("rolling", [False, True])
async def test_empty_numeric_selection_first_complete_is_recoverable(tmp_path, rolling):
    pair = material("R1 mass 1 kg.")
    plan = numeric_plan((pair,), needs=(Need("shipment-count", operation="count"),))
    if rolling:
        plan = namespace_plan(plan)
    output = response(row("R1", "1", "kg", needs=("n0",)))
    output["support"] = ["n0"]
    engine = EmptySelectionEngine(output)
    store = SessionStore(tmp_path / "state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", plan.task_state.raw_question, 0)
    arrived = time.monotonic()
    executor = ReadingExecutor(store, engine, ByteTemplate(), request_arrived=arrived)
    interrupt_after_save(store, lambda record, stage:
        stage == "COMPLETE" and record.task.operation == "numeric_select")
    with pytest.raises(Interrupted):
        if rolling:
            state = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
            while True:
                task = advance_read_plan(state.plan, state.results)[0]
                candidate = compile_step(state.plan, task, state, snapshot, ByteTemplate(),
                                         outputs=(task.output_budget,))[0]
                state = await executor.execute_one(attempt.attempt_id, snapshot, state.plan,
                                                  task, state, candidate=candidate)
        else:
            await executor.execute_reads(attempt.attempt_id, snapshot, plan)
    selector = store.execution_steps(attempt.attempt_id)[-1]
    assert selector.result.reduction.basis_claim_ids
    assert selector.result.reduction.frontier_claim_ids == ()
    if rolling:
        runtime = RollingPlanRuntime(SpecifiedGainFixture(), NativeObservationFixture(),
            reader_checkpoint="reader-fixture", reader_profile_id="profile-fixture",
            output_budgets=(8000,))
        runtime.mode = "observed_actions"
        captured = []

        def stop(catalog, state, *args):
            captured.append(state)
            raise Interrupted()

        runtime.action_proposals = stop
        from controller.events import EventLog
        with pytest.raises(Interrupted):
            await runtime.run(attempt.attempt_id, snapshot, (plan,), executor, engine, ByteTemplate(),
                SpecifiedCosts(), None, arrived=arrived, auxiliary_usage=dict(ZERO_AUX), events=EventLog())
        outcome = captured[0]
    else:
        outcome = await executor.execute_reads(attempt.attempt_id, snapshot,
                                              store.context_plan(attempt.attempt_id))
    count = next(claim for claim in outcome.claims if claim.operation == "count" and claim.kind == "derived")
    assert count.value == 0 and count.parent_claim_ids
    assert ClaimGraph(outcome.claims).source_spans(count.claim_id) == pair[1].source_spans
    assert engine.reads == engine.selections == 1
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_complete", [False, True])
async def test_gap_result_ready_resumes_saved_pool_without_retrieval(tmp_path, after_complete):
    controller, store, engine, _, plan = setup_controller(tmp_path, operation="infer")
    runtime = controller.rolling_policy
    runtime.mode = "observed_actions"
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", plan.task_state.raw_question, 0)
    arrived = time.monotonic()
    executor = ReadingExecutor(store, engine, ByteTemplate(), request_arrived=arrived)
    need = OpenNeed("n", "terminal", ("item",))
    state = ReadingOutcome(plan, open_needs=(need,), auxiliary_usage=dict(ZERO_AUX))
    request = GapRetrievalRequest((need,), "terminal", {"fixture": True},
                                  {"max_units": 2, "max_passages": 2})
    task = ReadTask("policy:gap:paid", "GAP", ("n",), operation="retrieve")
    block, unit = material("Terminal evidence.", "terminal")
    calls = []

    async def provider(*args):
        calls.append(args)
        return EvidencePool((unit,), source_blocks=(block,), auxiliary_usage={
            "input_tokens": 7, "output_tokens": 2, "compute_ms": 3, "models": []})

    if after_complete:
        save = store.save_reading_progress

        def interrupted(*args, **kwargs):
            save(*args, **kwargs)
            if kwargs.get("completed_step") is not None:
                store.save_reading_progress = save
                raise Interrupted()

        store.save_reading_progress = interrupted
    else:
        interrupt_after_save(store, lambda record, stage: stage == "RESULT_READY")
    with pytest.raises(Interrupted):
        await runtime._run_gap(attempt.attempt_id, task, request, state, snapshot, ByteTemplate(),
            executor, set(), provider, controller.events, arrived)
    captured = []

    def stop(catalog, state, *args):
        captured.append(state)
        raise Interrupted()

    runtime.action_proposals = stop
    with pytest.raises(Interrupted):
        await runtime.run(attempt.attempt_id, snapshot, (plan,), executor, engine, ByteTemplate(),
            controller.cost_model, None, arrived=arrived, auxiliary_usage=dict(ZERO_AUX),
            events=controller.events, open_need_provider=provider)
    recovered = captured[0]
    assert len(calls) == 1
    assert store.execution_step(attempt.attempt_id, task.step_id)[0] == "COMPLETE"
    assert any(current.unit_id == unit.unit_id for current in recovered.plan.selected_set.selected_units)
    assert recovered.auxiliary_usage["input_tokens"] == 7
    assert recovered.auxiliary_usage["output_tokens"] == 2
    assert sum(record.task.kind == "GAP" for record in recovered.steps) == 1
    assert len({task.step_id for task in recovered.plan.steps}) == len(recovered.plan.steps)
    assert sum(task.kind == "READ" for task in recovered.plan.steps) == len(plan.steps) + 1
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
async def test_rolling_complete_read_resumes_prefix_callback_before_retirement(tmp_path):
    controller, store, engine, _, plan = setup_controller(tmp_path)
    plan = namespace_plan(replace(plan, read_layout="source_first"))
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", plan.task_state.raw_question, 0)
    arrived = time.monotonic()
    callbacks = []

    async def prefix(current_plan, task, units, rendered, record):
        callbacks.append(record.step_id)
        assert rendered.input_ids == record.input_ids
        if len(callbacks) == 1:
            raise Interrupted()
        return {"owner_id": "recovered-prefix", "state": "RETAINED"}

    executor = ReadingExecutor(store, engine, ByteTemplate(), request_arrived=arrived,
                               observe_prefix=prefix)
    state = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
    task = plan.steps[0]
    candidate = compile_step(plan, task, state, snapshot, ByteTemplate(), outputs=(512,))[0]
    with pytest.raises(Interrupted):
        await executor.execute_one(attempt.attempt_id, snapshot, plan, task, state, candidate=candidate)
    assert store.execution_step(attempt.attempt_id, task.step_id)[0] == "COMPLETE"
    assert len(engine.constraints) == 1
    runtime = controller.rolling_policy
    runtime.mode = "observed_actions"
    captured = []

    def stop(catalog, state, *args):
        captured.append(state)
        raise Interrupted()

    runtime.action_proposals = stop
    with pytest.raises(Interrupted):
        await runtime.run(attempt.attempt_id, snapshot, (plan,), executor, engine, ByteTemplate(),
            controller.cost_model, None, arrived=arrived, auxiliary_usage=dict(ZERO_AUX),
            events=controller.events)
    record = captured[0].steps[0]
    assert callbacks == [task.step_id, task.step_id]
    assert record.metrics["evidence_prefix"]["owner_id"] == "recovered-prefix"
    assert engine._epochs[record.engine_epoch.engine_session_id].state == "RETIRED"
    assert len(engine.constraints) == 1
    await controller.aclose()
    store.close()
