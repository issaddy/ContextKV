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
from controller.reading import advance_read_plan, parse_read_result
from controller.rolling_policy import RollingPlanRuntime, ZERO_AUX, compile_step, namespace_plan
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from policy_helpers import SpecifiedGainFixture
from rolling_helpers import NativeObservationFixture, SpecifiedCosts, setup_controller
from test_shared_numeric_reading import NumericEngine, material, numeric_plan, response, row


class Interrupted(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_native_read_response_error_recovers_once_by_persisted_submission(tmp_path):
    from controller.engine_client import EngineError

    controller, store, engine, _, plan = setup_controller(tmp_path)
    controller.rolling_policy = None
    plan = replace(plan, selected_set=replace(plan.selected_set, disposition="partial"))
    generate = engine.generate

    async def lose_first_response(**kwargs):
        result = await generate(**kwargs)
        if len(engine.constraints) == 1:
            raise EngineError("INVALID_ENGINE_RESPONSE")
        return result

    engine.generate = lose_first_response
    with pytest.raises(EngineError, match="INVALID_ENGINE_RESPONSE"):
        await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    pending = store.pending_attempt("s")
    assert store.has_submitted_model_step(pending.attempt_id)
    result = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    assert result.attempt_id == pending.attempt_id
    assert result.read_count == 1
    assert len(engine.constraints) == 2
    assert len(store.get_snapshot("s").raw_history) == 1
    await controller.aclose()
    store.close()


def test_terminal_native_token_usage_survives_duplicate_ledger_snapshots(tmp_path):
    from controller.context_schema import CompletionReceipt, StepRecord
    from controller.templates import token_hash
    from profiling.native_usage_index import index

    def ledger(name, outputs):
        store = SessionStore(tmp_path/name)
        store.read_snapshot("session")
        attempt = store.begin_attempt("session", "question", "hash", "How?", 0)
        inputs = (10, 20)
        receipt = CompletionReceipt("read", "read", "epoch", 1, "restart", token_hash(inputs), "stop", True,
                                    actual_prefill_tokens=2, actual_output_tokens=outputs)
        task = ReadTask("read", "READ")
        store.save_execution_step(StepRecord(attempt.attempt_id, task.step_id, task, input_ids=inputs,
                                             receipt=receipt), "FAILED")
        store.close()
        return str(store.path)

    measured, missing = ledger("a.db", 7), ledger("b.db", None)
    report = index([measured, missing])
    assert report["online_token_ledger"]["total_tokens"] == 9
    partial, = report["partial_native_calls"]
    assert partial["output_tokens"] == 7 and len(partial["locations"]) == 2
    conflict = index([measured, ledger("c.db", 8)])
    assert conflict["online_token_ledger"]["reasons"] == ["conflicting_native_receipts"]


@pytest.mark.asyncio
async def test_truncated_teacher_completion_is_charged_before_task_parsing(tmp_path):
    import httpx

    from adapters.history_state import TeacherTaskModel
    from adapters.query_selector import OpenAICompletionClient, SelectorConfig
    from adapters.retrieval import run_retrieval_worker
    from controller.context_schema import RetrievalState
    from controller.schema import OnlineTurn
    from controller.templates import ByteTokenizer
    from profiling.native_usage_index import index

    class Tokenizer(ByteTokenizer):
        chat_template = "test"

        def apply_chat_template(self, messages, **kwargs):
            return self.encode("".join(message["content"] for message in messages))

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "choices": [{"message": {"content": "{"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 8000, "completion_tokens": 2048}}))
    client = OpenAICompletionClient(SelectorConfig("http://teacher", "teacher", "fixture",
        max_input_tokens=16000, max_new_tokens=2048), tokenizer=Tokenizer(), transport=transport)
    model = TeacherTaskModel(client)
    controller, store, engine, _, _ = setup_controller(tmp_path)

    async def retrieve(snapshot):
        turn = OnlineTurn(snapshot.session_id, "r", 1, "corpus", "How?", (), (), "P3")
        await run_retrieval_worker(model.predict, turn, (), RetrievalState(0))

    with pytest.raises(RuntimeError, match="unfinished: completed/length"):
        await controller.serve_turn("s", "How?", request_id="r", evidence_provider=retrieve)
    attempt = store.find_request("s", "r")
    assert attempt.stage == "FAILED" and attempt.retrieval_ms > 0
    assert attempt.auxiliary_usage["input_tokens"] == model.usage["input_tokens"] == 8000
    assert attempt.auxiliary_usage["output_tokens"] == model.usage["output_tokens"] == 2048
    records = store.db.execute("SELECT payload FROM auxiliary_calls WHERE attempt_id=?",
                               (attempt.attempt_id,)).fetchall()
    receipts = [json.loads(record["payload"]) for record in records]
    group, receipt = receipts
    assert receipt["finish_reason"] == "length" and receipt["group_id"] == group["call_id"]
    ledger = index([str(store.path)])
    usage = ledger["auxiliary"]
    assert usage["unique_calls"] == 1
    assert usage["totals"]["input_tokens"]["observed"] == 8000
    assert usage["totals"]["output_tokens"]["observed"] == 2048
    assert ledger["online_token_ledger"]["total_tokens"] == 10048
    assert not engine.constraints
    client.close()
    await controller.aclose()
    store.close()


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
async def test_reused_numeric_atoms_restore_once_and_compute_current_condition_view(tmp_path):
    from controller.context_budget import direct_answer_reason
    from controller.fact_reuse import apply_reuse_to_plan

    needs = (Need("shipment-count", operation="count", condition_ids=("current",)),
             Need("shipment-mass", operation="sum", condition_ids=("current",)))
    plan = numeric_plan((material("R1 mass 1 kg."),), needs=needs)
    task = replace(plan.steps[0], step_id="policy:reuse:complete")
    plan = replace(plan, steps=(task,))
    read = parse_read_result(task, json.dumps(response(row("R1", "1", "kg"))),
                             plan.selected_set.selected_units)
    atoms = tuple(replace(read.claims[0], claim_id=f"history:{index}:atom",
                          conditions=(f"old-{index}",)) for index in range(2))
    read = replace(read, claims=atoms)
    seed = ReadingOutcome(plan, (read,), claims=atoms)
    assert direct_answer_reason(seed, None) == "numeric_selection_incomplete"
    store = SessionStore(tmp_path / "state.db")
    snapshot = store.read_snapshot("s")
    current_units = tuple(replace(unit, unit_id="fresh:"+unit.unit_id,
        source_spans=tuple(replace(span, span_id="fresh:"+span.span_id) for span in unit.source_spans))
        for unit in plan.selected_set.selected_units)
    current_plan = replace(plan, selected_set=replace(plan.selected_set, selected_units=current_units),
        coverage_units=tuple(unit.unit_id for unit in current_units), steps=(replace(plan.steps[0],
            step_id="new:read", input_unit_ids=tuple(unit.unit_id for unit in current_units)),))
    seed = apply_reuse_to_plan(current_plan, seed, ByteTemplate(), snapshot)
    plan = seed.plan
    attempt = store.begin_attempt("s", "r", "hash", plan.task_state.raw_question, 0)
    store.save_context_plan(attempt.attempt_id, plan, initial_outcome=seed)
    engine = NumericEngine(None)
    executor = ReadingExecutor(store, engine, ByteTemplate())
    outcome = await executor.execute_reads(attempt.attempt_id, snapshot, plan,
        initial_outcome=ReadingOutcome(plan))
    computed = {claim.operation: claim for claim in outcome.claims if claim.operation in {"count", "sum"}}
    assert computed["count"].value == 1 and computed["sum"].value == "1000"
    assert all(claim.conditions == ("current",) for claim in outcome.claims if claim.operation == "numeric_select")
    assert tuple(claim.conditions for claim in outcome.claims[:2]) == (("old-0",), ("old-1",))
    assert engine.reads == 0 and engine.selections == 2
    assert direct_answer_reason(outcome, None) is None
    assert len(outcome.steps) == len(store.execution_steps(attempt.attempt_id)) == 4
    assert outcome.plan.online_tokens_spent == sum(len(step.input_ids)+len(step.output_ids) for step in outcome.steps)
    store.save_context_plan(attempt.attempt_id, outcome.plan, initial_outcome=outcome)
    restored = await executor.execute_reads(attempt.attempt_id, snapshot, outcome.plan)
    assert restored.claims == outcome.claims and restored.results == outcome.results
    assert restored.plan.online_tokens_spent == outcome.plan.online_tokens_spent
    assert len(restored.steps) == 4 and engine.reads == 0 and engine.selections == 2
    store.close()


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

        def stop(catalog, state, *args, **kwargs):
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

    def stop(catalog, state, *args, **kwargs):
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

    async def prefix(current_plan, task, units, record):
        callbacks.append(record.step_id)
        assert record.input_ids == candidate.input_ids
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

    def stop(catalog, state, *args, **kwargs):
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
