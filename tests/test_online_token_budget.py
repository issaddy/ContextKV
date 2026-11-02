from dataclasses import replace

import pytest

from controller.context_schema import ContextBudget, EvidencePool, EvidenceSet, Need, TaskState
from controller.engine_client import UnknownOutcomeError
from controller.execution import OnlineTokenBudgetUnmeasured, ReadingBudgetExceeded, ReadingExecutor
from controller.planner import PlannerKnobs
from controller.reading import build_read_plan, step_messages
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from test_reading_runtime import ScriptedReadingEngine, fact, material, read_output


def usage(inputs, outputs=0):
    return {"input_tokens": inputs, "output_tokens": outputs, "compute_ms": 0.0, "models": []}


def request_plan(*, limit=100000, read_output=2048):
    block, unit = material("A belongs to B.")
    state = TaskState("task", "Question", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("set", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(max_total_tokens=limit, read_output_tokens=read_output),
        template=ByteTemplate())
    return plan, unit


@pytest.mark.asyncio
@pytest.mark.parametrize("initial,exception", [(usage(6, 4), ReadingBudgetExceeded),
                                               (usage(None, 4), OnlineTokenBudgetUnmeasured)])
async def test_exhausted_or_unknown_initial_auxiliary_work_stops_before_first_read(tmp_path, initial, exception):
    plan, unit = request_plan(limit=10)
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    with pytest.raises(exception, match="READ"):
        await controller.serve_turn("s", plan.task_state.raw_question, request_id="r",
                                    context_plans=(plan,), auxiliary_usage=initial)
    assert not engine.constraints
    attempt = controller.store.find_request("s", "r")
    assert attempt.stage == "FAILED" and attempt.auxiliary_usage == initial
    assert controller.store.context_plan(attempt.attempt_id).online_tokens_spent == (
        10 if initial["input_tokens"] is not None else None)
    assert not controller.store.get_snapshot("s").raw_history
    await controller.drain()


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_gap_auxiliary_charge_stops_successor_and_answer_generation(tmp_path, unknown):
    plan, unit = request_plan()
    next_block, next_unit = material("B started in 2001.", "b")
    gap = {"need_id": "n", "missing_relation": "start date", "bound_entities": ["B"],
           "temporal_scope": "current", "parent_claim_ids": ["read:0:claim:0"]}
    engine = ScriptedReadingEngine([read_output(fact("a", "B"), support=(), open_needs=(gap,))])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    observed = []

    async def extend(current, needs, claims):
        pending = controller.store.pending_attempt("s")
        first = controller.store.execution_steps(pending.attempt_id)[0]
        assert current.online_tokens_spent == 13+len(first.input_ids)+len(first.output_ids)
        observed.append(current.online_tokens_spent)
        charge = None if unknown else current.budget.max_total_tokens-current.online_tokens_spent
        return EvidencePool((next_unit,), (next_block,), auxiliary_usage=usage(charge))

    exception = OnlineTokenBudgetUnmeasured if unknown else ReadingBudgetExceeded
    with pytest.raises(exception, match="OpenNeed result"):
        await controller.serve_turn("s", plan.task_state.raw_question, request_id="r",
            context_plans=(plan,), auxiliary_usage=usage(10, 3), open_need_provider=extend)
    assert len(observed) == len(engine.constraints) == 1
    attempt = controller.store.find_request("s", "r")
    persisted = controller.store.context_plan(attempt.attempt_id)
    assert persisted.online_tokens_spent == (None if unknown else plan.budget.max_total_tokens)
    assert len(controller.store.execution_steps(attempt.attempt_id)) == 1
    assert not controller.store.get_snapshot("s").raw_history
    assert controller.store.reading_progress(attempt.attempt_id).auxiliary_usage["input_tokens"] == (
        None if unknown else plan.budget.max_total_tokens-observed[0])
    await controller.drain()


@pytest.mark.asyncio
async def test_answer_admission_includes_auxiliary_and_completed_read_tokens(tmp_path):
    plan, unit = request_plan(read_output=512)
    task = plan.steps[0]
    rendered = ByteTemplate().render(step_messages(plan, task, (unit,)))
    plan = replace(plan, budget=replace(plan.budget,
        max_total_tokens=13+len(rendered.input_ids)+task.output_budget))
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    controller = Controller(SessionStore(tmp_path/"state.db"), engine,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    with pytest.raises(ReadingBudgetExceeded, match="ANSWER"):
        await controller.serve_turn("s", plan.task_state.raw_question, request_id="r",
                                    context_plans=(plan,), auxiliary_usage=usage(10, 3))
    assert len(engine.constraints) == 1
    attempt = controller.store.find_request("s", "r")
    first = controller.store.execution_steps(attempt.attempt_id)[0]
    assert controller.store.context_plan(attempt.attempt_id).online_tokens_spent == (
        13+len(first.input_ids)+len(first.output_ids))
    assert not controller.store.get_snapshot("s").raw_history
    await controller.drain()


@pytest.mark.asyncio
async def test_resuming_a_submitted_read_reuses_its_existing_token_charge(tmp_path):
    plan, unit = request_plan(read_output=512)
    task = plan.steps[0]
    inputs = ByteTemplate().render(step_messages(plan, task, (unit,))).input_ids
    plan = replace(plan, budget=replace(plan.budget, max_total_tokens=13+len(inputs)+task.output_budget))
    engine = ScriptedReadingEngine([read_output(fact("a"))])
    store = SessionStore(tmp_path/"state.db")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s", "r", "hash", plan.task_state.raw_question, 0)
    executor = ReadingExecutor(store, engine, ByteTemplate())
    engine.drop_next_generate_response = True
    with pytest.raises(UnknownOutcomeError):
        await executor.execute_reads(attempt.attempt_id, snapshot, plan, initial_auxiliary_usage=usage(10, 3))
    outcome = await executor.execute_reads(attempt.attempt_id, snapshot,
        store.context_plan(attempt.attempt_id), initial_auxiliary_usage=usage(10, 3))
    assert len(engine.constraints) == 1
    assert outcome.plan.online_tokens_spent == 13+len(inputs)+len(outcome.steps[0].output_ids)
    assert outcome.plan.online_tokens_spent <= plan.budget.max_total_tokens
