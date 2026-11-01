"""Deadlines cover retrieval and generation while retaining owned-work cleanup."""
import asyncio
from copy import deepcopy
from pathlib import Path
import threading
import time

import pytest

from adapters.mtrag import load_rows
from bench.run_sessions import run_sessions


ROWS = load_rows(Path(__file__).parent / "fixtures" / "mtrag_small.jsonl")[:3]


def two_sessions(turns):
    rows = []
    for session in ("first", "second"):
        for original in ROWS[:turns]:
            row = deepcopy(original)
            row["conversation_id"] = session
            row["task_id"] = f"{session}<::>{row['turn']}"
            rows.append(row)
    return rows


async def test_open_loop_session_queue_times_out_before_the_other_session_finishes_thinking():
    calls = []

    async def handler(turn):
        calls.append((turn.conversation_id, turn.turn))
        return "answer"

    report = await run_sessions(two_sessions(2), handler, concurrency=1, arrival_rate=1_000_000,
                                think_time=.2, per_turn_timeout=.03)
    failed, = [row for row in report.turns if row.status == "timeout"]
    completed = [row for row in report.turns if row.status == "success"]
    second_turn, = [row for row in completed if row.turn == 2]
    assert failed.turn == 1 and failed.t_done < second_turn.t_arrive
    assert failed.t_done - failed.t_arrive < .15
    assert failed.conversation_id not in {session for session, _ in calls}
    assert report.unfinished_turns == 1


async def test_turn_scope_releases_pipeline_slot_during_think_time_and_keeps_history():
    calls = []
    active, peak = 0, 0

    async def handler(turn):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append((turn.conversation_id, turn.turn, tuple(m.text for m in turn.history)))
        await asyncio.sleep(.005)
        active -= 1
        return f"answer-{turn.conversation_id}-{turn.turn}"

    report = await run_sessions(two_sessions(2), handler, concurrency=1, concurrency_scope="turn", think_time=.08)
    assert report.concurrency_scope == "turn" and peak == 1
    assert len(report.turns) == 4 and all(row.status == "success" for row in report.turns)
    assert {calls[0][0], calls[1][0]} == {"first", "second"}
    for session, turn, history in calls:
        if turn == 2:
            assert history[-1] == f"answer-{session}-1"


async def test_turn_scope_deadline_includes_waiting_for_the_pipeline_slot():
    calls = []

    async def handler(turn):
        calls.append(turn.task_id)
        await asyncio.sleep(1)
        return "too late"

    report = await run_sessions(two_sessions(1), handler, concurrency=1, concurrency_scope="turn",
                                arrival_rate=1_000_000, per_turn_timeout=.03)
    assert len(report.turns) == 2 and all(row.status == "timeout" for row in report.turns)
    assert len(calls) == 1
    assert all(row.t_done - row.t_arrive < .15 for row in report.turns)
    assert report.unfinished_turns == 0


async def test_open_loop_arrival_counts_event_loop_delay_in_deadline():
    calls = []

    async def handler(turn):
        calls.append(turn.task_id)
        return "answer"

    def sink(result):
        if result.status == "success":
            time.sleep(.06)

    report = await run_sessions(two_sessions(1), handler, concurrency=2, arrival_rate=1_000_000,
                                per_turn_timeout=.02, on_result=sink)
    failed, = [row for row in report.turns if row.status == "timeout"]
    assert failed.t_done - failed.t_arrive >= .05
    assert failed.task_id not in calls


@pytest.mark.parametrize("slow_stage", ["retrieval", "context"])
@pytest.mark.parametrize("scope", ["session", "turn"])
async def test_p3_deadline_drains_synchronous_work_and_never_generates(slow_stage, scope):
    started, release = threading.Event(), threading.Event()
    actions = []

    def slow():
        started.set()
        assert release.wait(3)
        actions.append("worker_finished")

    class Retriever:
        def retrieve_turn(self, turn, *, prior_context=None):
            if slow_stage == "retrieval":
                slow()
            return turn

    def context(turn):
        if slow_stage == "context":
            slow()
        return None

    async def handler(turn):
        actions.append("generated")
        return "must not execute"

    async def owner():
        try:
            return await run_sessions(ROWS, handler, protocol="P3", retriever=Retriever(), concurrency_scope=scope,
                                      retrieval_context_provider=context, per_turn_timeout=.01)
        finally:
            actions.append("closed")

    task = asyncio.create_task(owner())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        await asyncio.sleep(.05)
        assert not task.done()  # Cancellation waits for the worker before close.
        assert actions == []
    finally:
        release.set()
        report = await task
    assert actions == ["worker_finished", "closed"]
    assert [row.status for row in report.turns] == ["timeout"]
    assert report.turns[0].assistant_text is None
    assert report.unfinished_turns == 2


async def test_p3_generation_receives_only_the_remaining_turn_budget():
    generated = asyncio.Event()
    cancelled = []

    class Retriever:
        def retrieve_turn(self, turn):
            time.sleep(.03)
            return turn

    async def handler(turn):
        generated.set()
        try:
            await asyncio.sleep(.03)
        except asyncio.CancelledError:
            cancelled.append(turn.task_id)
            raise
        return "would fit a fresh generation budget"

    report = await run_sessions(ROWS, handler, protocol="P3", retriever=Retriever(), per_turn_timeout=.05)
    assert generated.is_set()
    assert cancelled == [ROWS[0]["task_id"]]
    assert [row.status for row in report.turns] == ["timeout"]
    assert report.turns[0].t_retrieval_done is not None
    assert report.unfinished_turns == 2


async def test_returned_committed_result_and_its_sink_survive_turn_deadline():
    observed = []

    async def handler(turn):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            # A durable handler may finish an in-flight commit while draining.
            # Its returned committed answer must not be turned into retryable work.
            return "committed answer"

    async def sink(result):
        await asyncio.sleep(.03)
        observed.append((result.status, result.assistant_text))

    report = await run_sessions(ROWS[:1], handler, per_turn_timeout=.01, on_result=sink)
    assert observed == [("success", "committed answer")]
    assert report.turns[0].assistant_text == "committed answer"
    assert report.unfinished_turns == 0


async def test_drain_during_committed_result_publication_never_duplicates_turn():
    async def handler(turn):
        return "committed answer"

    async def sink(result):
        await asyncio.sleep(1)

    report = await run_sessions(ROWS, handler, per_turn_timeout=.01,
                                drain_timeout=.03, on_result=sink)
    assert [row.status for row in report.turns] == ["success"]
    assert report.unfinished_turns == 2
    assert report.timed_out_sessions == ("session-a",)
