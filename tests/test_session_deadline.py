"""Deadlines cover retrieval and generation while retaining owned-work cleanup."""
import asyncio
from pathlib import Path
import threading
import time

import pytest

from adapters.mtrag import load_rows
from bench.run_sessions import run_sessions


ROWS = load_rows(Path(__file__).parent / "fixtures" / "mtrag_small.jsonl")[:3]


@pytest.mark.parametrize("slow_stage", ["retrieval", "context"])
async def test_p3_deadline_drains_synchronous_work_and_never_generates(slow_stage):
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
            return await run_sessions(ROWS, handler, protocol="P3", retriever=Retriever(),
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
