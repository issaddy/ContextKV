"""Dependent session workloads: serial turns, concurrent sessions, explicit drain.

The scheduling layer owns raw scripted tasks. A serving handler receives only the
current whitelisted OnlineTurn. Public future questions and labels are inaccessible
through that request. Failed turns stop their session; dependent turns are counted
as unfinished, never executed with invented history.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import inspect
import random
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from adapters.mtrag import OnlineSession, OnlineTurn, Protocol, group_sessions


@dataclass(frozen=True)
class TurnRunResult:
    conversation_id: str
    task_id: str
    turn: int
    protocol: str
    status: str
    t_arrive: float
    t_done: float
    assistant_text: str | None = None
    error: str | None = None
    result: Any = None
    online_turn: OnlineTurn | None = None
    t_retrieval_done: float | None = None


@dataclass(frozen=True)
class RunReport:
    mode: str
    seed: int
    started_at: float
    arrival_window_end: float
    completed_at: float
    turns: tuple[TurnRunResult, ...]
    scheduled_sessions: int
    arrived_sessions: int
    unfinished_turns: int
    timed_out_sessions: tuple[str, ...]
    session_makespans: tuple[tuple[str, float], ...]
    time_origin: str

    @property
    def drain_seconds(self) -> float:
        return max(0.0, self.completed_at - self.arrival_window_end)


def _answer(result: Any) -> str:
    if isinstance(result, str):
        return result
    for name in ("assistant_text", "text", "answer"):
        value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
        if isinstance(value, str):
            return value
    raise ValueError("handler must return persisted assistant_text (or a string)")


async def run_sessions(sessions: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
                       handler: Callable[[OnlineTurn], Awaitable[Any]], *, protocol: Protocol = "P2",
                       concurrency: int = 1, think_time: float = 0.0, arrival_rate: float | None = None,
                       seed: int = 42, arrival_window: float | None = None,
                       drain_timeout: float | None = None, per_turn_timeout: float | None = None,
                       retriever: Any | None = None,
                       on_result: Callable[[TurnRunResult], Any] | None = None) -> RunReport:
    """Run a fixed cohort or Poisson session arrivals, then drain all arrived work.

    concurrency bounds active sessions, including their think time. For open-loop
    load, arrivals continue independently while excess sessions wait for a slot;
    their first-turn response time includes that wait. No future turn is submitted
    until the preceding result is successfully committed by handler. Timing uses
    the local monotonic clock; no unaligned wall-clock subtraction is performed.
    """
    if concurrency < 1 or think_time < 0:
        raise ValueError("invalid concurrency or think_time")
    for name, value in (("arrival_rate", arrival_rate), ("arrival_window", arrival_window),
                        ("drain_timeout", drain_timeout), ("per_turn_timeout", per_turn_timeout)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if protocol == "P3" and retriever is None:
        raise ValueError("P3 requires a real retriever")
    grouped = dict(sessions) if isinstance(sessions, Mapping) else group_sessions(sessions)
    if any(str(sid) != str(turns[0]["conversation_id"]) for sid, turns in grouped.items() if turns):
        raise ValueError("session key must equal conversation_id")
    ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(ids)
    offsets, offset = [], 0.0
    for sid in ids:
        if arrival_rate is not None:
            offset += rng.expovariate(arrival_rate)
        if arrival_window is not None and offset > arrival_window:
            break
        offsets.append((sid, offset))
    started = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[TurnRunResult] = []
    makespans: dict[str, float] = {}
    unfinished: dict[str, int] = {}
    arrived: set[str] = set()
    timed_out: set[str] = set()
    arrival_end = started + (arrival_window if arrival_window is not None else (offsets[-1][1] if offsets else 0))

    async def publish(item: TurnRunResult):
        results.append(item)
        if on_result is not None:
            callback_result = on_result(item)
            if inspect.isawaitable(callback_result):
                await callback_result

    async def one_session(sid: str, delay: float):
        gate = OnlineSession(grouped[sid], protocol)
        unfinished[sid] = gate.remaining
        await asyncio.sleep(max(0.0, started + delay - time.monotonic()))
        arrived.add(sid)
        session_arrive = time.monotonic()
        active: OnlineTurn | None = None
        turn_arrive = session_arrive
        try:
            async with semaphore:
                if arrival_rate is None:
                    session_arrive = turn_arrive = time.monotonic()
                while not gate.done:
                    active = gate.current()
                    retrieval_done = None
                    try:
                        if protocol == "P3":
                            active = await asyncio.to_thread(retriever.retrieve_turn, active)
                        retrieval_done = time.monotonic()
                        operation = handler(active)
                        result = await asyncio.wait_for(operation, per_turn_timeout) if per_turn_timeout else await operation
                        if (isinstance(result, Mapping) and result.get("success") is False) or getattr(result, "success", True) is False:
                            raise RuntimeError("handler reported unsuccessful generation")
                        answer = _answer(result)
                        gate.commit(active.task_id, answer)
                        unfinished[sid] = gate.remaining
                        completed = TurnRunResult(sid, active.task_id, active.turn, protocol, "success",
                                                  turn_arrive, time.monotonic(), answer, result=result,
                                                  online_turn=active, t_retrieval_done=retrieval_done)
                    except Exception as exc:
                        status = "timeout" if isinstance(exc, TimeoutError) else "failed"
                        failed = TurnRunResult(sid, active.task_id, active.turn, protocol, status,
                                               turn_arrive, time.monotonic(), error=f"{type(exc).__name__}: {exc}",
                                               online_turn=active, t_retrieval_done=retrieval_done)
                        # The failed task is accounted separately from dependent unfinished work.
                        unfinished[sid] = max(0, gate.remaining - 1)
                        active = None
                        await publish(failed)
                        break
                    # A sink failure must not reclassify an already committed generation.
                    active = None
                    await publish(completed)
                    if not gate.done:
                        await asyncio.sleep(think_time)
                        turn_arrive = time.monotonic()
                makespans[sid] = time.monotonic() - session_arrive
        except asyncio.CancelledError:
            timed_out.add(sid)
            if active is not None:
                await publish(TurnRunResult(sid, active.task_id, active.turn, protocol, "cancelled",
                                            turn_arrive, time.monotonic(), error="cohort drain timeout", online_turn=active))
                unfinished[sid] = max(0, gate.remaining - 1)
            makespans[sid] = time.monotonic() - session_arrive
            raise

    tasks = [asyncio.create_task(one_session(sid, delay)) for sid, delay in offsets]
    try:
        if tasks:
            # Fixed-work closed-loop has no prescribed observation window: complete the cohort.
            if drain_timeout is None:
                await asyncio.gather(*tasks)
            else:
                deadline = max(0.0, arrival_end - time.monotonic()) + drain_timeout
                done, pending = await asyncio.wait(tasks, timeout=deadline)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in done:
                    if task.exception():
                        raise task.exception()
            if arrival_window is not None:
                # A finite cohort may finish before the fixed observation window.
                # Keep the agreed denominator/window, without inventing more sessions.
                await asyncio.sleep(max(0.0, arrival_end - time.monotonic()))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    completed = time.monotonic()
    if arrival_rate is None and arrival_window is None:
        arrival_end = completed
    return RunReport("session_open_loop" if arrival_rate is not None else "closed_loop", seed,
                     started, arrival_end, completed, tuple(sorted(results, key=lambda r: (r.conversation_id, r.turn))),
                     len(offsets), len(arrived), sum(unfinished.values()), tuple(sorted(timed_out)),
                     tuple(sorted(makespans.items())),
                     "pipeline_arrival" if protocol == "P3" else "fixed_retrieval_input_ready")
