"""Opt-in task/Future observations; no retries, deadlines or request mutation."""
from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
import inspect
import json
import os
from pathlib import Path
import signal
import time


def _identity(value):
    return hex(id(value))


def _future(value):
    return isinstance(value, (asyncio.Future, ConcurrentFuture))


def _callback(callback):
    base = callback.func if isinstance(callback, partial) else callback
    references = [value for value in callback.args if _future(value)] if isinstance(callback, partial) else []
    owner = getattr(base, "__self__", None)
    if _future(owner):
        references.append(owner)
    if isinstance(owner, asyncio.TaskGroup) and owner._parent_task is not None:
        references.append(owner._parent_task)
    function = getattr(base, "__func__", base)
    if inspect.isfunction(function):
        references.extend(value for value in inspect.getclosurevars(function).nonlocals.values() if _future(value))
    return {"callback": getattr(base, "__qualname__", type(base).__qualname__),
            "future_references": [_identity(value) for value in references]}, references


def _await_chain(coroutine):
    chain, seen = [], set()
    while coroutine is not None and id(coroutine) not in seen:
        seen.add(id(coroutine))
        frame = (getattr(coroutine, "cr_frame", None) or getattr(coroutine, "gi_frame", None)
                 or getattr(coroutine, "ag_frame", None))
        node = {"type": type(coroutine).__qualname__, "id": _identity(coroutine)}
        if frame is not None:
            node.update(file=frame.f_code.co_filename, line=frame.f_lineno,
                        function=frame.f_code.co_qualname)
        chain.append(node)
        coroutine = (getattr(coroutine, "cr_await", None) or getattr(coroutine, "gi_yieldfrom", None)
                     or getattr(coroutine, "ag_await", None))
    return chain


def capture_async_state(*, state_provider=None):
    """Read suspended await chains and Future callback links without results/locals."""
    loop = asyncio.get_running_loop()
    pending = list(asyncio.all_tasks(loop))
    nodes, seen = [], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        node = {"id": _identity(value), "type": type(value).__qualname__,
                "module": type(value).__module__, "state": value._state}
        error = getattr(value, "_exception", None)
        node["exception_type"] = type(error).__qualname__ if error is not None else None
        node["result_type"] = type(getattr(value, "_result", None)).__qualname__ if value._state == "FINISHED" else None
        if isinstance(value, asyncio.Task):
            waiter = value._fut_waiter
            node.update(name=value.get_name(), cancelling=value.cancelling(),
                await_chain=_await_chain(value.get_coro()), waiter=_identity(waiter) if waiter is not None else None)
            if waiter is not None:
                pending.append(waiter)
        callbacks = (value._done_callbacks if isinstance(value, ConcurrentFuture) else value._callbacks) or ()
        callback_records = []
        for entry in callbacks:
            callback = entry if isinstance(value, ConcurrentFuture) else entry[0]
            description, references = _callback(callback)
            callback_records.append(description)
            pending.extend(references)
        node["callbacks"] = callback_records
        nodes.append(node)
    executor = loop._default_executor
    pool = None if executor is None else {"type": type(executor).__qualname__,
        "shutdown": executor._shutdown, "broken": bool(executor._broken),
        "queued_work_items": executor._work_queue.qsize(),
        "threads": [{"name": thread.name, "ident": thread.ident, "native_id": thread.native_id,
                     "alive": thread.is_alive()} for thread in executor._threads]}
    return {"pid": os.getpid(), "monotonic_ns": time.monotonic_ns(), "loop_debug": loop.get_debug(),
        "futures": sorted(nodes, key=lambda node: node["id"]), "default_executor": pool,
        "application": state_provider() if state_provider is not None else None,
        "scope": "task_awaits_future_states_callback_dependencies_no_user_locals_or_results"}


def study_wait_state(service, provider):
    pending = service.store.db.execute("SELECT attempt_id,session_id,request_id,stage FROM attempts "
        "WHERE stage NOT IN ('COMMITTED','FAILED') ORDER BY rowid").fetchall()
    retirement = service._retire_task
    return {"pending_attempts": [dict(row) for row in pending],
        "session_locks": {session: {"locked": lock.locked(),
            "waiters": [_identity(waiter) for waiter in lock._waiters or ()]}
            for session, lock in service._locks.items()},
        "provider_sync_lock_locked": provider._lock.locked(),
        "retirement_lock_locked": service._retire_lock.locked(),
        "retirement_task": _identity(retirement) if retirement is not None else None,
        "retirement_task_state": retirement._state if retirement is not None else None,
        "retirement_requested": service._retire_requested,
        "outbox_count": len(service.store.outbox()), "active_generations": service._active_generations}


class AsyncTaskDump:
    def __init__(self, path, *, state_provider=None, signum=signal.SIGUSR2):
        self.loop = asyncio.get_running_loop()
        self.path, self.state_provider, self.signum = Path(path), state_provider, signum
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous_signal = signal.getsignal(signum)
        if self.previous_signal != signal.SIG_DFL:
            raise ValueError("async diagnostic signal already has a handler")
        self.loop.add_signal_handler(signum, self.dump, signal.Signals(signum).name)
        self.active = True
        with self.path.open("a") as output:
            output.write(json.dumps({"event": "async_dump_installed", "pid": os.getpid(),
                "signal": signal.Signals(signum).name, "path": str(self.path.resolve())})+"\n")

    def dump(self, reason="manual"):
        if not self.active:
            return
        record = {"event": "async_task_dump", "reason": reason,
                  **capture_async_state(state_provider=self.state_provider)}
        with self.path.open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False)+"\n")

    def close(self):
        self.active = False
        self.loop.remove_signal_handler(self.signum)


def install_from_env(*, state_provider=None):
    path = os.environ.get("CONTEXTKV_ASYNC_DUMP")
    return AsyncTaskDump(path, state_provider=state_provider) if path else None
