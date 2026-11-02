import asyncio
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import replace
import json
import signal
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from adapters.retrieval import run_retrieval_worker
from bench.async_diagnostics import capture_async_state, install_from_env, study_wait_state
from controller.config import RunConfig, create_controller
from controller.service import EvidenceSelection


async def test_snapshot_shows_await_chain_and_future_bridge_without_user_locals():
    source = ConcurrentFuture()
    destination = asyncio.wrap_future(source)
    async def wait_for_result():
        private_question = "PRIVATE_QUESTION_MUST_NOT_BE_DUMPED"
        value = await destination
        return private_question, value
    task = asyncio.create_task(wait_for_result(), name="test-waiting-on-bridge")
    try:
        await asyncio.sleep(0)
        # Observe the normal source-complete / delivery-pending boundary before
        # allowing the event loop to run its cross-thread transfer callback.
        source.set_result("PRIVATE_RESULT_MUST_NOT_BE_DUMPED")
        dump = capture_async_state()
        by_id = {node["id"]: node for node in dump["futures"]}
        current = next(node for node in dump["futures"] if node.get("name") == "test-waiting-on-bridge")
        assert current["state"] == "PENDING" and current["waiter"] == hex(id(destination))
        assert any(item.get("function", "").endswith("wait_for_result") for item in current["await_chain"])
        assert by_id[hex(id(source))]["state"] == "FINISHED"
        assert by_id[hex(id(destination))]["state"] == "PENDING"
        assert "PRIVATE_" not in json.dumps(dump)
    finally:
        await task


async def test_controlled_undelivered_thread_result_reproduces_started_with_idle_executor(tmp_path, monkeypatch):
    """Injected delivery hold reproduces the signature, not the production cause."""
    loop = asyncio.get_running_loop()
    real_submit = loop.run_in_executor
    delivered = asyncio.Event()
    held = loop.create_future()
    submitted, operations = [], []
    def hold_delivery(executor, operation, *arguments):
        actual = real_submit(executor, operation, *arguments)
        submitted.append(actual)
        actual.add_done_callback(lambda _: delivered.set())
        def observe_source(_):
            return actual.done()
        held.add_done_callback(observe_source)
        return held
    monkeypatch.setattr(loop, "run_in_executor", hold_delivery)
    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path)))
    provider = SimpleNamespace(_lock=Lock())
    def synchronous_retrieval():
        operations.append("retrieved_once")
        return EvidenceSelection((), ())
    async def retrieve(snapshot):
        return await run_retrieval_worker(synchronous_retrieval)
    request = asyncio.create_task(service.serve_turn("s", "specified question", request_id="q",
        evidence_provider=retrieve), name="controlled-started-retrieval")
    try:
        await asyncio.wait_for(delivered.wait(), 2)
        attempt = service.store.find_request("s", "q")
        assert attempt.stage == "STARTED" and attempt.query_context is None and not attempt.evidence
        assert submitted[0].done() and operations == ["retrieved_once"]
        dump = capture_async_state(state_provider=lambda: study_wait_state(service, provider))
        assert dump["default_executor"]["queued_work_items"] == 0
        assert dump["application"]["pending_attempts"][0]["stage"] == "STARTED"
        assert dump["application"]["outbox_count"] == 0
        assert dump["application"]["retirement_lock_locked"] is False
        chain = next(node for node in dump["futures"] if node.get("name") == "controlled-started-retrieval")["await_chain"]
        assert any(item.get("function") == "Controller.serve_turn" for item in chain)
        assert any(item.get("function") == "run_retrieval_worker" for item in chain)
        assert any(node["state"] == "FINISHED" and node["result_type"] == "EvidenceSelection" for node in dump["futures"])
    finally:
        monkeypatch.setattr(loop, "run_in_executor", real_submit)
        held.set_result(submitted[0].result())
        await request
        await service.aclose()
        service.store.close()


async def test_opt_in_dump_signal_registration_and_disabled_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("CONTEXTKV_ASYNC_DUMP", raising=False)
    assert install_from_env() is None
    path = tmp_path/"async.jsonl"
    monkeypatch.setenv("CONTEXTKV_ASYNC_DUMP", str(path))
    previous = signal.getsignal(signal.SIGUSR2)
    diagnostic = install_from_env(state_provider=lambda: {"stage": "specified-observation"})
    try:
        diagnostic.dump("unit-test-manual-observation")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["event"] == "async_dump_installed"
        assert rows[1]["event"] == "async_task_dump"
        assert rows[1]["application"] == {"stage": "specified-observation"}
    finally:
        diagnostic.close()
    assert signal.getsignal(signal.SIGUSR2) == previous


async def test_cancelled_retrieval_is_started_only_while_its_sync_worker_is_still_draining(tmp_path):
    entered, release = Event(), Event()
    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path)))
    provider = SimpleNamespace(_lock=Lock())
    def operation():
        with provider._lock:
            entered.set()
            assert release.wait(3), "test must release its explicitly blocked worker"
            return EvidenceSelection((), ())
    async def retrieve(snapshot):
        return await run_retrieval_worker(operation)
    request = asyncio.create_task(service.serve_turn("s", "specified cancellation", request_id="q",
        evidence_provider=retrieve), name="controlled-cancelled-retrieval")
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        request.cancel()
        await asyncio.sleep(0)
        observed = capture_async_state(state_provider=lambda: study_wait_state(service, provider))
        node = next(node for node in observed["futures"] if node.get("name") == "controlled-cancelled-retrieval")
        assert node["cancelling"] == 1 and observed["application"]["pending_attempts"][0]["stage"] == "STARTED"
        assert observed["application"]["provider_sync_lock_locked"] is True
        assert any(item["type"] == "Future" and item["state"] == "RUNNING" for item in observed["futures"])
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert service.store.find_request("s", "q").stage == "FAILED"
        assert service.store.get_snapshot("s").raw_history == ()
        await service.aclose()
        service.store.close()
