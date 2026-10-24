"""HTTP shutdown retains resources until request cancellation/commit has finished."""
import asyncio
from dataclasses import replace
from threading import Event

import httpx
import pytest

from adapters.mtrag import OnlineContext
from adapters.serving_retrieval import ServingRetriever
from controller.api import create_app
from controller.config import RunConfig, create_controller
from controller.session_store import SessionStore


@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_shutdown_drains_cancelled_retrieval_before_closing_store(tmp_path, cancel_shutdown):
    entered, release = Event(), Event()
    actions = []

    class Retriever:
        corpus_snapshot_id = "snapshot"
        manifest = {"kind": "blocked-cpu-fixture"}

        def retrieve_turn(self, turn):
            entered.set()
            assert release.wait(5), "test did not release retrieval"
            actions.append("retrieved")
            return replace(turn, contexts=(OnlineContext("p", "Original source."),))

        def close(self):
            actions.append("closed")

    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path)))
    app = create_app(service, retrieval=ServingRetriever(Retriever(), corpus_id="manual"))
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        request = asyncio.create_task(client.post("/sessions/s/query",
            json={"query": "Question?", "request_id": "q"}))
        shutdown = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            request.cancel()  # Uvicorn cancels timed-out requests without awaiting them.
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(0)
            if cancel_shutdown:
                shutdown.cancel()
                await asyncio.sleep(0)
            assert not request.done() and not shutdown.done()
            assert not app.state.ready and actions == []
            assert service.store.find_request("s", "q").stage == "STARTED"
            for path, body in (("/sessions/new/query", {"query": "q", "request_id": "new"}),
                               ("/sessions/new/turns", {"query": "q", "request_id": "new"}),
                               ("/sessions/s/close", {})):
                response = await client.post(path, json=body)
                assert response.status_code == 503
                assert response.json()["code"] == "SERVICE_SHUTTING_DOWN"
        finally:
            release.set()
            results = await asyncio.gather(request, return_exceptions=True)
            if shutdown is not None:
                shutdown_results = await asyncio.gather(shutdown, return_exceptions=True)
            else:
                await lifespan.__aexit__(None, None, None)
        assert isinstance(results[0], asyncio.CancelledError)
        if cancel_shutdown:
            assert isinstance(shutdown_results[0], asyncio.CancelledError)
        else:
            assert shutdown_results == [False]
    assert actions == ["retrieved", "closed"]
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        assert store.find_request("s", "q").stage == "FAILED"
        snapshot = store.get_snapshot("s")
        assert snapshot.version == 0 and snapshot.raw_history == () and snapshot.current_epoch is None
        assert not store.outbox()
    finally:
        store.close()
    assert not any(event["event"] == "generate_complete" for event in service.engine.events)


async def test_shutdown_drains_active_generation_and_queued_session_close(tmp_path):
    config = replace(RunConfig(), state_dir=str(tmp_path))
    config = replace(config, planner=replace(config.planner, strategy="always_rebuild"))
    service = create_controller(config)
    app = create_app(service)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    entered, release = asyncio.Event(), asyncio.Event()
    original_generate = service.engine.generate

    async def blocked_generate(**kwargs):
        entered.set()
        await release.wait()
        return await original_generate(**kwargs)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/sessions/s/turns", json={"query": "First?", "request_id": "1"})
        assert first.status_code == 200
        service.engine.generate = blocked_generate
        request = asyncio.create_task(client.post("/sessions/s/turns",
            json={"query": "Second?", "request_id": "2"}))
        close = shutdown = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            close = asyncio.create_task(client.post("/sessions/s/close"))
            await asyncio.sleep(0)  # Accept close before shutdown; it waits on the session lock.
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(0)
            assert not request.done() and not close.done() and not shutdown.done()
            assert service.store.find_request("s", "2").stage == "SUBMITTED"
        finally:
            release.set()
            responses = await asyncio.gather(request, *([close] if close is not None else []))
            if shutdown is not None:
                await shutdown
            else:
                await lifespan.__aexit__(None, None, None)
        assert all(response.status_code == 200 for response in responses)
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        assert store.find_request("s", "2").stage == "COMMITTED"
        assert len(store.get_snapshot("s").raw_history) == 2
        assert store.db.execute("SELECT closed FROM sessions WHERE session_id='s'").fetchone()[0]
        assert not store.outbox("s")
    finally:
        store.close()
    assert sum(event["event"] == "generate_complete" for event in service.engine.events) == 2
