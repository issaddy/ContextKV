from fastapi.testclient import TestClient

from controller.api import create_app
from controller.config import RunConfig, create_controller
from dataclasses import replace
import asyncio

import httpx
import pytest

from controller.planner import PlannerKnobs


def test_http_lifecycle_and_whitelist(tmp_path):
    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path)))
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["engine"]["engine_kind"] == "mock"
        assert client.get("/sessions/not-created").status_code == 404
        payload = {"query": "timeout?", "request_id": "r1", "evidence": [
            {"document_id": "a", "text": "timeout=30", "source_uri": "docs://a"}]}
        first = client.post("/sessions/s/turns", json=payload)
        assert first.status_code == 200
        assert first.json()["action"] == "REBUILD"
        assert client.post("/sessions/s/turns", json=payload).json()["idempotent_replay"]
        payload["targets"] = "must never enter serving"
        assert client.post("/sessions/s/turns", json=payload).status_code == 422
        state = client.get("/sessions/s").json()
        assert len(state["raw_history"]) == 1
        assert client.post("/sessions/s/close").status_code == 200
        assert client.post("/sessions/s/turns", json={"query": "q2", "request_id": "r2"}).status_code == 409


@pytest.mark.parametrize("failure", ["ack_mismatch", "task_cancelled", "program_error"])
async def test_periodic_retirement_recovers_without_new_requests(tmp_path, monkeypatch, failure):
    monkeypatch.setattr("controller.api.RETIRE_RETRY_SECONDS", .01)
    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path),
                                       planner=PlannerKnobs(strategy="always_rebuild")))
    original_retire, original_drain = service.engine.retire, service.drain
    retry_started, release_retry, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def retire(*args):
        calls.append(args)
        if len(calls) == 1:
            if failure == "task_cancelled":
                raise asyncio.CancelledError()
            if failure == "program_error":
                raise RuntimeError("injected maintenance failure")
            return replace(await original_retire(*args), operation_id="wrong-operation")
        retry_started.set()
        await release_retry.wait()
        return await original_retire(*args)

    async def drain():
        await original_drain()
        if len(calls) > 1 and not service.store.outbox():
            drained.set()

    monkeypatch.setattr(service.engine, "retire", retire)
    monkeypatch.setattr(service, "drain", drain)
    app = create_app(service)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await service.serve_turn("s", "first", request_id="1")
            await original_drain()
            await service.serve_turn("s", "second", request_id="2")
            try:
                await asyncio.wait_for(retry_started.wait(), 2)
                response = await client.get("/health")
                assert response.status_code == 503
                assert response.json()["status"] == "degraded"
                assert response.json()["retirement_maintenance"]["running"] is True
                assert response.json()["retirement_maintenance"]["error"]
                assert len(service.store.outbox()) == 1
                assert len(service.store.get_snapshot("s").raw_history) == 2
            finally:
                release_retry.set()
            await asyncio.wait_for(drained.wait(), 2)
            assert calls[0] == calls[1]  # Retry the same durable operation and generation.
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert response.json()["retirement_maintenance"] == {"running": True, "error": None}
            assert response.json()["pending_retire_operations"] == 0
