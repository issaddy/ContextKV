from fastapi.testclient import TestClient

from controller.api import create_app
from controller.config import RunConfig, create_controller
from dataclasses import replace
import asyncio

import httpx
import pytest

from controller.planner import PlannerKnobs


def test_http_retry_rebuilds_future_gap_callback_after_controller_restart(tmp_path):
    from adapters.study_retrieval import StudyEvidenceProvider
    from controller.context_schema import ContextBudget, EvidencePool, EvidenceSet, Need, TaskState
    from controller.reading import build_read_plan
    from controller.service import Controller, EvidenceSelection
    from controller.session_store import SessionStore
    from controller.templates import ByteTemplate
    from test_reading_runtime import ScriptedReadingEngine, fact, material, read_output

    first, a = material("A belongs to B.", "a")
    second, b = material("B is funded by C.", "b")
    third, c = material("C started in 2001.", "c")
    state = TaskState("task", "When did the fund supporting A's parent start?", needs=(Need("n"),))
    plan = build_read_plan(state, EvidenceSet("set", (a,), source_blocks=(first,)),
                           "selected_sources", ContextBudget(), template=ByteTemplate())
    first_gap = {"need_id": "n", "missing_relation": "fund", "bound_entities": ["B"],
                 "temporal_scope": "current", "parent_claim_ids": ["read:0:claim:0"]}
    second_gap = {"need_id": "n", "missing_relation": "start date", "bound_entities": ["C"],
                  "temporal_scope": "current", "parent_claim_ids": ["hop:1:read:0:claim:0"]}
    second_fact, third_fact = fact("b", "B"), fact("c", "C", "2001")
    second_fact["parent_claim_ids"] = ["read:0:claim:0"]
    third_fact["parent_claim_ids"] = ["hop:1:read:0:claim:0"]

    class LostFirstHop(ScriptedReadingEngine):
        lost = False
        async def generate(self, **kwargs):
            if ":hop:1:" in kwargs["attempt_id"] and not self.lost:
                self.lost = True
                self.drop_next_generate_response = True
            return await super().generate(**kwargs)

    engine = LostFirstHop([read_output(fact("a", "A"), support=(), open_needs=(first_gap,)),
        read_output(second_fact, support=(), open_needs=(second_gap,)), read_output(third_fact)])
    zero = {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []}

    class Provider(StudyEvidenceProvider):
        corpus_snapshot_id = "v1"
        def __init__(self, restarted=False):
            self.restarted, self.retrieve_calls, self.gap_calls = restarted, 0, []
        @property
        def manifest(self):
            return {"kind": "test_fixture"}
        async def retrieve(self, snapshot, question, request_id):
            assert not self.restarted
            self.retrieve_calls += 1
            async def extend(plan, needs, claims):
                self.gap_calls.append(needs)
                return EvidencePool((b,), (second,), auxiliary_usage=zero)
            return EvidenceSelection((first,), (first.key,), context_plans=(plan,),
                                     open_need_provider=extend, auxiliary_usage=zero)
        async def extend_for_plan(self, snapshot, plan, needs, claims):
            assert self.restarted and snapshot.version == 0
            assert {unit.unit_id for unit in plan.selected_set.selected_units} == {"a", "b"}
            assert needs[0].bound_entities == ("C",)
            self.gap_calls.append(needs)
            return EvidencePool((c,), (third,), auxiliary_usage=zero)
        def close(self):
            pass

    database = tmp_path/"state.db"
    original = Provider()
    controller = Controller(SessionStore(database), engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    payload = {"query": state.raw_question, "request_id": "r"}
    with TestClient(create_app(controller, retrieval=original)) as client:
        assert client.post("/sessions/s/query", json=payload).status_code == 503
        assert controller.store.get_snapshot("s").version == 0
    restarted = Provider(restarted=True)
    recovered = Controller(SessionStore(database), engine, knobs=PlannerKnobs(strategy="always_rebuild"))
    with TestClient(create_app(recovered, retrieval=restarted)) as client:
        response = client.post("/sessions/s/query", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["read_count"] == 3
        assert original.retrieve_calls == 1 and restarted.retrieve_calls == 0
        assert len(original.gap_calls) == len(restarted.gap_calls) == 1
        snapshot = recovered.store.get_snapshot("s")
        assert snapshot.version == 1 and len(snapshot.raw_history) == 1


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
