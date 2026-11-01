"""Snapshot changes retain history and commit with a rebuilt native epoch."""
import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from adapters.corpus import CorpusIndex
from adapters.retrieval import BM25SRetriever
from adapters.serving_retrieval import ServingRetriever
from bench.mock_engine import MockEngine
from controller.api import create_app
from controller.engine_client import UnknownOutcomeError
from controller.evidence import EvidenceError, evidence_index
from controller.planner import PlannerKnobs
from controller.service import Controller
from controller.session_store import IdempotencyConflict, PendingAttempt, SessionStore, VersionConflict


def retriever(snapshot, revision, limit):
    rows = [
        {"passage_id": "limit", "document_id": "manual", "version": revision,
         "text": f"The upload limit is {limit} MB.", "source": "docs://manual/limit"},
        {"passage_id": "condition", "document_id": "manual", "version": "stable",
         "text": "The upload limit applies to Linux clients.", "source": "docs://manual/condition"},
    ]
    corpus = {row["passage_id"]: row for row in rows}
    sparse = BM25SRetriever(corpus, corpus_snapshot_id=snapshot)
    index = CorpusIndex.from_records(rows, corpus_id="manuals", snapshot_id=snapshot)
    return ServingRetriever(sparse, corpus_id="manuals", corpus_index=index)


def controller(tmp_path, *, engine=None, epoch_retirement=True):
    store = SessionStore(tmp_path / "state.sqlite3")
    service = Controller(store, engine or MockEngine(), knobs=PlannerKnobs(strategy="always_append"),
                         epoch_retirement=epoch_retirement)
    return service, store, service.engine


async def send(service, adapter, request_id, query, *, previous=None):
    return await service.serve_turn("s", query, request_id=request_id,
        corpus_snapshot_id=adapter.corpus_snapshot_id, from_corpus_snapshot_id=previous,
        evidence_provider=lambda snapshot: adapter.retrieve(snapshot, query, request_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("epoch_retirement", [True, False])
async def test_transition_commits_history_new_snapshot_and_epoch_together(tmp_path, monkeypatch, epoch_retirement):
    service, store, engine = controller(tmp_path, epoch_retirement=epoch_retirement)
    old, new = retriever("snapshot-a", "r1", 640), retriever("snapshot-b", "r2", 720)
    first = await send(service, old, "first", "What is the upload limit?")
    before = store.get_snapshot("s")
    with pytest.raises(ValueError, match="explicitly provide"):
        await send(service, new, "missing-precondition", "What is the current limit?")
    with pytest.raises(VersionConflict):
        await send(service, new, "wrong-precondition", "What is the current limit?", previous="unrelated")
    assert store.get_snapshot("s") == before

    generate = engine.generate
    async def check_before_commit(**kwargs):
        assert store.get_snapshot("s") == before
        assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state != "RETIRED"
        return await generate(**kwargs)
    monkeypatch.setattr(engine, "generate", check_before_commit)
    second = await send(service, new, "change", "What is the current upload limit?", previous="snapshot-a")
    after = store.get_snapshot("s")
    assert second.action == "REBUILD" and second.branch_reason == "corpus_snapshot_change"
    assert after.corpus_snapshot_id == after.execution.corpus_snapshot_id == "snapshot-b"
    assert after.current_epoch.epoch_no > before.current_epoch.epoch_no
    assert [row.user_text for row in after.raw_history] == ["What is the upload limit?", "What is the current upload limit?"]
    assert after.raw_history[0].assistant_text == first.assistant_text
    assert {key.version for key in after.execution.visible_blocks} == {"r2", "stable"}
    condition, = [b for b in after.archive if b.key.passage_id == "condition"]
    assert {"corpus_snapshot:snapshot-a", "corpus_snapshot:snapshot-b"} <= set(condition.provenance)
    assert {b.key.version for b in after.archive if b.key.passage_id == "limit"} == {"r1", "r2"}
    replay = await send(service, new, "first", "What is the upload limit?")
    assert replay.idempotent_replay and replay.attempt_id == first.attempt_id
    with pytest.raises(IdempotencyConflict):
        await send(service, new, "first", "A different request")
    with pytest.raises(VersionConflict):
        await send(service, old, "stale", "Old worker tries to switch back", previous="snapshot-a")
    await service.drain()
    assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state == "RETIRED"
    assert not store.outbox()
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["lost_response", "commit"])
async def test_transition_failure_recovers_without_advancing_snapshot(tmp_path, monkeypatch, fault):
    service, store, engine = controller(tmp_path)
    old, new = retriever("old", "r1", 640), retriever("new", "r2", 720)
    await send(service, old, "first", "Upload limit?")
    before = store.get_snapshot("s")
    commit = store.commit_turn_and_outbox
    if fault == "lost_response":
        engine.drop_next_generate_response = True
    else:
        def fail_commit(_):
            raise VersionConflict("injected commit failure")
        monkeypatch.setattr(store, "commit_turn_and_outbox", fail_commit)
    with pytest.raises(UnknownOutcomeError if fault == "lost_response" else VersionConflict):
        await send(service, new, "change", "What is the current upload limit?", previous="old")
    assert store.get_snapshot("s") == before
    assert store.pending_attempt("s").stage == ("SUBMITTED" if fault == "lost_response" else "RESULT_READY")
    assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state != "RETIRED"
    assert not store.outbox()
    if fault == "commit":
        monkeypatch.setattr(store, "commit_turn_and_outbox", commit)
    restored = (await send(service, new, "change", "What is the current upload limit?", previous="old")
                if fault == "lost_response" else await service.recover("s"))
    assert restored.idempotent_replay and store.get_snapshot("s").corpus_snapshot_id == "new"
    assert len(store.get_snapshot("s").raw_history) == 2
    await service.drain()
    assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state == "RETIRED"
    store.close()


@pytest.mark.asyncio
async def test_failed_new_retrieval_leaves_the_old_snapshot_usable(tmp_path):
    service, store, engine = controller(tmp_path)
    old = retriever("old", "r1", 640)
    await send(service, old, "first", "Upload limit?")
    before = store.get_snapshot("s")
    async def fail(snapshot):
        assert snapshot.corpus_snapshot_id == "new"
        raise RuntimeError("new index unavailable")
    with pytest.raises(RuntimeError, match="new index unavailable"):
        await service.serve_turn("s", "Current limit?", request_id="failed", corpus_snapshot_id="new",
            from_corpus_snapshot_id="old", evidence_provider=fail)
    assert store.get_snapshot("s") == before and store.pending_attempt("s") is None
    assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state != "RETIRED"
    response = await send(service, old, "continue", "What is the Linux condition?")
    assert response.ledger_version == 2 and store.get_snapshot("s").corpus_snapshot_id == "old"
    await service.drain()
    store.close()


@pytest.mark.asyncio
async def test_transition_rolls_back_if_retirement_outbox_write_fails(tmp_path, monkeypatch):
    service, store, engine = controller(tmp_path)
    old, new = retriever("old", "r1", 640), retriever("new", "r2", 720)
    await send(service, old, "first", "Upload limit?")
    before = store.get_snapshot("s")
    enqueue = store._enqueue
    def fail_enqueue(*args, **kwargs):
        raise RuntimeError("outbox transaction failed")
    monkeypatch.setattr(store, "_enqueue", fail_enqueue)
    with pytest.raises(RuntimeError, match="outbox transaction failed"):
        await send(service, new, "change", "What is the current upload limit?", previous="old")
    assert store.get_snapshot("s") == before
    assert store.pending_attempt("s").stage == "RESULT_READY" and not store.outbox()
    assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state != "RETIRED"
    monkeypatch.setattr(store, "_enqueue", enqueue)
    await service.recover("s")
    assert store.get_snapshot("s").corpus_snapshot_id == "new"
    await service.drain()
    assert (await engine.status(engine_session_id=before.current_epoch.engine_session_id)).state == "RETIRED"
    store.close()


@pytest.mark.asyncio
async def test_two_controllers_do_not_migrate_a_pending_or_stale_snapshot(tmp_path):
    first, store, engine = controller(tmp_path)
    other_store = SessionStore(tmp_path / "state.sqlite3")
    other = Controller(other_store, engine, knobs=PlannerKnobs(strategy="always_append"))
    old, new = retriever("a", "r1", 640), retriever("b", "r2", 720)
    await send(first, old, "one", "Upload limit?")
    entered, release = asyncio.Event(), asyncio.Event()
    async def paused(snapshot):
        assert snapshot.corpus_snapshot_id == "b"
        entered.set()
        await release.wait()
        return await new.retrieve(snapshot, "Current upload limit?", "two")
    task = asyncio.create_task(first.serve_turn("s", "Current upload limit?", request_id="two",
        corpus_snapshot_id="b", from_corpus_snapshot_id="a", evidence_provider=paused))
    await entered.wait()
    with pytest.raises(PendingAttempt):
        await send(other, old, "concurrent", "Another question")
    assert store.get_snapshot("s").corpus_snapshot_id == "a"
    release.set()
    await task
    with pytest.raises(VersionConflict):
        await send(other, old, "stale", "Another question", previous="a")
    assert store.get_snapshot("s").corpus_snapshot_id == "b"
    await first.drain()
    other_store.close()
    store.close()


def test_only_snapshot_provenance_may_merge_for_same_identity():
    first = retriever("a", "r1", 640).corpus_index.blocks[0]
    second = replace(first, provenance=tuple(x.replace("snapshot:a", "snapshot:b") for x in first.provenance))
    merged = evidence_index((second,), committed=(first,))[first.key]
    assert {"corpus_snapshot:a", "corpus_snapshot:b"} <= set(merged.provenance)
    with pytest.raises(EvidenceError, match="inconsistent"):
        evidence_index((replace(second, source_uri="docs://other"),), committed=(first,))


def test_http_transition_requires_existing_expected_snapshot(tmp_path):
    service, store, engine = controller(tmp_path)
    old, new = retriever("a", "r1", 640), retriever("b", "r2", 720)
    app = create_app(service, retrieval=old)
    with TestClient(app) as client:
        assert client.post("/sessions/missing/query", json={"request_id": "x", "query": "q",
                           "from_corpus_snapshot_id": "a"}).status_code == 404
        first = client.post("/sessions/s/query", json={"request_id": "one", "query": "Upload limit?"})
        assert first.status_code == 200
        app.state.retrieval = new  # A separately loaded index, as after a service restart.
        assert client.post("/sessions/s/query", json={"request_id": "two", "query": "Current limit?"}).status_code == 422
        changed = client.post("/sessions/s/query", json={"request_id": "two", "query": "Current limit?",
                              "from_corpus_snapshot_id": "a"})
        assert changed.status_code == 200 and changed.json()["branch_reason"] == "corpus_snapshot_change"
        assert client.get("/sessions/s").json()["corpus_snapshot_id"] == "b"
        replay = client.post("/sessions/s/query", json={"request_id": "one", "query": "Upload limit?"})
        assert replay.status_code == 200 and replay.json()["idempotent_replay"]
    old.close()


def test_retrieval_key_error_is_not_reported_as_a_missing_session(tmp_path, monkeypatch):
    service, store, engine = controller(tmp_path)
    old = retriever("a", "r1", 640)
    app = create_app(service, retrieval=old)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/sessions/s/query", json={"request_id": "one", "query": "Upload limit?"}).status_code == 200
        async def fail(*args, **kwargs):
            raise KeyError("missing retrieval record")
        monkeypatch.setattr(old, "retrieve", fail)
        response = client.post("/sessions/s/query", json={"request_id": "two", "query": "Current limit?",
                               "from_corpus_snapshot_id": "a"})
        assert response.status_code == 500
