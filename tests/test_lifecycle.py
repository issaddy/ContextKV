import asyncio
from dataclasses import replace

import pytest

from controller.candidates import CandidatePolicy
from controller.engine_client import EngineError, UnknownOutcomeError
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import TurnRecord
from controller.service import ContextOverflow, Controller
from controller.session_store import IdempotencyConflict, PendingAttempt, SessionStore, StoreError, VersionConflict


def system(tmp_path, **kwargs):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    engine = MockEngine()
    return Controller(store, engine, **kwargs), store, engine


@pytest.mark.asyncio
async def test_success_dual_ledger_rebuild_and_retire(tmp_path):
    controller, store, engine = system(tmp_path)
    v1 = make_evidence("The timeout is thirty seconds", "timeout")
    v2 = make_evidence("The timeout is sixty seconds", "timeout", version="v2")
    first = await controller.serve_turn("s", "What timeout?", (v1,), request_id="a")
    old = store.read_snapshot("s").current_epoch
    second = await controller.serve_turn("s", "Latest timeout?", (v2,), request_id="b")
    snap = store.read_snapshot("s")
    assert first.action == second.action == "REBUILD"
    assert snap.current_epoch.epoch_no > old.epoch_no
    assert len(snap.raw_history) == 2
    assert snap.execution.history_version == snap.raw_history_version
    assert snap.execution.last_output_ids == second.output_ids
    assert v2.key in snap.execution.visible_blocks and v1.key not in snap.execution.visible_blocks
    assert snap.execution.evidence_spans
    await controller.drain()
    assert not store.outbox()
    assert (await engine.status(engine_session_id=old.engine_session_id)).state == "RETIRED"


@pytest.mark.asyncio
async def test_idempotent_retry_and_conflict(tmp_path):
    controller, store, engine = system(tmp_path)
    evidence = (make_evidence("a fact", "a"),)
    original = await controller.serve_turn("s", "q", evidence, request_id="same")
    replay = await controller.serve_turn("s", "q", evidence, request_id="same")
    assert replay.idempotent_replay and original.output_ids == replay.output_ids
    assert len(store.read_snapshot("s").raw_history) == 1
    with pytest.raises(IdempotencyConflict):
        await controller.serve_turn("s", "another q", evidence, request_id="same")


@pytest.mark.asyncio
async def test_lost_success_response_recovers_without_regeneration(tmp_path):
    controller, store, engine = system(tmp_path)
    engine.drop_next_generate_response = True
    evidence = (make_evidence("a fact", "a"),)
    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", "q", evidence, request_id="lost")
    pending = store.pending_attempt("s")
    assert pending.stage == "SUBMITTED"
    assert store.read_snapshot("s").raw_history == ()
    replay = await controller.serve_turn("s", "q", evidence, request_id="lost")
    assert replay.idempotent_replay
    assert replay.attempt_id == pending.attempt_id
    assert len(store.read_snapshot("s").raw_history) == 1


@pytest.mark.asyncio
async def test_failed_append_keeps_current_epoch_and_forces_recovery(tmp_path):
    controller, store, engine = system(tmp_path)
    evidence = (make_evidence("a fact", "a"),)
    await controller.serve_turn("s", "q", evidence, request_id="1")
    old = store.read_snapshot("s").current_epoch
    engine.fail_next_generate = True
    with pytest.raises(EngineError):
        await controller.serve_turn("s", "follow up", evidence, request_id="2")
    await controller.recover("s", abandon=True)
    snap = store.read_snapshot("s")
    assert len(snap.raw_history) == 1
    assert snap.current_epoch == old and snap.force_new_epoch_for_recovery
    assert (await engine.status(engine_session_id=old.engine_session_id)).state != "RETIRED"
    resumed = await controller.serve_turn("s", "retry", evidence, request_id="3")
    assert resumed.action == "REBUILD"
    await controller.drain()
    assert (await engine.status(engine_session_id=old.engine_session_id)).state == "RETIRED"


@pytest.mark.asyncio
async def test_retire_failure_does_not_rollback_committed_answer(tmp_path):
    controller, store, engine = system(tmp_path, knobs=PlannerKnobs(strategy="always_rebuild"))
    evidence = (make_evidence("a fact", "a"),)
    await controller.serve_turn("s", "q", evidence, request_id="1")
    await controller.drain()
    engine.drop_next_retire_response = True
    await controller.serve_turn("s", "q2", evidence, request_id="2")
    await asyncio.gather(*tuple(controller._background))
    assert len(store.read_snapshot("s").raw_history) == 2
    assert len(store.outbox()) == 1
    await controller.drain()
    assert not store.outbox()


@pytest.mark.asyncio
async def test_restart_rebuilds_from_committed_history(tmp_path):
    controller, store, engine = system(tmp_path)
    evidence = (make_evidence("a fact", "a"),)
    await controller.serve_turn("s", "q", evidence)
    old = store.read_snapshot("s").current_epoch
    await engine.restart()
    next_turn = await controller.serve_turn("s", "follow up", evidence)
    snap = store.read_snapshot("s")
    assert next_turn.action == "REBUILD"
    assert len(snap.raw_history) == 2
    assert snap.current_epoch.engine_restart_id != old.engine_restart_id
    await controller.drain()
    assert not store.outbox()


@pytest.mark.asyncio
async def test_p1_history_sync_keeps_execution_audit_then_rebuilds(tmp_path):
    controller, store, engine = system(tmp_path)
    evidence = (make_evidence("a fact", "a"),)
    await controller.serve_turn("s", "q", evidence)
    before = store.read_snapshot("s")
    synced = store.sync_history("s", (TurnRecord(1, "q", "official prior answer"),), before.version)
    assert synced.execution == before.execution
    assert synced.raw_history_version != synced.execution.history_version
    result = await controller.serve_turn("s", "q2", evidence)
    after = store.read_snapshot("s")
    assert result.action == "REBUILD"
    assert after.current_epoch.reason == "reference_history_sync"
    assert after.raw_history[0].assistant_text == "official prior answer"
    assert after.execution.history_version == after.raw_history_version
    await controller.drain()


@pytest.mark.asyncio
async def test_overflow_is_failure_not_silent_truncation(tmp_path):
    controller, store, engine = system(tmp_path, policy=CandidatePolicy(max_total_model_tokens=100, max_new_tokens=10))
    with pytest.raises(ContextOverflow):
        await controller.serve_turn("s", "q", (make_evidence("long evidence " * 50, "a"),), request_id="1")
    assert store.pending_attempt("s") is None
    assert store.find_request("s", "1").stage == "FAILED"
    assert store.read_snapshot("s").execution is None


def test_cross_connection_pending_and_history_cas(tmp_path):
    first = SessionStore(tmp_path / "sessions.sqlite3")
    second = SessionStore(tmp_path / "sessions.sqlite3")
    snap = first.read_snapshot("s")
    first.begin_attempt("s", "one", "hash", "q", snap.version)
    with pytest.raises(PendingAttempt):
        second.begin_attempt("s", "two", "hash2", "q2", snap.version)
    with pytest.raises(PendingAttempt):
        second.begin_attempt("s", "one", "hash", "q", snap.version)
    with pytest.raises(PendingAttempt):
        second.sync_history("s", (), snap.version)
    first.close()
    second.close()


@pytest.mark.asyncio
async def test_commit_failure_keeps_durable_result_and_old_epoch(tmp_path, monkeypatch):
    controller, store, engine = system(tmp_path, knobs=PlannerKnobs(strategy="always_rebuild"))
    evidence = (make_evidence("a fact", "a"),)
    await controller.serve_turn("s", "q", evidence, request_id="1")
    old = store.read_snapshot("s").current_epoch
    real_commit = store.commit_turn_and_outbox
    def failed_commit(attempt_id):
        raise VersionConflict("injected commit failure")
    monkeypatch.setattr(store, "commit_turn_and_outbox", failed_commit)
    with pytest.raises(VersionConflict):
        await controller.serve_turn("s", "q2", evidence, request_id="2")
    assert store.pending_attempt("s").stage == "RESULT_READY"
    assert store.read_snapshot("s").current_epoch == old
    assert (await engine.status(engine_session_id=old.engine_session_id)).state != "RETIRED"
    monkeypatch.setattr(store, "commit_turn_and_outbox", real_commit)
    result = await controller.recover("s")
    assert result.idempotent_replay
    await controller.drain()
    assert (await engine.status(engine_session_id=old.engine_session_id)).state == "RETIRED"


@pytest.mark.asyncio
async def test_same_session_serial_across_concurrent_calls(tmp_path):
    controller, store, engine = system(tmp_path)
    engine.generation_delay = .01
    evidence = (make_evidence("a fact", "a"),)
    responses = await asyncio.gather(*(controller.serve_turn("s", f"q{i}", evidence, request_id=str(i)) for i in range(4)))
    assert sorted(r.ledger_version for r in responses) == [1, 2, 3, 4]
    assert len(store.read_snapshot("s").raw_history) == 4


@pytest.mark.asyncio
async def test_bad_receipt_is_never_committed(tmp_path, monkeypatch):
    controller, store, engine = system(tmp_path)
    original = engine.generate
    async def corrupted(**kwargs):
        result = await original(**kwargs)
        return replace(result, receipt=replace(result.receipt, input_hash="wrong"))
    monkeypatch.setattr(engine, "generate", corrupted)
    with pytest.raises(EngineError, match="RECEIPT_IDENTITY_MISMATCH"):
        await controller.serve_turn("s", "q", ())
    assert not store.read_snapshot("s").raw_history
    assert store.pending_attempt("s").stage == "RESULT_READY"


@pytest.mark.asyncio
async def test_closed_session_rejects_new_turn(tmp_path):
    controller, store, engine = system(tmp_path)
    await controller.serve_turn("s", "q", ())
    await controller.close_session("s")
    with pytest.raises(StoreError, match="closed"):
        await controller.serve_turn("s", "q2", ())


@pytest.mark.asyncio
async def test_lost_prepare_ack_can_be_abandoned_safely(tmp_path, monkeypatch):
    controller, store, engine = system(tmp_path)
    original = engine.prepare
    async def lost(*args):
        await original(*args)
        raise UnknownOutcomeError("TRANSPORT_ERROR")
    monkeypatch.setattr(engine, "prepare", lost)
    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", "q", ())
    assert store.pending_attempt("s").stage == "PREPARING"
    with pytest.raises(PendingAttempt):
        await controller.recover("s")
    monkeypatch.setattr(engine, "prepare", original)
    await controller.recover("s", abandon=True)
    assert store.pending_attempt("s") is None and not store.outbox()
    assert not store.read_snapshot("s").raw_history


def test_reopen_store_preserves_immutable_state(tmp_path):
    async def run():
        controller, store, _ = system(tmp_path)
        await controller.serve_turn("s", "q", (make_evidence("fact", "a"),))
        await controller.aclose()
        snapshot = store.read_snapshot("s")
        store.close()
        reopened = SessionStore(tmp_path / "sessions.sqlite3")
        assert reopened.read_snapshot("s") == snapshot
        assert isinstance(reopened.read_snapshot("s").execution.visible_blocks, frozenset)
        reopened.close()
    asyncio.run(run())
