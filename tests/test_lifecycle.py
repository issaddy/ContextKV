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
@pytest.mark.parametrize("lost_response", [False,True])
async def test_cost_tie_compacts_evidence_and_retires_only_after_commit(tmp_path,monkeypatch,lost_response):
    from controller.cost_model import CostModel
    from controller.schema import CostEstimate

    model = CostModel()
    monkeypatch.setattr(model,"predict_native",lambda candidate,*args,**kwargs:
        CostEstimate(candidate.candidate_id,100,error_ms=10,calibrated=True))
    controller,store,engine = system(tmp_path,cost_model=model)
    old_evidence = make_evidence("Old material. "*200,"old")
    new_evidence = make_evidence("Current material.","new")
    shared = make_evidence("Shared relevant condition.", "shared")
    first = await controller.serve_turn("s","Summarize the old material.",(old_evidence,shared),request_id="1")
    old = store.read_snapshot("s").current_epoch
    retired = []
    original_retire = engine.retire

    async def retire_after_commit(session_id,*args,**kwargs):
        if session_id == old.engine_session_id:
            snapshot = store.read_snapshot("s")
            assert store.find_request("s","2").stage == "COMMITTED"
            assert snapshot.current_epoch != old
            assert len(snapshot.raw_history) == 2
            retired.append(session_id)
        return await original_retire(session_id,*args,**kwargs)

    monkeypatch.setattr(engine,"retire",retire_after_commit)
    if lost_response:
        engine.drop_next_generate_response = True
        with pytest.raises(UnknownOutcomeError):
            await controller.serve_turn("s","Summarize the new material.",(new_evidence,shared),request_id="2")
        assert store.read_snapshot("s").current_epoch == old
        assert not retired and (await engine.status(engine_session_id=old.engine_session_id)).state != "RETIRED"
    second = await controller.serve_turn("s","Summarize the new material.",(new_evidence,shared),request_id="2")
    snapshot = store.read_snapshot("s")
    assert second.action == "REBUILD" and second.reason == "context_size_tie_break"
    assert second.idempotent_replay == lost_response
    assert snapshot.raw_history[0].assistant_text == first.assistant_text
    assert snapshot.execution.visible_blocks == frozenset((new_evidence.key,shared.key))
    await controller.drain()
    assert retired == [old.engine_session_id] and not store.outbox()


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
async def test_coalesced_writes_preserve_durable_barriers_before_engine_calls(tmp_path,monkeypatch):
    controller,store,engine = system(tmp_path,knobs=PlannerKnobs(strategy="always_append"))
    observed = []
    prepare,generate = engine.prepare,engine.generate
    async def checked_prepare(*args,**kwargs):
        attempt = store.pending_attempt("s")
        assert attempt.stage == "PREPARING" and attempt.plan is not None
        assert attempt.epoch.engine_session_id == args[0]
        observed.append("prepare")
        return await prepare(*args,**kwargs)
    async def checked_generate(**kwargs):
        attempt = store.get_attempt(kwargs["attempt_id"])
        assert attempt.stage == "SUBMITTED"
        assert attempt.epoch.session_generation == kwargs["expected_generation"]
        assert attempt.plan.candidate.input_ids == kwargs["input_ids"]
        observed.append("generate")
        return await generate(**kwargs)
    monkeypatch.setattr(engine,"prepare",checked_prepare)
    monkeypatch.setattr(engine,"generate",checked_generate)
    evidence = (make_evidence("Stable evidence","source"),)
    await controller.serve_turn("s","first",evidence,request_id="1")
    second = await controller.serve_turn("s","follow up",evidence,request_id="2")
    replay = await controller.serve_turn("s","follow up",evidence,request_id="2")
    assert second.action == "APPEND" and replay.idempotent_replay
    assert observed == ["prepare","generate","generate"]
    await controller.drain()


def test_plan_and_epoch_allocation_roll_back_together(tmp_path,monkeypatch):
    from controller.candidates import build_candidates
    from controller.schema import Plan
    store = SessionStore(tmp_path/"state.sqlite3")
    snapshot = store.read_snapshot("s")
    attempt = store.begin_attempt("s","one","hash","question",snapshot.version)
    plan = Plan(build_candidates(snapshot,"question",())[0])
    allocate = store._allocate_epoch
    def fail_after_write(*args):
        allocate(*args)
        raise RuntimeError("injected transaction failure")
    monkeypatch.setattr(store,"_allocate_epoch",fail_after_write)
    with pytest.raises(RuntimeError,match="transaction failure"):
        store.plan_new_epoch(replace(attempt,stage="PLANNED",plan=plan),"restart","initial")
    assert store.get_attempt(attempt.attempt_id) == attempt
    assert store.db.execute("SELECT epoch_counter FROM sessions WHERE session_id='s'").fetchone()[0] == 0
    store.close()


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
async def test_slow_retirement_coalesces_turns_and_drains_new_outbox_entries(tmp_path, monkeypatch):
    controller, store, engine = system(tmp_path, knobs=PlannerKnobs(strategy="always_rebuild"))
    evidence = (make_evidence("The timeout is thirty seconds", "timeout"),)
    await controller.serve_turn("initial", "q", evidence, request_id="1")
    await controller.drain()
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    original = engine.retire

    async def slow_retire(*args):
        calls.append(args[2])
        if len(calls) == 1:
            started.set()
            await release.wait()
        return await original(*args)

    monkeypatch.setattr(engine, "retire", slow_retire)
    try:
        await controller.serve_turn("initial", "q2", evidence, request_id="2")
        await asyncio.wait_for(started.wait(), timeout=1)

        async def other_conversation(index):
            session = f"peer-{index}"
            await controller.serve_turn(session, "q", evidence, request_id="1")
            await controller.serve_turn(session, "q2", evidence, request_id="2")
            assert len(store.read_snapshot(session).raw_history) == 2

        # Generations and commits continue while the first retirement is held.
        await asyncio.gather(*(other_conversation(i) for i in range(6)))
        assert len(store.outbox()) == 7
        assert len([task for task in controller._background if not task.done()]) == 1
        release.set()
        await asyncio.gather(*tuple(controller._background))
        # Entries committed after the first flush snapshot must be picked up
        # without a new request, periodic HTTP worker or explicit final drain.
        assert not store.outbox()
        assert len(calls) == len(set(calls)) == 7
        assert all(store.get_snapshot(f"peer-{i}").current_epoch is not None for i in range(6))
    finally:
        release.set()
        await controller.aclose()
        store.close()


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
    async def lost(*args, **kwargs):
        await original(*args, **kwargs)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong", [
    {"engine_session_id": "wrong"}, {"engine_restart_id": "wrong"},
    {"operation_id": "wrong"}, {"session_generation": 0},
    {"business_session_id": "wrong"},
])
async def test_recovery_rejects_mismatched_prepare_ack_before_saving_generation(tmp_path, monkeypatch, wrong):
    controller, store, engine = system(tmp_path)
    original = engine.prepare

    async def lost(*args, **kwargs):
        await original(*args, **kwargs)
        raise UnknownOutcomeError("TRANSPORT_ERROR")

    monkeypatch.setattr(engine, "prepare", lost)
    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", "q", ())
    pending = store.pending_attempt("s")

    async def corrupted(*args, **kwargs):
        return replace(await original(*args, **kwargs), **wrong)

    monkeypatch.setattr(engine, "prepare", corrupted)
    with pytest.raises(EngineError, match="PREPARE_ACK_MISMATCH"):
        await controller.recover("s", abandon=True)
    assert store.pending_attempt("s") == pending
    assert not store.read_snapshot("s").raw_history
    assert not store.outbox()
    monkeypatch.setattr(engine, "prepare", original)
    await controller.recover("s", abandon=True)
    assert store.pending_attempt("s") is None


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
