"""Retrieval owns the same durable turn and session lock as supplied evidence."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from controller.engine_client import UnknownOutcomeError
from controller.evidence import make_evidence
from bench.mock_engine import MockEngine
from controller import service as service_module
from controller.service import Controller, EvidenceSelection
from controller.session_store import IdempotencyConflict, PendingAttempt, SessionStore, StoreError


def selection():
    block = make_evidence("A complete source body.", "source")
    return EvidenceSelection((block,), (block.key,))


@pytest.mark.asyncio
async def test_cross_connection_start_owner_precedes_retrieval(tmp_path):
    first_store = SessionStore(tmp_path / "shared.sqlite")
    second_store = SessionStore(tmp_path / "shared.sqlite")
    engine = MockEngine()
    first, second = Controller(first_store, engine), Controller(second_store, engine)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def provider(snapshot):
        assert first_store.pending_attempt("s").stage == "STARTED"
        calls.append(snapshot.version)
        entered.set()
        await release.wait()
        return selection()

    task = asyncio.create_task(first.serve_turn("s", "q", request_id="one", evidence_provider=provider))
    await entered.wait()
    for request_id in ("one", "two"):
        with pytest.raises(PendingAttempt):
            await second.serve_turn("s", "q", request_id=request_id, evidence_provider=provider)
    assert calls == [0]
    release.set()
    await task
    assert len(first_store.get_snapshot("s").raw_history) == 1
    await first.aclose()
    await second.aclose()
    first_store.close()
    second_store.close()


@pytest.mark.asyncio
async def test_lost_result_recovers_after_reopen_then_replays_original_audit(tmp_path):
    store = SessionStore(tmp_path / "state.sqlite")
    engine = MockEngine()
    engine.drop_next_generate_response = True
    controller = Controller(store, engine)
    calls = []

    async def provider(snapshot):
        calls.append(snapshot.version)
        assert len(calls) == 1
        return selection()

    with pytest.raises(UnknownOutcomeError):
        await controller.serve_turn("s", "q", request_id="one", evidence_provider=provider)
    pending = store.pending_attempt("s")
    assert pending.stage == "SUBMITTED" and pending.retrieval_ms is not None
    await controller.aclose()
    store.close()
    # The same engine still owns the outcome; the controller/store can restart.
    store = SessionStore(tmp_path / "state.sqlite")
    controller = Controller(store, engine)
    replay = await controller.serve_turn("s", "q", request_id="one", evidence_provider=provider)
    assert replay.idempotent_replay and replay.attempt_id == pending.attempt_id
    assert replay.retrieval_ms == pending.retrieval_ms
    assert replay.visible_evidence_manifest == pending.plan.candidate.evidence_spans
    assert store.find_request("s", "one").evidence == pending.evidence
    assert calls == [0] and len(engine._attempts) == 1
    await controller.serve_turn("s", "later question", (make_evidence("Another body.", "later"),),
                                request_id="later")
    old = await controller.serve_turn("s", "q", request_id="one", evidence_provider=provider)
    assert old.idempotent_replay and old.ledger_version == 1
    assert old.output_ids == replay.output_ids
    assert old.visible_evidence_manifest == pending.plan.candidate.evidence_spans
    assert old.retrieval_ms == pending.retrieval_ms
    assert calls == [0] and len(engine._attempts) == 2
    assert len(store.get_snapshot("s").raw_history) == 2
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_first", [False, True])
async def test_route_identity_cannot_replay_same_material(tmp_path, provider_first):
    store = SessionStore(tmp_path / "state.sqlite")
    controller = Controller(store, MockEngine())
    material = selection()
    calls = []

    async def provider(snapshot):
        calls.append(snapshot.version)
        return material

    direct = {"evidence": material.evidence, "retrieval_priority": material.retrieval_priority}
    retrieved = {"evidence_provider": provider}
    first, second = (retrieved, direct) if provider_first else (direct, retrieved)
    await controller.serve_turn("s", "q", request_id="one", **first)
    with pytest.raises(IdempotencyConflict):
        await controller.serve_turn("s", "changed q", request_id="one", **first)
    with pytest.raises(IdempotencyConflict):
        await controller.serve_turn("s", "q", request_id="one", **second)
    assert len(calls) == int(provider_first)
    assert len(store.get_snapshot("s").raw_history) == 1
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
async def test_query_waits_for_manual_turn_commit_before_reading_history(tmp_path):
    store = SessionStore(tmp_path / "state.sqlite")
    engine = MockEngine(response_text="actual first answer")
    controller = Controller(store, engine)
    entered, release = asyncio.Event(), asyncio.Event()
    original_generate = engine.generate

    async def delayed(**kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await original_generate(**kwargs)

    engine.generate = delayed
    views = []

    async def provider(snapshot):
        views.append(snapshot)
        return selection()

    first = asyncio.create_task(controller.serve_turn("s", "manual q", request_id="one"))
    await entered.wait()
    second = asyncio.create_task(controller.serve_turn("s", "automatic q", request_id="two",
                                                      evidence_provider=provider))
    await asyncio.sleep(0)
    assert not views
    release.set()
    await asyncio.gather(first, second)
    assert len(views) == 1 and views[0].version == 1
    assert [(t.user_text, t.assistant_text) for t in views[0].raw_history] == [
        ("manual q", "actual first answer")]
    assert store.get_snapshot("s").version == 2
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_provider_failure_needs_new_id_and_has_no_epoch_or_qa(tmp_path, cancel):
    store = SessionStore(tmp_path / "state.sqlite")
    engine = MockEngine()
    controller = Controller(store, engine)
    entered = asyncio.Event()
    calls = []

    async def provider(snapshot):
        calls.append(snapshot.version)
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError("retrieval unavailable")

    task = asyncio.create_task(controller.serve_turn("s", "q", request_id="one", evidence_provider=provider))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert store.find_request("s", "one").stage == "FAILED"
    assert store.get_snapshot("s").raw_history == () and not engine._attempts
    assert store.get_snapshot("s").current_epoch is None and not engine._epochs
    with pytest.raises(StoreError, match="Previous attempt failed"):
        await controller.serve_turn("s", "q", request_id="one", evidence_provider=provider)
    assert calls == [0]

    async def available(snapshot):
        calls.append(snapshot.version)
        return selection()

    result = await controller.serve_turn("s", "q", request_id="new", evidence_provider=available)
    assert result.ledger_version == 1 and calls == [0, 0]
    assert len(engine._attempts) == 1
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
async def test_close_and_recover_wait_for_provider_and_closed_session_skips_it(tmp_path):
    store = SessionStore(tmp_path / "state.sqlite")
    controller = Controller(store, MockEngine())
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def provider(snapshot):
        calls.append(snapshot.version)
        entered.set()
        await release.wait()
        return selection()

    turn = asyncio.create_task(controller.serve_turn("s", "q", request_id="one", evidence_provider=provider))
    await entered.wait()
    recover = asyncio.create_task(controller.recover("s"))
    close = asyncio.create_task(controller.close_session("s"))
    await asyncio.sleep(0)
    assert not recover.done() and not close.done()
    assert store.pending_attempt("s").stage == "STARTED"
    release.set()
    response, recovered, _ = await asyncio.gather(turn, recover, close)
    assert response.ledger_version == 1 and recovered is None
    with pytest.raises(StoreError, match="closed"):
        await controller.serve_turn("s", "later", request_id="two", evidence_provider=provider)
    assert calls == [0] and len(store.get_snapshot("s").raw_history) == 1
    await controller.aclose()
    store.close()


@pytest.mark.asyncio
async def test_synthetic_clock_includes_retrieval_in_completion_not_control(tmp_path, monkeypatch):
    # Arithmetic contract only: these are deliberately injected seconds, not a
    # latency benchmark, and no process-wide time or asyncio clock is patched.
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr(service_module, "time", SimpleNamespace(monotonic=lambda: clock.value))
    store = SessionStore(tmp_path / "state.sqlite")
    engine = MockEngine()
    controller = Controller(store, engine)
    observed_control = []
    choose = service_module.choose_plan

    def measured_choose(*args, **kwargs):
        observed_control.append(kwargs["control_ms"])
        clock.value += .1
        return choose(*args, **kwargs)

    monkeypatch.setattr(service_module, "choose_plan", measured_choose)
    generate = engine.generate

    async def measured_generate(**kwargs):
        result = await generate(**kwargs)
        clock.value += .4
        return replace(result, ttft_ms=100, last_token_ms=350, elapsed_ms=400)

    monkeypatch.setattr(engine, "generate", measured_generate)
    commit = store.commit_turn_and_outbox

    def measured_commit(attempt_id):
        result = commit(attempt_id)
        clock.value += .05
        return result

    monkeypatch.setattr(store, "commit_turn_and_outbox", measured_commit)

    async def provider(snapshot):
        clock.value += 2
        return selection()

    response = await controller.serve_turn("s", "q", request_id="one", evidence_provider=provider)
    assert response.retrieval_ms == pytest.approx(2000)
    assert response.complete_ms == pytest.approx(2550)
    assert response.ttft_ms == pytest.approx(2200)
    assert response.control_ms == pytest.approx(150)
    assert len(observed_control) == 1
    assert observed_control[0] == pytest.approx({"APPEND": 0, "REBUILD": 0})
    assert response.timing.t_submit - response.timing.t_arrive == pytest.approx(2.1)
    assert response.timing.t_commit - response.timing.t_arrive == pytest.approx(2.55)
    await controller.aclose()
    store.close()
