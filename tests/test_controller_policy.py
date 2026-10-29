import asyncio
from dataclasses import replace

import pytest

from controller.config import AblationConfig, CostSettings, RunConfig
from controller.evidence import make_evidence
from bench.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.service import Controller
from controller.session_store import SessionStore


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy,quotes", [("always_append", 0), ("always_rebuild", 0),
    ("fixed_threshold", 0), ("fixed_period", 0), ("contextpilot", 0), ("redundancy_threshold", 0),
    ("cost", 1), ("warm_idle", 1)])
async def test_quotes_only_when_policy_uses_them(tmp_path, monkeypatch, strategy, quotes):
    engine = MockEngine()
    controller = Controller(SessionStore(tmp_path/"sessions.sqlite3"), engine,
                            knobs=PlannerKnobs(strategy=strategy))
    observed = []
    original = engine.quote_batch
    async def quote_batch(*args):
        observed.append(args)
        return await original(*args)
    monkeypatch.setattr(engine, "quote_batch", quote_batch)
    await controller.serve_turn("s", "q1", ())
    await controller.serve_turn("s", "q2", ())
    assert len(observed) == quotes
    await controller.aclose()


@pytest.mark.asyncio
async def test_no_c_keeps_epoch_for_text_rebuild_but_p1_still_reopens(tmp_path):
    store, engine = SessionStore(tmp_path/"sessions.sqlite3"), MockEngine()
    controller = Controller(store, engine, epoch_retirement=False,
                            knobs=PlannerKnobs(strategy="always_rebuild"))
    evidence = (make_evidence("The timeout is thirty seconds", "timeout"),)
    first = await controller.serve_turn("s", "q1", evidence)
    second = await controller.serve_turn("s", "q2", evidence)
    assert first.epoch_no == second.epoch_no
    assert first.branch_reason == "initial"
    assert second.branch_reason == "always_rebuild"
    before = store.read_snapshot("s")
    official = tuple(replace(turn, assistant_text="official", assistant_output_ids=(), evidence_keys=())
                     for turn in before.raw_history)
    third = await controller.serve_turn("s", "q3", evidence, reference_history=official)
    assert third.epoch_no > second.epoch_no
    assert third.branch_reason == "reference_history_sync"
    after = store.read_snapshot("s")
    assert after.raw_history[0].evidence_keys == before.raw_history[0].evidence_keys
    assert after.raw_history[0].assistant_text == "official"
    await controller.aclose()


@pytest.mark.asyncio
async def test_client_timeline_never_uses_server_clock(tmp_path, monkeypatch):
    engine = MockEngine()
    controller = Controller(SessionStore(tmp_path/"sessions.sqlite3"), engine)
    original = engine.generate
    async def generate(**kwargs):
        await asyncio.sleep(.005)
        result = await original(**kwargs)
        return replace(result, ttft_ms=1, last_token_ms=3, server_ttft_ms=10000,
                       server_elapsed_ms=20000, timing_source="test_client_chunks")
    monkeypatch.setattr(engine, "generate", generate)
    response = await controller.serve_turn("s", "q", ())
    timing = response.timing
    assert timing.t_arrive <= timing.t_submit < timing.t_first < timing.t_last_token < timing.t_response_done <= timing.t_commit
    assert response.ttft_ms < response.complete_ms
    assert response.server_ttft_ms == 10000
    assert timing.token_observation == "chunk"
    await controller.aclose()


def test_online_total_residual_does_not_restore_ablated_costs():
    with pytest.raises(ValueError, match="frozen calibration"):
        RunConfig(cost_model=CostSettings(online_calibration=True), ablation=AblationConfig(host_cost=False))


@pytest.mark.asyncio
async def test_attempt_retains_original_retrieval_priority_for_offline_replay(tmp_path):
    store = SessionStore(tmp_path/"sessions.sqlite3")
    controller = Controller(store, MockEngine())
    evidence = (make_evidence("First fact", "a"), make_evidence("Condition", "b"))
    priority = (evidence[0].key,)
    response = await controller.serve_turn("s", "q", evidence, retrieval_priority=priority)
    assert store.get_attempt(response.attempt_id).retrieval_priority == priority
    await controller.aclose()
