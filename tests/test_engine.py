"""Engine protocol invariants; GPU timings and model quality are not simulated."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from controller.engine_client import (EngineError, SGLangEngineClient, UnknownOutcomeError,
                                      SGLANG_COMMIT, token_hash)
from controller.mock_engine import MockEngine
from controller.schema import ConfigFingerprint, PromptCandidate
from engine_patches.contextkv_runtime import ContextKVError, ContextKVRuntime


def candidate(ids, name="a"):
    return PromptCandidate(name, "APPEND", 0, (), tuple(ids), token_hash(ids), frozenset(), frozenset())


def run(coro):
    return asyncio.run(coro)


def test_quotes_are_read_only_order_independent_page_aligned_and_pending_aware():
    async def scenario():
        engine = MockEngine(page_size=2)
        await engine.seed_cache([1, 2], tier="gpu")
        await engine.seed_cache([1, 2, 3, 4], tier="host")
        await engine.seed_cache([1, 2, 3, 4, 5, 6], tier="gpu", pending=True)
        before = engine.state_digest()
        a, b = candidate([1, 2, 3, 4, 5], "a"), candidate([1, 2, 9, 10, 11], "b")
        ab = await engine.quote_batch([a, b], engine.fingerprint)
        ba = await engine.quote_batch([b, a], engine.fingerprint)
        assert engine.state_digest() == before
        assert ab.quotes == tuple(reversed(ba.quotes))
        q = ab.quotes[0]
        assert q.match_limit == 4
        assert q.ready_gpu_prefix_tokens == 0  # seed explicitly marked all H2D pending
        assert q.inflight_h2d_tokens == 4
        assert q.native_estimate_uncertain
        await engine.complete_transfers()
        q = (await engine.quote_batch([a], engine.fingerprint)).quotes[0]
        assert q.ready_gpu_prefix_tokens == 4
    run(scenario())


def test_prepare_generation_guard_and_restart():
    async def scenario():
        engine = MockEngine()
        info = await engine.info()
        p = await engine.prepare("s", "p", info.engine_restart_id)
        assert p == await engine.prepare("s", "p", info.engine_restart_id)
        with pytest.raises(EngineError, match="EPOCH_CONFLICT"):
            await engine.prepare("s", "different", info.engine_restart_id)
        with pytest.raises(EngineError, match="STALE_GENERATION"):
            await engine.generate(input_ids=(1, 2), session_id="s", expected_generation=p.session_generation+1,
                expected_restart_id=p.engine_restart_id, attempt_id="a", fingerprint=engine.fingerprint)
        await engine.restart()
        with pytest.raises(EngineError, match="STALE_RESTART"):
            await engine.prepare("s2", "p2", info.engine_restart_id)
    run(scenario())


def test_shared_unique_slots_survive_retire_and_late_requests_rejected():
    async def scenario():
        engine = MockEngine(response_text="answer")
        restart = (await engine.info()).engine_restart_id
        a, b = await engine.prepare("a", "pa", restart), await engine.prepare("b", "pb", restart)
        for p in [a, b]:
            await engine.generate(input_ids=(1, 2, 3), session_id=p.engine_session_id,
                expected_generation=p.session_generation, expected_restart_id=restart,
                attempt_id="g"+p.engine_session_id, fingerprint=engine.fingerprint)
        count = engine.unique_gpu_tokens
        assert count == 3 + len(b"answer") - 1
        ack = await engine.retire("a", a.session_generation, "ra", restart)
        assert ack.acknowledged and ack.freed_gpu_tokens == 0
        assert engine.unique_gpu_tokens == count
        assert any("b" in s.refs for s in engine._slots.values())
        assert all("a" not in s.refs for s in engine._slots.values())
        assert ack == await engine.retire("a", a.session_generation, "ra", restart)
        with pytest.raises(EngineError, match="EPOCH_RETIRED"):
            await engine.generate(input_ids=(1, 2, 3), session_id="a", expected_generation=a.session_generation,
                expected_restart_id=restart, attempt_id="late", fingerprint=engine.fingerprint)
        with pytest.raises(EngineError, match="EPOCH_CONFLICT"):
            await engine.prepare("a", "pa", restart)
    run(scenario())


def test_retirement_waits_for_inflight_and_lost_success_is_recoverable():
    async def scenario():
        engine = MockEngine(generation_delay=0.02)
        restart = (await engine.info()).engine_restart_id
        p = await engine.prepare("s", "p", restart)
        engine.drop_next_generate_response = True
        task = asyncio.create_task(engine.generate(input_ids=(1, 2, 3), session_id="s",
            expected_generation=p.session_generation, expected_restart_id=restart,
            attempt_id="a", fingerprint=engine.fingerprint))
        await asyncio.sleep(0)
        assert (await engine.status(attempt_id="a")).inflight
        ack = await engine.retire("s", p.session_generation, "r", restart)
        assert not ack.acknowledged and ack.state == "RETIRING"
        with pytest.raises(UnknownOutcomeError):
            await task
        status = await engine.status(attempt_id="a")
        assert status.terminal and status.result is not None
        assert status.receipt.generation_complete
        assert not status.receipt.session_ref_registered
        assert (await engine.status(engine_session_id="s")).state == "RETIRED"
    run(scenario())


def test_http_client_never_sends_guards_to_stock_server():
    async def scenario():
        calls = []
        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(404, json={"detail": "Not Found"})
        client = SGLangEngineClient("http://test", ConfigFingerprint(), transport=httpx.MockTransport(handler))
        with pytest.raises(EngineError):
            await client.generate(input_ids=(1,), session_id="s", expected_generation=1,
                expected_restart_id="r", attempt_id="a", fingerprint=ConfigFingerprint())
        assert calls == ["/context/engine/info"]
        await client.aclose()
    run(scenario())


def test_http_rejects_unverified_engine_capabilities_and_fingerprint():
    async def scenario():
        fp = ConfigFingerprint()
        info = dict(engine_restart_id="r", protocol_version="contextkv-v1", engine_kind="sglang",
                    guarded_generate_supported=True, scheduler_owned=False,
                    completion_status_supported=True, sglang_commit=SGLANG_COMMIT, fingerprint=asdict(fp))
        client = SGLangEngineClient("http://test", fp, transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=info)))
        with pytest.raises(EngineError, match="UNSUPPORTED_ENGINE"):
            await client.info()
        info["scheduler_owned"] = True
        info["fingerprint"]["template_hash"] = "wrong"
        with pytest.raises(EngineError, match="FINGERPRINT_MISMATCH"):
            await client.info()
        await client.aclose()
    run(scenario())


class FakeNativeCache:
    """Only the existing native tracker boundary is mocked in these tests."""
    page_size = 1
    _tree_core_backend = "python"
    cache_controller = None

    def __init__(self):
        self.session_refs = SimpleNamespace(_session_generations={})
        self.counter = 0
        self.released = []

    def open_radix_session(self, sid):
        self.counter += 1
        self.session_refs._session_generations[sid] = self.counter
        return self.counter

    def release_radix_session(self, sid):
        self.session_refs._session_generations.pop(sid, None)
        self.released.append(sid)


@pytest.mark.parametrize("stream", [False, True])
def test_real_patch_runtime_reuses_tracker_generation_and_blocks_auto_reopen(stream):
    cache = FakeNativeCache()
    runtime = ContextKVRuntime(cache, asdict(ConfigFingerprint()))
    args = dict(engine_session_id="s", operation_id="p", expected_restart_id=runtime.restart_id)
    a = runtime.prepare(args)
    assert runtime.prepare(args) == a
    assert cache.counter == 1
    recv = SimpleNamespace(session_id="s", expected_restart_id=runtime.restart_id,
        expected_generation=a["session_generation"], contextkv_fingerprint=runtime.fingerprint,
        contextkv_attempt_id="g", rid="g", stream=stream, session_params=None, lora_id=None,
        mm_inputs=None, input_embeds=None, return_logprob=False, cache_salt=None, extra_key=None,
        input_ids=[1, 2, 3])
    req = SimpleNamespace()
    runtime.admit(recv, req)
    ack = runtime.retire(dict(engine_session_id="s", expected_generation=a["session_generation"],
                             operation_id="r", expected_restart_id=runtime.restart_id))
    assert ack["state"] == "RETIRING" and not ack["acknowledged"]
    with pytest.raises(ContextKVError, match="EPOCH_RETIRED"):
        runtime.admit(recv, SimpleNamespace())
    req.finished = lambda: True
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    req.rid = "g"
    req.session_id = "s"
    req.session_generation = a["session_generation"]
    req.cached_tokens_device = req.cached_tokens_host = req.cached_tokens = 0
    req.origin_input_ids = [1, 2, 3]
    req.kv = SimpleNamespace(cache_inserted_len=3)
    req.customized_info = None
    req.output_ids = [4, 5, 6]
    req.finished_len = 2  # Native stop trimming excludes later accepted IDs.
    req.time_stats = SimpleNamespace(scheduler_recv_time=.9, forward_entry_time=.95,
                                    prefill_finished_time=1, completion_time=1.5)
    runtime.finish(req)
    assert req.customized_info["contextkv_timing"][0]["tpot_ms"] == 500
    assert cache.released == ["s"]
    assert runtime.status({"attempt_id": "g"})["terminal"]
    assert runtime.status({"attempt_id": "g"})["result"] is None
    assert runtime.status({"attempt_id": "g"})["receipt"]["inserted_cache_tokens"] == 3
    assert runtime.retire(dict(engine_session_id="s", expected_generation=a["session_generation"],
        operation_id="r", expected_restart_id=runtime.restart_id))["acknowledged"]
    assert cache.released == ["s"]
    with pytest.raises(ContextKVError, match="EPOCH_CONFLICT"):
        runtime.prepare(args)


def test_real_patch_runtime_checks_live_native_generation_not_only_registry():
    cache = FakeNativeCache()
    runtime = ContextKVRuntime(cache, {})
    p = runtime.prepare(dict(engine_session_id="s", operation_id="p", expected_restart_id=runtime.restart_id))
    cache.session_refs._session_generations.clear()
    recv = SimpleNamespace(session_id="s", expected_restart_id=runtime.restart_id,
                            contextkv_fingerprint={}, expected_generation=p["session_generation"])
    with pytest.raises(ContextKVError, match="STALE_GENERATION"):
        runtime.admit(recv, SimpleNamespace())


def test_real_patch_extend_counter_counts_chunk_and_retry_work_not_prompt_minus_hits():
    from engine_patches.contextkv_runtime import complete_extend_work, snapshot_extend_work
    req = SimpleNamespace(rid="r", contextkv_runtime=object(), contextkv_prefill_tokens=0)
    decode = SimpleNamespace(rid="decode", contextkv_runtime=object(), contextkv_prefill_tokens=0)
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True),
                            decoding_reqs=[decode], reqs=[req, decode], extend_lens=[5, 1])
    work = snapshot_extend_work(batch)
    batch.extend_lens = [999, 1]  # the captured batch lengths must remain stable
    complete_extend_work(work)
    batch.extend_lens = [3, 1]
    complete_extend_work(snapshot_extend_work(batch))
    batch.extend_lens = [5, 1]  # a retracted chunk really executed again
    complete_extend_work(snapshot_extend_work(batch))
    assert req.contextkv_prefill_tokens == 13
    assert decode.contextkv_prefill_tokens == 0


def test_real_patch_quote_uses_native_array_key_without_mutations(monkeypatch):
    import sys
    from array import array
    from types import ModuleType
    radix_module = ModuleType("sglang.srt.mem_cache.radix_cache")
    class RadixKey:
        def __init__(self, ids):
            assert type(ids) is array and ids.typecode == "q"
            self.ids = ids
    radix_module.RadixKey = RadixKey
    req_module = ModuleType("sglang.srt.managers.schedule_batch")
    req_module.Req = SimpleNamespace(_compute_max_prefix_len=lambda req, n: max(n-1, 0))
    monkeypatch.setitem(sys.modules, radix_module.__name__, radix_module)
    monkeypatch.setitem(sys.modules, req_module.__name__, req_module)
    cache = FakeNativeCache()
    cache.ongoing_load_back = {}
    cache.tree_core = SimpleNamespace(peek_full_prefix=lambda key: dict(
        raw_match_tokens=len(key.ids), ready_gpu_prefix_tokens=len(key.ids),
        recoverable_prefix_tokens=len(key.ids), host_only_load_tokens=0,
        inflight_h2d_tokens=0, inflight_d2h_tokens=0))
    cache.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 100)
    runtime = ContextKVRuntime(cache, {})
    request = {"request_id": "q", "fingerprint": {}, "candidates": [
        {"candidate_id": "a", "snapshot_version": 0, "input_ids": [1, 2, 3], "input_hash": token_hash([1, 2, 3])}]}
    quote = runtime.quote(request)
    assert quote["quotes"][0]["ready_gpu_prefix_tokens"] == 2
    assert quote["host_quote_supported"] is True
    assert runtime.observed_seq == 0 and not runtime.epochs and not runtime.attempts
    cache.ongoing_load_back[1] = object()
    runtime.quote(request)
    assert len(cache.ongoing_load_back) == 1  # quote does not process transfer ACKs


def test_native_timing_uses_completion_clock_without_inventing_gpu_phases():
    from engine_patches.contextkv_runtime import native_timing
    timing = native_timing(SimpleNamespace(scheduler_recv_time=1, forward_entry_time=1.05,
                                           prefill_finished_time=1.2, completion_time=1.6), 5)
    assert timing["server_ttft_ms"] == pytest.approx(200)
    assert timing["server_elapsed_ms"] == pytest.approx(600)
    assert timing["forward_to_first_token_ms"] == pytest.approx(150)
    assert timing["decode_ms"] == pytest.approx(400)
    assert timing["tpot_ms"] == pytest.approx(100)
    assert "prefill_ms" not in timing and "queue_ms" not in timing
    assert native_timing(None, 0)["server_ttft_ms"] is None
    assert native_timing(SimpleNamespace(scheduler_recv_time=1, prefill_finished_time=1.1,
                                        completion_time=1.1), 1)["tpot_ms"] is None


def test_native_exact_result_replay_has_ttl_and_byte_bounds(monkeypatch):
    import engine_patches.contextkv_runtime as module
    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    runtime = ContextKVRuntime(FakeNativeCache(), {}, result_ttl_seconds=2, result_max_bytes=64)
    receipt = dict(finish_reason="stop", generation_complete=True)
    runtime.attempts["a"] = dict(state="COMPLETED", terminal=True, receipt=receipt, result=None)
    raw = dict(text="exact native stop-trimmed answer", output_ids=[1, 2],
               meta_info={"contextkv_timing": [{"server_ttft_ms": 3}]})
    assert runtime.publish_result({"attempt_id": "a", "response": raw})["stored"]
    assert runtime.status({"attempt_id": "a"})["result"]["text"] == raw["text"]
    assert runtime.status({"attempt_id": "a"})["result"]["server_ttft_ms"] == 3
    clock[0] = 13
    status = runtime.status({"attempt_id": "a"})
    assert status["terminal"] and status["receipt"] == receipt and status["result"] is None
    raw["text"] = "x" * 65
    assert not runtime.publish_result({"attempt_id": "a", "response": raw})["stored"]
    assert runtime.result_bytes == 0


def test_native_metrics_use_unique_allocator_counts_and_completed_ack_events():
    cache = FakeNativeCache()
    cache.token_to_kv_pool_allocator = SimpleNamespace(size=100, available_size=lambda: 40)
    cache.cache_controller = SimpleNamespace(mem_pool_host=SimpleNamespace(size=200, available_size=lambda: 150))
    runtime = ContextKVRuntime(cache, {}, event_limit=2)
    ack = SimpleNamespace(start_event=SimpleNamespace(elapsed_time=lambda end: 2.5),
                          finish_event=object(), timing_enabled=True, num_bytes=4096, num_tokens=32)
    runtime.record_transfer("D2H", ack)
    ack.timing_enabled = False
    runtime.record_transfer("H2D", ack)
    metrics = runtime.metrics({})
    assert metrics["pool"]["unique_gpu_tokens"] == 60
    assert metrics["pool"]["unique_host_tokens"] == 50
    assert metrics["events"][0]["duration_ms"] == 2.5
    assert metrics["events"][0]["clock"] == "cuda_event"
    assert metrics["events"][1]["duration_ms"] is None
    runtime.record_transfer("D2H", ack)
    assert runtime.metrics({"after_seq": 0})["events_truncated"]
    assert not runtime.metrics({"after_seq": 1})["events_truncated"]
    with pytest.raises(ContextKVError, match="PROFILING_DISABLED"):
        runtime.profile_prepare({"mode": "cold"})


def test_client_keeps_client_and_server_timing_separate_and_decodes_replay():
    async def scenario():
        fp = ConfigFingerprint()
        receipt = dict(rid="a", attempt_id="a", engine_session_id="s", session_generation=1,
                       engine_restart_id="r", input_hash=token_hash([1,2]),
                       finish_reason="stop", generation_complete=True)
        timing = dict(server_ttft_ms=5, server_elapsed_ms=9, decode_ms=4,
                      forward_to_first_token_ms=3, timing_source="sglang_server_monotonic", tpot_ms=4)
        def handler(request):
            if request.url.path == "/context/engine/info":
                return httpx.Response(200, json=dict(engine_restart_id="r", protocol_version="contextkv-v1",
                    guarded_generate_supported=True, scheduler_owned=True, completion_status_supported=True,
                    sglang_commit=SGLANG_COMMIT, fingerprint=asdict(fp), metrics_supported=True,
                    result_replay_supported=True, result_ttl_seconds=900, profiling_enabled=False))
            if request.url.path == "/generate":
                return httpx.Response(200, json=dict(text="answer", output_ids=[3,4],
                    meta_info={"contextkv_receipt":[receipt], "contextkv_timing":[timing]}))
            return httpx.Response(200, json=dict(state="COMPLETED", engine_restart_id="r", terminal=True,
                result=dict(text="answer",output_ids=[3,4],finish_reason="stop",receipt=receipt,**timing)))
        engine = SGLangEngineClient("http://test", fp, transport=httpx.MockTransport(handler), stream=False)
        result = await engine.generate(input_ids=[1,2],session_id="s",expected_generation=1,
            expected_restart_id="r",attempt_id="a",fingerprint=fp)
        assert result.server_ttft_ms == 5 and result.decode_ms == 4
        assert result.elapsed_ms is not None
        assert result.ttft_ms is result.last_token_ms is result.prefill_ms is result.queue_ms is None
        replay = await engine.status(attempt_id="a")
        assert replay.result.output_ids == (3,4)
        assert replay.result.server_elapsed_ms == 9
        await engine.aclose()
    run(scenario())
