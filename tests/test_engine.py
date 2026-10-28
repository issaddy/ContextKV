"""Engine protocol invariants; GPU timings and model quality are not simulated."""
import asyncio
from collections import OrderedDict
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from controller.engine_client import (EngineError, SGLangEngineClient, UnknownOutcomeError,
                                      SGLANG_COMMIT, token_hash)
from controller.mock_engine import MockEngine
from controller.schema import ConfigFingerprint, NativePrefixMatch, NativeRequestSummary, PromptCandidate
from engine_patches.contextkv_runtime import ContextKVError, ContextKVRuntime


def candidate(ids, name="a"):
    return PromptCandidate(name, "APPEND", 0, (), tuple(ids), token_hash(ids), frozenset(), frozenset())


def test_mock_slot_events_scale_with_operations_and_preserve_pool_totals():
    async def check():
        engine = MockEngine(gpu_capacity_tokens=12000, host_capacity_tokens=20000)
        await engine.seed_cache(range(10000), tier="gpu")
        await engine.seed_cache(range(10000), tier="gpu")  # Shared slots are not allocated twice.
        await engine.seed_cache(range(10000), tier="host")
        engine._admit(tuple(range(20000, 30000)))
        await engine.seed_cache(range(20000, 30000), tier="gpu")
        metrics = await engine.metrics()
        assert metrics["pool"]["unique_gpu_tokens"] == 12000
        assert metrics["pool"]["unique_host_tokens"] == 10000
        assert [(e["event"], e["tier"], e["slots"]) for e in metrics["events"]] == [
            ("alloc", "gpu", 10000), ("alloc", "host", 10000),
            ("free", "gpu", 8000), ("alloc", "gpu", 10000)]
    asyncio.run(check())


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


@pytest.mark.parametrize("workload", ["legacy", "null", "empty", "one"])
def test_http_quote_decodes_optional_native_workload_contract(workload):
    async def scenario():
        fp = ConfigFingerprint()
        item = candidate((1,2,3))
        info = dict(engine_restart_id="r", protocol_version="contextkv-v1", engine_kind="sglang",
                    guarded_generate_supported=True, scheduler_owned=True,
                    completion_status_supported=True, sglang_commit=SGLANG_COMMIT, fingerprint=asdict(fp))
        summary = NativeRequestSummary(input_tokens=100,generated_tokens=4,max_new_tokens=20,
            kv_allocated_tokens=108,kv_committed_tokens=103,cache_protected_tokens=64,
            native_finished=False,last_native_match=NativePrefixMatch(1000,100,32,32,64))
        wire = dict(request_id="q",observed_seq=0,observed_time_ns=1,engine_restart_id="r",
            cache_config_hash="c",free_gpu_tokens=0,free_host_tokens=10,fingerprint=asdict(fp),
            quotes=[dict(candidate_id="a",input_hash=item.input_hash,input_len=3,match_limit=2)])
        if workload != "legacy":
            wire.update(gpu_capacity_tokens=16384,evictable_gpu_tokens=128,evictable_host_tokens=None,
                native_running_requests=(None if workload == "null" else
                                         [asdict(summary)] if workload == "one" else []),
                native_waiting_requests=[])
            wire["quotes"][0].update(native_estimate_uncertain=True,
                                      uncertainty_reasons=["capacity_pressure"])
        if workload == "one":
            wire.update(native_new_token_ratio=.125,native_max_new_tokens_clip=4096,
                        native_running_budget_tokens=2.0,native_running_budget_requests=[asdict(summary)])
        client = SGLangEngineClient("http://test",fp,transport=httpx.MockTransport(
            lambda request:httpx.Response(200,json=info if request.url.path == "/context/engine/info" else wire)))
        try:
            result = await client.quote_batch([item],fp)
            assert result.native_running_requests == ((summary,) if workload == "one" else
                                                     () if workload == "empty" else None)
            if workload != "legacy":
                assert result.native_waiting_requests == ()
                assert result.gpu_capacity_tokens == 16384 and result.evictable_host_tokens is None
                assert result.quotes[0].uncertainty_reasons == ("capacity_pressure",)
            else:
                assert result.gpu_capacity_tokens is None
                assert result.quotes[0].uncertainty_reasons == ()
            if workload == "one":
                assert result.native_running_budget_requests == (summary,)
                assert result.native_new_token_ratio == .125 and result.native_running_budget_tokens == 2
                assert isinstance(result.native_running_requests[0].last_native_match,NativePrefixMatch)
            else:
                assert result.native_new_token_ratio is result.native_max_new_tokens_clip is None
                assert result.native_running_budget_requests is result.native_running_budget_tokens is None
        finally:
            await client.aclose()
    run(scenario())


def test_nonstream_native_top_level_409_preserves_message_and_http_status():
    async def scenario():
        # Actual pinned SGLang /generate body observed after a retired epoch.
        body={"object":"error","message":"EPOCH_RETIRED: EPOCH_RETIRED",
              "type":"409","param":None,"code":409}
        client=SGLangEngineClient("http://test",ConfigFingerprint(),stream=False,
            transport=httpx.MockTransport(lambda request:httpx.Response(409,json=body)))
        try:
            with pytest.raises(EngineError) as caught:
                await client._post("/generate",{})
            assert caught.value.code=="HTTP_ERROR"
            assert str(caught.value)=="HTTP_ERROR: HTTP 409: EPOCH_RETIRED: EPOCH_RETIRED"
        finally:
            await client.aclose()
    run(scenario())


def test_nested_context_error_retains_its_original_protocol_code():
    async def scenario():
        client=SGLangEngineClient("http://test",ConfigFingerprint(),
            transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
                "error":{"code":"STALE_GENERATION","message":"generation changed"}})))
        try:
            with pytest.raises(EngineError) as caught:
                await client._post("/context/epoch/retire",{})
            assert caught.value.code=="STALE_GENERATION"
            assert str(caught.value)=="STALE_GENERATION: generation changed"
        finally:
            await client.aclose()
    run(scenario())


class FakeNativeCache:
    """Only the existing native tracker boundary is mocked in these tests."""
    page_size = 1
    _tree_core_backend = "python"
    cache_controller = None

    def __init__(self, closed_limit=8192):
        self.session_refs = SimpleNamespace(_session_generations={},_closed_session_ids=OrderedDict())
        self.closed_limit = closed_limit
        self.counter = 0
        self.released = []

    def open_radix_session(self, sid):
        self.session_refs._closed_session_ids.pop(sid,None)
        self.counter += 1
        self.session_refs._session_generations[sid] = self.counter
        return self.counter

    def release_radix_session(self, sid):
        self.session_refs._session_generations.pop(sid, None)
        self.session_refs._closed_session_ids[sid] = None
        self.session_refs._closed_session_ids.move_to_end(sid)
        while len(self.session_refs._closed_session_ids) > self.closed_limit:
            self.session_refs._closed_session_ids.popitem(last=False)
        self.released.append(sid)


def prepare_runtime_epoch(runtime, sid, operation=None):
    return runtime.prepare(dict(engine_session_id=sid, operation_id=operation or 'prepare:'+sid,
                                expected_restart_id=runtime.restart_id))


def retire_runtime_epoch(runtime, prepared, operation=None):
    return runtime.retire(dict(engine_session_id=prepared['engine_session_id'],
        expected_generation=prepared['session_generation'],
        operation_id=operation or 'retire:'+prepared['engine_session_id'],
        expected_restart_id=runtime.restart_id))


def test_terminal_metadata_is_bounded_without_pruning_active_or_retiring_epochs():
    cache = FakeNativeCache(closed_limit=8)
    runtime = ContextKVRuntime(cache, {}, receipt_limit=3)
    live = prepare_runtime_epoch(runtime,'live')
    pending = prepare_runtime_epoch(runtime,'pending')
    runtime.epochs['pending']['inflight'].add('unfinished')
    assert not retire_runtime_epoch(runtime,pending)['acknowledged']
    for i in range(30):
        prepared = prepare_runtime_epoch(runtime,f'closed-{i}')
        assert retire_runtime_epoch(runtime,prepared)['acknowledged']
        assert len(runtime.retired_epochs) <= 3
        assert len(runtime.epochs) <= 5
        assert len(runtime.operation_receipts) <= 3
    # New operation IDs for an already closed epoch must not accumulate either.
    for i in range(30):
        assert retire_runtime_epoch(runtime,prepared,f'retry-{i}')['acknowledged']
    assert set(runtime.operations) == {'prepare:live','prepare:pending','retire:pending'}
    assert len(runtime.operation_receipts) == 3
    assert runtime.status({'operation_id':'prepare:pending'})['state'] == 'RETIRING'
    assert runtime.status({'engine_session_id':'live'})['state'] == 'PREPARED'
    assert cache.session_refs._session_generations == {
        'live':live['session_generation'],'pending':pending['session_generation']}
    assert 'pending' not in cache.released
    cache.token_to_kv_pool_allocator = SimpleNamespace(size=100,available_size=lambda:100)
    assert runtime.metrics({})['control_metadata'] == {
        'active_epochs':2,'retained_retired_epochs':3,'active_operations':3,
        'retained_operation_receipts':3,'terminal_record_limit':3}


def test_lost_retire_ack_recovers_after_metadata_expiry_without_touching_new_incarnation():
    cache = FakeNativeCache(closed_limit=2)
    runtime = ContextKVRuntime(cache, {}, receipt_limit=2)
    old = prepare_runtime_epoch(runtime,'old')
    original_ack = retire_runtime_epoch(runtime,old)
    for i in range(5):
        retire_runtime_epoch(runtime,prepare_runtime_epoch(runtime,str(i)))
    assert 'old' not in runtime.epochs and 'old' not in cache.session_refs._closed_session_ids
    assert runtime.status({'engine_session_id':'old'})['state']=='UNKNOWN'
    releases = list(cache.released)
    assert retire_runtime_epoch(runtime,old) == original_ack
    assert cache.released == releases and 'old' not in cache.session_refs._session_generations

    # Explicit preparation after native close-history expiry gets a fresh fence.
    new = prepare_runtime_epoch(runtime,'old','new-operation')
    assert new['session_generation'] > old['session_generation']
    late = SimpleNamespace(session_id='old',expected_generation=old['session_generation'],
        expected_restart_id=runtime.restart_id,contextkv_fingerprint={})
    with pytest.raises(ContextKVError,match='STALE_GENERATION'):
        runtime.admit(late,SimpleNamespace())
    assert retire_runtime_epoch(runtime,old) == original_ack
    assert cache.released == releases
    assert cache.session_refs._session_generations['old'] == new['session_generation']
    assert runtime.epochs['old']['state']=='PREPARED'
    future = {**new,'session_generation':new['session_generation']+1}
    with pytest.raises(ContextKVError,match='STALE_GENERATION'):
        retire_runtime_epoch(runtime,future,'future-generation')
    assert 'future-generation' not in runtime.operations


def test_native_close_guard_survives_expiry_of_research_metadata():
    cache = FakeNativeCache(closed_limit=8)
    runtime = ContextKVRuntime(cache, {}, receipt_limit=1)
    old = prepare_runtime_epoch(runtime,'old')
    retire_runtime_epoch(runtime,old)
    retire_runtime_epoch(runtime,prepare_runtime_epoch(runtime,'next'))
    assert 'old' not in runtime.epochs and 'old' in cache.session_refs._closed_session_ids
    generation = cache.counter
    with pytest.raises(ContextKVError,match='EPOCH_CONFLICT'):
        prepare_runtime_epoch(runtime,'old','another-operation')
    assert cache.counter == generation


def test_native_cache_reset_rotates_restart_before_reusing_generation_values():
    from engine_patches.contextkv_runtime import get_runtime
    cache = FakeNativeCache()
    runtime = ContextKVRuntime(cache, {}, receipt_limit=2,event_limit=7)
    scheduler = SimpleNamespace(tree_cache=cache,contextkv_runtime=runtime)
    old = prepare_runtime_epoch(runtime,'same')
    assert get_runtime(scheduler) is runtime
    # Exact pinned reset behavior: a fresh generation dictionary and counter.
    cache.session_refs._session_generations = {}
    cache.session_refs._closed_session_ids = OrderedDict()
    cache.counter = 0
    replacement = get_runtime(scheduler)
    assert replacement is scheduler.contextkv_runtime is cache.contextkv_runtime
    assert replacement.restart_id != runtime.restart_id
    assert replacement.receipt_limit==2 and replacement.events.maxlen==7
    current = prepare_runtime_epoch(replacement,'same')
    assert current['session_generation']==old['session_generation']
    with pytest.raises(ContextKVError,match='STALE_RESTART'):
        replacement.admit(SimpleNamespace(session_id='same',expected_restart_id=old['engine_restart_id'],
            expected_generation=old['session_generation']),SimpleNamespace())
    with pytest.raises(ContextKVError,match='STALE_RESTART'):
        replacement.retire(dict(engine_session_id='same',expected_generation=old['session_generation'],
            operation_id='late-retire',expected_restart_id=old['engine_restart_id']))
    assert cache.session_refs._session_generations['same']==current['session_generation']


def test_durable_outbox_can_acknowledge_expired_retirement_metadata(tmp_path):
    from controller.engine_client import RetireAck
    from controller.epoch import EpochCoordinator
    from controller.events import EventLog
    from controller.schema import ContextEpoch
    from controller.session_store import SessionStore

    async def scenario():
        cache = FakeNativeCache(closed_limit=2)
        runtime = ContextKVRuntime(cache, {}, receipt_limit=2)
        old = prepare_runtime_epoch(runtime,'old')
        store = SessionStore(tmp_path/'state.sqlite3')
        store._enqueue('business',ContextEpoch('business',1,'old',old['session_generation'],
            engine_restart_id=runtime.restart_id),'replaced',1)
        retire_runtime_epoch(runtime,old)  # Successful release; ACK was lost.
        for i in range(5):
            retire_runtime_epoch(runtime,prepare_runtime_epoch(runtime,str(i)))

        class Engine:
            async def retire(self,sid,generation,operation,restart,reason):
                return RetireAck(**runtime.retire(dict(engine_session_id=sid,
                    expected_generation=generation,operation_id=operation,
                    expected_restart_id=restart,reason=reason)))
        try:
            assert len(store.outbox())==1 and 'old' not in runtime.epochs
            releases = list(cache.released)
            assert await EpochCoordinator(store,Engine(),EventLog()).flush()==1
            assert not store.outbox() and cache.released==releases
        finally:
            store.close()
    run(scenario())


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


def test_rejections_are_bounded_and_do_not_evict_active_or_newly_finished_results():
    runtime = ContextKVRuntime(FakeNativeCache(), {}, receipt_limit=3)
    prepared = prepare_runtime_epoch(runtime, "live")
    recv = SimpleNamespace(session_id="live", expected_restart_id=runtime.restart_id,
        expected_generation=prepared["session_generation"], contextkv_fingerprint={},
        contextkv_attempt_id="live-attempt", rid="live-attempt", session_params=None,
        lora_id=None, mm_inputs=None, input_embeds=None, return_logprob=False,
        cache_salt=None, extra_key=None, input_ids=[1, 2])
    req = SimpleNamespace()
    runtime.admit(recv, req)
    for index in range(20):
        rejected = SimpleNamespace(**{**vars(recv), "rid": f"rejected-{index}",
            "contextkv_attempt_id": f"rejected-{index}", "expected_generation": 0})
        with pytest.raises(ContextKVError) as error:
            runtime.admit(rejected, SimpleNamespace())
        runtime.record_rejection(rejected, error.value)
        assert len(runtime.attempts) <= 4  # Three terminal records plus one active request.
        assert runtime.status({"attempt_id": "live-attempt"})["inflight"]
    assert runtime.status({"attempt_id": "rejected-0"})["state"] == "UNKNOWN"
    assert runtime.status({"attempt_id": "rejected-19"})["state"] == "REJECTED"
    req.finished = lambda: True
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
    req.rid, req.session_id = "live-attempt", "live"
    req.session_generation = prepared["session_generation"]
    req.cached_tokens_device = req.cached_tokens_host = req.cached_tokens = 0
    req.kv = SimpleNamespace(cache_inserted_len=2)
    req.customized_info = None
    req.output_ids = [3]
    runtime.finish(req)
    assert len(runtime.attempts) == 3
    assert runtime.publish_result({"attempt_id": "live-attempt",
                                   "response": {"text": "answer", "output_ids": [3]}})["stored"]
    assert runtime.status({"attempt_id": "live-attempt"})["result"]["text"] == "answer"


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
    import json
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
    cache.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 100,size=256)
    cache.full_evictable_size = lambda:64
    fp = ConfigFingerprint()
    class ActiveRequest:
        rid = "actual-native-request"
        origin_input_ids = (1,2,3)
        output_ids = (4,)
        sampling_params = SimpleNamespace(max_new_tokens=8,ignore_eos=False)
        kv = SimpleNamespace(kv_allocated_len=4,kv_committed_len=3,cache_protected_len=2)
        def finished(self):
            return False
    active = ActiveRequest()
    from test_engine_host_quote import install_native_policy
    install_native_policy(monkeypatch)
    scheduler = SimpleNamespace(collect_inflight_reqs=lambda:{active},waiting_queue=[],chunked_req=None,
        running_batch=SimpleNamespace(reqs=[active]),new_token_ratio_tracker=SimpleNamespace(current=.5))
    runtime = ContextKVRuntime(cache,asdict(fp),scheduler=scheduler)
    active.prefix_len, active.host_hit_length, active.num_matched_prefix_tokens = 2,0,2
    runtime.record_native_match(active,3)
    match = NativePrefixMatch(**active.contextkv_last_native_match)
    request = {"request_id": "q", "fingerprint": asdict(fp), "candidates": [
        {"candidate_id": "a", "snapshot_version": 0, "input_ids": [1, 2, 3], "input_hash": token_hash([1, 2, 3])}]}
    quote = runtime.quote(request)
    assert quote["observed_time_ns"] >= match.observed_time_ns
    assert quote["quotes"][0]["ready_gpu_prefix_tokens"] == 2
    assert quote["host_quote_supported"] is True
    assert runtime.observed_seq == 0 and not runtime.epochs and not runtime.attempts
    cache.ongoing_load_back[1] = object()
    runtime.quote(request)
    assert len(cache.ongoing_load_back) == 1  # quote does not process transfer ACKs
    async def scenario():
        def handler(request):
            handler = runtime.info if request.url.path == "/context/engine/info" else runtime.quote
            return httpx.Response(200,json=handler(json.loads(request.content)))
        client = SGLangEngineClient("http://test",fp,transport=httpx.MockTransport(handler))
        try:
            decoded = await client.quote_batch([candidate((1,2,3))],fp)
            assert decoded.active_requests == 1 and decoded.queued_requests == 0
            assert decoded.native_running_requests == (NativeRequestSummary(
                input_tokens=3,generated_tokens=1,max_new_tokens=8,kv_allocated_tokens=4,
                kv_committed_tokens=3,cache_protected_tokens=2,native_finished=False,last_native_match=match,
                request_id=active.rid),)
            assert decoded.native_running_budget_requests == decoded.native_running_requests
            assert decoded.native_running_budget_tokens == 3.5
            assert decoded.native_new_token_ratio == .5 and decoded.native_max_new_tokens_clip == 4096
            assert decoded.native_waiting_requests == () and decoded.evictable_host_tokens is None
        finally:
            await client.aclose()
    run(scenario())


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
