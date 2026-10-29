"""Controlled C attribution protocol checks. These doubles are not GPU evidence."""
import asyncio
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import pytest

from bench.mock_engine import MockEngine
from engine_patches.contextkv_runtime import ContextKVRuntime, ContextKVError
from profiling.lifecycle_probe import LifecycleProbe
from test_engine import FakeNativeCache


class LifecycleDouble(MockEngine):
    async def info(self):
        return replace(await super().info(),engine_kind="sglang",profiling_enabled=True)

    async def profile_prepare(self,mode,*,tier=None,tokens=None):
        if mode == "cold":
            await self.restart()
            return {}
        if mode == "settle":
            for slot in self._slots.values():
                slot.host |= slot.gpu
            return {}
        assert mode == "pressure"
        before = (await self.metrics())["pool"]
        for _ in range(tokens):
            allocated = {index for index,slot in self._slots.items() if getattr(slot,tier)}
            parents = {parent for (parent,_),child in self._children.items() if child in allocated}
            leaves = allocated-parents
            if not leaves:
                break
            victim = min(leaves,key=lambda index:(bool(self._slots[index].refs),
                len(self._slots[index].refs),self._slots[index].touched,index))
            setattr(self._slots[victim],tier,False)
        after = (await self.metrics())["pool"]
        freed = before[f"unique_{tier}_tokens"]-after[f"unique_{tier}_tokens"]
        return dict(mode=mode,tier=tier,requested_tokens=tokens,native_evicted_tokens=freed,
            pool_before=before,pool_after=after,refs_released=False,
            gpu_tokens_freed=before["unique_gpu_tokens"]-after["unique_gpu_tokens"],
            host_tokens_freed=before["unique_host_tokens"]-after["unique_host_tokens"])


def test_paired_lifecycle_probe_attributes_native_pressure_not_retire_ack():
    async def scenario():
        engine = LifecycleDouble(response_text="x")
        report = await LifecycleProbe(engine,range(10),shared_tokens=4,branch_tokens=4).run()
        assert report["status"] == "effect_observed"
        for tier in ("gpu","host"):
            pair = report["trials"][tier]
            assert pair["new_prefix_preserved_gain_tokens"] == 4
            assert pair["treatment"]["retained_prefix_tokens"] == {"old":4,"new":8,"shared":4}
            assert pair["control"]["retained_prefix_tokens"] == {"old":8,"new":4,"shared":4}
            assert pair["treatment"]["retirement_pool_delta"] == {"unique_gpu_tokens":0,"unique_host_tokens":0}
            assert not pair["treatment"]["pressure"]["refs_released"]
    asyncio.run(scenario())


def test_lifecycle_probe_rejects_non_profiler():
    with pytest.raises(ValueError,match="dedicated"):
        asyncio.run(LifecycleProbe(MockEngine(),range(10)).run())


@pytest.mark.parametrize("tier",["gpu","host"])
def test_native_pressure_is_profiler_gated_and_preserves_epoch_refs(monkeypatch,tier):
    module = ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    module.EvictParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setenv("CONTEXTKV_ENABLE_PROFILING","1")
    free = {"gpu":40,"host":60}
    cache = FakeNativeCache()
    cache.token_to_kv_pool_allocator = SimpleNamespace(size=100,available_size=lambda:free["gpu"])
    cache.cache_controller = SimpleNamespace(io_backend="kernel",write_policy="write_through",
        mem_pool_host=SimpleNamespace(size=100,layout="layer_first",available_size=lambda:free["host"]))
    cache.flush_pending_backups = lambda:None
    cache.writing_check = lambda **kwargs:None
    cache.loading_check = lambda:None
    def evict_gpu(params):
        free["gpu"] += params.num_tokens
        return SimpleNamespace(num_tokens_evicted=params.num_tokens)
    def evict_host(tokens):
        free["host"] += tokens
        return tokens
    cache.evict,cache.evict_host = evict_gpu,evict_host
    scheduler = SimpleNamespace(is_fully_idle=lambda:True)
    runtime = ContextKVRuntime(cache,{},scheduler=scheduler,transfer_tma_enabled=False)
    ack = runtime.prepare({"engine_session_id":"current","operation_id":"prepare",
                           "expected_restart_id":runtime.restart_id})
    output = runtime.profile_prepare({"mode":"pressure","tier":tier,"tokens":10,
                                      "expected_restart_id":runtime.restart_id})
    assert output["native_evicted_tokens"] == 10 and not output["refs_released"]
    assert cache.released == []
    assert cache.session_refs._session_generations["current"] == ack["session_generation"]
    assert runtime.cache_config["host_layout"] == "layer_first"
    assert runtime.cache_config["hicache_io_backend"] == "kernel"
    assert runtime.cache_config["tma_transfer_enabled"] is False
    runtime.profiling_enabled = False
    with pytest.raises(ContextKVError,match="PROFILING_DISABLED"):
        runtime.profile_prepare({"mode":"pressure","tier":tier,"tokens":10,
                                 "expected_restart_id":runtime.restart_id})
