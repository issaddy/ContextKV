"""Shared native pool observations include retained cache and preserve missingness."""
import asyncio

import httpx
import pytest

from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series, pool_snapshot
from bench.report import summarize_run


def scrape(free=4, cached=12, used=0, host=24, rank=0):
    labels = f'model_name="fixture",tp_rank="{rank}",pp_rank="0"'
    return "\n".join(f'{name}{{{labels}}} {value}' for name,value in (
        ("sglang:kv_available_tokens", free), ("sglang:kv_evictable_tokens", cached),
        ("sglang:kv_used_tokens", used), ("sglang:hicache_host_used_tokens", host)))+"\n"


def test_native_slots_include_evictable_cache_without_summing_replicated_ranks():
    text = scrape()+scrape(free=0,cached=16,rank=1)
    assert pool_snapshot(text,gpu_capacity=16,host_enabled=True,model="fixture") == {
        "gpu_slots": 12, "host_slots": 24}
    assert pool_snapshot(scrape(free=0,cached=16),gpu_capacity=16,host_enabled=False,model="fixture")["gpu_slots"] == 16
    with pytest.raises(ValueError,match="uninitialized"):
        pool_snapshot(scrape(free=0,cached=0),gpu_capacity=16,host_enabled=True,model="fixture")


def observation(samples):
    return {"samples": samples, "measurement":"sampled_unique_pool_slots",
            "interval_seconds":.5,"scope":"test fixture","errors":[]}


def test_series_clips_to_request_window_and_does_not_backfill_from_future():
    samples=[{"timestamp":0,"gpu_slots":0,"host_slots":0},
             {"timestamp":2,"gpu_slots":4,"host_slots":8},
             {"timestamp":5,"gpu_slots":6,"host_slots":10}]
    values,info=pool_series(observation(samples),1,4)
    assert [v["timestamp"] for v in values]==[1,2,4]
    assert occupancy_integral(values)["gpu_slot_seconds"]==8
    assert info["complete_window_coverage"]
    partial,info=pool_series(observation(samples[1:]),1,4)
    assert partial[0]["timestamp"]==2 and not info["complete_window_coverage"]


def test_scrape_gap_only_covers_the_integrated_window():
    samples = [{"timestamp": t, "gpu_slots": 1, "host_slots": 2}
               for t in (0, 30, 30.4, 30.8, 31.2, 60)]
    _, info = pool_series(observation(samples), 30.2, 31)
    assert info["complete_window_coverage"]
    assert info["max_scrape_gap_seconds"] == pytest.approx(.4)


def test_http_observation_failure_does_not_abort_work_and_sampler_closes():
    requests=[]
    def respond(request):
        requests.append(request)
        return httpx.Response(503) if len(requests)==1 else httpx.Response(200,text=scrape())
    async def run():
        sampler=NativePoolSampler("http://engine",gpu_capacity=16,host_enabled=True,
            model="fixture",initial_empty=True,interval=60,transport=httpx.MockTransport(respond))
        await sampler.start()
        await sampler.stop()
        result=sampler.observation()
        assert result["samples"][0]["basis"]=="fresh_owned_engine_no_generation_warmup"
        assert result["samples"][-1]["gpu_slots"]==12
        assert len(result["errors"])==1 and sampler._client.is_closed
    asyncio.run(run())
    assert len(requests)==2


def test_report_uses_same_pool_basis_and_does_not_invent_transfer_measurements():
    sample=observation([{"timestamp":0,"gpu_slots":4,"host_slots":8},
                        {"timestamp":11,"gpu_slots":0,"host_slots":0}])
    payload={"engine_kind":"sglang","native_pool_observation":sample,
        "report":{"started_at":1,"arrival_window_end":10,"completed_at":10,
            "time_origin":"pipeline_arrival","scheduled_sessions":1,"arrived_sessions":1,
            "unfinished_turns":0,"timed_out_sessions":[],"session_makespans":[["s",2]],
            "turns":[{"task_id":"s:1","conversation_id":"s","status":"success",
                "assistant_text":"fixture","t_arrive":1,"t_done":3,
                "result":{"output_ids":[1],"truncated":False}}]}}
    result=summarize_run(payload)
    assert result["occupancy"]["gpu_slot_seconds"]==36
    assert result["resource_efficiency"]["gpu_slot_seconds_per_nontruncated_turn"]==36
    assert result["transfers"] is None
    payload["native_pool_observation"]["samples"][0]["timestamp"]=2
    result=summarize_run(payload)
    assert result["resource_efficiency"]["gpu_slot_seconds_per_nontruncated_turn"] is None
