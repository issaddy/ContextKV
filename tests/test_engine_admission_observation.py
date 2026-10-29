"""Pinned native observation seams, pure CPU: never simulate GPU performance."""
from array import array
import copy
import dataclasses
from types import SimpleNamespace

import pytest

from controller.schema import CacheQuoteBatch, NativePrefixMatch
from controller.serialization import decode, dumps, loads
from engine_patches.contextkv_runtime import ContextKVRuntime, HostAdmissionPolicy
from test_engine_host_quote import (extract_native_class, extract_native_method,
                                    install_native_policy, native_source)


class Request:
    def __init__(self, n=100, generated=4, maximum=1024, *, done=False, ignore_eos=False):
        self.rid=f"native-{id(self)}"
        self.origin_input_ids=array("q",range(n))
        self.output_ids=array("q",range(generated))
        self.sampling_params=SimpleNamespace(max_new_tokens=maximum,ignore_eos=ignore_eos)
        self.kv=SimpleNamespace(kv_allocated_len=n+generated,kv_committed_len=n+generated-1,
                                cache_protected_len=32)
        self.done=done
    def finished(self):
        return self.done


def bridge(running=(),waiting=(),overlap=()):
    tracker=extract_native_class(native_source("NewTokenRatioTracker"),"NewTokenRatioTracker",
                                 {"dataclass":dataclasses.dataclass})
    runtime=object.__new__(ContextKVRuntime)
    runtime.host_admission=HostAdmissionPolicy()
    runtime.scheduler=SimpleNamespace(running_batch=SimpleNamespace(reqs=list(running)),
        new_token_ratio_tracker=tracker(init=.7,min=.098,decay=.1,current=.7),
        waiting_queue=list(waiting),chunked_req=None,
        collect_inflight_reqs=lambda:set(running)|set(overlap))
    return runtime


def test_native_budget_uses_actual_batch_members_and_ratio_without_cleanup(monkeypatch):
    offset=install_native_policy(monkeypatch)
    active=Request(generated=8)
    early_eos=Request(generated=5,done=True)
    retracted=Request(generated=20)
    overlap=Request(generated=1)
    fixed=Request(generated=3,ignore_eos=True)
    runtime=bridge([active,early_eos,retracted,fixed],[retracted],[overlap])
    before=copy.deepcopy([r.__dict__ for r in (active,early_eos,retracted,fixed,overlap)])
    snapshot=runtime.workload_snapshot()
    assert len(snapshot["native_running_requests"])==3  # active, fixed, overlap
    assert len(snapshot["native_running_budget_requests"])==4  # exact native list
    assert snapshot["native_running_budget_requests"][1]["native_finished"] is True
    assert [r["request_id"] for r in snapshot["native_running_budget_requests"]] == [r.rid for r in runtime.scheduler.running_batch.reqs]
    owner=SimpleNamespace(new_token_ratio=.7)
    assert snapshot["native_running_budget_tokens"]==sum(offset(owner,r) for r in runtime.scheduler.running_batch.reqs)
    runtime.scheduler.new_token_ratio_tracker.decay_step()
    newer=runtime.workload_snapshot()
    assert newer["native_new_token_ratio"]==pytest.approx(.6)
    assert newer["native_running_budget_tokens"] < snapshot["native_running_budget_tokens"]
    assert newer["native_running_budget_requests"]==snapshot["native_running_budget_requests"]
    assert [r.__dict__ for r in (active,early_eos,retracted,fixed,overlap)]==before
    assert runtime.scheduler.running_batch.reqs==[active,early_eos,retracted,fixed]
    assert runtime.scheduler.waiting_queue==[retracted]


@pytest.mark.parametrize("maximum,generated,ignore_eos,expected",[
    (10000,5000,False,4096*.7), (10000,8000,False,2000*.7),
    (10000,5000,True,4096), (1024,1024,False,0),
])
def test_native_offset_remaining_then_clip_and_ignore_eos(monkeypatch,maximum,generated,ignore_eos,expected):
    install_native_policy(monkeypatch)
    req=Request(maximum=maximum,generated=generated,ignore_eos=ignore_eos)
    assert bridge([req]).workload_snapshot()["native_running_budget_tokens"]==expected


def test_native_match_hook_observes_only_native_call_and_does_not_promise_fresh_residency(monkeypatch):
    install_native_policy(monkeypatch)
    req=Request()
    req._compute_max_prefix_len=lambda n:n-1
    runtime=bridge(waiting=[req])
    observed=SimpleNamespace(device_prefix_len=32,last_device_node=object(),last_host_node=object(),
        best_match_node=object(),host_hit_length=64,swa_host_hit_length=0,mamba_host_hit_length=0,
        swa_branching_seqlen=None,mamba_branching_seqlen=None,cache_protected_len=None)
    calls=[]
    tree=SimpleNamespace(contextkv_runtime=runtime,match_prefix=lambda p:(calls.append(p) or observed))
    namespace={"MatchPrefixParams":SimpleNamespace,"_req_radix_key":lambda *args:args[2],
               "envs":SimpleNamespace(SGLANG_RADIX_FORCE_MISS=SimpleNamespace(get=lambda:False))}
    original=extract_native_method(native_source("match_kv_cache"),"match_kv_cache",namespace)
    patched=extract_native_method(native_source("match_kv_cache",patched=True),"match_kv_cache",namespace)
    old_req=copy.deepcopy(req)
    assert original(old_req,tree) is observed
    assert not hasattr(old_req,"contextkv_last_native_match")
    ticks=iter([1000,2500])
    monkeypatch.setattr("engine_patches.contextkv_runtime.time.monotonic_ns",lambda:next(ticks))
    assert runtime.workload_snapshot()["native_waiting_requests"][0]["last_native_match"] is None
    assert patched(req,tree) is observed
    assert len(calls)==2  # exactly one original match per invocation
    assert {k:v for k,v in req.__dict__.items() if k!="contextkv_last_native_match"}==old_req.__dict__
    expected=dict(observed_time_ns=1000,input_tokens=104,device_prefix_tokens=32,
                  host_hit_tokens=64,matched_prefix_tokens=96)
    assert req.contextkv_last_native_match==expected
    first=runtime.workload_snapshot()
    # Native scheduling can subsequently update these live fields. The saved
    # observation remains old; reads must not claim a new match or new residency.
    req.prefix_len,req.host_hit_length=0,0
    assert runtime.workload_snapshot()==first
    first["native_waiting_requests"][0]["last_native_match"]["device_prefix_tokens"]=999
    assert req.contextkv_last_native_match==expected  # exported dict is a copy
    patched(req,tree)
    assert req.contextkv_last_native_match["observed_time_ns"]==2500 and len(calls)==3
    assert set(req.contextkv_last_native_match)==set(expected)  # one bounded snapshot
    wire=dict(request_id="q",observed_seq=1,observed_time_ns=3000,engine_restart_id="r",
              cache_config_hash="c",free_gpu_tokens=0,free_host_tokens=0,quotes=[],**runtime.workload_snapshot())
    batch=decode(CacheQuoteBatch,wire)
    saved=batch.native_waiting_requests[0].last_native_match
    assert isinstance(saved,NativePrefixMatch) and batch.observed_time_ns-saved.observed_time_ns==500
    assert loads(CacheQuoteBatch,dumps(batch))==batch
    tree.contextkv_runtime=None
    uncaptured=Request()
    uncaptured._compute_max_prefix_len=lambda n:n-1
    assert patched(uncaptured,tree) is observed
    assert not hasattr(uncaptured,"contextkv_last_native_match")


def test_unknown_match_is_not_zero_hit_and_legacy_batch_roundtrips():
    wire=dict(request_id="q",observed_seq=0,observed_time_ns=1,engine_restart_id="r",
              cache_config_hash="c",free_gpu_tokens=0,free_host_tokens=0,quotes=[],
              native_waiting_requests=[dict(input_tokens=100,generated_tokens=0,max_new_tokens=10,
                  kv_allocated_tokens=0,kv_committed_tokens=0,cache_protected_tokens=0)])
    old=decode(CacheQuoteBatch,wire)
    assert old.native_running_budget_tokens is old.native_new_token_ratio is None
    assert old.native_running_budget_requests is None
    assert old.native_waiting_requests[0].last_native_match is None
    assert loads(CacheQuoteBatch,dumps(old))==old
