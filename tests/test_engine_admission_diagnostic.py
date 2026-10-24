"""Opt-in observations of fixed native gates; no CUDA/performance simulation."""
import ast
from collections import namedtuple
from contextlib import nullcontext
from enum import Enum, auto
import json
from types import MethodType, SimpleNamespace
import textwrap

import pytest

from engine_patches.contextkv_runtime import (
    AdmissionDiagnostic, ContextKVRuntime, record_admission_gate,
    record_admission_outcome, record_scheduler_gate,
)
from test_engine import FakeNativeCache
from test_engine_host_quote import extract_native_class, extract_native_method, native_source


class Result(Enum):
    CONTINUE = auto()
    NO_TOKEN = auto()
    OTHER = auto()


def request(n=257, diagnostic=True):
    req = SimpleNamespace(rid="r", origin_input_ids=list(range(n)),
        full_untruncated_fill_ids=list(range(n)), prefix_len=0, host_hit_length=0,
        swa_host_hit_length=0, auxiliary_swa_host_hit_length=0, output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=128, ignore_eos=False),
        last_node=object(), best_match_node=object(), needs_host_load_back=lambda:False)
    if diagnostic:
        req.contextkv_admission_diagnostic = AdmissionDiagnostic()
    return req


def adder(capacity=450, *, patched=True):
    # Execute the original pinned single-pool admission math and patched native
    # add_one_req/_select_prefill_admission. Only allocation/locking is doubled.
    budget_type = extract_native_class(native_source("PrefillBudget"), "PrefillBudget")
    tree = SimpleNamespace(supports_mamba=lambda:False, evictable_size=lambda:0,
                           disable=False, buffer_pipeline=None)
    allocator = SimpleNamespace(available_size=lambda:capacity, page_size=64)
    budget = budget_type(allocator, tree)
    owner = SimpleNamespace(page_size=64, rem_input_tokens=8192, rem_chunk_tokens=None,
        new_token_ratio=.7, can_run_list=[], memory_budget=budget, tree_cache=tree,
        prefill_max_requests=None, per_req_token_overhead=64, rem_mamba_slots=None,
        prefill_delayer_single_pass=None, dllm_config=None, exact_chunk_fill=False,
        _mamba_gap_budget_for_req=lambda *a,**k:0, _request_slot_reserve_for_req=lambda *a,**k:0,
        _lock_node=lambda *a,**k:nullcontext(), _replay_len=lambda *a:0,
        ceil_paged_tokens=budget.ceil_paged_tokens,
        _kv_shard_reserve_scratch=lambda **kw:True,
        _check_prefill_tile_budget=lambda n:None,
        _defer_staged_for_capacity_if_idle=lambda *a,**k:None)
    # Native property values must remain live as reserve changes their offsets.
    class Owner(SimpleNamespace):
        @property
        def rem_total_tokens(self):
            return self.memory_budget.remaining_total
        @property
        def cur_rem_tokens(self):
            return self.memory_budget.remaining_current
    owner = Owner(**vars(owner))
    owner._check_prefill_budget = lambda req,**kw:budget.check_prefill(
        max_new_tokens=128, input_tokens=len(req.origin_input_ids), **kw)
    def commit(req, selected, gap):
        owner.can_run_list.append(req)
        budget.reserve(selected.extend_len, selected.max_new_tokens)
        owner.rem_input_tokens -= budget.ceil_paged_tokens(selected.extend_len)
    owner._commit_prefill_admission = commit
    owner.budget_state = lambda:(Result.NO_TOKEN if not budget.has_capacity() else
        Result.OTHER if owner.rem_input_tokens <= 0 else Result.CONTINUE)
    ns = dict(AddReqResult=Result, CLIP_MAX_NEW_TOKENS=4096,
        _PrefillAdmission=namedtuple("Shape", "prefix_len extend_len max_new_tokens is_chunked replay_len"),
        record_admission_gate=record_admission_gate, UnifiedRadixCache=type("Unified",(),{}),
        InitLoadBackParams=SimpleNamespace, nullcontext=nullcontext)
    for method in ("_select_prefill_admission", "add_one_req"):
        setattr(owner, method, MethodType(extract_native_method(
            native_source(method, patched=patched), method, ns), owner))
    return owner


def test_native_accept_then_exhaust_is_not_a_rejection():
    req, owner = request(), adder()
    before = len(owner.can_run_list)
    result = owner.add_one_req(req, False, None)
    assert result is Result.NO_TOKEN  # raw 449 fits; page-rounded reservation is 512
    assert owner.can_run_list == [req]
    record_admission_outcome(req, owner, before, result)
    diag = req.contextkv_admission_diagnostic
    assert diag.outcomes == {"accepted:NO_TOKEN":1}
    assert "prefill_budget_rejected" not in diag.gates
    checked = diag.gates["prefill_budget_accepted"]["first"]
    assert checked["requested_total_tokens"] == 449 < checked["remaining_total"] == 450
    assert diag.gates["adder_accepted"]["last"]["remaining_total"] == -62


def test_native_capacity_failure_and_repeated_attempt_are_observations_not_durations():
    req, owner = request(), adder(capacity=449)
    for _ in range(100):
        before = len(owner.can_run_list)
        result = owner.add_one_req(req, False, None)
        record_admission_outcome(req, owner, before, result)
    assert not owner.can_run_list
    diag = req.contextkv_admission_diagnostic
    assert diag.outcomes == {"not_accepted:NO_TOKEN":100}
    assert len(diag.gates) == 2  # first/last summaries, never an unbounded attempt log
    row = diag.gates["prefill_budget_rejected"]
    assert row["count"] == 100 and row["first"]["observed_at"] <= row["last"]["observed_at"]
    assert row["last"]["requested_total_tokens"] == row["last"]["remaining_total"] == 449


@pytest.mark.parametrize("which", ["accepted_other", "request_limit", "host_load"])
def test_other_result_covers_different_native_outcomes(which):
    req, owner = request(n=128), adder(capacity=2048)
    if which == "accepted_other":
        owner.rem_input_tokens = 64
    elif which == "request_limit":
        owner.prefill_max_requests = 0
    else:
        req.host_hit_length = 64
        req.needs_host_load_back = lambda:True
        owner.tree_cache.init_load_back = lambda params:None
    result = owner.add_one_req(req, False, None)
    assert result is Result.OTHER
    record_admission_outcome(req, owner, 0, result)
    diag = req.contextkv_admission_diagnostic
    if which == "accepted_other":
        assert diag.outcomes == {"accepted:OTHER":1}
    else:
        assert diag.outcomes == {"not_accepted:OTHER":1}
        assert ("prefill_request_limit" if which=="request_limit" else "load_back_unavailable") in diag.gates


@pytest.mark.parametrize("capacity,n", [(449,257),(450,257),(4096,128)])
def test_original_and_patched_native_decisions_remain_equal(capacity,n):
    old, new = adder(capacity,patched=False), adder(capacity)
    oldreq,newreq = request(n,False),request(n,False)
    assert old.add_one_req(oldreq,False,None) is new.add_one_req(newreq,False,None)
    assert len(old.can_run_list) == len(new.can_run_list)
    assert (old.rem_total_tokens,old.cur_rem_tokens,old.rem_input_tokens)==(
        new.rem_total_tokens,new.cur_rem_tokens,new.rem_input_tokens)
    assert not hasattr(newreq,"contextkv_admission_diagnostic")


def test_disabled_hooks_do_not_read_budgets_or_clock(monkeypatch):
    monkeypatch.setattr("engine_patches.contextkv_runtime.time.perf_counter",
                        lambda:pytest.fail("off diagnostic read clock"))
    req = SimpleNamespace()
    record_admission_gate(req, object(), "unused")
    record_admission_outcome(req, object(), 0, object())
    record_scheduler_gate(SimpleNamespace(waiting_queue=[req]),object(),"unused")
    assert vars(req)=={}


def test_scheduler_gate_samples_only_current_head_not_whole_queue():
    head,behind=request(),request()
    scheduler=SimpleNamespace(waiting_queue=[head,behind],new_token_ratio_tracker=SimpleNamespace(current=.3))
    batch=SimpleNamespace(reqs=[object()],batch_is_full=True)
    record_scheduler_gate(scheduler,batch,"scheduler_batch_full")
    first=head.contextkv_admission_diagnostic.gates["scheduler_batch_full"]["first"]
    assert first["budget"] is None and first["scope"]=="scheduler_pass_head_observation"
    assert first["new_token_ratio"]==.3
    assert not behind.contextkv_admission_diagnostic.gates
    assert scheduler.waiting_queue==[head,behind]


@pytest.mark.parametrize("profiling,enabled,expected", [(False,False,False),(False,True,False),
                                                       (True,False,False),(True,True,True)])
def test_diagnostics_require_explicit_profiler_and_keep_cache_identity(monkeypatch,profiling,enabled,expected):
    monkeypatch.delenv("CONTEXTKV_ADMISSION_DIAGNOSTICS",raising=False)
    baseline=ContextKVRuntime(FakeNativeCache(),{})
    monkeypatch.setenv("CONTEXTKV_ENABLE_PROFILING",str(int(profiling)))
    monkeypatch.setenv("CONTEXTKV_ADMISSION_DIAGNOSTICS",str(int(enabled)))
    owner=ContextKVRuntime(FakeNativeCache(),{})
    assert owner.admission_diagnostics_enabled is expected
    assert owner.info({}).get("admission_diagnostics_enabled",False) is expected
    assert owner.cache_config == baseline.cache_config
    assert owner.cache_config_hash == baseline.cache_config_hash
    p=owner.prepare(dict(engine_session_id="s",operation_id="p",expected_restart_id=owner.restart_id))
    recv=SimpleNamespace(session_id="s",expected_restart_id=owner.restart_id,contextkv_fingerprint={},
        expected_generation=p["session_generation"],contextkv_attempt_id="r",rid="r",session_params=None,
        lora_id=None,mm_inputs=None,input_embeds=None,return_logprob=False,cache_salt=None,extra_key=None,input_ids=[1])
    req=SimpleNamespace()
    owner.admit(recv,req)
    assert hasattr(req,"contextkv_admission_diagnostic") is expected


def test_terminal_copies_original_native_clock_values_and_preserves_reentry_semantics(monkeypatch):
    req=request()
    req.time_stats=SimpleNamespace(scheduler_recv_time=10.,forward_entry_time=10.5,
        prefill_finished_time=11.,completion_time=13.,wait_queue_entry_time=12.,queue_duration_s=1.)
    monkeypatch.setattr("engine_patches.contextkv_runtime.time.perf_counter",
                        lambda:pytest.fail("terminal must copy original timestamps"))
    saved=req.contextkv_admission_diagnostic.finish(req)
    assert saved["timestamps"]=={name:getattr(req.time_stats,name) for name in saved["timestamps"]}
    assert saved["timestamps"]["wait_queue_entry_time"] > saved["timestamps"]["forward_entry_time"]
    assert saved["queue_duration_s"]==1. and "reentry" in saved["queue_duration_scope"]
    assert json.loads(json.dumps(saved))==saved


@pytest.mark.parametrize("enabled", [False, True])
def test_terminal_emits_diagnostic_once_only_for_enabled_guarded_requests(monkeypatch,enabled):
    monkeypatch.setenv("CONTEXTKV_ENABLE_PROFILING","1")
    monkeypatch.setenv("CONTEXTKV_ADMISSION_DIAGNOSTICS",str(int(enabled)))
    owner=ContextKVRuntime(FakeNativeCache(),{})
    p=owner.prepare(dict(engine_session_id="s",operation_id="p",expected_restart_id=owner.restart_id))
    recv=SimpleNamespace(session_id="s",expected_restart_id=owner.restart_id,contextkv_fingerprint={},
        expected_generation=p["session_generation"],contextkv_attempt_id="r",rid="r",session_params=None,
        lora_id=None,mm_inputs=None,input_embeds=None,return_logprob=False,cache_salt=None,extra_key=None,input_ids=[1])
    req=SimpleNamespace()
    owner.admit(recv,req)
    req.rid,req.session_id,req.session_generation="r","s",p["session_generation"]
    req.finished=lambda:True
    req.finished_reason=SimpleNamespace(to_json=lambda:{"type":"stop"})
    req.cached_tokens_device=req.cached_tokens_host=req.cached_tokens=0
    req.kv=SimpleNamespace(cache_inserted_len=1)
    req.customized_info=None
    req.output_ids=[2,3]
    req.time_stats=SimpleNamespace(scheduler_recv_time=1.,forward_entry_time=2.,
        prefill_finished_time=3.,completion_time=4.,wait_queue_entry_time=1.5,queue_duration_s=.5)
    owner.finish(req)
    owner.finish(req)
    terminal=[event for event in owner.events if event["event"]=="request_terminal"]
    assert len(terminal)==1
    assert ("admission_diagnostic" in terminal[0]) is enabled
    assert "admission_diagnostic" not in req.customized_info  # no new generated response schema
    assert terminal[0]["timing"]["server_elapsed_ms"]==3000
    if enabled:
        assert terminal[0]["admission_diagnostic"]["timestamps"]["completion_time"]==4.


def test_native_retry_wait_field_is_latest_and_queue_accumulates_without_moving_first_forward():
    native_forward=extract_native_method(native_source("set_forward_entry_time"),"set_forward_entry_time")
    stats=SimpleNamespace(forward_entry_time=0.,last_forward_entry_time=0.,wait_queue_entry_time=1.,
                          queue_duration_s=0.,enable_metrics=False,trace_ctx=SimpleNamespace(tracing_enable=False))
    native_forward(stats,ts=2.)
    # Native set_retract_time resets last_forward_entry, not first forward.
    stats.last_forward_entry_time=0.
    stats.wait_queue_entry_time=5.
    native_forward(stats,ts=7.)
    assert stats.forward_entry_time==2. and stats.last_forward_entry_time==7.
    assert stats.queue_duration_s==3.  # cumulative; cannot subtract it from TTFT


def test_actual_scheduler_calls_outcome_after_add_and_before_return_handling():
    source=textwrap.dedent(native_source("_get_new_batch_prefill_raw",patched=True))
    module=ast.parse(source)
    loop=next(n for n in ast.walk(module) if isinstance(n,ast.For) and ast.unparse(n.iter)=="self.waiting_queue")
    rendered=ast.unparse(loop)
    assert rendered.index("admitted_before = len(adder.can_run_list)") < rendered.index("res = adder.add_one_req")
    assert rendered.index("record_admission_outcome") < rendered.index("if res != AddReqResult.CONTINUE")
    assert "scheduler_pass_budget" in rendered and "scheduler_request_slots_after_admission" in rendered
