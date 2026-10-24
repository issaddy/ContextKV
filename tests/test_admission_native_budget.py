"""Native FULL gate differentials and explicitly approximate waiting scenarios."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from controller.admission import estimate_native_budget_wait
from controller.schema import CacheQuoteBatch, NativePrefixMatch, NativeRequestSummary
from test_engine_host_quote import extract_native_class, native_source


def req(rid="a", n=50, g=10, maximum=30, allocated=60, *, fixed=False, done=False, match=None):
    return NativeRequestSummary(n,g,maximum,allocated,max(0,allocated-1),20,
        fixed,done,match,rid)


def batch(*, capacity=200, free=0, evictable=0, running=(), waiting=(), budget=None, ratio=.5, clip=4096):
    members=running if budget is None else budget
    offset=sum(min(r.max_new_tokens-r.generated_tokens,clip)*(1 if r.ignore_eos else ratio) for r in members)
    return CacheQuoteBatch("q",0,10_000,"r","c",free,0,(),gpu_capacity_tokens=capacity,
        evictable_gpu_tokens=evictable,native_running_requests=running,native_waiting_requests=waiting,
        native_running_budget_requests=members,native_running_budget_tokens=offset,
        native_new_token_ratio=ratio,native_max_new_tokens_clip=clip)


def estimate(snapshot, *, n=30, resident=0, maximum=10, page=4, lengths=(20,30), forward=None, queued=None, decode=None, pin=None):
    return estimate_native_budget_wait(n,resident,maximum,snapshot,
        forward or (lambda n,p,c:(float(n-p),True)),
        decode or (lambda context,steps,c:(float(steps),True)),
        queued or (lambda n,p,h,c:(float(n-p-h)+h*.1,True)),lengths,1.0,page,
        initial_pinned_tokens=pin)


def test_shared_prefix_observation_corrects_false_wait_against_native_full_gate():
    # X/Y are equal 4096-token resident branches. The running peer locks X
    # (shared) or Y (unrelated); the old scalar quote is identical in both.
    active=req("peer",n=6144,g=1,maximum=512,allocated=6208,fixed=True)
    snapshot=batch(capacity=16384,free=6080,evictable=4096,running=(active,))
    native=extract_native_class(native_source("PrefillBudget"),"PrefillBudget")
    legacy=estimate(snapshot,n=12288,resident=4096,maximum=512,page=64)
    assert legacy.mean_ms > 1
    for pin,accepted in ((0,True),(4096,False)):
        original=native(SimpleNamespace(page_size=64,available_size=lambda:6080),
            SimpleNamespace(supports_mamba=lambda:False,evictable_size=lambda:4096-pin))
        original.total_offset=snapshot.native_running_budget_tokens
        gate,_=original.check_prefill(extend_input_len=8192,total_tokens=8768,
            max_new_tokens=512,input_tokens=8192,swa_host_hit_length=0,chunk_limit=None)
        allocation=original.can_allocate_prefill(paged_input=8192,extend_input_len=8192,
            max_new_tokens=512,chunk_limit=None)
        revised=estimate(snapshot,n=12288,resident=4096,maximum=512,page=64,pin=pin)
        assert bool(gate and allocation) == accepted == (revised.mean_ms == 1)
        assert revised.supported
        assert "observed_initial_gpu_pin" in revised.reason
    assert estimate(snapshot,n=12288,resident=4096,maximum=512,page=64,pin=4096).mean_ms == legacy.mean_ms


def test_initial_pin_is_not_reused_after_virtual_queue_admission_or_completion():
    active=req("peer",n=100,g=1,maximum=20,allocated=100,fixed=True)
    queued=req("queued",n=20,g=0,maximum=20,allocated=0,fixed=True)
    snapshot=batch(capacity=200,free=80,evictable=20,running=(active,),waiting=(queued,))
    assert estimate(snapshot,n=70,resident=20,pin=0) == estimate(snapshot,n=70,resident=20)
    # Even observed zero pin does not grant the candidate newly-unlocked slots
    # twice after a completion; the legacy release proxy remains unchanged.
    snapshot=replace(snapshot,native_waiting_requests=())
    old=estimate(snapshot,n=130,resident=20,maximum=20)
    revised=estimate(snapshot,n=130,resident=20,maximum=20,pin=0)
    assert old.mean_ms > 1 and revised.mean_ms == old.mean_ms


def test_full_native_gate_matches_original_source_for_strict_boundary_and_pages():
    native=extract_native_class(native_source("PrefillBudget"),"PrefillBudget")
    for page in (4,64):
        for capacity in (32,64,256):
            for n in (1,28,60,63,64,70):
                for maximum in (0,8,64):
                    for carry in ((),(req("done",g=0,maximum=25,allocated=0,done=True),)):
                        snapshot=batch(capacity=capacity,free=capacity,budget=carry,clip=64)
                        original=native(SimpleNamespace(page_size=page,available_size=lambda:capacity),
                            SimpleNamespace(supports_mamba=lambda:False,evictable_size=lambda:0))
                        original.total_offset=snapshot.native_running_budget_tokens
                        gate,_=original.check_prefill(extend_input_len=n,total_tokens=n+min(maximum,64)+page,
                            max_new_tokens=maximum,input_tokens=original.ceil_paged_tokens(n),
                            swa_host_hit_length=0,chunk_limit=None)
                        allocation=original.can_allocate_prefill(paged_input=original.ceil_paged_tokens(n),
                            extend_input_len=n,max_new_tokens=maximum,chunk_limit=None)
                        observed=estimate(snapshot,n=n,maximum=maximum,page=page)
                        assert observed.supported == (gate and allocation)
                        if observed.supported:
                            assert observed.mean_ms == 1


def test_declared_capacity_and_empirical_service_are_distinct():
    active=req(n=110,g=10,maximum=50,allocated=120)
    snap=batch(free=80,running=(active,))  # total=60, candidate declared gate=84
    short=estimate(snap,maximum=50,lengths=(20,))
    long=estimate(snap,maximum=50,lengths=(50,))
    assert short.mean_ms == 11 and long.mean_ms == 41
    # Changing service lengths cannot make the same initial gate fit.
    assert short.supported and long.supported
    assert short.model_kind == "analytical_native_budget"


def test_ratio_and_ignore_eos_affect_headroom_not_service_residual():
    active=req(g=10,maximum=50,allocated=100)
    low=batch(capacity=200,free=100,running=(active,),ratio=.1)
    high=replace(low,native_new_token_ratio=1,native_running_budget_tokens=40)
    assert estimate(low,n=60,maximum=20,lengths=(20,)).mean_ms == 1
    assert estimate(high,n=60,maximum=20,lengths=(20,)).mean_ms == 11
    fixed=batch(capacity=200,free=100,running=(replace(active,ignore_eos=True),),ratio=.1)
    assert estimate(fixed,n=60,maximum=20,lengths=(20,)).mean_ms == 41


def test_budget_releases_follow_request_id_across_different_member_order():
    slow=req("slow",g=10,maximum=50,allocated=50,fixed=True)
    fast=req("fast",g=10,maximum=12,allocated=50,fixed=True)
    snap=batch(capacity=200,free=50,running=(fast,slow),budget=(slow,fast))
    result=estimate(snap,n=90,maximum=20)
    assert result.supported and result.mean_ms == 41


def test_finished_and_retracted_budget_without_service_completion_stays_held():
    carry=req("early-eos",g=3,maximum=10,allocated=0,done=True)
    snap=batch(capacity=100,free=100,budget=(carry,))
    assert estimate(snap,n=90,maximum=0).supported
    blocked=estimate(snap,n=93,maximum=0)
    assert not blocked.supported and blocked.reason == "unattributed_native_capacity_or_budget"
    assert "unmatched_native_budget_held" in estimate(snap,n=90,maximum=0).reason


def test_overlap_service_member_does_not_invent_a_native_budget_offset():
    overlap=req("overlap",maximum=20,fixed=True)
    snap=batch(capacity=100,free=20,running=(overlap,),budget=())
    result=estimate(snap,n=40)
    assert result.mean_ms == 11 and result.supported
    assert "service_member_outside_budget_batch" in result.reason


def test_retracted_queue_gate_and_final_prefill_commit_have_different_headroom():
    pending=req("retracted",n=35,g=5,maximum=20,allocated=0,fixed=True)
    # Gate:40+15+4=59. Full-request commit:40+20+4=64. Remaining=36,
    # candidate gate=36 fails the strict native check until this job completes.
    result=estimate(batch(capacity=100,free=100,waiting=(pending,)),n=31,maximum=1,
                    queued=lambda *args:(10.,True))
    assert result.mean_ms == result.upper_ms == 25


def test_saved_host_match_is_service_scenario_only_not_capacity_or_freshness():
    calls=[]
    def queued(n,p,h,c):
        calls.append((n,p,h,c))
        return float(n-p-h)+h*.1,True
    match=NativePrefixMatch(1,60,20,30,50)  # deliberately old, still labelled historical
    pending=req("queued",n=60,g=0,maximum=10,allocated=0,match=match)
    result=estimate(batch(capacity=120,free=120,waiting=(pending,)),n=50,lengths=(5,),queued=queued)
    assert (result.mean_ms,result.upper_ms) == (18,65)
    assert calls == [(60,20,30,1),(60,0,0,1)]
    assert "last_match_service_scenario_not_residency" in result.reason
    # Both scenarios must wait: old device hits never reduce the cold gate.
    assert "fifo_waiting_cold_capacity" in result.reason
    mismatch=replace(pending,last_native_match=replace(match,input_tokens=59))
    assert estimate(batch(capacity=120,free=120,waiting=(mismatch,)),n=50,lengths=(5,)).mean_ms == 65


def test_shared_physical_release_is_normalized_once_and_simultaneous_events_clear():
    a=req("a",g=10,maximum=15,allocated=150,fixed=True)
    b=req("b",g=10,maximum=30,allocated=450,fixed=True)
    result=estimate(batch(capacity=100,running=(a,b)),n=30,maximum=10)
    assert result.mean_ms == 21  # first 25-slot proxy cannot fit 44-token gate
    equal=replace(b,max_new_tokens=15)
    result=estimate(batch(capacity=100,running=(a,equal)),n=30,maximum=10)
    assert result.mean_ms == 6


def test_completion_cannot_double_credit_candidate_previously_locked_prefix():
    active=req("shared",n=90,g=10,maximum=10,allocated=100,fixed=True)
    result=estimate(batch(capacity=100,running=(active,)),n=98,resident=80,maximum=5)
    # After release, treating 80 resident tokens as free as well would make a
    # 27-token gate fit. Pinning the potentially unlocked prefix leaves only 20.
    assert not result.supported and result.reason == "unattributed_native_capacity_or_budget"


@pytest.mark.parametrize("change,reason",[
    ({"native_running_budget_tokens":None},"missing_native_budget"),
    ({"native_running_requests":None},"missing_native_workload"),
    ({"gpu_capacity_tokens":None},"missing_gpu_capacity"),
    ({"native_running_budget_requests":(req(None),)},"missing_native_request_identity"),
])
def test_missing_contract_is_explicit_without_legacy_defaults(change,reason):
    result=estimate(replace(batch(free=200),**change))
    assert not result.supported and result.reason == reason


def test_missing_page_and_batch_or_required_timing_are_unsupported():
    assert estimate(None).reason == "missing_native_workload"
    assert estimate(batch(),page=None).reason == "missing_native_page_size"
    assert estimate(batch(running=(req(),)),decode=lambda *args:(1.,False)).reason == "unsupported_running_timing"


def test_immediate_admission_does_not_require_unrelated_timing():
    def unused(*args):
        raise AssertionError("no completion is needed")
    result=estimate(batch(free=150,running=(req(),)),forward=unused,decode=unused)
    assert result.supported and result.mean_ms == result.upper_ms == 1
    pending=req("queued",n=20,g=0,maximum=10,allocated=0)
    result=estimate(batch(free=150,running=(req(),),waiting=(pending,)),
                    forward=unused,queued=unused,decode=unused)
    assert result.supported and result.mean_ms == result.upper_ms == 1
