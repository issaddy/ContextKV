"""Analytical workload scenarios; these are not native scheduler equivalence tests."""
from dataclasses import FrozenInstanceError

import pytest

from controller.admission import estimate_wait
from controller.schema import CacheQuoteBatch, NativeRequestSummary


def req(n=50, generated=10, budget=30, allocated=60, committed=59, protected=20, fixed=False):
    return NativeRequestSummary(n,generated,budget,allocated,committed,protected,fixed)


def batch(*, capacity=200, free=0, evictable=0, running=(), waiting=()):
    return CacheQuoteBatch("q",0,0,"r","c",free,0,(),gpu_capacity_tokens=capacity,
        evictable_gpu_tokens=evictable,native_running_requests=running,native_waiting_requests=waiting)


def estimate(snapshot, *, n=30, resident=0, mean=10, p90=20, lengths=(20,30), forward=None, decode=None):
    return estimate_wait(n,resident,mean,p90,snapshot,
        forward or (lambda n,p,c:(float(n-p),True)),
        decode or (lambda context,steps,c:(float(steps),True)),lengths,1.0)


def test_empty_or_evictable_pool_can_admit_without_any_wait_service_timing():
    def unused(*args):
        raise AssertionError("candidate forward is excluded from wait")
    result = estimate(batch(free=200),forward=unused,decode=unused)
    assert result.mean_ms == result.upper_ms == 1 and result.supported
    result = estimate(batch(free=0,evictable=200),forward=unused,decode=unused)
    assert result.mean_ms == 1 and result.model_kind == "analytical_native_workload"
    with pytest.raises(FrozenInstanceError):
        result.mean_ms = 2


@pytest.mark.parametrize("snapshot,reason", [
    (None,"missing_native_workload"),
    (batch(capacity=None),"missing_gpu_capacity"),
    (batch(running=None),"missing_native_workload"),
    (batch(waiting=None),"missing_native_workload"),
    (batch(),"unattributed_locked_capacity"),
])
def test_unknown_observations_do_not_become_zero_wait_support(snapshot,reason):
    result = estimate(snapshot)
    assert not result.supported and result.reason == reason


def test_running_conditioning_uses_only_empirical_survivors_and_known_budget():
    calls = []
    def decode(context,steps,c):
        calls.append((context,steps,c))
        return float(steps),True
    # L=5 is impossible once 10 tokens have been observed. Survivors have
    # residuals 10 and 20, so scenarios are 15 and 20 future decode steps.
    result = estimate(batch(running=(req(),)),lengths=(5,20,30),decode=decode)
    assert (result.mean_ms,result.upper_ms) == (16,21)
    assert calls == [(60,15,1),(60,20,1)]
    assert result.supported and "logical_release_proxy" in result.reason


@pytest.mark.parametrize("lengths,fixed,expected,flag", [
    ((5,8),False,20,"development_tail_budget_fallback"),
    ((),False,20,"missing_development_lengths_budget_fallback"),
    ((11,),True,20,None),
    ((200,),False,20,None),  # empirical observations are capped to this budget
])
def test_tail_and_ignore_eos_use_remaining_declared_budget(lengths,fixed,expected,flag):
    result = estimate(batch(running=(req(fixed=fixed),)),lengths=lengths)
    assert result.mean_ms == result.upper_ms == 1+expected
    assert result.supported
    if flag:
        assert flag in result.reason


def test_normalized_locked_pool_is_released_in_proportion_not_per_request_sum():
    # Logical allocations sum to 600 because prefixes can be shared. Only 100
    # physical locked tokens exist; the first completion releases a 25-token
    # proxy, insufficient for candidate need 40, so wait for the second.
    a = req(generated=10,budget=15,allocated=150,fixed=True)
    b = req(generated=10,budget=30,allocated=450,fixed=True)
    result = estimate(batch(capacity=100,running=(a,b)),mean=10,p90=10)
    assert result.mean_ms == result.upper_ms == 21


def test_resident_prefix_is_not_counted_twice_with_global_evictable_slots():
    snapshot = batch(capacity=100,evictable=50,running=(req(fixed=True),))
    # Without the min(resident,evictable) correction this would need 10 slots
    # and incorrectly start immediately, despite reserving its 50-token prefix.
    result = estimate(snapshot,n=50,resident=50,mean=10,p90=10)
    assert result.mean_ms == result.upper_ms == 21


def test_waiting_cold_prefill_produces_next_token_including_retracted_requests():
    calls = []
    def forward(n,p,c):
        calls.append(("forward",n,p,c))
        return 10.0,True
    def decode(context,steps,c):
        calls.append(("decode",context,steps,c))
        return float(steps),True
    pending = req(n=50,generated=2,budget=7,allocated=0,committed=0,fixed=True)
    result = estimate(batch(capacity=100,free=100,waiting=(pending,)),n=50,mean=10,p90=10,
                      forward=forward,decode=decode)
    assert result.mean_ms == result.upper_ms == 15
    assert calls == [("forward",52,0,1),("decode",53,4,1)]*2
    assert result.supported and "fifo_waiting_cold_prefix" in result.reason


def test_running_without_first_token_uses_committed_prefix_and_decode_minus_one():
    calls = []
    def forward(n,p,c):
        calls.append((n,p,c))
        return 10.0,True
    active = req(n=50,generated=0,budget=5,committed=20,fixed=True)
    result = estimate(batch(running=(active,)),forward=forward)
    assert (result.mean_ms,result.upper_ms) == (15,15)
    assert calls == [(50,20,1)]*2


def test_simultaneous_completions_are_removed_before_next_admission():
    concurrency = []
    def forward(n,p,c):
        concurrency.append(c)
        return 10.0,True
    active = req(budget=15,fixed=True)
    queued = req(n=80,generated=0,budget=5,allocated=0,fixed=True)
    result = estimate(batch(capacity=100,running=(active,active),waiting=(queued,)),
                      n=20,mean=5,p90=5,forward=forward)
    assert result.mean_ms == result.upper_ms == 20  # 5 + 10 + 4 + idle
    assert concurrency == [1,1]


def test_timing_support_and_capacity_failures_are_explicit():
    assert estimate(batch(running=(req(),)),decode=lambda *args:(3,False)).reason == "unsupported_running_timing"
    assert estimate(batch(capacity=100,free=100,waiting=(req(),)),forward=lambda *args:(3,False)).reason == "unsupported_waiting_timing"
    assert estimate(batch(free=200),n=200).reason == "candidate_reservation_exceeds_capacity"
    assert estimate(batch(free=200,waiting=(req(n=201),))).reason == "queued_reservation_exceeds_capacity"


def test_upper_scenario_is_at_least_mean_even_for_nonmonotone_callback():
    result = estimate(batch(running=(req(),)),decode=lambda context,steps,c:(100-steps,True))
    assert result.mean_ms == result.upper_ms == 86


def test_running_future_headroom_can_prevent_immediate_admission():
    snapshot = batch(capacity=100,free=50,running=(req(generated=10,budget=50,fixed=True),))
    result = estimate(snapshot,mean=10,p90=10)
    assert result.mean_ms == result.upper_ms == 41  # 50 free minus 40 headroom < 40 need


def test_immediate_admission_does_not_require_unrelated_service_timing():
    def unused(*args):
        raise AssertionError("no completion is needed")
    snapshot = batch(free=150,running=(req(),))
    result = estimate(snapshot,forward=unused,decode=unused)
    assert result.mean_ms == result.upper_ms == 1 and result.supported


def test_zero_time_completion_does_not_inflate_virtual_concurrency():
    calls = []
    decode_calls = []
    def forward(n,p,c):
        calls.append(c)
        return float(n-p),True
    def decode(context,steps,c):
        decode_calls.append(c)
        return float(steps),True
    finished_budget = req(generated=10,budget=10,fixed=True)
    active = req(generated=10,budget=20,fixed=True)
    queued = req(n=20,generated=0,budget=5,allocated=0,fixed=True)
    snapshot = batch(capacity=100,free=60,running=(finished_budget,active),waiting=(queued,))
    result = estimate(snapshot,n=60,mean=10,p90=10,forward=forward,decode=decode)
    assert result.mean_ms == result.upper_ms == 11
    assert calls == [2,2]
    assert decode_calls == [1,2,1,2]
