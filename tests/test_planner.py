from dataclasses import replace

import pytest

from controller.cost_model import CostModel, DecodePoint, LookupPoint, output_length_prior
from controller.planner import PlannerKnobs, choose_plan
from controller.schema import (CacheQuote, CacheQuoteBatch, CostEstimate, NoFeasiblePlan,
                               PromptCandidate, SessionSnapshot, TurnRecord)
from controller.templates import token_hash


def candidate(action,n):
    ids = tuple(range(n))
    return PromptCandidate(action,action,0,(),ids,token_hash(ids),frozenset(),frozenset())


def batch(*candidates):
    return CacheQuoteBatch("q",1,123,"restart","config",100000,100000,
        tuple(CacheQuote(c.candidate_id,c.input_hash,len(c.input_ids),len(c.input_ids)-1)
              for c in candidates))


class MeasuredModel(CostModel):
    def __init__(self,a=100,r=50,error=0,calibrated=True):
        super().__init__()
        self.values,self.error,self.calibrated={"APPEND":a,"REBUILD":r},error,calibrated

    def predict_native(self,candidate,quote,*args,**kwargs):
        return CostEstimate(candidate.candidate_id,self.values[candidate.action],
                            error_ms=self.error,calibrated=self.calibrated)


def test_cost_saving_requires_absolute_relative_and_error_margin():
    a,r = candidate("APPEND",1000),candidate("REBUILD",500)
    snap = SessionSnapshot("s",turns_since_rebuild=2)
    model = MeasuredModel(a=100,r=90,error=6)
    assert choose_plan((a,r),batch(a,r),model,snap).candidate == a
    model = MeasuredModel(a=100,r=50)
    assert choose_plan((a,r),batch(a,r),model,snap).candidate == r
    assert choose_plan((a,r),batch(a,r),model,snap,PlannerKnobs(delta_abs=51)).candidate == a
    assert choose_plan((a,r),batch(a,r),model,snap,PlannerKnobs(delta_rel=.51)).candidate == a


def test_minimum_stable_turns_and_forced_rebuild_precedence():
    a,r = candidate("APPEND",1000),candidate("REBUILD",500)
    snap = SessionSnapshot("s",turns_since_rebuild=0)
    assert choose_plan((a,r),batch(a,r),MeasuredModel(),snap).candidate == a
    assert choose_plan((a,r),batch(a,r),MeasuredModel(),replace(snap,must_rebuild=True)).candidate == r
    invalid = replace(a,feasible=False,reason="invalid_visible_version")
    assert choose_plan((invalid,r),None,MeasuredModel(),snap).candidate == r


def test_uncalibrated_and_quote_failure_use_common_fixed_threshold():
    a,r = candidate("APPEND",1000),candidate("REBUILD",500)
    snap = SessionSnapshot("s",turns_since_rebuild=5)
    knobs = PlannerKnobs(fixed_token_threshold=800)
    plan = choose_plan((a,r),batch(a,r),CostModel(),snap,knobs)
    assert plan.candidate == r and plan.reason == "uncalibrated_cost_fallback"
    assert choose_plan((a,r),None,CostModel(),snap,knobs).candidate == r


@pytest.mark.parametrize("strategy,expected",[("always_append","APPEND"),
    ("stable_layout","APPEND"),("always_rebuild","REBUILD"),("normal_rag","REBUILD")])
def test_fixed_baselines_share_hard_constraints(strategy,expected):
    a,r = candidate("APPEND",100),candidate("REBUILD",50)
    plan = choose_plan((a,r),None,CostModel(),SessionSnapshot("s"),PlannerKnobs(strategy=strategy))
    assert plan.candidate.action == expected
    invalid = replace(a,feasible=False)
    assert choose_plan((invalid,r),None,CostModel(),SessionSnapshot("s"),
                       PlannerKnobs(strategy=strategy)).candidate == r


def test_no_valid_candidate_and_stale_snapshot_fail_explicitly():
    a,r = candidate("APPEND",100),candidate("REBUILD",50)
    assert isinstance(choose_plan((replace(a,feasible=False),replace(r,feasible=False)),
        None,CostModel(),SessionSnapshot("s")),NoFeasiblePlan)
    with pytest.raises(ValueError,match="snapshot"):
        choose_plan((a,r),None,CostModel(),SessionSnapshot("s",version=1))


def test_output_prior_uses_only_past_four_completed_outputs_with_shrinkage():
    history = tuple(TurnRecord(i,"q","a",tuple(range(n)))
                    for i,n in enumerate((999,10,20,30,40)))
    prior = output_length_prior(SessionSnapshot("s",raw_history=history),
                                global_mean=100,global_p90=120,max_new_tokens=512)
    assert prior.mean_tokens == 50
    assert prior.p90_tokens == 120


def test_decode_excludes_first_token_and_joint_cost_is_not_double_counted():
    c = candidate("REBUILD",100)
    model = CostModel((LookupPoint(0,100,1,10),),
                     (DecodePoint(100,1,2),DecodePoint(110,1,2)),
                     default_output_tokens=4,p90_output_tokens=4,max_new_tokens=4)
    estimate = model.predict_native(c,batch(c).quotes[0])
    assert estimate.restore_prefill_ms == 10
    assert estimate.decode_ms == 6
    assert estimate.cost == 16 and estimate.calibrated


def test_host_unknown_or_pending_never_becomes_ready_or_calibrated():
    c = candidate("REBUILD",100)
    quote = replace(batch(c).quotes[0],ready_gpu_prefix_tokens=20,
        recoverable_prefix_tokens=80,host_only_load_tokens=60,native_load_estimate=60,
        inflight_h2d_tokens=15,native_estimate_uncertain=True)
    model = CostModel((LookupPoint(80,20,1,5,host_load_tokens=60),),
        (DecodePoint(100,1,2),DecodePoint(110,1,2)),default_output_tokens=2,p90_output_tokens=2)
    assert model.predict_native(c,quote).restore_prefill_ms == 5
    assert not model.predict_native(c,quote).calibrated
    unknown = replace(batch(c),host_quote_supported=False)
    assert not model.predict_native(c,replace(quote,native_estimate_uncertain=False),batch=unknown).calibrated


def test_mock_profiles_are_not_accepted_as_hardware_calibration():
    rows = [{"phase":"prefill","prompt_tokens":100,"prefix_tokens":0,
             "concurrency":1,"elapsed_ms":2,"source":"mock"}]
    assert CostModel.from_measurements(rows).prefill_points == ()
    assert len(CostModel.from_measurements(rows,allow_mock=True).prefill_points) == 1
    unproven = {key:value for key,value in rows[0].items() if key != "source"}
    assert CostModel.from_measurements((unproven,),allow_mock=True).prefill_points == ()


def test_input_hash_binding_prevents_quote_for_other_input():
    c = candidate("APPEND",100)
    quotes = batch(c)
    with pytest.raises(ValueError,match="candidate input"):
        choose_plan((c,),replace(quotes,quotes=(replace(quotes.quotes[0],input_hash="wrong"),)),
                    CostModel(),SessionSnapshot("s"))


def test_capacity_risk_does_not_credit_retired_branch_as_freed():
    c = candidate("REBUILD",100)
    small = replace(batch(c),free_gpu_tokens=10,evictable_gpu_tokens=0)
    assert CostModel().predict_native(c,small.quotes[0],batch=small).capacity_risk


def test_host_gap_restoration_preserves_gpu_ready_suffix():
    c = candidate("REBUILD", 100)
    # GPU [0,20), host-only [20,40), GPU [40,80): restore 20 tokens to
    # reach 80, rather than incorrectly treating that boundary as 20+20.
    quote = replace(batch(c).quotes[0], ready_gpu_prefix_tokens=20,
                    recoverable_prefix_tokens=80, host_only_load_tokens=20, native_load_estimate=20)
    model = CostModel((LookupPoint(80,20,1,5,host_load_tokens=20),),
                      default_output_tokens=1,p90_output_tokens=1,max_new_tokens=1)
    prediction = model.predict_native(c,quote)
    assert prediction.calibrated and prediction.restore_prefill_ms == 5
    assert not model.predict_native(c,replace(quote,native_load_estimate=10)).calibrated


def test_fixed_period_rebuilds_every_four_turns_after_last_rebuild():
    a,r = candidate("APPEND",100),candidate("REBUILD",50)
    knobs = PlannerKnobs(strategy="fixed_period",period=4)
    for elapsed in (0,1,2):
        snap = SessionSnapshot("s",turns_since_rebuild=elapsed)
        assert choose_plan((a,r),None,CostModel(),snap,knobs).candidate == a
    snap = SessionSnapshot("s",turns_since_rebuild=3)
    assert choose_plan((a,r),None,CostModel(),snap,knobs).candidate == r


def test_warm_idle_uses_real_partial_ready_prefix_threshold():
    a,r = candidate("APPEND",101),candidate("REBUILD",50)
    quotes = batch(a,r)
    knobs = PlannerKnobs(strategy="warm_idle",warm_ratio=.5)
    snap = SessionSnapshot("s")
    for ready,expected in ((49,r),(50,a),(75,a)):
        observed = replace(quotes,quotes=(replace(quotes.quotes[0],
            ready_gpu_prefix_tokens=ready,recoverable_prefix_tokens=100,
            host_only_load_tokens=100-ready,inflight_h2d_tokens=100-ready),quotes.quotes[1]))
        plan = choose_plan((a,r),observed,CostModel(),snap,knobs)
        assert plan.candidate == expected
    missing = choose_plan((a,r),None,CostModel(),snap,knobs)
    assert missing.reason == "warm_quote_unavailable_fallback"


def test_lookup_total_input_range_is_enforced_without_optional_fitting_dependencies():
    points = [LookupPoint(0,2048,1,3), LookupPoint(1024,1024,1,2),
              LookupPoint(4096,4096,1,5), LookupPoint(7360,832,1,2), LookupPoint(0,8192,1,8)]
    model = CostModel(points)
    assert not model._prefill(6000,4000,0,1)[2]
    assert model._prefill(4096,4096,0,1)[2]
