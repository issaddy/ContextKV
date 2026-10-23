"""Stage estimates preserve legacy behavior and expose analytical eligibility."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from controller.admission import WaitEstimate
from controller.config import CostSettings, RunConfig, create_controller, load_config
from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.planner import PlannerKnobs, choose_plan
from controller.schema import (
    CacheQuote,
    CacheQuoteBatch,
    ConfigFingerprint,
    CostEstimate,
    NativeRequestSummary,
    PromptCandidate,
    SessionSnapshot,
)
from controller.serialization import dumps, loads
from controller.templates import token_hash
from profiling.decision_pairs import _model_and_knobs

FP = ConfigFingerprint()


def candidate(n=100, action="REBUILD"):
    ids = tuple(range(n))
    return PromptCandidate(
        action, action, 0, (), ids, token_hash(ids), frozenset(), frozenset(), fingerprint=FP
    )


def quote(value, **kwargs):
    return CacheQuote(
        value.candidate_id,
        value.input_hash,
        len(value.input_ids),
        len(value.input_ids) - 1,
        **kwargs,
    )


def batch(quotes, **kwargs):
    return CacheQuoteBatch(
        "q",
        0,
        0,
        "restart",
        "cache",
        0,
        100,
        tuple(quotes),
        FP,
        evictable_gpu_tokens=500,
        gpu_capacity_tokens=500,
        native_running_requests=(),
        native_waiting_requests=(),
        **kwargs,
    )


def stage():
    cells = [
        {"features": [p, u, load, 1, budget, 500], "mean_ms": value, "requests": 3, "batches": 3}
        for p, u, load, value in (
            (0, 100, 0, 1.0),
            (0, 200, 0, 2.0),
            (50, 50, 0, 1.0),
            (50, 50, 50, 3.0),
        )
        for budget in (5, 10)
    ]
    return {
        "version": 1,
        "waiting_calibration_scope": "homogeneous_burst_only",
        "pre_forward": {"cells": cells, "online_heterogeneous_queue_calibrated": False},
        "development_output_lengths": {"lengths": [3, 4, 5, 10]},
    }


def model(*, staged=True, compatible=True, **kwargs):
    scope = "forward_first" if staged else "first_token"
    points = [
        LookupPoint(0, 100, 1, 10.0, error_ms=0.2, scope=scope),
        LookupPoint(0, 200, 1, 50.0, error_ms=0.3, scope=scope),
        LookupPoint(50, 50, 1, 10.0, error_ms=0.2, scope=scope),
        LookupPoint(50, 50, 1, 10.0, error_ms=0.2, host_load_tokens=50, scope=scope),
    ]
    result = CostModel(
        points,
        [DecodePoint(150, 1, 2.0, error_ms=0.05, context_start=0, context_end=1000)],
        default_output_tokens=3,
        p90_output_tokens=5,
        max_new_tokens=10,
        stage_model=stage() if staged else None,
        admission_estimator="native_workload" if staged else "strict",
        **kwargs,
    )
    result.calibration_metadata = {
        "profile_manifest": {"engine": {"fingerprint": asdict(FP), "cache_config_hash": "cache"}}
    }
    if compatible:
        result.check_calibration_scope({"fingerprint": asdict(FP), "cache_config_hash": "cache"})
    return result


def test_strict_ttft_behavior_ignores_new_workload_and_never_calls_wait(monkeypatch):
    def unused(**kwargs):
        raise AssertionError("strict total-TTFT model must not add analytical waiting")

    monkeypatch.setattr("controller.admission.estimate_wait", unused)
    c = candidate()
    q = quote(c)
    estimate = model(staged=False).predict_native(
        c, q, batch=batch([q]), control_ms=2, queue_ms=999
    )
    assert estimate.cost == 16 and estimate.risk_cost == 20 and estimate.queue_ms == 0
    assert estimate.calibrated and estimate.selection_eligible is None
    assert estimate.waiting_estimate_kind is None and estimate.waiting_upper_ms is None
    pressure = replace(
        q, native_estimate_uncertain=True, uncertainty_reasons=("capacity_pressure",)
    )
    assert not model(staged=False).predict_native(c, pressure, batch=batch([pressure])).calibrated


def test_explicit_mode_and_stage_scope_are_required_at_artifact_boundary():
    with pytest.raises(ValueError, match="explicit admission_estimator"):
        CostModel(stage_model=stage())
    with pytest.raises(ValueError, match="double-count"):
        CostModel(
            [LookupPoint(0, 100, 1, 10, scope="first_token")], admission_estimator="native_workload"
        )
    with pytest.raises(ValueError, match="forward_first points require"):
        CostModel([LookupPoint(0, 100, 1, 10, scope="forward_first")])
    with pytest.raises(ValueError, match="only forward_first"):
        CostModel(
            [LookupPoint(0, 100, 1, 10, scope="first_token")],
            stage_model=stage(),
            admission_estimator="native_workload",
        )


def test_wait_mean_is_added_once_and_p90_scenario_only_affects_risk(monkeypatch):
    def estimate(**kwargs):
        assert kwargs["idle_ms"] == 1
        assert kwargs["development_lengths"] == [3, 4, 5, 10]
        return WaitEstimate(7.0, 50.0, True, "p90_input_scenario")

    monkeypatch.setattr("controller.admission.estimate_wait", estimate)
    c = candidate()
    q = quote(c)
    value = model().predict_native(c, q, batch=batch([q]), control_ms=2, queue_ms=999)
    assert value.restore_prefill_ms == 10 and value.decode_ms == 4 and value.queue_ms == 7
    assert value.cost == 23 and value.risk_cost == 70
    assert value.error_ms == pytest.approx(0.3)  # waiting upper is not an empirical CI/error
    assert value.waiting_upper_ms == 50 and value.waiting_reason == "p90_input_scenario"
    assert value.waiting_estimate_kind == "analytical_native_workload"
    assert value.selection_eligible and not value.calibrated and value.timing_components_supported


def test_single_request_setup_preserves_candidate_host_restore_difference():
    c = candidate()
    gpu = quote(c, ready_gpu_prefix_tokens=50, recoverable_prefix_tokens=50)
    host = quote(c, recoverable_prefix_tokens=50, host_only_load_tokens=50, native_load_estimate=50)
    a = model().predict_native(c, gpu, batch=batch([gpu]))
    b = model().predict_native(c, host, batch=batch([host]))
    assert a.restore_prefill_ms == b.restore_prefill_ms == 10
    assert a.queue_ms == 1 and b.queue_ms == 3 and b.cost - a.cost == 2
    assert a.selection_eligible and b.selection_eligible
    no_host = model(include_host_cost=False).predict_native(c, host, batch=batch([host]))
    assert no_host.cost == a.cost


def test_known_capacity_pressure_is_eligible_but_unknown_native_state_is_not():
    c = candidate()
    m = model()
    q = quote(c, native_estimate_uncertain=True, uncertainty_reasons=("capacity_pressure",))
    value = m.predict_native(c, q, batch=batch([q]))
    assert value.selection_eligible and not value.calibrated
    for altered in (
        replace(q, uncertainty_reasons=()),
        replace(q, uncertainty_reasons=("unknown_future_restore",)),
        replace(q, inflight_h2d_tokens=1),
        replace(q, inflight_d2h_tokens=1),
    ):
        assert not m.predict_native(c, altered, batch=batch([altered])).selection_eligible


@pytest.mark.parametrize(
    "missing", ["batch", "capacity", "running", "waiting", "host_quote", "binding"]
)
def test_missing_observation_or_scope_falls_back_without_calibration_claim(missing):
    c = candidate()
    q = quote(c)
    b = batch([q])
    m = model(compatible=missing != "binding")
    if missing == "batch":
        b = None
    elif missing == "capacity":
        b = replace(b, gpu_capacity_tokens=None)
    elif missing == "running":
        b = replace(b, native_running_requests=None)
    elif missing == "waiting":
        b = replace(b, native_waiting_requests=None)
    elif missing == "host_quote":
        b = replace(b, host_quote_supported=False)
    value = m.predict_native(c, q, batch=b)
    assert not value.selection_eligible and not value.calibrated
    if missing in {"batch", "capacity", "running", "waiting"}:
        assert value.waiting_reason in {"missing_gpu_capacity", "missing_native_workload"}
    assert value.waiting_estimate_kind == "analytical_native_workload"


def test_planner_uses_eligible_stage_estimates_without_relabeling_them_calibrated():
    candidates = (candidate(200, "APPEND"), candidate(100, "REBUILD"))
    quotes = [
        quote(c, native_estimate_uncertain=True, uncertainty_reasons=("capacity_pressure",))
        for c in candidates
    ]
    snapshot = SessionSnapshot("s", turns_since_rebuild=10)
    knobs = PlannerKnobs(delta_rel=0, z=0, min_stable_turns=0, fallback_strategy="always_append")
    planned = choose_plan(candidates, batch(quotes), model(), snapshot, knobs)
    assert planned.reason == "cost_saving" and planned.candidate.action == "REBUILD"
    assert all(p.selection_eligible and not p.calibrated for p in planned.predicted)
    legacy = choose_plan(candidates, batch(quotes), model(staged=False), snapshot, knobs)
    assert legacy.reason == "uncalibrated_cost_fallback" and legacy.candidate.action == "APPEND"
    missing = choose_plan(
        candidates, replace(batch(quotes), native_running_requests=None), model(), snapshot, knobs
    )
    assert missing.reason == "uncalibrated_cost_fallback" and missing.candidate.action == "APPEND"


def test_stage_artifact_save_load_and_old_schema_defaults(tmp_path):
    original = model()
    path = tmp_path / "stage.json"
    original.save(path)
    restored = CostModel.load(path)
    assert (
        restored.stage_model == original.stage_model
        and restored.admission_estimator == "native_workload"
    )
    restored.check_calibration_scope({"fingerprint": asdict(FP), "cache_config_hash": "cache"})
    c = candidate()
    q = quote(c)
    b = batch([q])
    estimate = original.predict_native(c, q, batch=b)
    assert restored.predict_native(c, q, batch=b) == estimate
    assert loads(CostEstimate, dumps(estimate)) == estimate
    raw = asdict(estimate)
    for key in (
        "selection_eligible",
        "waiting_estimate_kind",
        "waiting_upper_ms",
        "waiting_reason",
        "timing_components_supported",
    ):
        raw.pop(key)
    assert loads(CostEstimate, json.dumps(raw)).selection_eligible is None
    with pytest.raises(ValueError, match="explicit admission_estimator"):
        CostModel.load(path, admission_estimator="strict")


@pytest.mark.asyncio
async def test_stage_configuration_assembles_online_and_offline_consistently(tmp_path):
    artifact = tmp_path / "fitter-stage.json"
    model().save(artifact)
    # The offline stage fitter preserves original settings. Explicit config is
    # what enables this newly supplied artifact, rather than its filename.
    payload = json.loads(artifact.read_text())
    payload["settings"].pop("admission_estimator")
    artifact.write_text(json.dumps(payload))
    config = RunConfig(
        cost_model=CostSettings(artifact=str(artifact), admission_estimator="native_workload"),
        state_dir=str(tmp_path / "state"),
    )
    config = replace(config, context=replace(config.context, max_new_tokens=10))
    file = tmp_path / "config.yaml"
    file.write_text(json.dumps(asdict(config)))
    loaded = load_config(file)
    serving = create_controller(loaded)
    try:
        offline, _ = _model_and_knobs(
            loaded, (SimpleNamespace(max_new_tokens=10, candidates=(candidate(),)),), str(artifact)
        )
        assert (
            offline.admission_estimator
            == serving.cost_model.admission_estimator
            == "native_workload"
        )
        assert offline.stage_model == serving.cost_model.stage_model == payload["stage_model"]
    finally:
        await serving.aclose()
        serving.store.close()


def test_default_offline_assembly_keeps_legacy_strict_mode():
    value, _ = _model_and_knobs(
        RunConfig(), (SimpleNamespace(max_new_tokens=10, candidates=(candidate(),)),), None
    )
    assert value.admission_estimator == "strict" and not value.stage_model


def test_native_quote_schema_roundtrips_workload_without_inventing_missing_lists():
    c = candidate()
    q = quote(c, uncertainty_reasons=("capacity_pressure",))
    current = replace(
        batch([q]), native_running_requests=(NativeRequestSummary(10, 2, 8, 12, 11, 5, False),)
    )
    assert loads(CacheQuoteBatch, dumps(current)) == current
    old = asdict(current)
    for name in ("gpu_capacity_tokens", "native_running_requests", "native_waiting_requests"):
        old.pop(name)
    legacy = loads(CacheQuoteBatch, json.dumps(old))
    assert legacy.native_running_requests is None and legacy.native_waiting_requests is None


def test_native_budget_mode_is_optional_and_roundtrips_with_bound_page_size(tmp_path):
    original=model()
    original.calibration_metadata["profile_manifest"]["cache_config"]={"page_size":4}
    path=tmp_path/"stage.json"
    original.save(path)
    old=CostModel.load(path)
    changed=CostModel.load(path,admission_estimator="native_budget")
    assert old.admission_estimator == "native_workload"
    changed.check_calibration_scope({"fingerprint":asdict(FP),"cache_config_hash":"cache"})
    c=candidate()
    q=quote(c)
    snapshot=replace(batch([q]),native_running_budget_requests=(),native_running_budget_tokens=0.,
                     native_new_token_ratio=.7,native_max_new_tokens_clip=4096)
    value=changed.predict_native(c,q,batch=snapshot)
    assert value.selection_eligible and not value.calibrated
    assert value.waiting_estimate_kind == "analytical_native_budget"
    changed.save(path)
    assert CostModel.load(path).admission_estimator == "native_budget"
    cfg=RunConfig(cost_model=CostSettings(artifact=str(path),admission_estimator="native_budget"))
    assembled,_=_model_and_knobs(cfg,(SimpleNamespace(max_new_tokens=10,candidates=(c,)),),None)
    assert assembled.admission_estimator == "native_budget"
    assert loads(CostEstimate,dumps(value)) == value


def test_native_budget_missing_identity_and_page_do_not_become_eligible():
    value=model()
    value.admission_estimator="native_budget"
    c=candidate()
    q=quote(c)
    snap=replace(batch([q]),native_running_budget_requests=(),native_running_budget_tokens=0.,
                 native_new_token_ratio=.7,native_max_new_tokens_clip=4096)
    missing_page=value.predict_native(c,q,batch=snap)
    assert not missing_page.selection_eligible and missing_page.waiting_reason == "missing_native_page_size"
    value.calibration_metadata["profile_manifest"]["cache_config"]={"page_size":4}
    old_request=NativeRequestSummary(10,2,8,12,11,5)
    missing_identity=value.predict_native(c,q,batch=replace(snap,native_running_requests=(old_request,)))
    assert not missing_identity.selection_eligible
    assert missing_identity.waiting_reason == "missing_native_request_identity"


def test_queued_phase_uses_host_specific_idle_setup_without_total_ttft():
    value=model()
    # Same forward-first cost 10, but empiric idle is 1 for GPU and 3 for host.
    assert value._queued_workload_forward(100,50,0,1) == (11.,True)
    assert value._queued_workload_forward(100,0,50,1) == (13.,True)
