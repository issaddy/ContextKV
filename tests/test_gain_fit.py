from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.engine_client import EngineInfo
from controller.planner import PlannerKnobs
from controller.schema import CacheQuote, CacheQuoteBatch, ConfigFingerprint, PromptCandidate, SessionSnapshot
from controller.templates import token_hash
from profiling.decision_pairs import DecisionCase
from profiling.gain_fit import (FEATURE_NAMES, FAMILIES, NestedFeatures, cells, decide,
                                fit_exploration, fit_gain, fit_outer, one_sided_threshold, predict_gain,
                                prediction_features, prepared_pairs)
from profiling.length_fit import build_cases


def inputs():
    cases, rows = [], []
    fp = ConfigFingerprint()
    info = EngineInfo("restart", cache_config_hash="cache", fingerprint=fp)
    for i in range(5):
        candidates = tuple(PromptCandidate(f"c{i}:{a}", a, 0, (), (i,)*n,
            token_hash((i,)*n), frozenset(), frozenset(), fingerprint=fp)
            for a, n in (("APPEND", 100+i*5), ("REBUILD", 80+i*3)))
        case = DecisionCase(f"c{i}", f"s{i}", f"r{i}",
            SessionSnapshot(f"s{i}", turns_since_rebuild=i % 3), candidates, (), 64, "APPEND")
        cases.append(case)
        quote = CacheQuoteBatch("q", 1, 1, "restart", "cache", 4096, 0,
            tuple(CacheQuote(c.candidate_id, c.input_hash, len(c.input_ids), len(c.input_ids)-1) for c in candidates),
            fingerprint=fp)
        for state in ("cold", "gpu_ready", "host_only"):
            for repetition in range(2):
                rows.append({"case_id": case.case_id, "conversation_id": case.conversation_id,
                    "request_id": case.request_id, "cache_state": state, "repetition": repetition,
                    "status": "completed", "measurement_source": "hardware",
                    "execution_order": ["APPEND", "REBUILD"], "selected_action": "APPEND", "plan_reason": "hysteresis",
                    "observations": {c.action: {"input_hash": c.input_hash, "input_tokens": len(c.input_ids),
                        "output_tokens": 8+i if c.action == "APPEND" else 4+i,
                        "server_elapsed_ms": 80.+20*i+repetition if c.action == "APPEND" else 90.+2*i,
                        "quote": asdict(quote), "prediction": {},
                        "result": {"receipt": {"input_hash": c.input_hash}, "finish_reason": "stop"}}
                        for c in candidates}})
    model = CostModel([LookupPoint(0, len(c.input_ids), 1, 3.) for case in cases for c in case.candidates],
        [DecodePoint(200, 1, 1., context_start=1, context_end=1024)],
        default_output_tokens=5, p90_output_tokens=10, max_new_tokens=64)
    knobs = PlannerKnobs(delta_abs=2., delta_rel=.05)
    pairs = prepared_pairs(cases, rows, model, knobs, info)
    lengths, _ = build_cases([asdict(c) for c in cases], rows)
    return cases, lengths, pairs, model, knobs


@pytest.mark.parametrize("family", FAMILIES)
def test_outer_heldout_timing_and_length_labels_cannot_change_its_fit_or_selection(family):
    cases, lengths, pairs, model, knobs = inputs()
    first = NestedFeatures(cases, lengths, pairs, model, knobs)
    records, fold = fit_outer(first, family, "s4", alphas=(.1, 10.), quantiles=(0., .8))
    altered_lengths = [replace(c, targets={a: 63. for a in c.targets},
        observations={a: (63,)*len(v) for a, v in c.observations.items()}) if c.session_id == "s4" else c for c in lengths]
    altered_pairs = [dict(p, actual_gain_ms=-9000., actual_costs_ms={"APPEND": 100., "REBUILD": 9100.})
                     if p["session_id"] == "s4" else p for p in pairs]
    second = NestedFeatures(cases, altered_lengths, altered_pairs, model, knobs)
    changed, changed_fold = fit_outer(second, family, "s4", alphas=(.1, 10.), quantiles=(0., .8))
    assert changed_fold == fold
    for key in records:
        for old, new in zip(records[key], changed[key]):
            assert {k: old[k] for k in ("features", "predicted_gain_ms", "decision_threshold_ms", "length_means", "selected_action")} == {
                k: new[k] for k in ("features", "predicted_gain_ms", "decision_threshold_ms", "length_means", "selected_action")}
    assert first.fit_audit == second.fit_audit
    assert all("s4" not in item["training_sessions"] for item in first.fit_audit)
    assert all(set(item["training_sessions"]).isdisjoint(item["testing_sessions"]) for item in first.fit_audit)


def test_gain_features_and_replication_weighting_do_not_use_identifiers_or_current_labels():
    cases, lengths, pairs, model, knobs = inputs()
    features = NestedFeatures(cases, lengths, pairs, model, knobs)
    rows = features.crossfit("shared_history", {"s0", "s1", "s2", "s3"})
    assert set(rows[0]["features"]) == set(FEATURE_NAMES)
    original = fit_gain(rows, 1.)
    replicated = fit_gain(rows + deepcopy([r for r in rows if r["session_id"] == "s0"]), 1.)
    assert replicated == original
    assert len(cells(rows)) == 12  # 24 repeated pairs are 12 state cells, 4 cases.
    row = rows[0]
    changed = dict(row, actual_gain_ms=1e9, case_id="oracle", actual_costs_ms={"APPEND": 1e9})
    assert predict_gain(row, original) == predict_gain(changed, original)


def test_one_sided_threshold_uses_case_cells_not_repeated_observation_count():
    rows = [{"case_id": case, "cache_state": state, "eligible": True,
             "gain_ms": gain, "actual_gain_ms": 0.}
            for case, gain in (("a", 10.), ("b", 30.)) for state in ("cold", "warm")]
    first = one_sided_threshold(rows, .5)
    assert first["threshold_ms"] == 20. and first["cases"] == 2 and first["cells"] == 4
    assert one_sided_threshold(rows + rows[:2]*10, .5) == first
    assert one_sided_threshold(rows, 0.)["threshold_ms"] == 0.


def test_diagnostic_margin_never_changes_gain_choice_or_unsupported_fallback():
    row = {"eligible": True, "margin_ms": 20., "baseline_action": "APPEND",
           "baseline_reason": "hysteresis", "actual_gain_ms": 100.}
    assert decide(row, 25., 10.)["selected_action"] == "REBUILD"
    assert decide(row, 10.)["selected_action"] == "REBUILD"
    assert decide(dict(row, baseline_action="REBUILD"), 0.)["selected_action"] == "REBUILD"
    fallback = dict(row, eligible=False, baseline_reason="uncalibrated_cost_fallback")
    decision = decide(fallback, 1e6)
    assert decision["selected_action"] == "APPEND" and decision["reason"] == "uncalibrated_cost_fallback"


@pytest.mark.parametrize("gain", [-10., 0., 10.])
def test_configured_baseline_matches_serving_even_below_margin(monkeypatch, gain):
    from controller.schema import CostEstimate
    cases, _, pairs, model, knobs = inputs()
    monkeypatch.setattr(model, "predict_native", lambda candidate, *args, **kwargs:
        CostEstimate(candidate.candidate_id, 50. + (gain if candidate.action == "APPEND" else 0.),
                     calibrated=True))
    row = prediction_features(cases[0], pairs[0]["quote"], model, replace(knobs, delta_abs=40.))
    assert row["margin_ms"] == 40.
    assert decide({**row, "actual_gain_ms": 0.}, row["baseline_gain_ms"])["selected_action"] == row["baseline_action"]


def test_complete_replay_reports_fixed_baselines_without_fake_gain_coverage():
    cases, lengths, pairs, model, knobs = inputs()
    records, report = fit_exploration(NestedFeatures(cases, lengths, pairs, model, knobs),
        families=("shared_history",), alphas=(1.,), quantiles=(0., .8))
    assert len(records) == 3 and all(len(rows) == 30 for rows in records.values())
    for action, result in report["fixed_actions"].items():
        assert result["selected_actions"] == {action: 30}
        assert result["supported_gain_mae_ms"] is None
        assert result["equal_session_mean_regret_ms"] >= 0


def test_forced_rebuild_remains_a_constraint_not_a_gain_target():
    cases, _, pairs, model, knobs = inputs()
    case = replace(cases[0], snapshot=replace(cases[0].snapshot, must_rebuild=True))
    features = prediction_features(case, pairs[0]["quote"], model, knobs)
    assert not features["eligible"] and features["baseline_reason"] == "forced_rebuild"
    assert decide(dict(features, actual_gain_ms=-100.), -1e6)["selected_action"] == "REBUILD"
