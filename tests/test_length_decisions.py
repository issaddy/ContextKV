from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from controller.config import RunConfig
from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.engine_client import EngineInfo
from controller.schema import CacheQuote, CacheQuoteBatch, ConfigFingerprint, PromptCandidate, SessionSnapshot
from controller.templates import token_hash
from profiling.decision_pairs import DecisionCase
from profiling.length_decisions import FAMILIES, OOFLengthPrior, evaluate_decisions


def replay_inputs(tmp_path):
    cases, rows = [], []
    fingerprint = ConfigFingerprint()
    info = EngineInfo("restart", cache_config_hash="cache", fingerprint=fingerprint)
    for index in range(2):
        candidates = tuple(PromptCandidate(f"c{index}:{action}", action, 0, (), (index,)*n,
            token_hash((index,)*n), frozenset(), frozenset(), fingerprint=fingerprint)
            for action, n in (("APPEND", 100), ("REBUILD", 80)))
        case = DecisionCase(f"c{index}", f"s{index}", f"r{index}",
            SessionSnapshot(f"s{index}", turns_since_rebuild=3), candidates, (), 64, "APPEND")
        cases.append(case)
        quote = CacheQuoteBatch("q", 1, 1, "restart", "cache", 4096, 0,
            tuple(CacheQuote(c.candidate_id, c.input_hash, len(c.input_ids), len(c.input_ids)-1) for c in candidates),
            fingerprint=fingerprint)
        rows.append({"case_id": case.case_id, "conversation_id": case.conversation_id, "request_id": case.request_id,
            "cache_state": "cold", "repetition": 0, "status": "completed", "measurement_source": "simulation",
            "execution_order": ["APPEND", "REBUILD"], "selected_action": "APPEND", "plan_reason": "hysteresis",
            "observations": {c.action: {"input_hash": c.input_hash, "input_tokens": len(c.input_ids), "output_tokens": 5,
                "server_elapsed_ms": 50. if c.action == "APPEND" else 10., "quote": asdict(quote), "prediction": {},
                "result": {"text": "measured answer", "output_ids": [1]*5, "receipt": {"input_hash": c.input_hash}}}
                for c in candidates}})
    validation = {"scope": "exploratory_development_session_LOSO_not_final_test", "global_priors": [5, 10], "models": {}}
    for family in FAMILIES:
        means = {"APPEND": 5, "REBUILD": 5} if family == "shared_history" else {"APPEND": 20, "REBUILD": 3}
        validation["models"][family] = {
            "folds": [{"heldout_session": f"s{i}", "training_sessions": [f"s{1-i}"],
                "uncertainty_scope": "frozen shared prior" if family == "shared_history" else "training-session inner OOF only"}
                for i in range(2)],
            "oof_predictions": [{"case_id": c.case_id, "session_id": c.conversation_id,
                "predicted_tokens": dict(means), "p90_tokens": {a: n+2 for a, n in means.items()},
                "error_tokens": {a: None if family == "shared_history" else 500. for a in means},
                "within_training_range": {a: True for a in means},
                "target_mean_tokens": {a: 5 for a in means}} for c in cases]}
    spec = {"coefficients": [2., 0., 0., 0., 0., 0., 0., 0.], "fit_sources": ["hardware"],
        "uncertainty_sufficient": True, "error_p99_ms": 100.,
        "conditional_uncertainty": {"minimum_feature_cells": 20, "strata": [{"concurrency": 1,
            "host_load": None, "host_load_states": [False], "feature_cells": 20, "error_p99_ms": 1.}]}}
    artifact = tmp_path/"cost.json"
    CostModel([LookupPoint(0, n, 1, 2.) for n in (80, 100)],
        [DecodePoint(200, 1, 1., context_start=1, context_end=1024)],
        default_output_tokens=5, p90_output_tokens=10, max_new_tokens=64,
        prefill_estimator="positive_basis", positive_prefill=spec, uncertainty_mode="conditional").save(artifact)
    return cases, rows, validation, artifact, info


def test_oof_replay_uses_real_planner_and_preserves_measurements(tmp_path):
    cases, rows, validation, artifact, info = replay_inputs(tmp_path)
    original = deepcopy(rows)
    records, summaries = evaluate_decisions(cases, rows, validation, RunConfig(), artifact, info)
    assert len(records) == 8
    assert summaries["global-candidate"]["plan_reasons"] == {"cost_saving": 2}
    assert summaries["conditional-candidate"]["plan_reasons"] == {"cost_saving": 2}
    assert summaries["conditional-shared_history"]["mean_selected_regret_ms"] == 0.
    assert summaries["conditional-candidate"]["mean_selected_regret_ms"] == 0.
    assert all(arm["prediction"]["output_length_error_tokens"] == 500.
               for row in records["conditional-candidate"] for arm in row["observations"].values())
    assert rows == original
    for row, old in zip(records["conditional-candidate"], original):
        for action in ("APPEND", "REBUILD"):
            assert {k: v for k, v in row["observations"][action].items() if k != "prediction"} == {
                k: v for k, v in old["observations"][action].items() if k != "prediction"}


def test_changing_current_outcomes_and_report_labels_cannot_change_oof_selection(tmp_path):
    cases, rows, validation, artifact, info = replay_inputs(tmp_path)
    original, _ = evaluate_decisions(cases, rows, validation, RunConfig(), artifact, info)
    changed = deepcopy(rows)
    for row in changed:
        for arm in row["observations"].values():
            arm.update(server_elapsed_ms=100000., output_tokens=60)
            arm["result"].update(text="changed current outcome", output_ids=[9]*60)
    altered = deepcopy(validation)
    for model in altered["models"].values():
        for prediction in model["oof_predictions"]:
            prediction["target_mean_tokens"] = {"APPEND": 9999, "REBUILD": 1}
    replayed, _ = evaluate_decisions(cases, changed, altered, RunConfig(), artifact, info)
    for variant in original:
        for first, second in zip(original[variant], replayed[variant]):
            assert (first["selected_action"], first["plan_reason"]) == (second["selected_action"], second["plan_reason"])
            assert all(first["observations"][a]["prediction"] == second["observations"][a]["prediction"]
                       for a in ("APPEND", "REBUILD"))


@pytest.mark.parametrize("problem", ["missing_case", "schema", "fold_leakage", "budget", "full_fit"])
def test_invalid_oof_binding_is_rejected_before_prediction(tmp_path, problem):
    cases, _, validation, _, _ = replay_inputs(tmp_path)
    if problem == "missing_case":
        validation["models"]["candidate"]["oof_predictions"].pop()
    elif problem == "schema":
        validation["models"]["candidate"]["oof_predictions"][0].pop("p90_tokens")
    elif problem == "fold_leakage":
        validation["models"]["candidate"]["folds"][0]["training_sessions"].append("s0")
    elif problem == "budget":
        validation["models"]["candidate"]["oof_predictions"][0]["predicted_tokens"]["APPEND"] = 65
    else:
        validation = {"kind": "contextkv_output_length_ridge_v1", "model_kind": "candidate"}
    with pytest.raises(ValueError, match="OOF|LOSO|budget|held-out"):
        OOFLengthPrior(validation, "candidate", cases)


def test_replay_cannot_silently_mix_configured_full_fit_artifact(tmp_path):
    cases, rows, validation, artifact, info = replay_inputs(tmp_path)
    config = RunConfig()
    config = replace(config, cost_model=replace(config.cost_model, output_length_artifact="full-fit.json"))
    with pytest.raises(ValueError, match="full-fit"):
        evaluate_decisions(cases, rows, validation, config, artifact, info)
