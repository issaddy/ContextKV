from copy import deepcopy
from dataclasses import replace

import pytest

from controller.context_schema import ReadingOutcome
from controller.plan_policy import FEATURE_NAMES, plan_features
from controller.schema import SessionSnapshot, TurnRecord
from training.train_plan_gain import (PlanGainModel, _supervision, actual_history,
                                     build_gain_record, working_state_key)
from test_plan_policy import make_plans


def observed_pair():
    """Specified receipt/score fixtures for testing data binding, not real runs."""
    base, direct, plan = make_plans()
    state = ReadingOutcome(base)
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(1, "Use Windows", "Windows permits 10"),))
    shared = {"measurement_kind": "hardware", "history_origin": "frozen_actual_rollout",
        "question": base.task_state.raw_question, "actual_history": actual_history(snapshot),
        "state_key": working_state_key(state, snapshot), "reader_checkpoint": "reader-fixture",
        "reader_profile_id": "profile-fixture", "split": "train", "answer": "specified answer",
        "scorer_identity": {"model": "specified-test-scorer", "metric": "quality", "normalization": "identity_0_1"},
        "comparison_budget": {"body_tokens": 8192, "answer_output_tokens": 64},
        "native_receipts": [{"input_hash": "native-test-input", "generation_complete": True}],
        "cost": {"complete_ms": 20, "gpu_gib_seconds": .2, "host_gib_seconds": .1,
                 "total_input_tokens": 100, "total_output_tokens": 30}}
    direct_run = {**deepcopy(shared), "semantic_kind": "direct_answer", "plan_id": direct.plan_id,
                  "read_count": 0, "quality": .2}
    counterfactual = {**deepcopy(shared), "semantic_kind": "complementary_evidence", "plan_id": plan.plan_id,
                      "read_count": 1, "quality": .8}
    return plan, state, snapshot, direct_run, counterfactual


def test_counterfactual_labels_equal_actual_answer_score_delta_and_keep_full_costs():
    plan, state, snapshot, direct, counterfactual = observed_pair()
    # Existing support/set quality is deliberately unrelated to this score gap.
    plan = replace(plan, context_plan=replace(plan.context_plan, selected_set=replace(
        plan.context_plan.selected_set, quality_prediction=.01)))
    record = build_gain_record(plan, state, snapshot, direct, counterfactual)
    assert record["delta_q"] == pytest.approx(.6)
    assert record["direct_run"] == direct and record["counterfactual_run"] == counterfactual
    assert record["features"] == plan_features(plan, state)
    assert record["counterfactual_run"]["cost"]["gpu_gib_seconds"] == .2
    assert _supervision([record], split="train")[:2] == ("reader-fixture", "profile-fixture")


@pytest.mark.parametrize("field,value,match", (
    ("reader_checkpoint", "reader-next", "different reader"),
    ("reader_profile_id", "runtime-next", "reader_profile_id"),
    ("split", "test", "test answers"),
    ("measurement_kind", "mock", "hardware"),
    ("history_origin", "author_reference", "actual model history"),
    ("scorer_identity", {"model": "different-scorer"}, "scorer_identity"),
    ("comparison_budget", {"answer_output_tokens": 128}, "comparison_budget"),
    ("question", "other question", "frozen logical state"),
    ("plan_id", "other-plan", "labelled complete plan"),
))
def test_gain_labels_reject_changed_checkpoint_history_split_budget_or_rubric(field, value, match):
    plan, state, snapshot, direct, counterfactual = observed_pair()
    counterfactual[field] = value
    with pytest.raises(ValueError, match=match):
        build_gain_record(plan, state, snapshot, direct, counterfactual)


def test_gain_labels_require_actual_history_completed_receipts_and_measured_resources():
    plan, state, snapshot, direct, counterfactual = observed_pair()
    changed = deepcopy(counterfactual)
    changed["actual_history"][0]["assistant_text"] = "unobserved author answer"
    with pytest.raises(ValueError, match="frozen logical state"):
        build_gain_record(plan, state, snapshot, direct, changed)
    changed = deepcopy(counterfactual)
    changed["native_receipts"][0]["generation_complete"] = False
    with pytest.raises(ValueError, match="completed native"):
        build_gain_record(plan, state, snapshot, direct, changed)
    changed = deepcopy(counterfactual)
    changed["cost"]["host_gib_seconds"] = None
    with pytest.raises(ValueError, match="physical integrals"):
        build_gain_record(plan, state, snapshot, direct, changed)


def test_imported_gain_supervision_cannot_substitute_support_or_test_labels():
    plan, state, snapshot, direct, counterfactual = observed_pair()
    record = build_gain_record(plan, state, snapshot, direct, counterfactual)
    record["delta_q"] = .99
    with pytest.raises(ValueError, match="observed counterfactual quality"):
        _supervision([record], split="train")
    record = build_gain_record(plan, state, snapshot, direct, counterfactual)
    record["source"] = "support_scores"
    with pytest.raises(ValueError, match="actual answer counterfactuals"):
        _supervision([record], split="train")
    record["source"] = "actual_answer_counterfactual"
    record["split"] = "test"
    with pytest.raises(ValueError, match="actual answer counterfactuals"):
        _supervision([record], split="train")


def test_portable_gain_asset_predicts_and_loads_only_its_bound_reader(tmp_path):
    # Explicit fixture coefficients test loading and feature arithmetic; no fit
    # or model training is executed by this test.
    model = PlanGainModel("reader-fixture", "profile-fixture", "test-only-explicit-coefficients", FEATURE_NAMES,
        (0.0,)*len(FEATURE_NAMES), (1.0,)*len(FEATURE_NAMES),
        tuple(.2 if name == "read_steps" else 0.0 for name in FEATURE_NAMES), -.1,
        {"source": "unit_test_explicit_coefficients"})
    path = tmp_path/"fixture-gain.json"
    model.save(path)
    loaded = PlanGainModel.load(path, reader_checkpoint="reader-fixture", reader_profile_id="profile-fixture")
    plan, state, _, _, _ = observed_pair()
    assert loaded.predict(plan_features(plan, state)) == pytest.approx(.1)
    with pytest.raises(ValueError, match="measured reader"):
        PlanGainModel.load(path, reader_checkpoint="new-reader", reader_profile_id="profile-fixture")
