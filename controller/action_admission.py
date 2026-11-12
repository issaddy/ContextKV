"""Deployment gate for learned continuation, separate from benchmark acceptance."""
from __future__ import annotations

import math


TERMINAL_COST_TARGET = "terminal_attempt_incurred_cost_v2"
DEFAULT_VALIDATION = {
    "min_paired_states": 30,
    "min_conversations": 5,
    "min_family_records": 10,
    "min_read_records": 20,
    "min_sign_agreement": 0.60,
    "min_objective_delta": 0.0,
}


def validated_action_asset(model, thresholds=None):
    """Return an auditable decision; missing validation never implies approval.

    These are conservative engineering admission thresholds, not a claim of
    statistical significance or a replacement for independent benchmark tests.
    Explicit ``action_policy=learned`` remains available for research ablations.
    """
    supplied = thresholds or {}
    unknown = set(supplied)-DEFAULT_VALIDATION.keys()
    if unknown:
        raise ValueError(f"unknown action validation thresholds: {sorted(unknown)}")
    limits = {**DEFAULT_VALIDATION, **supplied}
    for key, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"action validation {key} must be finite numeric")
        if key != "min_objective_delta" and value < 0:
            raise ValueError(f"action validation {key} must be nonnegative")
    if not 0 <= limits["min_sign_agreement"] <= 1 or limits["min_objective_delta"] < 0:
        raise ValueError("action validation cannot admit negative held-out benefit")

    reasons = []
    if model.cost_provenance.get("cost_target_version") != TERMINAL_COST_TARGET:
        reasons.append("cost_target_requires_terminal_failure_refit")
    validation = model.provenance.get("heldout_decision_validation") or {}
    report = validation.get("report") or {}
    if validation.get("split") != "dev" or validation.get("model_asset_id") != model.asset_id:
        reasons.append("missing_asset_bound_development_validation")
    conversations = set(validation.get("conversation_ids", ()))
    training = set(model.provenance.get("training_conversations", ()))
    if (not validation.get("development_digest") or conversations & training
            or validation.get("train_conversations_disjoint") is not True):
        reasons.append("unverified_development_separation")
    if len(conversations) < limits["min_conversations"]:
        reasons.append("insufficient_development_conversations")
    if report.get("paired_direct_reference_states", 0) < limits["min_paired_states"]:
        reasons.append("insufficient_paired_direct_states")
    delta = report.get("mean_paired_objective_delta_vs_direct")
    if (isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta)
            or delta <= limits["min_objective_delta"]):
        reasons.append("no_positive_heldout_objective_gain")
    families = set(model.provenance.get("trained_semantic_kinds", ())) - {"direct_answer"}
    families.update(model.provenance.get("representation_gains", ()))
    families.add("read_once")
    for family in sorted(families):
        row = report.get("by_family", {}).get(family, {})
        minimum = limits["min_read_records"] if family == "read_once" else limits["min_family_records"]
        if row.get("records", 0) < minimum:
            reasons.append(f"insufficient_family_records:{family}")
        sign = row.get("sign_agreement")
        if (isinstance(sign, bool) or not isinstance(sign, (int, float)) or not math.isfinite(sign)
                or not limits["min_sign_agreement"] <= sign <= 1):
            reasons.append(f"unreliable_gain_sign:{family}")
    if any(report.get("untrained_family_counts", {}).values()):
        reasons.append("untrained_development_actions")
    return {"admitted": not reasons, "policy": "learned" if not reasons else "rule",
            "asset_id": model.asset_id, "reasons": tuple(reasons), "thresholds": limits,
            "validation_source_digest": validation.get("development_digest"),
            "independent_test_required": True,
            "latency_calibration_uses_same_development_partition": validation.get(
                "latency_calibration_uses_same_development_partition")}
