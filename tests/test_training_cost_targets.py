"""Failure-inclusive cost fitting and honest compatibility for historical assets."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from controller.plan_gain_protocol import input_protocol
from controller.plan_policy import OBSERVED_FEATURE_NAMES, NATIVE_COST_FEATURE_NAMES
from controller.schema import ConfigFingerprint
from training.plan_gain_completion import (COST_TARGET_VERSION, RULE, calibrate_completion_latency,
    digest, evaluate_paired_decisions, fit_effective, require_terminal_cost_target,
    with_heldout_decision_validation)
from training.train_plan_gain import COST_FIELDS, PlanGainModel


def branch(identity, kind, *, completion=1, cost=100.0, conversation="training", reads=1):
    features = {name: 0.0 for name in OBSERVED_FEATURE_NAMES}
    features[kind] = 1.0
    native = {name: 0.0 for name in NATIVE_COST_FEATURE_NAMES}
    native[kind] = 1.0
    return dict(record_id=identity, conversation_id=conversation, plan_id=identity,
        semantic_kind=kind, answer_representation="existing", features=features,
        cost_features=native, completion=completion,
        completion_reason="committed_QA" if completion else "confirmed_terminal_no_QA",
        effective_utility=(0.6 if kind == "direct_answer" else 0.7) if completion else 0.0,
        actual_cost={name: cost for name in COST_FIELDS},
        continuation_gap_limits={"max_queries": 1}, cache_start="cold", first_output_tokens=0,
        new_action_counts={"READ": reads if kind == "read_once" else 0},
        observed_state_binding={"resident_model_manifest_digest": "fixed-resident-models",
            "action_domain": "test-bounded-actions", "paid_cost": {
                "latency_ms": 0, "input_tokens": 0, "output_tokens": 0}},
        budget={"lambda_time": 0, "lambda_gpu": 0, "lambda_host": 0,
            "max_complete_ms": None, "max_total_tokens": None, "max_steps": 64})


def pair(direct, action, *, split="train"):
    return dict(source="actual_completion_weighted_counterfactual", split=split,
        utility_rule=RULE, equivalent_representation=False,
        reader_checkpoint="test-reader", reader_profile_id="test-profile",
        input_protocol=input_protocol(ConfigFingerprint(model_revision="test-reader")),
        evidence_selection_contract=None, observed_state_binding=action["observed_state_binding"],
        scorer_identity={"id": "test-fixed-judge"}, semantic_kind=action["semantic_kind"],
        conversation_id=action["conversation_id"], state_key="state-"+direct["record_id"],
        direct=direct, counterfactual=action, features=action["features"],
        answer_representation="existing",
        completion_direct=direct["completion"], completion_counterfactual=action["completion"],
        raw_Q_direct=direct["effective_utility"] if direct["completion"] else None,
        raw_Q_counterfactual=action["effective_utility"] if action["completion"] else None,
        utility_direct=direct["effective_utility"], utility_counterfactual=action["effective_utility"],
        effective_delta_q=action["effective_utility"]-direct["effective_utility"])


def records():
    direct = branch("direct", "direct_answer")
    success = branch("read-success", "read_once", reads=2)
    failure = branch("read-failure", "read_once", completion=0, cost=1000, reads=1)
    return [pair(direct, success), pair(direct, failure)]


class TerminalAttemptCostTests(unittest.TestCase):
    def test_terminal_failure_costs_change_the_fitted_cost_not_only_quality(self):
        training = records()
        fitted = fit_effective(training, quality_head="linear")
        success_only = fit_effective(training[:1], quality_head="linear")
        provenance = fitted.cost_provenance
        self.assertEqual(provenance["cost_target_version"], COST_TARGET_VERSION)
        self.assertEqual(provenance["training_samples"], 3)
        self.assertEqual(provenance["unique_direct_samples"], 1)
        self.assertEqual(provenance["action_samples"], 2)
        self.assertEqual(provenance["training_completion_counts"], {"1": 2, "0": 1})
        self.assertIn("read-failure", provenance["record_ids"])
        features = training[1]["counterfactual"]["cost_features"]
        predicted = fitted.predict_costs(features)["branch_complete_cost"]
        baseline = success_only.predict_costs(features)["branch_complete_cost"]
        for field in COST_FIELDS:
            self.assertGreater(predicted[field], baseline[field]+100)
        profile = provenance["read_count_profile"]
        self.assertEqual(profile["scope"], "unique_measured_terminal_cost_training_branches")
        self.assertEqual(profile["groups"]["read_once"]["without_infer"], {"mean": 1.5, "samples": 2})

    def test_failed_direct_branch_is_not_dropped_or_counted_per_pair(self):
        direct = branch("failed-direct", "direct_answer", completion=0, cost=900)
        rows = [pair(direct, branch("action-a", "read_once")),
                pair(direct, branch("action-b", "read_once"))]
        fitted = fit_effective(rows, quality_head="linear")
        self.assertEqual(fitted.cost_provenance["record_ids"].count("failed-direct"), 1)
        self.assertEqual(fitted.cost_provenance["unique_direct_samples"], 1)
        self.assertEqual(fitted.cost_provenance["training_samples"], 3)
        self.assertEqual(fitted.provenance["measured_paid_cost_outcomes"]["confirmed_terminal_no_QA"], 1)

    def test_missing_nonfinite_and_negative_costs_are_never_zero_imputed(self):
        for missing in (None, math.nan, math.inf, -1.0, True):
            training = records()
            training[1]["counterfactual"]["actual_cost"]["total_output_tokens"] = missing
            fitted = fit_effective(training, quality_head="linear")
            self.assertEqual(fitted.cost_provenance["training_samples"], 2)
            self.assertNotIn("read-failure", fitted.cost_provenance["record_ids"])
            self.assertEqual(fitted.cost_provenance["excluded_cost_records"],
                {"missing_or_invalid_measured_cost": 1})
            self.assertEqual(fitted.provenance["training_records"], 2)
            # Failure quality remains a real zero-utility label even though its
            # incomplete cost vector cannot enter the cost head.
            self.assertEqual(fitted.provenance["completion_counts"], {"1": 1, "0": 1})

    def test_known_zero_cost_is_distinct_from_missing_cost(self):
        training = records()
        training[1]["counterfactual"]["actual_cost"] = {name: 0.0 for name in COST_FIELDS}
        fitted = fit_effective(training, quality_head="linear")
        self.assertIn("read-failure", fitted.cost_provenance["record_ids"])
        self.assertEqual(fitted.cost_provenance["excluded_cost_records"], {})

    def test_unresolved_attempt_is_not_a_complete_cost_target(self):
        training = records()
        training[1]["counterfactual"]["completion"] = None
        fitted = fit_effective(training, quality_head="linear")
        self.assertNotIn("read-failure", fitted.cost_provenance["record_ids"])
        self.assertEqual(fitted.cost_provenance["excluded_cost_records"],
            {"unresolved_terminal_outcome": 1})

    def test_latency_calibration_retains_measured_terminal_failures(self):
        fitted = fit_effective(records(), quality_head="linear")
        development = [pair(branch("direct-"+name, "direct_answer", conversation=name),
            branch("failed-"+name, "read_once", completion=0, cost=cost, conversation=name), split="dev")
            for name, cost in (("development-a", 5000), ("development-b", 9000))]
        calibrated = calibrate_completion_latency(fitted, development)
        residuals = calibrated.cost_provenance["completion_reserve_calibration"]["read_once"]
        self.assertEqual(residuals["completion_counts"], {"0": 2})
        self.assertEqual(residuals["cost_target_version"], COST_TARGET_VERSION)
        self.assertGreater(residuals["latency_margin_ms"], 8000)
        report = evaluate_paired_decisions(calibrated, development)
        self.assertEqual(report["paired_direct_reference_states"], 2)
        self.assertIsNotNone(report["mean_paired_objective_delta_vs_direct"])

    def test_historical_asset_loads_for_audit_but_cannot_claim_new_cost_target(self):
        fitted = fit_effective(records(), quality_head="linear")
        old_provenance = deepcopy(fitted.cost_provenance)
        del old_provenance["cost_target_version"]
        legacy = replace(fitted, cost_provenance=old_provenance)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/"legacy-asset.json"
            legacy.save(path)
            loaded = PlanGainModel.load(path, reader_checkpoint="test-reader", reader_profile_id="test-profile")
            with self.assertRaisesRegex(ValueError, "refit.*measured successful and confirmed failed"):
                require_terminal_cost_target(loaded)
            with self.assertRaisesRegex(ValueError, "relabel"):
                calibrate_completion_latency(loaded, [])
            with self.assertRaisesRegex(ValueError, "terminal_attempt_incurred_cost_v2"):
                evaluate_paired_decisions(loaded, [])
        require_terminal_cost_target(fitted)

    def test_heldout_report_is_bound_to_asset_and_partition_without_inventing_pass(self):
        fitted = fit_effective(records(), quality_head="linear")
        development = [pair(branch("dev-direct", "direct_answer", conversation="dev-conversation"),
            branch("dev-action", "read_once", conversation="dev-conversation"), split="dev")]
        report = {"asset_id": fitted.asset_id, **evaluate_paired_decisions(fitted, development)}
        bound = with_heldout_decision_validation(fitted, development, report)
        validation = bound.provenance["heldout_decision_validation"]
        self.assertEqual(validation["split"], "dev")
        self.assertTrue(validation["train_conversations_disjoint"])
        self.assertEqual(validation["model_asset_id"], fitted.asset_id)
        self.assertEqual(validation["development_digest"], digest(development))
        self.assertEqual(validation["conversation_ids"], ["dev-conversation"])
        self.assertEqual(validation["report"], report)
        self.assertEqual(report["by_family"]["read_once"]["records"], 1)
        self.assertEqual(report["by_family"]["read_once"]["conversations"], 1)
        self.assertNotIn("passed", validation)
        with self.assertRaisesRegex(ValueError, "different fitted asset"):
            with_heldout_decision_validation(fitted, development, {**report, "asset_id": "different"})
        with self.assertRaisesRegex(ValueError, "dev partition"):
            with_heldout_decision_validation(fitted, records(), report)
        overlapping = deepcopy(development)
        overlapping[0]["conversation_id"] = "training"
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            with_heldout_decision_validation(fitted, overlapping, report)


if __name__ == "__main__":
    unittest.main()
