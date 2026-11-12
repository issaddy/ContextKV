import unittest
from copy import deepcopy
from types import SimpleNamespace

from controller.action_admission import TERMINAL_COST_TARGET, validated_action_asset


def asset():
    return SimpleNamespace(asset_id="gain:1", cost_provenance={"cost_target_version": TERMINAL_COST_TARGET},
        provenance={"trained_semantic_kinds": ["read_once"], "training_conversations": ["train"],
            "heldout_decision_validation": {"split": "dev", "model_asset_id": "gain:1",
                "development_digest": "digest", "train_conversations_disjoint": True,
                "conversation_ids": [f"dev:{i}" for i in range(5)], "report": {
                    "paired_direct_reference_states": 40, "mean_paired_objective_delta_vs_direct": 0.01,
                    "by_family": {"read_once": {"records": 40, "sign_agreement": 0.8}},
                    "untrained_family_counts": {}}}})


class ActionAdmissionTests(unittest.TestCase):
    def test_positive_bound_development_allows_learned_policy(self):
        verdict = validated_action_asset(asset())
        self.assertTrue(verdict["admitted"])
        self.assertEqual(verdict["policy"], "learned")
        self.assertTrue(verdict["independent_test_required"])

    def test_reported_negative_objective_and_bad_read_sign_fall_back(self):
        model = asset()
        report = model.provenance["heldout_decision_validation"]["report"]
        report["mean_paired_objective_delta_vs_direct"] = -0.001167
        report["by_family"]["read_once"].update(records=26, sign_agreement=0.3846)
        verdict = validated_action_asset(model)
        self.assertFalse(verdict["admitted"])
        self.assertEqual(verdict["policy"], "rule")
        self.assertIn("no_positive_heldout_objective_gain", verdict["reasons"])
        self.assertIn("unreliable_gain_sign:read_once", verdict["reasons"])

    def test_legacy_delivered_only_cost_asset_is_not_deployed(self):
        model = asset()
        model.cost_provenance.clear()
        self.assertIn("cost_target_requires_terminal_failure_refit", validated_action_asset(model)["reasons"])

    def test_absent_mismatched_or_leaking_validation_fails_closed(self):
        baseline = asset()
        for kind in ("absent", "wrong_asset", "train_leak", "no_digest"):
            model = deepcopy(baseline)
            validation = model.provenance["heldout_decision_validation"]
            if kind == "absent":
                del model.provenance["heldout_decision_validation"]
            elif kind == "wrong_asset":
                validation["model_asset_id"] = "another-model"
            elif kind == "train_leak":
                validation["conversation_ids"].append("train")
            else:
                validation.pop("development_digest")
            with self.subTest(kind=kind):
                self.assertFalse(validated_action_asset(model)["admitted"])

    def test_sparse_new_action_family_and_fixed_representation_are_gated(self):
        model = asset()
        model.provenance["trained_semantic_kinds"].append("gap_retrieval")
        model.provenance["representation_gains"] = {"facts": {"delta_q": 0.05}}
        verdict = validated_action_asset(model)
        self.assertIn("insufficient_family_records:gap_retrieval", verdict["reasons"])
        self.assertIn("insufficient_family_records:facts", verdict["reasons"])

    def test_invalid_thresholds_are_rejected(self):
        for thresholds in ({"min_sign_agreement": float("nan")}, {"min_objective_delta": -1},
                           {"min_read_records": True}, {"bogus": 1}):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                validated_action_asset(asset(), thresholds)


if __name__ == "__main__":
    unittest.main()
