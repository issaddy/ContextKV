"""Default validation audits incompatible assets without executing their weights."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from controller.action_admission import TERMINAL_COST_TARGET
from controller.action_proposals import BOUNDED_CONTINUATION, finite_action_domain
from controller.context_budget import COMPLETION_RESERVE_SCOPE
from controller.plan_gain_protocol import input_protocol, validate_input_protocol
from controller.plan_policy import (NATIVE_COST_FEATURE_NAMES, NATIVE_COST_SCHEMA,
    OBSERVED_FEATURE_NAMES, OBSERVED_FEATURE_SCHEMA)
from controller.plan_profiles import load_action_gain_asset, load_rolling_runtime
from controller.schema import ConfigFingerprint
from training.train_plan_gain import COST_FIELDS, PlanGainModel


FINGERPRINT = ConfigFingerprint(model_revision="reader-v2")


def asset_data():
    asset = "validated-test-asset"
    provenance = {
        "input_protocol": input_protocol(FINGERPRINT),
        "observed_feature_schema": OBSERVED_FEATURE_SCHEMA,
        "continuation_protocol": BOUNDED_CONTINUATION,
        "continuation_gap_limits": {"max_units": 96, "max_passages": 10},
        "action_domain": finite_action_domain((2048,), 64),
        "label": "actual_completion_weighted_final_utility_minus_direct_utility",
        "training_split": "train", "training_records": 100,
        "training_conversations": ["training-conversation"],
        "no_answer_utility": 0, "raw_missing_quality_preserved": True,
        "trained_semantic_kinds": ["read_once"], "direct_baseline_identity": True,
        "heldout_decision_validation": {"schema_version": 1, "split": "dev", "model_asset_id": asset,
            "cost_target_version": TERMINAL_COST_TARGET,
            "development_digest": "test-fixture-development-digest", "train_conversations_disjoint": True,
            "conversation_ids": [f"dev-{i}" for i in range(5)],
            "report": {"asset_id": asset, "paired_direct_reference_states": 40,
                "mean_paired_objective_delta_vs_direct": 0.01,
                "by_family": {"read_once": {"records": 40, "conversations": 5,
                    "sign_agreement": 0.8, "mae": 0.1}}, "untrained_family_counts": {}}},
    }
    return {
        "schema_version": 3, "model_type": "reader_bound_plan_gain_cost_ridge",
        "reader_checkpoint": FINGERPRINT.model_revision, "reader_profile_id": "profile-v2",
        "asset_id": asset, "feature_names": list(OBSERVED_FEATURE_NAMES),
        "center": [0]*len(OBSERVED_FEATURE_NAMES), "scale": [1]*len(OBSERVED_FEATURE_NAMES),
        "coefficients": [0]*len(OBSERVED_FEATURE_NAMES), "intercept": 0,
        "provenance": provenance,
        "cost_provenance": {"feature_schema": NATIVE_COST_SCHEMA,
            "feature_names": list(NATIVE_COST_FEATURE_NAMES), "scale": [1]*len(NATIVE_COST_FEATURE_NAMES),
            "observed_feature_schema": OBSERVED_FEATURE_SCHEMA,
            "trained_semantic_kinds": ["direct_answer", "read_once"],
            "completion_reserve_scope": COMPLETION_RESERVE_SCOPE, "cost_target_version": TERMINAL_COST_TARGET},
        "cost_models": {"branch_complete_cost": {name: {
            "coefficients": [0]*len(NATIVE_COST_FEATURE_NAMES), "intercept": 1} for name in COST_FIELDS}},
        "quality_tree": None,
    }


def old_input_protocol():
    protocol = input_protocol(FINGERPRINT)
    for name in ("schema_version", "need_dependencies", "read_recovery", "terminal_gaps"):
        del protocol[name]
    protocol["numeric_selection"] = "original_source_excerpts_v1"
    protocol["completion_admission"] = "mandatory_future_work_v1"
    return protocol


class ActionRuntimeLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/"gain.json"
        self.settings = {
            "gain_asset": str(self.path), "reader_checkpoint": FINGERPRINT.model_revision,
            "reader_profile_id": "profile-v2", "mode": "observed_actions", "joint_context": True,
            "gpu_devices": [], "process_ids": [], "sampling_interval_seconds": 0.5,
            "output_budgets": [2048], "output_tokens_per_fact": 64,
            "gap_limits": {"max_units": 96, "max_passages": 10},
        }

    def write(self, data):
        self.path.write_text(json.dumps(data))

    def test_current_validated_asset_passes_the_real_strict_loader(self):
        self.write(asset_data())
        with patch.object(PlanGainModel, "load", wraps=PlanGainModel.load) as loader:
            model, admission, metadata = load_action_gain_asset(self.settings, reader_fingerprint=FINGERPRINT)
        self.assertIsInstance(model, PlanGainModel)
        self.assertTrue(admission["admitted"])
        self.assertTrue(admission["loaded_for_prediction"])
        self.assertEqual(admission["asset_inspection"], "strict_model_load")
        self.assertEqual(loader.call_count, 1)
        self.assertFalse(hasattr(metadata, "predict"))
        self.assertFalse(hasattr(metadata, "predict_costs"))
        self.assertEqual(metadata.provenance["action_domain"], finite_action_domain((2048,), 64))

    def test_old_protocol_is_audited_without_constructing_model(self):
        data = asset_data()
        data["provenance"]["input_protocol"] = old_input_protocol()
        data["coefficients"] = "not loaded or used for predictions"
        self.write(data)
        with patch.object(PlanGainModel, "load", side_effect=AssertionError("must not load stale weights")):
            model, admission, metadata = load_action_gain_asset(self.settings, reader_fingerprint=FINGERPRINT)
        self.assertIsNone(model)
        self.assertEqual(admission["policy"], "rule")
        self.assertFalse(admission["admitted"])
        self.assertFalse(admission["loaded_for_prediction"])
        self.assertEqual(admission["asset_inspection"], "metadata_only")
        self.assertIn("input_protocol_requires_recollection", admission["reasons"])
        self.assertEqual(metadata.provenance["continuation_gap_limits"], self.settings["gap_limits"])
        self.assertEqual(admission["asset_sha256"], metadata.source_sha256)

    def test_default_runtime_survives_old_asset_with_explicit_rule_diagnostic(self):
        data = asset_data()
        data["provenance"]["input_protocol"] = old_input_protocol()
        self.write(data)
        with patch.object(PlanGainModel, "load", side_effect=AssertionError("must not load stale weights")):
            runtime = load_rolling_runtime(self.settings, template=SimpleNamespace(fingerprint=FINGERPRINT),
                store=None, sampling={}, answer_system_prompt="answer", engine_base_url="http://unused")
        self.assertIsNone(runtime.policy.gain_model)
        self.assertTrue(runtime.joint_context)
        self.assertEqual(runtime.native.fingerprint, FINGERPRINT)
        self.assertIn("input_protocol_requires_recollection", runtime.action_policy_admission["reasons"])
        self.assertFalse(runtime.native.paid.started)

    def test_preflight_without_template_fingerprint_uses_same_metadata_fallback(self):
        data = asset_data()
        data["provenance"]["input_protocol"] = old_input_protocol()
        self.write(data)
        with patch.object(PlanGainModel, "load", side_effect=AssertionError("must not load stale weights")):
            model, admission, metadata = load_action_gain_asset(self.settings)
        self.assertIsNone(model)
        self.assertEqual(metadata.reader_profile_id, "profile-v2")
        self.assertEqual(admission["policy"], "rule")

    def test_poor_development_evidence_never_constructs_predictor(self):
        data = asset_data()
        report = data["provenance"]["heldout_decision_validation"]["report"]
        report["mean_paired_objective_delta_vs_direct"] = -0.001167
        report["by_family"]["read_once"]["sign_agreement"] = 0.3846
        self.write(data)
        with patch.object(PlanGainModel, "load", side_effect=AssertionError("must not load unvalidated weights")):
            model, admission, _ = load_action_gain_asset(self.settings, reader_fingerprint=FINGERPRINT)
        self.assertIsNone(model)
        self.assertIn("no_positive_heldout_objective_gain", admission["reasons"])
        self.assertIn("unreliable_gain_sign:read_once", admission["reasons"])

    def test_explicit_learned_rejects_old_protocol_instead_of_silent_downgrade(self):
        data = asset_data()
        data["provenance"]["input_protocol"] = old_input_protocol()
        self.write(data)
        with self.assertRaisesRegex(ValueError, "recollect frozen branches"):
            load_action_gain_asset({**self.settings, "action_policy": "learned"}, reader_fingerprint=FINGERPRINT)
        with self.assertRaisesRegex(ValueError, "actual source/compiler/reader scope"):
            PlanGainModel.load(self.path, reader_checkpoint=FINGERPRINT.model_revision,
                reader_profile_id="profile-v2", reader_fingerprint=FINGERPRINT)

    def test_explicit_learned_can_run_current_protocol_research_ablation(self):
        data = asset_data()
        del data["provenance"]["heldout_decision_validation"]
        self.write(data)
        model, admission, _ = load_action_gain_asset({**self.settings, "action_policy": "learned"},
            reader_fingerprint=FINGERPRINT)
        self.assertIsInstance(model, PlanGainModel)
        self.assertTrue(admission["loaded_for_prediction"])
        self.assertFalse(admission["validation_checked"])

    def test_reader_profile_and_fingerprint_mismatches_remain_hard_errors(self):
        for field in ("reader_checkpoint", "reader_profile_id", "fingerprint"):
            data = asset_data()
            if field == "fingerprint":
                data["provenance"]["input_protocol"]["reader_fingerprint"] = asdict(
                    replace(FINGERPRINT, tokenizer_hash="different-tokenizer"))
            else:
                data[field] = "another-reader-or-profile"
            self.write(data)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reader|profile|fingerprint"):
                load_action_gain_asset(self.settings, reader_fingerprint=FINGERPRINT)
        self.write(asset_data())
        with self.assertRaisesRegex(ValueError, "actual reader checkpoint"):
            load_action_gain_asset(self.settings, reader_fingerprint=replace(FINGERPRINT, model_revision="different"))

    def test_invalid_training_provenance_does_not_sneak_through_passing_metrics(self):
        data = asset_data()
        data["provenance"]["training_split"] = "test"
        self.write(data)
        with patch.object(PlanGainModel, "load", side_effect=AssertionError("must not load test-trained asset")):
            model, admission, _ = load_action_gain_asset(self.settings, reader_fingerprint=FINGERPRINT)
        self.assertIsNone(model)
        self.assertIn("invalid_training_provenance", admission["reasons"])

    def test_rule_mode_does_not_require_or_read_a_gain_asset(self):
        settings = {**self.settings, "action_policy": "rule"}
        del settings["gain_asset"]
        model, admission, metadata = load_action_gain_asset(settings, reader_fingerprint=FINGERPRINT)
        self.assertEqual((model, metadata), (None, None))
        self.assertEqual(admission["policy"], "rule")

    def test_protocol_binds_grounding_dependencies_and_bounded_recovery(self):
        protocol = input_protocol(FINGERPRINT)
        self.assertEqual(validate_input_protocol(protocol, reader_fingerprint=FINGERPRINT), protocol)
        with self.assertRaisesRegex(ValueError, "do not relabel archived inputs"):
            validate_input_protocol(old_input_protocol(), reader_fingerprint=FINGERPRINT)
        for key in ("numeric_selection", "need_dependencies", "terminal_gaps", "read_recovery"):
            changed = deepcopy(protocol)
            del changed[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_input_protocol(changed, reader_fingerprint=FINGERPRINT)
        with patch("controller.read_recovery.MAX_SPLIT_DEPTH", 3), self.assertRaises(ValueError):
            validate_input_protocol(protocol, reader_fingerprint=FINGERPRINT)


if __name__ == "__main__":
    unittest.main()
