"""Explicit learned policies keep reader and measured-cost compatibility strict."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from controller.action_proposals import BOUNDED_CONTINUATION, finite_action_domain
from controller.context_budget import COMPLETION_RESERVE_SCOPE
from controller.plan_gain_protocol import input_protocol
from controller.plan_policy import (NATIVE_COST_FEATURE_NAMES, NATIVE_COST_SCHEMA,
    OBSERVED_FEATURE_NAMES, OBSERVED_FEATURE_SCHEMA)
from controller.plan_profiles import load_action_gain_asset, load_rolling_runtime
from controller.schema import ConfigFingerprint
from training.plan_gain_completion import COST_TARGET_VERSION
from training.train_plan_gain import COST_FIELDS, PlanGainModel


FINGERPRINT = ConfigFingerprint(model_revision="reader-v2")


def asset_data():
    asset = "test-gain-asset"
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
            "completion_reserve_scope": COMPLETION_RESERVE_SCOPE, "cost_target_version": COST_TARGET_VERSION},
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

    def test_default_and_explicit_rule_runtime_need_no_gain_asset(self):
        for policy in (None, "rule"):
            settings = {key: value for key, value in self.settings.items() if key != "gain_asset"}
            if policy is not None:
                settings["action_policy"] = policy
            with self.subTest(policy=policy):
                runtime = load_rolling_runtime(settings, template=SimpleNamespace(fingerprint=FINGERPRINT),
                    store=None, sampling={}, answer_system_prompt="answer", engine_base_url="http://unused")
                self.assertIsNone(runtime.policy.gain_model)
                self.assertTrue(runtime.joint_context)
                self.assertEqual(runtime.native.fingerprint, FINGERPRINT)
                self.assertFalse(runtime.native.paid.started)

    def test_explicit_learned_runtime_loads_current_bound_asset(self):
        self.write(asset_data())
        runtime = load_rolling_runtime({**self.settings, "action_policy": "learned"},
            template=SimpleNamespace(fingerprint=FINGERPRINT), store=None, sampling={},
            answer_system_prompt="answer", engine_base_url="http://unused")
        self.assertIsInstance(runtime.policy.gain_model, PlanGainModel)
        self.assertEqual(runtime.policy.gain_model.asset_id, "test-gain-asset")

    def test_explicit_learned_rejects_incompatible_assets_without_fallback(self):
        changes = (
            ("reader checkpoint", ("reader_checkpoint",), "another-reader"),
            ("reader profile", ("reader_profile_id",), "another-profile"),
            ("input semantics", ("provenance", "input_protocol"), old_input_protocol()),
            ("reader fingerprint", ("provenance", "input_protocol", "reader_fingerprint"),
                asdict(replace(FINGERPRINT, tokenizer_hash="different-tokenizer"))),
            ("cost target", ("cost_provenance", "cost_target_version"), "successful_only"),
            ("training split", ("provenance", "training_split"), "test"),
        )
        for name, keys, value in changes:
            data = asset_data()
            target = data
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            self.write(data)
            with self.subTest(contract=name), self.assertRaises(ValueError):
                load_action_gain_asset({**self.settings, "action_policy": "learned"},
                    reader_fingerprint=FINGERPRINT)

    def test_rule_runtime_still_binds_actual_reader_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "actual reader checkpoint"):
            load_rolling_runtime(self.settings,
                template=SimpleNamespace(fingerprint=replace(FINGERPRINT, model_revision="different")),
                store=None, sampling={}, answer_system_prompt="answer", engine_base_url="http://unused")


if __name__ == "__main__":
    unittest.main()
