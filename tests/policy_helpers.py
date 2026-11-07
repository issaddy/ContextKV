"""Shared CPU protocol fixtures for core property tests."""
from controller.cost_model import CostModel


class SpecifiedGainFixture:
    """Explicit unit-test gains, not a deployed or trained quality model."""
    reader_checkpoint = "reader-fixture"
    reader_profile_id = "profile-fixture"
    asset_id = "test-only-specified-gains"
    def predict(self, features):
        return .6 if features["read_steps"] else .1


def bind_fixture_timing(model, candidates):
    """Explicit protocol bindings for specified test costs, not real profiles."""
    for candidate in candidates:
        kind = candidate.generation_kind
        child = model if kind == "ANSWER" else model.step_models.setdefault(kind, CostModel())
        child.calibration_metadata = {"generation_kind": kind,
            "output_prior_system_prompt": candidate.messages[0].content,
            "step_control_scope": ["rolling_proposals_compilation_start"]}
        child.step_output_priors[f"{kind}:{candidate.output_budget}"] = {
            "mean_tokens": 1, "p90_tokens": 1, "mean_control_ms": 0,
            "mean_remaining_control_ms": 0}
