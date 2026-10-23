"""Optional candidate lengths change decode cost, without changing error semantics."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.schema import CacheQuote, ConfigFingerprint, PromptCandidate


def model(output_length_model=None):
    return CostModel([LookupPoint(0, 100, 1, 2, error_ms=.1)],
        [DecodePoint(110, 1, 3, error_ms=.01, context_start=101, context_end=120)],
        default_output_tokens=5, p90_output_tokens=10, max_new_tokens=10,
        output_length_model=output_length_model)


def test_candidate_mean_and_p90_affect_cost_and_risk_but_length_error_is_diagnostic():
    class Predictor:
        def predict_prior(self, candidate, session, max_new_tokens, shared_prior):
            assert max_new_tokens == 10
            assert (shared_prior.mean_tokens, shared_prior.p90_tokens) == (5, 10)
            mean, p90 = (3, 5) if candidate.action == "APPEND" else (7, 9)
            return SimpleNamespace(mean_tokens=mean, p90_tokens=p90, error_tokens=500.,
                                   model_kind="candidate", within_training_range=True)
    estimator = model(Predictor())
    results = []
    for action in ("APPEND", "REBUILD"):
        candidate = SimpleNamespace(input_ids=(1,)*100, candidate_id=action, action=action)
        quote = CacheQuote(action, "unused", 100, 99)
        results.append(estimator.predict_native(candidate, quote))
    append, rebuild = results
    assert append.decode_ms == 6 and rebuild.decode_ms == 18
    assert append.risk_cost == 14 and rebuild.risk_cost == 26
    assert append.error_ms == pytest.approx(.12) and rebuild.error_ms == pytest.approx(.16)
    assert append.calibrated and rebuild.calibrated
    assert append.output_length_error_tokens == rebuild.output_length_error_tokens == 500.
    assert append.output_length_model == "candidate" and append.output_length_in_range is True


def test_out_of_range_length_fallback_preserves_shared_timing_calibration():
    class OutOfRange:
        def predict_prior(self, candidate, session, max_new_tokens, shared_prior):
            return SimpleNamespace(mean_tokens=shared_prior.mean_tokens, p90_tokens=shared_prior.p90_tokens,
                error_tokens=700., model_kind="shared_fallback", within_training_range=False)
    candidate = SimpleNamespace(input_ids=(1,)*100, candidate_id="test")
    quote = CacheQuote("test", "unused", 100, 99)
    shared = model().predict_native(candidate, quote)
    fallback = model(OutOfRange()).predict_native(candidate, quote)
    for key in ("cost", "risk_cost", "error_ms", "calibrated", "capacity_risk", "predicted_output_tokens", "p90_output_tokens"):
        assert getattr(fallback, key) == getattr(shared, key)
    assert shared.output_length_model == "shared" and shared.output_length_error_tokens is None
    assert fallback.output_length_model == "shared_fallback" and fallback.output_length_in_range is False


def test_saved_optional_length_model_integrates_with_cost_predictions(tmp_path):
    from controller.output_length import OutputLengthModel
    spec = {"schema_version": 1, "kind": "contextkv_output_length_ridge_v1", "model_kind": "action_only",
        "target": {"fingerprint": asdict(ConfigFingerprint()), "max_new_tokens": 10},
        "feature_schema": {"version": 1, "names": ["is_rebuild"]},
        "regression": {"target_scale": "linear", "centers": [0], "scales": [1],
                       "coefficients": [-2], "intercept": 0},
        "training_range": {"input_tokens": [50, 150]},
        "uncertainty": {"absolute_error_p90_tokens": 300, "positive_residual_p90_tokens": 2}}
    artifact = tmp_path/"length.json"
    OutputLengthModel(spec).save(artifact)
    predictor = OutputLengthModel.load(artifact, expected_fingerprint=ConfigFingerprint(), max_new_tokens=10)
    estimator = model(predictor)
    values = []
    for action in ("APPEND", "REBUILD"):
        candidate = PromptCandidate(action, action, 0, (), (1,)*100, "unused", frozenset(), frozenset())
        values.append(estimator.predict_native(candidate, CacheQuote(action, "unused", 100, 99)))
    assert [value.predicted_output_tokens for value in values] == [5, 3]
    assert [value.p90_output_tokens for value in values] == [7, 5]
    assert [value.decode_ms for value in values] == [12, 6]
    assert all(value.calibrated and value.output_length_model == "action_only" for value in values)
    assert all(value.output_length_error_tokens == 300 and value.error_ms < 1 for value in values)
