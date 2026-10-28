from dataclasses import replace
from itertools import product

import pytest

from controller.cost_model import CostModel, LookupPoint
from controller.schema import CacheQuote, CacheQuoteBatch, PromptCandidate
from controller.templates import token_hash
from profiling.fit import evaluate, fit, output_priors


def row(prompt=100, *, phase="first_token", output=4, elapsed=10, source="hardware"):
    return dict(phase=phase, prompt_tokens=prompt, prefix_tokens=0, output_tokens=output,
                concurrency=1, host_load_tokens=0, elapsed_ms=elapsed, source=source)


def candidate_and_quote():
    ids = tuple(range(100))
    candidate = PromptCandidate("r", "REBUILD", 0, (), ids, token_hash(ids), frozenset(), frozenset())
    return candidate, CacheQuote("r", candidate.input_hash, 100, 99)


def test_interpolation_is_bounded_by_measured_support():
    points = [LookupPoint(prefix, suffix, 1, prefix+2*suffix)
              for prefix, suffix in product((0, 100), (20, 40))]
    model = CostModel(points)
    cost, _, calibrated = model._prefill(50, 30, 0, 1)
    assert calibrated and cost == 110
    assert not model._prefill(101, 30, 0, 1)[2]
    assert not model._prefill(50, 30, 1, 1)[2]


def test_decode_measurement_covers_actual_interval_and_does_not_double_charge_queue():
    candidate, quote = candidate_and_quote()
    model = CostModel.from_measurements([row(), row(phase="decode", elapsed=6)],
                                       default_output_tokens=4, p90_output_tokens=4)
    prediction = model.predict_native(candidate, quote, queue_ms=999, control_ms=2)
    assert prediction.calibrated
    assert prediction.cost == 18
    assert prediction.queue_ms == 0 and prediction.decode_ms == 6
    assert model._decode(101, 1)[2] and model._decode(103, 1)[2]
    assert not model._decode(104, 1)[2]


def test_host_and_decode_control_ablations_remove_only_selected_terms():
    candidate, quote = candidate_and_quote()
    quote = replace(quote, ready_gpu_prefix_tokens=20, recoverable_prefix_tokens=80,
                    host_only_load_tokens=60, native_load_estimate=60)
    points = (LookupPoint(80, 20, 1, 5), LookupPoint(80, 20, 1, 15, host_load_tokens=60))
    all_terms = CostModel(points, default_output_tokens=1, p90_output_tokens=1)
    no_host = CostModel(points, default_output_tokens=1, p90_output_tokens=1, include_host_cost=False)
    no_control = CostModel(points, default_output_tokens=1, p90_output_tokens=1, include_control_cost=False)
    assert all_terms.predict_native(candidate, quote, control_ms=2).cost == 17
    assert no_host.predict_native(candidate, quote, control_ms=2).cost == 7
    assert no_control.predict_native(candidate, quote, control_ms=2).cost == 15
    no_decode = CostModel(points, include_decode_cost=False)
    assert no_decode.predict_native(candidate, quote).calibrated
    assert no_decode.predict_native(candidate, quote).decode_ms == 0
    batch = CacheQuoteBatch("q", 1, 0, "restart", "config", 25, 0, (quote,))
    assert all_terms.predict_native(candidate, quote, batch=batch).capacity_risk


def test_frozen_artifact_preserves_points_but_not_runtime_adaptation(tmp_path):
    model = CostModel.from_measurements([row(), row(phase="decode", elapsed=6)],
                                       default_output_tokens=4, p90_output_tokens=4)
    candidate, quote = candidate_and_quote()
    before = model.predict_native(candidate, quote)
    model.observe(before, before.cost+100)
    assert model.predict_native(candidate, quote).cost > before.cost
    path = tmp_path / "model.json"
    model.save(path, metadata={"split": "development"})
    frozen = CostModel.load(path)
    assert frozen.predict_native(candidate, quote) == before
    mock = CostModel.from_measurements([row(source="mock")], allow_mock=True)
    mock.save(path)
    assert not CostModel.load(path).predict_native(candidate, quote).calibrated


def test_fit_holds_out_whole_cells_and_ignores_control_as_validation_target():
    rows = [row(prompt, elapsed=prompt/10) for prompt in (100, 200, 300) for _ in range(3)]
    rows += [row(phase="control", elapsed=4), row(phase="D2H", elapsed=8)]
    model, metadata = fit(rows, holdout_fraction=.3)
    assert len(model.prefill_points) == 3
    validation = metadata["validation"]
    assert validation["training_cells"] == 2 and validation["heldout_cells"] == 1
    assert validation["samples"] == 3
    with pytest.raises(ValueError, match="no first-token"):
        fit([row(phase="control")])
    with pytest.raises(ValueError, match="one timing scope"):
        fit([row(), row(phase="prefill")])
    _, single = fit([row(), row()])
    assert single["validation"]["status"] == "insufficient_independent_cells"
    assert single["validation"]["samples"] == 0


def test_output_prior_does_not_take_failed_or_reference_output_lengths():
    report = {"turns": [{"status": "success", "result": {"output_tokens": 10}},
                        {"status": "success", "result": {"output_tokens": 30}},
                        {"status": "failed", "result": {"output_tokens": 1000}}]}
    assert output_priors(report) == (20, 30)


def test_holdout_groups_match_cost_model_features_not_unmodeled_labels():
    rows = [row(output=n) for n in (4, 8, 16)]
    rows += [dict(row(phase="decode", output=8), prefix_tokens=p, host_load_tokens=p)
             for p in (0, 20, 40)]
    _, metadata = fit(rows, holdout_fraction=.5)
    validation = metadata["validation"]
    assert validation["training_cells"] == validation["heldout_cells"] == 1
    assert validation["samples"] == 3


def test_validation_separates_phase_units_supported_errors_and_equal_feature_cells():
    class ObservedModel:
        def _prefill(self, prefix, suffix, load, concurrency):
            return 8, 0, suffix <= 100

        def _decode(self, context, concurrency):
            return 2, 0, context <= 150

    rows = [row(100, elapsed=10), row(200, elapsed=20)]
    # Five prior cache states repeat the same decode features, with one other
    # independent context. They must not become five independent feature cells.
    rows += [dict(row(100, phase="decode", elapsed=3), prefix_tokens=p, host_load_tokens=p)
             for p in (0, 20, 40, 60, 80)]
    rows += [row(200, phase="decode", elapsed=30)]
    result = evaluate(ObservedModel(), rows)
    assert result["aggregation"] == "mixed_phase_diagnostic"
    assert result["samples"] == 8 and result["supported_samples"] == 6
    first = result["by_phase"]["first_token"]
    decode = result["by_phase"]["decode"]
    assert first["unit"] == "ms" and decode["unit"] == "ms_per_output_token"
    assert first["mae_ms"] == 7 and first["supported_only"]["mae_ms"] == 2
    assert first["coverage"] == .5
    assert decode["mae_ms"] == pytest.approx(13/6)
    assert decode["supported_only"]["mae_ms"] == 1 and decode["coverage"] == pytest.approx(5/6)
    cells = decode["feature_cells_equal_weight"]
    assert cells["cells"] == 2 and cells["supported_cells"] == 1
    assert cells["mae_ms"] == 4.5 and cells["coverage"] == .5
    assert cells["supported_only"]["mae_ms"] == 1


def test_phase_summary_handles_empty_support_and_zero_denominator_without_dropping_absolute_error():
    class UnsupportedModel:
        def _prefill(self, *args):
            return 2, 0, False

    result = evaluate(UnsupportedModel(), [row(elapsed=0)])
    phase = result["by_phase"]["first_token"]
    assert phase["mae_ms"] == 2 and phase["mean_relative_error"] is None
    assert phase["relative_error_samples"] == 0
    assert phase["coverage"] == 0 and phase["supported_only"]["samples"] == 0
    assert phase["supported_only"]["mae_ms"] is None
    assert phase["feature_cells_equal_weight"]["supported_only"]["cells"] == 0
    empty = evaluate(UnsupportedModel(), [])
    assert empty["samples"] == 0 and empty["coverage"] is None and empty["by_phase"] == {}


def test_decode_support_changes_when_sparse_distance_setting_changes():
    from controller.cost_model import DecodePoint
    model = CostModel(decode_points=[DecodePoint(100, 1, 2.), DecodePoint(200, 4, 5.)])
    first = model._decode(150, 1)
    assert first[2] is True
    model.max_normalized_distance = .1
    changed = model._decode(150, 1)
    assert changed[:2] == first[:2] and changed[2] is False
    model.max_normalized_distance = .5
    assert model._decode(150, 1) == first
