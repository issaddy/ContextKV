"""Positive first-token estimation is opt-in and keeps the original support gate."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("scipy")

from controller.cost_model import CostModel, LookupPoint, positive_prefill_features
from profiling.compare_models import folds, nonnegative_least_squares
from profiling.fit import feature_cell_key, fit
from profiling.positive_prefill import conditional_uncertainty, fit_positive_prefill, main as uncertainty_main


def observations():
    coefficients = (3, .005, .00000012, .0008, 2, .002, .00000002, .0002)
    rows = []
    for total in (2048, 4096, 8192):
        for fraction in (0, .5, .875):
            prefix = int(total*fraction)
            for load in sorted({0, prefix}):
                for concurrency in (1, 4):
                    measured = sum(a*b for a, b in zip(positive_prefill_features(prefix, total-prefix, load, concurrency), coefficients))
                    for index, noise in enumerate((-1, 1)):
                        rows.append({"phase": "first_token", "prompt_tokens": total, "prefix_tokens": prefix,
                            "host_load_tokens": load, "concurrency": concurrency, "output_tokens": 128 if index == 0 else 512,
                            "source": "hardware", "elapsed_ms": measured+noise, "timing_scope": "server_request"})
    return rows


def test_scipy_nnls_and_feature_cell_splits_are_deterministic_without_repeat_leakage():
    coefficients, residual = nonnegative_least_squares([[1, 0], [0, 1], [1, 1]], [2, 3, 5])
    assert coefficients == pytest.approx((2, 3)) and residual < 1e-20
    coefficients, residual = nonnegative_least_squares([[1], [2]], [-1, -2])
    assert coefficients == (0,) and residual == pytest.approx(5)
    rows = observations()
    for name, training, held in folds(rows):
        assert {feature_cell_key(r) for r in training}.isdisjoint(feature_cell_key(r) for r in held)
        if name.startswith("leave_length_"):
            assert {r["prompt_tokens"] for r in training}.isdisjoint(r["prompt_tokens"] for r in held)


def test_oof_errors_are_from_held_out_feature_cells_and_include_request_variation(monkeypatch):
    import profiling.positive_prefill as module
    original = module.fit_coefficients
    trained_keys = []
    def record(cells):
        trained_keys.append({cell["key"] for cell in cells})
        return original(cells)
    monkeypatch.setattr(module, "fit_coefficients", record)
    spec = fit_positive_prefill(observations())
    assert spec["uncertainty_sufficient"]
    assert spec["error_p99_ms"] >= spec["error_p95_ms"] >= .99
    assert spec["validation_protocol"]["oof_feature_cells"] == 30
    assert spec["training_cell_sse"] < 1e-12
    for held in spec["validation_protocol"]["oof_predictions"]:
        assert tuple(held["feature_cell"]) not in trained_keys[held["fold"]+1]
        assert held["observations"] == 2  # different output caps stay together


def test_positive_model_preserves_lookup_support_and_runs_without_scipy_at_inference(tmp_path):
    rows = observations()
    spec = fit_positive_prefill(rows)
    lookup = CostModel.from_measurements(rows)
    positive = CostModel.from_measurements(rows, prefill_estimator="positive_basis", positive_prefill=spec)
    targets = [(p.prefix_tokens, p.suffix_tokens, p.host_load_tokens, p.concurrency) for p in lookup.prefill_points]
    rng = random.Random(7)
    targets += [(rng.randrange(0, 9000), rng.randrange(1, 9000), rng.randrange(0, 7000), rng.choice((1, 2, 3, 4, 5))) for _ in range(60)]
    for target in targets:
        assert positive._prefill(*target)[2] == lookup._prefill(*target)[2]
    path = tmp_path / "positive.json"
    positive.save(path)
    loaded = CostModel.load(path)
    assert loaded.prefill_estimator == "positive_basis"
    assert loaded._prefill(*targets[0]) == positive._prefill(*targets[0])
    code = """import sys
from controller.cost_model import CostModel
model=CostModel.load(sys.argv[1])
assert 'scipy' not in sys.modules and 'numpy' not in sys.modules
assert model._prefill(0,2048,0,1)[2]
print('stdlib-only-inference-ok')
"""
    output = subprocess.check_output([sys.executable, "-S", "-c", code, str(path)], text=True,
                                     cwd=Path(__file__).resolve().parents[1])
    assert output.strip() == "stdlib-only-inference-ok"


def test_small_dataset_cannot_claim_positive_calibration_and_old_artifacts_remain_lookup(tmp_path):
    rows = observations()[:2]
    spec = fit_positive_prefill(rows)
    assert not spec["uncertainty_sufficient"] and spec["error_p99_ms"] is None
    model = CostModel.from_measurements(rows, prefill_estimator="positive_basis", positive_prefill=spec)
    estimate, error, supported = model._prefill(0, 2048, 0, 1)
    assert not supported and error >= estimate
    tiny = rows + [{**rows[0], "phase": "decode", "output_tokens": 4, "elapsed_ms": 3}]
    _, metadata = fit(tiny, prefill_estimator="positive_basis")
    if metadata["validation"]["evaluated_estimator"] != "positive_basis":
        assert metadata["validation"]["status"] == "insufficient_estimator_training_cells"
    legacy = {"settings": {}, "prefill_points": [asdict(LookupPoint(0, 100, 1, 2))], "decode_points": []}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    assert CostModel.load(path).prefill_estimator == "lookup"


def test_fit_positive_option_preserves_default_and_reports_outer_holdout_protocol():
    rows = observations()
    lookup, _ = fit(rows)
    positive, metadata = fit(rows, prefill_estimator="positive_basis")
    assert lookup.prefill_estimator == "lookup" and positive.prefill_estimator == "positive_basis"
    assert metadata["validation"]["requested_estimator"] == metadata["validation"]["evaluated_estimator"] == "positive_basis"
    assert metadata["validation"]["heldout_cells"] == 6
    assert positive.decode_points == lookup.decode_points


def test_both_estimators_reject_total_length_outside_original_sparse_measurements():
    # The real profiler uses n=2048/4096/8192 with page-rounded 0/.5/.9 prefixes.
    rows = []
    for total in (2048,4096,8192):
        for fraction in (0,.5,.9):
            prefix = int(total*fraction)//64*64
            for concurrency in (1,4):
                for noise in (-.1,.1):
                    rows.append(dict(phase="first_token",prompt_tokens=total,prefix_tokens=prefix,
                        host_load_tokens=0,concurrency=concurrency,output_tokens=128,source="hardware",
                        elapsed_ms=1+.01*(total-prefix)+noise,timing_scope="server_request"))
    lookup = CostModel.from_measurements(rows)
    positive = CostModel.from_measurements(rows,prefill_estimator="positive_basis",
                                          positive_prefill=fit_positive_prefill(rows))
    outside = (6000,4000,0,1)  # Within individual P/U axes, but total n=10000.
    samples = [((p.prefix_tokens,p.suffix_tokens,p.host_load_tokens,p.concurrency),p.elapsed_ms,p.error_ms)
               for p in lookup.prefill_points]
    old_diagnostic = lookup._interpolate(outside,samples)
    assert old_diagnostic[2]  # Demonstrates the original sparse-axis loophole.
    assert lookup._prefill(*outside)[:2] == old_diagnostic[:2]
    for model in (lookup,positive):
        assert not model._prefill(*outside)[2]
        assert not model._prefill(1000,1000,0,1)[2]  # Below measured min n=2048.
        assert model._prefill(4096,4096,0,1)[2]  # Exact upper boundary n=8192.
        assert model._prefill(2000,4000,0,1)[2]  # Sparse interior n=6000.


def test_prefill_phase_aliases_do_not_inflate_independent_cells_or_leak_across_folds():
    rows = [dict(phase=phase,prompt_tokens=1000+i*100,prefix_tokens=0,host_load_tokens=0,
                 concurrency=1,output_tokens=8,source="hardware",elapsed_ms=10.,timing_scope="server_phase")
            for i in range(5) for phase in ("prefill","restore_prefill")]
    spec = fit_positive_prefill(rows)
    assert spec["training_feature_cells"] == 5
    assert not spec["uncertainty_sufficient"] and spec["error_p99_ms"] is None
    def coordinates(row):
        return (row["prefix_tokens"],row["prompt_tokens"]-row["prefix_tokens"],
                row["host_load_tokens"],row["concurrency"])
    for _,training,held in folds(rows):
        assert set(map(coordinates,training)).isdisjoint(map(coordinates,held))


@pytest.mark.parametrize("family",["first_token","decode"])
def test_fitting_and_comparison_reject_incompatible_timing_scopes(family):
    from profiling.compare_models import compare
    rows = observations()
    if family == "first_token":
        rows.append({**rows[0],"timing_scope":"cuda_event"})
    else:
        rows.extend([{**rows[0],"phase":"decode","timing_scope":scope}
                     for scope in ("server_request","cuda_event")])
    for operation in (fit,fit_positive_prefill,compare):
        with pytest.raises(ValueError,match="timing scope"):
            operation(rows)


def test_mock_fitted_coefficients_cannot_become_hardware_by_filtering_points_on_load(tmp_path):
    rows = [dict(phase="first_token",prompt_tokens=1000+i*100,prefix_tokens=0,host_load_tokens=0,
                 concurrency=1,output_tokens=8,source=source,elapsed_ms=latency,timing_scope="server_request")
            for i in range(10) for source,latency in (("hardware",10.),("mock",1000.))]
    model,metadata = fit(rows,allow_mock=True,prefill_estimator="positive_basis")
    assert model.positive_prefill["fit_sources"] == ["hardware","mock"]
    assert model._prefill(0,1000,0,1)[2]  # Explicit diagnostic opt-in only.
    path = tmp_path/"mixed.json"
    model.save(path,metadata=metadata)
    restored = CostModel.load(path)
    estimate,error,supported = restored._prefill(0,1000,0,1)
    assert estimate == pytest.approx(505) and error == pytest.approx(495)
    assert not restored.allow_mock_calibration and not supported


def test_legacy_positive_sources_need_explicit_hardware_metadata(tmp_path):
    rows = observations()
    model,metadata = fit(rows,prefill_estimator="positive_basis")
    path = tmp_path/"legacy-positive.json"
    model.save(path,metadata=metadata)
    payload = json.loads(path.read_text())
    payload["positive_prefill"].pop("fit_sources")
    path.write_text(json.dumps(payload))
    assert CostModel.load(path)._prefill(0,2048,0,1)[2]
    payload["metadata"].pop("measurement_sources")
    path.write_text(json.dumps(payload))
    assert not CostModel.load(path)._prefill(0,2048,0,1)[2]


def uncertainty_predictions(*, counts=(13, 9), concurrencies=(1, 4)):
    return [{"feature_cell": ["first_token", 1000+i*100, 100, 100 if host else 0, concurrency, "server_request"],
             "max_absolute_error_ms": 7. if concurrency == 1 else 500., "observations": 1000}
            for concurrency in concurrencies for host, count in enumerate(counts) for i in range(count)]


def uncertainty_spec(predictions):
    return {"coefficients": [1., 0., 0., 0., 0., 0., 0., 0.], "fit_sources": ["hardware"],
            "uncertainty_sufficient": True, "error_p95_ms": 90., "error_p99_ms": 100.,
            "conditional_uncertainty": conditional_uncertainty(predictions),
            "validation_protocol": {"oof_predictions": predictions}}


def test_conditional_uncertainty_counts_cells_not_repeats_and_uses_coarse_concurrency():
    predictions = uncertainty_predictions()
    table = conditional_uncertainty(predictions * 100)
    narrow = [row for row in table["strata"] if row["host_load"] is not None]
    broad = [row for row in table["strata"] if row["host_load"] is None]
    assert sorted(row["feature_cells"] for row in narrow) == [9, 9, 13, 13]
    assert all(not row["sufficient"] for row in narrow)
    assert [row["feature_cells"] for row in broad] == [22, 22]
    assert all(row["sufficient"] for row in broad)
    spec = uncertainty_spec(predictions)
    global_model = CostModel(prefill_estimator="positive_basis", positive_prefill=spec)
    conditional = CostModel(prefill_estimator="positive_basis", positive_prefill=spec, uncertainty_mode="conditional")
    assert global_model._positive_error(1, 0) == 100.
    assert conditional._positive_error(1, 0) == conditional._positive_error(1, 100) == 7.
    assert conditional._positive_error(4, 0) == 500.  # A conditional error can also be larger.
    assert conditional._positive_error(2, 0) == 100.  # No observed c=2 stratum.
    assert conditional._positive_error(3, 100) == 100.
    alias_rows = [{**row, "feature_cell": [phase, *row["feature_cell"][1:]]}
                  for row in predictions for phase in ("prefill", "restore_prefill")]
    aliases = conditional_uncertainty(alias_rows)
    assert [row["feature_cells"] for row in aliases["strata"]] == [row["feature_cells"] for row in table["strata"]]


def test_conditional_uncertainty_missing_host_state_or_insufficient_cells_falls_back():
    for counts, load in (((19, 0), 0), ((20, 0), 100)):
        spec = uncertainty_spec(uncertainty_predictions(counts=counts, concurrencies=(1,)))
        model = CostModel(prefill_estimator="positive_basis", positive_prefill=spec, uncertainty_mode="conditional")
        assert model._positive_error(1, load) == 100.
    spec.pop("conditional_uncertainty")
    assert CostModel(prefill_estimator="positive_basis", positive_prefill=spec,
                     uncertainty_mode="conditional")._positive_error(1, 0) == 100.


def test_conditional_uncertainty_uses_true_quote_load_even_in_no_host_cost_ablation():
    from controller.schema import CacheQuote
    predictions = uncertainty_predictions(counts=(20, 20), concurrencies=(1,))
    for row in predictions:
        row["max_absolute_error_ms"] = 8. if row["feature_cell"][3] else 2.
    model = CostModel(prefill_estimator="positive_basis", positive_prefill=uncertainty_spec(predictions),
                     uncertainty_mode="conditional", include_host_cost=False, include_decode_cost=False,
                     default_output_tokens=1, p90_output_tokens=1)
    candidate = SimpleNamespace(input_ids=(1,)*1000, candidate_id="test")
    quote = CacheQuote("test", "unused", 1000, 999, recoverable_prefix_tokens=64,
                       host_only_load_tokens=64, native_load_estimate=64)
    assert model._positive_error(1, 0) == 2.
    assert model.predict_native(candidate, quote).error_ms == 8.


def test_conditional_artifact_reuses_frozen_oof_and_preserves_default_and_override(tmp_path):
    spec = uncertainty_spec(uncertainty_predictions())
    model = CostModel(prefill_estimator="positive_basis", positive_prefill=spec)
    source, target = tmp_path/"global.json", tmp_path/"conditional.json"
    model.save(source)
    original = source.read_bytes()
    uncertainty_main(["--input-artifact", str(source), "--output", str(target)])
    assert source.read_bytes() == original
    loaded = CostModel.load(target, uncertainty_mode=None)
    assert loaded.uncertainty_mode == "conditional"
    assert loaded.positive_prefill["coefficients"] == spec["coefficients"]
    assert loaded.positive_prefill["validation_protocol"] == spec["validation_protocol"]
    assert loaded._positive_error(1, 0) == 7.
    assert CostModel.load(target, uncertainty_mode="global")._positive_error(1, 0) == 100.
    assert CostModel.load(source).uncertainty_mode == "global"
    subprocess.check_call([sys.executable, "-S", "-c",
        "import sys; from controller.cost_model import CostModel; "
        "assert CostModel.load(sys.argv[1])._positive_error(1, 0) == 7.; "
        "assert 'scipy' not in sys.modules and 'numpy' not in sys.modules", str(target)],
        cwd=Path(__file__).resolve().parents[1])
    with pytest.raises(ValueError, match="uncertainty_mode"):
        CostModel(uncertainty_mode="unknown")
