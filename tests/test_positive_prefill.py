"""Positive first-token estimation is opt-in and keeps the original support gate."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

pytest.importorskip("scipy")

from controller.cost_model import CostModel, LookupPoint, positive_prefill_features
from profiling.compare_models import folds, nonnegative_least_squares
from profiling.fit import feature_cell_key, fit
from profiling.positive_prefill import fit_positive_prefill


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
