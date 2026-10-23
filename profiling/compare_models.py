"""Bounded offline first-token model comparison; never writes a serving artifact.

Only measured calibration features/latencies are read. Paired generations and
their answer lengths are not inputs. The current split is reproduced exactly,
with fixed feature-cell CV and leave-input-length-out diagnostics added.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import random
import statistics
import time

from bench.metrics import percentile
from controller.cost_model import CostModel
from profiling.common import load_measurements
from profiling.fit import _cell_summary, _observation_summary, evaluate, feature_cell_key, validate_timing_scopes

FIRST_PHASES = {"prefill", "restore_prefill", "first_token"}
BASES = ("constant", "suffix", "attention", "host_load", "pressure", "pressure_suffix", "pressure_attention", "pressure_host_load")


def normalized_coordinates(prefix, suffix, load, concurrency):
    total = prefix + suffix
    return total, prefix/max(1, total), load/max(1, total), concurrency


def positive_bases(prefix, suffix, load, concurrency):
    base = (1.0, float(suffix), float(suffix*prefix + suffix*suffix/2), float(load))
    pressure = max(0, concurrency-1)
    return base + tuple(pressure*value for value in base)


def cell_means(rows):
    groups = {}
    for row in rows:
        if row["phase"] in FIRST_PHASES:
            groups.setdefault(feature_cell_key(row), []).append(row)
    cells = []
    for samples in groups.values():
        mean = statistics.mean(r["elapsed_ms"] for r in samples)
        cells.append({**samples[0], "elapsed_ms": mean,
                      "within_cell_p95_ms": percentile([abs(r["elapsed_ms"]-mean) for r in samples], .95),
                      "observation_count": len(samples)})
    return cells


class GridCoordinates:
    """Existing interpolation/error policy, with only the feature map changed."""
    support_definition = "existing_multilinear_or_bounded_nearest_in_n_prefix_fraction_host_fraction_concurrency"
    uncertainty_definition = "original_within_cell_deviation_and_existing_sparse_interpolation_heuristic"

    def __init__(self, rows):
        self.original = CostModel.from_measurements(rows)
        self.samples = [(normalized_coordinates(p.prefix_tokens, p.suffix_tokens, p.host_load_tokens, p.concurrency),
                         p.elapsed_ms, p.error_ms) for p in self.original.prefill_points]

    def _prefill(self, prefix, suffix, load, concurrency):
        return self.original._interpolate(normalized_coordinates(prefix, suffix, load, concurrency), self.samples)

    def description(self):
        return {"points": len(self.samples), "feature_map": ["n=P+U", "P/n", "L/n", "concurrency"],
                "prediction_complexity": "O(measured_cells log measured_cells) worst-case sparse lookup"}


def nonnegative_least_squares(matrix, target):
    """Fixed eight-basis NNLS, with column scaling and no validation tuning."""
    import numpy as np
    from scipy.optimize import nnls
    matrix, target = np.asarray(matrix, dtype=float), np.asarray(target, dtype=float)
    scale = np.maximum(np.max(np.abs(matrix), axis=0), 1)
    coefficients, residual_norm = nnls(matrix/scale, target, maxiter=1000)
    return tuple(float(value) for value in coefficients/scale), float(residual_norm**2)


class PositiveBasis:
    support_definition = "within_training_n_fraction_concurrency_ranges_only_not_an_error_guarantee"
    uncertainty_definition = "training_cell_residual_p95_plus_mean_within_cell_p95_not_a_confidence_interval"

    def __init__(self, rows):
        cells = cell_means(rows)
        features = [positive_bases(r["prefix_tokens"], r["prompt_tokens"]-r["prefix_tokens"],
                                   r.get("host_load_tokens", 0), r["concurrency"]) for r in cells]
        target = [r["elapsed_ms"] for r in cells]
        self.coefficients, self.training_sse = nonnegative_least_squares(features, target)
        coordinates = [normalized_coordinates(r["prefix_tokens"], r["prompt_tokens"]-r["prefix_tokens"],
                                              r.get("host_load_tokens", 0), r["concurrency"]) for r in cells]
        self.ranges = tuple((min(v[i] for v in coordinates), max(v[i] for v in coordinates)) for i in range(4))
        residuals = [abs(sum(a*b for a, b in zip(x, self.coefficients))-y) for x, y in zip(features, target)]
        self.error_ms = percentile(residuals, .95) + statistics.mean(r["within_cell_p95_ms"] for r in cells)
        self.cell_count = len(cells)

    def _prefill(self, prefix, suffix, load, concurrency):
        features = positive_bases(prefix, suffix, load, concurrency)
        predicted = sum(a*b for a, b in zip(features, self.coefficients))
        coordinates = normalized_coordinates(prefix, suffix, load, concurrency)
        covered = all(low <= value <= high for value, (low, high) in zip(coordinates, self.ranges))
        return predicted, self.error_ms, covered

    def description(self):
        return {"training_cells": self.cell_count, "coefficients": dict(zip(BASES, self.coefficients)),
                "training_cell_sse": self.training_sse, "prediction_complexity": "8 basis terms, O(1)",
                "fit_complexity": "scipy.optimize.nnls with eight scaled coefficients; offline only",
                "dependency_versions": {"scipy": version("scipy"), "numpy": version("numpy")},
                "error_ms_heuristic": self.error_ms, "training_coordinate_ranges": self.ranges}


def folds(rows, *, seed=42, cv_seed=2026, cv_folds=5):
    fitted = [r for r in rows if r["phase"] in FIRST_PHASES or (r["phase"] == "decode" and r["output_tokens"] > 1)]
    grouped = {}
    for row in fitted:
        grouped.setdefault(feature_cell_key(row), []).append(row)
    keys = sorted(grouped, key=repr)
    initial = list(keys)
    random.Random(seed).shuffle(initial)
    count = max(1, round(len(initial)*.2)) if len(initial) > 1 else 0
    holdouts = [("original_feature_holdout", initial[:count])]
    cross = list(keys)
    random.Random(cv_seed).shuffle(cross)
    holdouts += [(f"feature_cv_{index}", cross[index::cv_folds]) for index in range(cv_folds)]
    for name, selected in holdouts:
        held_keys = set(selected)
        yield name, [r for key in keys if key not in held_keys for r in grouped[key]], [r for key in keys if key in held_keys for r in grouped[key]]
    for length in sorted({r["prompt_tokens"] for r in fitted}):
        yield f"leave_length_{length}", [r for r in fitted if r["prompt_tokens"] != length], [r for r in fitted if r["prompt_tokens"] == length]


def large_error_diagnostics(model, rows):
    observed = []
    for row in rows:
        predicted, uncertainty, supported = model._prefill(row["prefix_tokens"], row["prompt_tokens"]-row["prefix_tokens"],
                                                          row.get("host_load_tokens", 0), row["concurrency"])
        error = abs(predicted-row["elapsed_ms"])
        observed.append({"error": error, "uncertainty": uncertainty, "supported": supported,
                         "relative_error": error/row["elapsed_ms"] if row["elapsed_ms"] > 0 else None})
    supported = [r for r in observed if r["supported"]]
    def subset(values):
        return {"samples": len(values), "p95_absolute_error_ms": percentile([r["error"] for r in values], .95),
                "max_absolute_error_ms": max((r["error"] for r in values), default=None),
                "errors_over_50ms": sum(r["error"] > 50 for r in values),
                "relative_errors_over_50pct": sum(r["relative_error"] is not None and r["relative_error"] > .5 for r in values),
                "heuristic_uncertainty_capture_fraction": statistics.mean(r["error"] <= r["uncertainty"] for r in values) if values else None}
    return {"all_observations": subset(observed), "supported_only": subset(supported)}


def compare(rows, *, seed=42, cv_seed=2026, cv_folds=5):
    rows = [r for r in rows if r["source"] == "hardware"]
    if not any(r["phase"] in FIRST_PHASES for r in rows):
        raise ValueError("comparison requires real calibration first-token/prefill observations")
    validate_timing_scopes(rows)
    output = []
    for name, training, held in folds(rows, seed=seed, cv_seed=cv_seed, cv_folds=cv_folds):
        first_training = [r for r in training if r["phase"] in FIRST_PHASES]
        first_held = [r for r in held if r["phase"] in FIRST_PHASES]
        if not first_training or not first_held:
            continue
        for method, constructor in (("current_p_u_l", CostModel.from_measurements),
                                    ("normalized_grid", GridCoordinates), ("positive_basis_nnls", PositiveBasis)):
            started = time.perf_counter()
            model = constructor(first_training)
            fit_ms = (time.perf_counter()-started)*1000
            result = evaluate(model, first_held)
            probe_row = first_held[0]
            samples = []
            for _ in range(101):
                started = time.perf_counter()
                model._prefill(probe_row["prefix_tokens"], probe_row["prompt_tokens"]-probe_row["prefix_tokens"],
                               probe_row.get("host_load_tokens", 0), probe_row["concurrency"])
                samples.append((time.perf_counter()-started)*1e6)
            output.append({"fold": name, "method": method,
                "training_feature_cells": len({feature_cell_key(r) for r in first_training}),
                "heldout_feature_cells": len({feature_cell_key(r) for r in first_held}),
                "heldout_lengths": sorted({r["prompt_tokens"] for r in first_held}),
                "validation": result, "large_errors": large_error_diagnostics(model, first_held),
                "support_definition": getattr(model, "support_definition", "production_P_U_L_concurrency_lookup_with_observed_total_input_range"),
                "uncertainty_definition": getattr(model, "uncertainty_definition", "production_within_cell_deviation_and_sparse_interpolation_heuristic"),
                "local_cpu_fit_ms": fit_ms, "local_cpu_prediction_median_us": statistics.median(samples),
                "model_description": model.description() if hasattr(model, "description") else {"points": len(model.prefill_points)}})
    family_summaries = {}
    for family in ("original_feature_holdout", "feature_cv", "leave_length"):
        family_summaries[family] = {}
        for method in ("current_p_u_l", "normalized_grid", "positive_basis_nnls"):
            selected = [result for result in output if result["method"] == method and result["fold"].startswith(family)]
            cases = [case for result in selected for case in result["validation"]["cases"]]
            family_summaries[family][method] = {**_observation_summary(cases), "folds": len(selected),
                "feature_cells_equal_weight": _cell_summary(cases),
                "p95_absolute_error_ms": percentile([case["absolute_error_ms"] for case in cases], .95),
                "max_absolute_error_ms": max((case["absolute_error_ms"] for case in cases), default=None)}
    return {"production_change": False, "paired_generation_data_used": False,
            "selection_status": "offline_comparison_only_requires_independent_validation_before_adoption",
            "seeds": {"original_holdout": seed, "feature_cv": cv_seed}, "cv_folds": cv_folds,
            "fit_target": "first_token/restore-prefill joint latency, not isolated CUDA kernel time",
            "nnls_weighting": "equal training feature cell means, unweighted squared latency error",
            "support_warning": "support definitions differ; none is an empirical error guarantee; compare all-observation error too",
            "positive_uncertainty_warning": "training-residual heuristic shown only to diagnose its failure; production optional artifact uses whole-feature-cell OOF error instead",
            "fold_results": output, "family_summaries": family_summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.measurements)
    rows = [asdict(row) for row in load_measurements(path)]
    result = compare(rows)
    result["source_measurements"] = {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(), "rows": len(rows)}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"output": str(target.resolve()), "fold_model_evaluations": len(result["fold_results"]), "production_change": False}))


if __name__ == "__main__":
    main()
