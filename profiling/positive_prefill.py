"""Optional SciPy NNLS fitter; serving uses only saved coefficients and geometry."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
import statistics

from bench.metrics import percentile
from controller.cost_model import CostModel, positive_prefill_features
from profiling.fit import feature_cell_key, validate_timing_scopes

BASES = ("constant", "suffix", "attention", "host_load", "pressure", "pressure_suffix", "pressure_attention", "pressure_host_load")
FIRST_PHASES = {"prefill", "restore_prefill", "first_token"}


def conditional_uncertainty(predictions, *, minimum_feature_cells=20):
    """Pool OOF cell errors by observable load, never by measured pair outcomes.

    Twenty distinct feature cells are required before using a conditional tail.
    Narrow host strata fall back to the same concurrency, then the global error.
    These are empirical maxima quantiles, not coverage/confidence guarantees.
    """
    groups = {}
    for row in predictions:
        raw_key = row["feature_cell"]
        key = ("restore_prefill" if raw_key[0] in {"prefill", "restore_prefill"} else raw_key[0], *raw_key[1:])
        concurrency, host_load = key[4], key[3] > 0
        for group in ((concurrency, host_load), (concurrency, None)):
            cells = groups.setdefault(group, {})
            cells[key] = max(cells.get(key, 0), row["max_absolute_error_ms"])
    strata = []
    for (concurrency, host_load), cells in sorted(groups.items(), key=lambda item: repr(item[0])):
        errors = list(cells.values())
        strata.append({"concurrency": concurrency, "host_load": host_load,
            "host_load_states": sorted({key[3] > 0 for key in cells}),
            "feature_cells": len(cells), "sufficient": len(cells) >= minimum_feature_cells,
            "error_p95_ms": percentile(errors, .95), "error_p99_ms": percentile(errors, .99)})
    return {"kind": "oof_concurrency_host_hierarchy_v1", "minimum_feature_cells": minimum_feature_cells,
            "matching": "exact_concurrency_and_observed_host_load_state",
            "fallback_order": ["concurrency_and_host_load", "concurrency", "global"],
            "definition": "P95/P99 of per-feature-cell maximum absolute OOF request error; empirical, not a confidence guarantee",
            "strata": strata}


def calibration_cells(rows):
    groups = {}
    for row in rows:
        if row["phase"] in FIRST_PHASES:
            groups.setdefault(feature_cell_key(row), []).append(row)
    cells = []
    for key, observations in sorted(groups.items(), key=lambda item: repr(item[0])):
        first = observations[0]
        mean = statistics.mean(r["elapsed_ms"] for r in observations)
        cells.append({"key": key, "rows": observations, "mean_ms": mean,
                      "features": positive_prefill_features(first["prefix_tokens"], first["prompt_tokens"]-first["prefix_tokens"],
                                                             first.get("host_load_tokens", 0), first["concurrency"])})
    return cells


def fit_coefficients(cells):
    try:
        import numpy as np
        from scipy.optimize import nnls
    except ImportError as exc:
        raise RuntimeError("positive_basis fitting requires the optional calibration extra (NumPy and SciPy); serving does not") from exc
    matrix = np.asarray([cell["features"] for cell in cells], dtype=float)
    target = np.asarray([cell["mean_ms"] for cell in cells], dtype=float)
    scales = np.maximum(np.max(np.abs(matrix), axis=0), 1)
    coefficients, residual_norm = nnls(matrix/scales, target, maxiter=1000)
    return tuple(float(x) for x in coefficients/scales), float(residual_norm**2)


def fit_positive_prefill(rows, *, seed=2026, folds=5, minimum_oof_cells=10):
    rows = list(rows)
    validate_timing_scopes(rows)
    cells = calibration_cells(rows)
    if not cells:
        raise ValueError("positive prefill fitting requires first-token or prefill observations")
    coefficients, sse = fit_coefficients(cells)
    sufficient = len(cells) >= minimum_oof_cells and folds >= 2
    predictions = []
    if sufficient:
        indexes = list(range(len(cells)))
        random.Random(seed).shuffle(indexes)
        for fold in range(folds):
            held = set(indexes[fold::folds])
            training = [cell for index, cell in enumerate(cells) if index not in held]
            weights, _ = fit_coefficients(training)
            lookup = CostModel.from_measurements([row for cell in training for row in cell["rows"]], allow_mock=True)
            for index in sorted(held):
                cell = cells[index]
                predicted = sum(x*w for x, w in zip(cell["features"], weights))
                errors = [abs(predicted-row["elapsed_ms"]) for row in cell["rows"]]
                first = cell["rows"][0]
                _, _, supported = lookup._prefill(first["prefix_tokens"], first["prompt_tokens"]-first["prefix_tokens"],
                                                  first.get("host_load_tokens", 0), first["concurrency"])
                predictions.append({"feature_cell": list(cell["key"]), "fold": fold,
                    "observations": len(errors), "predicted_ms": predicted, "measured_mean_ms": cell["mean_ms"],
                    "absolute_error_of_mean_ms": abs(predicted-cell["mean_ms"]),
                    "mean_absolute_error_ms": statistics.mean(errors), "max_absolute_error_ms": max(errors),
                    "supported_by_training_lookup": supported})
    # One conservative error observation per held-out feature cell. Max within
    # each cell preserves observed request variability without giving a c=4 cell
    # four votes. These empirical quantiles are not confidence guarantees.
    maxima = [row["max_absolute_error_ms"] for row in predictions]
    return {"kind": "positive_prefill_nnls_v1", "basis_names": list(BASES), "coefficients": list(coefficients),
            "training_feature_cells": len(cells), "training_cell_sse": sse,
            "fit_sources": sorted({row["source"] for cell in cells for row in cell["rows"]}),
            "fit_weighting": "equal_feature_cell_means", "solver": "scipy.optimize.nnls",
            "dependency_versions": {"scipy": version("scipy"), "numpy": version("numpy")},
            "uncertainty_sufficient": sufficient, "error_p95_ms": percentile(maxima, .95),
            "error_p99_ms": percentile(maxima, .99),
            "conditional_uncertainty": conditional_uncertainty(predictions),
            "uncertainty_definition": "P95/P99 of per-feature-cell maximum absolute OOF request error; not a confidence guarantee",
            "support_definition": "shared_P_U_L_concurrency_lookup_geometry_and_observed_total_input_range",
            "validation_protocol": {"kind": "out_of_fold_whole_first_token_feature_cells", "seed": seed,
                "folds": folds, "minimum_feature_cells": minimum_oof_cells,
                "oof_feature_cells": len(predictions), "status": "available" if sufficient else "insufficient_feature_cells",
                "lookup_support_coverage": statistics.mean(row["supported_by_training_lookup"] for row in predictions) if predictions else None,
                "oof_predictions": predictions}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derive opt-in conditional uncertainty from an existing positive artifact's OOF cells")
    parser.add_argument("--input-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("choose a new output artifact; the original calibration stays frozen")
    model = CostModel.load(args.input_artifact)
    if model.prefill_estimator != "positive_basis":
        raise ValueError("conditional OOF uncertainty requires a positive_basis artifact")
    predictions = model.positive_prefill.get("validation_protocol", {}).get("oof_predictions", [])
    model.positive_prefill["conditional_uncertainty"] = conditional_uncertainty(predictions)
    model.uncertainty_mode = "conditional"
    model.calibration_metadata["conditional_uncertainty_update"] = {
        "source_artifact": str(Path(args.input_artifact).resolve()),
        "minimum_feature_cells": 20, "source": "existing_whole_feature_cell_oof_predictions",
        "new_coefficients_fitted": False, "generation_outcomes_used": False,
        "status": "development_exploration_not_independent_validation"}
    model.save(target)
    print(json.dumps({"artifact": str(target), "conditional_uncertainty": model.positive_prefill["conditional_uncertainty"]}, indent=2))


if __name__ == "__main__":
    main()
