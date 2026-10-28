"""Fit and evaluate a frozen cost artifact from measured development samples."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics

from controller.cost_model import CostModel
from controller.sampling import bound_recipe
from profiling.common import load_measurements


def measurement_provenance(rows, measurements_path, *, profile_manifest_path=None, development_report_path=None):
    """Bind an offline artifact to original files once; no request-path hashing."""
    phases = {}
    sizes = ("prompt_tokens", "prefix_tokens", "output_tokens", "host_load_tokens", "concurrency", "transfer_bytes")
    for phase in sorted({row["phase"] for row in rows}):
        selected = [r for r in rows if r["phase"] == phase]
        phases[phase] = {"sample_count": len(selected),
                         "timing_scopes": sorted({r.get("timing_scope", "unspecified") for r in selected}),
                         "cache_states": sorted({r.get("cache_state", "unspecified") for r in selected}),
                         "ranges": {key: {"min": min(r.get(key, 0) for r in selected),
                                          "max": max(r.get(key, 0) for r in selected)} for key in sizes}}
    manifest_paths = ([profile_manifest_path] if isinstance(profile_manifest_path, (str, Path))
                      else list(profile_manifest_path or []))
    manifests = [json.loads(Path(path).read_text()) for path in manifest_paths]
    recipes = [bound_recipe(entry.get("sampling_recipe")) for entry in manifests]
    manifest = manifests[0] if manifests else None
    run_ids = sorted({r.get("run_id") for r in rows if r.get("run_id")})
    binding = "not_provided"
    if len(manifests) > 1:
        manifest_ids = [entry.get("run_id") for entry in manifests]
        if (not all(manifest_ids) or len(set(manifest_ids)) != len(manifest_ids)
                or set(run_ids) != set(manifest_ids) or not all(r.get("run_id") for r in rows)):
            raise ValueError("measurement run_ids must exactly match distinct supplied profile manifests")
        identities = [{"fingerprint": entry.get("fingerprint") or entry.get("engine", {}).get("fingerprint"),
                       "cache_config_hash": entry.get("engine", {}).get("cache_config_hash")}
                      for entry in manifests]
        if any(not all(identity.values()) or identity != identities[0] for identity in identities):
            raise ValueError("profile manifests must have the same fingerprint and cache_config_hash to merge")
        if any(recipe != recipes[0] for recipe in recipes):
            raise ValueError("profile manifests must have the same sampling recipe to merge")
        binding = "run_id_matched"
    elif manifest is not None:
        if run_ids and run_ids != [manifest.get("run_id")]:
            raise ValueError("measurement run_ids do not match the supplied profile manifest")
        binding = "run_id_matched" if run_ids and all(r.get("run_id") for r in rows) else "operator_declared_unlinked"
    paths = {"measurements": measurements_path,
             "development_report": development_report_path}
    files = {name: {"path": str(Path(path).resolve()), "sha256": sha256(Path(path).read_bytes()).hexdigest()}
             for name, path in paths.items() if path}
    manifest_files = [{"run_id": entry.get("run_id"), "path": str(Path(path).resolve()),
                       "sha256": sha256(Path(path).read_bytes()).hexdigest()}
                      for entry, path in zip(manifests, manifest_paths)]
    if manifest_files:
        files["profile_manifest"] = {key: manifest_files[0][key] for key in ("path", "sha256")}
        files["profile_manifests"] = manifest_files
    prior_binding = ({"output_prior_sampling":
                      json.loads(Path(development_report_path).read_text()).get("sampling_recipe")}
                     if development_report_path else {})
    return {**prior_binding, "profile_manifest": manifest, "profile_manifests": manifests,
            "measurement_provenance": {"binding_status": binding, "run_ids": run_ids,
                "sampling_recipe_binding": "explicit" if recipes and all(r is not None for r in recipes) else "unknown_legacy_unbound",
                "files": files,
                "sampling": phases, "cross_configuration_validation": False}}


def output_priors(report: dict) -> tuple[int, int]:
    """Use the method's completed development outputs, never reference answers."""
    turns = report.get("report", report).get("turns", [])
    lengths = [int(row["result"]["output_tokens"]) for row in turns
               if row.get("status") == "success" and row.get("result", {}).get("output_tokens", 0) > 0]
    if not lengths:
        raise ValueError("development report has no completed output lengths")
    lengths.sort()
    return round(statistics.mean(lengths)), lengths[math.ceil(.9*len(lengths))-1]


def feature_cell_key(row):
    """The predictor's independent feature cells, shared by splitting/reporting."""
    if row["phase"] == "decode":
        return ("decode", row["prompt_tokens"], row["output_tokens"], row["concurrency"], row.get("timing_scope"))
    phase = "first_token" if row["phase"] == "first_token" else "restore_prefill"
    return (phase, row["prompt_tokens"], row["prefix_tokens"], row.get("host_load_tokens", 0),
            row["concurrency"], row.get("timing_scope"))


def validate_timing_scopes(rows):
    """Phase aliases share a predictor; different clock/stage scopes cannot."""
    first = {(feature_cell_key(row)[0], row.get("timing_scope")) for row in rows
             if row["phase"] in {"prefill", "restore_prefill", "first_token"}}
    decode = {row.get("timing_scope") for row in rows if row["phase"] == "decode"}
    if len(first) > 1 or len(decode) > 1:
        raise ValueError("fit one timing scope per predictor at a time: first_token or queue-excluding prefill; do not mix clocks/stages")


def _error_summary(cases):
    relative = [row["relative_error"] for row in cases if row["relative_error"] is not None]
    return {"samples": len(cases), "mae_ms": statistics.mean(row["absolute_error_ms"] for row in cases) if cases else None,
            "mean_relative_error": statistics.mean(relative) if relative else None,
            "relative_error_samples": len(relative)}


def _observation_summary(cases):
    supported = [row for row in cases if row["supported"]]
    return {**_error_summary(cases), "supported_samples": len(supported),
            "coverage": len(supported)/len(cases) if cases else None,
            "supported_only": _error_summary(supported)}


def _cell_summary(cases):
    groups = {}
    for case in cases:
        groups.setdefault(tuple(case["feature_cell"]), []).append(case)
    cells = [_observation_summary(rows) for rows in groups.values()]
    supported = [cell["supported_only"] for cell in cells if cell["supported_samples"]]

    def equal_mean(rows, field):
        values = [row[field] for row in rows if row[field] is not None]
        return statistics.mean(values) if values else None

    return {"cells": len(cells), "supported_cells": len(supported),
            "weighting": "equal_feature_cell_mean_of_within_cell_observations",
            "coverage": equal_mean(cells, "coverage"), "mae_ms": equal_mean(cells, "mae_ms"),
            "mean_relative_error": equal_mean(cells, "mean_relative_error"),
            "relative_error_cells": sum(cell["mean_relative_error"] is not None for cell in cells),
            "supported_only": {"cells": len(supported), "mae_ms": equal_mean(supported, "mae_ms"),
                               "mean_relative_error": equal_mean(supported, "mean_relative_error"),
                               "relative_error_cells": sum(cell["mean_relative_error"] is not None for cell in supported)}}


def evaluate(model: CostModel, rows: list[dict]) -> dict:
    cases = []
    for row in rows:
        phase = row["phase"]
        if phase in {"prefill", "restore_prefill", "first_token"}:
            expected = float(row["elapsed_ms"])
            predicted, _, supported = model._prefill(row["prefix_tokens"],
                row["prompt_tokens"]-row["prefix_tokens"], row.get("host_load_tokens", 0), row["concurrency"])
        elif phase == "decode" and row["output_tokens"] > 1:
            expected = float(row["elapsed_ms"])/(row["output_tokens"]-1)
            predicted, _, supported = model._decode(row["prompt_tokens"]+row["output_tokens"]//2, row["concurrency"])
        else:
            continue
        error = abs(predicted-expected)
        cases.append({"phase": phase, "measured_ms": expected, "predicted_ms": predicted,
                      "absolute_error_ms": error, "relative_error": error/expected if expected > 0 else None,
                      "supported": supported, "feature_cell": list(feature_cell_key(row))})
    by_phase = {}
    for phase in sorted({case["phase"] for case in cases}):
        selected = [case for case in cases if case["phase"] == phase]
        by_phase[phase] = {**_observation_summary(selected),
                           "target": "decode_tpot" if phase == "decode" else "phase_latency",
                           "unit": "ms_per_output_token" if phase == "decode" else "ms",
                           "aggregation": "observation_weighted_within_phase",
                           "feature_cells_equal_weight": _cell_summary(selected)}
    return {**_observation_summary(cases), "cases": cases, "by_phase": by_phase,
            "aggregation": "mixed_phase_diagnostic" if len(by_phase) > 1 else "single_phase_observation_weighted"}


def fit(rows: list[dict], *, seed=42, holdout_fraction=.2, priors=None, allow_mock=False, prefill_estimator="lookup"):
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    allowed = {"hardware", "mock"} if allow_mock else {"hardware"}
    rows = [row for row in rows if row.get("source") in allowed]
    if not rows:
        raise ValueError("no permitted calibration observations")
    # Transfer/control observations remain available for reporting, but cannot
    # constitute a validation cell for the first-token/decode predictor.
    fitted = [row for row in rows if row["phase"] in {"prefill", "restore_prefill", "first_token"}
              or (row["phase"] == "decode" and row.get("output_tokens", 0) > 1)]
    if not fitted:
        raise ValueError("no first-token, prefill or decode calibration observations")
    if prefill_estimator == "positive_basis" and not any(row["phase"] != "decode" for row in fitted):
        raise ValueError("positive_basis requires first-token or prefill observations")
    validate_timing_scopes(fitted)
    # Hold out complete cells, not repeats of a cell that training already saw.
    groups = {}
    for row in fitted:
        # Match the predictor's actual feature cells: a different output cap
        # does not create an independent prefill cell, and the preceding cache
        # state does not create an independent decode interval.
        groups.setdefault(feature_cell_key(row), []).append(row)
    cells = sorted(groups, key=repr)
    random.Random(seed).shuffle(cells)
    count = max(1, round(len(cells)*holdout_fraction)) if len(cells) > 1 else 0
    held = [row for key in cells[:count] for row in groups[key]]
    training = [row for key in cells[count:] for row in groups[key]]
    settings = {} if priors is None else {"default_output_tokens": priors[0], "p90_output_tokens": priors[1]}
    if prefill_estimator not in {"lookup", "positive_basis"}:
        raise ValueError("prefill_estimator must be lookup or positive_basis")

    def fitted_model(observations):
        extra = {}
        if prefill_estimator == "positive_basis" and any(row["phase"] in {"prefill", "restore_prefill", "first_token"} for row in observations):
            from profiling.positive_prefill import fit_positive_prefill
            # Outer heldout cells never enter this inner OOF error calibration.
            extra = {"prefill_estimator": "positive_basis", "positive_prefill": fit_positive_prefill(observations)}
        return CostModel.from_measurements(observations, allow_mock=allow_mock, **settings, **extra)

    validation_model = fitted_model(training)
    validation = evaluate(validation_model, held)
    validation_status = ("insufficient_independent_cells" if not held else
                         "insufficient_estimator_training_cells" if validation_model.prefill_estimator != prefill_estimator else "evaluated")
    validation.update(seed=seed, training_cells=len(cells)-count, heldout_cells=count,
                      status=validation_status,
                      requested_estimator=prefill_estimator, evaluated_estimator=validation_model.prefill_estimator)
    # Freeze all dev observations after reporting held-out support and error.
    model = fitted_model(rows)
    return model, {"split": "development", "validation": validation, "prefill_estimator": prefill_estimator,
                   "output_priors_source": "completed_development_outputs" if priors else "defaults_not_fitted",
                   "measurement_sources": sorted({row["source"] for row in rows})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements")
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-report")
    parser.add_argument("--profile-manifest", action="append",
                        help="bind an original profiler manifest; repeat for runs with identical engine/cache configuration")
    parser.add_argument("--prefill-estimator", choices=["lookup", "positive_basis"], default="lookup",
                        help="positive_basis is opt-in and requires the calibration extra for offline fitting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=.2)
    parser.add_argument("--allow-mock", action="store_true", help="diagnostics only; artifact remains marked mock")
    args = parser.parse_args(argv)
    rows = [asdict(row) for row in load_measurements(args.measurements)]
    priors = output_priors(json.loads(Path(args.development_report).read_text())) if args.development_report else None
    model, metadata = fit(rows, seed=args.seed, holdout_fraction=args.holdout_fraction,
                          priors=priors, allow_mock=args.allow_mock, prefill_estimator=args.prefill_estimator)
    metadata.update(measurement_provenance(rows, args.measurements, profile_manifest_path=args.profile_manifest,
                                           development_report_path=args.development_report))
    model.save(args.output, metadata=metadata)
    print(json.dumps({"artifact": args.output, **metadata}, indent=2))


if __name__ == "__main__":
    main()
