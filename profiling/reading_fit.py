"""Freeze independent READ/REDUCE timing and length profiles from development."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from controller.reading import step_generation_kind


def _generated_steps(rows):
    for row in rows:
        if row["split"] != "dev":
            raise ValueError("reading priors are selected on development steps")
        outcome = row.get("reading_outcome")
        steps = outcome["steps"] if outcome is not None else row.get("steps", ())
        mode = outcome.get("plan", {}).get("reading_mode", "typed_reduce") if outcome is not None else "typed_reduce"
        for step in steps:
            task = step["task"]
            if task["kind"] in {"READ", "REDUCE"} and step["input_ids"]:
                kind = step_generation_kind(task["kind"], task.get("operation", "extract"), mode)
                yield row, step, kind


def fit_reading_priors(rows):
    groups = defaultdict(list)
    controls = defaultdict(list)
    tasks = defaultdict(set)
    for row, step, kind in _generated_steps(rows):
        key = f"{kind}:{step['task']['output_budget']}"
        groups[key].append(len(step["output_ids"]))
        tasks[key].add(row["task_id"])
        metrics = step.get("metrics", {})
        if metrics.get("step_wall_ms") is not None and metrics.get("last_token_ms") is not None:
            controls[key].append(metrics["step_wall_ms"]-metrics["last_token_ms"])
    return {key: {"mean_tokens": round(statistics.fmean(values)),
        "p90_tokens": sorted(values)[math.ceil(.9*len(values))-1],
        "steps": len(values), "tasks": len(tasks[key]), "split": "dev",
        "control_steps": len(controls[key]),
        "control_basis": "step_wall_minus_last_observed_token_including_receipt_and_retire",
        **({"mean_control_ms": statistics.fmean(controls[key])} if controls[key] else {}),
        "length_basis": "actual served outputs including budget-limited completions"}
        for key, values in groups.items()}


def fit_step_models(rows, profile_manifest, *, source="hardware", seed=42, records_path=None):
    from controller.reading import (COVERAGE_READ_SYSTEM, READ_SYSTEM, REDUCE_SYSTEM,
                                    SUMMARY_READ_SYSTEM, SUMMARY_REDUCE_SYSTEM)
    from profiling.fit import fit, measurement_provenance
    from profiling.reading_measurements import reading_measurements

    manifest = json.loads(Path(profile_manifest).read_text())
    prompts = {"READ": READ_SYSTEM, "REDUCE": REDUCE_SYSTEM,
               "READ_SUMMARY": SUMMARY_READ_SYSTEM, "REDUCE_SUMMARY": SUMMARY_REDUCE_SYSTEM,
               "READ_COVERAGE": COVERAGE_READ_SYSTEM}
    expected = {"fingerprint": manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint"),
                "cache_config_hash": manifest.get("engine", {}).get("cache_config_hash")}
    if not all(expected.values()):
        raise ValueError("reading profile manifest requires fingerprint and cache_config_hash")
    observed = defaultdict(list)
    for row, step, kind in _generated_steps(rows):
        identity = f"{row['task_id']}:{step['step_id']}"
        metrics = step.get("metrics", {})
        quote = metrics.get("cache_quote") or {}
        actual = {name: quote.get(name) for name in expected}
        if not all(actual.values()):
            raise ValueError(f"{identity} lacks observed cache_quote fingerprint/cache_config_hash")
        if actual != expected:
            raise ValueError(f"{identity} cache_quote configuration differs from profile manifest")
        if metrics.get("generation_kind", kind) != kind:
            raise ValueError(f"{identity} recorded generation_kind differs from its reading plan")
        systems = [message["content"] for message in step.get("input_messages", ()) if message["role"] == "system"]
        if systems != [prompts[kind]]:
            raise ValueError(f"{identity} actual system prompt differs from declared {kind} prompt or is unobserved")
        observed[kind].append({"task_id": row["task_id"], "step_id": step["step_id"],
            "fingerprint": actual["fingerprint"], "cache_config_hash": actual["cache_config_hash"],
            "quote_request_id": quote.get("request_id"), "observed_seq": quote.get("observed_seq"),
            "engine_restart_id": quote.get("engine_restart_id"), "system_prompt": systems[0]})
    # Exported measurements do not observe the operator's profile run_id.
    groups, ledger = reading_measurements(rows, source=source, run_id="")
    if not groups:
        raise ValueError("reading profile has no observed native step timing")
    lengths = fit_reading_priors(rows)
    models, validation = {}, {}
    for kind, observations in groups.items():
        measured = [asdict(item) for item in observations]
        model, evaluation = fit(measured, seed=seed, allow_mock=source == "mock")
        model.step_output_priors = {key: value for key, value in lengths.items() if key.startswith(kind+":")}
        provenance = measurement_provenance(measured, records_path,
                                           profile_manifest_path=profile_manifest)
        provenance["measurement_provenance"].update(
            binding_status="observed_step_configuration_and_prompt_matched",
            run_id_binding="operator_declared_unverified",
            declared_profile_run_id=manifest.get("run_id"),
            observed_step_bindings=observed[kind],
            step_ledger="embedded:reading_profile_validation.step_ledger")
        model.calibration_metadata = {**provenance, "generation_kind": kind,
            "output_prior_sampling": manifest["sampling_recipe"],
            "output_prior_system_prompt": prompts[kind],
            "output_prior_fingerprint": expected["fingerprint"],
            "output_prior_cache_config_hash": expected["cache_config_hash"], "calibration_split": "dev"}
        models[kind] = model.to_spec()
        validation[kind] = evaluation
    return models, {"validation": validation, "step_ledger": ledger,
                    "length_profiles": lengths, "split": "dev"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--cost-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-manifest", help="Bind independent native timing profiles to this engine run")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    artifact = json.loads(Path(args.cost_artifact).read_text())
    artifact["settings"]["step_output_priors"] = fit_reading_priors(rows)
    artifact["metadata"]["reading_profile_results"] = args.results
    if args.profile_manifest:
        artifact["step_models"], report = fit_step_models(rows, args.profile_manifest, records_path=args.results)
        artifact["metadata"]["reading_profile_validation"] = report
    Path(args.output).write_text(json.dumps(artifact, indent=2)+"\n")


if __name__ == "__main__":
    main()
