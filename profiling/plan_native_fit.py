"""Fit actual native first-token/decode/control assets for plan selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import math
import json
from pathlib import Path
import statistics

from controller.resident_models import resident_model_digest
from profiling.common import Measurement
from profiling.fit import fit, measurement_provenance
from profiling.retention_trace import digest, write_json


def timing_phases(observation):
    """Keep token timing and complete-phase control as separate observations."""
    result = observation.get("generation_result")
    if result is None or result.get("ttft_ms") is None or result.get("last_token_ms") is None:
        raise ValueError("actual streamed generation token timing is unobserved")
    first, last = result["ttft_ms"], result["last_token_ms"]
    control = observation["latency_ms"]-last
    if not 0 <= first <= last or control < 0:
        raise ValueError("observed token times do not fit inside the actual full phase")
    receipt = observation["receipt"]
    prefix = receipt["actual_device_hit_tokens"]+receipt["actual_host_loaded_tokens"]
    base = dict(prompt_tokens=observation["input_tokens"], prefix_tokens=prefix,
        output_tokens=observation["output_tokens"], concurrency=observation["concurrency"],
        transfer_bytes=0, host_load_tokens=receipt["actual_host_loaded_tokens"], source="hardware",
        run_id="native-phase:"+digest([observation["source_sha256"], observation["observation_id"]]),
        cache_state="actual_executed_"+observation["cache_config_hash"])
    phases = [Measurement(**base, phase="first_token", elapsed_ms=first,
        timing_scope="client_generate_to_first_observed_new_token_including_queue_transport")]
    if observation["output_tokens"] > 1:
        phases.append(Measurement(**base, phase="decode", elapsed_ms=last-first,
            timing_scope="client_first_to_last_observed_new_token_chunk"))
    phases.append(Measurement(**base, phase="control", elapsed_ms=control,
        timing_scope="complete_phase_wall_minus_generate_to_last_observed_new_token_including_head_and_tail"))
    return phases


def fit_native_assets(observations, output_dir, *, seed=42, observation_ledger=None):
    """Emit a real ANSWER base asset plus separately bound model-step assets."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups, rejected = defaultdict(list), []
    for row in observations:
        if row["generation_kind"] == "CPU_REDUCE":
            continue
        if row["split"] != "dev":
            rejected.append(dict(observation_id=row["observation_id"], reason="native timing freeze requires development observations"))
            continue
        try:
            phases = timing_phases(row)
        except ValueError as error:
            rejected.append(dict(observation_id=row["observation_id"], reason=str(error)))
            continue
        groups[row["generation_kind"]].append((row, phases))
    paths, validations, priors = {}, {}, {}
    for kind, entries in groups.items():
        first = entries[0][0]
        binding_names = ("fingerprint", "cache_config_hash", "sampling_recipe", "system_prompt")
        if any(any(row[name] != first[name] for name in binding_names) for row, _ in entries):
            raise ValueError("native timing kind cannot mix reader/cache/sampling/prompt scopes")
        if len({row["phase_window"].get("start_definition", "reading_step_entry") for row, _ in entries}) > 1:
            raise ValueError("native control timing kind cannot mix actual phase start definitions")
        rows = [asdict(phase) for _, phases in entries for phase in phases]
        measurements_path = output/(kind.lower()+"-measurements.jsonl")
        measurements_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        lengths = [row["output_tokens"] for row, _ in entries]
        natural_priors = round(statistics.fmean(lengths)), sorted(lengths)[math.ceil(.9*len(lengths))-1]
        model, validation = fit(rows, seed=seed, priors=natural_priors)
        if not model.prefill_points or not model.decode_points:
            rejected.append(dict(generation_kind=kind, reason="native selection needs both first-token and decode observations"))
            continue
        run_ids = [row["run_id"] for row in rows]
        manifest = dict(run_id="phase-cohort:"+digest(run_ids), fingerprint=first["fingerprint"],
            engine={"fingerprint": first["fingerprint"], "cache_config_hash": first["cache_config_hash"]},
            sampling_recipe=first["sampling_recipe"], system_prompt=first["system_prompt"],
            generation_kind=kind, resource_scope=first["resource_scope"],
            resource_scopes_observed=list({digest(row["resource_scope"]): row["resource_scope"] for row, _ in entries}.values()),
            residency_binding_scope="provenance only for native token timing; not a physical residency calibration",
            resident_model_manifest=first["resident_model_manifest"],
            resident_model_manifest_digest=resident_model_digest(first["resident_model_manifest"]),
            resident_model_manifests_raw=list({digest(row.get("resident_model_manifest_raw", row["resident_model_manifest"])):
                row.get("resident_model_manifest_raw", row["resident_model_manifest"]) for row, _ in entries}.values()),
            source_observation_run_ids=sorted(set(run_ids)),
            scope="fresh actual complete phase / actual input receipt / actual native quote")
        manifest_path = output/(kind.lower()+"-manifest.json")
        write_json(manifest_path, manifest)
        by_output = defaultdict(list)
        for row, phases in entries:
            by_output[row["output_budget"]].append((row, phases[-1].elapsed_ms))
        length_profiles = {}
        for budget, values in by_output.items():
            values_lengths = sorted(row["output_tokens"] for row, _ in values)
            remaining = [(row["phase_window"]["end"]-row["selection_completed_s"])*1000
                         -row["generation_result"]["last_token_ms"] for row, _ in values
                         if row.get("selection_completed_s") is not None]
            if any(value < 0 for value in remaining):
                raise ValueError("actual remaining control extends outside the post-selection phase")
            length_profiles[f"{kind}:{budget}"] = dict(mean_tokens=round(statistics.fmean(values_lengths)),
                p90_tokens=values_lengths[math.ceil(.9*len(values_lengths))-1], steps=len(values),
                mean_control_ms=statistics.fmean(control for _, control in values),
                control_basis="complete measured phase wall minus client last-token offset, retaining head and tail",
                control_range_ms=[min(control for _, control in values), max(control for _, control in values)],
                sampling_recipe=first["sampling_recipe"], system_prompt=first["system_prompt"],
                actual_output_budget=budget, split=first["split"])
            if len(remaining) == len(values):
                length_profiles[f"{kind}:{budget}"].update(
                    mean_remaining_control_ms=statistics.fmean(remaining),
                    remaining_control_samples=len(remaining),
                    remaining_control_basis="actual terminal selection_completed to phase_end minus generate-to-last-new-token offset",
                    remaining_control_range_ms=[min(remaining), max(remaining)])
        # The native predictor already consumes this per-kind/output-cap path,
        # including ANSWER. Control is never hidden inside a fabricated TTFT.
        model.step_output_priors = length_profiles
        provenance = measurement_provenance(rows, measurements_path)
        provenance["measurement_provenance"].update(
            binding_status="actual_quote_receipt_token_timings_and_complete_phase_window_matched",
            files={**provenance["measurement_provenance"]["files"],
                   "observation_ledger": {"path": str(Path(observation_ledger).resolve())} if observation_ledger else None},
            source_files=sorted({path for row, _ in entries for path in row["source_files"]}),
            source_sha256={path: value for row, _ in entries for path, value in row["source_sha256"].items()},
            actual_observation_ids=[row["observation_id"] for row, _ in entries],
            native_quote_bindings=[row["quote_binding"] for row, _ in entries])
        model.calibration_metadata = {**provenance, "profile_manifest": manifest,
            "profile_manifests": [manifest], "generation_kind": kind,
            "output_prior_sampling": first["sampling_recipe"],
            "output_prior_system_prompt": first["system_prompt"],
            "timing_system_prompt_sha256": first["system_prompt_sha256"],
            "output_prior_fingerprint": first["fingerprint"],
            "output_prior_cache_config_hash": first["cache_config_hash"],
            "calibration_split": first["split"], "native_phase_validation": validation,
            "native_phase_outcomes": dict(
                success_count=sum(row.get("phase_success", "semantic_outcome" not in row) for row, _ in entries),
                failure_count=sum(not row.get("phase_success", "semantic_outcome" not in row) for row, _ in entries),
                semantic_outcomes=[row.get("semantic_outcome", "phase_completed") for row, _ in entries],
                qa_quality_label=None, interpretation="calibrated execution cost does not predict QA success"),
            "step_control_scope": sorted({row["phase_window"].get("start_definition", "reading_step_entry")
                                           for row, _ in entries}),
            "control_coverage": "empirical per-output-cap mean; input-dependent head/tail variation retained in observations, not independently modelled"}
        path = output/(kind.lower()+"-native.json")
        model.save(path)
        paths[kind], validations[kind], priors[kind] = str(path.resolve()), validation, length_profiles
    available = sorted(paths)
    manifest = dict(format="contextkv_plan_native_timing_assets_v1", source="hardware",
        available_kinds=available, base_artifact=paths.get("ANSWER"),
        step_artifacts={kind: path for kind, path in paths.items() if kind != "ANSWER"},
        validation=validations, length_and_control_profiles=priors, excluded=rejected,
        selection_status="requires actual scope, sampling/prompt and lookup support checks at runtime",
        missing_kinds=sorted({"ANSWER", "READ", "REDUCE"}-set(paths)),
        config_overlay={"cost_model": {"artifact": paths.get("ANSWER"),
            "step_artifacts": {kind: path for kind, path in paths.items() if kind != "ANSWER"}},
            **({"context": {"system_prompt": groups["ANSWER"][0][0]["system_prompt"]}} if "ANSWER" in paths else {})})
    write_json(output/"native-assets.json", manifest)
    return manifest


def main(argv=None):
    from profiling.plan_cost_fit import UnqualifiedPhase, capture_events, phase_events, phase_scope_exclusion, qualify_phase
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-entry", action="append", default=[])
    parser.add_argument("--phase-entry", action="append", default=[])
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--current-read-prompt", action="store_true")
    parser.add_argument("--answer-start-definition", default="rolling_proposals_compilation_start")
    args = parser.parse_args(argv)
    observations, exclusions = [], []
    groups = [capture_events(path) for path in args.capture_entry]
    groups.extend(phase_events(path) for path in args.phase_entry)
    for events, samples, windows in groups:
        for event in events:
            try:
                if event.get("source_exclusion"):
                    raise UnqualifiedPhase(event["source_exclusion"])
                if event["split"] != args.split:
                    raise UnqualifiedPhase("source_partition_differs_from_native_fit")
                excluded_scope = phase_scope_exclusion(event, current_read_prompt=args.current_read_prompt,
                    answer_start_definition=args.answer_start_definition)
                if excluded_scope:
                    raise UnqualifiedPhase(excluded_scope)
                row = qualify_phase(event, samples=samples, all_windows=windows,
                                    isolation="dedicated_service_device")
                if row["fingerprint"]["model_revision"] != args.reader_checkpoint:
                    raise ValueError("native timing source differs from the requested reader checkpoint")
                observations.append(row)
            except UnqualifiedPhase as error:
                exclusions.append(dict(observation_id=event["observation_id"], reason=str(error)))
    path = Path(args.output_dir)/"native-observations.json"
    write_json(path, dict(observations=observations, excluded=exclusions,
        scope="native timing only; historical resident identity remains provenance, not new physical-role qualification"))
    manifest = fit_native_assets(observations, args.output_dir, observation_ledger=path)
    print(json.dumps({"native_kinds": manifest["available_kinds"], "output": args.output_dir,
                      "excluded": len(exclusions)}))
    return manifest


if __name__ == "__main__":
    main()
