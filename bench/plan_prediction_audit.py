"""Audit saved rolling current-step cost predictions against completed phases.

Read-only SQLite/events/physical observations; no generation, fitting or
request-share allocation. The paid prefix and unexecuted future tail are
removed before comparing the selected step with its own actual phase.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from bench.study_resources import integrate_physical
from controller.serialization import to_jsonable
from profiling.retention_trace import SourceLedger


METRICS = ("latency_ms", "gpu_gib_ms", "host_gib_ms")


def source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def selected_prediction(event):
    next_step, action, output = event["decision"]
    candidates = [row for row in event["evaluations"] if row["feasible"]]
    selected = max(candidates, key=lambda row: (row["objective"], -row["cost"]["latency_ms"],
        -row["cost"]["gpu_gib_ms"], -row["cost"]["host_gib_ms"], row["plan_id"]))
    if (selected["plan_id"] != next_step["plan_id"] or selected["lookahead"][0] != [
            next_step["task"]["step_id"], action, output]):
        return None, "saved_evaluation_does_not_identify_actual_selected_decision"
    current = {key: selected["cost"][key]-event["paid_cost"][key] for key in METRICS}
    choices = [choice for item in event["future_step_forecasts"] if item["plan_id"] == selected["plan_id"]
        and item["step_id"] != next_step["task"]["step_id"] for choice in item["choices"]]
    tail = []
    for identity in selected["forecast_sources"][1:]:
        matched = [row for row in choices if row["scope"]+":"+row["calibration_id"] == identity]
        costs = {tuple(row["cost"][key] for key in METRICS) for row in matched}
        if len(costs) != 1:
            return None, "missing_or_ambiguous_recorded_future_tail_cost:"+identity
        value = dict(zip(METRICS, next(iter(costs)), strict=True))
        tail.append({"forecast_source": identity, "cost": value})
        for key in METRICS:
            current[key] -= value[key]
    if any(not math.isfinite(value) or value < 0 for value in current.values()):
        return None, "negative_or_nonfinite_current_cost_after_exact_paid_tail_subtraction"
    return {"current": current, "evaluation": selected, "subtracted_paid": event["paid_cost"],
            "subtracted_future_tail": tail}, None


def scope_key(observation):
    return tuple(sorted(observation["gpu_devices"])), tuple(sorted(observation["process_roots"]))


def physical_match(observations, paid, start, end):
    expected = scope_key(paid)
    expected_devices = {(row["index"], row["uuid"]) for row in paid["samples"][-1]["devices"]}
    reasons = []
    for observation in observations:
        if scope_key(observation) != expected:
            reasons.append("physical_declared_gpu_or_process_scope_differs")
            continue
        samples = sorted(observation["samples"], key=lambda row: row["timestamp"])
        before = [row for row in samples if row["timestamp"] <= start]
        after = [row for row in samples if row["timestamp"] >= end]
        if not before or not after:
            reasons.append("physical_phase_boundaries_not_bracketed")
            continue
        used = [before[-1], *(row for row in samples if start < row["timestamp"] < end), after[0]]
        if any({(row["index"], row["uuid"]) for row in sample["devices"]} != expected_devices for sample in used):
            reasons.append("physical_gpu_uuid_scope_differs")
            continue
        if any(not set(expected[1]) <= {row["pid"] for row in sample["process_pss"]} for sample in used):
            reasons.append("physical_declared_process_root_pss_missing")
            continue
        physical = integrate_physical(samples, start, end)
        return {"gpu_gib_ms": physical["gpu_gib_seconds"]*1000,
            "host_gib_ms": physical["host_gib_seconds"]*1000,
            "source": observation["source"], "scope": {"gpu_devices": expected[0], "process_roots": expected[1]},
            "basis": "complete physical device/process envelope, no request-share or marginal attribution"}, None
    return None, sorted(set(reasons)) or ["no_physical_observation"]


def calibration_ids(directory):
    """Retain actual declared profile observations for exact overlap checks."""
    manifest_path = directory/"generation_manifest.json"
    if not manifest_path.exists():
        return set(), [], "generation_manifest_missing"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    settings = config["execution"].get("plan_gain")
    if settings is None:
        return set(), [], "rolling_profile_settings_missing"
    paths = [settings["execution_profile_asset"]]
    run_path = Path(config["generation"]["run_config"])
    import yaml
    run = yaml.safe_load(run_path.read_text())
    timing = config["execution"].get("cost_artifact") or run.get("cost_model", {}).get("artifact")
    if timing:
        paths.append(timing)
    paths.extend(run.get("cost_model", {}).get("step_artifacts", {}).values())
    identities, provenance, missing = set(), [], []
    for path in dict.fromkeys(paths):
        path = Path(path)
        if not path.exists():
            missing.append(str(path))
            continue
        value = json.loads(path.read_text())
        identities.update(identity for row in value.get("rows", ()) for identity in row["observation_ids"])
        identities.update(value.get("metadata", {}).get("measurement_provenance", {}).get("actual_observation_ids", ()))
        provenance.append(source(path))
    return identities, provenance, "missing_artifact_files:"+str(missing) if missing else "actual_declared_asset_observation_ids"


def audit_directory(directory):
    directory = Path(directory).resolve()
    journal_path = directory/"state"/"events.jsonl"
    database_path = directory/"state"/"sessions.sqlite3"
    journal = journal_path.read_text().splitlines(keepends=True)
    unfinished = bool(journal and not journal[-1].endswith("\n"))
    if unfinished:
        journal = journal[:-1]
    events = [json.loads(line) for line in journal if line.strip()]
    decisions = [row for row in events if row["event"] == "rolling_plan_decision"]
    commits = {row["attempt_id"]: row for row in events if row["event"] == "ledger_committed"}
    observations, internal = [], defaultdict(dict)
    for path in sorted((directory/"measurements").glob("*/physical_resources.json")):
        item = json.loads(path.read_text())
        observations.append({**item, "source": source(path)})
    for row in decisions:
        paid = row.get("physical_paid_observation")
        if paid is not None:
            internal[scope_key(paid)].update({sample["timestamp"]: sample for sample in paid["samples"]})
    for (devices, roots), samples in internal.items():
        observations.append({"gpu_devices": devices, "process_roots": roots,
            "samples": list(samples.values()), "source": {**source(journal_path), "kind": "recorded_paid_sampler_samples"}})
    fitted_ids, assets, overlap_basis = calibration_ids(directory)
    store = SourceLedger(database_path)
    store.db.execute("BEGIN")
    rows = []
    try:
        for number, event in enumerate(decisions):
            next_step, action, output = event["decision"]
            task = next_step["task"]
            kind = (next_step["candidate"]["generation_kind"] if next_step["candidate"] is not None
                    else "CPU_REDUCE" if task["kind"] == "REDUCE" else task["kind"])
            identity = event["attempt_id"]+"/"+("ANSWER" if task["kind"] == "ANSWER" else task["step_id"])
            row = {"generation_dir": str(directory), "event_index": number, "observation_id": identity,
                "attempt_id": event["attempt_id"], "step_id": task["step_id"], "kind": kind,
                "logical_kind": task["kind"],
                "operation": task["operation"], "context_action": action, "output_budget": output,
                "actual": {}, "errors": {}, "unmatched_reasons": [],
                "calibration_overlap": identity in fitted_ids,
                "heldout_qualified": bool(fitted_ids) and overlap_basis == "actual_declared_asset_observation_ids"
                    and identity not in fitted_ids,
                "overlap_check_basis": overlap_basis, "profile_assets": assets}
            rows.append(row)
            if event.get("engine_kind") != "sglang":
                row["unmatched_reasons"].append("source_is_not_actual_sglang_hardware")
                continue
            prediction, reason = selected_prediction(event)
            if reason:
                row["unmatched_reasons"].append(reason)
                continue
            row["prediction"] = prediction
            paid = event.get("physical_paid_observation")
            if paid is None:
                row["unmatched_reasons"].append("actual_paid_boundary_missing")
                continue
            attempt = store.get_attempt(event["attempt_id"])
            if task["kind"] == "ANSWER":
                window = commits.get(event["attempt_id"], {}).get("answer_phase_window")
                if (attempt.stage != "COMMITTED" or attempt.result is None or
                        not attempt.result.receipt.generation_complete or window is None):
                    row["unmatched_reasons"].append("final_qa_receipt_or_actual_commit_window_not_completed")
                    continue
                if window.get("clock") != "client_monotonic" or window.get("scope") != "answer_until_commit":
                    row["unmatched_reasons"].append("actual_complete_answer_phase_scope_missing")
                    continue
                overlap = paid.get("current_answer_phase_overlap")
                if overlap is None or overlap["start"] != window["start"]:
                    row["unmatched_reasons"].append("selected_answer_phase_start_does_not_match_actual_window")
                    continue
                if next_step["candidate"]["input_hash"] != attempt.result.receipt.input_hash:
                    row["unmatched_reasons"].append("actual_answer_receipt_does_not_match_selected_input")
                    continue
                start, end = paid["end"], window["end"]
                row["phase_definition"] = "current remaining ANSWER after actual paid boundary through commit/response tail"
                row["whole_answer_phase_window"] = window
            else:
                saved = store.execution_step(event["attempt_id"], task["step_id"])
                if saved is None or saved[0] != "COMPLETE" or saved[1].result is None:
                    row["unmatched_reasons"].append("logical_step_not_completed")
                    continue
                _, step, _ = saved
                if step.task.output_budget != output:
                    row["unmatched_reasons"].append("actual_step_output_budget_differs_from_selected_budget")
                    continue
                window = step.metrics.get("phase_window")
                if window is None or window.get("clock") != "client_monotonic" or window.get("scope") != "step_wall":
                    row["unmatched_reasons"].append("actual_complete_step_phase_window_missing")
                    continue
                if next_step["candidate"] is not None and (step.receipt is None or
                        not step.receipt.generation_complete or
                        next_step["candidate"]["input_hash"] != step.receipt.input_hash):
                    row["unmatched_reasons"].append("actual_step_receipt_does_not_match_selected_input")
                    continue
                start, end = window["start"], window["end"]
                row["phase_definition"] = "complete actual READ/model REDUCE/CPU phase; paid prefix excluded"
                row["selection_to_phase_gap_ms"] = (start-paid["end"])*1000
            if not paid["end"] <= start < end:
                row["unmatched_reasons"].append("selected_phase_does_not_follow_actual_paid_boundary")
                continue
            row["phase_window"] = {"start": start, "end": end, "clock": "client_monotonic"}
            row["actual"]["latency_ms"] = (end-start)*1000
            physical, reason = physical_match(observations, paid, start, end)
            if physical is None:
                row["unmatched_reasons"].extend(reason)
            else:
                row["actual"].update({key: physical[key] for key in ("gpu_gib_ms", "host_gib_ms")})
                row["physical_match"] = physical
            row["errors"] = {key: prediction["current"][key]-value for key, value in row["actual"].items()}
    finally:
        store.db.close()
    return rows, {"generation_dir": str(directory), "journal_unfinished_tail_omitted": unfinished,
        "events": source(journal_path), "state_database": source(database_path), "rolling_decisions": len(decisions)}


def summaries(rows):
    groups = {"all": rows}
    groups.update({kind: [row for row in rows if row["kind"] == kind] for kind in sorted({row["kind"] for row in rows})})
    result = {}
    for key, selected in groups.items():
        heldout = [row for row in selected if row["heldout_qualified"]]
        def metric_errors(qualified):
            metrics = {}
            for metric in METRICS:
                errors = [row["errors"][metric] for row in qualified if metric in row["errors"]]
                metrics[metric] = {"count": len(errors),
                    "mae": statistics.fmean(abs(value) for value in errors) if errors else None,
                    "bias_prediction_minus_actual": statistics.fmean(errors) if errors else None,
                    "rmse": math.sqrt(statistics.fmean(value*value for value in errors)) if errors else None}
            return metrics
        result[key] = {"decisions": len(selected), "heldout_qualified": len(heldout),
            "exact_calibration_overlap_excluded": sum(row["calibration_overlap"] for row in selected),
            "heldout_metrics": metric_errors(heldout), "all_observed_metrics": metric_errors(selected),
            "unmatched_reasons": dict(Counter(reason for row in selected for reason in row["unmatched_reasons"]))}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="Saved study/workflow root; discovers state/events.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    directories = sorted({path.parent.parent for root in args.root for path in Path(root).resolve().rglob("state/events.jsonl")})
    rows, sources = [], []
    for directory in directories:
        actual, provenance = audit_directory(directory)
        rows.extend(actual)
        sources.append(provenance)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"predictions.jsonl").write_text("".join(json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False)+"\n" for row in rows))
    summary = {"format": "contextkv_rolling_prediction_error_audit_v1", "source_directories": sources,
        "summaries": summaries(rows), "error_sign": "prediction minus actual",
        "physical_unit": "GiB milliseconds", "physical_scope": "whole declared device/process envelope, not marginal or shared",
        "heldout_rule": "exclude exact observation IDs in declared calibration assets; missing asset proof is retained in overlap_check_basis",
        "method": "read saved selected predictions and actual completed phases; no calls, refitting, score imputation or paid-cost addition"}
    (destination/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
