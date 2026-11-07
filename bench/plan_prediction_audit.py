"""Audit saved rolling predictions against current steps or actual request tails.

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
    if settings.get("mode") == "observed_actions":
        if settings.get("action_policy") == "rule":
            return set(), [], "coverage_and_quoted_work_rule"
        path = Path(settings["gain_asset"])
        return set(), [source(path)] if path.exists() else [], "bounded_rollout_cost_asset"
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


def observed_remaining(row, event, attempt, terminal, observations):
    step, _, _ = event["decision"]
    key = step["candidate"]["candidate_id"] if step["candidate"] is not None else step["task"]["step_id"]
    option = event["option_diagnostics"].get(key, {})
    forecast = option.get("rollout_expected_cost")
    row["prediction_scope"] = "fixed_bounded_rollout"
    row["actual_scope"] = "online_replanning_until_actual_terminal_state"
    row["terminal_state"] = attempt.stage
    row["completion_reserve"] = option.get("completion_reserve")
    row["completion_cost"] = option.get("completion_cost")
    row["prediction"] = forecast
    if forecast is None:
        row["unmatched_reasons"].append("recorded_bounded_rollout_forecast_missing")
        return
    paid = event.get("physical_paid_observation")
    if paid is None or terminal is None or attempt.stage not in {"COMMITTED", "FAILED"}:
        row["unmatched_reasons"].append("paid_boundary_or_authoritative_terminal_event_missing")
        return
    start = paid["end"]
    window = terminal.get("answer_phase_window")
    end = window["end"] if window is not None else terminal["monotonic_timestamp_ns"]/1e9
    if end <= start:
        row["unmatched_reasons"].append("terminal_boundary_does_not_follow_paid_boundary")
        return
    row["phase_window"] = {"start": start, "end": end, "clock": "client_monotonic"}
    row["actual"]["latency_ms"] = (end-start)*1000
    physical, reason = physical_match(observations, paid, start, end)
    if physical is None:
        row["unmatched_reasons"].extend(reason)
    else:
        row["actual"].update({key: physical[key] for key in ("gpu_gib_ms", "host_gib_ms")})
        row["physical_match"] = physical
    if attempt.stage == "COMMITTED" and attempt.result is not None:
        steps = attempt.reading_outcome.steps if attempt.reading_outcome else ()
        usage = attempt.auxiliary_usage or {"input_tokens": 0, "output_tokens": 0}
        for name, answer_tokens, step_field in (("input_tokens", attempt.plan.candidate.input_ids, "input_ids"),
                ("output_tokens", attempt.result.output_ids, "output_ids")):
            if usage[name] is None:
                continue
            actual = usage[name]+len(answer_tokens)+sum(len(getattr(step, step_field)) for step in steps)
            if actual >= event["paid_cost"][name]:
                row["actual"][name] = actual-event["paid_cost"][name]
    row["errors"] = {key: forecast[key]-value for key, value in row["actual"].items()}
    if row["completion_cost"] is not None and attempt.stage == "COMMITTED":
        row["latency_reserve_underestimated"] = row["actual"]["latency_ms"] > row["completion_cost"]["latency_ms"]


def observed_choice_trace(store, event, attempt, task, candidate, prefix_control):
    from controller.admission import quote_realization, quoted_native_work
    from controller.schema import CacheQuote
    from controller.serialization import decode

    step_id = task["step_id"]
    batch = event["quote"]
    key = candidate["candidate_id"] if candidate is not None else step_id
    option = event.get("option_diagnostics", {}).get(key, {})
    saved = store.execution_step(attempt.attempt_id, step_id) if task["kind"] != "ANSWER" else None
    result = saved[1].result if saved is not None else None
    if task["kind"] == "ANSWER" and attempt.result is not None:
        receipt = attempt.result.receipt
        native = quote_realization(attempt.answer_quote, receipt)
    else:
        receipt = saved[1].receipt if saved is not None else None
        native = saved[1].metrics.get("native_quote_work") if saved is not None else None
    if receipt is not None:
        native = {**(native or {}), "actual_device_hit_tokens": receipt.actual_device_hit_tokens,
            "actual_host_loaded_tokens": receipt.actual_host_loaded_tokens, "actual_prefill_tokens": receipt.actual_prefill_tokens}
    alternatives = []
    for raw in batch["quotes"]:
        quote = decode(CacheQuote, raw)
        prefill, host = quoted_native_work(quote)
        alternatives.append({"candidate_id": quote.candidate_id, "input_tokens": quote.input_len,
                             "quoted_prefill_tokens": prefill, "quoted_host_load_tokens": host})
    return {"request_id": attempt.request_id, "quote_seq": batch["observed_seq"],
        "candidate_id": key, "representation": candidate.get("representation") if candidate is not None else None,
        "quoted_alternatives": alternatives, "selected_native_work": native,
        "joint_demand": event.get("joint_context", option.get("joint_context")),
        "selection_rank": event.get("selection_rank"),
        "prefix_control": prefix_control, "step_stage": saved[0] if saved is not None else None,
        "new_claim_ids": tuple(claim.claim_id for claim in result.claims) if result is not None else (),
        "covered_unit_ids": result.covered_units if result is not None else (),
        "supported_needs": result.support if result is not None else (),
        "terminal_stage": attempt.stage}


def audit_directory(directory):
    directory = Path(directory).resolve()
    journal_path = directory/"state"/"events.jsonl"
    database_path = directory/"state"/"sessions.sqlite3"
    journal = journal_path.read_text().splitlines(keepends=True)
    unfinished = bool(journal and not journal[-1].endswith("\n"))
    if unfinished:
        journal = journal[:-1]
    events = [json.loads(line) for line in journal if line.strip()]
    decisions = [row for row in events if row["event"] in {"rolling_plan_decision", "rolling_required_scope_step"}]
    commits = {row["attempt_id"]: row for row in events if row["event"] == "ledger_committed"}
    recovered = {row["attempt_id"]: row for row in events if row["event"] == "attempt_recovered"}
    failures = {row["attempt_id"]: row for row in events if row["event"] == "attempt_failed"}
    controls = {(row["attempt_id"], row["choice"], row["quote_seq"]):
        {key: row[key] for key in ("elapsed_ms", "owner_status")}
        for row in events if row["event"] == "joint_prefix_control"}
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
            required = event["event"] == "rolling_required_scope_step"
            if required:
                task, candidate = event["task"], event.get("selected_candidate")
                action = candidate["action"] if candidate is not None else None
                output = candidate["output_budget"] if candidate is not None else task["output_budget"]
            else:
                next_step, action, output = event["decision"]
                task, candidate = next_step["task"], next_step["candidate"]
            kind = (candidate["generation_kind"] if candidate is not None
                    else "CPU_REDUCE" if task["kind"] == "REDUCE" and task["operation"] in
                        {"union", "count", "sum", "group"} else task["kind"])
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
            if required or event["decision_mode"] == "observed_actions":
                row["prediction_scope"] = "mandatory_scope_reservation" if required else "fixed_bounded_rollout"
                attempt = store.get_attempt(event["attempt_id"])
                key = candidate["candidate_id"] if candidate is not None else task["step_id"]
                control = controls.get((attempt.attempt_id, key, event["quote"]["observed_seq"]))
                row["trajectory"] = observed_choice_trace(store, event, attempt, task, candidate, control)
            if event.get("engine_kind") != "sglang":
                row["unmatched_reasons"].append("source_is_not_actual_sglang_hardware")
                continue
            if required:
                row["completion_reserve"] = event["completion_reserve"]
                continue
            if event["decision_mode"] == "observed_actions":
                terminal = ((commits.get(attempt.attempt_id) or recovered.get(attempt.attempt_id))
                            if attempt.stage == "COMMITTED" else failures.get(attempt.attempt_id))
                observed_remaining(row, event, attempt, terminal, observations)
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
                if candidate["input_hash"] != attempt.result.receipt.input_hash:
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
                if candidate is not None and (step.receipt is None or
                        not step.receipt.generation_complete or
                        candidate["input_hash"] != step.receipt.input_hash):
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


def trajectory_examples(rows):
    examples = {}
    for row in rows:
        trace = row.get("trajectory")
        if not trace or row["unmatched_reasons"] == ["source_is_not_actual_sglang_hardware"]:
            continue
        work = trace["selected_native_work"] or {}
        tags = {
            "observed_prefill": work.get("actual_prefill_tokens", 0),
            "observed_gpu_prefix_hit": work.get("actual_device_hit_tokens", 0),
            "observed_host_reload": work.get("actual_host_loaded_tokens", 0),
            "read_added_evidence": row["logical_kind"] == "READ" and trace["new_claim_ids"],
            "applied_demand_retention": any(owner.get("state") == "RETAINED" for owner in
                (trace["prefix_control"] or {}).get("owner_status", {}).values()),
        }
        for name, present in tags.items():
            if present and name not in examples:
                examples[name] = {"generation_dir": row["generation_dir"], "observation_id": row["observation_id"],
                                  "request_id": trace["request_id"]}
    return examples


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
    remaining = [row for row in rows if row.get("prediction_scope") == "fixed_bounded_rollout"]
    summary = {"format": "contextkv_rolling_prediction_error_audit_v1", "source_directories": sources,
        "summaries": summaries([row for row in rows if row.get("prediction_scope") is None]),
        "observed_rollout_remaining": summaries(remaining),
        "mandatory_scope_decisions": dict(Counter(row["kind"] for row in rows
            if row.get("prediction_scope") == "mandatory_scope_reservation")),
        "trajectory_examples": trajectory_examples(rows),
        "delivered_latency_reserve_underestimated": sum(row.get("latency_reserve_underestimated") is True for row in remaining),
        "delivered_latency_reserve_checked": sum("latency_reserve_underestimated" in row for row in remaining),
        "error_sign": "prediction minus actual",
        "physical_unit": "GiB milliseconds", "physical_scope": "whole declared device/process envelope, not marginal or shared",
        "heldout_rule": "exclude exact observation IDs in declared calibration assets; missing asset proof is retained in overlap_check_basis",
        "method": "read saved selected predictions and actual completed phases; no calls, refitting, score imputation or paid-cost addition"}
    (destination/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
