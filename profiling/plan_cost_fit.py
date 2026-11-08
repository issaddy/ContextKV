"""Build reader/cache-bound full-phase physical profiles from timed native runs.

No request-cost share is substituted for a phase measurement.  Each accepted
phase has a client-monotonic start/end, a real pre-generation quote, an actual
input/receipt, and physical observations bracketing both boundaries.  The
physical quantity is the complete dedicated device/process-group envelope;
it is not a causal marginal-memory measurement of one request.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from bench.study_resources import integrate_physical
from controller.reading import step_generation_kind
from controller.resident_models import resident_model_digest, resident_model_identity
from controller.serialization import to_jsonable
from controller.templates import token_hash
from profiling.retention_trace import SourceLedger, digest, event_id, write_json


INPUT_EDGES = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
PREFIX_EDGES = (64, 256, 1024, 4096, 16384, 65536)


class UnqualifiedPhase(ValueError):
    pass


def file_provenance(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}


def resident_binding(report):
    raw = report.get("resident_model_manifest")
    return dict(resident_model_manifest=resident_model_identity(raw), resident_model_manifest_raw=raw)


def resource_scope(observation, native_endpoint):
    """Use the actual sampler's device identities and declared PSS roots."""
    devices = [{"index": row["index"], "uuid": row["uuid"]}
        for row in observation["samples"][0]["devices"]]
    devices.sort(key=lambda row: row["index"])
    for sample in observation["samples"]:
        current = sorted(({"index": row["index"], "uuid": row["uuid"]} for row in sample["devices"]),
                         key=lambda row: row["index"])
        if current != devices:
            raise ValueError("physical device identity changed within the measured capture")
        observed_pids = {row["pid"] for row in sample["process_pss"]}
        if not set(observation["process_roots"]) <= observed_pids:
            raise ValueError("declared physical process root lacks an actual PSS observation")
    if sorted(observation["gpu_devices"]) != [row["index"] for row in devices]:
        raise ValueError("physical samples do not cover the declared GPU devices")
    return dict(gpu_devices=sorted(observation["gpu_devices"]), devices=devices,
        process_roots=sorted(observation["process_roots"]), host_basis=observation["host_basis"],
        native_endpoint=native_endpoint.rstrip("/"))


def observed_window(value, scope):
    if (value is None or value.get("clock") != "client_monotonic"
            or value.get("scope") != scope or value["end"] <= value["start"]):
        raise UnqualifiedPhase("missing_absolute_complete_phase_window")
    return value["start"], value["end"]


def qualify_phase(event, *, samples, all_windows, isolation):
    """Qualify observed facts before fitting any feature bin or prediction."""
    scope = "answer_until_commit" if event["generation_kind"] == "ANSWER" else "step_wall"
    start, end = observed_window(event.get("phase_window"), scope)
    if event["generation_kind"] == "ANSWER" and not event["phase_window"].get("start_definition"):
        raise UnqualifiedPhase("answer_start_definition_unobserved")
    if any(identity != event["observation_id"] and left < end and right > start
           for identity, left, right in all_windows):
        raise UnqualifiedPhase("concurrent_phase_envelope_not_individual_step_measurement")
    if isolation != "dedicated_service_device":
        raise UnqualifiedPhase("dedicated_device_scope_not_declared")
    if event["generation_kind"] == "CPU_REDUCE":
        workload = event.get("native_workload")
        if workload is None or workload.get("active_requests") is None or workload.get("queued_requests") is None:
            raise UnqualifiedPhase("cpu_native_workload_unobserved")
        if workload["active_requests"] or workload["queued_requests"]:
            raise UnqualifiedPhase("concurrent_native_workload")
        concurrency = workload["active_requests"]+1
        ready, host = 0, 0  # No model input or native prefix exists for this CPU operation.
        fingerprint = event["fingerprint"]
        cache_hash = hashlib.sha256(json.dumps(workload["cache_config"], sort_keys=True).encode()).hexdigest()
        quote_binding = {"basis": "CPU has no KV input; actual native workload recorded"}
    else:
        ids, receipt, quote = event["input_ids"], event.get("receipt"), event.get("cache_quote")
        if receipt is None or not receipt["generation_complete"] or token_hash(ids) != receipt["input_hash"]:
            raise UnqualifiedPhase("actual_input_receipt_proof_missing")
        if quote is None or quote.get("fingerprint") is None:
            raise UnqualifiedPhase("pre_generation_native_quote_unobserved")
        matching = [item for item in quote["quotes"] if item["input_hash"] == receipt["input_hash"]
                    and item["input_len"] == len(ids)]
        if not matching or receipt["engine_restart_id"] != quote["engine_restart_id"]:
            raise UnqualifiedPhase("actual_quote_input_restart_mismatch")
        if quote.get("active_requests") is None or quote.get("queued_requests") is None:
            raise UnqualifiedPhase("native_workload_unobserved")
        if quote["active_requests"] or quote["queued_requests"]:
            raise UnqualifiedPhase("concurrent_native_workload")
        if any(item["inflight_h2d_tokens"] or item["inflight_d2h_tokens"] for item in matching):
            raise UnqualifiedPhase("native_transfer_pending")
        submission = event.get("generation_window")
        if submission is None or not start <= submission["start"] < submission["end"] <= end:
            raise UnqualifiedPhase("actual_generation_not_bracketed_by_phase")
        concurrency = quote["active_requests"]+1
        ready, host = matching[0]["ready_gpu_prefix_tokens"], matching[0]["host_only_load_tokens"]
        fingerprint, cache_hash = quote["fingerprint"], quote["cache_config_hash"]
        quote_binding = dict(request_id=quote["request_id"], observed_seq=quote["observed_seq"],
            engine_restart_id=quote["engine_restart_id"], input_hash=receipt["input_hash"])
    physical = integrate_physical(sorted(samples, key=lambda row: row["timestamp"]), start, end)
    if not physical["complete_window_coverage"]:
        raise UnqualifiedPhase("physical_phase_boundary_not_bracketed")
    if event.get("system_prompt") is None or event.get("sampling_recipe") is None:
        raise UnqualifiedPhase("actual_prompt_or_sampling_unobserved")
    return {**event, "fingerprint": fingerprint, "cache_config_hash": cache_hash,
        "concurrency": concurrency, "ready_gpu_tokens": ready, "host_load_tokens": host,
        "latency_ms": (end-start)*1000, "gpu_gib_ms": physical["gpu_gib_seconds"]*1000,
        "host_gib_ms": physical["host_gib_seconds"]*1000,
        "gpu_peak_gib": physical["gpu_peak_gib"], "host_peak_gib": physical["host_peak_gib"],
        "quote_binding": quote_binding, "cost_scope": scope,
        "physical_basis": "observed complete dedicated device/process-group envelope; not marginal, not request-share allocation",
        "isolation_basis": "operator-declared dedicated service/device + recorded quote active/queued counts + no overlapping recorded phase",
        "system_prompt_sha256": hashlib.sha256(event["system_prompt"].encode()).hexdigest()}


def capture_events(entry_path):
    """Read typed actual StepRecords and durable ANSWER commit/response windows."""
    entry_path = Path(entry_path).resolve()
    root = entry_path.parent
    entry = json.loads(entry_path.read_text())
    if entry["source"] != "hardware":
        raise ValueError("execution calibration requires real hardware capture")
    trace_path, wire_path = root/"capture-trace.json", root/"capture-wire.json"
    if not trace_path.exists():
        # An entry that completed no canonical turn still has real collection
        # cost. Preserve its declared failures as exclusions, never phase data.
        sources = file_provenance((entry_path, wire_path))
        cases = entry.get("case_outcomes")
        if not cases:
            cases = [entry["current_source"] if entry["current_source"] is not None else
                     {"reason": "capture_initialization_failed_before_source_selection"}]
        events = [dict(observation_id="unexported-capture:"+str(index),
            generation_kind=case.get("phase", "UNEXPORTED_NATIVE"), split=case.get("split"),
            phase_window=None, source_files=list(sources), source_sha256=sources,
            source_exclusion="no_committed_frozen_phase; "+case.get("reason", str(entry["error"])))
            for index, case in enumerate(cases)]
        return events, entry["physical_observation"]["samples"], []
    trace, wire = json.loads(trace_path.read_text()), json.loads(wire_path.read_text())
    if trace["config"]["engine"]["kind"] != "sglang":
        raise ValueError("execution calibration needs an actual SGLang source capture")
    physical_scope = resource_scope(entry["physical_observation"], trace["config"]["engine"]["base_url"])
    calls = {item["receipt"]["attempt_id"]: item for item in wire if "receipt" in item}
    recipes = {digest(item["sampling"]): item["sampling"] for item in wire}
    run_sampling = next(iter(recipes.values())) if len(recipes) == 1 else None
    state_path = Path(entry["state_database"])
    events_path = state_path.parent/"events.jsonl"
    journal = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    commits = {item["attempt_id"]: item for item in journal
        if item["event"] in {"ledger_committed", "turn_committed"}}
    terminal_choices = {item["attempt_id"]: item for item in journal
                        if item["event"] == "cost_probe_terminal_choice"}
    sources = file_provenance((entry_path, trace_path, wire_path, state_path, events_path))
    events, windows = [], []
    store = SourceLedger(state_path)
    try:
        for turn in trace["turns"]:
            attempt = store.get_attempt(turn["attempt_id"])
            for raw in store.execution_steps(attempt.attempt_id):
                stage, step, _ = store.execution_step(attempt.attempt_id, raw.step_id)
                kind = step_generation_kind(step.task.kind, step.task.operation, turn["context_plan"]["reading_mode"])
                cpu = not step.input_ids
                call = calls.get(step.receipt.attempt_id) if step.receipt is not None else None
                quote = call.get("cache_quote") if call else None
                if quote is None:
                    quote = step.metrics.get("cache_quote")
                identity = attempt.attempt_id+"/"+step.step_id
                observed = dict(observation_id=identity, generation_kind="CPU_REDUCE" if cpu else kind,
                    context_action="CPU" if cpu else "REBUILD", input_ids=list(step.input_ids),
                    input_tokens=len(step.input_ids), output_budget=0 if cpu else step.task.output_budget,
                    output_tokens=len(step.output_ids), phase_window=step.metrics.get("phase_window"),
                    receipt=to_jsonable(step.receipt), cache_quote=quote,
                    fingerprint=trace["fingerprint"], cache_config_hash=(
                        quote["cache_config_hash"] if quote else step.metrics.get("cache_config_hash")),
                    system_prompt="" if cpu else step.input_messages[0].content,
                    native_workload=step.metrics.get("native_workload"),
                    sampling_recipe=call["sampling"] if call else run_sampling,
                    operation=step.task.operation, input_claim_count=len(step.task.input_claim_ids),
                    generation_window={"start": call["started_s"], "end": call["completed_s"]} if call else None,
                    generation_result=call["result"] if call else None,
                    split=turn["split"], dataset=turn["dataset"], resource_scope=physical_scope,
                    **resident_binding(entry),
                    source_files=list(sources), source_sha256=sources,
                    original_stage=stage)
                events.append(observed)
            final = attempt.plan.candidate
            call = calls.get(attempt.result.receipt.attempt_id)
            commit = commits.get(attempt.attempt_id, {})
            window = commit.get("answer_phase_window")
            # Prefer a response-tail window only when that exact interval was recorded.
            response_window = turn["source_metadata"].get("response", {}).get("answer_phase_window")
            if response_window is not None:
                window = response_window
            events.append(dict(observation_id=attempt.attempt_id+"/ANSWER", generation_kind="ANSWER",
                context_action=final.action, input_ids=list(final.input_ids), input_tokens=len(final.input_ids),
                output_budget=final.output_budget or turn["source_metadata"]["max_new_tokens"],
                output_tokens=len(attempt.result.output_ids), phase_window=window,
                receipt=to_jsonable(attempt.result.receipt), cache_quote=call.get("cache_quote") if call else None,
                system_prompt=final.messages[0].content, sampling_recipe=call["sampling"] if call else None,
                generation_window={"start": call["started_s"], "end": call["completed_s"]} if call else None,
                generation_result=call["result"] if call else None,
                selection_completed_s=terminal_choices.get(attempt.attempt_id, {}).get("selection_completed"),
                phase_success=attempt.result.finish_reason not in {"length", "max_tokens"},
                semantic_outcome=("answer_budget_exhausted_no_quality_label"
                    if attempt.result.finish_reason in {"length", "max_tokens"} else "answer_phase_completed_no_quality_label"),
                split=turn["split"], dataset=turn["dataset"], resource_scope=physical_scope,
                **resident_binding(entry),
                source_files=list(sources), source_sha256=sources))
    finally:
        store.close()
    for event in events:
        window = event["phase_window"]
        if window is not None:
            windows.append((event["observation_id"], window["start"], window["end"]))
    # Failed/unexportable turns still contribute real overlapping native pressure.
    known = {event["receipt"]["attempt_id"]: event["observation_id"] for event in events if event.get("receipt")}
    windows.extend((known.get(call["request"]["attempt_id"], "unexported:"+call["request"]["attempt_id"]),
        call["started_s"], call.get("completed_s", entry["resource_window"]["end"])) for call in wire)
    return events, entry["physical_observation"]["samples"], windows


def replay_events(trace_path, report_path):
    """Audit existing C native reports; missing quotes/tails stay unqualified."""
    trace_path, report_path = Path(trace_path).resolve(), Path(report_path).resolve()
    trace, report = json.loads(trace_path.read_text()), json.loads(report_path.read_text())
    if report["source"] != "hardware" or report["status"] != "completed" or report["intervention"] is not None:
        raise ValueError("execution calibration consumes completed native C arms only")
    if report["config"]["engine"]["kind"] != "sglang":
        raise ValueError("execution calibration needs an actual SGLang native report")
    physical_scope = resource_scope(report["physical_observation"], report["config"]["engine"]["base_url"])
    provenance = file_provenance((trace_path, report_path))
    events, windows = [], []
    for turn in trace["turns"]:
        for source in turn["events"]:
            if source["kind"] == "GAP":
                continue
            identity = event_id(turn, source)
            actual = report["events"][identity]
            cpu = source["kind"] == "CPU_REDUCE"
            kind = "CPU_REDUCE" if cpu else step_generation_kind(source["kind"],
                source["operation"], turn["context_plan"]["reading_mode"])
            window = actual["window"]
            windows.append((identity, window["start"], window["end"]))
            events.append(dict(observation_id=identity, generation_kind=kind,
                context_action=source.get("context_action", "CPU" if cpu else None),
                input_ids=source.get("input_ids", []), input_tokens=len(source.get("input_ids", [])),
                output_budget=0 if cpu else actual["output_budget"],
                output_tokens=0 if cpu else actual["actual_output_tokens"],
                phase_window={**window, "clock": "client_monotonic", "scope": "step_wall"},
                receipt=None if cpu else actual["result"]["receipt"], cache_quote=actual.get("cache_quote"),
                fingerprint=report["engine"]["fingerprint"], cache_config_hash=report["engine"]["cache_config_hash"],
                native_workload=actual.get("native_workload"), system_prompt=source.get("system_prompt"),
                operation=source["operation"], input_claim_count=len(source.get("input_claims", ())),
                sampling_recipe=source.get("sampling"), generation_window=actual.get("generation_window"),
                generation_result=actual.get("result"),
                split=turn["split"], dataset=turn["dataset"], resource_scope=physical_scope,
                **resident_binding(report),
                source_files=list(provenance), source_sha256=provenance,
                **({"source_exclusion": "controlled_decode_length"} if not cpu and report.get("decode_work_rule") else {}),
                replay_scope="fixed DAG; ANSWER has no business commit and is excluded"))
    return events, report["physical_observation"]["samples"], windows


def phase_events(entry_path):
    """Standalone REDUCE has a COMPLETE step, not a synthetic QA commit."""
    entry_path = Path(entry_path).resolve()
    root = entry_path.parent
    entry = json.loads(entry_path.read_text())
    phase_path = root/"phase-capture.json"
    if not phase_path.exists():
        return capture_events(entry_path)
    capture = json.loads(phase_path.read_text())
    if capture["format"] != "contextkv_phase_capture_v1" or capture["engine_kind"] != "sglang":
        raise ValueError("standalone phase calibration needs a real native phase capture")
    scope = resource_scope(entry["physical_observation"], capture["config"]["engine"]["base_url"])
    provenance = file_provenance((entry_path, phase_path, root/"capture-wire.json",
                                 root/"claim-input-origins.json", Path(entry["state_database"])))
    events = [{**event, "resource_scope": scope,
        **resident_binding(entry),
        "source_files": list(provenance), "source_sha256": provenance}
        for event in capture["events"]]
    windows = [(event["observation_id"], event["phase_window"]["start"], event["phase_window"]["end"])
               for event in events]
    return events, entry["physical_observation"]["samples"], windows


def feature_bucket(value, edges):
    # A measured no-hit state must not cover positive prefix reuse, or vice versa.
    return ("zero",) if value == 0 else ("positive", bisect_right(edges, value))


def phase_scope_exclusion(event, *, current_read_prompt=False,
                          answer_start_definition="rolling_proposals_compilation_start"):
    """Select protocols without changing any historical phase definition."""
    if current_read_prompt and event["generation_kind"] == "READ":
        from controller.reading import READ_SYSTEM
        if event.get("system_prompt") != READ_SYSTEM:
            return "obsolete_READ_system_prompt; actual_sha256="+hashlib.sha256(
                (event.get("system_prompt") or "").encode()).hexdigest()
    if event["generation_kind"] == "ANSWER":
        actual = (event.get("phase_window") or {}).get("start_definition")
        if actual != answer_start_definition:
            return "ANSWER_control_scope_not_selected; actual="+str(actual)+"; selected="+answer_start_definition
    return None


def fit_profiles(observations, *, reader_checkpoint, reader_profile_id, split, input_edges=INPUT_EDGES):
    if not observations:
        raise ValueError("no qualified complete measured phases")
    reference = observations[0]
    binding = {name: reference[name] for name in ("fingerprint", "cache_config_hash", "sampling_recipe",
                                                "resource_scope", "resident_model_manifest")}
    if binding["fingerprint"]["model_revision"] != reader_checkpoint:
        raise ValueError("reader checkpoint label differs from the actually observed fingerprint")
    groups = defaultdict(list)
    answer_definitions = {row["phase_window"]["start_definition"] for row in observations
                          if row["generation_kind"] == "ANSWER"}
    if len(answer_definitions) > 1:
        raise ValueError("one execution asset cannot mix ANSWER phase start definitions")
    for row in observations:
        if row["split"] != split or split not in {"train", "dev"}:
            raise ValueError("execution calibration must use its declared train/development partition")
        if any(row[name] != value for name, value in binding.items()):
            raise ValueError("one execution asset cannot mix reader/cache/sampling scopes")
        if row["context_action"] not in {"REBUILD", "APPEND", "CPU"}:
            raise ValueError("phase context action was not recorded")
        measured_inputs = row["input_claim_count"] if row["generation_kind"] == "CPU_REDUCE" else row["input_tokens"]
        key = (row["generation_kind"], row["context_action"], row["output_budget"],
            row["system_prompt_sha256"], bisect_right(input_edges, measured_inputs),
            row["concurrency"], feature_bucket(row["ready_gpu_tokens"], PREFIX_EDGES),
            feature_bucket(row["host_load_tokens"], PREFIX_EDGES),
            row["operation"] if row["generation_kind"] == "CPU_REDUCE" else None,
            row["phase_window"].get("start_definition", "reading_step_entry"))
        groups[key].append(row)
    profiles = []
    for key, values in groups.items():
        first = values[0]
        profile = dict(profile_id="measured-phase:"+digest(values),
            generation_kind=first["generation_kind"], context_action=first["context_action"],
            output_budget=first["output_budget"], system_prompt_sha256=first["system_prompt_sha256"],
            cost_scope=first["cost_scope"],
            observation_ids=[row["observation_id"] for row in values],
            source_files=sorted({path for row in values for path in row["source_files"]}),
            source_sha256={path: value for row in values for path, value in row["source_sha256"].items()},
            samples=len(values), fit_method="empirical maximum of observed same-protocol input bin; prediction over declared bin is an estimate",
            success_count=sum(row.get("phase_success", "semantic_outcome" not in row) for row in values),
            failure_count=sum(not row.get("phase_success", "semantic_outcome" not in row) for row in values),
            success_definition="phase parsing/handling without a recorded semantic or output-cap failure; not QA correctness",
            qa_quality_label=None,
            physical_basis=first["physical_basis"], isolation_basis=first["isolation_basis"],
            start_definition=first["phase_window"].get("start_definition", "reading_step_entry"),
            observed_operations=sorted({row.get("operation", "answer") for row in values}),
            observed_semantic_outcomes=sorted({row.get("semantic_outcome", "successful_phase") for row in values}),
            input_claim_count_range=[min(row.get("input_claim_count", 0) for row in values),
                                     max(row.get("input_claim_count", 0) for row in values)],
            actual_output_tokens_range=[min(row["output_tokens"] for row in values), max(row["output_tokens"] for row in values)])
        if first["generation_kind"] == "CPU_REDUCE":
            profile.update(input_measure="claim_count", operation=first["operation"])
        else:
            profile["input_measure"] = "tokens"
        for name, feature in (("input_range", "input_claim_count" if first["generation_kind"] == "CPU_REDUCE" else "input_tokens"), ("concurrency_range", "concurrency"),
                              ("ready_gpu_range", "ready_gpu_tokens"), ("host_load_range", "host_load_tokens")):
            profile[name] = [min(row[feature] for row in values), max(row[feature] for row in values)]
        bucket = key[4]
        profile["observed_input_range"] = list(profile["input_range"])
        profile["prediction_bin"] = [0 if bucket == 0 else input_edges[bucket-1],
            input_edges[bucket]-1 if bucket < len(input_edges) else input_edges[-1]*2-1]
        profile["prediction_status"] = "empirical_bin_estimate_not_an_additional_measured_phase"
        for name in ("latency_ms", "gpu_gib_ms", "host_gib_ms", "gpu_peak_gib", "host_peak_gib"):
            profile[name] = max(row[name] for row in values)
        profile["development_leave_one_observation_out"] = {
            name: dict(samples=len(values), mean_absolute_error=(sum(abs(row[name]-max(
                other[name] for index, other in enumerate(values) if index != position))
                for position, row in enumerate(values))/len(values) if len(values) > 1 else None))
            for name in ("latency_ms", "gpu_gib_ms", "host_gib_ms")}
        profile["independent_held_out_error"] = None
        profiles.append(profile)
    return dict(schema_version=1, measurement_kind="hardware", reader_checkpoint=reader_checkpoint,
        reader_profile_id=reader_profile_id, **binding, calibration_split=split,
        output_budgets=sorted({row["output_budget"] for row in observations if row["generation_kind"] != "CPU_REDUCE"}),
        rows=profiles, coverage=dict(generation_kinds=sorted({row["generation_kind"] for row in observations}),
            missing_full_tail_kinds=sorted({"READ", "ANSWER"}-{row["generation_kind"] for row in observations})),
        feature_schema=dict(input_edges=list(input_edges), prefix_edges=list(PREFIX_EDGES),
            range_rule="observed_input_range remains min/max actual; prediction_bin uses preregistered input edges with an empirical maximum estimate; zero reuse is separate"),
        reader_profile_binding="operator profile label bound to actual native fingerprint/cache/sampling",
        resident_model_manifest_digest=resident_model_digest(binding["resident_model_manifest"]),
        resident_model_manifests_raw=list({digest(row.get("resident_model_manifest_raw", row["resident_model_manifest"])):
            row.get("resident_model_manifest_raw", row["resident_model_manifest"]) for row in observations}.values()),
        answer_start_definition=next(iter(answer_definitions)) if answer_definitions else None,
        calibration_status="empirical input-bin prediction from measured phases; development leave-one-observation-out errors recorded; independent held-out error unmeasured",
        profile_usage_scope=("same verified resident provider/device/process envelope" if binding["resident_model_manifest"] is not None
                             else "declared device/process envelope; auxiliary model residency identity unverified"),
        value_scope="full isolated phase device/process-group envelope; not C retention net benefit or per-request resource allocation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-entry", action="append", default=[])
    parser.add_argument("--phase-entry", action="append", default=[], help="Actual standalone model REDUCE capture entry")
    parser.add_argument("--replay-pair", nargs=2, metavar=("TRACE", "NATIVE_REPORT"), action="append", default=[])
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--reader-profile-id", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--isolation", choices=("dedicated_service_device",), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-timing-output-dir",
        help="Also freeze actual first-token/decode/control ANSWER and per-kind native timing assets")
    parser.add_argument("--claim-capacity-asset", help="Actual tokenizer/native literal-key grammar capacity bound")
    parser.add_argument("--current-read-prompt", action="store_true",
        help="Exclude historical READ prompts differing from the imported actual READ_SYSTEM")
    parser.add_argument("--answer-start-definition", default="rolling_proposals_compilation_start",
        help="Select the actually recorded ANSWER control start; other scopes remain excluded in the ledger")
    args = parser.parse_args(argv)
    observations, ledger = [], []
    groups = [capture_events(path) for path in args.capture_entry]
    groups.extend(phase_events(path) for path in args.phase_entry)
    groups.extend(replay_events(*pair) for pair in args.replay_pair)
    for events, samples, windows in groups:
        for event in events:
            identity = dict(observation_id=event["observation_id"], generation_kind=event["generation_kind"],
                            source_files=event["source_files"])
            try:
                if "source_exclusion" in event:
                    raise UnqualifiedPhase(event["source_exclusion"])
                if event["split"] != args.split:
                    raise UnqualifiedPhase("source_partition_differs_from_declared_fit_partition")
                excluded_scope = phase_scope_exclusion(event, current_read_prompt=args.current_read_prompt,
                    answer_start_definition=args.answer_start_definition)
                if excluded_scope:
                    raise UnqualifiedPhase(excluded_scope)
                observed = qualify_phase(event, samples=samples, all_windows=windows, isolation=args.isolation)
                observations.append(observed)
                ledger.append({**identity, "status": "qualified", "observation": observed})
            except UnqualifiedPhase as error:
                ledger.append({**identity, "status": "excluded", "reason": str(error)})
    ledger_path = Path(args.output).with_suffix(".observations.json")
    write_json(ledger_path, dict(source="hardware", source_scope="actual canonical phase windows",
        observations=ledger, status_counts=dict(Counter(row["status"] for row in ledger)),
        unavailable_reason=None if observations else "no qualified full phase data; no calibrated asset emitted"))
    if not observations:
        raise ValueError("no qualified full phases; inspect the observations ledger")
    artifact = fit_profiles(observations, reader_checkpoint=args.reader_checkpoint,
        reader_profile_id=args.reader_profile_id, split=args.split)
    artifact["observation_ledger"] = str(ledger_path.resolve())
    artifact["selected_protocols"] = dict(current_read_prompt=args.current_read_prompt,
        answer_start_definition=args.answer_start_definition)
    if args.claim_capacity_asset is not None:
        capacity = json.loads(Path(args.claim_capacity_asset).read_text())
        if capacity["tokenizer_hash"] != artifact["fingerprint"]["tokenizer_hash"]:
            raise ValueError("claim capacity bound differs from the actual reader tokenizer")
        artifact["claim_capacity"] = capacity
    if args.native_timing_output_dir is not None:
        from profiling.plan_native_fit import fit_native_assets
        artifact["native_timing_assets"] = fit_native_assets(observations, args.native_timing_output_dir,
            observation_ledger=ledger_path)
    write_json(args.output, artifact)
    print(json.dumps({"profiles": len(artifact["rows"]), "qualified": len(observations),
        "excluded": len(ledger)-len(observations), "coverage": artifact["coverage"], "output": args.output}))
    return artifact


if __name__ == "__main__":
    main()
