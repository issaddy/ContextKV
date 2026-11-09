"""Export actual pre-ANSWER states from completed, measured READ branches."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from bench.study_resources import integrate_physical
from controller.action_proposals import build_observed_action_plans, gap_key, gap_query, gap_scope_available, GapRetrievalRequest
from controller.context_budget import completed_source_coverage
from controller.context_schema import ContextPlan, ReadingOutcome, ReadResult, ReadTask, StepRecord
from controller.plan_gain_protocol import validate_input_protocol
from controller.plan_policy import CompleteCost
from controller.resident_models import evidence_selection_contract, resident_model_digest
from controller.schema import SessionSnapshot
from controller.serialization import decode, to_jsonable
from training.reader_data import file_digest


TERMINAL_FAMILIES = ("read_once", "reduce_once", "gap_retrieval")


def restore_read_catalog(state, original_plan):
    """Retain actual terminal tasks and the original, still-unread READ packs."""
    tasks = {task.step_id: task for task in state.plan.steps}
    units = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
    reads = [(tasks[result.step_id], result) for result in state.results
             if result.step_id in tasks and tasks[result.step_id].kind == "READ"]
    for task in original_plan.steps:
        if task.kind != "READ" or task.step_id in tasks:
            continue
        matching = tuple(result for actual, result in reads
            if set(task.need_ids) <= set(actual.need_ids) and task.condition_ids == actual.condition_ids)
        if not completed_source_coverage(state.plan, matching,
                tuple(units[identity] for identity in task.input_unit_ids)):
            tasks[task.step_id] = task
    return replace(state, plan=replace(state.plan, steps=tuple(tasks.values())))


def terminal_observation(before, run, source_task, answer_event, online, template, provenance):
    from training.plan_gain_rollout import digest

    attempt = run["attempt"]
    attempt_id = attempt["attempt_id"]
    original = source_task["source_capture"]
    original_attempt = original.get("response") or original["attempt"]
    source_packet = source_task["observed_packet"]
    observed = run["observed_state_binding"]
    snapshot = decode(SessionSnapshot, before["snapshot"])
    if (observed != before["observed_state_binding"]
            or source_packet["observation_id"] != observed["observation_id"]
            or source_packet["attempt_id"] != original_attempt["attempt_id"]
            or source_packet["source_capture_sha256"] != digest(original)):
        raise ValueError("terminal observation and READ branch refer to different source captures")
    state = decode(ReadingOutcome, attempt["reading_outcome"])
    validate_input_protocol(run["input_protocol"], reader_fingerprint=template.fingerprint)

    terminal = run["actual_native_calls"][-1]
    candidate = attempt["plan"]["candidate"]
    quote = terminal["cache_quote"]
    quoted = next(item for item in quote["quotes"] if item["candidate_id"] == terminal["quote_candidate_id"])
    request = terminal["request"]
    # The executor already validated the committed result. Bind the recording
    # and quote-return event to that result before cutting its cost window.
    if (terminal["receipt"] != attempt["result"]["receipt"] or answer_event["actual_quote"] != quote
            or quote["observed_seq"] != attempt["plan"]["quote_seq"]
            or quoted["input_hash"] != candidate["input_hash"]):
        raise ValueError("terminal Native quote does not bind the actual compiled ANSWER")

    execution = run["actual_branch_execution_cost"]
    start = execution["window"]["start"]
    end = answer_event["monotonic_timestamp_ns"] / 1e9
    answer_window = run["response"]["answer_phase_window"]
    if (online["clock"] != "client_monotonic" or answer_window["clock"] != "client_monotonic"
            or not start <= answer_window["start"] <= end <= terminal["started_s"]):
        raise ValueError("terminal quote event is outside the measured pre-ANSWER interval")
    samples = sorted(online["samples"], key=lambda sample: sample["timestamp"])
    physical = integrate_physical(samples, start, end)
    if not physical["complete_window_coverage"]:
        raise ValueError("terminal observation requires measured samples on both cost boundaries")
    if execution["total_input_tokens"] is None or execution["total_output_tokens"] is None:
        raise ValueError("terminal observation requires measured branch and ANSWER token costs")
    prior = decode(CompleteCost, observed["paid_cost"])
    before_end = [sample for sample in samples if sample["timestamp"] <= end]
    window_samples = [next(sample for sample in reversed(before_end) if sample["timestamp"] <= start),
                      *(sample for sample in before_end if start < sample["timestamp"])]
    for tier in ("gpu", "host"):
        physical[f"{tier}_peak_gib"] = max(sample[f"{tier}_gib"] for sample in window_samples)
    paid = prior + CompleteCost((end-start)*1000, physical["gpu_gib_seconds"]*1000,
        physical["host_gib_seconds"]*1000, execution["total_input_tokens"]-len(request["input_ids"]),
        execution["total_output_tokens"]-len(terminal["result"]["output_ids"]),
        physical["gpu_peak_gib"], physical["host_peak_gib"])

    # Fixed continuation stores a real GAP separately from its Native step ledger.
    expanded = set()
    if run["new_action_counts"]["GAP_RETRIEVAL"]:
        gap = run["gap_result"]
        retrieved = decode(GapRetrievalRequest, gap["request"])
        expanded = {gap_key(need) for need in retrieved.open_needs}
        after_read = run["new_step_ids"][0]
        gap_id = f"terminal-recorded-gap:{attempt_id}"
        task = ReadTask(gap_id, "GAP", tuple(dict.fromkeys(need.need_id for need in retrieved.open_needs)),
            input_claim_ids=tuple(dict.fromkeys(parent for need in retrieved.open_needs for parent in need.parent_claim_ids)),
            predecessor_step_ids=(after_read,), input_budget=0, output_budget=0, operation="retrieve")
        result = ReadResult(gap_id, elapsed_ms=gap["elapsed_ms"])
        step = StepRecord(attempt_id, gap_id, task, result=result, metrics={"gap_request": gap["request"],
            "auxiliary_usage": gap["pool"]["auxiliary_usage"], "source": "actual_branch_gap_result"})
        def after_first_read(rows, record):
            index = next(i for i, row in enumerate(rows) if row.step_id == after_read) + 1
            return (*rows[:index], record, *rows[index:])

        state = replace(state, steps=after_first_read(state.steps, step),
            results=after_first_read(state.results, result),
            plan=replace(state.plan, steps=after_first_read(state.plan.steps, task)))
    # New branches retain their complete selected organization. Restoring from
    # the pre-selection state would resurrect packs replaced by a regrouping.
    catalog = (decode(ContextPlan, before["plan_candidate"]["context_plan"])
        if "registered_context_plan_id" in run else decode(ReadingOutcome, before["working_state"]).plan)
    state = restore_read_catalog(state, catalog)
    state = replace(state, plan=replace(state.plan, online_tokens_spent=paid.input_tokens+paid.output_tokens))
    resident = run["resident_model_manifest"]
    resident_digest = resident_model_digest(resident)
    retriever = {"implementation": "StudyEvidenceProvider.extend_for_plan",
                 "resident_model_manifest_digest": resident_digest}
    gaps = tuple(need for need in state.open_needs if gap_key(need) not in expanded)
    gap_request = (GapRetrievalRequest(gaps, gap_query(state.plan.task_state, gaps), retriever,
        run["continuation_gap_limits"]) if gaps and gap_scope_available(state) else None)
    domain = observed.get("action_domain")
    plans, _ = build_observed_action_plans(state, snapshot, template,
        reader_checkpoint=run["reader_checkpoint"], gap_request=gap_request,
        read_choices=domain["read_choices"] if domain is not None else 1)
    return to_jsonable({"schema_version": 1, "decision_mode": "branch_terminal", "history_origin": run["history_origin"],
        "observation_id": f"{attempt_id}:branch_terminal:{quote['request_id']}", "attempt_id": attempt_id,
        "request_id": attempt["request_id"], "task_id": run["task_id"], "cohort_split": run["cohort_split"],
        "cohort_manifest_sha256": run["cohort_manifest_sha256"], "decision_seq": len(state.steps),
        "source_capture_attempt_id": original_attempt["attempt_id"], "source_capture_sha256": source_packet["source_capture_sha256"],
        "terminal_branch": provenance, "working_state": state, "snapshot": snapshot, "paid_cost": paid,
        "native_condition": quote, "native_condition_scope": "actual_terminal_answer_only",
        "paid_observation": {"clock": "client_monotonic", "start": start, "end": end,
            "basis": "original paid state plus measured branch work through terminal quote return", "branch_physical": physical},
        "reader_checkpoint": run["reader_checkpoint"], "reader_profile_id": run["reader_profile_id"],
        "input_protocol": run["input_protocol"], "resident_model_manifest_digest": resident_digest,
        "evidence_selection_contract": evidence_selection_contract(resident),
        "action_domain": domain, "legal_answer_representations": (candidate["representation"],),
        "gap_request": gap_request, "legal_actions": [plan.semantic_kind for plan in plans],
        "requested_families": TERMINAL_FAMILIES})


def export_terminal_observations(collection, template, output):
    from training.plan_gain_rollout import digest

    packets, excluded = [], []
    root = Path(collection)
    phases = (root/"phase-resources.json",) if (root/"phase-resources.json").is_file() else sorted(root.glob("*/phase-resources.json"))
    for phase_path in phases:
        directory = phase_path.parent
        phase = json.loads(phase_path.read_text())["online"]
        event_path = directory / "state/events.jsonl"
        with event_path.open() as stream:
            events = {event["attempt_id"]: event for line in stream if line.strip()
                for event in (json.loads(line),) if event["event"] == "plan_gain_experimental_answer"}
        phase_sha = file_digest(phase_path)
        for task in sorted((directory / "tasks").iterdir()):
            for run_path in sorted(task.glob("branch-*/run.json")):
                raw = run_path.read_bytes()
                run = json.loads(raw)
                if run["semantic_kind"] != "read_once":
                    continue
                if run["status"] != "completed":
                    excluded.append({"run": str(run_path.resolve()), "reason": "READ branch did not commit ANSWER"})
                    continue
                if not run["new_action_counts"]["READ"]:
                    excluded.append({"run": str(run_path.resolve()), "reason": "branch executed no new READ"})
                    continue
                before_path = run_path.with_name("before.json")
                before_raw = before_path.read_bytes()
                family = json.loads((task / "registered-family.json").read_text())
                event = events[run["attempt"]["attempt_id"]]
                provenance = {"run_path": str(run_path.resolve()), "run_sha256": hashlib.sha256(raw).hexdigest(),
                    "before_path": str(before_path.resolve()), "before_sha256": hashlib.sha256(before_raw).hexdigest(),
                    "events_path": str(event_path.resolve()), "answer_event_sha256": digest(event),
                    "phase_resources_path": str(phase_path.resolve()), "phase_resources_sha256": phase_sha}
                packets.append(terminal_observation(json.loads(before_raw), run, family["source_task"],
                    event, phase, template, provenance))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.writelines(json.dumps(packet, ensure_ascii=False, allow_nan=False)+"\n" for packet in packets)
    coverage = {split: {action: sorted({packet["task_id"] for packet in packets
        if packet["cohort_split"] == split and action in packet["legal_actions"]})
        for action in TERMINAL_FAMILIES} for split in ("train", "dev")}
    Path(str(output)+".manifest.json").write_text(json.dumps({"scope": "actual READ branch terminal states",
        "packets": len(packets), "action_coverage": coverage, "excluded": excluded,
        "output": str(output.resolve())}, indent=2)+"\n")
