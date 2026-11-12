"""Settle paid request work even when no successful TurnResponse exists.

The durable model ledgers are authoritative. Provider groups include their
child auxiliary calls, and a rolling ANSWER may also be the final native step;
neither is charged twice. Missing receipts remain unknown, with a separately
reported lower bound from the receipts that did survive.
"""
from __future__ import annotations

import math


TOKEN_COST_FIELDS = (
    "total_input_tokens", "total_output_tokens", "reader_input_tokens",
    "reader_output_tokens", "auxiliary_input_tokens", "auxiliary_output_tokens",
    "actual_prefill_tokens", "actual_device_hit_tokens", "actual_host_loaded_tokens",
)
ADDITIVE_COST_FIELDS = (*TOKEN_COST_FIELDS, "auxiliary_ms", "online_compute_ms",
                        "gpu_gib_seconds", "host_gib_seconds")
NATIVE_FIELDS = ("reader_input_tokens", "reader_output_tokens", "actual_prefill_tokens",
                 "actual_device_hit_tokens", "actual_host_loaded_tokens", "compute_ms")


def _number(value):
    return value if (not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value) and value >= 0) else None


def _sum(values):
    return sum(values) if all(value is not None for value in values) else None


def _receipt_key(receipt):
    return (receipt.engine_restart_id, receipt.engine_session_id,
            receipt.session_generation, receipt.rid, receipt.attempt_id)


def _native_call(stage, input_ids, output_ids, receipt, generation, metrics):
    # A terminal receipt confirms the request's input; a planned/prepared
    # candidate without a submission is not model work.
    observed = generation is not None or receipt is not None
    output = _number(receipt.actual_output_tokens) if receipt is not None else None
    if output is None and generation is not None and generation.generation_complete:
        output = len(generation.output_ids)
    if (output is None and generation is None and receipt is not None and receipt.generation_complete
            and stage in {"COMPLETE", "RESULT_READY"}):
        output = len(output_ids)
    value = {
        "reader_input_tokens": len(input_ids) if observed and input_ids is not None else None,
        "reader_output_tokens": output,
        **{name: _number(getattr(receipt, name)) if receipt is not None else None for name in (
            "actual_prefill_tokens", "actual_device_hit_tokens", "actual_host_loaded_tokens")},
        "compute_ms": _number(generation.elapsed_ms if generation is not None else metrics.get("elapsed_ms"))}
    known = {field: number or 0 for field, number in value.items()}
    if output is None:
        known["reader_output_tokens"] = len(generation.output_ids if generation is not None else output_ids)
    return {**value, "known": known}


def _auxiliary_groups(calls):
    """Use each aggregate once; children only supply a known lower bound."""
    groups = {call["call_id"]: call for call in calls if call.get("kind") == "provider_usage"}
    children = {}
    for call in calls:
        if call.get("kind") != "provider_usage":
            children.setdefault(call.get("group_id"), []).append(call)
    values = []
    for identity, group in groups.items():
        members = children.pop(identity, ())
        value, known = {}, {}
        for field in ("input_tokens", "output_tokens", "compute_ms"):
            value[field] = _number(group.get(field))
            child_field = "elapsed_ms" if field == "compute_ms" else field
            lower = sum(_number(call.get(child_field)) or 0 for call in members)
            if field != "compute_ms" and value[field] is not None and value[field] < lower:
                value[field] = None  # An inconsistent aggregate is not an exact total.
            known[field] = value[field] if value[field] is not None else lower
        values.append({**value, "known": known})
    # Older or interrupted journals can contain children without a group row.
    # They are distinct actual calls, not an additional aggregate.
    for members in children.values():
        for call in members:
            value = {field: _number(call.get("elapsed_ms" if field == "compute_ms" else field))
                     for field in ("input_tokens", "output_tokens", "compute_ms")}
            values.append({**value, "known": {key: number or 0 for key, number in value.items()}})
    return values


def failed_request_cost(store, attempt):
    """Read-only settlement, before state cleanup; never infer missing usage as zero."""
    native, seen = [], set()
    stages = store.execution_steps_with_stage(attempt.attempt_id) if attempt is not None else ()
    for stage, record in stages:
        if record.engine_epoch is None or stage in {"PLANNED", "PREPARED"}:
            continue  # CPU reductions and unsubmitted native preparations.
        generation = store.execution_step(attempt.attempt_id, record.step_id)[2]
        key = _receipt_key(record.receipt) if record.receipt is not None else ("step", record.step_id)
        if key in seen:
            continue
        seen.add(key)
        native.append(_native_call(stage, record.input_ids, record.output_ids,
                                   record.receipt, generation, record.metrics))
    if attempt is not None:
        generation = attempt.result
        receipt = generation.receipt if generation is not None else attempt.terminal_receipt
        final_recorded = any(record.step_id == attempt.final_step_id for _, record in stages)
        final_submitted = (attempt.stage in {"SUBMITTED", "RESULT_READY", "COMMITTED"}
                           or attempt.failed_from_stage in {"SUBMITTED", "RESULT_READY"})
        key = _receipt_key(receipt) if receipt is not None else ("answer", attempt.attempt_id)
        if key not in seen and (receipt is not None or generation is not None
                                or final_submitted and not final_recorded):
            native.append(_native_call(attempt.stage,
                attempt.plan.candidate.input_ids if attempt.plan is not None else None,
                generation.output_ids if generation is not None else (), receipt, generation, {}))
        calls = store.auxiliary_calls(attempt.attempt_id)
        auxiliary = _auxiliary_groups(calls)
        if not calls:
            # Direct callers can provide aggregate usage without a capture
            # journal. Before final planning, reading progress owns gap usage.
            usages = [attempt.auxiliary_usage] if attempt.auxiliary_usage is not None else []
            progress = store.reading_progress(attempt.attempt_id)
            if progress is not None and (attempt.stage == "STARTED" or attempt.failed_from_stage == "STARTED"):
                if progress.auxiliary_usage is not None:
                    usages.append(progress.auxiliary_usage)
            auxiliary = [{**{field: _number(usage.get(field)) for field in ("input_tokens", "output_tokens", "compute_ms")},
                "known": {field: _number(usage.get(field)) or 0
                          for field in ("input_tokens", "output_tokens", "compute_ms")}}
                for usage in usages]
    else:
        auxiliary = []
    totals = {field: _sum([call[field] for call in native]) for field in NATIVE_FIELDS}
    lower = {field: sum(call["known"][field] for call in native) for field in NATIVE_FIELDS}
    for target, field in (("auxiliary_input_tokens", "input_tokens"),
                          ("auxiliary_output_tokens", "output_tokens"), ("auxiliary_ms", "compute_ms")):
        totals[target] = _sum([call[field] for call in auxiliary])
        lower[target] = sum(call["known"][field] for call in auxiliary)
    for total, parts in (("total_input_tokens", ("reader_input_tokens", "auxiliary_input_tokens")),
                         ("total_output_tokens", ("reader_output_tokens", "auxiliary_output_tokens")),
                         ("online_compute_ms", ("compute_ms", "auxiliary_ms"))):
        totals[total] = _sum([totals[field] for field in parts])
        lower[total] = sum(lower[field] for field in parts)
    totals.pop("compute_ms")
    lower.pop("compute_ms")
    return {**totals, "online_compute_basis": "durable native generation and auxiliary group receipts",
        "accounting": {"basis": "durable_attempt_step_and_auxiliary_receipts",
            "known_lower_bound": lower}}


def unsubmitted_cost():
    """A blocked task made no call; no per-request memory was attributed to it."""
    return {**dict.fromkeys(ADDITIVE_COST_FIELDS, 0), "complete_ms": None,
        "accounting": {"basis": "blocked_before_submission", "known_lower_bound": {}}}
