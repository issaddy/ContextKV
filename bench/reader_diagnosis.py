"""Execute frozen-history E0/E1/E2 × R0/R1 diagnoses through serve_turn.

Oracle annotation and selection are offline preparation. Generation, steps,
native receipts and physical costs are measured; quality is scored separately.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from bench.evidence_diagnosis import (ApplicableScopeReview, OracleAnnotations, diagnose_evidence)
from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.quality import export_evaluation
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from controller.candidates import PLAN_SYSTEM_PROMPT, build_plan_candidates
from controller.config import create_controller, load_config
from controller.context_schema import AnswerPlan, AnswerPoint, ContextPlan, EvidencePool, TaskState
from controller.reading import build_read_plan, step_messages
from controller.schema import OnlineContext, OnlineMessage, OnlineTurn, TurnRecord
from controller.serialization import decode, to_jsonable


@dataclass(frozen=True)
class ReaderSpec:
    reader: str
    run_config: str | None
    gpu_devices: tuple[int, ...]
    checkpoint: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class DiagnosisProtocol:
    body_tokens: int
    output_tokens: int | None = None
    read_output_tokens: int | None = None
    read_input_tokens: int | None = None
    window_tokens: int | None = None
    reading_mode: str | None = None
    read_layout: str | None = None
    answer_layout: str | None = None
    output_tokens_per_fact: int = 64
    cache_start: str = "cold"
    read_packing: str | None = None


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)+"\n")


def _jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(to_jsonable(row), ensure_ascii=False)+"\n" for row in rows))


def _budget(plan, protocol):
    return replace(plan.budget, body_tokens=protocol.body_tokens, **{name: value for name, value in (
        ("output_tokens", protocol.output_tokens), ("read_output_tokens", protocol.read_output_tokens),
        ("read_input_tokens", protocol.read_input_tokens), ("window_tokens", protocol.window_tokens)) if value is not None})


def prepare_packet(packet, tokenizer, protocol):
    e0 = decode(ContextPlan, packet.get("e0_initial_plan") or packet["e0_plan"])
    pool = decode(EvidencePool, packet["candidate_pool"])
    full = decode(EvidencePool, packet["full_applicable_pool"])
    history = decode(tuple[TurnRecord, ...], packet["actual_history"])
    annotations = decode(OracleAnnotations, packet["annotations"])
    review = decode(ApplicableScopeReview, packet["scope_review"])
    diagnosis = diagnose_evidence(e0, pool, annotations, actual_history=history,
        full_applicable_pool=full, scope_review=review, body_budgets=(protocol.body_tokens,), tokenizer=tokenizer)
    runtime_bindings = any(task.input_claim_ids or task.kind == "REDUCE" for task in e0.steps) or any(
        point.claim_ids for point in e0.answer_plan.points)
    initial_available = packet.get("e0_plan_stage") != "runtime_expanded_only" and not runtime_bindings
    return {"e0": e0, "history": history, "diagnosis": diagnosis,
        "reading_enabled": any(task.kind == "READ" for task in e0.steps),
        "initial_plan_available": initial_available,
        "selections": {name: getattr(diagnosis, name.lower())[0] for name in ("E0", "E1", "E2")},
        "budget": _budget(e0, protocol), "verified_task_state": decode(TaskState, packet["verified_task_state"])
            if packet.get("verified_task_state") is not None else None}


def build_cell_plan(prepared, evidence_level, binding, protocol, template, snapshot):
    point = prepared["selections"][evidence_level]
    if not point.budget_feasible:
        return None
    original = prepared["e0"]
    state = original.task_state if binding == "inferred" else prepared["verified_task_state"]
    if state.raw_question != original.task_state.raw_question:
        raise ValueError("verified task replacement must preserve the actual user question")
    if not prepared["initial_plan_available"]:
        raise ValueError("initial plan is unavailable; runtime claim bindings cannot be replayed across readers")
    selected = (replace(original.selected_set, token_cost=point.evidence_set.token_cost) if evidence_level == "E0" else
                replace(point.evidence_set, set_id=f"{original.selected_set.set_id}:{evidence_level}"))
    read_settings = {"reading_mode": protocol.reading_mode or original.reading_mode,
                     "read_layout": protocol.read_layout or original.read_layout,
                     "read_packing": protocol.read_packing or original.read_packing}
    same_read_settings = all(getattr(original, name) == value for name, value in read_settings.items()) and all(
        getattr(original.budget, name) == getattr(prepared["budget"], name) for name in (
            "read_input_tokens", "read_output_tokens", "window_tokens"))
    preserve_initial = evidence_level == "E0" and binding == "inferred" and same_read_settings
    if not prepared["reading_enabled"] or preserve_initial:
        plan = replace(original, task_state=state, selected_set=selected, budget=prepared["budget"],
            steps=original.steps if prepared["reading_enabled"] else (), online_tokens_spent=0,
            auxiliary_ms=(), predicted_cost_ms=0.0,
            candidate_qualities=original.candidate_qualities if preserve_initial else (), **read_settings)
    else:
        plan = build_read_plan(state, selected, original.coverage_scope, prepared["budget"],
            template=template, snapshot=snapshot, output_tokens_per_fact=protocol.output_tokens_per_fact,
            separate_needs=original.separate_needs, numeric_identity=original.numeric_identity, **read_settings)
    layouts = (protocol.answer_layout,) if protocol.answer_layout is not None else original.layouts
    if preserve_initial:
        return replace(plan, layouts=layouts)
    catalog = tuple(span for unit in selected.selected_units for span in unit.source_spans)
    old_points = {point.need_id: point for point in original.answer_plan.points}
    points = []
    for need in state.needs:
        old = old_points.get(need.need_id)
        retained = tuple(span for span in old.source_spans if any(
            source.source_key == span.source_key and source.start <= span.start and span.end <= source.end
            for source in catalog)) if old is not None else ()
        points.append(AnswerPoint(need.need_id, source_spans=retained or catalog,
            citation_ids=old.citation_ids if old is not None and retained == old.source_spans else ()))
    return replace(plan, layouts=layouts,
                   answer_plan=AnswerPlan(tuple(points), original.answer_plan.disposition))


def _run_config(reader, directory):
    run = load_config(reader.run_config)
    return replace(run, engine=replace(run.engine, **({"base_url": reader.base_url} if reader.base_url else {})),
        context=replace(run.context, system_prompt=run.context.system_prompt or PLAN_SYSTEM_PROMPT),
        planner=replace(run.planner, strategy="always_rebuild", fallback_strategy="always_rebuild"),
        prefix_retention={"mode": "native"}, calibration_path=None,
        cost_model=replace(run.cost_model, artifact=None, step_artifacts={}, online_calibration=False),
        state_dir=str(directory/"state"))


def _reference(packet, history, question):
    value = {**packet.get("reference", {}), "task_id": packet["task_id"]}
    value.setdefault("input", [message for row in history for message in (
        {"speaker": "user", "text": row.user_text}, {"speaker": "agent", "text": row.assistant_text})]
        +[{"speaker": "user", "text": question}])
    for name in ("offline_reference", "goal_conditions", "targets"):
        if name in packet:
            value[name] = packet[name]
    return value


def _initial_contract(plan, template, snapshot):
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    return [{"step_id": task.step_id, "kind": task.kind, "operation": task.operation,
        "output_budget": task.output_budget,
        "input_ids": template.render(step_messages(plan, task,
            tuple(units[identity] for identity in task.input_unit_ids), snapshot=snapshot)).input_ids}
        for task in plan.steps if task.kind == "READ" and not task.predecessor_step_ids]


def initial_plan_report(packet, prepared, plan, level, binding, template, snapshot, policy):
    original = prepared["e0"]
    expected = _initial_contract(original, template, snapshot)
    actual = _initial_contract(plan, template, snapshot)
    expanded = packet.get("e0_actual_expanded_plan")
    initial_ids = {task.step_id for task in original.steps}
    report = {"comparison_scope": "initial_plan", "reading_enabled": prepared["reading_enabled"],
        "initial_plan_origin": packet.get("e0_plan_origin", "packet_e0_plan"),
        "initial_dag_preserved": original.steps == plan.steps,
        "initial_read_inputs_preserved": expected == actual,
        "actual_expansion_step_ids": [task["step_id"] for task in expanded["steps"]
            if task["step_id"] not in initial_ids] if expanded else [],
        "source_runtime_gap_read_step_ids": [task["step_id"] for task in expanded["steps"]
            if task["kind"] == "READ" and task["step_id"] not in initial_ids] if expanded else [],
        "runtime_reduction_policy": "derive_from_fresh_reader_results",
        "runtime_claim_bindings_replayed": False,
        "source_context_action": packet.get("source_context_action"), "context_action_protocol": "always_rebuild",
        "evidence_level": level, "binding": binding,
        "rebuild_reason": "preserved_direct_answer" if not prepared["reading_enabled"] else
            "original_initial_dag" if level == "E0" and binding == "inferred" and original.steps == plan.steps
            else "new_evidence_binding_or_read_budget"}
    observed = {step["step_id"]: tuple(step["input_ids"]) for step in packet.get("source_initial_read_inputs", ())}
    report["recorded_initial_read_inputs_match"] = (all(observed.get(step["step_id"]) == tuple(step["input_ids"])
        for step in actual) if observed and level == "E0" and binding == "inferred" else None)
    if not prepared["reading_enabled"] and level == "E0" and binding == "inferred":
        old = build_plan_candidates(snapshot, original.task_state.raw_question, original, template, policy=policy)
        new = build_plan_candidates(snapshot, plan.task_state.raw_question, plan, template, policy=policy)
        report["direct_answer_inputs_preserved"] = {(c.action, c.layout_id, c.input_ids) for c in old} == {
            (c.action, c.layout_id, c.input_ids) for c in new}
        recorded = packet.get("source_e0_answer_input_ids")
        report["recorded_direct_answer_input_candidate_match"] = any(
            candidate.input_ids == tuple(recorded) for candidate in new) if recorded is not None else None
    return report


async def run_diagnosis(packets, readers, output, protocol, *, packets_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"protocol.json").open("x") as stream:
        json.dump(to_jsonable({"protocol": protocol, "readers": readers,
            "packets_sha256": hashlib.sha256(json.dumps(packets, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "packets_path": str(Path(packets_path).resolve()) if packets_path is not None else None,
            "B": "always_rebuild", "C": "native", "quality": "offline_scoring_pending"}), stream, ensure_ascii=False, indent=2)
    rows, prepared_packets, contracts, reference_protocol = {}, None, {}, None
    cells = [(reader.reader, binding, level, number, packet) for reader in readers for number, packet in enumerate(packets)
        for binding in (("inferred", "verified") if packet.get("verified_task_state") is not None else ("inferred",))
        for level in ("E0", "E1", "E2")]
    for reader, binding, level, number, packet in cells:
        key = reader, binding, level, packet["task_id"]
        rows[key] = {"task_id": packet["task_id"], "reader": reader, "binding": binding,
            "evidence_level": level, "status": "pending", "metrics": {}, "record_number": number,
            "method": f"{level}_{reader}_{binding}", "dataset": packet.get("dataset", "reader_diagnosis"),
            "split": packet.get("split", "dev"), "conversation_id": packet.get("conversation_id", packet["task_id"]),
            "question": packet["e0_plan"]["task_state"]["raw_question"],
            "review_status": packet.get("review_status", "reviewed"),
            "actual_history": packet["actual_history"], "history_origin": "frozen_actual_rollout",
            "body_budget": protocol.body_tokens,
            "cost": {"complete_ms": None, "total_input_tokens": None, "total_output_tokens": None,
                     "gpu_gib_seconds": None, "host_gib_seconds": None}}

    def save_matrix():
        values = list(rows.values())
        _write(output/"matrix.json", {"protocol": protocol, "cells": values,
            "execution_status": "executed_unscored" if all(row["status"] == "completed" for row in values) else "partial",
            "quality_status": "offline_scoring_pending", "six_grid_passed": False})
        for reader in readers:
            for binding in ("inferred", "verified"):
                for level in ("E0", "E1", "E2"):
                    directory = output/f"{level}_{reader.reader}_{binding}"
                    selected = [row for row in values if (row["reader"], row["binding"], row["evidence_level"]) ==
                                (reader.reader, binding, level)]
                    if selected:
                        _jsonl(directory/"results.unscored.jsonl", selected)
    save_matrix()
    for reader in readers:
        if reader.run_config is None:
            for key, row in rows.items():
                if key[0] == reader.reader:
                    row["reason"] = "reader_checkpoint_or_runtime_pending"
            save_matrix()
            continue
        reader_dir = output/reader.reader
        run = _run_config(reader, reader_dir)
        initialized_at = time.monotonic()
        service = create_controller(run)
        initialization_ms = (time.monotonic()-initialized_at)*1000
        physical = native = None
        physical_started = False
        windows = {}
        try:
            protocol_identity = (service.template.fingerprint.tokenizer_hash, service.template.fingerprint.template_hash,
                                 run.context.system_prompt, run.sampling)
            if reference_protocol is not None and protocol_identity != reference_protocol:
                raise ValueError("R0/R1 must share tokenizer, prompt template and generation sampling")
            reference_protocol = protocol_identity
            if prepared_packets is None:
                started = time.monotonic()
                prepared_packets = [prepare_packet(packet, service.template.tokenizer, protocol) for packet in packets]
                _write(output/"offline_preparation.json", {"elapsed_ms": (time.monotonic()-started)*1000,
                    "diagnoses": [item["diagnosis"] for item in prepared_packets],
                    "original_preparation_usage": [packet.get("original_preparation_usage") for packet in packets],
                    "scope": "shared_offline_oracle_preparation_separate_from_served_cells"})
            info, allocator = await asyncio.gather(service.engine.info(), service.engine.metrics())
            _write(reader_dir/"runtime.json", {"reader": reader, "run_config": run, "engine": info, "allocator": allocator,
                "controller_initialization_ms": initialization_ms,
                "reader_checkpoint": reader.checkpoint or run.model.revision,
                "checkpoint_binding": "model_scope_identity" if reader.checkpoint is None else "explicit_checkpoint_registry",
                "B": "always_rebuild", "C": "native"})
            endpoint = urlsplit(run.engine.base_url)
            pids = listening_processes((endpoint.port or (443 if endpoint.scheme == "https" else 80),))
            physical = StudyResourceSampler(reader.gpu_devices, pids)
            native = NativePoolSampler(run.engine.base_url, gpu_capacity=allocator["pool"]["gpu_capacity_tokens"],
                host_enabled=allocator["pool"]["host_capacity_tokens"] > 0, model=run.model.name)
            await physical.start()
            physical_started = True
            await native.start()
            for number, (packet, prepared) in enumerate(zip(packets, prepared_packets, strict=True)):
                for binding in (("inferred", "verified") if prepared["verified_task_state"] is not None else ("inferred",)):
                    for level in ("E0", "E1", "E2"):
                        key = reader.reader, binding, level, packet["task_id"]
                        row = rows[key]
                        point = prepared["selections"][level]
                        row.update(method=f"{level}_{reader.reader}_{binding}", dataset=packet.get("dataset", "reader_diagnosis"),
                            split=packet.get("split", "dev"), history_origin="frozen_actual_rollout",
                            measurement_kind="hardware" if run.engine.kind == "sglang" else "simulation",
                            body_tokens=point.evidence_set.token_cost, body_budget=protocol.body_tokens,
                            read_layout=protocol.read_layout or prepared["e0"].read_layout,
                            read_packing=protocol.read_packing or prepared["e0"].read_packing,
                            reading_enabled=prepared["reading_enabled"],
                            initial_plan_available=prepared["initial_plan_available"],
                            witness_coverage={"completed": point.completed_requirements, "missing": point.missing_requirements},
                            full_applicable_scope_reviewed=prepared["diagnosis"].provenance["full_applicable_scope_reviewed"],
                            oracle_minimum_scope=prepared["diagnosis"].provenance["minimum_scope"],
                            original_e0_plan=packet["e0_plan"], input_packet_index=number,
                            original_preparation_usage=packet.get("original_preparation_usage"))
                        directory = output/row["method"]
                        row["generation_dir"] = str(directory.resolve())
                        if not prepared["initial_plan_available"]:
                            row.update(status="initial_plan_unavailable",
                                reason="only_runtime_expanded_claim_bound_plan_available")
                            save_matrix()
                            continue
                        if level != "E0" and packet.get("review_status") == "review_incomplete":
                            row.update(status="review_incomplete", reason=packet["review_reason"])
                            save_matrix()
                            continue
                        if not point.budget_feasible:
                            row.update(status="infeasible", reason="frozen_E0_exceeds_body_budget")
                            save_matrix()
                            continue
                        session = f"reader-diagnosis:{row['method']}:{packet['task_id']}"
                        corpus = packet.get("corpus_snapshot_id", "frozen-diagnosis-packet")
                        setup_started = time.monotonic()
                        before = service.store.read_snapshot(session, corpus)
                        before = service.store.sync_history(session, prepared["history"], before.version)
                        service.policy = replace(service.policy, max_new_tokens=prepared["budget"].output_tokens,
                                                 max_total_model_tokens=prepared["budget"].window_tokens)
                        cache_start = (await service.engine.profile_prepare("cold") if protocol.cache_start == "cold" else None)
                        started = time.monotonic()
                        response = plan = None
                        try:
                            plan = build_cell_plan(prepared, level, binding, protocol, service.template, before)
                            contract = _initial_contract(plan, service.template, before)
                            contract_key = packet["task_id"], binding, level
                            if contract_key in contracts and contracts[contract_key] != contract:
                                raise ValueError("paired readers compile different initial READ inputs or step budgets")
                            contracts[contract_key] = contract
                            report = initial_plan_report(packet, prepared, plan, level, binding,
                                service.template, before, service.policy)
                            row["initial_plan_report"] = report
                            _write(directory/"plans"/f"{number:06d}.json", {"plan": plan, "initial_read_contract": contract,
                                "initial_plan_report": report, "actual_expanded_plan": packet.get("e0_actual_expanded_plan"),
                                "cache_start": cache_start, "binding": binding, "reader": reader})
                            response = await service.serve_turn(session, plan.task_state.raw_question,
                                request_id=packet["task_id"], corpus_snapshot_id=corpus, context_plans=(plan,),
                                auxiliary_usage={"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0,
                                    "models": [], "scope": "frozen_evidence_execution_no_online_auxiliary_calls"})
                        finally:
                            ended = time.monotonic()
                            attempt = service.store.find_request(session, packet["task_id"])
                            steps = service.store.execution_steps(attempt.attempt_id) if attempt is not None else ()
                            after = service.store.get_snapshot(session)
                            windows[key] = started, ended
                            if response is not None and level == "E0" and binding == "inferred":
                                report = row["initial_plan_report"]
                                observed = packet.get("source_e0_answer_input_ids")
                                report["executed_direct_answer_input_matches_recorded"] = (
                                    attempt.plan.candidate.input_ids == tuple(observed)
                                    if not prepared["reading_enabled"] and observed is not None else None)
                                observed_reads = {item["step_id"]: tuple(item["input_ids"])
                                                  for item in packet.get("source_initial_read_inputs", ())}
                                actual_reads = {step.step_id: step.input_ids for step in steps if step.task.kind == "READ"}
                                report["executed_initial_read_inputs_match_recorded"] = (
                                    all(actual_reads.get(identity) == ids for identity, ids in observed_reads.items())
                                    if observed_reads else None)
                            profile = {"fingerprint": info.fingerprint, "cache_config_hash": info.cache_config_hash,
                                "sampling": run.sampling, "answer_prompt": run.context.system_prompt,
                                "reading_mode": plan.reading_mode if plan is not None else prepared["e0"].reading_mode,
                                "read_layout": plan.read_layout if plan is not None else prepared["e0"].read_layout,
                                "read_packing": plan.read_packing if plan is not None else prepared["e0"].read_packing}
                            profile_id = "reader-profile:"+hashlib.sha256(json.dumps(
                                to_jsonable(profile), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                            elapsed = ([response.engine_elapsed_ms, *(step.get("elapsed_ms")
                                for step in response.step_metrics)] if response is not None else [])
                            row.update(status="completed" if response is not None else "failed",
                                error=str(sys.exc_info()[1]) if response is None else None,
                                question=prepared["e0"].task_state.raw_question, actual_history=to_jsonable(before.raw_history),
                                conversation_id=packet.get("conversation_id", packet["task_id"]),
                                turn=len(before.raw_history)+1, protocol="P3_frozen_actual_history",
                                context_plan=to_jsonable(attempt.context_plan) if attempt is not None else to_jsonable(plan),
                                reading_outcome=to_jsonable(attempt.reading_outcome) if attempt is not None else None,
                                steps=to_jsonable(steps), attempt=to_jsonable(attempt), response=to_jsonable(response),
                                answer=response.assistant_text if response is not None else None,
                                qa_increment=len(after.raw_history)-len(before.raw_history),
                                version_increment=after.version-before.version, request_window=[started, ended],
                                offline_cell_setup_ms=(started-setup_started)*1000,
                                reader_checkpoint=reader.checkpoint or run.model.revision,
                                selected_answer_layout=attempt.plan.candidate.layout_id if response is not None else None,
                                reader_profile_id=profile_id, reader_runtime_profile=to_jsonable(profile),
                                cost={"complete_ms": (ended-started)*1000,
                                    "total_input_tokens": response.total_input_tokens if response else None,
                                    "total_output_tokens": response.total_output_tokens if response else None,
                                    "reader_input_tokens": response.reader_input_tokens if response else None,
                                    "reader_output_tokens": response.reader_output_tokens if response else None,
                                    "auxiliary_input_tokens": 0, "auxiliary_output_tokens": 0,
                                    "actual_device_hit_tokens": response.total_device_hit_tokens if response else None,
                                    "actual_host_loaded_tokens": response.total_host_loaded_tokens if response else None,
                                    "actual_prefill_tokens": response.total_prefill_tokens if response else None,
                                    "online_compute_ms": sum(elapsed) if elapsed and all(value is not None for value in elapsed) else None,
                                    "gpu_gib_seconds": None, "host_gib_seconds": None,
                                    "scope": "frozen_evidence_READ_REDUCE_ANSWER_execution"})
                            _write(directory/"records"/f"{number:06d}.json", row)
                            save_matrix()
                        if row["qa_increment"] != 1 or row["version_increment"] != 1:
                            raise ValueError("a diagnosis cell must commit exactly one current QA")
                        blocks = tuple(after.archive)
                        turn = OnlineTurn(row["conversation_id"], packet["task_id"], row["turn"],
                            packet.get("collection", "reader_diagnosis"), row["question"],
                            tuple(message for old in before.raw_history for message in (
                                OnlineMessage("user", old.user_text), OnlineMessage("assistant", old.assistant_text))),
                            tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri, block.title)
                                  for block in plan.selected_set.source_blocks), "P3", query_context=attempt.query_context)
                        export_evaluation(turn, response.assistant_text, official_path=directory/"official.jsonl",
                            audit_path=directory/"audit.jsonl", visible_evidence_manifest=response.visible_evidence_manifest,
                            audit_evidence=blocks, input_ids=attempt.plan.candidate.input_ids, output_ids=response.output_ids,
                            truncated=response.truncated, citation_scope=response.citation_scope,
                            measurement_kind=row["measurement_kind"])
                        with (directory/"references.jsonl").open("a") as stream:
                            stream.write(json.dumps(_reference(packet, before.raw_history, row["question"]), ensure_ascii=False)+"\n")
                        await service.close_session(session)
        finally:
            if native is not None:
                await native.stop()
            if physical_started:
                await physical.stop()
                _write(reader_dir/"physical_resources.json", physical.observation())
            if native is not None:
                observation = native.observation()
                _write(reader_dir/"native_resources.json", observation)
                for key, (start, end) in windows.items():
                    rows[key]["cost"].update(integrate_physical(physical.samples, start, end))
                    series, metadata = pool_series(observation, start, end)
                    rows[key]["native_resource_window"] = {**occupancy_integral(series), **metadata}
                    _write(Path(rows[key]["generation_dir"])/"records"/f"{rows[key]['record_number']:06d}.json", rows[key])
            save_matrix()
            await service.aclose()
            service.store.close()
    return list(rows.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", required=True)
    parser.add_argument("--r0-config", required=True)
    parser.add_argument("--r0-gpu", type=int, action="append", required=True)
    parser.add_argument("--r0-checkpoint")
    parser.add_argument("--r0-url")
    parser.add_argument("--r1-config")
    parser.add_argument("--r1-gpu", type=int, action="append", default=[])
    parser.add_argument("--r1-checkpoint")
    parser.add_argument("--r1-url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--body-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--read-output-tokens", type=int)
    parser.add_argument("--read-input-tokens", type=int)
    parser.add_argument("--window-tokens", type=int)
    parser.add_argument("--reading-mode", choices=("typed_reduce", "coverage_only"))
    parser.add_argument("--read-layout", choices=("task_first", "source_first"))
    parser.add_argument("--read-packing", choices=("mixed", "source_windows"))
    parser.add_argument("--answer-layout", choices=("existing", "need_order", "stable_evidence"))
    parser.add_argument("--cache-start", choices=("cold", "observed"), default="cold")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    packets = [json.loads(line) for line in Path(args.packets).read_text().splitlines() if line.strip()]
    readers = (ReaderSpec("R0", args.r0_config, tuple(args.r0_gpu), args.r0_checkpoint, args.r0_url),
               ReaderSpec("R1", args.r1_config, tuple(args.r1_gpu), args.r1_checkpoint, args.r1_url))
    protocol = DiagnosisProtocol(args.body_tokens, args.output_tokens, args.read_output_tokens,
        args.read_input_tokens, args.window_tokens, args.reading_mode, args.read_layout,
        args.answer_layout, cache_start=args.cache_start, read_packing=args.read_packing)
    asyncio.run(run_diagnosis(packets[:args.limit] if args.limit else packets, readers, args.output, protocol,
                             packets_path=args.packets))


if __name__ == "__main__":
    main()
