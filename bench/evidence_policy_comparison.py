"""Evidence planner × reading protocol with fixed Qwen weights and actual P3 history."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse

import yaml

from bench.study import dataset_rows, merge_config, method_config, run_dataset
from controller.reading import READ_PROTOCOL_VERSION, SHARED_NUMERIC_PROTOCOL_VERSION, NUMERIC_SELECT_PROTOCOL_VERSION
from controller.serialization import decode, to_jsonable
from controller.context_schema import TaskState
from controller.schema import OnlineContext, OnlineMessage, OnlineTurn, TurnRecord
from training.reader_data import digest, file_digest, rows, write_rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)+"\n")


READ_ALLOCATION_ERRORS = frozenset((
    "need-isolated full coverage exceeds the request step allocation",
    "READ dependency closure needs finer original units or predecessor facts",
    "one READ fact and task exceed the input/output allocation",
    "full coverage exceeds the request step allocation",
    "selected step leaves no terminal ANSWER allocation",
    "initial plan exceeds the original passage allocation",
    "one claim exceeds the reduction input allocation",
    "reading exhausted the request step allocation",
    "reading exhausted the complete-time allocation"))


def terminal_budget_text(text):
    return (text in READ_ALLOCATION_ERRORS or text == "task interpretation unfinished: completed/length"
        or text.partition(":")[0] == "READ_STEP_OUTPUT_TRUNCATED"
        or re.fullmatch(r"[^\n]+: online token allocation \d+ cannot admit \d+ after \d+ spent", text) is not None
        or re.fullmatch(r"[^\n]+ input exceeds \d+", text) is not None
        or re.fullmatch(r"[^\n]+ exceeds the model window", text) is not None)


def terminal_model_budget(error):
    from controller.engine_client import EngineError
    from controller.execution import ReadingBudgetExceeded
    return (isinstance(error, EngineError) and error.code == "READ_STEP_OUTPUT_TRUNCATED"
            or type(error) is RuntimeError and str(error) == "task interpretation unfinished: completed/length"
            or isinstance(error, ReadingBudgetExceeded)
            or type(error) is ValueError and str(error) in READ_ALLOCATION_ERRORS)


READING_PROTOCOLS = {
    "task_first_isolated": {"layout": "task_first", "numeric_need_isolation": True},
    "source_first_shared": {"layout": "source_first", "numeric_need_isolation": False},
}
CELL_METHODS = tuple(policy+"_"+protocol for policy in ("P0", "P1") for protocol in READING_PROTOCOLS)
CELL_CONTRASTS = {
    "planner_at_task_first_isolated": ["P0_task_first_isolated", "P1_task_first_isolated"],
    "planner_at_source_first_shared": ["P0_source_first_shared", "P1_source_first_shared"],
    "protocol_at_P0": ["P0_task_first_isolated", "P0_source_first_shared"],
    "protocol_at_P1": ["P1_task_first_isolated", "P1_source_first_shared"],
}


def cell_method(policy, protocol):
    return {"id": policy+"_"+protocol, "strategy": "B7",
        "overrides": {"reading": READING_PROTOCOLS[protocol]}}


def allocation_identity(config):
    return {"execution": {name: config["execution"][name] for name in (
        "window_tokens", "history_tokens", "output_tokens", "max_steps", "max_total_tokens")},
        "reading_targets_and_options": {name: config["reading"][name] for name in (
            "input_target_tokens", "output_tokens", "output_tokens_per_fact", "output_budget_options")},
        "output_allocation": "shared total limits; READ/REDUCE output shares may be redistributed by the existing allocator"}


def fixed_reader_identity(reader_spec):
    from controller.config import load_config
    from controller.sampling import SamplingRecipe
    run_path = reader_spec["run_config"]
    run = load_config(run_path)
    return {"run_config": run_path, "run_config_sha256": file_digest(run_path),
        "checkpoint": run.model.revision, "profile_id": reader_spec["profile_id"],
        "scope_sha256": file_digest(run.model.scope_path),
        "dtype": run.model.scope.manifest()["record"]["config"]["torch_dtype"],
        "sampling": (run.sampling or SamplingRecipe()).record(), "weights_updated": False}


def cell_config(config, policy, protocol):
    settings = config["policy_comparison"]
    asset, reader_spec = settings["policies"][policy], settings["reader"]
    run_path = reader_spec["run_config"]
    fixed_reader = fixed_reader_identity(reader_spec)
    query_asset = settings["query_assets"][policy]
    actual = merge_config(config, {"generation": {"run_config": run_path,
        "teacher_url": settings["teacher_url"], "helper_run_config": settings["helper_run_config"]},
        "task_model": {"device": settings["encoder_device"]},
        "retrieval": {"task_query_asset": query_asset},
        "support_model": {"mode": "trained", "asset": asset, "preference_asset": asset,
            "device": settings["support_device"]}, "selection": {"utility": "learned"},
        "execution": {"prefix_retention": {"mode": "native"}, "plan_gain": None},
        "reading": READING_PROTOCOLS[protocol]})
    metadata = json.loads((Path(asset)/"support-model.json").read_text())
    identity = {"policy": policy, "read_protocol": protocol,
        "reading": READING_PROTOCOLS[protocol], "read_protocol_version": READ_PROTOCOL_VERSION,
        "shared_numeric_protocol_version": SHARED_NUMERIC_PROTOCOL_VERSION,
        "numeric_selection_protocol_version": NUMERIC_SELECT_PROTOCOL_VERSION,
        "policy_asset": asset, "policy_heads_sha256": file_digest(Path(asset)/"heads.pt"),
        "policy_checkpoint": metadata.get("policy_checkpoint"),
        "query_asset": query_asset,
        "reader_binding_used_for_policy_training": metadata.get("reader_binding"),
        "fixed_reader": fixed_reader,
        "evaluation_scope": "evidence planner x read protocol; fixed model weights and common total budgets"}
    if policy == "P1":
        from training.train_evidence_policy import require_policy_reader_binding
        from training.preference_rollout import query_encoder_binding
        binding = require_policy_reader_binding(asset, reader_checkpoint=fixed_reader["checkpoint"],
                                               reader_profile_id=reader_spec["profile_id"])
        query_identity = query_encoder_binding({"task_asset": query_asset})
        if binding["query_encoder"] != query_identity:
            raise ValueError("P1 comparison query asset differs from the actual policy-training query encoder")
        identity["query_encoder"] = query_identity
        identity["policy_protocol_evaluation"] = {
            "training_read_protocol": settings["p1_training_read_protocol"],
            "evaluation_read_protocol": protocol,
            "cross_protocol_transfer": protocol != settings["p1_training_read_protocol"],
            "reader_checkpoint_and_profile_unchanged": True}
    return actual, reader_spec, identity


def frozen_inputs(record):
    from bench.dynamic_goal_packets import _initial_e0
    from controller.context_schema import ContextPlan, EvidencePool
    initial, origin, stage = _initial_e0(record)
    return (decode(ContextPlan, initial), decode(EvidencePool, record["candidate_pool"]),
            {"task_origin": origin, "plan_stage": stage, "pool_origin": "actual E0 candidate_pool; not oracle/verified"})


def frozen_reference(source, actual):
    reference = {**source, "input": [message for turn in actual["actual_history"] for message in (
        {"speaker": "user", "text": turn["user_text"]}, {"speaker": "assistant", "text": turn["assistant_text"]})]
        +[{"speaker": "user", "text": actual["question"]}]}
    decision = actual.get("question_policy")
    if decision is not None:
        reference.update(offline_reference=decision["metadata"]["offline_reference"],
            goal_conditions=decision["metadata"]["goal_conditions"], goal_id=decision["goal_id"])
    return reference


async def frozen_cell(config, matrix, dataset, split, records, directory, *, policy, protocol,
                      expected_task_ids=None):
    from adapters.study_retrieval import create_study_retriever
    from bench.native_metrics import NativePoolSampler, pool_series
    from bench.metrics import occupancy_integral
    from bench.quality import export_evaluation
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.config import GenerationProtocol, apply_study_config, create_controller, load_config
    from controller.engine_client import EngineError
    actual, reader_spec, identity = cell_config(config, policy, protocol)
    spec = matrix["datasets"][dataset]
    actual = method_config(actual, cell_method(policy, protocol), spec, split)
    identity["allocation"] = allocation_identity(actual)
    audited = actual["policy_comparison"].get("audited_history")
    export_answers = not audited or split != "train"
    observation = actual["execution"].pop("state_observation", None)
    if not audited or (policy, protocol) != ("P1", "source_first_shared"):
        observation = None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"records").mkdir(exist_ok=True)
    devices = sorted(set((*actual["policy_comparison"]["frozen_shared_gpu_devices"], *reader_spec["gpu_devices"])))
    ports = [urlparse(reader_spec["base_url"]).port]
    if audited and actual["task_model"]["mode"] == "teacher":
        ports.append(urlparse(actual["generation"]["teacher_url"]).port)
    process_ids = sorted(set((*listening_processes(ports), *matrix["resources"].get("process_ids", ()))))
    if observation:
        write_json(directory/"frozen_a_study_config.json", actual)
        actual["execution"]["state_observation"] = {**observation,
            "cohort_manifest": audited["cohort_manifest"], "cohort_split": audited["cohort_split"],
            "observation_path": str(directory/"observations.jsonl"),
            "reader_checkpoint": identity["fixed_reader"]["checkpoint"],
            "reader_profile_id": reader_spec["profile_id"], "gpu_devices": devices, "process_ids": process_ids}
    run = apply_study_config(load_config(reader_spec["run_config"]), actual)
    run = replace(run, state_dir=str(directory/"state"),
        engine=replace(run.engine, base_url=reader_spec["base_url"]),
        planner=replace(run.planner, strategy="always_rebuild"), prefix_retention={"mode": "native"})
    if "answer_system_prompt" in spec:
        run = replace(run, context=replace(run.context, system_prompt=spec["answer_system_prompt"]))
    identity["generation_protocol"] = GenerationProtocol.from_config(run).record()
    service = create_controller(run)
    physical = StudyResourceSampler(devices, process_ids)
    await physical.start()
    started = time.monotonic()
    from adapters.history_state import HistoryIndex
    from adapters.hybrid_retrieval import LocalBGE
    from training.plan_gain_protocol import AuditedTaskInterpreter
    if audited:
        from training.plan_gain_rollout import create_audited_provider
        provider, interpreter = await asyncio.to_thread(create_audited_provider, actual, service.template)
    else:
        frozen_history = await asyncio.to_thread(LocalBGE, actual["task_model"].get("base_path", "models/bge-m3"),
            device=actual["policy_comparison"]["encoder_device"], dtype="bfloat16", max_length=8192)
        interpreter = AuditedTaskInterpreter(HistoryIndex(frozen_history.tokenizer, frozen_history.encode))
        provider = await asyncio.to_thread(create_study_retriever, actual, service.template,
            task_interpreter=interpreter, frozen_history=frozen_history)
    service.bind_resident_models(provider.manifest)
    initialization_ms = (time.monotonic()-started)*1000
    info = await service.engine.info()
    if info.fingerprint.model_revision != run.model.revision:
        raise ValueError("comparison reader service differs from its explicit run config")
    if actual["execution"].get("cache_start") == "cold":
        await service.engine.profile_prepare("cold")
    allocator = await service.engine.metrics()
    native = NativePoolSampler(run.engine.base_url, gpu_capacity=allocator["pool"]["gpu_capacity_tokens"],
        host_enabled=allocator["pool"]["host_capacity_tokens"] > 0, model=run.model.name)
    await native.start()
    sources = {row["task_id"]: row for row in (records if not export_answers else dataset_rows(spec, split,
        task_ids=matrix.get("cohorts", {}).get(split, {}).get(dataset)))}
    results = {row["task_id"]: row for path in (directory/"records").glob("*.json") for row in [json.loads(path.read_text())]}
    windows = {}
    try:
        for index, frozen in enumerate(records):
            task_id, query = frozen["task_id"], frozen["question"]
            if task_id in results:
                continue
            source = sources[task_id]
            session = "policy-frozen:"+policy+":"+protocol+":"+digest(task_id)[:20]
            history = decode(tuple[TurnRecord, ...], frozen["actual_history"])
            before = service.store.read_snapshot(session, provider.corpus_snapshot_id)
            before = service.store.sync_history(session, history, before.version)
            if audited:
                task_state = decode(TaskState, frozen["task_state"])
                frozen_origin = {"task_origin": "audited_full_cohort", "pool_origin": "live_cell_retrieval"}
                frozen_pool_hash = None
            else:
                original, frozen_pool, frozen_origin = frozen_inputs(frozen)
                task_state = original.task_state
                frozen_pool_hash = digest(to_jsonable(frozen_pool))
                provider.budget = original.budget
                provider.config["reading"].update(enabled=any(step.kind == "READ" for step in original.steps),
                    coverage_mode=original.coverage_scope)
                provider.config["selection"]["initial_gap_search_rounds"] = 0
                from adapters.history_state import restore_source_metadata
                restore_source_metadata(provider.directory, frozen_pool.units)
                provider.directory.add_archive(frozen_pool.source_blocks)
                fixed_pool = replace(frozen_pool, retrieval_ms=0, auxiliary_ms=(), auxiliary_usage=None)
                provider.directory.retrieve_pool = lambda state, memory: fixed_pool
                provider.directory.retrieve_gap_pool = lambda state, memory, needs: fixed_pool
            interpreter.bind(frozen, task_state)
            async def retrieve(snapshot):
                return await provider.retrieve(snapshot, query, task_id)
            async def extend(plan, needs, claims):
                return await provider.extend_for_plan(before, plan, needs, claims)
            arrived = time.monotonic()
            previous = service.store.find_request(session, task_id)
            response = None
            if previous is not None and previous.stage == "FAILED":
                if not terminal_budget_text(previous.error):
                    raise ValueError(previous.error)
                status, error, window = "failed", previous.error, None
            else:
                try:
                    response = await service.serve_turn(session, query, request_id=task_id,
                        corpus_snapshot_id=provider.corpus_snapshot_id, evidence_provider=retrieve,
                        open_need_provider=extend)
                    status, error = "completed", None
                except EngineError as exc:
                    if exc.code != "READ_STEP_OUTPUT_TRUNCATED":
                        raise
                    status, error = "failed", str(exc)
                except ValueError as exc:
                    if not terminal_model_budget(exc):
                        raise
                    status, error = "failed", str(exc)
                window = [arrived, time.monotonic()]
            attempt = service.store.find_request(session, task_id)
            committed = service.store.get_snapshot(session)
            steps = service.store.execution_steps_with_stage(attempt.attempt_id)
            value = {"task_id": task_id, "conversation_id": frozen["conversation_id"], "turn": frozen["turn"],
                "status": status, "error": error, "dataset": dataset, "split": split,
                "method": policy+"_"+protocol, "read_protocol": protocol, "policy_identity": identity,
                "measurement_kind": "hardware" if run.engine.kind == "sglang" else "simulation",
                "question": query, "actual_history": frozen["actual_history"], "history_origin": "frozen_actual_rollout",
                "frozen_input_origin": frozen_origin, "frozen_task_state": to_jsonable(task_state),
                "frozen_candidate_pool_sha256": frozen_pool_hash,
                "original_preparation_usage": frozen.get("planning_trace", {}).get("auxiliary_usage"),
                "answer": response.assistant_text if response else None, "response": to_jsonable(response),
                "qa_increment": len(committed.raw_history)-len(history),
                "planning_trace": provider.traces.pop((session, task_id), {}),
                "attempt": to_jsonable(attempt), "step_records": [
                    {"stage": stage, "record": to_jsonable(record)} for stage, record in steps],
                "request_window": window, "reference": frozen_reference(source, frozen),
                "cost": {"complete_ms": (window[1]-window[0])*1000 if window is not None else None,
                    "total_input_tokens": response.total_input_tokens if response else None,
                    "total_output_tokens": response.total_output_tokens if response else None}, "metrics": {}}
            if audited:
                value.update(cohort_split=frozen["cohort_split"], cohort_manifest_sha256=frozen["cohort_manifest_sha256"])
            if window is not None:
                windows[task_id] = window
            results[task_id] = value
            write_json(directory/"records"/f"{index:06}.json", value)
            if response and export_answers:
                blocks = {block.key: block for block in committed.archive}
                selected = [blocks[key] for key in sorted(attempt.plan.candidate.required_blocks)]
                online = OnlineTurn(frozen["conversation_id"], task_id, frozen["turn"], source["Collection"], query,
                    tuple(message for turn in history for message in (OnlineMessage("user", turn.user_text),
                        OnlineMessage("assistant", turn.assistant_text))),
                    tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri, block.title) for block in selected),
                    "frozen_actual_P3_state", query_context=attempt.query_context)
                export_evaluation(online, response.assistant_text, official_path=directory/"official.jsonl",
                    audit_path=directory/"audit.jsonl", visible_evidence_manifest=response.visible_evidence_manifest,
                    audit_evidence=tuple(blocks.values()), input_ids=attempt.plan.candidate.input_ids,
                    output_ids=response.output_ids, truncated=response.truncated, citation_scope=response.citation_scope,
                    measurement_kind=value["measurement_kind"])
            await service.close_session(session)
    finally:
        provider.close()
        await service.aclose()
        service.store.close()
        await native.stop()
        await physical.stop()
        ended = time.monotonic()
        stamp = time.time_ns()
        write_json(directory/f"physical.{stamp}.json", physical.observation())
        write_json(directory/f"native.{stamp}.json", native.observation())
        write_json(directory/f"batch_cost.{stamp}.json", {"physical": integrate_physical(physical.samples, started, ended),
            "start": started, "end": ended, "initialization_ms": initialization_ms, "policy_identity": identity})
        for task_id, window in windows.items():
            value = results[task_id]
            value["cost"].update(integrate_physical(physical.samples, *window))
            series, coverage = pool_series(native.observation(), *window)
            value["native_resource_window"] = {**occupancy_integral(series), **coverage}
            write_json(directory/"records"/f"{next(i for i, row in enumerate(records) if row['task_id'] == task_id):06}.json", value)
        ordered = [results[row["task_id"]] for row in records if row["task_id"] in results]
        write_rows(directory/"results.unscored.jsonl", [row for row in ordered if row["status"] == "completed"])
        write_rows(directory/"failed_cases.jsonl", [row for row in ordered if row["status"] == "failed"])
        write_rows(directory/"references.jsonl", [row["reference"] for row in ordered])
        write_json(directory/"generation_manifest.json", {"policy_identity": identity, "config": actual,
            "expected_tasks": expected_task_ids if expected_task_ids is not None else [row["task_id"] for row in records],
            "scheduled_task_ids": [row["task_id"] for row in records],
            "completed": sum(row["status"] == "completed" for row in ordered),
            "failed": sum(row["status"] == "failed" for row in ordered),
            "history_protocol": "fixed_actual_history_live_retrieval" if audited else "same frozen actual QA for all four cells",
            "engine": info, "provider": provider.manifest, "reference_used_online": False,
            "retrieval_protocol": "live_cell_retrieval" if audited else "frozen_pool"})
    return ordered


async def own_p3_cell(config, matrix, dataset, split, output, *, policy, protocol, goal_policy=None, limit_conversations=None):
    from controller.config import GenerationProtocol, apply_study_config, load_config
    actual, reader_spec, identity = cell_config(config, policy, protocol)
    spec = matrix["datasets"][dataset]
    actual = method_config(actual, cell_method(policy, protocol), spec, split)
    actual["execution"]["cache_start"] = "cold"
    actual["execution"].pop("state_observation", None)
    identity["allocation"] = allocation_identity(actual)
    concurrency = config["policy_comparison"]["concurrency"]
    identity["concurrency"] = concurrency
    if spec.get("protocol", "P3") != "P3":
        raise ValueError("own-history comparison requires P3, not imported authored history")
    source_rows = dataset_rows(spec, split, task_ids=matrix.get("cohorts", {}).get(split, {}).get(dataset))
    declared_ids = [row["task_id"] for row in source_rows]
    grouped = {}
    for row in source_rows:
        grouped.setdefault(row["conversation_id"], []).append(row)
    identities = sorted(grouped)[:limit_conversations]
    scheduled_ids = [row["task_id"] for conversation in identities for row in grouped[conversation]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    task_path = output/"tasks.jsonl"
    write_rows(task_path, source_rows)
    local_matrix = merge_config(matrix, {
        "resources": {"gpu_devices": sorted(set(
            (*config["policy_comparison"]["shared_gpu_devices"], *reader_spec["gpu_devices"])))},
        "datasets": {dataset: {"inputs": {split: str(task_path)},
            "reuse_provider": False, "continue_conversation_failures": True}},
        "cohorts": {split: {dataset: declared_ids}}})
    reader_run = yaml.safe_load(Path(reader_spec["run_config"]).read_text())
    reader_run["engine"]["base_url"] = reader_spec["base_url"]
    if "answer_system_prompt" in spec:
        reader_run.setdefault("context", {})["system_prompt"] = spec["answer_system_prompt"]
    run_path = output/"reader_run.yaml"
    run_path.write_text(yaml.safe_dump(reader_run, sort_keys=False))
    actual["generation"]["run_config"] = str(run_path)
    identity["generation_protocol"] = GenerationProtocol.from_config(
        apply_study_config(load_config(run_path), actual)).record()
    manifest = {"status": "running", "policy_identity": identity,
        "history_protocol": "each conversation advances sequentially on its own committed answers",
        "concurrency": concurrency, "cache_start_protocol": "cold_once_per_method_dataset",
        "task_ids": declared_ids, "scheduled_task_ids": scheduled_ids,
        "tasks_path": str(task_path.resolve()),
        "requested_conversations": identities}
    write_json(output/"manifest.json", manifest)
    collected = await run_dataset(actual, local_matrix, cell_method(policy, protocol),
        dataset, split, output, concurrency=concurrency, limit_conversations=limit_conversations,
        question_fn=goal_policy.question_fn if goal_policy else None)
    failed = json.loads((output/"failed_conversations.json").read_text())
    manifest.update(status="completed", completed_turns=len(collected),
        missing_execution_task_ids=sorted(set(declared_ids)-{row["task_id"] for row in collected}),
        failed_conversation_count=len(failed),
        failed_conversations_path=str((output/"failed_conversations.json").resolve()),
        state_db=str((output/"state"/"sessions.sqlite3").resolve()))
    write_json(output/"manifest.json", manifest)
    return collected


def persist_p3_failure(directory, conversation, task_id, session_id, attempt, error):
    failure = {"conversation_id": conversation, "task_id": task_id, "session_id": session_id,
        "error": error, "attempt_id": attempt.attempt_id if attempt is not None else None,
        "blocked_task_ids": [], "state_db": str((directory/"state"/"sessions.sqlite3").resolve())}
    write_json(directory/"terminal_read_failures"/(digest(conversation)[:20]+".json"), failure)
    return failure


async def compare(config, matrix, *, dataset, split, output, phase, frozen_records=None,
                  goal_policy=None, limit_states=None, limit_conversations=None, cells=None):
    audited = config["policy_comparison"].get("audited_history") if phase in {"frozen", "both"} else None
    records = rows(frozen_records) if phase in {"frozen", "both"} else []
    expected_task_ids = None
    if audited:
        if split != audited["cohort_split"]:
            raise ValueError("audited history must use its declared cohort split")
        from training.plan_gain_rollout import audited_cohort, verify_actual_history
        records, cohort = audited_cohort(records, rows(Path(config["training"]["data_dir"])/"sources.jsonl"),
            split=split, limit=len(records), cohort_manifest=audited["cohort_manifest"])
        verify_actual_history(records, rows(audited["actual_rollouts"]))
        expected_task_ids = cohort["declared_task_ids"]
        requested = expected_task_ids[:limit_states]
        by_task = {row["task_id"]: row for row in records}
        records = [by_task[task_id] for task_id in requested if task_id in by_task]
        write_json(Path(output)/"capture_cohort.json", {**cohort, "requested_task_ids": requested,
            "scheduled_task_ids": [row["task_id"] for row in records],
            "missing_task_ids": [task_id for task_id in requested if task_id not in by_task]})
    else:
        records = records[:limit_states]
    if not audited and any(row["status"] != "completed" for row in records):
        raise ValueError("frozen comparison states must come from completed actual QA records")
    result = []
    for policy in ("P0", "P1"):
        for protocol in READING_PROTOCOLS:
            cell = policy+"_"+protocol
            if cells is not None and cell not in cells:
                continue
            if phase in {"frozen", "both"}:
                selected = await frozen_cell(config, matrix, dataset, split, records, Path(output)/"frozen"/cell,
                                             policy=policy, protocol=protocol, expected_task_ids=expected_task_ids)
                result.append({"cell": cell, "phase": "frozen", "observed_records": len(selected)})
            if phase in {"p3", "both"}:
                selected = await own_p3_cell(config, matrix, dataset, split, Path(output)/"p3"/cell,
                    policy=policy, protocol=protocol, goal_policy=goal_policy, limit_conversations=limit_conversations)
                result.append({"cell": cell, "phase": "p3", "observed_records": len(selected)})
    write_json(Path(output)/"comparison_manifest.json", {"cells": result, "quality": "unscored; score saved actual outputs separately",
        "purpose": "B_observation_capture" if audited and split == "train" else "A_factorial_comparison",
        "frozen_protocol": "fixed_actual_history_live_retrieval" if audited else "frozen_pool",
        "declared_frozen_states": len(expected_task_ids) if audited else len(records),
        "comparison_axes": {"evidence_planner": ["P0", "P1"], "read_protocol": list(READING_PROTOCOLS)},
        "declared_cohorts": matrix.get("cohorts", {}).get(split, {}),
        "paired_contrasts_baseline_candidate": CELL_CONTRASTS,
        "reading_protocols": READING_PROTOCOLS, "read_protocol_version": READ_PROTOCOL_VERSION,
        "shared_numeric_protocol_version": SHARED_NUMERIC_PROTOCOL_VERSION,
        "numeric_selection_protocol_version": NUMERIC_SELECT_PROTOCOL_VERSION,
        "query_assets": config["policy_comparison"]["query_assets"],
        "fixed_reader": fixed_reader_identity(config["policy_comparison"]["reader"]),
        "common_allocation": allocation_identity(method_config(config,
            {"id": "common", "strategy": "B7"}, matrix["datasets"][dataset], split)),
        "cache_policy": "native", "plan_gain_enabled": False,
        "own_p3_concurrency": config["policy_comparison"]["concurrency"], "frozen_concurrency": 1,
        "frozen_records": str(frozen_records) if frozen_records else None,
        "frozen_records_sha256": file_digest(frozen_records) if frozen_records else None,
        "planned_frozen_states": len(records), "dataset": dataset, "split": split})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("frozen", "p3", "both"))
    parser.add_argument("--config", default="configs/evidence_policy_comparison.yaml")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--records")
    parser.add_argument("--goal-policy")
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--limit-conversations", type=int)
    parser.add_argument("--cells", nargs="+", choices=CELL_METHODS)
    args = parser.parse_args(argv)
    if args.phase in {"frozen", "both"} and args.records is None:
        parser.error("--records is required for frozen actual-history comparison")
    config = yaml.safe_load(Path(args.config).read_text())
    config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    from bench.study_workloads import load_goal_policy
    policy = load_goal_policy(args.goal_policy) if args.goal_policy else None
    result = asyncio.run(compare(config, matrix, dataset=args.dataset, split=args.split, output=args.output,
        phase=args.phase, frozen_records=args.records, goal_policy=policy, limit_states=args.limit_states,
        limit_conversations=args.limit_conversations, cells=args.cells))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
