"""P0/P1 × R0/R1 on frozen actual history and each pair's own P3 history."""
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
from controller.serialization import decode, to_jsonable
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


def recorded_terminal_budget(directory):
    import sqlite3
    from controller.session_store import Attempt, load_record
    database = directory/"state"/"sessions.sqlite3"
    if not database.exists():
        return []
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    failed = [load_record(connection, Attempt, payload) for payload, in connection.execute(
        "SELECT payload FROM attempts WHERE stage='FAILED'")]
    connection.close()
    known = []
    for attempt in failed:
        if terminal_budget_text(attempt.error):
            known.append(attempt.error)
        else:
            raise ValueError(f"unrecognized prior failed attempt: {attempt.error}")
    return known


def cell_config(config, policy, reader):
    settings = config["policy_comparison"]
    asset = settings["policies"][policy]
    reader_spec = settings["readers"][reader]
    run_path = reader_spec["run_config"]
    actual = merge_config(config, {"generation": {"run_config": run_path,
        "teacher_url": settings["teacher_url"], "helper_run_config": settings["helper_run_config"]},
        "task_model": {"device": settings["encoder_device"]},
        "support_model": {"mode": "trained", "asset": asset, "preference_asset": asset,
            "device": settings["support_device"]}, "selection": {"utility": "learned"},
        "execution": {"prefix_retention": {"mode": "native"}, "plan_gain": None},
        "reading": {"layout": "source_first", "packing": "source_windows"}})
    metadata = json.loads((Path(asset)/"support-model.json").read_text())
    identity = {"policy": policy, "reader": reader, "policy_asset": asset,
        "policy_heads_sha256": file_digest(Path(asset)/"heads.pt"),
        "policy_checkpoint": metadata.get("policy_checkpoint"),
        "reader_binding_used_for_policy_training": metadata.get("reader_binding"),
        "reader_run_config_sha256": file_digest(run_path),
        "evaluation_scope": "actual-prefix utility before/after reader-bound preferences; not strongest Incumbent selection"}
    if policy == "P1":
        from training.train_evidence_policy import require_policy_reader_binding
        trained = metadata["reader_binding"]
        require_policy_reader_binding(asset, reader_checkpoint=trained["reader_checkpoint"],
                                      reader_profile_id=trained["reader_profile_id"])
        identity["cross_reader_transfer_cell"] = reader != "R1"
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


async def frozen_cell(config, matrix, dataset, split, records, directory, *, policy, reader):
    from adapters.study_retrieval import create_study_retriever
    from bench.native_metrics import NativePoolSampler, pool_series
    from bench.metrics import occupancy_integral
    from bench.quality import export_evaluation
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.config import apply_study_config, create_controller, load_config
    from controller.engine_client import EngineError
    actual, reader_spec, identity = cell_config(config, policy, reader)
    spec = matrix["datasets"][dataset]
    actual = method_config(actual, {"id": policy+"_"+reader, "strategy": "B7"}, spec, split)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"records").mkdir(exist_ok=True)
    run = apply_study_config(load_config(reader_spec["run_config"]), actual)
    run = replace(run, state_dir=str(directory/"state"),
        engine=replace(run.engine, base_url=reader_spec["base_url"]),
        planner=replace(run.planner, strategy="always_rebuild"), prefix_retention={"mode": "native"})
    service = create_controller(run)
    devices = sorted(set((*actual["policy_comparison"]["frozen_shared_gpu_devices"], *reader_spec["gpu_devices"])))
    ports = [urlparse(run.engine.base_url).port]
    physical = StudyResourceSampler(devices, listening_processes(ports))
    await physical.start()
    started = time.monotonic()
    from adapters.history_state import HistoryIndex
    from adapters.hybrid_retrieval import LocalBGE
    from training.plan_gain_protocol import AuditedTaskInterpreter
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
    sources = {row["task_id"]: row for row in dataset_rows(spec, split)}
    results = {row["task_id"]: row for path in (directory/"records").glob("*.json") for row in [json.loads(path.read_text())]}
    windows = {}
    try:
        for index, frozen in enumerate(records):
            task_id, query = frozen["task_id"], frozen["question"]
            if task_id in results:
                continue
            source = sources[task_id]
            session = "policy-frozen:"+policy+":"+reader+":"+digest(task_id)[:20]
            history = decode(tuple[TurnRecord, ...], frozen["actual_history"])
            before = service.store.read_snapshot(session, provider.corpus_snapshot_id)
            before = service.store.sync_history(session, history, before.version)
            original, frozen_pool, frozen_origin = frozen_inputs(frozen)
            interpreter.bind(frozen, original.task_state)
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
                "method": policy+"_"+reader, "policy_identity": identity,
                "measurement_kind": "hardware" if run.engine.kind == "sglang" else "simulation",
                "question": query, "actual_history": frozen["actual_history"], "history_origin": "frozen_actual_rollout",
                "frozen_input_origin": frozen_origin, "frozen_task_state": to_jsonable(original.task_state),
                "frozen_candidate_pool_sha256": digest(to_jsonable(frozen_pool)),
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
            if window is not None:
                windows[task_id] = window
            results[task_id] = value
            write_json(directory/"records"/f"{index:06}.json", value)
            if response:
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
            "expected_tasks": [row["task_id"] for row in records], "completed": sum(row["status"] == "completed" for row in ordered),
            "failed": sum(row["status"] == "failed" for row in ordered), "history_protocol": "same frozen actual QA for all four cells",
            "engine": info, "provider": provider.manifest, "reference_used_online": False})
    return ordered


async def own_p3_cell(config, matrix, dataset, split, output, *, policy, reader, goal_policy=None, limit_conversations=None):
    from controller.engine_client import EngineError
    actual, reader_spec, identity = cell_config(config, policy, reader)
    spec = matrix["datasets"][dataset]
    if spec.get("protocol", "P3") != "P3":
        raise ValueError("own-history comparison requires P3, not imported authored history")
    grouped = {}
    for row in dataset_rows(spec, split):
        grouped.setdefault(row["conversation_id"], []).append(row)
    identities = sorted(grouped)[:limit_conversations]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    local_matrix = merge_config(matrix, {"resources": {"gpu_devices": sorted(set(
        (*config["policy_comparison"]["shared_gpu_devices"], *reader_spec["gpu_devices"])))}})
    # run_dataset uses the reader's explicit base URL through this scoped config.
    reader_run = yaml.safe_load(Path(reader_spec["run_config"]).read_text())
    reader_run["engine"]["base_url"] = reader_spec["base_url"]
    run_path = output/"reader_run.yaml"
    run_path.write_text(yaml.safe_dump(reader_run, sort_keys=False))
    actual["generation"]["run_config"] = str(run_path)
    collected, failed = [], []
    for conversation in identities:
        directory = output/digest(conversation)[:20]
        directory.mkdir(parents=True, exist_ok=True)
        task_path = directory/"tasks.jsonl"
        write_rows(task_path, grouped[conversation])
        one_matrix = merge_config(local_matrix, {"datasets": {dataset: {"inputs": {split: str(task_path)}}}})
        recorded = recorded_terminal_budget(directory)
        if recorded:
            collected.extend(rows(directory/"results.unscored.jsonl"))
            failed.append(persist_p3_failure(directory, conversation, "; ".join(recorded)))
            continue
        try:
            result = await run_dataset(actual, one_matrix, {"id": policy+"_"+reader, "strategy": "B7"},
                dataset, split, directory, concurrency=1, question_fn=goal_policy.question_fn if goal_policy else None)
            collected.extend(result)
        except EngineError as exc:
            if exc.code != "READ_STEP_OUTPUT_TRUNCATED":
                raise
            collected.extend(rows(directory/"results.unscored.jsonl"))
            failed.append(persist_p3_failure(directory, conversation, str(exc)))
        except RuntimeError as exc:
            if not terminal_model_budget(exc):
                raise
            collected.extend(rows(directory/"results.unscored.jsonl"))
            failed.append(persist_p3_failure(directory, conversation, str(exc)))
        except ValueError as exc:
            if not terminal_model_budget(exc):
                raise
            collected.extend(rows(directory/"results.unscored.jsonl"))
            failed.append(persist_p3_failure(directory, conversation, str(exc)))
        except ExceptionGroup as group:
            known, unknown = group.split(terminal_model_budget)
            if unknown is not None:
                raise unknown
            collected.extend(rows(directory/"results.unscored.jsonl"))
            failed.append(persist_p3_failure(directory, conversation, str(known)))
    write_rows(output/"results.unscored.jsonl", collected)
    write_json(output/"manifest.json", {"policy_identity": identity, "history_protocol": "P3 along each pair's own committed answers",
        "requested_conversations": identities, "completed_turns": len(collected), "failed_conversations": failed})
    return collected


def persist_p3_failure(directory, conversation, error, *, teacher_ledger=None):
    import sqlite3
    from controller.context_schema import StepRecord
    from controller.schema import SessionSnapshot
    from controller.session_store import Attempt, load_record
    connection = sqlite3.connect(f"file:{(directory/'state'/'sessions.sqlite3').resolve()}?mode=ro", uri=True)
    attempts = [load_record(connection, Attempt, payload) for payload, in connection.execute(
        "SELECT payload FROM attempts WHERE stage='FAILED'")]
    records = []
    for attempt in attempts:
        steps = [{"stage": stage, "record": to_jsonable(load_record(connection, StepRecord, payload))}
            for stage, payload in connection.execute("SELECT stage,payload FROM execution_steps WHERE attempt_id=? ORDER BY rowid",
                                                     (attempt.attempt_id,))]
        snapshot = load_record(connection, SessionSnapshot, connection.execute(
            "SELECT snapshot FROM sessions WHERE session_id=?", (attempt.session_id,)).fetchone()[0])
        helper_calls = ([row for row in rows(teacher_ledger) if row["task_id"] == attempt.request_id]
                        if teacher_ledger is not None and Path(teacher_ledger).exists() else [])
        records.append({"attempt": to_jsonable(attempt), "step_records": steps, "quality": None,
            "failure_category": "terminal_model_budget", "qa_increment": snapshot.version-attempt.expected_version,
            "request_window": None, "batch_cost_directory": str(directory),
            "helper_calls_raw": helper_calls, "helper_calls_source": str(teacher_ledger) if teacher_ledger is not None else None,
            "helper_call_attribution": "raw same-task records retained; no invented unique-call cost for a prior window"})
    connection.close()
    failure = {"conversation_id": conversation, "error": error, "generation_dir": str(directory),
        "descendants": "not generated after uncommitted failed turn", "failed_attempts": records}
    write_json(directory/"terminal_read_failures.json", failure)
    return failure


async def compare(config, matrix, *, dataset, split, output, phase, frozen_records=None,
                  goal_policy=None, limit_states=None, limit_conversations=None, cells=None):
    records = rows(frozen_records)[:limit_states] if phase in {"frozen", "both"} else []
    if any(row["status"] != "completed" for row in records):
        raise ValueError("frozen comparison states must come from completed actual QA records")
    result = []
    for policy in ("P0", "P1"):
        for reader in ("R0", "R1"):
            cell = policy+"_"+reader
            if cells is not None and cell not in cells:
                continue
            if phase in {"frozen", "both"}:
                selected = await frozen_cell(config, matrix, dataset, split, records, Path(output)/"frozen"/cell,
                                             policy=policy, reader=reader)
                result.append({"cell": cell, "phase": "frozen", "observed_records": len(selected)})
            if phase in {"p3", "both"}:
                selected = await own_p3_cell(config, matrix, dataset, split, Path(output)/"p3"/cell,
                    policy=policy, reader=reader, goal_policy=goal_policy, limit_conversations=limit_conversations)
                result.append({"cell": cell, "phase": "p3", "observed_records": len(selected)})
    write_json(Path(output)/"comparison_manifest.json", {"cells": result, "quality": "unscored; score saved actual outputs separately",
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
    parser.add_argument("--cells", nargs="+", choices=("P0_R0", "P0_R1", "P1_R0", "P1_R1"))
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
