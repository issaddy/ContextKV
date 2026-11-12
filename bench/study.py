"""Prepare, run, freeze and report the task-driven evidence/reading study."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import time
import uuid
from urllib.parse import urlparse

import yaml

from adapters.mtrag import load_rows
from bench.quality import export_evaluation
from bench.result_records import retain_metrics, write_task_metrics
from bench.study_costs import failed_request_cost
from bench.study_metrics import _track_specs, paired_comparisons, select_development, summarize_methods
from controller.config import apply_study_config, create_controller, load_config
from controller.schema import (Message, OnlineContext, OnlineMessage, OnlineTurn,
                               from_history_messages, history_messages)
from controller.serialization import dumps, to_jsonable


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def saved_study_rows(directory):
    """Read the dataset's single result journal."""
    export = Path(directory)/"results.jsonl"
    if export.exists():
        return read_jsonl(export)
    index = Path(directory)/"result_sets.json"
    if index.exists():
        return [row for source in json.loads(index.read_text())["datasets"] for row in saved_study_rows(source)]
    return []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)+"\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(row)+"\n" for row in rows))


async def close_failed_session(service, session_id):
    """Require durable Native cleanup before a P1 continuation is admitted.

    Controller.close_session may defer RETIRE_PENDING or unknown owner release
    outcomes to its maintenance outbox. A return from that method alone is not
    a release receipt, so the benchmark must not immediately reopen the session.
    """
    await service.close_session(session_id)
    retirements = service.store.outbox(session_id)
    owners = tuple(owner for owner in service.store.prefix_owners(session_id, active_only=True)
                   if owner.policy == "joint_demand")
    if retirements or owners:
        raise RuntimeError("session cleanup unacknowledged: "+dumps({
            "retirement_ids": [operation.operation_id for operation in retirements],
            "demand_owners": [{"owner_id": owner.owner_id, "stage": owner.stage} for owner in owners]}))


def save_measurement_segment(directory, segment_id, *, batch):
    """Save the measured resource totals for this execution segment."""
    directory = Path(directory)
    segment = directory/"measurements"/segment_id
    segment.mkdir(parents=True, exist_ok=False)
    write_json(segment/"batch_cost.json", batch)
    write_json(directory/"measurement_segments.json", {"schema_version": 1,
        "segments": [str(path.relative_to(directory)) for path in sorted((directory/"measurements").iterdir())
                     if (path/"batch_cost.json").exists()]})
    return segment


def merge_config(base, overrides):
    result = dict(base)
    for name, value in overrides.items():
        result[name] = (merge_config(result[name] if isinstance(result.get(name), dict) else {}, value) if isinstance(value, dict)
                        else value)
    return result


def dataset_rows(spec, split, task_ids=None):
    path = spec["inputs"][split] if "inputs" in spec else spec["input"]
    rows = load_rows(path)
    if "split_manifest" in spec:
        allowed = set(json.loads(Path(spec["split_manifest"]).read_text())[split])
        rows = [row for row in rows if row["conversation_id"] in allowed]
    if "domain" in spec:
        rows = [row for row in rows if row["Collection"] == spec["domain"]]
    if task_ids is not None:
        by_id = {row["task_id"]: row for row in rows}
        return [by_id[task_id] for task_id in task_ids]
    if split == "dev" and "dev_conversations" in spec:
        identities = sorted({row["conversation_id"] for row in rows})
        selected = set(random.Random(spec["dev_seed"]).sample(identities, spec["dev_conversations"]))
        rows = [row for row in rows if row["conversation_id"] in selected]
    return rows


def declare_cohorts(matrix, *, splits=("dev", "test"), datasets=None):
    matrix = merge_config(matrix, {})
    cohorts = dict(matrix.get("cohorts", {}))
    for split in splits:
        cohorts[split] = {**cohorts.get(split, {}), **{name: [row["task_id"] for row in dataset_rows(spec, split)]
            for name, spec in _track_specs(matrix, split, datasets).items()}}
    matrix["cohorts"] = cohorts
    return matrix


def prepare(config, matrix=None):
    output = Path(config["study"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    from bench.study_assets import prepare_public_assets
    evaluation_assets = prepare_public_assets(config)
    write_json(output/"effective_config.json", config)
    manifest = {"evaluation_assets": evaluation_assets}
    if matrix is not None:
        # Evaluation assets are inventoried independently from training builders.
        manifest["datasets"] = {name: {"split_kind": spec["split_kind"],
            "primary_metric": spec["primary_metric"],
            "paths": spec.get("inputs", {"input": spec.get("input")})}
            for name, spec in matrix["datasets"].items()}
    write_json(output/"prepare_manifest.json", manifest)
    return manifest


def method_config(config, method, dataset, split):
    retrieval = {name: dataset[key] for name, key in (
        ("corpus_path", "corpus_path"), ("config_path", "retriever_config"),
        ("corpus_id", "corpus_id")) if key in dataset}
    reading = dict(dataset.get("reading", {}))
    reading["coverage_mode"] = dataset.get("coverage_mode", reading.get("coverage_mode", "selected_sources"))
    selection = dict(dataset.get("selection", {}))
    selection["max_passages"] = dataset.get("max_passages", selection.get("max_passages"))
    value = merge_config(config, {"study": {"strategy": method["strategy"]},
        "retrieval": retrieval, "reading": reading,
        "selection": selection})
    value = merge_config(value, {"retrieval": dataset.get("retrieval_by_split", {}).get(split, {})})
    return merge_config(value, method.get("overrides", {}))


async def run_dataset(config, matrix, method, dataset_name, split, directory, *, concurrency=1,
                      limit_conversations=None, limit_turns=None, workload=None, question_fn=None, request_namespace=None):
    from adapters.study_retrieval import create_study_retriever
    from controller.evidence import make_evidence
    from bench.study_resources import (StudyResourceSampler, attribute_physical,
        integrate_physical, listening_processes)
    from bench.study_workloads import WorkloadDispatcher

    spec = matrix["datasets"][dataset_name]
    online_started = time.monotonic()
    declared_tasks = matrix.get("cohorts", {}).get(split, {}).get(dataset_name)
    source_rows = dataset_rows(spec, split, task_ids=declared_tasks)
    catalogue_path = spec.get("supplied_source_catalogue")
    supplied_catalogue = ({row["source_id"]: {**row, "document_id": row["source_id"], "source": row["source_uri"]}
                           for row in read_jsonl(catalogue_path)} if catalogue_path else None)
    reuse_provider = spec.get("reuse_provider", False)
    protocol = spec.get("protocol", matrix.get("protocol", "P3"))
    independent_reference_turns = protocol == "P1" and question_fn is None
    continue_conversation_failures = spec.get("continue_conversation_failures", independent_reference_turns)

    def actual_request_id(task_id):
        return task_id if request_namespace is None else f"{request_namespace}:{task_id}"

    if workload is not None:
        concurrency = workload.concurrency
    grouped = {}
    for number, row in enumerate(source_rows):
        grouped.setdefault(row["conversation_id"], []).append((number, row))
    conversations = sorted(grouped)[:limit_conversations]
    if limit_turns is not None:
        remaining, scheduled = limit_turns, []
        for sid in conversations:
            if remaining <= 0:
                break
            grouped[sid] = sorted(grouped[sid], key=lambda item: item[1]["turn"])[:remaining]
            scheduled.append(sid)
            remaining -= len(grouped[sid])
        conversations = scheduled
    if reuse_provider and (concurrency != 1 or workload is not None
            or any(len(grouped[sid]) != 1 or grouped[sid][0][1]["turn"] != 1 for sid in conversations)):
        raise ValueError("reuse_provider requires serial independent one-turn conversations")
    expected = [row["task_id"] for sid in conversations for _, row in grouped[sid]]
    if workload is not None and set(workload.dispatch_order) != set(expected):
        raise ValueError("workload and dataset must contain exactly the same task IDs")

    config = method_config(config, method, spec, split)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    old_records = {row["task_id"]: row for row in saved_study_rows(directory)}
    if expected and set(expected) <= old_records.keys():
        return [old_records[key] for key in expected]
    segment_id = uuid.uuid4().hex
    run = load_config(config["generation"]["run_config"])
    run = replace(apply_study_config(run, config), state_dir=str(directory/"state"))
    resource_settings = matrix["resources"]
    ports = [urlparse(run.engine.base_url).port]
    if config["task_model"].get("mode") == "teacher" or config["support_model"].get("mode") == "teacher":
        ports.append(urlparse(config["generation"]["teacher_url"]).port)
    process_ids = tuple(sorted({os.getpid(), *listening_processes(ports), *resource_settings.get("process_ids", ())}))
    devices = tuple(resource_settings["gpu_devices"])
    if run.plan_gain is not None and run.plan_gain["mode"] == "observed_actions":
        run = replace(run, plan_gain={**run.plan_gain, "gpu_devices": devices, "process_ids": process_ids})
    if run.state_observation is not None:
        run = replace(run, state_observation={**run.state_observation,
            "gpu_devices": devices, "process_ids": process_ids})
    if "answer_system_prompt" in spec:
        run = replace(run, context=replace(run.context, system_prompt=spec["answer_system_prompt"]))
    service = create_controller(run)
    provider_started = time.monotonic()
    provider = await asyncio.to_thread(create_study_retriever, config, service.template)
    from controller.qa import DocumentQA
    qa = DocumentQA(service, provider)
    service.bind_resident_models(provider.manifest)
    await qa.start()
    initialization_ms = (time.monotonic()-provider_started)*1000
    engine_info = await service.engine.info()
    cache_protocol = config["execution"].get("cache_start", "observed")
    cache_start = (await service.engine.profile_prepare("cold")
                   if cache_protocol == "cold" and not reuse_provider else None)
    engine_info = await service.engine.info()
    allocator = await service.engine.metrics()
    generation_manifest = {"config": config, "method": method, "measurement_segment_id": segment_id,
        "dataset": dataset_name, "split": split, "protocol": protocol,
        "task_ids": [row["task_id"] for row in source_rows],
        "scheduled_task_ids": expected, "cohort_declared": declared_tasks is not None,
        "concurrency": concurrency, "retrieval": provider.manifest,
        "concurrency_scope": "request", "request_namespace": request_namespace,
        "response_latency_basis": "fixed scheduled arrival when supplied, otherwise observed arrival before semaphore, through committed response",
        "workload": to_jsonable(workload), "answer_driven_policy": question_fn is not None,
        "cache_start_protocol": cache_protocol, "cache_start_record": cache_start,
        "reuse_provider": reuse_provider, "conversation_cache_boundary": "cold" if reuse_provider else None,
        "auxiliary_initialization_ms": initialization_ms,
        "resource_scope": {"gpu_devices": devices, "process_roots": process_ids},
        "engine": engine_info, "allocator_initial": allocator,
        "model": run.model, "generation_protocol": run.sampling,
        "measurement_kind": "hardware" if run.engine.kind == "sglang" else "simulation",
        "capture_training": run.state_observation is not None or bool(config.get("policy_comparison"))}
    write_json(directory/"generation_manifest.json", generation_manifest)
    capture_training = generation_manifest["capture_training"]
    results, windows, request_starts = dict(old_records), {}, {}
    failures = {row["task_id"]: row
        for path in (directory/"terminal_read_failures").glob("*.json")
        for row in [json.loads(path.read_text())]} if reuse_provider else {}
    failure_path = directory/"failed_conversations.json"
    if continue_conversation_failures and not reuse_provider and failure_path.exists():
        pending = set(expected)-results.keys()
        failures = {row["task_id"]: row for row in json.loads(failure_path.read_text())
            if row["task_id"] in pending and row["request_id"] == actual_request_id(row["task_id"])
            and (row.get("attempt") or {}).get("stage") == "FAILED"}
    for failure in failures.values():
        attempt = service.store.find_request(failure["session_id"],
            failure.get("request_id", actual_request_id(failure["task_id"])))
        if attempt is not None and "accounting" not in failure.get("cost", {}):
            failure["cost"] = {**failure.get("cost", {}), **failed_request_cost(service.store, attempt)}
    semaphore = asyncio.Semaphore(concurrency)
    physical = StudyResourceSampler(devices, process_ids)
    await physical.start()
    measured_start = time.monotonic()
    dispatcher = WorkloadDispatcher(workload) if workload is not None else None
    if dispatcher is not None:
        dispatcher.restore_completed(results.keys())
        dispatcher.restore_failed(failures.keys())

    async def one_turn(sid, number, row, arrival=None, workload_arrival=None):
        session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
        start = time.monotonic() if arrival is None else arrival
        request_starts[row["task_id"]] = start
        query = row["input"][-1]["text"]
        before = service.store.read_snapshot(session_id, provider.corpus_snapshot_id)
        if protocol == "P1":
            reference_history = from_history_messages(tuple(
                Message("assistant" if item["speaker"] == "agent" else item["speaker"], item["text"])
                for item in row["input"][:-1]))
            before = service.store.sync_history(session_id, reference_history, before.version)
        decision = question_fn(row, before.raw_history) if question_fn is not None else None
        if decision is not None:
            query = decision.question
        history = tuple(OnlineMessage(message.role, message.content)
                        for message in history_messages(before.raw_history))
        task_id = row["task_id"]
        request_id = actual_request_id(task_id)
        supplied, metadata = None, None
        if (row.get("coverage_mode") or spec.get("supplied_documents",
                dataset_name in {"loong", "long_dev", "multihop_dev"})):
            contexts = (tuple(supplied_catalogue[key] for key in row["scope_source_ids"])
                        if supplied_catalogue is not None else row["contexts"])
            supplied = tuple(make_evidence(context["text"], context["document_id"],
                corpus_id=dataset_name, passage_id=context["document_id"],
                version=context.get("version", "v1"),
                source_uri=context.get("source") or f"{dataset_name}://{context['document_id']}",
                title=context.get("title", "")) for context in contexts)
            metadata = {block.key: {key: context[key] for key in (
                "document_family", "valid_from", "valid_to", "published_at", "applicability",
                "table_id", "heading", "current", "version_order") if key in context}
                for block, context in zip(supplied, contexts, strict=True)}
        response = await qa.ask(session_id, query, request_id, task_id=task_id,
            supplied_blocks=supplied, source_metadata=metadata, request_arrived=start,
            coverage_scope=row.get("coverage_mode", spec.get("coverage_mode", "selected_sources")))
        end = time.monotonic()
        attempt = service.store.get_attempt(response.attempt_id)
        snapshot = service.store.get_snapshot(session_id)
        blocks = {block.key: block for block in snapshot.archive}
        selected = tuple(blocks[key] for key in sorted(attempt.plan.candidate.required_blocks))
        turn = OnlineTurn(sid, task_id, row["turn"], row["Collection"], query, history,
            tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri,
                                block.title) for block in selected), protocol,
            query_context=attempt.query_context)
        export_evaluation(turn, response.assistant_text, official_path=directory/"official.jsonl",
            audit_path=directory/"audit.jsonl", visible_evidence_manifest=response.visible_evidence_manifest,
            audit_evidence=tuple(blocks.values()), input_ids=attempt.plan.candidate.input_ids,
            output_ids=response.output_ids, truncated=response.truncated,
            citation_scope=response.citation_scope, measurement_kind="hardware",
            reading_outcome=attempt.reading_outcome, reference_map=attempt.plan.candidate.reference_map)
        trace = provider.take_trace(session_id, task_id)
        internal_ms = [step["elapsed_ms"] for step in response.step_metrics if "elapsed_ms" in step]
        model_work_ms = (response.engine_elapsed_ms+sum(internal_ms)
                         if response.engine_elapsed_ms is not None
                         and all(value is not None for value in internal_ms) else None)
        cost = {"complete_ms": (end-start)*1000,
            "total_input_tokens": response.total_input_tokens,
            "total_output_tokens": response.total_output_tokens,
            "reader_input_tokens": response.reader_input_tokens,
            "reader_output_tokens": response.reader_output_tokens,
            "auxiliary_input_tokens": response.auxiliary_input_tokens,
            "auxiliary_output_tokens": response.auxiliary_output_tokens,
            "auxiliary_ms": response.auxiliary_model_ms,
            "gpu_capacity_tokens": allocator["pool"]["gpu_capacity_tokens"],
            "host_capacity_tokens": allocator["pool"]["host_capacity_tokens"],
            "actual_prefill_tokens": response.total_prefill_tokens,
            "actual_device_hit_tokens": response.total_device_hit_tokens,
            "actual_host_loaded_tokens": response.total_host_loaded_tokens,
            "online_compute_ms": (model_work_ms+response.auxiliary_model_ms
                if model_work_ms is not None and response.auxiliary_model_ms is not None else None),
            "online_compute_basis": "summed generation service elapsed and measured auxiliary pipeline time",
            "gpu_gib_seconds": None, "host_gib_seconds": None}
        value = {"method": method["id"], "dataset": dataset_name, "split": split,
            "task_id": task_id, "request_id": request_id, "conversation_id": sid,
            "turn": snapshot.raw_history[-1].turn_id,
            "status": "completed", "question": query,
            "history_origin": "author_reference_import" if protocol == "P1" else "model_actual", "protocol": protocol,
            "source_partition": split,
            "measurement_kind": "hardware" if run.engine.kind == "sglang" else "simulation",
            **(service.state_observer.cohort if service.state_observer is not None else {}),
            "answer": response.assistant_text,
            "response": {key: getattr(response, key) for key in (
                "attempt_id", "action", "output_tokens", "truncated", "read_count", "reduce_count")},
            "metrics": {}, "cost": cost,
            "request_window": [start, end], "record_number": number,
            "measurement_segment_id": segment_id,
            "generation_dir": str(directory.resolve())}
        if capture_training:
            value.update(actual_history=to_jsonable(before.raw_history), reference=row,
                response=to_jsonable(response), task_state=trace.get("task_state"),
                candidate_pool=trace.get("pool"), planning_trace=trace,
                context_plan=to_jsonable(attempt.context_plan), reading_outcome=to_jsonable(attempt.reading_outcome))
        value["workload_arrival"] = workload_arrival
        value["question_policy"] = to_jsonable(decision)
        if decision is not None:
            value["actual_history"] = to_jsonable(before.raw_history)
        results[task_id] = value
        windows[task_id] = (start, end)
        with (directory/"results.jsonl").open("a") as stream:
            stream.write(dumps(value)+"\n")

    async def record_failure(sid, row, error, client_started, *, blocked_rows=()):
        """Settle this request before releasing its session and scheduling another."""
        session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
        attempt = service.store.find_request(session_id, actual_request_id(row["task_id"]))
        start = request_starts.get(row["task_id"], client_started)
        failure = dict(conversation_id=sid, task_id=row["task_id"], session_id=session_id,
            request_id=actual_request_id(row["task_id"]), dataset=dataset_name, protocol=protocol,
            error_type=type(error).__name__, error=str(error),
            request_window=[start, time.monotonic()],
            failure_window_scope="failed_request_wall" if row["task_id"] in request_starts else "client_before_submission",
            measurement_segment_id=segment_id,
            attempt_id=attempt.attempt_id if attempt is not None else None,
            state_db=str((directory/"state"/"sessions.sqlite3").resolve()),
            blocked_task_ids=[value["task_id"] for value in blocked_rows],
            blocked_tasks=[{name: value[name] for name in ("task_id", "turn")} for value in blocked_rows],
            reference=row, source_partition=split, split=split, turn=row["turn"],
            status="failed", method=method["id"], metrics={},
            measurement_kind=generation_manifest["measurement_kind"],
            question=attempt.query if attempt is not None else row["input"][-1]["text"],
            history_origin="author_reference_import" if protocol == "P1" else "model_actual",
            attempt={"attempt_id": attempt.attempt_id, "stage": attempt.stage} if attempt is not None else None,
            cost={**failed_request_cost(service.store, attempt),
                "gpu_capacity_tokens": allocator["pool"]["gpu_capacity_tokens"],
                "host_capacity_tokens": allocator["pool"]["host_capacity_tokens"],
                "gpu_gib_seconds": None, "host_gib_seconds": None},
            **(service.state_observer.cohort if service.state_observer is not None else {}))
        if not capture_training:
            failure.pop("reference", None)
        failures[row["task_id"]] = failure
        try:
            await close_failed_session(service, session_id)
        except Exception as cleanup_error:
            # An unresolved native request must not be abandoned or replaced
            # silently. Reference-history independence cannot repair that state.
            failure["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
        failure["request_window"][1] = time.monotonic()
        failure["cost"]["complete_ms"] = (failure["request_window"][1]-start)*1000
        return failure

    async def conversation(sid, *, isolate_failures=False):
        session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
        pending = sorted(((number, row) for number, row in grouped[sid]
            if row["task_id"] not in results and row["task_id"] not in failures),
            key=lambda item: item[1]["turn"])
        if pending and (request_namespace is not None or independent_reference_turns):
            service.store.reopen_session(session_id)
        for position, (number, row) in enumerate(pending):
            client_started = time.monotonic()
            try:
                snapshot = service.store.read_snapshot(session_id, provider.corpus_snapshot_id)
                workload_arrival = (await dispatcher.wait_turn(row, snapshot.raw_history,
                    predecessor_terminal=independent_reference_turns) if dispatcher is not None else None)
                arrival = (dispatcher.origin+workload_arrival["actual_arrival_seconds"]
                           if dispatcher is not None else time.monotonic())
                request_starts[row["task_id"]] = arrival
                async with semaphore:
                    await one_turn(sid, number, row, arrival, workload_arrival)
            except Exception as error:
                if not isolate_failures:
                    raise
                later = [value for _, value in pending[position+1:]]
                failure = await record_failure(sid, row, error, client_started,
                    blocked_rows=() if independent_reference_turns else later)
                if dispatcher is not None:
                    await dispatcher.fail_turn(row["task_id"])
                if independent_reference_turns and "cleanup_error" not in failure:
                    service.store.reopen_session(session_id)
                    continue
                if independent_reference_turns:
                    failure.update(blocked_task_ids=[value["task_id"] for value in later],
                        blocked_tasks=[{name: value[name] for name in ("task_id", "turn")} for value in later],
                        blocked_reason="session_cleanup_unresolved")
                if dispatcher is not None:
                    await dispatcher.cancel_conversation(sid, failure.get("blocked_reason", str(error)))
                return
            if dispatcher is not None:
                dispatcher.complete_turn(row["task_id"])
        await service.close_session(session_id)

    async def isolated_conversation(sid):
        previous = tuple(failure for failure in failures.values() if failure["conversation_id"] == sid)
        if previous and not independent_reference_turns:
            if dispatcher is not None:
                await dispatcher.cancel_conversation(sid, previous[0]["error"])
            return
        unresolved = tuple(failure for failure in previous if failure.get("cleanup_error"))
        if unresolved:
            # A new measured segment cannot bypass the Native cleanup gate.
            # Its bounded drain is accounted in this segment's physical window,
            # without rewriting the original failed-request latency.
            error = None
            try:
                await close_failed_session(service, unresolved[0]["session_id"])
            except Exception as cleanup_error:
                error = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
            for failure in unresolved:
                if error:
                    failure["cleanup_error"] = error
                else:
                    failure.pop("cleanup_error", None)
                    failure.pop("blocked_reason", None)
                    failure.update(blocked_task_ids=[], blocked_tasks=[])
            if error:
                if dispatcher is not None:
                    await dispatcher.cancel_conversation(sid, "session_cleanup_unresolved")
                return
        await conversation(sid, isolate_failures=True)

    dispatch_window = None
    try:
        if reuse_provider:
            from bench.evidence_policy_comparison import (persist_p3_failure,
                terminal_budget_text, terminal_model_budget)
            from training.reader_data import digest
            for sid in conversations:
                number, row = grouped[sid][0]
                if row["task_id"] in results or row["task_id"] in failures:
                    continue
                session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
                previous = service.store.find_request(session_id, actual_request_id(row["task_id"]))
                if previous is not None and previous.stage == "FAILED":
                    if not terminal_budget_text(previous.error):
                        raise ValueError(f"unrecognized prior failed attempt: {previous.error}")
                    if row["task_id"] not in failures:
                        failure = persist_p3_failure(directory, sid, row["task_id"], session_id,
                            previous, previous.error)
                        failure.update(reference=row, source_partition=split, split=split,
                            dataset=dataset_name, protocol=protocol,
                            turn=row["turn"], status="failed", method=method["id"],
                            measurement_kind="hardware" if run.engine.kind == "sglang" else "simulation",
                            question=previous.query,
                            attempt={"attempt_id": previous.attempt_id, "stage": previous.stage},
                            history_origin="model_actual",
                            cost=failed_request_cost(service.store, previous),
                            **(service.state_observer.cohort if service.state_observer is not None else {}))
                        failures[row["task_id"]] = failure
                        write_json(directory/"terminal_read_failures"/(digest(sid)[:20]+".json"), failure)
                    await service.close_session(session_id)
                    continue
                boundary_start = time.monotonic()
                cold = await service.engine.profile_prepare("cold")
                write_json(directory/"cold_boundaries"/(digest(sid)[:20]+f".{segment_id}.json"),
                    {"conversation_id": sid, "measurement_segment_id": segment_id,
                     "window": [boundary_start, time.monotonic()], "receipt": cold})
                request_start = time.monotonic()
                try:
                    await conversation(sid)
                except Exception as error:
                    if isinstance(error, ExceptionGroup):
                        _, unknown = error.split(terminal_model_budget)
                        if unknown is not None:
                            raise unknown
                    elif not terminal_model_budget(error):
                        raise
                    attempt = service.store.find_request(session_id, actual_request_id(row["task_id"]))
                    failure = persist_p3_failure(directory, sid, row["task_id"], session_id, attempt, str(error))
                    failure.update(request_window=[request_start, time.monotonic()],
                        measurement_segment_id=segment_id, reference=row, source_partition=split,
                        dataset=dataset_name, protocol=protocol,
                        split=split, turn=row["turn"], status="failed", method=method["id"],
                        measurement_kind="hardware" if run.engine.kind == "sglang" else "simulation",
                        question=attempt.query if attempt is not None else row["input"][-1]["text"],
                        history_origin="model_actual",
                        attempt={"attempt_id": attempt.attempt_id, "stage": attempt.stage} if attempt else None,
                        cost=failed_request_cost(service.store, attempt),
                        **(service.state_observer.cohort if service.state_observer is not None else {}))
                    failures[row["task_id"]] = failure
                    write_json(directory/"terminal_read_failures"/(digest(sid)[:20]+".json"), failure)
                    await service.close_session(session_id)
        elif continue_conversation_failures:
            await asyncio.gather(*(isolated_conversation(sid) for sid in conversations))
        else:
            async with asyncio.TaskGroup() as group:
                for sid in conversations:
                    group.create_task(conversation(sid), name=f"study:{method['id']}:{dataset_name}:{split}:{sid}")
        if dispatcher is not None:
            dispatch_window = await dispatcher.finish_arrivals()
    finally:
        measured_end = time.monotonic()
        if dispatcher is not None and dispatch_window is None:
            dispatch_window = {"start": dispatcher.origin, "end": measured_end}
        await physical.stop()
        failure_windows = {("failure", sid): failure["request_window"]
            for sid, failure in failures.items() if failure.get("measurement_segment_id") == segment_id}
        memory_costs = attribute_physical(physical.samples, {**windows, **failure_windows})
        for task_id, window in windows.items():
            raw_window = integrate_physical(physical.samples, *window)
            results[task_id]["cost"].update(memory_costs[task_id], **{name: raw_window.get(name)
                for name in ("gpu_peak_gib", "host_peak_gib", "complete_window_coverage")})
        segment = save_measurement_segment(directory, segment_id,
            batch={"start": measured_start, "end": measured_end,
            "physical": integrate_physical(physical.samples, measured_start, measured_end),
            "attribution": "physical interval occupancy divided across concurrently active requests",
            "idle_cost": "included in batch physical integral, retained outside per-request attribution"})
        if reuse_provider or continue_conversation_failures:
            for sid, failure in failures.items():
                key = ("failure", sid)
                if key in failure_windows:
                    window = failure_windows[key]
                    raw_window = integrate_physical(physical.samples, *window)
                    failure["cost"] = {**failure.get("cost", {}),
                        "complete_ms": (window[1]-window[0])*1000, **memory_costs[key],
                        **{name: raw_window.get(name) for name in
                           ("gpu_peak_gib", "host_peak_gib", "complete_window_coverage")}}
                if reuse_provider:
                    failure_key = digest(failure["conversation_id"])[:20]
                    write_json(directory/"terminal_read_failures"/(failure_key+".json"), failure)
            write_json(segment/"failed_conversations.json", [failures[sid] for _, sid in failure_windows])
            write_json(failure_path, list(failures.values()))
        if dispatcher is not None:
            write_json(directory/"workload_dispatch.json", dict(origin=dispatcher.origin,
                clock="client_monotonic", observation_window=dispatch_window,
                actual_arrivals=dispatcher.arrivals, cancelled=dispatcher.cancelled,
                completion_seconds=dispatcher._completion_times,
                terminal_seconds=dispatcher._terminal_times,
                independent_clients=workload.dispatch_semantics == "independent_clients_after_own_prior_commit"))
        ordered = [results[key] for key in expected if key in results]
        write_jsonl(directory/"results.jsonl", ordered)
        write_task_metrics(directory, ordered, failures=list(failures.values()))
        references = []
        for _, row in (item for sid in conversations for item in grouped[sid]):
            record = results.get(row["task_id"])
            decision = record.get("question_policy") if record is not None else None
            if decision is None:
                references.append({key: value for key, value in row.items() if key != "contexts"})
                continue
            metadata = decision["metadata"]
            actual = [message for turn in record["actual_history"] for message in (
                {"speaker": "user", "text": turn["user_text"]},
                {"speaker": "assistant", "text": turn["assistant_text"]})]
            references.append({**{key: value for key, value in row.items() if key != "contexts"},
                "input": [*actual, {"speaker": "user", "text": record["question"]}],
                "goal_conditions": metadata["goal_conditions"],
                "offline_reference": metadata["offline_reference"],
                "goal_id": decision["goal_id"], "policy_protocol": metadata["protocol"]})
        write_jsonl(directory/"references.jsonl", references)
        provider.close()
        await service.aclose()
        service.store.close()
        write_json(segment/"online_window.json", {"start": online_started, "end": time.monotonic(),
            "measurement_kind": generation_manifest["measurement_kind"],
            "clock": "client_monotonic", "scope": "online preparation through dataset cleanup"})
    return ordered


async def evaluate(config, matrix, output, *, split, frozen=None, methods=None, datasets=None,
                   concurrency=1, limit_conversations=None, limit_turns=None, score=True, workload=None, question_fn=None,
                   request_namespace=None):
    from bench.study_scoring import score_dataset

    output = Path(output)
    write_json(output/"effective_config.json", config)
    selected_methods = matrix["methods"]
    if frozen is not None:
        selected_methods = frozen["test_methods"]
    if methods is not None:
        selected_methods = [m for m in selected_methods if m["id"] in methods]
    generated = []
    for method in selected_methods:
        for name, dataset in _track_specs(matrix, split, datasets).items():
            directory = output/method["id"]/name
            rows = await run_dataset(config, matrix, method, name, split, directory,
                concurrency=concurrency, limit_conversations=limit_conversations, limit_turns=limit_turns,
                workload=workload[name] if isinstance(workload, dict) else workload, question_fn=question_fn,
                request_namespace=request_namespace)
            generated.append((name, dataset, directory, rows))

    results = []
    for name, dataset, directory, rows in generated:
        if score and any(row.get("scoring", {}).get("status") != "scored" for row in rows):
            rows = await score_dataset(config, dataset, rows, directory,
                expected_task_ids=matrix.get("cohorts", {}).get(split, {}).get(name))
        if score:
            manifest = json.loads((directory/"generation_manifest.json").read_text())
            if not manifest["capture_training"] and all(row["scoring"]["status"] == "scored" for row in rows):
                retain_metrics(directory, rows)
        write_task_metrics(directory, rows)
        results.extend(rows)
    write_json(output/"result_sets.json", {"datasets": [str(directory.resolve())
        for _, _, directory, _ in generated]})
    return results


def freeze(config, matrix, rows, path):
    selected = select_development(matrix, rows)
    method = next(m for m in matrix["methods"] if m["id"] == selected["method"])
    test_ids = [*matrix["selection"]["test_comparators"], method["id"]]
    by_id = {m["id"]: m for m in matrix["methods"]}
    test_methods = [by_id[identity] for identity in dict.fromkeys(test_ids)]
    missing = [name for name in matrix["datasets"] if name not in matrix.get("cohorts", {}).get("test", {})]
    matrix = declare_cohorts(matrix, splits=("test",), datasets=missing)
    frozen = {"schema_version": 1, "baseline_method": matrix["selection"]["baseline"], "method_spec": method,
        "test_methods": test_methods,
        "selected": selected, "matrix": matrix, "config": config,
        "learning": config["study"].get("learning", True),
        "assets": {"task_backend": config["task_model"]["mode"],
                   "support_backend": config["support_model"]["mode"],
                   "task": (config["task_model"]["asset"]
                            if config["task_model"]["mode"] == "trained" else None),
                   "support": (config["support_model"]["asset"]
                               if config["support_model"]["mode"] == "trained" else None)},
        "development_results": str(Path(path).parent/"dev"/"result_sets.json")}
    write_json(path, frozen)
    return frozen


def report(frozen, rows, directory):
    directory = Path(directory)
    summaries = summarize_methods(frozen["matrix"], rows, split="test")
    paired = paired_comparisons(frozen["matrix"], rows)
    write_json(directory/"summaries.json", summaries)
    write_json(directory/"comparisons.json", paired)
    text = ["# ContextKV 任务状态驱动证据规划与预算化阅读", "",
            f"比较基线：`{frozen['baseline_method']}`。冻结方案：`{frozen['method_spec']['id']}`。", "",
            f"新增训练：{frozen['learning']}。任务后端：{frozen['assets']['task_backend']}；"
            f"支持后端：{frozen['assets']['support_backend']}。", "",
            "各数据集使用自己的原始指标，分别报告。"]
    text += ["", "## 原始指标与题集", "",
        "| 方法 | 轨道 | 预定 | 完成 | 主指标成功评分 | 主指标 | 原始分数 |",
        "|---|---|---:|---:|---:|---|---:|"]
    for method, summary in summaries.items():
        for name, track in summary["datasets"].items():
            text.append(f"| {method} | {name} | {track['planned']} | {track['completed']} | "
                f"{track['scored']} | {track['primary_metric']} | {track['primary_score']} |")
    text += ["", "## 完整在线成本", "",
        "| 方法 | 平均响应 ms | P95 ms | 全模型输入 | 全模型输出 | 辅助输入 | 辅助输出 | GPU GiB·s | Host GiB·s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, summary in summaries.items():
        cost = summary["cost"]
        fields = ("mean_complete_ms", "p95_complete_ms", "total_input_tokens", "total_output_tokens",
                  "auxiliary_input_tokens", "auxiliary_output_tokens", "gpu_gib_seconds", "host_gib_seconds")
        text.append("| "+method+" | "+" | ".join(str(cost[field]) for field in fields)+" |")
    mechanism = {}
    for method in summaries:
        selected = [row for row in rows if row["method"] == method]
        responses = [row["response"] for row in selected]
        mechanism[method] = {"actions": dict(Counter(response["action"] for response in responses)),
            "read_steps": sum(response["read_count"] for response in responses),
            "reduce_steps": sum(response["reduce_count"] for response in responses),
            "truncated_answers": sum(response["truncated"] for response in responses),
            "error_attribution": dict(Counter(row.get("error_attribution", "unannotated") for row in selected))}
    write_json(directory/"mechanisms.json", mechanism)
    text += ["", "## 机制与误差", "",
        "逐题步骤、实际动作、截断及人工错误归因计数保存在 `mechanisms.json`。", "",
        "| 方法 | READ | REDUCE | 动作计数 | 截断 |", "|---|---:|---:|---|---:|"]
    for method, value in mechanism.items():
        text.append(f"| {method} | {value['read_steps']} | {value['reduce_steps']} | "
                    f"{value['actions']} | {value['truncated_answers']} |")
    text += ["", "## 配对比较与开发消融", "",
        "`comparisons.json` 按数据集报告质量差与完整响应 P95 变化。", "",
        "冻结文件保存全部开发方案的质量、完整成本、预算资格与最终选择。"
        "GPU/Host 物理积分按并发请求分摊，空闲保留占用在各 `measurements/*/batch_cost.json` 中记录。", "",
        "评分模型调用成本保存在每组 `scoring/manifest.json`，与在线系统成本分开。", ""]
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"report.md").write_text("\n".join(text))
    return {"report": str((directory/"report.md").resolve()), "summaries": summaries,
            "comparisons": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "cohorts", "workloads", "rollout", "evaluate", "freeze", "report"))
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--matrix", default="configs/study_matrix.yaml")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--frozen")
    parser.add_argument("--results")
    parser.add_argument("--output")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit-conversations", type=int)
    parser.add_argument("--limit-turns", type=int, help="Cap rollout turns, or scheduled turns per evaluation method and dataset")
    parser.add_argument("--unscored", action="store_true")
    parser.add_argument("--workload-plan")
    parser.add_argument("--goal-policy")
    parser.add_argument("--cohorts", help="Existing split/dataset task-ID manifest to bind into the matrix")
    parser.add_argument("--request-namespace", help="Explicit execution identity for a new attempt; logical task IDs stay fixed")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    if args.cohorts:
        matrix["cohorts"] = json.loads(Path(args.cohorts).read_text())["cohorts"]
    root = Path(config["study"]["output_dir"])
    if args.command == "prepare":
        result = prepare(config, matrix)
    elif args.command == "cohorts":
        missing = [name for name in _track_specs(matrix, args.split, args.datasets)
            if name not in matrix.get("cohorts", {}).get(args.split, {})]
        matrix = declare_cohorts(matrix, splits=(args.split,), datasets=missing)
        target = Path(args.output or root/"matrix.cohorts.json")
        write_json(target, matrix)
        result = {"matrix": str(target.resolve()), "split": args.split,
            "tasks": {name: len(identities) for name, identities in matrix["cohorts"][args.split].items()}}
    elif args.command == "workloads":
        from bench.study_workloads import prepare_workloads
        result = prepare_workloads(config, matrix)
    elif args.command == "rollout":
        from training.task_rollout import run_actual_rollouts
        result = asyncio.run(run_actual_rollouts(config, min_conversations=config["training"]["min_rollout_conversations"],
            min_turns=config["training"]["min_rollout_turns"], limit_turns=args.limit_turns,
            concurrency=args.concurrency))
    elif args.command == "evaluate":
        from bench.study_workloads import read_workload_manifest, load_goal_policy
        workload = read_workload_manifest(args.workload_plan) if args.workload_plan else None
        goal_policy = load_goal_policy(args.goal_policy) if args.goal_policy else None
        frozen = json.loads(Path(args.frozen).read_text()) if args.frozen else None
        if frozen:
            config, matrix = frozen["config"], frozen["matrix"]
        else:
            missing = [name for name in _track_specs(matrix, args.split, args.datasets)
                if name not in matrix.get("cohorts", {}).get(args.split, {})]
            matrix = declare_cohorts(matrix, splits=(args.split,), datasets=missing)
        output = Path(args.output or root/args.split)
        write_json(output/"matrix.json", matrix)
        result = asyncio.run(evaluate(config, matrix, output, split=args.split, frozen=frozen,
            methods=args.methods, datasets=args.datasets, concurrency=args.concurrency,
            limit_conversations=args.limit_conversations, limit_turns=args.limit_turns, score=not args.unscored,
            workload=workload, question_fn=goal_policy.question_fn if goal_policy else None,
            request_namespace=args.request_namespace))
        result = {"rows": len(result), "output": str(output.resolve())}
    elif args.command == "freeze":
        rows = read_jsonl(args.results) if Path(args.results).is_file() else saved_study_rows(args.results)
        results_root = Path(args.results).parent if Path(args.results).is_file() else Path(args.results)
        recorded = results_root/"matrix.json"
        matrix = json.loads(recorded.read_text())
        config = json.loads((results_root/"effective_config.json").read_text())
        result = freeze(config, matrix, rows, args.output or root/"frozen.json")
    else:
        frozen = json.loads(Path(args.frozen).read_text())
        rows = (read_jsonl(args.results) if Path(args.results).is_file() else saved_study_rows(args.results)) if args.results else saved_study_rows(root/"test")
        result = report(frozen, rows, args.output or root/"report")
    print(dumps(result))


if __name__ == "__main__":
    main()
