"""Prepare, run, freeze and report the task-driven evidence/reading study."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import time
import uuid
from urllib.parse import urlparse

import yaml

from adapters.mtrag import load_rows
from bench.quality import export_evaluation
from bench.study_metrics import paired_comparisons, select_development, summarize_methods
from controller.config import apply_study_config, create_controller, load_config
from controller.schema import OnlineContext, OnlineMessage, OnlineTurn, TurnRecord
from controller.serialization import dumps, to_jsonable


BASELINE = "85f4ed4b1bc84de657691153d74e81a5ae9b4341"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)+"\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(row)+"\n" for row in rows))


def save_measurement_segment(directory, segment_id, *, physical, native, batch):
    """A resumed execution preserves all earlier raw observations and idle cost."""
    directory = Path(directory)
    segment = directory/"measurements"/segment_id
    segment.mkdir(parents=True, exist_ok=False)
    for name, value in (("physical_resources", physical), ("native_resources", native), ("batch_cost", batch)):
        write_json(segment/f"{name}.json", value)
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
    if task_ids is not None:
        by_id = {row["task_id"]: row for row in rows}
        return [by_id[task_id] for task_id in task_ids]
    if "split_manifest" in spec:
        allowed = set(json.loads(Path(spec["split_manifest"]).read_text())[split])
        rows = [row for row in rows if row["conversation_id"] in allowed]
    if "domain" in spec:
        rows = [row for row in rows if row["Collection"] == spec["domain"]]
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
            for name, spec in matrix["datasets"].items()
            if (datasets is None or name in datasets) and not (split == "dev" and spec.get("evaluation_only"))}}
    matrix["cohorts"] = cohorts
    return matrix


def prepare(config, matrix=None):
    from training.task_data import build_task_data
    from training.reading_data import build_reading_data

    output = Path(config["study"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    task_data = build_task_data(config)
    reading_data = build_reading_data(config)
    from bench.study_assets import prepare_public_assets
    evaluation_assets = prepare_public_assets(config)
    baseline = {"commit": BASELINE,
        "query_adapter": "models/bge-m3-dialogue-query-v1",
        "generation_config": "configs/qwen3_32b.yaml",
        "retrieval_config": "configs/study_multidoc_retriever.yaml",
        "strategy": "Dense5_REBUILD_native_Radix_HiCache"}
    write_json(output/"baseline.json", baseline)
    write_json(output/"effective_config.json", config)
    manifest = {"baseline": baseline, "task_data": task_data,
        "reading_data": reading_data, "evaluation_assets": evaluation_assets,
        "learning": config["study"].get("learning", True), "configured_assets": {
            "task": config["task_model"]["asset"], "support": config["support_model"]["asset"],
            "preference": config["support_model"]["preference_asset"]}}
    if matrix is not None:
        from bench.study_jobs import development_variants
        write_json(output/"development_jobs.json", development_variants(matrix))
        # Evaluation assets are inventoried independently from training builders.
        manifest["datasets"] = {name: {"split_kind": spec["split_kind"],
            "primary_metric": spec["primary_metric"], "weight": spec["weight"],
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
                      limit_conversations=None, workload=None, question_fn=None):
    from adapters.study_retrieval import create_study_retriever
    from controller.evidence import make_evidence
    from bench.native_metrics import NativePoolSampler, pool_series
    from bench.metrics import occupancy_integral
    from bench.study_resources import (StudyResourceSampler, attribute_physical,
        integrate_physical, listening_processes)
    from bench.study_workloads import WorkloadDispatcher

    spec = matrix["datasets"][dataset_name]
    declared_tasks = matrix.get("cohorts", {}).get(split, {}).get(dataset_name)
    source_rows = dataset_rows(spec, split, task_ids=declared_tasks)
    reuse_provider = spec.get("reuse_provider", False)
    continue_conversation_failures = spec.get("continue_conversation_failures", False)
    if workload is not None:
        concurrency = workload.concurrency
    config = method_config(config, method, spec, split)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    segment_id = uuid.uuid4().hex
    run = load_config(config["generation"]["run_config"])
    run = replace(apply_study_config(run, config), state_dir=str(directory/"state"))
    if "answer_system_prompt" in spec:
        run = replace(run, context=replace(run.context, system_prompt=spec["answer_system_prompt"]))
    service = create_controller(run)
    provider_started = time.monotonic()
    provider = await asyncio.to_thread(create_study_retriever, config, service.template)
    service.bind_resident_models(provider.manifest)
    initialization_ms = (time.monotonic()-provider_started)*1000
    engine_info = await service.engine.info()
    cache_protocol = config["execution"].get("cache_start", "observed")
    cache_start = (await service.engine.profile_prepare("cold")
                   if cache_protocol == "cold" and not reuse_provider else None)
    engine_info = await service.engine.info()
    allocator = await service.engine.metrics()
    grouped = {}
    for number, row in enumerate(source_rows):
        grouped.setdefault(row["conversation_id"], []).append((number, row))
    conversations = sorted(grouped)
    if limit_conversations is not None:
        conversations = conversations[:limit_conversations]
    if reuse_provider and (concurrency != 1 or workload is not None
            or any(len(grouped[sid]) != 1 or grouped[sid][0][1]["turn"] != 1 for sid in conversations)):
        raise ValueError("reuse_provider requires serial independent one-turn conversations")
    expected = [row["task_id"] for sid in conversations for _, row in grouped[sid]]
    if workload is not None and set(workload.dispatch_order) != set(expected):
        raise ValueError("workload and dataset must contain exactly the same task IDs")
    generation_manifest = {"config": config, "method": method, "measurement_segment_id": segment_id,
        "dataset": dataset_name, "split": split, "task_ids": [row["task_id"] for row in source_rows],
        "scheduled_task_ids": expected, "cohort_declared": declared_tasks is not None,
        "concurrency": concurrency, "retrieval": provider.manifest,
        "concurrency_scope": "request",
        "response_latency_basis": "fixed scheduled arrival when supplied, otherwise observed arrival before semaphore, through committed response",
        "workload": to_jsonable(workload), "answer_driven_policy": question_fn is not None,
        "cache_start_protocol": cache_protocol, "cache_start_record": cache_start,
        "reuse_provider": reuse_provider, "conversation_cache_boundary": "cold" if reuse_provider else None,
        "auxiliary_initialization_ms": initialization_ms,
        "engine": engine_info, "allocator_initial": allocator,
        "model": run.model, "generation_protocol": run.sampling,
        "measurement_kind": "hardware" if run.engine.kind == "sglang" else "simulation"}
    write_json(directory/"generation_manifest.json", generation_manifest)
    (directory/"records").mkdir(exist_ok=True)
    old_records = {row["task_id"]: row for path in (directory/"records").glob("*.json")
                   for row in [json.loads(path.read_text())]}
    results, windows, request_starts = dict(old_records), {}, {}
    failures = {row["conversation_id"]: row
        for path in (directory/"terminal_read_failures").glob("*.json")
        for row in [json.loads(path.read_text())]} if reuse_provider else {}
    semaphore = asyncio.Semaphore(concurrency)
    resource_settings = matrix["resources"]
    ports = [urlparse(run.engine.base_url).port]
    if config["task_model"].get("mode") == "teacher" or config["support_model"].get("mode") == "teacher":
        ports.append(urlparse(config["generation"]["teacher_url"]).port)
    process_ids = (*listening_processes(ports), *resource_settings.get("process_ids", ()))
    devices = list(resource_settings["gpu_devices"])
    physical = StudyResourceSampler(devices, process_ids)
    native = NativePoolSampler(run.engine.base_url, gpu_capacity=allocator["pool"]["gpu_capacity_tokens"],
        host_enabled=allocator["pool"]["host_capacity_tokens"] > 0, model=run.model.name)
    await physical.start()
    await native.start()
    measured_start = time.monotonic()
    dispatcher = WorkloadDispatcher(workload) if workload is not None else None
    if dispatcher is not None:
        dispatcher.restore_completed(results.keys())

    async def one_turn(sid, number, row, arrival=None, workload_arrival=None):
        session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
        start = time.monotonic() if arrival is None else arrival
        request_starts[row["task_id"]] = start
        query = row["input"][-1]["text"]
        before = service.store.read_snapshot(session_id, provider.corpus_snapshot_id)
        protocol = spec.get("protocol", "P3")
        if protocol == "P1" and not before.raw_history:
            given = row["input"][:-1]
            reference_history = tuple(TurnRecord(index//2+1,
                given[index]["text"], given[index+1]["text"]) for index in range(0, len(given), 2))
            before = service.store.sync_history(session_id, reference_history, before.version)
        decision = question_fn(row, before.raw_history) if question_fn is not None else None
        if decision is not None:
            query = decision.question
        history = tuple(message for turn in before.raw_history for message in (
            OnlineMessage("user", turn.user_text), OnlineMessage("assistant", turn.assistant_text)))
        request_id = row["task_id"]
        if (row.get("coverage_mode") or spec.get("supplied_documents",
                dataset_name in {"loong", "long_dev", "multihop_dev"})):
            supplied = tuple(make_evidence(context["text"], context["document_id"],
                corpus_id=dataset_name, passage_id=context["document_id"],
                version=context.get("version", "v1"),
                source_uri=context.get("source") or f"{dataset_name}://{context['document_id']}",
                title=context.get("title", "")) for context in row["contexts"])
            metadata = {block.key: {key: context[key] for key in (
                "document_family", "valid_from", "valid_to", "published_at", "applicability",
                "table_id", "heading", "current", "version_order") if key in context}
                for block, context in zip(supplied, row["contexts"], strict=True)}
            async def retrieve(snapshot):
                return await provider.retrieve_supplied(snapshot, query, request_id, supplied,
                    source_metadata=metadata, coverage_scope=row.get("coverage_mode",
                        spec.get("coverage_mode", "selected_sources")))
        else:
            async def retrieve(snapshot):
                return await provider.retrieve(snapshot, query, request_id)
        async def extend(plan, open_needs, claims):
            return await provider.extend_for_plan(before, plan, open_needs, claims)
        response = await service.serve_turn(session_id, query, request_id=request_id,
            corpus_snapshot_id=provider.corpus_snapshot_id, evidence_provider=retrieve,
            open_need_provider=extend, request_arrived=start)
        end = time.monotonic()
        attempt = service.store.get_attempt(response.attempt_id)
        snapshot = service.store.get_snapshot(session_id)
        blocks = {block.key: block for block in snapshot.archive}
        selected = tuple(blocks[key] for key in sorted(attempt.plan.candidate.required_blocks))
        turn = OnlineTurn(sid, request_id, row["turn"], row["Collection"], query, history,
            tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri,
                                block.title) for block in selected), protocol,
            query_context=attempt.query_context)
        export_evaluation(turn, response.assistant_text, official_path=directory/"official.jsonl",
            audit_path=directory/"audit.jsonl", visible_evidence_manifest=response.visible_evidence_manifest,
            audit_evidence=tuple(blocks.values()), input_ids=attempt.plan.candidate.input_ids,
            output_ids=response.output_ids, truncated=response.truncated,
            citation_scope=response.citation_scope, measurement_kind="hardware",
            reading_outcome=attempt.reading_outcome, reference_map=attempt.plan.candidate.reference_map)
        trace = provider.traces.pop((session_id, request_id), {})
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
            "task_id": request_id, "conversation_id": sid, "turn": row["turn"],
            "status": "completed", "question": query, "actual_history": to_jsonable(before.raw_history),
            "history_origin": "author_reference_import" if protocol == "P1" else "model_actual",
            "answer": response.assistant_text, "response": to_jsonable(response),
            "task_state": trace.get("task_state"), "candidate_pool": trace.get("pool"),
            "planning_trace": trace, "context_plan": to_jsonable(attempt.context_plan),
            "reading_outcome": to_jsonable(attempt.reading_outcome), "metrics": {}, "cost": cost,
            "request_window": [start, end], "record_number": number,
            "measurement_segment_id": segment_id,
            "generation_dir": str(directory.resolve())}
        value["workload_arrival"] = workload_arrival
        value["question_policy"] = to_jsonable(decision)
        results[request_id] = value
        windows[request_id] = (start, end)
        write_json(directory/"records"/f"{number:06d}.json", value)
        print(dumps({"method": method["id"], "dataset": dataset_name, "task": request_id,
                     "complete_ms": response.complete_ms}), flush=True)

    async def conversation(sid):
        session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
        for number, row in sorted(grouped[sid], key=lambda item: item[1]["turn"]):
            if row["task_id"] in results:
                continue
            snapshot = service.store.read_snapshot(session_id, provider.corpus_snapshot_id)
            workload_arrival = (await dispatcher.wait_turn(row, snapshot.raw_history)
                                if dispatcher is not None else None)
            arrival = (dispatcher.origin+workload_arrival["actual_arrival_seconds"]
                       if dispatcher is not None else time.monotonic())
            request_starts[row["task_id"]] = arrival
            async with semaphore:
                await one_turn(sid, number, row, arrival, workload_arrival)
            if dispatcher is not None:
                dispatcher.complete_turn(row["task_id"])
        await service.close_session(session_id)

    async def isolated_conversation(sid):
        client_started = time.monotonic()
        try:
            await conversation(sid)
        except Exception as error:
            if dispatcher is not None:
                await dispatcher.cancel_conversation(sid, str(error))
            ordered_rows = sorted(grouped[sid], key=lambda item: item[1]["turn"])
            row = next((row for _, row in ordered_rows if row["task_id"] not in results), ordered_rows[-1][1])
            session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
            attempt = service.store.find_request(session_id, row["task_id"])
            failure = dict(conversation_id=sid, task_id=row["task_id"], session_id=session_id,
                error_type=type(error).__name__, error=str(error),
                request_window=[request_starts[row["task_id"]] if row["task_id"] in request_starts else client_started,
                                time.monotonic()],
                failure_window_scope="failed_request_wall" if row["task_id"] in request_starts else "client_before_submission",
                measurement_segment_id=segment_id,
                attempt_id=attempt.attempt_id if attempt is not None else None,
                state_db=str((directory/"state"/"sessions.sqlite3").resolve()),
                blocked_task_ids=[value["task_id"] for _, value in ordered_rows
                    if value["task_id"] not in results and value["task_id"] != row["task_id"]])
            failures[sid] = failure
            try:
                await service.close_session(session_id)
            except Exception as cleanup_error:
                failure["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}

    try:
        if reuse_provider:
            from bench.evidence_policy_comparison import (persist_p3_failure,
                terminal_budget_text, terminal_model_budget)
            from training.reader_data import digest
            for sid in conversations:
                number, row = grouped[sid][0]
                if row["task_id"] in results or sid in failures:
                    continue
                session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
                previous = service.store.find_request(session_id, row["task_id"])
                if previous is not None and previous.stage == "FAILED":
                    if not terminal_budget_text(previous.error):
                        raise ValueError(f"unrecognized prior failed attempt: {previous.error}")
                    if sid not in failures:
                        failures[sid] = persist_p3_failure(directory, sid, row["task_id"], session_id,
                            previous, previous.error)
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
                    attempt = service.store.find_request(session_id, row["task_id"])
                    failure = persist_p3_failure(directory, sid, row["task_id"], session_id, attempt, str(error))
                    failure.update(request_window=[request_start, time.monotonic()],
                        measurement_segment_id=segment_id)
                    failures[sid] = failure
                    write_json(directory/"terminal_read_failures"/(digest(sid)[:20]+".json"), failure)
                    await service.close_session(session_id)
        elif continue_conversation_failures:
            await asyncio.gather(*(isolated_conversation(sid) for sid in conversations))
        else:
            async with asyncio.TaskGroup() as group:
                for sid in conversations:
                    group.create_task(conversation(sid), name=f"study:{method['id']}:{dataset_name}:{split}:{sid}")
    finally:
        dispatch_window = await dispatcher.finish_arrivals() if dispatcher is not None else None
        measured_end = time.monotonic()
        await native.stop()
        await physical.stop()
        observation = native.observation()
        failure_windows = {("failure", sid): failure["request_window"]
            for sid, failure in failures.items() if failure.get("measurement_segment_id") == segment_id}
        memory_costs = attribute_physical(physical.samples, {**windows, **failure_windows})
        for task_id, window in windows.items():
            raw_window = integrate_physical(physical.samples, *window)
            results[task_id]["cost"].update(memory_costs[task_id], **{name: raw_window.get(name)
                for name in ("gpu_peak_gib", "host_peak_gib", "complete_window_coverage")})
            results[task_id]["physical_window"] = raw_window
            series, metadata = pool_series(observation, *window)
            results[task_id]["native_resource_window"] = {**occupancy_integral(series), **metadata}
            write_json(directory/"records"/f"{results[task_id]['record_number']:06d}.json", results[task_id])
        segment = save_measurement_segment(directory, segment_id,
            physical=physical.observation(), native=observation, batch={"start": measured_start, "end": measured_end,
            "physical": integrate_physical(physical.samples, measured_start, measured_end),
            "attribution": "physical interval occupancy divided across concurrently active requests",
            "idle_cost": "included in batch physical integral, retained outside per-request attribution"})
        write_json(segment/"generation_manifest.json", generation_manifest)
        if reuse_provider or continue_conversation_failures:
            for sid, failure in failures.items():
                key = ("failure", sid)
                if key in failure_windows:
                    window = failure_windows[key]
                    raw_window = integrate_physical(physical.samples, *window)
                    failure["cost"] = {"complete_ms": (window[1]-window[0])*1000, **memory_costs[key],
                        **{name: raw_window.get(name) for name in
                           ("gpu_peak_gib", "host_peak_gib", "complete_window_coverage")}}
                    failure["physical_window"] = raw_window
                if reuse_provider:
                    failure_key = digest(failure["conversation_id"])[:20]
                    write_json(directory/"terminal_read_failures"/(failure_key+".json"), failure)
            write_json(directory/"failed_conversations.json", list(failures.values()))
        if dispatcher is not None:
            write_json(directory/"workload_dispatch.json", dict(origin=dispatcher.origin,
                clock="client_monotonic", observation_window=dispatch_window,
                actual_arrivals=dispatcher.arrivals, cancelled=dispatcher.cancelled,
                completion_seconds=dispatcher._completion_times,
                independent_clients=workload.dispatch_semantics == "independent_clients_after_own_prior_commit"))
        ordered = [results[key] for key in expected if key in results]
        write_jsonl(directory/"results.unscored.jsonl", ordered)
        references = []
        for row in source_rows:
            record = results.get(row["task_id"])
            decision = record["question_policy"] if record is not None else None
            if decision is None:
                references.append(row)
                continue
            metadata = decision["metadata"]
            actual = [message for turn in record["actual_history"] for message in (
                {"speaker": "user", "text": turn["user_text"]},
                {"speaker": "assistant", "text": turn["assistant_text"]})]
            references.append({**row, "input": [*actual, {"speaker": "user", "text": record["question"]}],
                "goal_conditions": metadata["goal_conditions"],
                "offline_reference": metadata["offline_reference"],
                "goal_id": decision["goal_id"], "policy_protocol": metadata["protocol"]})
        write_jsonl(directory/"references.jsonl", references)
        provider.close()
        await service.aclose()
        service.store.close()
    return ordered


async def evaluate(config, matrix, output, *, split, frozen=None, methods=None, datasets=None,
                   concurrency=1, limit_conversations=None, score=True, workload=None, question_fn=None):
    from bench.study_scoring import score_dataset

    output = Path(output)
    write_json(output/"effective_config.json", config)
    selected_methods = matrix["methods"]
    if frozen is not None:
        selected_methods = frozen["test_methods"]
    if methods is not None:
        selected_methods = [m for m in selected_methods if m["id"] in methods]
    results = []
    for method in selected_methods:
        for name, dataset in matrix["datasets"].items():
            if split == "dev" and dataset.get("evaluation_only"):
                continue
            if datasets is not None and name not in datasets:
                continue
            directory = output/method["id"]/name
            rows = await run_dataset(config, matrix, method, name, split, directory,
                concurrency=concurrency, limit_conversations=limit_conversations,
                workload=workload[name] if isinstance(workload, dict) else workload, question_fn=question_fn)
            if score:
                rows = await score_dataset(config, dataset, rows, directory,
                    expected_task_ids=matrix.get("cohorts", {}).get(split, {}).get(name))
            results.extend(rows)
            write_jsonl(output/"results.jsonl", results)
    return results


def freeze(config, matrix, rows, path):
    selected = select_development(matrix, rows)
    method = next(m for m in matrix["methods"] if m["id"] == selected["method"])
    test_ids = [*matrix["selection"]["test_comparators"], method["id"]]
    by_id = {m["id"]: m for m in matrix["methods"]}
    test_methods = [by_id[identity] for identity in dict.fromkeys(test_ids)]
    missing = [name for name in matrix["datasets"] if name not in matrix.get("cohorts", {}).get("test", {})]
    matrix = declare_cohorts(matrix, splits=("test",), datasets=missing)
    frozen = {"schema_version": 1, "baseline_commit": BASELINE, "method_spec": method,
        "test_methods": test_methods,
        "selected": selected, "matrix": matrix, "config": config,
        "learning": config["study"].get("learning", True),
        "assets": {"task_backend": config["task_model"]["mode"],
                   "support_backend": config["support_model"]["mode"],
                   "task": (config["task_model"]["asset"]
                            if config["task_model"]["mode"] == "trained" else None),
                   "support": (config["support_model"]["asset"]
                               if config["support_model"]["mode"] == "trained" else None)},
        "development_results": str(Path(path).parent/"dev"/"results.jsonl")}
    write_json(path, frozen)
    return frozen


def report(frozen, rows, directory):
    directory = Path(directory)
    summaries = summarize_methods(frozen["matrix"], rows, split="test")
    paired = paired_comparisons(frozen["matrix"], rows)
    write_json(directory/"summaries.json", summaries)
    write_json(directory/"paired_intervals.json", paired)
    text = ["# ContextKV 任务状态驱动证据规划与预算化阅读", "",
            f"评审基线：`{BASELINE}`。冻结方案：`{frozen['method_spec']['id']}`。", "",
            f"新增训练：{frozen['learning']}。任务后端：{frozen['assets']['task_backend']}；"
            f"支持后端：{frozen['assets']['support_backend']}。", "",
            "| 方法 | 工程 Q | 状态 |", "|---|---:|---|"]
    for method, summary in summaries.items():
        text.append(f"| {method} | {summary.get('quality')} | {summary.get('eligible')} |")
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
        "`paired_intervals.json` 按会话或文档任务重采样，保留原始指标单位及成本差。"
        "等成本标记同时要求计算量与完整响应落入预定匹配区间。", "",
        "冻结文件保存全部开发方案的质量、完整成本、预算资格与最终选择。"
        "GPU/Host 物理积分按并发请求分摊，空闲保留占用在各 `measurements/*/batch_cost.json` 中记录。", "",
        "评分模型调用成本保存在每组 `scoring/manifest.json`，与在线系统成本分开。", ""]
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"report.md").write_text("\n".join(text))
    return {"report": str((directory/"report.md").resolve()), "summaries": summaries,
            "paired_intervals": paired}


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
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--unscored", action="store_true")
    parser.add_argument("--workload-plan")
    parser.add_argument("--goal-policy")
    parser.add_argument("--cohorts", help="Existing split/dataset task-ID manifest to bind into the matrix")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    if args.cohorts:
        matrix["cohorts"] = json.loads(Path(args.cohorts).read_text())
    root = Path(config["study"]["output_dir"])
    if args.command == "prepare":
        result = prepare(config, matrix)
    elif args.command == "cohorts":
        missing = [name for name in matrix["datasets"]
            if (args.datasets is None or name in args.datasets)
            and name not in matrix.get("cohorts", {}).get(args.split, {})]
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
        goal_policy = (load_goal_policy(args.goal_policy,
            expected_sha256=workload.goal_policy_sha256 if workload else None) if args.goal_policy else None)
        frozen = json.loads(Path(args.frozen).read_text()) if args.frozen else None
        if frozen:
            config, matrix = frozen["config"], frozen["matrix"]
        else:
            if args.split == "dev" and matrix["selection"].get("include_ablations", False):
                from bench.study_jobs import development_variants
                matrix = {**matrix, "methods": development_variants(matrix)}
            missing = [name for name in matrix["datasets"]
                if (args.datasets is None or name in args.datasets)
                and name not in matrix.get("cohorts", {}).get(args.split, {})]
            matrix = declare_cohorts(matrix, splits=(args.split,), datasets=missing)
        output = Path(args.output or root/args.split)
        write_json(output/"matrix.json", matrix)
        result = asyncio.run(evaluate(config, matrix, output, split=args.split, frozen=frozen,
            methods=args.methods, datasets=args.datasets, concurrency=args.concurrency,
            limit_conversations=args.limit_conversations, score=not args.unscored,
            workload=workload, question_fn=goal_policy.question_fn if goal_policy else None))
        result = {"rows": len(result), "output": str(output.resolve())}
    elif args.command == "freeze":
        rows = read_jsonl(args.results)
        recorded = Path(args.results).parent/"matrix.json"
        matrix = json.loads(recorded.read_text())
        config = json.loads((Path(args.results).parent/"effective_config.json").read_text())
        result = freeze(config, matrix, rows, args.output or root/"frozen.json")
    else:
        frozen = json.loads(Path(args.frozen).read_text())
        rows = read_jsonl(args.results or root/"test"/"results.jsonl")
        result = report(frozen, rows, args.output or root/"report")
    print(dumps(result))


if __name__ == "__main__":
    main()
