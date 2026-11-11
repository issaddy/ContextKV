"""Run current methods on declared dev/test datasets with complete response costs."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
from tempfile import TemporaryFile
import time
from urllib.parse import urlparse

import yaml

from bench.metrics import TurnTiming, occupancy_integral, summarize_timings
from bench.native_metrics import NativePoolSampler, pool_series
from bench.plan_result_summary import batch_ledger
from bench.reassess_saved import paid_execution_index
from bench.study import declare_cohorts, dataset_rows, read_jsonl, saved_study_rows, write_json
from bench.study_metrics import (_cost_summary, _track_specs, declared_queue_summary,
                                 evaluation_track_protocol)
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from bench.study_workloads import declare_fixed_arrivals, freeze_workload, read_workload_manifest


def source_file(path):
    return {"path": str(Path(path).resolve())}


CONTEXT_METHODS = ("context_native", "context_cache_order", "context_cache_demand", "context_no_fact_reuse")
CONVENTIONAL_METHODS = {"RAG_native": "rag", "LlamaIndex_native": "llamaindex"}
RULE_CONTEXT_METHOD = "Rule_contextpilot_joint_demand"
FULL_QUEUE_METHODS = ("RAG_native", "context_cache_demand", "LlamaIndex_native")
OPTIONAL_FULL_QUEUE_METHODS = ("context_native", "context_cache_order", "context_no_fact_reuse", RULE_CONTEXT_METHOD)


def full_queue_method_specs(selected, gain_settings, compilation, *, rule_settings=None, demand_policy=None):
    """Materialize the three systems and their current single-factor controls."""
    specs = []
    for method in selected:
        gain, rule = method in CONTEXT_METHODS, method == RULE_CONTEXT_METHOD
        context = gain or rule
        external = method in CONVENTIONAL_METHODS
        dialogue = CONVENTIONAL_METHODS.get(method, "contextkv")
        contextpilot = context and method != "context_native"
        rolling = deepcopy(gain_settings if gain else rule_settings) if context else None
        if context:
            rolling["joint_context"] = True
        retention = {"mode": "native"}
        if method in {"context_cache_demand", "context_no_fact_reuse", RULE_CONTEXT_METHOD}:
            declared = demand_policy or {}
            retention = dict(mode="demand", allocation="next_use", ttl_seconds=15,
                             horizon_seconds_choices=[15, 60, 180], protection_budget_tokens=8192)
            retention.update({name: deepcopy(declared[name]) for name in (
                "allocation", "ttl_seconds", "horizon_seconds_choices", "protection_budget_tokens",
                "lambda_gpu", "lambda_host") if name in declared})
            if retention["allocation"] == "paired_value" and declared.get("pricing_asset"):
                retention["pricing_asset"] = str(Path(declared["pricing_asset"]).resolve())
        overrides = {"execution": {"dialogue_mode": dialogue, "plan_gain": rolling,
            "cache_compilation": deepcopy(compilation) if contextpilot else {"mode": "native"},
            "prefix_retention": retention},
            "reading": {"fact_reuse": context and method != "context_no_fact_reuse",
                        "numeric_record_projection": context}}
        specs.append(dict(id=method, strategy="B7" if context else "B6", overrides=overrides,
            evaluation_factors=dict(B="gain" if gain else "rule" if rule else "fixed",
                compilation="contextpilot" if contextpilot else "native",
                representation="finite" if context else "existing", retention=retention["mode"],
                retention_allocation=retention.get("allocation"), numeric_record_projection=context,
                fact_reuse=overrides["reading"]["fact_reuse"], dialogue=dialogue,
                role=("main_baseline" if method == "RAG_native" else "external_reference" if external
                      else "fact_reuse_control" if method == "context_no_fact_reuse"
                      else "rule_comparator" if rule else "context_method"))))
    return specs


def full_queue_gap_limits(execution, gain=None, *, max_units=None, max_passages=None):
    """Bind gain actions to their measured limits; explicit CLI values take precedence."""
    inherited = (gain.provenance["continuation_gap_limits"] if gain is not None else
        {"max_units": 96, "max_passages": 10,
         **((execution.get("state_observation") or {}).get("gap_limits") or {})})
    return {"max_units": inherited["max_units"] if max_units is None else max_units,
            "max_passages": inherited["max_passages"] if max_passages is None else max_passages}


def prepare(args, root):
    from controller.config import load_config
    from controller.context_schema import DEFAULT_FULL_SCOPE_OUTPUT_TOKENS
    from training.train_plan_gain import PlanGainModel

    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("reading", {}).setdefault("full_scope_output_tokens", DEFAULT_FULL_SCOPE_OUTPUT_TOKENS)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    registered = matrix["final_evaluation"]
    split = args.split
    if args.cohorts is not None:
        matrix["cohorts"] = yaml.safe_load(Path(args.cohorts).read_text())["cohorts"]
    specs = _track_specs(matrix, split, datasets=args.datasets)
    datasets = tuple(specs)
    matrix["datasets"] = {name: matrix["datasets"][name] for name in datasets}
    arrival_path = Path(args.arrival_plan) if args.arrival_plan else None
    arrivals, fixed = {}, None
    if arrival_path is not None:
        arrival_spec = json.loads(arrival_path.read_text())
        if arrival_spec.get("format") == "contextkv_fixed_arrivals_v1":
            fixed = arrival_spec["datasets"]
        else:
            arrivals = read_workload_manifest(arrival_path)
            if not isinstance(arrivals, dict):
                if len(datasets) != 1:
                    raise ValueError("a single workload requires one selected dataset")
                arrivals = {datasets[0]: arrivals}
        if args.cohorts is None:
            matrix.setdefault("cohorts", {})[split] = {
                name: list(fixed[name]["seconds"] if fixed is not None else arrivals[name].dispatch_order)
                for name in datasets}
    missing = [name for name in datasets if name not in matrix.get("cohorts", {}).get(split, {})]
    matrix = declare_cohorts(matrix, splits=(split,), datasets=missing)
    cohorts = {name: matrix["cohorts"][split][name] for name in datasets}
    matrix["cohorts"] = {split: cohorts}
    source_rows = {name: dataset_rows(matrix["datasets"][name], split, task_ids=cohorts[name])
                   for name in datasets}
    sessions = {name: len({row["conversation_id"] for row in rows}) for name, rows in source_rows.items()}
    run_path = Path(config["generation"]["run_config"]).resolve()
    run_data = yaml.safe_load(run_path.read_text())
    run_data["engine"]["base_url"] = args.reader_base_url
    run_path = root/"reader-run.yaml"
    run_path.write_text(yaml.safe_dump(run_data, sort_keys=False))
    run = load_config(run_path)
    config["generation"]["run_config"] = str(run_path)
    selected_methods = tuple(dict.fromkeys(args.methods or registered["main_methods"]))
    needs_gain = any(method in CONTEXT_METHODS for method in selected_methods)
    gain_path = Path(args.gain_asset).resolve() if needs_gain else None
    profile = json.loads(gain_path.read_text())["reader_profile_id"] if needs_gain else args.reader_profile_id
    gain = (PlanGainModel.load(gain_path, reader_checkpoint=run.model.revision, reader_profile_id=profile)
        if needs_gain else None)
    roots = args.online_process_roots
    if not roots:
        endpoints = [run.engine.base_url]
        if config["generation"].get("teacher_url"):
            endpoints.append(config["generation"]["teacher_url"])
        roots = set(listening_processes([urlparse(url).port for url in endpoints]))
        if not args.prepare_only:
            roots.add(os.getpid())
        roots = sorted(roots)
    scope = dict(gpu_devices=args.online_gpu_devices, process_roots=roots,
        native_endpoint=run.engine.base_url, gpu_capacity_tokens=run.engine.gpu_capacity_tokens,
        host_capacity_tokens=run.engine.host_capacity_tokens,
        host_enabled=run.engine.host_capacity_tokens > 0, model=run.model.name)
    matrix["resources"].update(gpu_devices=args.online_gpu_devices, process_ids=args.online_process_roots or ())
    execution = config["execution"]
    if args.lambda_time is not None:
        execution["lambda_time"] = args.lambda_time
    gap_limits = full_queue_gap_limits(execution, gain,
        max_units=args.gap_max_units, max_passages=args.gap_max_passages)
    declared_compilation = deepcopy(execution.get("cache_compilation") or {})
    completion_timing = (execution.get("plan_gain") or {}).get("completion_timing_asset")
    demand_policy = execution.get("prefix_retention") or {}
    execution.update(cache_start="cold", state_observation=None, plan_gain=None,
                     prefix_retention={"mode": "native"}, cache_compilation={"mode": "native"},
                     dialogue_mode="contextkv")
    read_outputs = args.read_outputs or (2048,)
    if needs_gain:
        from controller.action_proposals import finite_action_domain
        domain = gain.provenance["action_domain"]
        read_outputs = args.read_outputs or domain["output_budgets"]
        if domain != finite_action_domain(read_outputs, config["reading"]["output_tokens_per_fact"]):
            raise ValueError("read budgets differ from the fitted candidate domain")
    rolling_settings = dict(mode="observed_actions", reader_checkpoint=run.model.revision, reader_profile_id=profile,
        sampling_interval_seconds=args.sampling_interval, output_budgets=read_outputs,
        output_tokens_per_fact=config["reading"]["output_tokens_per_fact"], gap_limits=gap_limits, read_choices=2)
    if completion_timing is not None:
        rolling_settings["completion_timing_asset"] = str(Path(completion_timing).resolve())
    gain_settings = dict(rolling_settings, gain_asset=str(gain_path)) if needs_gain else None
    rule_settings = dict(rolling_settings, action_policy="rule")
    checkout = args.contextpilot_checkout or declared_compilation.get("checkout")
    needs_compilation = any(method in (*CONTEXT_METHODS[1:], RULE_CONTEXT_METHOD) for method in selected_methods)
    compilation = ({"mode": "contextpilot", "checkout": str(Path(checkout).resolve())}
                   if needs_compilation and checkout else None)
    if needs_compilation and compilation is None:
        raise ValueError("ContextPilot methods require --contextpilot-checkout or execution.cache_compilation.checkout")
    matrix["methods"] = full_queue_method_specs(selected_methods, gain_settings, compilation,
                                                rule_settings=rule_settings, demand_policy=demand_policy)
    pricing_asset = next((method["overrides"]["execution"]["prefix_retention"].get("pricing_asset")
        for method in matrix["methods"] if method["evaluation_factors"]["retention"] == "demand"), None)
    config_path = root/"frozen-study.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    for name in datasets:
        matrix["datasets"][name]["continue_conversation_failures"] = True
    matrix_path = root/"study-matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False))
    external_matrix_path = None
    external_retrievers = {}
    if set(CONVENTIONAL_METHODS).intersection(selected_methods):
        from bench.study import method_config
        external_matrix = deepcopy(matrix)
        external_method = next(item for item in matrix["methods"] if item["id"] in CONVENTIONAL_METHODS)
        for dataset in datasets:
            effective = method_config(config, external_method, matrix["datasets"][dataset], split)
            retriever = yaml.safe_load(Path(effective["retrieval"]["config_path"]).read_text())
            retriever.update(stage="hybrid_rerank", query_mode="current", selector=None,
                query_adapter_path=None, reranker_path=args.external_reranker,
                evidence_selection="ranked", query_fusion="single", rerank_query_mode="resolved", source_window=False)
            path = root/f"external-retriever-{dataset}.yaml"
            path.write_text(yaml.safe_dump(retriever, sort_keys=False))
            external_retrievers[dataset] = source_file(path)
            external_matrix["datasets"][dataset]["retriever_config"] = str(path.resolve())
            split_retrieval = external_matrix["datasets"][dataset].get("retrieval_by_split", {}).get(split, {})
            if "config_path" in split_retrieval:
                split_retrieval["config_path"] = str(path.resolve())
        external_matrix_path = root/"external-matrix.yaml"
        external_matrix_path.write_text(yaml.safe_dump(external_matrix, sort_keys=False))
    if fixed is not None:
        arrivals = {name: declare_fixed_arrivals(source_rows[name], fixed[name]["seconds"],
            name=fixed[name]["name"], concurrency=1) for name in datasets}
    jobs, workload_plans, workload_protocols = [], {}, {}
    for concurrency in args.concurrency:
        workload_paths = {}
        for dataset in datasets:
            rows = source_rows[dataset]
            workload_path = root/f"workload-{split}-c{concurrency}-{dataset}.json"
            source_plan = (replace(arrivals[dataset], concurrency=concurrency) if arrival_path is not None else
                freeze_workload(rows, name=f"{split}-{dataset}", concurrency=concurrency))
            if set(source_plan.dispatch_order) != {row["task_id"] for row in rows}:
                raise ValueError("arrival plan must cover the declared cohort exactly")
            source_plan.write(workload_path)
            workload_paths[dataset] = str(workload_path.resolve())
            workload_protocols[dataset] = dict(dispatch_semantics=source_plan.dispatch_semantics,
                arrival_source_basis=source_plan.arrival_source_basis,
                time_origin=source_plan.time_origin)
        workload_plans[str(concurrency)] = {name: source_file(path) for name, path in workload_paths.items()}
        for repetition in range(args.repetitions):
            order = selected_methods if repetition % 2 == 0 else tuple(reversed(selected_methods))
            for dataset in datasets:
                for method in order:
                    suffix = f"-r{repetition+1}" if args.repetitions > 1 else ""
                    name = f"{split}-{dataset}-c{concurrency}-{method}{suffix}"
                    jobs.append(dict(name=name, split=split, dataset=dataset, datasets=[dataset],
                        concurrency=concurrency, method=method, repetition=repetition,
                        requested_qa=len(cohorts[dataset]), requested_sessions=sessions[dataset],
                        protocol=matrix["datasets"][dataset].get("protocol", matrix.get("protocol", "P3")),
                        complete_slo_seconds=args.complete_slo_seconds,
                        matrix=str(external_matrix_path if method in CONVENTIONAL_METHODS else matrix_path),
                        workload=workload_paths[dataset], output=str(root/"jobs"/name/"generation")))
    acceptance = {**registered["acceptance"], "baseline": registered["baseline"],
        "resource_limits": {name: matrix["resources"][name] for name in (
            "gpu_capacity_tokens", "host_capacity_tokens", "max_online_tokens",
            "max_gpu_gib", "max_host_gib", "max_gpu_gib_seconds", "max_host_gib_seconds") if name in matrix["resources"]},
        "p95_slo_ms": matrix["resources"].get("p95_complete_ms"),
        "confidence": matrix["selection"].get("confidence", .95),
        "bootstrap_samples": matrix["selection"].get("bootstrap_samples", 5000), "seed": matrix["seed"],
        "p95_completion_rule": "all_declared_tasks_delivered",
        "cost_basis": "unique native receipts plus provider phase totals, and global online physical occupancy"}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.project_root,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.strip() or None
    manifest = dict(format="contextkv_dataset_queue_plan_v1", scope="per-dataset complete declared queue", split=split,
        execution_commit=revision, source_config=source_file(args.config), source_matrix=source_file(args.matrix),
        source_cohorts=source_file(args.cohorts) if args.cohorts is not None else None,
        reader_profile_id=profile, reader_run=source_file(run_path), resource_scope=scope,
        assets={"gain": source_file(gain_path) if needs_gain else None,
                "demand_pricing": source_file(pricing_asset) if pricing_asset is not None else None,
                "completion_timing": source_file(rolling_settings["completion_timing_asset"])
                if completion_timing is not None else None},
        cohort=cohorts, workload_plans=workload_plans, workload_protocols=workload_protocols,
        acceptance=acceptance, replicate_id=args.replicate_id,
        experiment_cells={method: method for method in selected_methods},
        experiment_design="ordinary retrieval QA versus ContextKV and LlamaIndex; optional mechanism controls",
        method_factors={method["id"]: method["evaluation_factors"] for method in matrix["methods"]},
        external_retrievers=external_retrievers, external_reranker=args.external_reranker,
        compilation_policy=compilation, source_arrival_plan=source_file(arrival_path) if arrival_path is not None else None,
        lambda_time=execution["lambda_time"], concurrency=args.concurrency, selected_methods=selected_methods,
        requested_qa=sum(job["requested_qa"] for job in jobs), jobs=jobs, config=str(config_path),
        cache_boundary="one cold boundary per method and dataset; never cold between turns",
        scoring="each dataset's declared metric and protocol, with the complete task denominator")
    write_json(root/"capacity-plan.json", manifest)
    return manifest


def delivery_summary(dataset_rows, failed, dispatch, *, complete_slo_seconds=None):
    """Request-level delivery status and observed arrival-to-completion window."""
    timings = [TurnTiming(row["conversation_id"], row["task_id"], "success",
        row["response"]["output_tokens"], row["request_window"][0],
        t_response_done=row["request_window"][1], t_commit=row["request_window"][1],
        truncated=row["response"]["truncated"]) for row in dataset_rows]
    timings += [TurnTiming(row["conversation_id"], row["task_id"], "failed", 0,
                row["request_window"][0]) for row in failed]
    windows = [row["request_window"] for row in (*dataset_rows, *failed)]
    if dispatch is not None:
        known = {timing.task_id for timing in timings}
        timings += [TurnTiming(value["conversation_id"], task_id, "blocked", 0,
            dispatch["origin"]+value["fixed_arrival_seconds"])
            for task_id, value in dispatch["cancelled"].items()
            if task_id not in known and value["fixed_arrival_seconds"] is not None]
        if dispatch.get("observation_window") is not None:
            window = dispatch["observation_window"]
            windows.append([window["start"], window["end"]])
    if windows:
        return summarize_timings(timings,
            window_start=min(window[0] for window in windows),
            window_end=math.nextafter(max(window[1] for window in windows), math.inf),
            ttft_slo=0, tpot_slo=0, response_mode="buffered",
            complete_slo=complete_slo_seconds)
    return None


def declared_execution_rows(job):
    """Combine saved delivery, scoring and dispatch evidence by task identity."""
    directory = Path(job["output"])/job["method"]
    rows, failure_ledgers, delivery = [], {}, {}
    for dataset in job["datasets"]:
        track = directory/dataset
        scored = track/"results.scored.jsonl"
        raw = saved_study_rows(track)
        results = {str(row["task_id"]): row for row in raw}
        if len(results) != len(raw):
            raise ValueError("duplicate saved executions in a declared track")
        if scored.exists():
            labelled = read_jsonl(scored)
            scores = {str(row["task_id"]): row for row in labelled}
            if len(scores) != len(labelled):
                raise ValueError("duplicate saved scores in a declared track")
            if not results:
                results = scores
            else:
                for task_id in results.keys() & scores.keys():
                    actual, score = results[task_id], scores[task_id]
                    if actual["response"]["attempt_id"] == score["response"]["attempt_id"]:
                        results[task_id] = {**actual, "metrics": score["metrics"], "scoring": score["scoring"]}
        failures = track/"failed_conversations.json"
        if failures.exists():
            failure_ledgers[dataset] = str(failures)
        failed = json.loads(failures.read_text()) if failures.exists() else []
        failed = [row for row in failed if str(row["task_id"]) not in results]
        dispatch_path = track/"workload_dispatch.json"
        dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else None
        delivery[dataset] = delivery_summary([row for row in results.values() if row["status"] == "completed"],
            failed, dispatch,
            complete_slo_seconds=job.get("complete_slo_seconds"))
        for row in failed:
            stage = (row.get("attempt") or {}).get("stage")
            status = "unknown" if stage == "COMMITTED" else "failed"
            results.setdefault(str(row["task_id"]), {**row, "status": status})
        if dispatch is not None:
            for task_id, row in dispatch["cancelled"].items():
                results.setdefault(str(task_id), {**row, "status": "blocked"})
        for failure in failed:
            for task_id in failure.get("blocked_task_ids", ()):
                results.setdefault(str(task_id), {"task_id": task_id,
                    "conversation_id": failure["conversation_id"], "status": "blocked"})
        rows.extend({**row, "dataset": dataset, "generation_dir": str(track)} for row in results.values())
    return rows, failure_ledgers, delivery


def summarize_full_queue(job, *, native_path=None):
    from controller.resident_models import resident_model_digest
    from profiling.study_cost_ledger import registration_key, registration_work

    directory = Path(job["output"])/job["method"]
    rows, failure_ledgers, delivery = declared_execution_rows(job)
    batch_tracks = [{"generation_dir": str(directory/name)} for name in job["datasets"]]
    windows = {}
    for name in job["datasets"]:
        windows[name] = []
        for path in sorted((directory/name/"measurements").glob("*/online_window.json")):
            manifest = json.loads((path.parent/"generation_manifest.json").read_text())
            windows[name].append({**json.loads(path.read_text()), "source": str(path),
                "resident_model_digest": resident_model_digest(manifest["retrieval"]),
                "measurement_kind": manifest["measurement_kind"], "engine_pool": {
                    key: manifest["allocator_initial"]["pool"][key] for key in ("gpu_capacity_tokens", "host_capacity_tokens")}})
    scoring = {}
    registration_events, event_sources = {}, []
    for name in job["datasets"]:
        events = directory/name/"state"/"events.jsonl"
        if events.exists():
            event_sources.append(str(events))
            with events.open() as stream:
                for line in stream:
                    if '"documents_registered"' in line and line.endswith("\n"):
                        event = json.loads(line)
                        if event["event"] == "documents_registered":
                            registration_events[registration_key(event)] = event
        path = directory/name/"scoring"/"manifest.json"
        if path.exists():
            scorer = json.loads(path.read_text())["identity"]
            for key in ("legacy_paid_scoring_path", "original_full_audit"):
                scorer.pop(key, None)
            if "judge" in scorer:
                scorer["judge"].pop("base_url", None)
            scoring[name] = scorer
    matrix = yaml.safe_load(Path(job["matrix"]).read_text())
    dataset, split = job["dataset"], job["split"]
    spec = _track_specs(matrix, split, datasets=[dataset])[dataset]
    track = declared_queue_summary(rows, spec, matrix["cohorts"][split][dataset])
    quality, normalization = track["queue_primary_score"], spec.get("normalization", 1.0)
    declared = {(dataset, str(task)) for task in matrix["cohorts"][split][dataset]}
    extras = sorted({(row["dataset"], str(row["task_id"])) for row in rows}-declared)
    return {"format": "contextkv_dataset_queue_result_v1", "job": job["name"],
        "method": job["method"], "concurrency": job["concurrency"],
        "dataset": dataset, "split": split, "protocol": job["protocol"],
        "requested_qa": job["requested_qa"],
        "completed_qa": track["completed"], "datasets": {dataset: track},
        "dataset_protocols": {dataset: evaluation_track_protocol(spec, split)},
        "queue_quality": quality/normalization if quality is not None else None,
        "queue_quality_bounds": [bound/normalization for bound in track["queue_score_bounds"]],
        "quality_scope": "single dataset, normalized original metric, complete declared denominator",
        "p95_complete_ms": _cost_summary([row for row in rows if (row["dataset"], str(row["task_id"])) in declared])[
            "p95_complete_ms"],
        "unexpected_execution_task_ids": extras, "online_windows": windows, "scorer_contracts": scoring,
        "registration_work": {**registration_work(registration_events.values()), "event_sources": event_sources},
        "latency_population": "delivered responses; completion counts use the entire declared queue",
        "failure_ledgers": failure_ledgers,
        "delivery_metrics": delivery,
        "cost_batches": batch_ledger(batch_tracks),
        "paid_native_usage": paid_execution_index(rows, [directory], native_path or Path(job["output"])/"native.json")}


def full_queue_occupancy(windows, physical, native):
    """Integrate unique global samples over the union of online windows."""
    from profiling.study_cost_ledger import merge_ranges

    merged = merge_ranges((row["start"], row["end"]) for rows in windows.values() for row in rows)
    physical_costs, native_costs, device_coverage = [], [], []
    for start, end in merged:
        physical_costs.append(integrate_physical(physical["samples"], start, end))
        pool, coverage = pool_series(native, start, end)
        native_costs.append({**occupancy_integral(pool), **coverage})
        inside = [row for row in physical["samples"] if start <= row["timestamp"] <= end]
        device_coverage.append(bool(inside) and all(
            {item["index"] for item in row["devices"]} == set(physical["gpu_devices"])
            and len({item["uuid"] for item in row["devices"]}) == len(physical["gpu_devices"]) for row in inside))
    def aggregate(costs, keys):
        return {key: (reducer(row[key] for row in costs) if costs
                      and all(row.get(key) is not None and row["complete_window_coverage"] for row in costs) else None)
                for key, reducer in keys.items()}
    result = aggregate(physical_costs, {"gpu_gib_seconds": sum, "host_gib_seconds": sum,
                                       "gpu_peak_gib": max, "host_peak_gib": max})
    result.update(aggregate(native_costs, {"gpu_slot_seconds": sum, "host_slot_seconds": sum,
                                          "gpu_peak_slots": max, "host_peak_slots": max}))
    return {**result, "clock": "client_monotonic", "windows": merged,
        "online_seconds": sum(end-start for start, end in merged),
        "hardware_only": bool(merged) and all(rows and all(row["measurement_kind"] == "hardware" for row in rows)
                                              for rows in windows.values()),
        "physical_coverage": bool(merged) and all(row["complete_window_coverage"] for row in physical_costs)
            and all(device_coverage),
        "native_coverage": bool(merged) and all(row["complete_window_coverage"] for row in native_costs)}


def refresh_saved_full_queue(job, original, cost, native_path):
    """Rebuild a scored cell from its saved records and original observation."""
    summary = summarize_full_queue(job, native_path=native_path)
    occupancy = full_queue_occupancy(summary["online_windows"],
        cost["physical_observation"], cost["native_observation"])
    finalized = (all(summary["online_windows"].values())
        and set(summary["scorer_contracts"]) == set(summary["datasets"])
        and summary["queue_quality"] is not None
        and not summary["unexpected_execution_task_ids"])
    return {**original, **summary, "online_occupancy": occupancy,
        "original_execution_status": original["execution_status"],
        "execution_status": "completed" if finalized else original["execution_status"]}


async def interrupt_job(process):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    return await process.wait()


async def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cpu_start = time.process_time()
    plan = prepare(args, root)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "qa": plan["requested_qa"], "plan": str(root/"capacity-plan.json")}))
        return plan
    scope = plan["resource_scope"]
    physical = StudyResourceSampler(tuple(scope["gpu_devices"]), tuple(scope["process_roots"]), interval=args.sampling_interval)
    native = NativePoolSampler(scope["native_endpoint"], gpu_capacity=scope["gpu_capacity_tokens"],
        host_enabled=scope["host_enabled"], model=scope["model"])
    await physical.start()
    await native.start()
    started = time.monotonic()
    deadline = datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    state = dict(format="contextkv_dataset_queue_workflow_v1", scope=plan["scope"], split=plan["split"],
        status="running", requested_qa=plan["requested_qa"], jobs=[], results=[])
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, task.cancel)
    try:
        for job in plan["jobs"]:
            if deadline is not None and time.time() >= deadline:
                state["jobs"].append({**job, "status": "not_started_deadline"})
                continue
            from profiling.runtime_manifest import collect_local_runtime, collect_server
            hardware = await asyncio.to_thread(collect_local_runtime, include_torch=False)
            native_runtime = await collect_server(scope["native_endpoint"], research=True)
            command = [args.python or sys.executable, "-m", "bench.study", "evaluate", "--config", plan["config"],
                "--matrix", job["matrix"], "--split", job["split"], "--methods", job["method"],
                "--concurrency", str(job["concurrency"]), "--output", job["output"],
                "--datasets", job["dataset"], "--workload-plan", job["workload"]]
            record = {**job, "command": command, "status": "running", "started_s": time.monotonic()}
            with TemporaryFile() as errors:
                process = await asyncio.create_subprocess_exec(*command, cwd=Path(args.project_root).resolve(),
                    stdout=asyncio.subprocess.DEVNULL, stderr=errors, start_new_session=True)
                record["pid"] = process.pid
                state["jobs"].append(record)
                write_json(root/"workflow.json", state)
                stopped = None
                try:
                    code = await asyncio.wait_for(process.wait(), max(.1, deadline-time.time())) if deadline else await process.wait()
                except TimeoutError:
                    stopped = "stopped_at_deadline"
                    code = await interrupt_job(process)
                except asyncio.CancelledError:
                    stopped = state["status"] = "interrupted"
                    code = await interrupt_job(process)
                if code and stopped is None:
                    errors.seek(0, os.SEEK_END)
                    errors.seek(max(0, errors.tell()-1024))
                    lines = errors.read().decode(errors="replace").strip().splitlines()
                    if lines:
                        record["error"] = lines[-1]
            record.update(returncode=code, ended_s=time.monotonic(),
                          status=stopped or ("completed" if code == 0 else "failed_with_existing_raw_preserved"))
            summary = summarize_full_queue(job)
            summary["runtime_hardware"] = hardware
            summary["native_cache_config"] = native_runtime["cache_config"]
            summary["execution_status"] = record["status"]
            summary["reader_runtime"] = {name: native_runtime["engine"].get(name)
                for name in ("fingerprint", "model_scope", "protocol_version", "sglang_commit")}
            state["results"].append(summary)
            write_json(root/"workflow.json", state)
            print(json.dumps({"job": job["name"], "status": record["status"],
                "completed": summary["completed_qa"], "requested": job["requested_qa"],
                "p95_ms": summary["p95_complete_ms"]}), flush=True)
            if state["status"] == "interrupted":
                break
    except asyncio.CancelledError:
        state["status"] = "interrupted"
    except Exception:
        state["status"] = "failed"
        raise
    else:
        if state["status"] == "running":
            state["status"] = "finished_with_actual_coverage"
    finally:
        end = time.monotonic()
        await native.stop()
        await physical.stop()
        physical_observation, native_observation = physical.observation(), native.observation()
        for summary in state["results"]:
            summary["online_occupancy"] = full_queue_occupancy(
                summary["online_windows"], physical_observation, native_observation)
        child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        series, coverage = pool_series(native.observation(), started, end)
        write_json(root/"capacity-cost.json", dict(format="contextkv_dataset_queue_cost_v1",
            source="hardware", resource_window=dict(start=started, end=end),
            resource_scope=scope, physical_observation=physical_observation,
            physical_cost=integrate_physical(physical.samples, started, end),
            native_observation=native_observation, native_cost={**occupancy_integral(series), **coverage},
            controller_cpu_seconds=time.process_time()-cpu_start,
            child_cpu_seconds=child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime,
            cost_scope="full sequential capacity jobs including child provider initialization, pauses, failed requests and idle gaps; nested per-job/response shares are not added"))
        counts = Counter()
        for row in state["results"]:
            counts[row["method"]] += row["completed_qa"]
        state.update(completed_by_method=dict(counts), performance_scope=plan["scope"],
            full_cost=str(root/"capacity-cost.json"))
        write_json(root/"workflow.json", state)
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Study configuration with frozen method assets")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--datasets", nargs="+", help="Select datasets available in this split")
    parser.add_argument("--arrival-plan", help="Recorded fixed arrivals or workload manifest; default follows conversation dependencies")
    parser.add_argument("--complete-slo-seconds", type=float, help="Arrival-to-completion SLO for request goodput")
    parser.add_argument("--repetitions", type=int, default=1, help="Independent queue repetitions per method")
    parser.add_argument("--replicate-id", help="Shared identifier when splitting one repeat across explicit workers")
    parser.add_argument("--methods", nargs="+", choices=(*FULL_QUEUE_METHODS, *OPTIONAL_FULL_QUEUE_METHODS))
    parser.add_argument("--cohorts", help="Frozen cohort manifest containing the cohorts mapping")
    parser.add_argument("--gain-asset")
    parser.add_argument("--external-reranker", default="models/bge-reranker-v2-m3")
    parser.add_argument("--contextpilot-checkout", help="Pinned author source checkout for ContextPilot compilation")
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 4, 8), default=[1, 4])
    parser.add_argument("--reader-profile-id")
    parser.add_argument("--online-gpu-devices", required=True, type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--online-process-roots", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--reader-base-url", required=True)
    parser.add_argument("--gap-max-units", type=int)
    parser.add_argument("--gap-max-passages", type=int)
    parser.add_argument("--lambda-time", type=float)
    parser.add_argument("--read-outputs", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--sampling-interval", type=float, default=.5)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python")
    parser.add_argument("--deadline")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    methods = args.methods or FULL_QUEUE_METHODS
    if any(method in CONTEXT_METHODS for method in methods):
        if not args.gain_asset:
            parser.error("learned ContextKV methods require --gain-asset")
    elif not args.reader_profile_id:
        parser.error("methods without a gain asset require --reader-profile-id")
    return asyncio.run(run(args))


if __name__ == "__main__":
    if main().get("status") == "interrupted":
        raise SystemExit(130)
