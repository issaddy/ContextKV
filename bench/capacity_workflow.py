"""Run current methods on declared dev/test datasets with complete response costs."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from tempfile import TemporaryFile
import time
from urllib.parse import urlparse

import yaml

from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.study import declare_cohorts, dataset_rows, saved_study_rows, write_json
from bench.study_metrics import _track_specs, declared_queue_summary
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
    config.setdefault("scoring", {})["gpu_devices"] = list(args.online_gpu_devices)
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
                        matrix=str(external_matrix_path if method in CONVENTIONAL_METHODS else matrix_path),
                        workload=workload_paths[dataset], output=str(root/"jobs"/name/"generation")))
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
        acceptance=registered["acceptance"],
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


def delivery_summary(dataset_rows, failed, dispatch):
    return {"completed": len(dataset_rows), "failed": len(failed),
            "blocked": len(dispatch["cancelled"]) if dispatch is not None else 0}


def declared_execution_rows(job):
    rows, delivery = [], {}
    for dataset in job["datasets"]:
        track = Path(job["output"])/job["method"]/dataset
        results = {str(row["task_id"]): row for row in saved_study_rows(track)}
        path = track/"failed_conversations.json"
        failed = json.loads(path.read_text()) if path.exists() else []
        failed = [row for row in failed if str(row["task_id"]) not in results]
        for row in failed:
            results[str(row["task_id"])] = {**row, "status": "failed"}
            for identity in row.get("blocked_task_ids", ()):
                results.setdefault(str(identity), {"task_id": identity,
                    "conversation_id": row["conversation_id"], "status": "blocked"})
        dispatch_path = track/"workload_dispatch.json"
        dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else None
        if dispatch is not None:
            for identity, row in dispatch["cancelled"].items():
                results.setdefault(str(identity), {**row, "status": "blocked"})
        delivery[dataset] = delivery_summary(
            [row for row in results.values() if row["status"] == "completed"], failed, dispatch)
        rows.extend({**row, "dataset": dataset} for row in results.values())
    return rows, delivery


def summarize_full_queue(job):
    rows, delivery = declared_execution_rows(job)
    dataset, split = job["dataset"], job["split"]
    matrix = yaml.safe_load(Path(job["matrix"]).read_text())
    spec = _track_specs(matrix, split, datasets=[dataset])[dataset]
    track = declared_queue_summary(rows, spec, matrix["cohorts"][split][dataset])
    windows = {dataset: []}
    directory = Path(job["output"])/job["method"]/dataset
    for path in sorted((directory/"measurements").glob("*/online_window.json")):
        windows[dataset].append(json.loads(path.read_text()))
    quality = track["queue_primary_score"]
    scoring_path = directory/"scoring"/"manifest.json"
    scoring = json.loads(scoring_path.read_text()) if scoring_path.exists() else {}
    return {"format": "contextkv_dataset_queue_result_v1", "job": job["name"],
        "method": job["method"], "concurrency": job["concurrency"], "repetition": job["repetition"],
        "dataset": dataset, "split": split, "protocol": job["protocol"],
        "primary_metric": spec["primary_metric"], "normalization": spec.get("normalization", 1),
        "requested_qa": track["declared"], "completed_qa": track["completed"], "scored_qa": track["scored"],
        "completion_rate": track["completion_rate"], "quality": quality,
        "benchmark_metrics": track["benchmark_metrics"],
        "groups": scoring.get("groups", {}), "scoring": {name: scoring[name] for name in (
            "identity", "judge_cost") if name in scoring},
        "queue_quality": quality/spec.get("normalization", 1) if quality is not None else None,
        "queue_quality_bounds": track["queue_score_bounds"],
        "cost": track["cost"], "p95_complete_ms": track["cost"]["p95_complete_ms"],
        "online_windows": windows, "delivery_metrics": delivery,
        "failures": {row["task_id"]: row["error"] for row in rows if row["status"] == "failed"}}


def full_queue_occupancy(windows, physical, native):
    """Integrate unique global samples over the union of online windows."""
    from bench.metrics import merge_ranges

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



async def interrupt_job(process):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    return await process.wait()


async def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
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
                          status=stopped or ("completed" if code == 0 else "failed"))
            summary = summarize_full_queue(job)
            summary["execution_status"] = record["status"]
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
        series, coverage = pool_series(native.observation(), started, end)
        write_json(root/"capacity-cost.json", dict(format="contextkv_dataset_queue_cost_v1",
            source="hardware", resource_window=dict(start=started, end=end),
            resource_scope=scope,
            physical_cost=integrate_physical(physical.samples, started, end),
            native_cost={**occupancy_integral(series), **coverage},
            cost_scope="full sequential capacity jobs including child provider initialization, pauses, failed requests and idle gaps; nested per-job/response shares are not added"))
        counts = Counter()
        for row in state["results"]:
            counts[row["method"]] += row["completed_qa"]
        state.update(completed_by_method=dict(counts), performance_scope=plan["scope"],
            full_cost=str(root/"capacity-cost.json"))
        for summary in state["results"]:
            summary.pop("online_windows", None)
        write_json(root/"workflow.json", state)
        if (root/"jobs").exists():
            shutil.rmtree(root/"jobs")
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
    parser.add_argument("--repetitions", type=int, default=1, help="Independent queue repetitions per method")
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
