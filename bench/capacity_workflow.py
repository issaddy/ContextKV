"""Own-history P3 capacity diagnostics and full-queue joint B/C evaluation."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import yaml

from bench.metrics import TurnTiming, occupancy_integral, summarize_timings
from bench.native_metrics import NativePoolSampler, pool_series
from bench.plan_result_summary import batch_ledger
from bench.reassess_saved import paid_execution_index
from bench.rolling_study_workflow import configure
from bench.study import dataset_rows, merge_config, read_jsonl, write_json, write_jsonl
from bench.study_metrics import (_cost_summary, _quantile, _track_specs, declared_queue_summary,
                                 evaluation_track_protocol)
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from bench.study_workloads import freeze_workload, load_goal_policy, read_workload_manifest


def snapshot(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare(args, root):
    if args.full_queue:
        return prepare_full_queue(args, root)
    source_path = Path(args.source_manifest).resolve()
    source = json.loads(source_path.read_text())
    source_matrix = yaml.safe_load(Path(source["diagnostic_matrix_path"]).read_text())
    settings = yaml.safe_load(Path(args.config).read_text())
    settings = merge_config(settings, {
        "support_model": {"mode": "trained", "asset": args.parent_policy_asset,
                          "preference_asset": args.parent_policy_asset},
        "selection": {"utility": "learned"},
        "reading": {"layout": "source_first", "packing": "source_windows"},
        "execution": {"prefix_retention": {"mode": "native"}}})
    parent = root/"parent-P0-study.yaml"
    parent.write_text(yaml.safe_dump(settings, sort_keys=False))
    scope = SimpleNamespace(config=str(parent), matrix=args.matrix or source["diagnostic_matrix_path"],
        execution_profile=args.execution_profile, native_assets=args.native_assets, gain_asset=args.gain_asset,
        reader_profile_id=args.reader_profile_id, sampling_interval=args.sampling_interval,
        read_outputs=args.read_outputs, baselines=["B6"], rolling_id="S3",
        lambda_time=(args.lambda_time if args.lambda_time is not None else settings["execution"]["lambda_time"],),
        failure_unit="client", dev_conversations_per_track=None, test_conversations_per_track=None)
    config, matrix, config_path = configure(scope, root)
    config["execution"].update(cache_start="cold", prefix_retention={"mode": "native"})
    config["reading"].update(layout="source_first", packing="source_windows")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    methods = []
    for method in matrix["methods"]:
        value = deepcopy(method)
        value["original_method_id"] = value["id"]
        value["id"] = "B6_native" if value["strategy"] == "B6" else "S3_native"
        value["overrides"] = merge_config(value.get("overrides", {}), {
            "reading": {"layout": "source_first", "packing": "source_windows"},
            "execution": {"cache_start": "cold", "prefix_retention": {"mode": "native"}}})
        if value["strategy"] == "B7":
            gain = value["overrides"]["execution"]["plan_gain"]
            gain["mode"] = args.gain_mode
            if args.value_model is not None:
                gain["prefix_directory"] = True
        methods.append(value)
    value_policy = None
    if args.value_model is not None:
        value_policy = dict(mode="value", model_path=str(Path(args.value_model).resolve()),
            horizon_seconds_choices=[15, 60, 180],
            protection_budget_tokens=8192, lambda_gpu=0.0, lambda_host=0.0)
        for method in tuple(methods):
            value = deepcopy(method)
            value["id"] = method["id"].removesuffix("_native")+"_value"
            value["overrides"]["execution"]["prefix_retention"] = deepcopy(value_policy)
            methods.append(value)
    matrix["methods"] = methods
    matrix["resources"].update(gpu_capacity_tokens=32768, host_capacity_tokens=61056)
    policy_path = Path(source["goal_policy_path"]).resolve()
    policy = load_goal_policy(policy_path, expected_sha256=source["goal_policy_sha256"])
    originals = read_jsonl(source["tasks_path"])
    grouped = {}
    for row in originals:
        if row["turn"] in (1, 2):
            grouped.setdefault(row["conversation_id"], {})[row["turn"]] = row
    clients = sorted(grouped)
    plans = []
    for concurrency in args.concurrency:
        rows = []
        for index in range(concurrency):
            original_id = clients[index % len(clients)]
            original = grouped[original_id]
            client = f"capacity-c{concurrency}-client{index:02d}"
            if original[1]["goal_graph_id"] not in policy.graphs:
                raise ValueError("actual source goal policy is missing the frozen client graph")
            for turn in (1, 2):
                rows.append({**deepcopy(original[turn]), "conversation_id": client,
                    "task_id": client+":"+str(turn),
                    "capacity_replica_of": original[turn]["task_id"],
                    "capacity_scope": "first two source turns only; original graph/source/family/version unchanged"})
        directory = root/"inputs"/f"c{concurrency}"
        task_path, workload_path = directory/"tasks.jsonl", directory/"workload.json"
        write_jsonl(task_path, rows)
        frozen = freeze_workload(rows, name=f"capacity-c{concurrency}-first2-pause30",
            concurrency=concurrency, think_time_seconds=30,
            goal_policy_sha256=source["goal_policy_sha256"])
        frozen = replace(frozen, dispatch_semantics="independent_clients_after_own_prior_commit")
        frozen.write(workload_path)
        job_matrix = deepcopy(matrix)
        source_spec = source_matrix["datasets"][source["dataset"]]
        source_spec = {name: value for name, value in source_spec.items() if name not in {
            "input", "inputs", "split_manifest", "dev_conversations", "dev_seed", "reuse_provider"}}
        source_spec.update(inputs={"dev": str(task_path.resolve())}, protocol="P3",
            supplied_documents=True, goal_policy=str(policy_path), continue_conversation_failures=True,
            diagnostic_only=True, include_in_public_quality_aggregate=False)
        job_matrix["datasets"] = {"dynamic_goal_dev": source_spec}
        job_matrix["cohorts"] = {"dev": {"dynamic_goal_dev": [row["task_id"] for row in rows]}}
        job_matrix["selection"].update(include_ablations=False, development_weights={"dynamic_goal_dev": 1.0})
        matrix_path = directory/"matrix.yaml"
        matrix_path.write_text(yaml.safe_dump(job_matrix, sort_keys=False))
        order = ("S3_native", "B6_native") if concurrency == 4 else ("B6_native", "S3_native")
        if args.value_model is not None:
            order += ("B6_value", "S3_value")
        for method in order:
            plans.append(dict(name=f"c{concurrency}-"+method, concurrency=concurrency,
                clients=concurrency, requested_qa=2*concurrency, method=method,
                tasks=str(task_path), matrix=str(matrix_path), workload=str(workload_path),
                goal_policy=str(policy_path), output=str(root/"jobs"/(f"c{concurrency}-"+method)/"generation")))
    per_method = Counter()
    for job in plans:
        per_method[job["method"]] += job["requested_qa"]
    manifest = dict(format="contextkv_two_turn_capacity_plan_v1", scope="two-turn P3 capacity only; not the full seven-turn/five-scenario experiment",
        source_manifest=snapshot(source_path), source_tasks=snapshot(source["tasks_path"]),
        source_goal_policy=snapshot(policy_path), source_config=snapshot(args.config),
        parent_P0_asset=str(Path(args.parent_policy_asset).resolve()),
        parent_P0_heads=snapshot(Path(args.parent_policy_asset)/"heads.pt"),
        reader_profile_id=args.reader_profile_id, assets={name: snapshot(path) for name, path in (
            ("gain", args.gain_asset), ("execution", args.execution_profile), ("native", args.native_assets))},
        first_arrivals="all clients planned at dispatcher time zero; actual scheduler drift recorded",
        second_arrivals="own first QA commit + 30 seconds; own actual answer selects graph branch",
        client_replication="sorted frozen source conversations cycled to each declared client count; original source metadata retained",
        methods=[method["id"] for method in methods], requested_qa=sum(job["requested_qa"] for job in plans),
        per_method_requested_qa=dict(per_method),
        value_asset=snapshot(args.value_model) if args.value_model is not None else None,
        value_policy=value_policy,
        gain_mode=args.gain_mode, concurrency=args.concurrency,
        joint_scope=("four cells per declared concurrency, each executes own P3 and may choose different DAGs"
                     if args.value_model is not None else None),
        cache_boundary="one cold boundary per method/concurrency job; never cold between clients/turns",
        question_protocol="P3 own committed generated QA; no authored/other-method answer imported",
        scoring="capacity/completion/Need-source coverage only; no public quality aggregate", jobs=plans,
        config=str(config_path),
        resource_scope={**json.loads(Path(args.execution_profile).read_text())["resource_scope"],
            "gpu_capacity_tokens": 32768, "host_enabled": True, "model": "Qwen3-32B"})
    write_json(root/"capacity-plan.json", manifest)
    return manifest


MAIN_COMPILATION_CELLS = {
    "E1": "B6_native", "E2": "S3_native",
    "E3": "B6_contextpilot", "E4": "S3_contextpilot",
}
FULL_QUEUE_METHODS = (*MAIN_COMPILATION_CELLS.values(),
    "LlamaIndex_native", "S3_contextpilot_no_fact_reuse")
OPTIONAL_FULL_QUEUE_METHODS = ("E0_native", "B6_value", "S3_value")


def full_queue_methods(concurrency, methods=None):
    # Both c1 and c4 answer the same factorial question and use the same queue.
    return tuple(methods) if methods is not None else FULL_QUEUE_METHODS


def full_queue_method_specs(selected, gain_settings, compilation, value_policy=None):
    """Keep compilation, B decisions and optional retention as separate factors."""
    specs = []
    for method in selected:
        gain = method.startswith("S3_")
        external = method == "LlamaIndex_native"
        contextpilot = "_contextpilot" in method
        overrides = {"execution": {
            "dialogue_mode": "llamaindex" if external else "contextkv",
            "plan_gain": deepcopy(gain_settings) if gain else None,
            "cache_compilation": deepcopy(compilation) if contextpilot else {"mode": "native"},
            "prefix_retention": deepcopy(value_policy) if method.endswith("_value") else {"mode": "native"}},
            "reading": {"fact_reuse": method != "S3_contextpilot_no_fact_reuse"}}
        if method == "E0_native":
            overrides["reading"].update(layout="task_first", packing="mixed", numeric_need_isolation=True)
        specs.append(dict(id=method, strategy="B7" if gain else "B6", overrides=overrides,
            evaluation_factors=dict(B="gain" if gain else "fixed", compilation="contextpilot" if contextpilot else "native",
                retention="value" if method.endswith("_value") else "native",
                fact_reuse=overrides["reading"]["fact_reuse"], dialogue="llamaindex" if external else "contextkv",
                role=("external_reference" if external else "fact_reuse_control" if method.endswith("_no_fact_reuse")
                      else "historical_reference" if method == "E0_native" else "optional_retention" if method.endswith("_value")
                      else "main_factorial"))))
    return specs


def full_queue_gap_limits(execution, gain=None, *, max_units=None, max_passages=None):
    """Bind gain actions to their measured limits; explicit CLI values take precedence."""
    inherited = (gain.provenance["continuation_gap_limits"] if gain is not None else
        {"max_units": 96, "max_passages": 10,
         **((execution.get("state_observation") or {}).get("gap_limits") or {})})
    return {"max_units": inherited["max_units"] if max_units is None else max_units,
            "max_passages": inherited["max_passages"] if max_passages is None else max_passages}


def compilation_source_identity(checkout):
    """Record the verified author tree and actual Python sources used by its index."""
    import subprocess
    from adapters.contextpilot import _verified_checkout

    root = _verified_checkout(str(checkout))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "contextpilot"], cwd=root,
                             check=True, capture_output=True).stdout.decode().split("\0")
    sources = [{"path": name, "sha256": hashlib.sha256((root/name).read_bytes()).hexdigest()}
               for name in sorted(tracked) if name.endswith(".py")]
    if not any(item["path"] == "contextpilot/server/live_index.py" for item in sources):
        raise ValueError("verified ContextPilot checkout is missing its author index source")
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return dict(checkout=str(root), author_commit=head, source_manifest=sources, sha256=digest,
                digest_scope="manifest of actual tracked contextpilot Python source content hashes")


def prepare_full_queue(args, root):
    from controller.config import load_config
    from training.train_plan_gain import PlanGainModel

    config = yaml.safe_load(Path(args.config).read_text())
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    run_path = Path(config["generation"]["run_config"]).resolve()
    run_data = yaml.safe_load(run_path.read_text())
    run_data["engine"]["base_url"] = args.reader_base_url
    run_path = root/"reader-run.yaml"
    run_path.write_text(yaml.safe_dump(run_data, sort_keys=False))
    run = load_config(run_path)
    config["generation"]["run_config"] = str(run_path)
    selected_methods = tuple(dict.fromkeys(method for concurrency in args.concurrency
        for method in full_queue_methods(concurrency, args.methods)))
    needs_gain = any(method.startswith("S3_") for method in selected_methods)
    needs_value = any(method.endswith("_value") for method in selected_methods)
    gain_path = Path(args.gain_asset).resolve() if needs_gain else None
    profile = json.loads(gain_path.read_text())["reader_profile_id"] if needs_gain else args.reader_profile_id
    gain = (PlanGainModel.load(gain_path, reader_checkpoint=run.model.revision, reader_profile_id=profile)
        if needs_gain else None)
    roots = args.online_process_roots
    if not roots:
        endpoints = [run.engine.base_url]
        if config["generation"].get("teacher_url"):
            endpoints.append(config["generation"]["teacher_url"])
        roots = sorted({os.getpid(), *listening_processes([urlparse(url).port for url in endpoints])})
    scope = dict(gpu_devices=args.online_gpu_devices, process_roots=roots,
        native_endpoint=run.engine.base_url, gpu_capacity_tokens=run.engine.gpu_capacity_tokens,
        host_capacity_tokens=run.engine.host_capacity_tokens,
        host_enabled=run.engine.host_capacity_tokens > 0, model="Qwen3-32B")
    matrix["resources"].update(gpu_devices=args.online_gpu_devices, process_ids=roots)
    execution = config["execution"]
    if args.lambda_time is not None:
        execution["lambda_time"] = args.lambda_time
    gap_limits = full_queue_gap_limits(execution, gain,
        max_units=args.gap_max_units, max_passages=args.gap_max_passages)
    declared_compilation = deepcopy(execution.get("cache_compilation") or {})
    execution.update(cache_start="cold", state_observation=None, plan_gain=None,
                     prefix_retention={"mode": "native"}, cache_compilation={"mode": "native"},
                     dialogue_mode="contextkv")
    gain_settings = dict(mode="observed_actions", gain_asset=str(gain_path),
        reader_checkpoint=gain.reader_checkpoint, reader_profile_id=gain.reader_profile_id,
        gpu_devices=args.online_gpu_devices, process_ids=roots,
        sampling_interval_seconds=args.sampling_interval, output_budgets=args.read_outputs,
        output_tokens_per_fact=config["reading"]["output_tokens_per_fact"],
        prefix_directory=True, gap_limits=gap_limits) if needs_gain else None
    value_policy = dict(mode="value", model_path=str(Path(args.value_model).resolve()),
        horizon_seconds_choices=[15, 60, 180], protection_budget_tokens=8192,
        lambda_gpu=0.0, lambda_host=0.0) if needs_value else None
    checkout = getattr(args, "contextpilot_checkout", None) or declared_compilation.get("checkout")
    needs_compilation = any("_contextpilot" in method for method in selected_methods)
    compilation = ({"mode": "contextpilot", "checkout": str(Path(checkout).resolve())}
                   if needs_compilation and checkout else None)
    if needs_compilation and compilation is None:
        raise ValueError("ContextPilot methods require --contextpilot-checkout or execution.cache_compilation.checkout")
    matrix["methods"] = full_queue_method_specs(selected_methods, gain_settings, compilation, value_policy)
    reference = "B6_native" if "B6_native" in selected_methods else matrix["methods"][0]["id"]
    matrix["selection"].update(include_ablations=False, baseline=reference,
                               test_comparators=[reference])
    config_path = root/"frozen-A-study.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    cohorts = matrix["cohorts"]["dev"]
    datasets = tuple(cohorts)
    requested = sum(len(task_ids) for task_ids in cohorts.values())
    for name in datasets:
        matrix["datasets"][name].update(protocol="P3", continue_conversation_failures=True)
    matrix_path = root/"joint-matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False))
    external_matrix_path = None
    external_retrievers = {}
    if "LlamaIndex_native" in selected_methods:
        from bench.study import method_config
        external_matrix = deepcopy(matrix)
        external_method = next(item for item in matrix["methods"] if item["id"] == "LlamaIndex_native")
        for dataset in datasets:
            effective = method_config(config, external_method, matrix["datasets"][dataset], "dev")
            retriever = yaml.safe_load(Path(effective["retrieval"]["config_path"]).read_text())
            retriever.update(stage="hybrid_rerank", query_mode="current", selector=None,
                query_adapter_path=None, reranker_path=args.external_reranker,
                evidence_selection="ranked", query_fusion="single", rerank_query_mode="resolved",
                source_window=False)
            path = root/f"external-retriever-{dataset}.yaml"
            path.write_text(yaml.safe_dump(retriever, sort_keys=False))
            external_retrievers[dataset] = snapshot(path)
            external_matrix["datasets"][dataset]["retriever_config"] = str(path.resolve())
            split_retrieval = external_matrix["datasets"][dataset].get("retrieval_by_split", {}).get("dev", {})
            if "config_path" in split_retrieval:
                split_retrieval["config_path"] = str(path.resolve())
        external_matrix_path = root/"external-matrix.yaml"
        external_matrix_path.write_text(yaml.safe_dump(external_matrix, sort_keys=False))
    jobs = []
    workload_plans = {}
    arrivals = read_workload_manifest(args.arrival_plan)
    for concurrency in args.concurrency:
        workload_paths = {}
        for dataset in datasets:
            rows = dataset_rows(matrix["datasets"][dataset], "dev", task_ids=cohorts[dataset])
            workload_path = root/f"workload-c{concurrency}-{dataset}.json"
            source_plan = arrivals[dataset]
            if source_plan.dispatch_semantics != "fixed_arrivals":
                raise ValueError("full queue requires fixed absolute relative arrivals")
            if set(source_plan.dispatch_order) != {row["task_id"] for row in rows}:
                raise ValueError("arrival plan must cover the declared cohort exactly")
            replace(source_plan, concurrency=concurrency).write(workload_path)
            workload_paths[dataset] = str(workload_path.resolve())
        workload_index = root/f"workload-c{concurrency}.json"
        write_json(workload_index, dict(format="contextkv_dataset_workload_index_v1", datasets=workload_paths))
        workload_plans[str(concurrency)] = {name: snapshot(path) for name, path in workload_paths.items()}
        for method in full_queue_methods(concurrency, args.methods):
            name = f"c{concurrency}-{method}"
            jobs.append(dict(name=name, concurrency=concurrency, method=method, datasets=datasets,
                requested_qa=requested, complete_slo_seconds=args.complete_slo_seconds,
                matrix=str(external_matrix_path if method == "LlamaIndex_native" else matrix_path),
                workload=str(workload_index.resolve()),
                output=str(root/"jobs"/name/"generation")))
    manifest = dict(format="contextkv_full_queue_joint_plan_v1", scope="full declared P3 development queue",
        source_config=snapshot(args.config), source_matrix=snapshot(args.matrix),
        reader_profile_id=profile, reader_run=snapshot(run_path), resource_scope=scope,
        assets={"gain": snapshot(gain_path) if needs_gain else None,
                "value": snapshot(args.value_model) if needs_value else None},
        cohort=cohorts, workload_plans=workload_plans,
        experiment_cells=MAIN_COMPILATION_CELLS,
        experiment_design="fixed/gain B x Native/ContextPilot compilation; retention Native; both c1 and c4",
        method_factors={method["id"]: method["evaluation_factors"] for method in matrix["methods"]},
        external_retrievers=external_retrievers, external_reranker=args.external_reranker,
        compilation_policy=compilation,
        compilation_asset=(compilation_source_identity(compilation["checkout"])
                           if compilation is not None else None),
        E0_protocol="P0 support/query assets with task_first, source_rows_pr3, numeric_need_isolation and Native; same public correctness code and arrival schedule",
        source_arrival_plan=snapshot(args.arrival_plan),
        arrival_protocol="fixed relative arrival timestamps shared across methods; own-history and semaphore waits included in complete response",
        value_policy=value_policy, gain_mode="observed_actions",
        lambda_time=execution["lambda_time"], concurrency=args.concurrency,
        selected_methods=selected_methods, requested_qa=requested*len(jobs), jobs=jobs, config=str(config_path),
        cache_boundary="one cold boundary per method and dataset; never cold between turns",
        scoring="declared development metrics and weights with the complete task denominator")
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


def summarize_full_queue(job):
    directory = Path(job["output"])/job["method"]
    rows, batch_tracks, failure_ledgers, delivery = [], [], {}, {}
    for dataset in job["datasets"]:
        track = directory/dataset
        batch_tracks.append({"generation_dir": str(track)})
        scored, unscored = track/"results.scored.jsonl", track/"results.unscored.jsonl"
        results = scored if scored.exists() else unscored
        if results.exists():
            rows.extend({**row, "generation_dir": str(track)} for row in read_jsonl(results))
        failures = track/"failed_conversations.json"
        if failures.exists():
            failure_ledgers[dataset] = str(failures)
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        failed = json.loads(failures.read_text()) if failures.exists() else []
        dispatch_path = track/"workload_dispatch.json"
        dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else None
        delivery[dataset] = delivery_summary(dataset_rows, failed, dispatch,
            complete_slo_seconds=job.get("complete_slo_seconds"))
    matrix = yaml.safe_load(Path(job["matrix"]).read_text())
    specs = _track_specs(matrix, "dev")
    tracks = {name: declared_queue_summary([row for row in rows if row["dataset"] == name],
        spec, matrix["cohorts"]["dev"][name]) for name, spec in specs.items()}
    quality = {name: track["queue_primary_score"] for name, track in tracks.items()}
    return {"job": job["name"], "method": job["method"], "concurrency": job["concurrency"],
        "requested_qa": job["requested_qa"],
        "completed_qa": sum(track["completed"] for track in tracks.values()),
        "datasets": tracks,
        "dataset_protocols": {name: evaluation_track_protocol(spec, "dev") for name, spec in specs.items()},
        "track_weights": {name: spec["weight"] for name, spec in specs.items()},
        "queue_quality": sum(spec["weight"]*quality[name]/spec.get("normalization", 1.0)
            for name, spec in specs.items()) if all(value is not None for value in quality.values()) else None,
        "queue_quality_bounds": [sum(spec["weight"]*tracks[name]["queue_score_bounds"][bound]
            /spec.get("normalization", 1.0) for name, spec in specs.items()) for bound in (0, 1)],
        "p95_complete_ms": _cost_summary(rows)["p95_complete_ms"],
        "latency_population": "delivered responses; completion counts use the entire declared queue",
        "failure_ledgers": failure_ledgers,
        "delivery_metrics": delivery,
        "cost_batches": batch_ledger(batch_tracks),
        "paid_native_usage": paid_execution_index(rows, [directory], Path(job["output"])/"native.json")}


def summarize_job(job):
    directory = Path(job["output"])/job["method"]/"dynamic_goal_dev"
    records = [json.loads(path.read_text()) for path in sorted((directory/"records").glob("*.json"))]
    failures_path = directory/"failed_conversations.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    dispatch_path = directory/"workload_dispatch.json"
    dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else None
    arrivals = ([dispatch["origin"]+row["actual_arrival_seconds"] for row in dispatch["actual_arrivals"].values()]
                if dispatch is not None else [])
    ends = [row["request_window"][1] for row in records]+[row["request_window"][1] for row in failures]
    elapsed = max(ends)-min(arrivals) if arrivals and ends else None
    latencies = [row["cost"]["complete_ms"] for row in records]
    traces, batch_costs = [], []
    for path in sorted((directory/"measurements").glob("*/batch_cost.json")):
        cost = json.loads(path.read_text())
        native = json.loads((path.parent/"native_resources.json").read_text())
        series, coverage = pool_series(native, cost["start"], cost["end"])
        batch_costs.append(dict(path=str(path), **cost,
            native_resource_window={**occupancy_integral(series), **coverage}))
    needs, supported, units, covered, open_needs = 0, 0, 0, 0, 0
    for row in records:
        outcome = row["reading_outcome"]
        if outcome is None:
            continue
        need_ids = {item["need_id"] for item in outcome["plan"]["task_state"]["needs"]}
        support = {need for result in outcome["results"] for need in result["support"]}
        unit_ids = {item["unit_id"] for item in outcome["plan"]["selected_set"]["selected_units"]}
        covered_units = {item[2] for item in outcome["coverage"]}
        needs, supported = needs+len(need_ids), supported+len(need_ids & support)
        units, covered = units+len(unit_ids), covered+len(unit_ids & covered_units)
        open_needs += len(outcome["open_needs"])
    for row in records:
        arrival = row["workload_arrival"]
        traces.append(dict(task_id=row["task_id"], conversation_id=row["conversation_id"], turn=row["turn"],
            actual_arrival=dispatch["origin"]+arrival["actual_arrival_seconds"] if dispatch is not None else None,
            planned_arrival_seconds=arrival["planned_arrival_seconds"], complete=row["request_window"][1],
            latency_ms=row["cost"]["complete_ms"], goal_id=row["question_policy"]["goal_id"],
            branch=row["question_policy"]["branch"], prior_qa_turns=len(row["actual_history"]),
            prefill_tokens=row["cost"]["actual_prefill_tokens"],
            device_hit_tokens=row["cost"]["actual_device_hit_tokens"],
            host_reload_tokens=row["cost"]["actual_host_loaded_tokens"]))
    return dict(job=job["name"], method=job["method"], concurrency=job["concurrency"],
        clients=job["clients"], requested_qa=job["requested_qa"], completed_qa=len(records),
        submitted_qa=len(arrivals), failed_clients=len(failures),
        completed_qa_coverage=len(records)/job["requested_qa"],
        throughput_qa_per_second=len(records)/elapsed if elapsed is not None and elapsed > 0 else None,
        makespan_seconds_including_pause=elapsed, p95_complete_ms=_quantile(latencies, .95),
        successful_qa_tokens={name: (sum(row["cost"][name] for row in records)
            if all(row["cost"][name] is not None for row in records) else None) for name in (
            "total_input_tokens", "total_output_tokens", "actual_prefill_tokens",
            "actual_device_hit_tokens", "actual_host_loaded_tokens")},
        failure_cost_scope="actual failed Attempt and StepRecords + full batch physical/native sampler; successful QA token sums do not include missing failure/auxiliary tokens",
        failed_clients_detail=failures, actual_arrival_completion=traces,
        need_support=dict(required=needs, supported=supported, open=open_needs,
            coverage=supported/needs if needs else None),
        source_unit_read_coverage=dict(selected=units, read=covered, coverage=covered/units if units else None),
        batch_costs=batch_costs,
        performance_scope="observed completed responses; failure coverage reported separately; shared physical batch counted once, not sum of request shares")


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
    state = dict(format=("contextkv_full_queue_joint_workflow_v1" if args.full_queue
                         else "contextkv_two_turn_capacity_workflow_v1"), scope=plan["scope"],
        status="running", requested_qa=plan["requested_qa"], jobs=[], results=[])
    try:
        for job in plan["jobs"]:
            if deadline is not None and time.time() >= deadline:
                state["jobs"].append({**job, "status": "not_started_deadline"})
                continue
            from profiling.runtime_manifest import collect_local_runtime, collect_server
            hardware = await asyncio.to_thread(collect_local_runtime, include_torch=False)
            native_runtime = await collect_server(scope["native_endpoint"], research=True)
            command = [args.python or sys.executable, "-m", "bench.study", "evaluate", "--config", plan["config"],
                "--matrix", job["matrix"], "--split", "dev", "--methods", job["method"],
                "--concurrency", str(job["concurrency"]), "--output", job["output"]]
            if args.full_queue:
                command += ["--datasets", *job["datasets"], "--workload-plan", job["workload"]]
            else:
                command += ["--datasets", "dynamic_goal_dev", "--workload-plan", job["workload"],
                            "--goal-policy", job["goal_policy"], "--unscored"]
            directory = root/"jobs"/job["name"]
            directory.mkdir(parents=True, exist_ok=True)
            record = {**job, "command": command, "status": "running", "started_s": time.monotonic()}
            state["jobs"].append(record)
            write_json(root/"workflow.json", state)
            with (directory/"process.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(*command, cwd=Path(args.project_root).resolve(),
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                record["pid"] = process.pid
                write_json(root/"workflow.json", state)
                try:
                    code = await asyncio.wait_for(process.wait(), max(.1, deadline-time.time())) if deadline else await process.wait()
                except TimeoutError:
                    process.terminate()
                    code = await process.wait()
                    record["deadline_terminated"] = True
            record.update(returncode=code, ended_s=time.monotonic(),
                          status="completed" if code == 0 else "failed_with_existing_raw_preserved")
            summary = summarize_full_queue(job) if args.full_queue else summarize_job(job)
            summary["runtime_hardware"] = hardware
            summary["native_cache_config"] = native_runtime["cache_config"]
            write_json(directory/"capacity-summary.json", summary)
            state["results"].append(summary)
            write_json(root/"workflow.json", state)
            print(json.dumps({"job": job["name"], "status": record["status"],
                "completed": summary["completed_qa"], "requested": job["requested_qa"],
                "p95_ms": summary["p95_complete_ms"]}), flush=True)
    finally:
        end = time.monotonic()
        await native.stop()
        await physical.stop()
        child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        series, coverage = pool_series(native.observation(), started, end)
        write_json(root/"capacity-cost.json", dict(format=("contextkv_full_queue_joint_cost_v1" if args.full_queue
                   else "contextkv_two_turn_capacity_cost_v1"),
            source="hardware", resource_window=dict(start=started, end=end),
            resource_scope=scope, physical_observation=physical.observation(),
            physical_cost=integrate_physical(physical.samples, started, end),
            native_observation=native.observation(), native_cost={**occupancy_integral(series), **coverage},
            controller_cpu_seconds=time.process_time()-cpu_start,
            child_cpu_seconds=child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime,
            cost_scope="full sequential capacity jobs including child provider initialization, pauses, failed requests and idle gaps; nested per-job/response shares are not added"))
        counts = Counter()
        for row in state["results"]:
            counts[row["method"]] += row["completed_qa"]
        state.update(status="finished_with_actual_coverage", completed_by_method=dict(counts),
            performance_scope=plan["scope"], full_cost=str(root/"capacity-cost.json"))
        write_json(root/"workflow.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Study configuration with frozen A for full-queue evaluation")
    parser.add_argument("--arrival-plan", help="Dataset workload index with explicit fixed_arrivals schedules")
    parser.add_argument("--complete-slo-seconds", type=float, help="Buffered arrival-to-completion SLO for request goodput")
    parser.add_argument("--full-queue", action="store_true", help="Run all declared development tasks in each B/C cell")
    parser.add_argument("--methods", nargs="+", choices=(*FULL_QUEUE_METHODS, *OPTIONAL_FULL_QUEUE_METHODS),
                        help="Select full-queue cells for this reader")
    parser.add_argument("--source-manifest", help="Source manifest for the two-turn capacity diagnostic")
    parser.add_argument("--matrix")
    parser.add_argument("--parent-policy-asset")
    parser.add_argument("--gain-asset")
    parser.add_argument("--execution-profile")
    parser.add_argument("--native-assets")
    parser.add_argument("--value-model", help="Asset for explicitly selected optional *_value retention cells")
    parser.add_argument("--external-reranker", default="models/bge-reranker-v2-m3",
                        help="Released reranker checkpoint for the hybrid LlamaIndex reference")
    parser.add_argument("--contextpilot-checkout", help="Pinned author source checkout for ContextPilot compilation cells")
    parser.add_argument("--concurrency", type=int, nargs="+", choices=(1, 4, 8), default=[1, 4])
    parser.add_argument("--gain-mode", choices=("observed_actions", "whole_plan"), default="observed_actions")
    parser.add_argument("--reader-profile-id")
    parser.add_argument("--online-gpu-devices", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--online-process-roots", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--reader-base-url")
    parser.add_argument("--gap-max-units", type=int)
    parser.add_argument("--gap-max-passages", type=int)
    parser.add_argument("--lambda-time", type=float)
    parser.add_argument("--read-outputs", type=lambda value: tuple(map(int, value.split(","))), default=(2048,))
    parser.add_argument("--sampling-interval", type=float, default=.5)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python")
    parser.add_argument("--deadline")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.methods is not None and not args.full_queue:
        parser.error("--methods requires --full-queue")
    if args.full_queue:
        if not args.matrix or not args.online_gpu_devices or not args.reader_base_url or not args.arrival_plan:
            parser.error("--full-queue requires --matrix, --online-gpu-devices, --reader-base-url and --arrival-plan")
        methods = {method for concurrency in args.concurrency
                   for method in full_queue_methods(concurrency, args.methods)}
        if any(method.startswith("S3_") for method in methods):
            if not args.gain_asset:
                parser.error("S3 full-queue methods require --gain-asset")
        elif not args.reader_profile_id:
            parser.error("fixed-B full-queue methods require --reader-profile-id")
        if any(method.endswith("_value") for method in methods) and not args.value_model:
            parser.error("value full-queue methods require --value-model")
    elif not all((args.source_manifest, args.parent_policy_asset, args.execution_profile,
                  args.native_assets, args.reader_profile_id, args.gain_asset)):
        parser.error("the two-turn diagnostic requires --source-manifest, --parent-policy-asset, "
                     "--execution-profile, --native-assets, --reader-profile-id and --gain-asset")
    return asyncio.run(run(args))


if __name__ == "__main__":
    main()
