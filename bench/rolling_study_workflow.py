"""Sequential actual dev selection, frozen evaluation and goal-workload jobs.

Each child has a durable command/log/result directory. A failed conversation
ends its remaining dependent turns; other conversations continue. Model calls
are never retried and missing answers or scores are not assigned quality zero.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

from bench.study import (dataset_rows, declare_cohorts, freeze, merge_config,
                         read_jsonl, report, write_json, write_jsonl)
from bench.study_metrics import paired_comparisons, summarize_methods


def snapshot_file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def task_spec(spec, path, split):
    value = {key: item for key, item in spec.items() if key not in {
        "input", "inputs", "split_manifest", "domain", "dev_conversations", "dev_seed"}}
    return {**value, "inputs": {split: str(Path(path).resolve())}}


def configure(args, root):
    config = yaml.safe_load(Path(args.config).read_text())
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    execution = json.loads(Path(args.execution_profile).read_text())
    native = json.loads(Path(args.native_assets).read_text())
    gain = json.loads(Path(args.gain_asset).read_text())
    if native["base_artifact"] is None:
        raise ValueError("rolling evaluation requires the actual ANSWER native timing asset")
    profile = execution["reader_profile_id"]
    checkpoint = execution["reader_checkpoint"]
    if profile != args.reader_profile_id:
        raise ValueError("the declared reader profile differs from the measured execution asset")
    run = yaml.safe_load(Path(config["generation"]["run_config"]).read_text())
    run["engine"]["base_url"] = execution["resource_scope"]["native_endpoint"]
    baseline_path = root/"baseline-run.yaml"
    baseline_path.write_text(yaml.safe_dump(run, allow_unicode=True, sort_keys=False))
    run = merge_config(run, native["config_overlay"])
    run_path = root/"run.yaml"
    run_path.write_text(yaml.safe_dump(run, allow_unicode=True, sort_keys=False))
    settings = dict(mode="whole_plan", gain_asset=str(Path(args.gain_asset).resolve()),
        execution_profile_asset=str(Path(args.execution_profile).resolve()),
        reader_checkpoint=checkpoint, reader_profile_id=profile,
        gpu_devices=execution["resource_scope"]["gpu_devices"],
        process_ids=execution["resource_scope"]["process_roots"],
        sampling_interval_seconds=args.sampling_interval,
        output_budgets=args.read_outputs, prefix_directory=False,
        output_tokens_per_fact=config.get("reading", {}).get("output_tokens_per_fact", 64))
    config = merge_config(config, {"generation": {"run_config": str(baseline_path.resolve())},
        "execution": {"plan_gain": None}})
    methods = []
    by_id = {method["id"]: method for method in matrix["methods"]}
    for name in args.baselines:
        method = deepcopy(by_id[name])
        method["overrides"] = merge_config(method.get("overrides", {}),
                                            {"execution": {"plan_gain": None}})
        methods.append(method)
    for weight in args.lambda_time:
        identity = args.rolling_id+"-lambda"+format(weight, ".9g")
        methods.append({"id": identity, "strategy": "B7", "comparison": "actual_reader_bound_plan_gain",
            "overrides": {"generation": {"run_config": str(run_path.resolve())},
                "execution": {"plan_gain": settings, "lambda_time": weight,
                              "cost_artifact": native["base_artifact"]}}})
    methods.sort(key=lambda method: method.get("comparison") != "actual_reader_bound_plan_gain")
    matrix = merge_config(matrix, {"methods": methods, "selection": {
        "include_ablations": False, "baseline": args.baselines[0],
        "test_comparators": args.baselines}})
    matrix["resources"]["gpu_devices"] = settings["gpu_devices"]
    matrix["resources"]["process_ids"] = settings["process_ids"]
    config_path = root/"study.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    provenance = {"files": {name: snapshot_file(path) for name, path in (
        ("study", args.config), ("matrix", args.matrix), ("gain", args.gain_asset),
        ("execution_profile", args.execution_profile), ("native_assets", args.native_assets))},
        "reader_profile_id": profile, "reader_checkpoint": checkpoint,
        "gain_identity": {name: gain.get(name) for name in ("asset_id", "reader_checkpoint", "reader_profile_id")},
        "resource_scope": execution["resource_scope"], "failure_unit": args.failure_unit,
        "cache_start": config["execution"].get("cache_start", "observed"),
        "cohort_limits": {"dev_conversations_per_track": args.dev_conversations_per_track,
                          "test_conversations_per_track": args.test_conversations_per_track},
        "profile_domain": "actual resource/input/output/concurrency support is checked at each real decision"}
    write_json(root/"binding.json", provenance)
    return config, matrix, config_path


def cohort_matrix(matrix, split, output, limit):
    value = deepcopy(matrix)
    for name, spec in matrix["datasets"].items():
        if split == "dev" and spec.get("evaluation_only"):
            continue
        rows = dataset_rows(spec, split)
        if limit is not None:
            selected = set(sorted({row["conversation_id"] for row in rows})[:limit])
            rows = [row for row in rows if row["conversation_id"] in selected]
        path = output/"cohorts"/split/(name+".jsonl")
        write_jsonl(path, rows)
        value["datasets"][name] = task_spec(spec, path, split)
    return declare_cohorts(value, splits=(split,))


async def workflow(args):
    from bench.study_resources import StudyResourceSampler, integrate_physical
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config, matrix, config_path = configure(args, root)
    profile = json.loads(Path(args.execution_profile).read_text())
    sampler = StudyResourceSampler(tuple(profile["resource_scope"]["gpu_devices"]),
                                   tuple(profile["resource_scope"]["process_roots"]))
    state = {"format": "contextkv_rolling_study_workflow_v1", "status": "running", "jobs": [],
        "public_quality_aggregate": "dev/test tracks only; goal-workload diagnostics remain separate"}

    def progress():
        write_json(root/"workflow.json", state)

    async def command(name, module, argv, *, selection_status=None):
        directory = root/"jobs"/name
        directory.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, "-m", module, *map(str, argv)]
        job = {"name": name, "command": cmd, "status": "running", "started": time.monotonic(),
            "log": str(directory/"process.log"), "selection_status": selection_status}
        state["jobs"].append(job)
        progress()
        with (directory/"process.log").open("w") as log:
            child = await asyncio.create_subprocess_exec(*cmd, stdout=log, stderr=asyncio.subprocess.STDOUT)
            job["pid"] = child.pid
            progress()
            code = await child.wait()
        job.update(status="completed" if code == 0 else "failed", returncode=code,
                   elapsed_ms=(time.monotonic()-job["started"])*1000)
        progress()
        print(json.dumps({"job": name, "status": job["status"], "returncode": code}), flush=True)
        return directory, code

    async def evaluate_group(stage, split, effective, methods, *, selection_status):
        destination = root/stage
        rows = []
        write_json(destination/"matrix.json", effective)
        write_json(destination/"effective_config.json", config)
        for method in methods:
            for dataset, spec in effective["datasets"].items():
                if split == "dev" and spec.get("evaluation_only"):
                    continue
                inputs = dataset_rows(spec, split)
                groups = {}
                for row in inputs:
                    key = row["conversation_id"] if args.failure_unit == "conversation" else "dataset"
                    groups.setdefault(key, []).append(row)
                for index, (conversation, source) in enumerate(sorted(groups.items())):
                    name = f"{stage}-{method['id']}-{dataset}-{index:05d}"
                    directory = root/"jobs"/name
                    input_path = destination/"inputs"/method["id"]/dataset/f"{index:05d}.jsonl"
                    write_jsonl(input_path, source)
                    worker = merge_config(effective, {"methods": [method], "datasets": {}})
                    worker["datasets"] = {dataset: task_spec(spec, input_path, split)}
                    worker = declare_cohorts(worker, splits=(split,))
                    matrix_path = input_path.with_suffix(".yaml")
                    matrix_path.write_text(yaml.safe_dump(worker, allow_unicode=True, sort_keys=False))
                    generation = directory/"generation"
                    write_json(destination/"scheduled"/(name+".json"), {
                        "conversation_id": conversation, "task_ids": [row["task_id"] for row in source],
                        "method": method, "matrix": snapshot_file(matrix_path)})
                    await command(name, "bench.study", ["evaluate", "--config", config_path,
                        "--matrix", matrix_path, "--split", split, "--methods", method["id"],
                        "--datasets", dataset, "--output", generation], selection_status=selection_status)
                    actual = generation/method["id"]/dataset
                    raw = actual/"results.unscored.jsonl"
                    scored = actual/"results.scored.jsonl"
                    if (raw.exists() and not scored.exists() and not (actual/"scoring").exists()
                            and read_jsonl(raw)):
                        # Score existing completed outputs once after a failed
                        # generation job; no request or QA is generated again.
                        await command(name+"-score", "bench.study_scoring", ["--config", config_path,
                            "--matrix", matrix_path, "--dataset", dataset, "--generation-dir", actual])
                    if scored.exists() or raw.exists():
                        rows.extend({**row, "evaluation_selection": {
                            "status": selection_status, "used_to_select_method": split == "dev"}}
                            for row in read_jsonl(scored if scored.exists() else raw))
                    write_jsonl(destination/"results.jsonl", rows)
        write_json(destination/"summaries.json", summarize_methods(effective, rows, split))
        return rows

    await sampler.start()
    started = time.monotonic()
    try:
        dev = cohort_matrix(matrix, "dev", root, args.dev_conversations_per_track)
        prespecified = cohort_matrix(matrix, "test", root, args.test_conversations_per_track)
        write_json(root/"prespecified-test.json", {"methods": matrix["methods"],
            "task_ids": prespecified["cohorts"]["test"],
            "protocol": "registered before dev outcomes; no public test outcomes used for selection"})
        dev_rows = await evaluate_group("dev", "dev", dev, dev["methods"],
                                       selection_status="development_candidate")
        summaries = summarize_methods(dev, dev_rows, "dev")
        if any(value["eligible"] for value in summaries.values()):
            # Keep the original test partition rules. The materialized dev
            # files contain only dev rows and cannot serve as test inputs.
            freeze_matrix = deepcopy(matrix)
            freeze_matrix["cohorts"] = dev["cohorts"]
            frozen = freeze(config, freeze_matrix, dev_rows, root/"frozen.json")
            state["development_freeze"] = {"status": "completed", "method": frozen["method_spec"]["id"]}
            progress()
            chosen = frozen["test_methods"]
            selection_status = "frozen_development_choice_and_prespecified_comparators"
        else:
            state["development_freeze"] = {"status": "ineligible", "summaries": summaries}
            chosen = matrix["methods"]
            selection_status = "unselected_prespecified"
            progress()
        if not args.skip_test:
            test_rows = await evaluate_group("test", "test", prespecified, chosen,
                                             selection_status=selection_status)
            state["test_execution"] = {"selection_status": selection_status,
                                       "methods": [method["id"] for method in chosen]}
            if state["development_freeze"]["status"] == "completed":
                frozen["matrix"] = prespecified
                write_json(root/"frozen.json", frozen)
                report(frozen, test_rows, root/"report")
            else:
                write_json(root/"report"/"summaries.json", summarize_methods(prespecified, test_rows, "test"))
                write_json(root/"report"/"paired_intervals.json", paired_comparisons(prespecified, test_rows))
                write_json(root/"report"/"selection.json", {"status": selection_status,
                    "development_freeze": "ineligible", "test_results_used_for_selection": False})
        if args.load_manifest:
            load = json.loads(Path(args.load_manifest).read_text())
            diagnostic = yaml.safe_load(Path(load["diagnostic_matrix_path"]).read_text())
            diagnostic["methods"] = chosen
            diagnostic["selection"]["include_ablations"] = False
            diagnostic["resources"]["gpu_devices"] = profile["resource_scope"]["gpu_devices"]
            path = root/"load-matrix.yaml"
            path.write_text(yaml.safe_dump(diagnostic, allow_unicode=True, sort_keys=False))
            for schedule in load["schedules"]:
                if args.load_schedules is not None and schedule["name"] not in args.load_schedules:
                    continue
                for method in chosen:
                    name = "load-"+schedule["name"]+"-"+method["id"]
                    await command(name, "bench.study", ["evaluate", "--config", config_path,
                        "--matrix", path, "--split", "dev", "--methods", method["id"],
                        "--output", root/"jobs"/name/"generation", "--workload-plan", schedule["path"],
                        "--goal-policy", load["goal_policy_path"]], selection_status=selection_status)
        state["status"] = "completed" if all(job["status"] == "completed" for job in state["jobs"]) else "finished_with_failed_jobs"
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        ended = time.monotonic()
        await sampler.stop()
        state.update(elapsed_ms=(ended-started)*1000,
            physical_cost=integrate_physical(sampler.samples, started, ended),
            physical_observation=sampler.observation(),
            cost_scope="whole workflow including initialization/offline scoring; per-answer online costs stay in original records; do not add overlapping integrals")
        progress()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "matrix", "output", "gain-asset", "execution-profile", "native-assets", "reader-profile-id"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--baselines", nargs="+", default=["B0", "B3", "B6"])
    parser.add_argument("--rolling-id", default="S3")
    parser.add_argument("--lambda-time", type=lambda value: tuple(map(float, value.split(","))),
                        default=(0.0, .000001, .00001, .0001))
    parser.add_argument("--read-outputs", type=lambda value: tuple(map(int, value.split(","))),
                        default=(512, 1024, 2048))
    parser.add_argument("--sampling-interval", type=float, default=.1)
    parser.add_argument("--failure-unit", choices=("conversation", "dataset"), default="conversation")
    parser.add_argument("--dev-conversations-per-track", type=int)
    parser.add_argument("--test-conversations-per-track", type=int)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--load-manifest")
    parser.add_argument("--load-schedules", nargs="+")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(workflow(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
