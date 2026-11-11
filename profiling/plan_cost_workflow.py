"""One sequential reader-bound collection, native fit and coverage workflow."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import resource
import sys
import time

import httpx

from controller.claim_capacity import claim_count_ceiling
from controller.config import load_config
from profiling.retention_capture_cost import CaptureRunAccounting
from profiling.retention_trace import write_json


def entry_path(value):
    path = Path(value).resolve()
    return path/"capture-entry.json" if path.is_dir() else path


async def fit_existing_only(args, output, manifest, checkpoint):
    """Offline fitting/audit never contacts or clears the active B reader."""
    from bench.study_resources import StudyResourceSampler, integrate_physical
    sampler = StudyResourceSampler((), (), interval=.2)
    await sampler.start()
    started, own_cpu = time.monotonic(), time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    stages = manifest["stages"]
    commands = []
    sources = []
    for path in args.existing_capture:
        sources += ["--capture-entry", str(entry_path(path))]
    if sources:
        commands.append(("c2-native-fit", "profiling.plan_native_fit", [*sources,
            "--reader-checkpoint", checkpoint, "--current-read-prompt",
            "--output-dir", str(output/"c2-native")]))
    if args.coverage_trace:
        commands.append(("coverage", "profiling.retention_coverage", ["--trace", str(Path(args.coverage_trace).resolve()),
            "--horizons", args.horizons, "--output", str(output/"coverage.json")]))
    try:
        for name, module, arguments in commands:
            row = dict(name=name, status="running", started_s=time.monotonic())
            stages.append(row)
            write_json(output/"workflow.json", manifest)
            with (output/(name+".log")).open("wb") as log:
                process = await asyncio.create_subprocess_exec(args.python or sys.executable,
                    "-m", module, *arguments, cwd=Path(args.project_root).resolve(),
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                row["pid"] = process.pid
                code = await process.wait()
            row.update(returncode=code, status="completed" if code == 0 else "unavailable", ended_s=time.monotonic())
            write_json(output/"workflow.json", manifest)
            print(json.dumps({"stage": name, "status": row["status"]}), flush=True)
    finally:
        end = time.monotonic()
        await sampler.stop()
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        write_json(output/"offline-fit-cost.json", dict(scope="offline fitting/audit process tree only; no reader RPC or GPU inference",
            window={"start": started, "end": end}, elapsed_seconds=end-started,
            own_cpu_seconds=time.process_time()-own_cpu,
            child_cpu_seconds=(child_after.ru_utime+child_after.ru_stime-child_before.ru_utime-child_before.ru_stime),
            physical_cost=integrate_physical(sampler.samples, started, end),
            selected_gpu_devices=[], active_B_reader_excluded=True))
    manifest.update(status="offline_complete", mode="fit_existing_only",
        reader_mutations=0, native_asset_manifest=str(output/"c2-native"/"native-assets.json")
            if (output/"c2-native"/"native-assets.json").exists() else None)
    write_json(output/"workflow.json", manifest)
    return manifest


async def workflow(args):
    project, output = Path(args.project_root).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    from dataclasses import replace
    config = replace(config, engine=replace(config.engine, base_url=args.base_url),
                     state_dir=str(output/"driver-state"))
    checkpoint = args.reader_checkpoint or config.model.revision
    deadline = datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None
    stages, new_captures, new_phases, blocked = [], [], [], False
    manifest = dict(format="contextkv_plan_cost_workflow_v1", status="running", stages=stages,
        reader_checkpoint=checkpoint, reader_profile_id=args.reader_profile_id,
        native_endpoint=args.base_url, prior_capture_entries=[str(entry_path(path)) for path in args.existing_capture],
        deadline=args.deadline, generation_scope="each collection stage runs sequentially on this independent reader",
        cost_rule="workflow capture-entry integral counts resident devices/processes once; nested stage costs are not summed",
        runtime_integration={"claim_capacity": "S3 consumer implemented; actual bound/scope qualification required",
                             "current_answer_remaining_control": "S3 consumer implemented; actual per-cap/control-scope qualification required"})
    if args.fit_existing_only:
        return await fit_existing_only(args, output, manifest, checkpoint)
    if args.study_config is None or not args.claim_source_entry or args.reader_profile_id is None:
        raise ValueError("collection mode requires study-config, claim-source-entry and the actual reader-profile-id")

    def progress():
        write_json(output/"workflow.json", manifest)

    python = args.python or sys.executable
    resources = ["--gpu-device", str(args.gpu_device)]
    if args.resource_gpu_devices:
        resources += ["--resource-gpu-devices", args.resource_gpu_devices]
    if args.resource_process_roots:
        resources += ["--resource-process-roots", args.resource_process_roots]
    environment = ["--config", str(Path(args.config).resolve()), "--base-url", args.base_url, *resources,
                   "--study-config", str(Path(args.study_config).resolve())]
    sources = []
    for path in args.source_trace:
        sources += ["--source-trace", str(Path(path).resolve())]
    for path in args.source_bank:
        sources += ["--source-bank", str(Path(path).resolve())]
    claim_sources = []
    for path in args.claim_source_entry:
        claim_sources += ["--source-capture-entry", str(entry_path(path))]

    async with CaptureRunAccounting(config, output, args.gpu_device,
        resource_gpu_devices=tuple(map(int, args.resource_gpu_devices.split(","))) if args.resource_gpu_devices else None,
        resource_process_roots=tuple(map(int, args.resource_process_roots.split(","))) if args.resource_process_roots else ()) as accounting:
        accounting.case_outcomes = stages

        async def stage(name, module, arguments, *, collected=None, fit=False):
            nonlocal blocked
            if blocked and collected is not None:
                stages.append(dict(name=name, status="not_started_unresolved_native_outcome"))
                progress()
                return False
            if collected is not None and deadline is not None and time.time() >= deadline:
                stages.append(dict(name=name, status="not_started_deadline"))
                progress()
                return False
            if collected is not None:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(args.base_url.rstrip("/")+"/context/metrics", json={})
                    response.raise_for_status()
                    workload = response.json()
                    if workload["active_requests"] or workload["queued_requests"]:
                        blocked = True
                        stages.append(dict(name=name, status="reader_not_idle", native_workload=workload))
                        progress()
                        return False
            command = [python, "-m", module, *arguments]
            row = dict(name=name, status="running", command=command, started_s=time.monotonic())
            stages.append(row)
            progress()
            with (output/(name+".log")).open("wb") as log:
                process = await asyncio.create_subprocess_exec(*command, cwd=project,
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                row["pid"] = process.pid
                progress()
                code = await process.wait()
            row.update(returncode=code, ended_s=time.monotonic(),
                       status="completed" if code == 0 else "unavailable" if fit else "failed")
            if collected is not None:
                entry = collected/"capture-entry.json"
                if entry.exists():
                    data = json.loads(entry.read_text())
                    row["capture_entry"] = str(entry)
                    row["case_statuses"] = [item["status"] for item in data.get("case_outcomes") or ()]
                    wire = collected/"capture-wire.json"
                    if wire.exists():
                        accounting.calls.extend(json.loads(wire.read_text()))
                    unresolved = data["generations"]["unresolved_native_attempts"]
                    if unresolved:
                        blocked = True
                        row.update(status="unresolved_native_outcome", unresolved=unresolved)
                    elif code and not data.get("case_outcomes"):
                        blocked = True
                        row["status"] = "infrastructure_error"
            progress()
            print(json.dumps({"stage": name, "status": row["status"], "returncode": code}), flush=True)
            return code == 0

        async with httpx.AsyncClient(timeout=30) as client:
            info_response = await client.post(args.base_url.rstrip("/")+"/context/engine/info", json={})
            info_response.raise_for_status()
            info = info_response.json()
        if info["fingerprint"]["model_revision"] != checkpoint:
            raise ValueError("workflow reader endpoint and scoped checkpoint differ")
        manifest["actual_engine_info"] = info
        progress()

        old = []
        for path in args.existing_capture:
            old += ["--capture-entry", str(entry_path(path))]
        if old:
            await stage("01-c2-native", "profiling.plan_native_fit", [*old,
                "--reader-checkpoint", checkpoint, "--current-read-prompt",
                "--output-dir", str(output/"c2-native")], fit=True)

        c3 = output/"c3-role"
        await stage("02-c3-role", "profiling.plan_cost_collect", [*environment, *sources,
            "--split", "dev", "--read-inputs", args.read_ceiling, "--read-outputs", args.read_output,
            "--answer-inputs", args.answer_ceiling, "--answer-output", args.answer_output,
            "--cache-states", "cold", "--max-total-generations", "3", "--output-dir", str(c3)], collected=c3)
        if (c3/"capture-trace.json").exists():
            new_captures.append(c3/"capture-entry.json")

        grid = output/"c3-current-grid"
        grid_calls = (2*len(args.read_grid.split(","))+len(args.answer_grid.split(",")))
        await stage("03-current-grid", "profiling.plan_cost_collect", [*environment, *sources,
            "--split", "dev", "--read-inputs", args.read_grid, "--read-outputs", args.read_output,
            "--answer-inputs", args.answer_grid, "--answer-output", args.answer_output,
            "--cache-states", "cold", "--max-total-generations", str(grid_calls), "--output-dir", str(grid)], collected=grid)
        if (grid/"capture-trace.json").exists():
            new_captures.append(grid/"capture-entry.json")

        capacity = output/"claim-capacity.json"
        await stage("04-claim-capacity", "profiling.claim_capacity_fit", ["--config", str(Path(args.config).resolve()),
            "--output", str(capacity)], fit=True)
        if capacity.exists():
            bound = json.loads(capacity.read_text())
            ceiling = claim_count_ceiling((int(args.parent_read_output),)*args.max_parent_reads, bound)
            basis = "actual tokenizer required-literal coverage bound"
        else:
            ceiling = int(args.parent_read_output)*args.max_parent_reads
            basis = "safe one-token-per-claim capacity; tighter grammar bound unavailable"
        counts = {1, ceiling}
        for boundary in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
            if boundary <= ceiling:
                counts.add(boundary)
            if boundary-1 <= ceiling:
                counts.add(boundary-1)
        manifest["cpu_capacity"] = dict(max_parent_reads=args.max_parent_reads,
            parent_output_budget=int(args.parent_read_output), ceiling=ceiling, basis=basis,
            measured_claim_counts=sorted(counts))
        progress()

        small = output/"reduce-current"
        await stage("05-reduce-current", "profiling.reduce_cost_collect", [*environment, *claim_sources,
            "--claim-counts", args.reduce_claim_counts, "--output-budgets", args.read_output,
            "--cache-states", "cold", "--replicate-frozen", "--max-total-generations",
            str(len(args.reduce_claim_counts.split(","))), "--output-dir", str(small)], collected=small)
        if (small/"phase-capture.json").exists():
            new_phases.append(small/"capture-entry.json")
        large = output/"reduce-capacity"
        await stage("06-reduce-capacity", "profiling.reduce_cost_collect", [*environment, *claim_sources,
            "--target-input-tokens", args.reduce_ceiling, "--max-input-tokens", args.reduce_ceiling,
            "--output-budgets", args.read_output, "--cache-states", "cold", "--replicate-frozen",
            "--reference-name-stress", "--max-total-generations", "1", "--output-dir", str(large)], collected=large)
        if (large/"phase-capture.json").exists():
            new_phases.append(large/"capture-entry.json")
        for operation in args.cpu_operations.split(","):
            directory = output/("cpu-"+operation)
            await stage("07-cpu-"+operation, "profiling.reduce_cost_collect", [*environment, *claim_sources,
                "--operation", operation, "--claim-counts", ",".join(map(str, sorted(counts))),
                "--replicate-frozen", "--output-dir", str(directory)], collected=directory)
            if (directory/"phase-capture.json").exists():
                new_phases.append(directory/"capture-entry.json")

        fit_sources = []
        for path in new_captures:
            fit_sources += ["--capture-entry", str(path)]
        for path in new_phases:
            fit_sources += ["--phase-entry", str(path)]
        fitting = [*fit_sources, "--reader-checkpoint", checkpoint, "--reader-profile-id", args.reader_profile_id,
            "--split", "dev", "--current-read-prompt", "--isolation", "dedicated_service_device",
            "--output", str(output/"execution-assets.json"),
            "--native-timing-output-dir", str(output/"native")]
        if capacity.exists():
            fitting += ["--claim-capacity-asset", str(capacity)]
        await stage("08-assets-fit", "profiling.plan_cost_fit", fitting, fit=True)
        if args.coverage_trace:
            await stage("09-coverage", "profiling.retention_coverage", ["--trace", str(Path(args.coverage_trace).resolve()),
                "--horizons", args.horizons, "--output", str(output/"coverage.json")], fit=True)
        manifest.update(status="unresolved_native_outcome" if blocked else "finished_with_observed_domains",
            new_capture_entries=list(map(str, new_captures)), new_phase_entries=list(map(str, new_phases)),
            execution_asset=str(output/"execution-assets.json") if (output/"execution-assets.json").exists() else None,
            native_asset_manifest=str(output/"native"/"native-assets.json") if (output/"native"/"native-assets.json").exists() else None,
            domain_reporting="actual input ranges, unavailable cases and stage errors remain in source ledgers; no extrapolated zero costs")
        progress()
    cost_path = output/"capture-entry.json"
    cost = json.loads(cost_path.read_text())
    cost.update(format="contextkv_cost_workflow_entry_v1", scope="full sequential workflow including fitting and inter-stage residency",
        nested_cost_rule="this entry is the full new workflow integral; do not sum nested collection integrals",
        prior_cost_entries=manifest["prior_capture_entries"])
    write_json(cost_path, cost)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reader-checkpoint")
    parser.add_argument("--reader-profile-id")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--resource-gpu-devices")
    parser.add_argument("--resource-process-roots")
    parser.add_argument("--study-config")
    parser.add_argument("--existing-capture", action="append", default=[])
    parser.add_argument("--source-trace", action="append", default=[])
    parser.add_argument("--source-bank", action="append", default=[])
    parser.add_argument("--claim-source-entry", action="append", default=[])
    parser.add_argument("--fit-existing-only", action="store_true", help="Offline C2 native fit and coverage only; zero reader RPC/cold")
    parser.add_argument("--coverage-trace")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deadline", help="Stop starting stages at this ISO timezone-aware wall time; do not kill in-flight native work")
    parser.add_argument("--read-ceiling", default="12288")
    parser.add_argument("--answer-ceiling", default="31744")
    parser.add_argument("--reduce-ceiling", default="30720")
    parser.add_argument("--read-output", default="2048")
    parser.add_argument("--answer-output", default="1024")
    parser.add_argument("--read-grid", default="1100,2000,2100,4000,4200,8000")
    parser.add_argument("--answer-grid", default="1100,2000,2100,4000,4200,8000")
    parser.add_argument("--reduce-claim-counts", default="1,4,16,64")
    parser.add_argument("--cpu-operations", default="count,sum,union,group")
    parser.add_argument("--parent-read-output", default="4096")
    parser.add_argument("--max-parent-reads", type=int, default=24)
    parser.add_argument("--horizons", default="1,2,5,8,15,30,60,120")
    args = parser.parse_args(argv)
    return asyncio.run(workflow(args))


if __name__ == "__main__":
    main()
