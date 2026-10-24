"""Plan finite-capacity research-controller comparisons; execute only explicitly.

Execution owns one local patched server per capacity/repetition and reuses the
experiment runner's cold hook, generation records and summaries. No SSH, stock
server changes, online artifact rewriting or implicit calibration is performed.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import yaml

from bench import native_policy_sweep as server
from bench.experiments import execute_job, run_matrix
from controller.config import load_config
from controller.cost_model import CostModel


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "Ours": {"planner": {"strategy": "cost"}, "ablation": {"epoch_retirement": True}},
    "Ours_no_retire": {"planner": {"strategy": "cost"}, "ablation": {"epoch_retirement": False}},
    "Fixed_4096": {"planner": {"strategy": "fixed_threshold", "fixed_token_threshold": 4096},
                   "ablation": {"epoch_retirement": True}},
    "Fixed_8192": {"planner": {"strategy": "fixed_threshold", "fixed_token_threshold": 8192},
                   "ablation": {"epoch_retirement": True}},
    "Rebuild": {"planner": {"strategy": "always_rebuild"}, "ablation": {"epoch_retirement": True}},
    "Rebuild_no_retire": {"planner": {"strategy": "always_rebuild"}, "ablation": {"epoch_retirement": False}},
}


def artifact_plan(path, capacity):
    """Read existing calibration provenance once; never relabel a larger pool."""
    if path is None or not Path(path).is_file():
        return {"status": "requires_calibration", "reason": "artifact_missing", "path": str(path) if path else None}
    metadata = json.loads(Path(path).read_text()).get("metadata", {})
    manifest = metadata.get("profile_manifest") or {}
    manifests = metadata.get("profile_manifests") or ([manifest] if manifest else [])
    capacities = {m.get("cache_config", {}).get("gpu_capacity_tokens", m.get("pool", {}).get("gpu_capacity_tokens"))
                  for m in manifests}
    bound = manifest.get("engine", {}).get("cache_config_hash") and (
        manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint"))
    compatible = bound and capacities == {capacity}
    return {"status": "requires_runtime_scope_check" if compatible else "requires_calibration",
            "reason": "declared_capacity_matches" if compatible else "capacity_mismatch_or_unbound",
            "path": str(Path(path).resolve()), "declared_gpu_capacities": sorted(capacities, key=str),
            "profile_run_ids": [m.get("run_id") for m in manifests],
            "engine_configuration_only": True, "hardware_equivalence_verified": False}


def build_plan(args):
    base = load_config(args.config)
    artifacts = {}
    for item in args.cost_artifact:
        capacity, path = item.split("=", 1)
        capacity = int(capacity)
        if capacity in artifacts or capacity not in args.gpu_tokens:
            raise ValueError("cost artifacts must name distinct requested capacities: TOKENS=PATH")
        artifacts[capacity] = str(Path(path).resolve())
    cost_methods = any(METHODS[name]["planner"]["strategy"] == "cost" for name in args.methods)
    groups = []
    for repetition in range(args.repetitions):
        capacities = args.gpu_tokens[repetition % len(args.gpu_tokens):] + args.gpu_tokens[:repetition % len(args.gpu_tokens)]
        names = args.methods[repetition % len(args.methods):] + args.methods[:repetition % len(args.methods)]
        for capacity in capacities:
            directory = args.output/f"gpu-{capacity}"/f"r{repetition+1}"
            artifact = artifacts.get(capacity, base.cost_model.artifact)
            calibration = artifact_plan(artifact, capacity)
            config = replace(base,
                model=replace(base.model, name=str(args.model_path), revision=server.MODEL_REVISION, local_files_only=True),
                engine=replace(base.engine, kind="sglang", base_url=f"http://127.0.0.1:{args.port}", stream=True, page_size=64),
                context=replace(base.context, max_total_model_tokens=32768, max_new_tokens=1024),
                planner=replace(base.planner, concurrency=4),
                cost_model=replace(base.cost_model, artifact=calibration["path"] if cost_methods else None,
                                   online_calibration=False), calibration_path=None, shadow=False)
            config_path = directory/"controller.yaml"
            matrix = {"schema_version": 1, "base_config": str(config_path), "input": str(args.input),
                "split_manifest": str(args.split_manifest), "domain": "cloud", "protocols": ["P2"],
                "concurrency": [4], "seeds": [42], "reference_method": "Rebuild", "tuning": {"equal_budget": 1},
                "lifecycle": {"before_job": [sys.executable, "-m", "scripts.prepare_cold_job",
                    "--base-url", config.engine.base_url, "--config", str(config_path)], "cwd": str(ROOT),
                    "cache_state_protocol": "dedicated_research_cold_reset_no_generation_warmup", "timeout_seconds": 180},
                "methods": [{"id": name, "overrides": deepcopy(METHODS[name])} for name in names]}
            launch = server.server_command(argparse.Namespace(port=args.port, gpu_tokens=capacity))
            environment = {"CUDA_VISIBLE_DEVICES": "0", "SGLANG_CHECKOUT": str(args.checkout),
                "CONTEXTKV_MODEL_PATH": str(args.model_path), "CONTEXTKV_ENABLE_HICACHE": "1",
                "CONTEXTKV_HICACHE_GB": "16", "CONTEXTKV_WRITE_POLICY": "write_through",
                "CONTEXTKV_HICACHE_LAYOUT": "layer_first", "CONTEXTKV_HICACHE_IO_BACKEND": "kernel",
                "CONTEXTKV_PAGE_SIZE": "64", "SGLANG_HICACHE_TMA_TRANSFER": "0",
                "CONTEXTKV_SAMPLE_POOL": "1", "CONTEXTKV_ENABLE_PROFILING": "1", "RAYON_NUM_THREADS": "4"}
            groups.append({"directory": str(directory), "requested_gpu_tokens": capacity,
                "model_context_tokens": 32768, "repetition": repetition+1, "server_argv": launch,
                "server_environment": environment, "config": asdict(config), "matrix": matrix,
                "calibration": calibration, "cost_methods_requested": cost_methods,
                "process_protocol": "new server per capacity/repetition; KV-only cold reset between methods; JIT/kernel state shared",
                "native_initial_state_matched": False, "b6_comparison_established": False,
                "matched_retirement_pairs": [[a, b] for a, b in (("Ours", "Ours_no_retire"), ("Rebuild", "Rebuild_no_retire"))
                                             if a in names and b in names],
                "scope": "controller-only Cloud dev P2/c4/cap1024; fixed parameters, no tuning or quality eligibility"})
    return {"execute": False, "groups": groups,
            "controller_thread_environment": {"RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
                "scope": "parent process launch settings; actual worker count not measured"},
            "calibration_required_for_cost_methods": True, "test_data_used": False,
            "sampling_note": "calibrate each actual pool separately; profiler does not precheck GPU prefix residency; see docs/controller_capacity_sweep.md"}


def check_research_checkout(checkout):
    if (checkout/".contextkv-source-only").exists():
        raise ValueError("source-only checkouts cannot launch a server")
    head = server.command(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if head != server.COMMIT:
        raise ValueError("research checkout must use the pinned SGLang commit")
    server.command(["git", "-C", str(checkout), "apply", "--reverse", "--check",
                    str(ROOT/"engine_patches/0001-contextkv-protocol.patch")])
    return {"path": str(checkout), "commit": head, "research_patch_applied": True,
            "verification": "read-only reverse-apply check of the current patch"}


def validate_runtime(group, runtime, *, expected_host_tokens=None):
    """Actual scheduler capacities and unchanged cost binding, before generation."""
    capacity = group["requested_gpu_tokens"]
    info, pool, cache = runtime["engine"], runtime["pool"], runtime["cache_config"]
    expected = {"gpu_capacity_tokens": capacity, "page_size": 64, "hicache": True,
        "host_memory_mode": "cache", "host_layout": "layer_first", "hicache_io_backend": "kernel",
        "write_policy": group.get("server_environment", {}).get("CONTEXTKV_WRITE_POLICY", "write_through"),
        "tma_transfer_enabled": False, "sample_pool": True}
    mismatch = {key: {"expected": value, "observed": cache.get(key)} for key, value in expected.items()
                if cache.get(key) != value}
    if (info.get("engine_kind") != "sglang" or not info.get("profiling_enabled")
            or info.get("sglang_commit") != server.COMMIT or mismatch
            or pool["gpu_capacity_tokens"] != capacity or pool["host_capacity_tokens"] <= 0
            or runtime["server_info"].get("hicache_size") != 16
            or (expected_host_tokens is not None and pool["host_capacity_tokens"] != expected_host_tokens)):
        raise ValueError(f"actual research server pool/host settings differ from the sweep: {mismatch}")
    scope = None
    if group["cost_methods_requested"]:
        options = group.get("config", {}).get("cost_model", {})
        estimator = options.get("admission_estimator")
        model = CostModel.load(group["calibration"]["path"], max_new_tokens=1024,
                               **({"admission_estimator": estimator} if estimator else {}))
        scope = model.check_calibration_scope(info)
        if scope["status"] != "matching_scope":
            raise ValueError(f"Ours requires a newly bound cost artifact for this actual pool: {scope}")
    return {"pool": pool, "cache_config": cache, "calibration_scope": scope,
            "gpu_hardware_equivalence_verified": False}


async def execute_matrix(spec, output, timeout):
    async def bounded_executor(spec, job, directory):
        return await asyncio.wait_for(execute_job(spec, job, directory), timeout=timeout)
    return await run_matrix(spec, output, executor=bounded_executor)


def execute(args, plan):
    from controller.templates import HuggingFaceTemplate
    from profiling.runtime_manifest import build_manifest

    for group in plan["groups"]:
        if group["cost_methods_requested"] and group["calibration"]["status"] == "requires_calibration":
            raise ValueError(f"GPU pool {group['requested_gpu_tokens']} needs its own calibration; fixed methods may be run separately")
    source = check_research_checkout(args.checkout)
    template_config = load_config(args.config).model
    template = HuggingFaceTemplate(str(args.model_path), server.MODEL_REVISION,
                                  namespace_hash=template_config.namespace_hash, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    server.write_json(args.output/"plan.json", dict(plan, execute=True, source_checkout=source))
    results, host_tokens = [], None
    for group in plan["groups"]:
        directory = Path(group["directory"])
        directory.mkdir(parents=True)
        (directory/"controller.yaml").write_text(yaml.safe_dump(group["config"], sort_keys=False))
        (directory/"matrix.yaml").write_text(yaml.safe_dump(group["matrix"], sort_keys=False))
        env = dict(os.environ, **group["server_environment"], CONTEXTKV_FINGERPRINT_JSON=json.dumps(asdict(template.fingerprint)))
        env["PYTHONPATH"] = str(args.checkout/"python")+os.pathsep+str(ROOT)
        env["PATH"] = str(Path(sys.executable).parent)+os.pathsep+env["PATH"]
        origin = server.command([sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('sglang').origin)"], env=env)
        if not Path(origin).resolve().is_relative_to(args.checkout/"python"):
            raise ValueError("Python resolves a different SGLang checkout")
        # Never adopt/stop an existing process; the operator owns other servers.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        process, record = None, {"requested_gpu_tokens": group["requested_gpu_tokens"], "repetition": group["repetition"], "status": "started"}
        try:
            with (directory/"startup.log").open("w") as log:
                process = subprocess.Popen(group["server_argv"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                server.write_json(directory/"owned-server.json", {"pid": process.pid, "process_group": process.pid})
                base_url = group["config"]["engine"]["base_url"]
                server.wait_ready(process, base_url, args.startup_timeout)
                runtime = asyncio.run(build_manifest(base_url=base_url, research=True, source_checkout=args.checkout))
                server.write_json(directory/"runtime-before.json", runtime)
                scope = validate_runtime(group, runtime, expected_host_tokens=host_tokens)
                host_tokens = scope["pool"]["host_capacity_tokens"]
                server.write_json(directory/"scope.json", scope)
                jobs = asyncio.run(execute_matrix(group["matrix"], directory/"matrix-results", args.job_timeout))
                record.update(status="completed" if all(j["status"] == "completed" and
                    j["summary"]["timing"]["failure_fraction"] == 0 and j["summary"]["cohort"]["unfinished_turns"] == 0
                    for j in jobs) else "generation_failures", jobs=[{"job_id": j["job_id"], "status": j["status"]} for j in jobs])
                server.write_json(directory/"runtime-after.json", asyncio.run(build_manifest(base_url=base_url, research=True, source_checkout=args.checkout)))
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if process is not None:
                server.stop_owned_server(process)
                record["server_returncode_after_cleanup"] = process.returncode
            server.write_json(directory/"group.json", record)
            results.append(record)
            server.write_json(args.output/"results.json", results)
        print(json.dumps(record), flush=True)
    return 0 if all(result["status"] == "completed" for result in results) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/hardware_dev_controller.yaml")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=ROOT/"vendor/sglang")
    parser.add_argument("--input", type=Path, default=ROOT/"data/mtrag/RAG.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=ROOT/"data/mtrag/split_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/controller-capacity-sweep")
    parser.add_argument("--gpu-tokens", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--cost-artifact", action="append", default=[], metavar="TOKENS=PATH")
    parser.add_argument("--methods", choices=tuple(METHODS), nargs="+", default=list(METHODS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for name in ("config", "model_path", "checkout", "input", "split_manifest", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.repetitions < 1 or len(set(args.gpu_tokens)) != len(args.gpu_tokens)
            or any(n < 64 or n % 64 for n in args.gpu_tokens) or len(set(args.methods)) != len(args.methods)):
        parser.error("positive repetitions, distinct capacities divisible by 64 and distinct methods required")
    plan = build_plan(args)
    if args.execute:
        return execute(args, plan)
    else:
        print(json.dumps(plan, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
