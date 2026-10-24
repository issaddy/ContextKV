"""Plan a finite 13-configuration P2 dev study; --execute owns one server per cell.

This phase is not a complete B6/test acceptance. It uses the existing matrix
executor and requires saved independent quality evidence before any B6 freeze.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import yaml

from bench import controller_capacity_sweep as research
from bench import native_policy_sweep as server
from bench.experiments import execute_job, run_matrix
from controller.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TEXT_POLICIES = {
    "Rebuild": {"strategy": "always_rebuild"},
    "Fixed4096": {"strategy": "fixed_threshold", "fixed_token_threshold": 4096},
    "Redundancy40": {"strategy": "redundancy_threshold", "redundancy_threshold": .4},
}
COMMON_SERVER = {"attention_backend": "fa3", "chunked_prefill_size": 8192,
    "schedule_policy": "fcfs", "max_running_requests": 8, "page_size": 64,
    "context_length": 32768, "max_total_tokens": 16384, "random_seed": 42,
    "skip_server_warmup": True, "tp_size": 1, "dp_size": 1}


def build_plan(args):
    base = load_config(args.config)
    config = replace(base,
        model=replace(base.model, name=str(args.model_path), revision=server.MODEL_REVISION,
                      namespace_hash="default", local_files_only=True),
        engine=replace(base.engine, kind="sglang", base_url=f"http://127.0.0.1:{args.port}", stream=True,
                       gpu_capacity_tokens=16384, page_size=64),
        context=replace(base.context, max_total_model_tokens=32768, max_new_tokens=1024, system_prompt=None),
        planner=replace(base.planner, strategy="always_rebuild", concurrency=4),
        cost_model=replace(base.cost_model, artifact=None, output_length_artifact=None,
                           online_calibration=False, admission_estimator=None, uncertainty_mode=None),
        calibration_path=None, shadow=False,
        ablation=replace(base.ablation, epoch_retirement=True))
    profiles, methods = {}, []
    for kind in ("patched", "stock"):
        for policy in server.POLICIES:
            profile_id = f"{kind}_{policy}"
            profiles[profile_id] = {"kind": kind, "write_policy": policy, "hicache": True,
                "session_radix": True, "gpu_tokens": 16384, "hicache_gb": 16,
                "checkout": str(args.patched_checkout if kind == "patched" else args.stock_checkout)}
    profiles["stock_B0"] = {"kind": "stock", "write_policy": None, "hicache": False,
        "session_radix": False, "gpu_tokens": 16384, "hicache_gb": 0, "checkout": str(args.stock_checkout)}
    for name, knobs in TEXT_POLICIES.items():
        for policy in server.POLICIES:
            methods.append({"id": f"{name}_{policy}", "family": name, "b6_eligible": True,
                "server_profile": f"patched_{policy}", "overrides": {"planner": knobs,
                    "ablation": {"epoch_retirement": True}}})
    for policy in server.POLICIES:
        methods.append({"id": f"B1_{policy}", "family": "B1", "kind": "native", "b6_eligible": True,
            "server_profile": f"stock_{policy}", "native": {"baseline": "B1", "base_url": config.engine.base_url,
                "model_path": str(args.model_path), "local_files_only": True, "audit_prompt_ids": True}})
    methods.append({"id": "B0", "family": "B0_reference", "kind": "native", "b6_eligible": False,
        "server_profile": "stock_B0", "native": {"baseline": "B0", "base_url": config.engine.base_url,
            "model_path": str(args.model_path), "local_files_only": True, "audit_prompt_ids": True}})
    spec = {"schema_version": 1, "base_config": str(args.output/"controller.yaml"),
        "input": str(args.input), "split_manifest": str(args.split_manifest), "domain": "cloud",
        "protocols": ["P2"], "concurrency": [4], "seeds": [42], "reference_method": "B0",
        "server_profiles": profiles, "methods": methods,
        "tuning": {"equal_budget": 1, "require_quality_evidence": True,
                   "quality_constraints": {"reference": "B0", "epsilon": .01}},
        "finite_enumeration": {"family_counts": {"Rebuild": 3, "Fixed4096": 3, "Redundancy40": 3, "B1": 3, "B0_reference": 1},
            "interpretation": "13 fixed configurations; three write policies per family, not equal family search effort inferred from equal_budget=1"},
        "lifecycle": {"cache_state_protocol": "fresh_process_empty_KV_no_generation_warmup"},
        "scope": "phase1 Cloud dev P2 only; no retriever, no Ours/B4/B5/ablations/P3/test claim"}
    argv = server.server_command(argparse.Namespace(port=args.port, gpu_tokens=16384)) + [
        "--attention-backend", "fa3", "--chunked-prefill-size", "8192", "--schedule-policy", "fcfs"]
    names = [m["id"] for m in methods]
    return {"execute": False, "config": asdict(config), "matrix": spec, "server_argv": argv,
        "common_server_expectations": COMMON_SERVER, "host_admission": "off", "normal_eos": True,
        "planned_cells": 13, "planned_turns": 533, "active_repetition": 1,
        "future_repeat_orders": [names[n:]+names[:n] for n in range(3)],
        "repetition_scope": "one repeat implemented; future orders predeclared, not new seeds or fabricated pooled results",
        "parent_thread_environment": {k: os.environ.get(k) for k in ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")},
        "server_initial_state": "new owned process per cell, skip generation warmup; disk/compiler caches and parent client process can persist",
        "quality_evaluated": False, "b6_selected": None, "test_content_used": False}


def server_environment(profile, args, fingerprint):
    checkout = Path(profile["checkout"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", SGLANG_CHECKOUT=str(checkout),
        CONTEXTKV_MODEL_PATH=str(args.model_path), CONTEXTKV_FINGERPRINT_JSON=json.dumps(fingerprint),
        CONTEXTKV_ENABLE_SESSION_RADIX="1" if profile["session_radix"] else "0",
        CONTEXTKV_ENABLE_HICACHE="1" if profile["hicache"] else "0",
        CONTEXTKV_HICACHE_GB="16", CONTEXTKV_WRITE_POLICY=profile["write_policy"] or "write_through",
        CONTEXTKV_HICACHE_LAYOUT="layer_first", CONTEXTKV_HICACHE_IO_BACKEND="kernel",
        CONTEXTKV_HOST_ADMISSION="off", CONTEXTKV_PAGE_SIZE="64", SGLANG_HICACHE_TMA_TRANSFER="0",
        CONTEXTKV_SAMPLE_POOL="1", CONTEXTKV_ENABLE_PROFILING="1" if profile["kind"] == "patched" else "0")
    env["PYTHONPATH"] = str(checkout/"python")+os.pathsep+str(ROOT)
    env["PATH"] = str(Path(sys.executable).parent)+os.pathsep+env["PATH"]
    return env


def check_common_runtime(info, runtime):
    mismatch = {k: {"expected": v, "observed": info.get(k)} for k, v in COMMON_SERVER.items() if info.get(k) != v}
    if mismatch:
        raise ValueError(f"actual engine scheduling/runtime settings differ: {mismatch}")
    gpu = next(g for g in runtime["gpus"] if g["index"] == 0)
    if runtime["collection_errors"] or any(not runtime["driver_cuda_torch_versions"].get(k) for k in ("driver", "cuda", "torch")):
        raise ValueError("actual GPU/driver/CUDA/torch runtime could not be observed")
    return {"gpu_uuid": gpu["uuid"], "gpu_model": gpu["name"], "gpu_memory_bytes": gpu["memory_bytes"],
            "driver_cuda_torch_versions": runtime["driver_cuda_torch_versions"],
            "cache_config_hash_is_not_runtime_equivalence_proof": True}


async def research_metrics(base_url, fingerprint):
    from controller.engine_client import SGLangEngineClient
    engine = SGLangEngineClient(base_url, fingerprint)
    try:
        return await engine.metrics()
    finally:
        await engine.aclose()


def quality_template(results, spec):
    """References real artifacts, but supplies no evaluator identity or scores."""
    return {"scope": "dev", "reference": "B0", "epsilon": spec["tuning"]["quality_constraints"]["epsilon"],
        "split_manifest": spec["split_manifest"], "frozen_protocol": None,
        "methods": {row["method"]: {"run": row.get("raw_output"), "audit": row.get("audit_output"),
            "manifest": row.get("run_manifest"),
            "server_record": row.get("server_execution", {}).get("record"),
            "scores": None, "generator": {"model": "Qwen/Qwen2.5-7B-Instruct", "revision": server.MODEL_REVISION,
                                           "base_family": "Qwen2.5"}}
            for row in results},
        "template_only": True, "note": "fill an independent frozen evaluator protocol and full answer-bound five-dimension scores; never copy eligibility booleans"}


def execute(args, plan):
    from adapters.mtrag import CLOUD_COLLECTION, load_rows
    from controller.templates import HuggingFaceTemplate
    from profiling.runtime_manifest import build_manifest

    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("launch the parent command with RAYON_NUM_THREADS=4; server-only settings do not configure the client")
    sources = {"patched": research.check_research_checkout(args.patched_checkout), "stock": server.check_stock(args.stock_checkout)}
    dev = set(json.loads(args.split_manifest.read_text())["dev"])
    cohort = [row for row in load_rows(args.input) if row["conversation_id"] in dev and row["Collection"] == CLOUD_COLLECTION]
    if len(cohort) != 41 or len({row["conversation_id"] for row in cohort}) != 5:
        raise ValueError("phase1 requires the complete five-session / 41-turn Cloud dev cohort")
    template = HuggingFaceTemplate(str(args.model_path), server.MODEL_REVISION, local_files_only=True)
    model_config = json.loads((args.model_path/"config.json").read_text())
    fingerprint = asdict(template.fingerprint)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"controller.yaml").write_text(yaml.safe_dump(plan["config"], sort_keys=False))
    (args.output/"matrix.yaml").write_text(yaml.safe_dump(plan["matrix"], sort_keys=False))
    server.write_json(args.output/"plan.json", {**plan, "execute": True, "source_checkouts": sources})
    hardware, host_tokens = None, None

    async def owned_executor(spec, job, directory):
        nonlocal hardware, host_tokens
        profile = spec["server_profiles"][job["server_profile"]]
        support = args.output/"servers"/job["job_id"]
        support.mkdir(parents=True)
        env = server_environment(profile, args, fingerprint)
        process, result = None, None
        record = {"job_id": job["job_id"], "server_profile": job["server_profile"], "profile": profile,
                  "status": "starting", "argv": plan["server_argv"], "source_checkout": sources[profile["kind"]]}
        base_url = f"http://127.0.0.1:{args.port}"
        try:
            origin = server.command([sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('sglang').origin)"], env=env)
            if not Path(origin).resolve().is_relative_to(Path(profile["checkout"])/"python"):
                raise ValueError("Python resolves the wrong SGLang checkout")
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", args.port))
            started = time.monotonic()
            with (support/"startup.log").open("w") as log:
                process = subprocess.Popen(plan["server_argv"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                server.write_json(support/"owned-server.json", {"pid": process.pid, "process_group": process.pid})
                server.wait_ready(process, base_url, args.startup_timeout)
                record["startup_seconds"] = time.monotonic()-started
                runtime = await build_manifest(base_url=base_url, research=profile["kind"] == "patched",
                                               source_checkout=profile["checkout"])
                server.write_json(support/"runtime-before.json", runtime)
                observed_hardware = check_common_runtime(runtime["server_info"], runtime["runtime"])
                if hardware is not None and hardware != observed_hardware:
                    raise ValueError("GPU/driver/runtime changed within this declared matrix")
                hardware = observed_hardware
                pools = server.pool_measurements(runtime["server_info"], (support/"startup.log").read_text(),
                    model_config, 16384, host_enabled=profile["hicache"])
                if profile["hicache"]:
                    if host_tokens is not None and pools["actual_host_pool_tokens"] != host_tokens:
                        raise ValueError("actual host token capacity differs across HiCache cells")
                    host_tokens = pools["actual_host_pool_tokens"]
                native_path = None
                if profile["kind"] == "patched":
                    scope = research.validate_runtime({"requested_gpu_tokens": 16384, "cost_methods_requested": False,
                        "server_environment": env}, runtime, expected_host_tokens=host_tokens)
                    metrics = await research_metrics(base_url, template.fingerprint)
                    server.write_json(support/"metrics-before.json", metrics)
                    if metrics["host_admission"]["config"]["mode"] != "off":
                        raise ValueError("host admission must actually be off")
                    if metrics["pool"]["unique_gpu_tokens"] or metrics["pool"]["unique_host_tokens"]:
                        raise ValueError("fresh research engine does not have empty initial KV")
                    record["scope"] = scope
                else:
                    native = server.native_manifest(runtime["server_info"], runtime["runtime"], pools, fingerprint,
                        profile["write_policy"], sources["stock"], 16384, baseline=job["native"]["baseline"])
                    native["server"].update(COMMON_SERVER)
                    native_path = support/"native-manifest.json"
                    server.write_json(native_path, native)
                record.update(status="running", pools=pools, hardware=hardware,
                    cache_state_protocol="fresh_process_empty_KV_no_generation_warmup",
                    jit_disk_cache_cleared=False, native_empty_state_basis="fresh owned process and skipped generation warmup")
                server.write_json(support/"server.json", record)
                result = await asyncio.wait_for(execute_job(spec, job, directory,
                    native_engine_manifest=native_path, verified_server=record), timeout=args.job_timeout)
                actual = json.loads(Path(result["raw_output"]).read_text())["report"]["turns"]
                if {r["task_id"] for r in actual} - {r["task_id"] for r in cohort}:
                    raise ValueError("job generated tasks outside its declared development cohort")
                record["status"] = "completed" if (len(actual) == 41
                    and {r["task_id"] for r in actual} == {r["task_id"] for r in cohort}
                    and all(r["status"] == "success" for r in actual)
                    and result["summary"]["cohort"]["unfinished_turns"] == 0) else "generation_failures"
                server.write_json(support/"runtime-after.json", await build_manifest(base_url=base_url,
                    research=profile["kind"] == "patched", source_checkout=profile["checkout"]))
                if profile["kind"] == "patched":
                    server.write_json(support/"metrics-after.json", await research_metrics(base_url, template.fingerprint))
                else:
                    (support/"metrics-after.txt").write_text(server.get_text(base_url, "/metrics"))
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if process is not None:
                server.stop_owned_server(process)
                record["server_returncode_after_cleanup"] = process.returncode
            server.write_json(support/"server.json", record)
        result["server_execution"] = {"record": str(support/"server.json"), **record}
        result["artifact_sha256"]["server_record"] = sha256((support/"server.json").read_bytes()).hexdigest()
        return result

    results = asyncio.run(run_matrix(plan["matrix"], args.output/"matrix-results", executor=owned_executor))
    server.write_json(args.output/"quality-evidence.template.json", quality_template(results, plan["matrix"]))
    complete = all(row["status"] == "completed" and row["summary"]["timing"]["failure_fraction"] == 0
                   and row["summary"]["cohort"]["unfinished_turns"] == 0
                   and row["server_execution"]["status"] == "completed" for row in results)
    server.write_json(args.output/"completion.json", {"status": "completed" if complete else "generation_failures",
        "cells": len(results), "quality_evaluated": False, "b6_selected": None,
        "freeze_requires_real_quality_evidence": True, "phase1_only": True})
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/hardware_dev_controller.yaml")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--patched-checkout", type=Path, default=ROOT/"vendor/sglang")
    parser.add_argument("--stock-checkout", type=Path, default=ROOT/"vendor/sglang-stock")
    parser.add_argument("--input", type=Path, default=ROOT/"data/mtrag/RAG.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=ROOT/"data/mtrag/split_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/b6-joint-dev-phase1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or min(args.startup_timeout, args.job_timeout) <= 0:
        parser.error("positive timeouts and a valid TCP port are required")
    for name in ("config", "model_path", "patched_checkout", "stock_checkout", "input", "split_manifest", "output"):
        setattr(args, name, getattr(args, name).resolve())
    plan = build_plan(args)
    if args.execute:
        return execute(args, plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
