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
from urllib.parse import urlparse
import socket
import subprocess
import sys
import time

import yaml

from bench import controller_capacity_sweep as research
from bench import native_policy_sweep as server
from bench.experiments import (execute_job, run_matrix, matrix_jobs, matrix_config,
                               merge, provenance)
from controller.config import RunConfig, load_config
from controller.cost_model import CostModel
from controller.serialization import decode

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
    checkout = Path(profile["checkout"]).resolve()
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


def check_common_runtime(info, runtime, *, gpu_tokens=16384, seed=42):
    expected = {**COMMON_SERVER, "max_total_tokens": gpu_tokens, "random_seed": seed}
    mismatch = {k: {"expected": v, "observed": info.get(k)} for k, v in expected.items() if info.get(k) != v}
    if mismatch:
        raise ValueError(f"actual engine scheduling/runtime settings differ: {mismatch}")
    gpu = next(g for g in runtime["gpus"] if g["index"] == 0)
    if runtime["collection_errors"] or any(not runtime["driver_cuda_torch_versions"].get(k) for k in ("driver", "cuda", "torch")):
        raise ValueError("actual GPU/driver/CUDA/torch runtime could not be observed")
    return {"gpu_uuid": gpu["uuid"], "gpu_model": gpu["name"], "gpu_memory_bytes": gpu["memory_bytes"],
            "driver_cuda_torch_versions": runtime["driver_cuda_torch_versions"],
            "cache_config_hash_is_not_runtime_equivalence_proof": True}


def frozen_jobs(spec, stage, frozen):
    """The existing frozen contract is checked before any owned process can start."""
    if stage != "dev":
        if frozen is None or frozen["provenance"] != provenance(spec):
            raise ValueError("owned test/ablation requires the exact frozen development selection and implementation")
        if frozen["simulation_only"]:
            raise ValueError("simulation-only tuning cannot authorize an owned hardware run")
    elif frozen is not None:
        raise ValueError("dev does not accept --frozen; use test or dev_ablation")
    jobs = matrix_jobs(spec, split="test" if stage == "test" else "dev", frozen=frozen,
                       ablations_only=stage == "dev_ablation")
    if not jobs:
        raise ValueError("owned matrix has no declared jobs for this stage")
    return jobs


def local_endpoint(base_url):
    parsed = urlparse(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None or not 1 <= parsed.port <= 65535 or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError("owned matrix requires one explicit local HTTP loopback endpoint")
    return parsed.port


def selected_cohort(spec, stage):
    from adapters.mtrag import CLOUD_COLLECTION, load_rows
    selected = set(json.loads(Path(spec["split_manifest"]).read_text())["test" if stage == "test" else "dev"])
    rows = [row for row in load_rows(spec["input"]) if row["conversation_id"] in selected
            and (spec.get("domain", "cloud") != "cloud" or row["Collection"] == CLOUD_COLLECTION)]
    if spec.get("limit_sessions"):
        selected = set(sorted({row["conversation_id"] for row in rows})[:spec["limit_sessions"]])
        rows = [row for row in rows if row["conversation_id"] in selected]
    if not rows:
        raise ValueError("owned matrix cohort is empty")
    return rows


def job_requirements(spec, job, shared):
    """Resolve the real job once; B5 quotes do not imply a fitted time model."""
    config = decode(RunConfig, merge(asdict(shared), job["overrides"]))
    if (config.engine.kind != "sglang" or config.engine.base_url != shared.engine.base_url
            or config.model != shared.model or config.engine.page_size != 64
            or config.context.max_total_model_tokens != 32768
            or config.context.max_new_tokens != shared.context.max_new_tokens):
        raise ValueError("owned jobs must use the shared SGLang endpoint/model/context/output budget and page 64")
    profile = spec.get("server_profiles", {}).get(job.get("server_profile"))
    fields = {"kind", "write_policy", "hicache", "session_radix", "gpu_tokens", "hicache_gb", "checkout"}
    if profile is None or set(profile) != fields:
        raise ValueError("owned jobs require a declared supported server_profile")
    capacity = profile["gpu_tokens"]
    if type(capacity) is not int or capacity < 64 or capacity % 64 or config.engine.gpu_capacity_tokens != capacity:
        raise ValueError("profile GPU capacity must match the job config and be positive/page-aligned")
    expected_kind = "stock" if job["kind"] == "native" else "patched"
    baseline = job["native"].get("baseline") if job["kind"] == "native" else None
    host = baseline != "B0"
    if (profile["kind"] != expected_kind or profile["hicache"] is not host
            or profile["session_radix"] is not host or profile["hicache_gb"] != (16 if host else 0)
            or (profile["write_policy"] not in server.POLICIES if host else profile["write_policy"] is not None)):
        raise ValueError("owned profiles support patched HiCache or stock B0/B1, host 16GB and native write policies")
    if job["kind"] == "native":
        native = job["native"]
        if (baseline not in {"B0", "B1"} or native.get("base_url") != shared.engine.base_url
                or not native.get("model_path")
                or Path(native["model_path"]).resolve() != Path(shared.model.name).resolve()
                or not native.get("local_files_only") or not native.get("audit_prompt_ids")):
            raise ValueError("native owned job must use the shared local model/endpoint and retain prompt IDs")
        if native.get("engine_manifest"):
            raise ValueError("owned native manifests are generated after startup; do not supply a static manifest")
    needs_cost = job["kind"] == "controller" and config.planner.strategy == "cost"
    return {"strategy": config.planner.strategy, "shadow": config.shadow,
        "executed_strategy": config.planner.fallback_strategy if config.shadow else config.planner.strategy,
        "fallback_strategy": config.planner.fallback_strategy,
        "needs_cost": needs_cost, "needs_quote": job["kind"] == "controller" and config.planner.strategy in {"cost", "warm_idle"},
        "calibration": research.artifact_plan(config.cost_model.artifact, capacity) if needs_cost else None}


def matrix_requirements(spec, jobs, shared):
    requirements = {job["job_id"]: job_requirements(spec, job, shared) for job in jobs}
    profiles = [spec["server_profiles"][job["server_profile"]] for job in jobs]
    if len({profile["gpu_tokens"] for profile in profiles}) != 1:
        raise ValueError("all profiles in one owned matrix must share the same GPU token capacity")
    kinds = {}
    for profile in profiles:
        path = str(Path(profile["checkout"]).resolve())
        if path in kinds and kinds[path] != profile["kind"]:
            raise ValueError("one checkout cannot serve as both patched and clean stock source")
        kinds[path] = profile["kind"]
    return requirements


def build_matrix_plan(args):
    spec = yaml.safe_load(args.matrix.read_text())
    frozen = json.loads(args.frozen.read_text()) if args.frozen else None
    jobs = frozen_jobs(spec, args.stage, frozen)
    config = matrix_config(spec)
    if (config.model.revision != server.MODEL_REVISION or not config.model.local_files_only
            or config.engine.kind != "sglang" or spec.get("domain", "cloud") not in {"cloud", "all"}):
        raise ValueError("owned matrix requires the pinned local Qwen7B, SGLang and a declared cloud/all domain")
    if spec.get("arrival_rate") is not None and spec.get("arrival_window") is not None:
        raise ValueError("owned full-cohort mode does not support arrival_rate with an arrival_window that can omit sessions")
    port = local_endpoint(config.engine.base_url)
    requirements = matrix_requirements(spec, jobs, config)
    cohort = selected_cohort(spec, args.stage)
    return {"execute": False, "phase1_only": False, "matrix_path": str(args.matrix),
        "matrix": spec, "config": asdict(config), "stage": args.stage, "frozen": frozen,
        "jobs": jobs, "job_requirements": requirements, "port": port,
        "planned_cells": len(jobs), "planned_turns": len(jobs)*len(cohort),
        "selected_sessions": sorted({row["conversation_id"] for row in cohort}),
        "cohort_turns": len(cohort), "host_admission": "off", "normal_eos": True,
        "server_initial_state": "new owned process per job; disk/compiler caches and parent process can persist",
        "quality_evaluated": False, "b6_selected": None}


def job_server_command(port, capacity):
    command = server.server_command(argparse.Namespace(port=port, gpu_tokens=capacity)) + [
        "--attention-backend", "fa3", "--chunked-prefill-size", "8192", "--schedule-policy", "fcfs"]
    return command


def fitted_scope(config, calibration, runtime):
    options = {"max_new_tokens": config.context.max_new_tokens,
        "include_host_cost": config.ablation.host_cost, "include_decode_cost": config.ablation.decode_cost,
        "include_control_cost": config.ablation.control_cost}
    for field in ("admission_estimator", "uncertainty_mode"):
        if getattr(config.cost_model, field) is not None:
            options[field] = getattr(config.cost_model, field)
    model = CostModel.load(calibration["path"], **options)
    scope = model.check_calibration_scope(runtime["engine"])
    if scope["status"] != "matching_scope":
        raise ValueError(f"cost job needs an artifact bound to the actual engine configuration: {scope}")
    if (not any(p.source == "hardware" for p in model.prefill_points)
            or config.ablation.decode_cost and not any(p.source == "hardware" for p in model.decode_points)):
        raise ValueError("cost job requires actual hardware prefill/decode samples, not an empty bound artifact")
    return scope


async def research_metrics(base_url, fingerprint):
    from controller.engine_client import SGLangEngineClient
    engine = SGLangEngineClient(base_url, fingerprint)
    try:
        return await engine.metrics()
    finally:
        await engine.aclose()


def quality_template(results, spec, *, by_job=False):
    """References real artifacts, but supplies no evaluator identity or scores."""
    return {"scope": "dev", "reference": spec["tuning"]["quality_constraints"]["reference"], "epsilon": spec["tuning"]["quality_constraints"]["epsilon"],
        "split_manifest": spec["split_manifest"], "frozen_protocol": None,
        "methods": {row["job_id"] if by_job else row["method"]: {"run": row.get("raw_output"), "audit": row.get("audit_output"),
            "manifest": row.get("run_manifest"),
            "server_record": row.get("server_execution", {}).get("record"),
            "scores": None, "generator": {"model": "Qwen/Qwen2.5-7B-Instruct", "revision": server.MODEL_REVISION,
                                           "base_family": "Qwen2.5"}}
            for row in results},
        "template_only": True, "note": "fill an independent frozen evaluator protocol and full answer-bound five-dimension scores; never copy eligibility booleans"}


def write_block_quality_templates(results, spec, output):
    """Incomplete scoring templates retain every actual job, never pool repeats."""
    fields = ("protocol", "concurrency", "seed", "repetition")
    blocks = {}
    for row in results:
        key = tuple(row.get(field, 1) if field == "repetition" else row[field] for field in fields)
        blocks.setdefault(key, []).append(row)
    directory = output/"quality-templates"
    directory.mkdir()
    entries = []
    for index, (key, rows) in enumerate(blocks.items()):
        value = quality_template(rows, spec, by_job=True)
        references = [row["job_id"] for row in rows if row["method"] == value["reference"]]
        value["reference"] = references[0] if len(references) == 1 else None
        if len(references) != 1:
            value["note"] += "; no unique declared reference job: this template cannot authorize a freeze"
        path = directory/f"block-{index}.json"
        server.write_json(path, value)
        entries.append({**dict(zip(fields, key)), "quality_config": str(path)})
    server.write_json(output/"quality-evidence.template.json", {"schema_version": 2, "blocks": entries})


def make_owned_executor(args, shared, sources, cohort, requirements):
    from profiling.runtime_manifest import build_manifest
    from controller.templates import HuggingFaceTemplate
    template = HuggingFaceTemplate(shared.model.name, shared.model.revision,
        namespace_hash=shared.model.namespace_hash, local_files_only=True)
    model_config = json.loads((Path(shared.model.name)/"config.json").read_text())
    fingerprint = asdict(template.fingerprint)
    hardware, host_tokens = None, None

    async def owned_executor(spec, job, directory):
        nonlocal hardware, host_tokens
        profile = spec["server_profiles"][job["server_profile"]]
        support = args.output/"servers"/job["job_id"]
        support.mkdir(parents=True)
        env = server_environment(profile, args, fingerprint)
        capacity = profile["gpu_tokens"]
        command = job_server_command(args.port, capacity)
        config = decode(RunConfig, merge(asdict(shared), job["overrides"]))
        requirement = requirements[job["job_id"]]
        process, result = None, None
        record = {"job_id": job["job_id"], "server_profile": job["server_profile"], "profile": profile,
                  "status": "starting", "argv": command, "requirements": requirement,
                  "source_checkout": sources[str(Path(profile["checkout"]).resolve())]}
        base_url = shared.engine.base_url.rstrip("/")
        try:
            origin = server.command([sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('sglang').origin)"], env=env)
            if not Path(origin).resolve().is_relative_to(Path(profile["checkout"]).resolve()/"python"):
                raise ValueError("Python resolves the wrong SGLang checkout")
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", args.port))
            started = time.monotonic()
            with (support/"startup.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                server.write_json(support/"owned-server.json", {"pid": process.pid, "process_group": process.pid})
                server.wait_ready(process, base_url, args.startup_timeout)
                record["startup_seconds"] = time.monotonic()-started
                runtime = await build_manifest(base_url=base_url, research=profile["kind"] == "patched",
                                               source_checkout=profile["checkout"])
                server.write_json(support/"runtime-before.json", runtime)
                observed_hardware = check_common_runtime(runtime["server_info"], runtime["runtime"], gpu_tokens=capacity)
                expected = {"model_path": str(Path(shared.model.name).resolve()), "revision": shared.model.revision,
                    "tokenizer_path": str(Path(shared.model.name).resolve()), "dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
                    "enable_session_radix_cache": profile["session_radix"], "enable_hierarchical_cache": profile["hicache"]}
                if any(runtime["server_info"].get(k) != v for k, v in expected.items()):
                    raise ValueError("actual owned model/tokenizer/dtype/cache flags differ from the shared job configuration")
                if hardware is not None and hardware != observed_hardware:
                    raise ValueError("GPU/driver/runtime changed within this declared matrix")
                hardware = observed_hardware
                pools = server.pool_measurements(runtime["server_info"], (support/"startup.log").read_text(),
                    model_config, capacity, host_enabled=profile["hicache"])
                if profile["hicache"]:
                    if host_tokens is not None and pools["actual_host_pool_tokens"] != host_tokens:
                        raise ValueError("actual host token capacity differs across HiCache cells")
                    host_tokens = pools["actual_host_pool_tokens"]
                native_path = None
                if profile["kind"] == "patched":
                    scope = research.validate_runtime({"requested_gpu_tokens": capacity, "cost_methods_requested": False,
                        "server_environment": env}, runtime, expected_host_tokens=host_tokens)
                    if runtime["engine"]["fingerprint"] != fingerprint:
                        raise ValueError("actual research tokenizer/template namespace differs from the shared fingerprint")
                    if requirement["needs_cost"]:
                        scope["calibration_scope"] = fitted_scope(config, requirement["calibration"], runtime)
                    metrics = await research_metrics(base_url, template.fingerprint)
                    server.write_json(support/"metrics-before.json", metrics)
                    if metrics["host_admission"]["config"]["mode"] != "off":
                        raise ValueError("host admission must actually be off")
                    if metrics["pool"]["unique_gpu_tokens"] or metrics["pool"]["unique_host_tokens"]:
                        raise ValueError("fresh research engine does not have empty initial KV")
                    record["scope"] = scope
                else:
                    native = server.native_manifest(runtime["server_info"], runtime["runtime"], pools, fingerprint,
                        profile["write_policy"], sources[str(Path(profile["checkout"]).resolve())], capacity, baseline=job["native"]["baseline"])
                    native["server"].update({**COMMON_SERVER, "max_total_tokens": capacity})
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
                    raise ValueError("job generated tasks outside its declared cohort")
                record["status"] = "completed" if (len(actual) == len(cohort)
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
        result.setdefault("artifact_sha256", {})["server_record"] = sha256((support/"server.json").read_bytes()).hexdigest()
        return result

    return owned_executor


def execute(args, plan):
    phase1 = plan.get("phase1_only", True)
    stage, frozen = plan.get("stage", "dev"), plan.get("frozen")
    spec, shared = plan["matrix"], decode(RunConfig, plan["config"])
    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("launch the parent command with RAYON_NUM_THREADS=4; server-only settings do not configure the client")
    # Phase1 writes its own new declarations; an existing matrix is never rewritten.
    if phase1:
        cohort = selected_cohort(spec, "dev")
        if len(cohort) != 41 or len({row["conversation_id"] for row in cohort}) != 5:
            raise ValueError("phase1 requires the complete five-session / 41-turn Cloud dev cohort")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/"controller.yaml").write_text(yaml.safe_dump(plan["config"], sort_keys=False))
        (args.output/"matrix.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    jobs = frozen_jobs(spec, stage, frozen)
    requirements = matrix_requirements(spec, jobs, shared)
    for need in requirements.values():
        if need["needs_cost"] and need["calibration"]["status"] == "requires_calibration":
            raise ValueError(f"cost job needs its own bound artifact before startup: {need['calibration']}")
    if not phase1:
        cohort = selected_cohort(spec, stage)
        args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for job in jobs:
        profile = spec["server_profiles"][job["server_profile"]]
        path = Path(profile["checkout"]).resolve()
        if str(path) not in sources:
            sources[str(path)] = (research.check_research_checkout(path) if profile["kind"] == "patched" else server.check_stock(path))
    server.write_json(args.output/"plan.json", {**plan, "execute": True, "source_checkouts": sources})
    executor = make_owned_executor(args, shared, sources, cohort, requirements)
    results = asyncio.run(run_matrix(spec, args.output/"matrix-results", stage=stage, frozen=frozen, executor=executor))
    if phase1:
        server.write_json(args.output/"quality-evidence.template.json", quality_template(results, spec))
    elif stage == "dev" and spec.get("tuning", {}).get("require_quality_evidence"):
        write_block_quality_templates(results, spec, args.output)
    complete = all(row["status"] == "completed" and row["summary"]["timing"]["failure_fraction"] == 0
                   and row["summary"]["cohort"]["unfinished_turns"] == 0
                   and row["server_execution"]["status"] == "completed" for row in results)
    server.write_json(args.output/"completion.json", {"status": "completed" if complete else "generation_failures",
        "cells": len(results), "expected_turns_per_job": len(cohort), "stage": stage,
        "quality_evaluated": False, "b6_selected": None,
        "freeze_requires_real_quality_evidence": True, "phase1_only": phase1,
        "quality_template_scope": "dev_only; test/ablation require their real matched scoring sources"})
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, help="existing declared matrix; never rewritten")
    parser.add_argument("--stage", choices=("dev", "test", "dev_ablation"), default="dev")
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--patched-checkout", type=Path)
    parser.add_argument("--stock-checkout", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if min(args.startup_timeout, args.job_timeout) <= 0:
        parser.error("positive timeouts are required")
    if args.matrix and args.output is None:
        parser.error("--matrix requires a separate new --output directory")
    args.output = (args.output or ROOT/"runs/b6-joint-dev-phase1").resolve()
    if args.matrix:
        if any(getattr(args, name) is not None for name in ("config", "model_path", "patched_checkout", "stock_checkout", "input", "split_manifest", "port")):
            parser.error("--matrix uses its existing config/profile paths; phase1 overrides cannot be combined")
        args.matrix = args.matrix.resolve()
        if args.frozen:
            args.frozen = args.frozen.resolve()
        plan = build_matrix_plan(args)
        args.port = plan["port"]
        args.model_path = Path(plan["config"]["model"]["name"]).resolve()
    else:
        if args.stage != "dev" or args.frozen:
            parser.error("test/dev_ablation/frozen require --matrix")
        if args.model_path is None:
            parser.error("phase1 requires --model-path")
        defaults = {"config": ROOT/"configs/hardware_dev_controller.yaml", "patched_checkout": ROOT/"vendor/sglang",
            "stock_checkout": ROOT/"vendor/sglang-stock", "input": ROOT/"data/mtrag/RAG.jsonl",
            "split_manifest": ROOT/"data/mtrag/split_manifest.json"}
        for name, default in defaults.items():
            setattr(args, name, (getattr(args, name) or default).resolve())
        args.model_path = args.model_path.resolve()
        args.port = 30000 if args.port is None else args.port
        if not 1 <= args.port <= 65535:
            parser.error("a valid TCP port is required")
        plan = build_plan(args)
    if args.execute:
        return execute(args, plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
