"""Run explicitly declared Qwen3 matrices with one owned server per job.

Plan by default. Saved independent quality evidence is required for B6 freeze.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
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
                               merge)
from controller.config import RunConfig, create_model_template, load_cost_model
from controller.serialization import decode
from controller.sampling import SamplingRecipe, request_sampling
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMMON_SERVER = {"attention_backend": "fa3", "chunked_prefill_size": 8192,
    "schedule_policy": "fcfs", "max_running_requests": 8, "page_size": 64,
    "context_length": 32768, "max_total_tokens": 16384, "random_seed": 42,
    "skip_server_warmup": True, "tp_size": 1, "dp_size": 1}


def server_environment(profile, args, fingerprint, *, model_scope):
    if model_scope is None:
        raise ValueError("owned generation requires an explicit Qwen3 model scope")
    checkout = Path(profile["checkout"]).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", SGLANG_CHECKOUT=str(checkout),
        CONTEXTKV_MODEL_PATH=str(args.model_path), CONTEXTKV_FINGERPRINT_JSON=json.dumps(fingerprint),
        CONTEXTKV_ENABLE_SESSION_RADIX="1" if profile["session_radix"] else "0",
        CONTEXTKV_ENABLE_HICACHE="1" if profile["hicache"] else "0",
        CONTEXTKV_HICACHE_GB=str(profile["hicache_gb"]), CONTEXTKV_WRITE_POLICY=profile["write_policy"] or "write_through",
        CONTEXTKV_HICACHE_LAYOUT="layer_first", CONTEXTKV_HICACHE_IO_BACKEND="kernel",
        CONTEXTKV_HOST_ADMISSION="off", CONTEXTKV_PAGE_SIZE="64", SGLANG_HICACHE_TMA_TRANSFER="0",
        CONTEXTKV_SAMPLE_POOL="1", CONTEXTKV_ENABLE_PROFILING="1" if profile["kind"] == "patched" else "0")
    # Do not inherit a different model identity from an operator shell. The
    # explicit scope owns model/tokenizer/revision through the existing launcher.
    env["CONTEXTKV_MODEL_SCOPE_PATH"] = model_scope.source_path
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
    from bench.experiments import needs_development_selection
    if needs_development_selection(spec, stage, frozen):
        if frozen is None:
            raise ValueError("owned test requires a frozen development selection")
    elif stage == "dev" and frozen is not None:
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
            or (config.sampling or SamplingRecipe()) != (shared.sampling or SamplingRecipe())
            or config.context.max_total_model_tokens != 32768
            or config.context.max_new_tokens != shared.context.max_new_tokens):
        raise ValueError("owned jobs must use the shared SGLang endpoint/model/context/output budget and page 64")
    profile = spec.get("server_profiles", {}).get(job.get("server_profile"))
    fields = {"kind", "write_policy", "hicache", "session_radix", "gpu_tokens", "hicache_gb", "checkout", "mem_fraction_static"}
    if profile is None or set(profile) != fields:
        raise ValueError("owned jobs require a declared supported server_profile")
    fraction = profile["mem_fraction_static"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("owned scope requires an explicit finite mem_fraction_static in (0,1)")
    capacity = profile["gpu_tokens"]
    if type(capacity) is not int or capacity < 64 or capacity % 64 or config.engine.gpu_capacity_tokens != capacity:
        raise ValueError("profile GPU capacity must match the job config and be positive/page-aligned")
    expected_kind = "stock" if job["kind"] == "native" else "patched"
    baseline = job["native"].get("baseline") if job["kind"] == "native" else None
    host = baseline != "B0"
    if (profile["kind"] != expected_kind or profile["hicache"] is not host
            or profile["session_radix"] is not host
            or type(profile["hicache_gb"]) not in (int, float) or not math.isfinite(profile["hicache_gb"])
            or (profile["hicache_gb"] <= 0 if host else profile["hicache_gb"] != 0)
            or (profile["write_policy"] not in server.POLICIES if host else profile["write_policy"] is not None)):
        raise ValueError("owned profiles support patched HiCache or stock B0/B1, explicit positive host GB (B0 zero) and native write policies")
    if job["kind"] == "native":
        native = job["native"]
        if (baseline not in {"B0", "B1"} or native.get("base_url") != shared.engine.base_url
                or (native.get("model_path") is not None
                    and Path(native["model_path"]).resolve() != Path(shared.model.name).resolve())
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
    if (len({profile["mem_fraction_static"] for profile in profiles}) != 1
            or len({profile["hicache_gb"] for profile in profiles if profile["hicache"]}) > 1):
        raise ValueError("all profiles must share mem_fraction_static and HiCache GB (B0 host zero is the exception)")
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
    if (config.model.scope is None or not config.model.local_files_only
            or config.engine.kind != "sglang" or spec.get("domain", "cloud") not in {"cloud", "all"}):
        raise ValueError("owned matrix requires an explicit local Qwen3 scope, SGLang and a declared cloud/all domain")
    if spec.get("arrival_rate") is not None and spec.get("arrival_window") is not None:
        raise ValueError("owned full-cohort mode does not support arrival_rate with an arrival_window that can omit sessions")
    port = local_endpoint(config.engine.base_url)
    requirements = matrix_requirements(spec, jobs, config)
    cohort = selected_cohort(spec, args.stage)
    return {"execute": False, "matrix_path": str(args.matrix),
        "matrix": spec, "config": asdict(config), "stage": args.stage, "frozen": frozen,
        "jobs": jobs, "job_requirements": requirements, "port": port,
        "planned_cells": len(jobs), "planned_turns": len(jobs)*len(cohort),
        "selected_sessions": sorted({row["conversation_id"] for row in cohort}),
        "cohort_turns": len(cohort), "host_admission": "off", "normal_eos": True,
        "server_initial_state": "new owned process per job; disk/compiler caches and parent process can persist",
        "quality_evaluated": False, "b6_selected": None}


def job_server_command(port, capacity, *, mem_fraction_static):
    command = server.server_command(argparse.Namespace(port=port, gpu_tokens=capacity),
                                    mem_fraction_static=mem_fraction_static) + [
        "--attention-backend", "fa3", "--chunked-prefill-size", "8192", "--schedule-policy", "fcfs"]
    return command


def fitted_scope(config, calibration, runtime):
    model = load_cost_model(config, artifact_path=calibration["path"])
    scope = model.check_calibration_scope(runtime["engine"])
    if scope["status"] != "matching_scope" or not model.sampling_scope_compatible:
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
    model = matrix_config(spec).model
    if model.scope is None:
        raise ValueError("owned quality templates require the actual explicit Qwen3 scope")
    generator = {"model": model.scope.model_path, "revision": model.scope.identity, "base_family": "Qwen3",
                 "identity_kind": model.scope.manifest()["record"]["identity_kind"],
                 "hf_revision": None, "weight_content_verified": False}
    return {"scope": "dev", "reference": spec["tuning"]["quality_constraints"]["reference"], "epsilon": spec["tuning"]["quality_constraints"]["epsilon"],
        "split_manifest": spec["split_manifest"], "frozen_protocol": None,
        "methods": {row["job_id"] if by_job else row["method"]: {"run": row.get("raw_output"), "audit": row.get("audit_output"),
            "manifest": row.get("run_manifest"),
            "server_record": row.get("server_execution", {}).get("record"),
            "scores": None, "generator": dict(generator)}
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
    template = create_model_template(shared.model)
    model_config = json.loads((Path(shared.model.name)/"config.json").read_text())
    fingerprint = asdict(template.fingerprint)
    hardware, host_tokens = None, None

    async def owned_executor(spec, job, directory):
        nonlocal hardware, host_tokens
        profile = spec["server_profiles"][job["server_profile"]]
        support = args.output/"servers"/job["job_id"]
        support.mkdir(parents=True)
        env = server_environment(profile, args, fingerprint, model_scope=shared.model.scope)
        capacity = profile["gpu_tokens"]
        command = job_server_command(args.port, capacity, mem_fraction_static=profile["mem_fraction_static"])
        config = decode(RunConfig, merge(asdict(shared), job["overrides"]))
        requirement = requirements[job["job_id"]]
        process, result = None, None
        record = {"job_id": job["job_id"], "server_profile": job["server_profile"], "profile": profile,
                  "status": "starting", "argv": command, "requirements": requirement,
                  "source_checkout": sources[str(Path(profile["checkout"]).resolve())]}
        record["model_scope"] = shared.model.scope.manifest()
        record["sampling_recipe"] = request_sampling(shared.sampling)
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
                expected = {"model_path": str(Path(shared.model.name).resolve()), "revision": None,
                    "tokenizer_path": str(Path(shared.model.name).resolve()), "dtype": "bfloat16", "kv_cache_dtype": "bfloat16",
                    "enable_session_radix_cache": profile["session_radix"], "enable_hierarchical_cache": profile["hicache"]}
                if any(runtime["server_info"].get(k) != v for k, v in expected.items()):
                    raise ValueError("actual owned model/tokenizer/dtype/cache flags differ from the shared job configuration")
                if runtime["server_info"].get("mem_fraction_static") != profile["mem_fraction_static"]:
                    raise ValueError("actual memory fraction differs from the declared shared resource")
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
                    shared.model.scope.verify_native_settings(SimpleNamespace(**runtime["server_info"]))
                    if runtime["server_info"].get("pp_size") != 1:
                        raise ValueError("owned local scope requires PP1")
                    scope = research.validate_runtime(capacity, runtime, write_policy=profile["write_policy"],
                        expected_host_tokens=host_tokens,
                        expected_hicache_gb=profile["hicache_gb"])
                    if runtime["engine"]["fingerprint"] != fingerprint:
                        raise ValueError("actual research tokenizer/template namespace differs from the shared fingerprint")
                    actual_scope = runtime["engine"].get("model_scope") or {}
                    if (actual_scope.get("identity") != shared.model.scope.identity
                            or actual_scope.get("record", {}).get("model_path") != shared.model.scope.model_path):
                        raise ValueError("actual research model scope differs from the shared local identity/path")
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
                        profile["write_policy"], sources[str(Path(profile["checkout"]).resolve())], capacity, baseline=job["native"]["baseline"],
                        model_scope=shared.model.scope, sampling=shared.sampling, hicache_gb=profile["hicache_gb"])
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
        return result

    return owned_executor


def execute(args, plan):
    stage, frozen = plan.get("stage", "dev"), plan.get("frozen")
    spec, shared = plan["matrix"], decode(RunConfig, plan["config"])
    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("launch the parent command with RAYON_NUM_THREADS=4; server-only settings do not configure the client")
    jobs = frozen_jobs(spec, stage, frozen)
    requirements = matrix_requirements(spec, jobs, shared)
    for need in requirements.values():
        if need["needs_cost"] and need["calibration"]["status"] == "requires_calibration":
            raise ValueError(f"cost job needs its own bound artifact before startup: {need['calibration']}")
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
    if stage == "dev" and spec.get("tuning", {}).get("require_quality_evidence"):
        write_block_quality_templates(results, spec, args.output)
    complete = all(row["status"] == "completed" and row["summary"]["timing"]["failure_fraction"] == 0
                   and row["summary"]["cohort"]["unfinished_turns"] == 0
                   and row["server_execution"]["status"] == "completed" for row in results)
    server.write_json(args.output/"completion.json", {"status": "completed" if complete else "generation_failures",
        "cells": len(results), "expected_turns_per_job": len(cohort), "stage": stage,
        "quality_evaluated": False, "b6_selected": None,
        "quality_template_scope": "dev_only; test/ablation require their real matched scoring sources"})
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="explicit Qwen3 matrix; never rewritten")
    parser.add_argument("--stage", choices=("dev", "test", "dev_ablation"), default="dev")
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if min(args.startup_timeout, args.job_timeout) <= 0:
        parser.error("positive timeouts are required")
    args.output = args.output.resolve()
    args.matrix = args.matrix.resolve()
    if args.frozen:
        args.frozen = args.frozen.resolve()
    plan = build_matrix_plan(args)
    args.port = plan["port"]
    args.model_path = Path(plan["config"]["model"]["name"]).resolve()
    if args.execute:
        return execute(args, plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
