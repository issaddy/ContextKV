"""Uniform experiment matrix, equal-budget development tuning and frozen test runs.

Jobs execute serially so one method does not compete with another for hardware.
Within each job, the existing session runner owns concurrency and P1/P2/P3 rules.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import time

import yaml

from adapters.mtrag import load_rows
from adapters.retriever_factory import retriever_config_sources
from bench.report import markdown_report, paired_latency, summarize_run
from controller.config import RunConfig, load_config
from controller.serialization import decode
from controller.templates import SYSTEM_PROMPT


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def matrix_config(spec):
    """One resolved prompt configuration shared by controller and native jobs."""
    config = load_config(spec.get("base_config"))
    text, filename = spec.get("system_prompt"), spec.get("system_prompt_file")
    if text is not None and filename is not None:
        raise ValueError("matrix system_prompt and system_prompt_file are mutually exclusive")
    if filename is not None:
        with Path(filename).open("r", encoding="utf-8", newline="") as stream:
            text = stream.read()
    if text is not None:
        config = replace(config, context=replace(config.context, system_prompt=text))
    return config


def validate_matrix(spec):
    if spec.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    if not spec.get("methods") or not set(spec.get("protocols", [])) <= {"P1", "P2", "P3"} or not spec.get("protocols"):
        raise ValueError("methods and valid protocols are required")
    if any(c < 1 for c in spec["concurrency"]) or not spec["concurrency"] or not spec["seeds"]:
        raise ValueError("nonempty positive concurrency and seeds are required")
    if "P3" in spec["protocols"] and not (spec.get("corpus") or spec.get("retriever_index") or spec.get("retriever_config")):
        raise ValueError("P3 requires corpus, retriever_index or retriever_config")
    if spec.get("retriever_config") and (spec.get("corpus") or spec.get("retriever_index")):
        raise ValueError("choose shared retriever_config or legacy corpus/retriever_index")
    names = [m["id"] for m in spec["methods"]]
    if len(names) != len(set(names)) or any(not name.replace("_", "").replace("-", "").isalnum() for name in names):
        raise ValueError("method IDs must be unique simple directory names")
    split = json.loads(Path(spec["split_manifest"]).read_text())
    dev, test = set(split["dev"]), set(split["test"])
    if not dev or not test or dev & test:
        raise ValueError("development and test must contain disjoint nonempty conversations")
    known = {r["conversation_id"] for r in load_rows(spec["input"])}
    if not (dev | test) <= known:
        raise ValueError("split manifest names conversations absent from the input")
    budget = spec.get("tuning", {}).get("equal_budget", 1)
    if budget < 1:
        raise ValueError("tuning budget must be positive")
    slo = spec.get("slo", {})
    if set(slo) - {"ttft_seconds", "tpot_seconds"} or any(not math.isfinite(v) or v < 0 for v in slo.values()):
        raise ValueError("matrix slo needs finite nonnegative ttft_seconds/tpot_seconds")
    methods_by_id = {m["id"]: m for m in spec["methods"]}
    shared_context = matrix_config(spec).context
    shared_context_values = asdict(shared_context)
    shared_prompt = shared_context.system_prompt
    effective_shared_prompt = SYSTEM_PROMPT if shared_prompt is None else shared_prompt
    for method in spec["methods"]:
        if method.get("kind", "controller") not in {"controller", "native"}:
            raise ValueError("method kind must be controller or native")
        candidates = method.get("candidates", [{}])
        for candidate in candidates:
            overrides = merge(method.get("overrides", {}), candidate)
            native = merge(method.get("native") or {}, overrides.get("native", {}))
            if any(key in source for source in (method, overrides, native)
                   for key in ("retriever_config", "retriever_index", "corpus")):
                raise ValueError("retriever configuration must be shared at the matrix root")
            if any(key in native for key in ("system_prompt", "system_prompt_file")):
                raise ValueError("matrix prompts must be shared at the matrix root or base_config.context")
            candidate_prompt = overrides.get("context", {}).get("system_prompt", shared_prompt)
            effective_candidate_prompt = SYSTEM_PROMPT if candidate_prompt is None else candidate_prompt
            if effective_candidate_prompt != effective_shared_prompt:
                raise ValueError("method-specific system_prompt differs from the shared matrix prompt")
            candidate_context = decode(type(shared_context), merge(shared_context_values, overrides.get("context", {})))
            # These fields determine the common material/budget contract. Layout
            # and annotations may vary for their explicit method/ablation studies.
            for field in ("revisit_mode", "revisit_history_turns", "allow_version_comparison",
                          "explicit_keys", "reference_prefix_length", "max_total_model_tokens", "max_new_tokens"):
                if getattr(candidate_context, field) != getattr(shared_context, field):
                    raise ValueError(f"method-specific context.{field} differs from the shared matrix policy")
            for key, field in (("context_length", "max_total_model_tokens"), ("max_new_tokens", "max_new_tokens")):
                if key in native and native[key] != getattr(shared_context, field):
                    raise ValueError(f"native {key} differs from the shared matrix budget")
            if "context_config" in native or "context_policy" in native:
                raise ValueError("native context policy must come from the shared matrix base_config")
        if method.get("tune", False) and len(candidates) != budget:
            raise ValueError(f"{method['id']} must have exactly equal_budget={budget} candidates")
        if not method.get("tune", False) and len(candidates) != 1:
            raise ValueError("multi-candidate methods must explicitly opt into equal-budget tuning")
        if method.get("ablation_of"):
            parent = methods_by_id.get(method["ablation_of"])
            if parent is None or parent is method or parent.get("ablation_of"):
                raise ValueError("ablation_of must name a primary method, not an ablation")
            if parent.get("kind", "controller") != "controller" or method.get("kind", "controller") != "controller":
                raise ValueError("matched ablations require a controller parent")
            if method.get("tune") or method.get("b6_eligible") or "candidates" in method:
                raise ValueError("ablations inherit the frozen parent candidate and do not participate in tuning/B6")
    lifecycle = spec.get("lifecycle", {})
    command = lifecycle.get("before_job")
    if command is not None and (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)):
        raise ValueError("lifecycle.before_job must be a nonempty argv list, never a shell string")
    if command is not None and not lifecycle.get("cache_state_protocol"):
        raise ValueError("before_job needs an explicit cache_state_protocol describing reset/warmup")
    return spec


def matrix_jobs(spec, *, split="dev", frozen=None, ablations_only=False):
    validate_matrix(spec)
    jobs = []
    methods = list(spec["methods"])
    if frozen is not None:
        by_id = {m["id"]: m for m in methods}
        if frozen.get("b6"):
            chosen = deepcopy(by_id[frozen["b6"]])
            chosen["id"], chosen["source_method"] = "B6", frozen["b6"]
            methods.append(chosen)
    for method in methods:
        source_id = method.get("source_method", method["id"])
        ablation_parent = method.get("ablation_of")
        if (ablation_parent and frozen is None) or (ablations_only and not ablation_parent):
            continue
        base_overrides = method.get("overrides", {})
        if ablation_parent:
            parent = next(m for m in spec["methods"] if m["id"] == ablation_parent)
            source_id = ablation_parent
            base_overrides = parent.get("overrides", {})
        if frozen is None:
            candidates = list(enumerate(method.get("candidates", [{}])))
        else:
            winner = frozen["winners"][source_id]
            candidates = [(winner["candidate_index"], winner["candidate_overrides"])]
        for (index, candidate), protocol, concurrency, seed in product(candidates, spec["protocols"], spec["concurrency"], spec["seeds"]):
            overrides = merge(base_overrides, candidate)
            if ablation_parent:
                overrides = merge(overrides, method.get("overrides", {}))
            native = merge(method.get("native") or {}, overrides.pop("native", {}))
            jobs.append({"job_id": f"{method['id']}-c{index}-{protocol}-n{concurrency}-s{seed}",
                         "method": method["id"], "source_method": source_id, "kind": method.get("kind", "controller"),
                         "split": split, "candidate_index": index, "candidate_overrides": candidate,
                         "overrides": overrides,
                         "native": native, "protocol": protocol,
                         "concurrency": concurrency, "seed": seed, "status": "not_run",
                         "ablation_of": ablation_parent})
    return jobs


def provenance(spec):
    config = matrix_config(spec)
    files = {spec.get("system_prompt_file"), spec.get("corpus"), spec.get("evidence_corpus"), spec.get("lifecycle", {}).get("warmup_input"), config.calibration_path,
             config.cost_model.artifact, config.cost_model.output_length_artifact}
    if spec.get("retriever_index"):
        files.add(str(Path(spec["retriever_index"]) / "manifest.json"))
    if spec.get("retriever_config"):
        files.update(retriever_config_sources(spec["retriever_config"]))
    for method in spec["methods"]:
        for candidate in method.get("candidates", [{}]):
            overrides = merge(method.get("overrides", {}), candidate)
            native = merge(method.get("native") or {}, overrides.get("native", {}))
            files.add(native.get("engine_manifest"))
            files.add(overrides.get("calibration_path"))
            files.add(overrides.get("cost_model", {}).get("artifact"))
            files.add(overrides.get("cost_model", {}).get("output_length_artifact"))
    return {"input_sha256": sha256(Path(spec["input"]).read_bytes()).hexdigest(),
            "split_sha256": sha256(Path(spec["split_manifest"]).read_bytes()).hexdigest(),
            # Config identity is a single experiment boundary check, never a request-path hash.
            "spec_sha256": sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "base_config_sha256": sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
            "external_inputs": {str(path): sha256(Path(path).read_bytes()).hexdigest() for path in sorted(p for p in files if p)}}


def generation_provenance(spec):
    """Offline quality eligibility can be added after generation; inputs cannot."""
    generation_spec = deepcopy(spec)
    generation_spec.get("tuning", {}).pop("quality_eligible", None)
    return provenance(generation_spec)


def select_development(spec, results):
    """All jobs of a candidate must complete; failures are not dropped from tuning.

    Each protocol/load/seed cell gets equal weight. Quality constraints, if used,
    must be precomputed offline and supplied as development eligibility, never
    inferred from mock answers or from held-out test scores.
    """
    expected = matrix_jobs(spec, split="dev")
    indexed = {r["job_id"]: r for r in results}
    if len(indexed) != len(results) or set(indexed) != {j["job_id"] for j in expected}:
        raise ValueError("development result coverage differs from the complete matrix")
    if any(r["split"] != "dev" for r in results):
        raise ValueError("test results cannot enter development selection")
    expected_provenance = generation_provenance(spec)
    if any(r.get("experiment_provenance") != expected_provenance for r in results):
        raise ValueError("development results belong to different or unrecorded inputs/configuration")
    grouped = {}
    for job in expected:
        row = indexed[job["job_id"]]
        grouped.setdefault((job["method"], job["candidate_index"]), []).append(row)
    winners = {}
    quality_gate = spec.get("tuning", {}).get("quality_eligible", {})
    for method in spec["methods"]:
        if method.get("ablation_of"):
            continue
        options = []
        for index, candidate in enumerate(method.get("candidates", [{}])):
            rows = grouped[(method["id"], index)]
            valid = all(r["status"] == "completed" and r["summary"]["timing"]["failure_fraction"] == 0
                        and r["summary"]["cohort"]["unfinished_turns"] == 0 for r in rows)
            eligible = quality_gate.get(method["id"], {}).get(str(index), False) if quality_gate else None
            if quality_gate and method.get("tune", False) and not eligible:
                valid = False
            if valid:
                objective = mean(r["summary"]["timing"]["turn_seconds"]["mean"] for r in rows)
                options.append((objective, index, candidate))
        if not options:
            raise ValueError(f"no fully completed eligible development candidate for {method['id']}")
        objective, index, candidate = min(options, key=lambda v: (v[0], v[1]))
        winners[method["id"]] = {"candidate_index": index, "candidate_overrides": candidate,
                                  "dev_mean_turn_seconds": objective,
                                  "quality_eligible": quality_gate.get(method["id"], {}).get(str(index), False) if quality_gate else None}
    simple = [m["id"] for m in spec["methods"] if m.get("b6_eligible", False)
              and (not quality_gate or winners[m["id"]]["quality_eligible"])]
    b6 = min(simple, key=lambda name: (winners[name]["dev_mean_turn_seconds"], name)) if simple else None
    simulation = any(r["summary"]["measurement_kind"] == "simulation" for r in results if r.get("summary"))
    return {"schema_version": 1, "selection_split": "dev", "provenance": provenance(spec),
            "winners": winners, "b6": b6, "objective": "equal_cell_weight_mean_complete_turn_seconds",
            "simulation_only": simulation, "quality_gate_applied": bool(quality_gate),
            "quality_noninferiority_established": False,
            "equal_budget": spec.get("tuning", {}).get("equal_budget", 1)}


async def execute_job(spec, job, directory):
    from controller.cli import benchmark
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    lifecycle = spec.get("lifecycle", {})
    hook = None
    if lifecycle.get("before_job"):
        started = time.monotonic()
        environment = {**os.environ, "CONTEXTKV_JOB_ID": job["job_id"],
                       "CONTEXTKV_JOB_SPLIT": job["split"], "CONTEXTKV_JOB_PROTOCOL": job["protocol"],
                       "CONTEXTKV_JOB_METHOD": job["method"]}
        command = lifecycle["before_job"]
        hook = {"argv": command, "cache_state_protocol": lifecycle["cache_state_protocol"]}
        try:
            completed = await asyncio.to_thread(subprocess.run, command, cwd=lifecycle.get("cwd"),
                                                env=environment, text=True, capture_output=True,
                                                timeout=lifecycle.get("timeout_seconds", 300))
            hook.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            hook.update(returncode=None, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            hook["elapsed_seconds"] = time.monotonic() - started
            (directory / "before-job.json").write_text(json.dumps(hook, indent=2))
        completed.check_returncode()
    args = dict(input=spec["input"], domain=spec.get("domain", "cloud"),
                split_manifest=spec["split_manifest"], split=job["split"], protocol=job["protocol"],
                corpus=spec.get("corpus"), retriever_index=spec.get("retriever_index"),
                retriever_config=spec.get("retriever_config"),
                evidence_corpus=spec.get("evidence_corpus"),
                concurrency=job["concurrency"], seed=job["seed"], limit_sessions=spec.get("limit_sessions"),
                think_time=spec.get("think_time", 0), arrival_rate=spec.get("arrival_rate"),
                arrival_window=spec.get("arrival_window"), drain_timeout=spec.get("drain_timeout"),
                per_turn_timeout=spec.get("per_turn_timeout"))
    shared_config = matrix_config(spec)
    if job["kind"] == "native":
        from bench.native_baseline import _run
        native = job["native"] or {}
        args.update(baseline=native["baseline"], base_url=native["base_url"], engine_manifest=native["engine_manifest"],
                    model_path=native.get("model_path"),
                    context_policy=shared_config.context,
                    system_prompt=shared_config.context.system_prompt, system_prompt_file=None,
                    table_relations=shared_config.context.table_relations,
                    table_annotation_style=shared_config.context.table_annotation_style,
                    table_structure_mode=shared_config.context.table_structure_mode,
                    output=str(directory), context_length=native.get("context_length", shared_config.context.max_total_model_tokens),
                    max_new_tokens=native.get("max_new_tokens", shared_config.context.max_new_tokens), local_files_only=native.get("local_files_only", False),
                    audit_prompt_ids=native.get("audit_prompt_ids", False), no_stream=native.get("no_stream", False))
        if job["overrides"]:
            raise ValueError("native engine/write policies require distinct preconfigured servers, not controller overrides")
        await _run(argparse.Namespace(**args))
        output = next(directory.glob("native-*.json"))
    else:
        base = asdict(shared_config)
        config = decode(RunConfig, merge(base, job["overrides"]))
        config = decode(RunConfig, {**asdict(config), "state_dir": str(directory)})
        await benchmark(argparse.Namespace(**args), config)
        output = next(directory.glob("benchmark-*.json"))
    payload = json.loads(output.read_text())
    default_cache_state = "fresh_mock_instance" if payload.get("engine_kind") == "mock" else "unspecified_hardware"
    protocol = {"cache_state_protocol": lifecycle.get("cache_state_protocol", default_cache_state),
                "before_job": hook, "warmup_input": lifecycle.get("warmup_input"),
                "fair_hardware_comparison_established": False}
    payload["experiment_protocol"] = protocol
    output.write_text(json.dumps(payload, indent=2))
    slo = spec.get("slo", {})
    summary = summarize_run(payload, **({"ttft_slo": slo["ttft_seconds"]} if "ttft_seconds" in slo else {}),
                            **({"tpot_slo": slo["tpot_seconds"]} if "tpot_seconds" in slo else {}))
    summary["experiment_protocol"] = protocol
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "summary.md").write_text(markdown_report(summary))
    return {**job, "status": "completed", "raw_output": str(output), "summary": summary}


async def run_matrix(spec, output, *, stage="dev", frozen=None, executor=execute_job):
    if stage not in {"dev", "test", "dev_ablation"}:
        raise ValueError("stage must be dev, test or dev_ablation")
    jobs = matrix_jobs(spec, split="dev" if stage == "dev_ablation" else stage,
                       frozen=frozen, ablations_only=stage == "dev_ablation")
    run_provenance = generation_provenance(spec)
    if stage in {"test", "dev_ablation"}:
        if frozen is None or frozen["provenance"] != provenance(spec):
            raise ValueError("test and matched dev ablations require a frozen development selection of this exact dataset/config/split")
        if frozen["simulation_only"]:
            hardware = any(j["kind"] == "native" for j in jobs) or load_config(spec.get("base_config")).engine.kind != "mock"
            hardware |= any(j["overrides"].get("engine", {}).get("kind") == "sglang" for j in jobs)
            if hardware:
                raise ValueError("simulation-only tuning cannot select configurations for a hardware test")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "results.jsonl"
    results = []
    with result_path.open("x") as stream:
        for job in jobs:
            try:
                result = await executor(spec, job, root / job["job_id"])
            except Exception as exc:
                result = {**job, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                hook_log = root / job["job_id"] / "before-job.json"
                if hook_log.exists():
                    result["before_job"] = json.loads(hook_log.read_text())
            result["experiment_provenance"] = run_provenance
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
    comparisons = []
    reference = spec.get("reference_method", "B3")
    for candidate in results:
        matches = [r for r in results if r["method"] == reference and r["status"] == "completed"
                   and all(r[k] == candidate[k] for k in ("protocol", "concurrency", "seed"))]
        if candidate["status"] == "completed" and candidate["method"] != reference and len(matches) == 1:
            try:
                comparison = paired_latency(candidate["summary"], matches[0]["summary"], samples=1000)
            except ValueError as exc:
                comparison = {"status": "not_comparable", "reason": str(exc)}
            comparisons.append({"candidate": candidate["job_id"], "reference": matches[0]["job_id"], **comparison})
    (root / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "tune", "freeze", "ablate-dev", "test"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen")
    parser.add_argument("--results", help="existing dev results.jsonl for offline quality-gated freeze")
    args = parser.parse_args(argv)
    spec = yaml.safe_load(Path(args.config).read_text())
    frozen = json.loads(Path(args.frozen).read_text()) if args.frozen else None
    if args.command == "plan":
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(matrix_jobs(spec, frozen=frozen), indent=2))
    elif args.command == "freeze":
        if not args.results:
            parser.error("freeze requires --results from the same generation configuration")
        results = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
        selected = select_development(spec, results)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            json.dump(selected, stream, indent=2)
    else:
        stage = {"tune": "dev", "test": "test", "ablate-dev": "dev_ablation"}[args.command]
        results = asyncio.run(run_matrix(spec, args.output, stage=stage, frozen=frozen))
        if args.command == "tune":
            selected = select_development(spec, results)
            with (Path(args.output) / "frozen.json").open("x") as stream:
                json.dump(selected, stream, indent=2)
        print(json.dumps({"jobs": len(results), "completed": sum(r["status"] == "completed" for r in results),
                          "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
