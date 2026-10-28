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
from bench.report import METRIC_SCHEMA_VERSION, markdown_report, paired_latency, summarize_run
from bench.official import remap
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


def retrieval_settings(spec, method=None):
    """Resolve the declared retrieval recipe for this physical execution."""
    if method and method.get("retriever_config"):
        return {"corpus": None, "retriever_index": None, "retriever_config": method["retriever_config"]}
    return {key: spec.get(key) for key in ("corpus", "retriever_index", "retriever_config")}


def needs_development_selection(spec, stage, frozen):
    """Predeclared fixed methods need no artificial tuning run before a test."""
    return stage == "dev_ablation" or (stage == "test" and (frozen is not None
        or any(m.get("tune") or m.get("ablation_of") for m in spec["methods"])))


def validate_matrix(spec, *, required_split="dev"):
    if spec.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    if not spec.get("methods") or not set(spec.get("protocols", [])) <= {"P1", "P2", "P3"} or not spec.get("protocols"):
        raise ValueError("methods and valid protocols are required")
    if any(c < 1 for c in spec["concurrency"]) or not spec["concurrency"] or not spec["seeds"]:
        raise ValueError("nonempty positive concurrency and seeds are required")
    if "repetitions" in spec:
        repetitions = spec["repetitions"]
        if (not isinstance(repetitions, list) or not repetitions
                or any(type(r) is not int or r < 1 for r in repetitions)
                or len(set(repetitions)) != len(repetitions)):
            raise ValueError("repetitions must be distinct positive integer run IDs, separate from seeds")
    scope = spec.get("comparison_scope", "execution")
    if scope not in {"execution", "system"}:
        raise ValueError("comparison_scope must be execution or system")
    if "P3" in spec["protocols"] and any(not any(retrieval_settings(spec, method).values()) for method in spec["methods"]):
        raise ValueError("P3 requires corpus, retriever_index or retriever_config")
    if spec.get("retriever_config") and (spec.get("corpus") or spec.get("retriever_index")):
        raise ValueError("choose shared retriever_config or legacy corpus/retriever_index")
    names = [m["id"] for m in spec["methods"]]
    if len(names) != len(set(names)) or any(not name.replace("_", "").replace("-", "").isalnum() for name in names):
        raise ValueError("method IDs must be unique simple directory names")
    split = json.loads(Path(spec["split_manifest"]).read_text())
    dev, test = set(split["dev"]), set(split["test"])
    if dev & test:
        raise ValueError("development and test conversations must be disjoint")
    if not split[required_split]:
        raise ValueError(f"requested {required_split} split contains no conversations")
    known = {r["conversation_id"] for r in load_rows(spec["input"])}
    if not (dev | test) <= known:
        raise ValueError("split manifest names conversations absent from the input")
    budget = spec.get("tuning", {}).get("equal_budget", 1)
    if budget < 1:
        raise ValueError("tuning budget must be positive")
    slo = spec.get("slo", {})
    if set(slo) - {"ttft_seconds", "tpot_seconds", "complete_seconds"} or any(not math.isfinite(v) or v < 0 for v in slo.values()):
        raise ValueError("matrix slo needs finite nonnegative ttft_seconds/tpot_seconds/complete_seconds")
    if spec.get("response_mode", "buffered") not in {"buffered", "streaming"}:
        raise ValueError("matrix response_mode must be buffered or streaming")
    methods_by_id = {m["id"]: m for m in spec["methods"]}
    if spec.get("tuning", {}).get("require_quality_evidence"):
        constraints = spec["tuning"].get("quality_constraints", {})
        epsilon = constraints.get("epsilon")
        if (set(constraints) != {"reference", "epsilon"} or constraints["reference"] not in methods_by_id
                or isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or not 0 < epsilon < 1):
            raise ValueError("required quality evidence needs a declared reference and finite epsilon in (0,1)")
    shared_config = matrix_config(spec)
    from controller.sampling import SamplingRecipe
    from controller.config import ModelConfig
    native_present = any(m.get("kind", "controller") == "native" for m in spec["methods"])
    if native_present and shared_config.model.scope is None:
        raise ValueError("native matrices require a shared explicit Qwen3 model.scope_path")
    shared_context = shared_config.context
    shared_context_values = asdict(shared_context)
    shared_prompt = shared_context.system_prompt
    effective_shared_prompt = SYSTEM_PROMPT if shared_prompt is None else shared_prompt
    for method in spec["methods"]:
        if method.get("retriever_config") and scope != "system":
            raise ValueError("method retriever_config requires comparison_scope=system; execution comparisons share retrieval")
        if method.get("kind", "controller") not in {"controller", "native"}:
            raise ValueError("method kind must be controller or native")
        if method.get("server_profile") is not None and method["server_profile"] not in spec.get("server_profiles", {}):
            raise ValueError("method server_profile must name a declared owned-server profile")
        candidates = method.get("candidates", [{}])
        for candidate in candidates:
            overrides = merge(method.get("overrides", {}), candidate)
            if native_present and "sampling" in overrides:
                actual = merge(asdict(shared_config), overrides)
                requested = decode(SamplingRecipe | None, actual["sampling"])
                if (requested or SamplingRecipe()) != (shared_config.sampling or SamplingRecipe()):
                    raise ValueError("method sampling differs from the shared native/controller recipe")
            if native_present and "model" in overrides:
                requested_model = decode(ModelConfig, merge(asdict(shared_config.model), overrides["model"]))
                if requested_model != shared_config.model:
                    raise ValueError("method model differs from the shared native/controller model")
            native = merge(method.get("native") or {}, overrides.get("native", {}))
            if any(key in native for key in ("model_scope", "model_config", "sampling", "sampling_params", "run_config", "revision")):
                raise ValueError("native model and sampling must come from the shared matrix base_config")
            if method.get("kind") == "native" and shared_config.model.scope is not None:
                if (native.get("model_path") is not None
                        and Path(native["model_path"]).resolve() != Path(shared_config.model.name)):
                    raise ValueError("native model_path differs from the shared local model scope")
                if native.get("local_files_only") is False:
                    raise ValueError("local model scope requires local_files_only")
            if (any(key in source for source in (overrides, native)
                    for key in ("retriever_config", "retriever_index", "corpus"))
                    or any(key in method for key in ("retriever_index", "corpus"))):
                raise ValueError("retriever configuration belongs at the matrix root or method.retriever_config for system comparison")
            if any(key in native for key in ("system_prompt", "system_prompt_file")):
                raise ValueError("matrix prompts must be shared at the matrix root or base_config.context")
            candidate_prompt = overrides.get("context", {}).get("system_prompt", shared_prompt)
            effective_candidate_prompt = SYSTEM_PROMPT if candidate_prompt is None else candidate_prompt
            if effective_candidate_prompt != effective_shared_prompt:
                raise ValueError("method-specific system_prompt differs from the shared matrix prompt")
            candidate_context = decode(type(shared_context), merge(shared_context_values, overrides.get("context", {})))
            # These fields determine the common material/budget contract. Layout
            # and annotations may vary for their explicit method/ablation studies.
            for field in ("allow_version_comparison",
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
            if ("server_profile" in method and method["server_profile"] != parent.get("server_profile")):
                raise ValueError("matched ablations must inherit the parent server_profile")
            if method.get("tune") or method.get("b6_eligible") or "candidates" in method:
                raise ValueError("ablations inherit the frozen parent candidate and do not participate in tuning/B6")
            if method.get("retriever_config") and method["retriever_config"] != parent.get("retriever_config"):
                raise ValueError("execution ablations inherit the parent's retriever_config")
    lifecycle = spec.get("lifecycle", {})
    command = lifecycle.get("before_job")
    if command is not None and (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)):
        raise ValueError("lifecycle.before_job must be a nonempty argv list, never a shell string")
    if command is not None and not lifecycle.get("cache_state_protocol"):
        raise ValueError("before_job needs an explicit cache_state_protocol describing reset/warmup")
    return spec


def matrix_jobs(spec, *, split="dev", frozen=None, ablations_only=False):
    validate_matrix(spec, required_split=split)
    jobs = []
    methods = list(spec["methods"])
    b6 = frozen.get("b6") if frozen is not None else None
    if b6 and b6 not in {m["id"] for m in methods}:
        raise ValueError("frozen B6 must identify a declared method")
    if b6 and b6 != "B6" and any(m["id"] == "B6" for m in methods):
        raise ValueError("B6 alias conflicts with a separately declared method")
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
                         "retrieval": retrieval_settings(spec, parent if ablation_parent else method),
                         "native": native, "protocol": protocol,
                         "concurrency": concurrency, "seed": seed, "status": "not_run",
                         "ablation_of": ablation_parent})
            if method["id"] == b6 and b6 != "B6":
                # The selected baseline is one physical run, not a second
                # method or repetition under a publication label.
                jobs[-1]["method_aliases"] = ["B6"]
            profile = parent.get("server_profile") if ablation_parent else method.get("server_profile")
            if profile is not None:
                jobs[-1]["server_profile"] = profile
    if "repetitions" in spec:
        jobs = [{**deepcopy(job), "job_id": f"{job['job_id']}-r{repetition}", "repetition": repetition}
                for repetition in spec["repetitions"] for job in jobs]
    return jobs


def implementation_manifest():
    """Snapshot local execution sources once at a matrix/selection boundary.

    Deliberately excludes Git HEAD, docs and tests. This is not attestation of
    the remote engine or model files, whose existing manifests remain separate.
    """
    root = Path(__file__).resolve().parents[1]
    paths = {p for folder in ("controller", "adapters", "bench", "profiling", "engine_patches", "scripts")
             for p in (root / folder).rglob("*.py")}
    paths.add(root / "scripts/launch_sglang.sh")
    paths.update(p for p in (root / "adapters/assets").rglob("*")
                 if p.suffix in {".txt", ".json", ".jinja", ".jinja2"}
                 and not p.name.upper().startswith(("LICENSE", "NOTICE")))
    content = {p: p.read_bytes() for p in sorted(paths) if p.is_file()}
    return {"scope": "source at this operation; saved generation retains its own manifest",
            "files": {str(p.relative_to(root)): sha256(data).hexdigest() for p, data in content.items()}}


def provenance(spec, *, implementation=None):
    config = matrix_config(spec)
    files = {spec.get("system_prompt_file"), spec.get("corpus"), spec.get("evidence_corpus"), spec.get("lifecycle", {}).get("warmup_input"), config.calibration_path,
             config.cost_model.artifact, config.cost_model.output_length_artifact}
    if spec.get("retriever_index"):
        files.add(str(Path(spec["retriever_index"]) / "manifest.json"))
    if spec.get("retriever_config"):
        files.update(retriever_config_sources(spec["retriever_config"]))
    for method in spec["methods"]:
        if method.get("retriever_config"):
            files.update(retriever_config_sources(method["retriever_config"]))
        for candidate in method.get("candidates", [{}]):
            overrides = merge(method.get("overrides", {}), candidate)
            native = merge(method.get("native") or {}, overrides.get("native", {}))
            files.add(native.get("engine_manifest"))
            files.add(overrides.get("calibration_path"))
            files.add(overrides.get("cost_model", {}).get("artifact"))
            files.add(overrides.get("cost_model", {}).get("output_length_artifact"))
    files.update(quality_evidence_files(spec))
    implementation = implementation if implementation is not None else implementation_manifest()
    return {"input_sha256": sha256(Path(spec["input"]).read_bytes()).hexdigest(),
            "split_sha256": sha256(Path(spec["split_manifest"]).read_bytes()).hexdigest(),
            # Config identity is a single experiment boundary check, never a request-path hash.
            "spec_sha256": sha256(json.dumps({k:v for k,v in spec.items() if k != "path_map"}, sort_keys=True).encode()).hexdigest(),
            "base_config_sha256": sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
            "implementation": implementation,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "external_inputs": {str(path): sha256(Path(remap(path, spec.get("path_map", {}))).read_bytes()).hexdigest() for path in sorted(p for p in files if p)}}


def generation_provenance(spec, *, implementation=None):
    """Offline quality eligibility can be added after generation; inputs cannot."""
    generation_spec = deepcopy(spec)
    generation_spec.get("tuning", {}).pop("quality_eligible", None)
    generation_spec.get("tuning", {}).pop("quality_evidence", None)
    return provenance(generation_spec, implementation=implementation)


def same_execution_provenance(recorded, current):
    """Code digests trace the run; declared method revisions bind behavior."""
    def inputs(value):
        return {key: item for key, item in value.items()
                if key not in {"implementation", "metric_schema_version"}}
    return bool(recorded and recorded.get("implementation")) and inputs(recorded) == inputs(current)


def validate_frozen_selection(recorded, current):
    if not same_execution_provenance(recorded, current):
        raise ValueError("frozen development selection inputs/configuration changed: regenerate development answers")
    if recorded["metric_schema_version"] != current["metric_schema_version"]:
        raise ValueError("metric version changed: recompute metrics and refreeze the saved development answers")


class QualityNotReady(ValueError):
    """A real acceptance report exists, but cannot authorize a B6 freeze."""
    def __init__(self, report):
        super().__init__("required quality evidence is incomplete or no B6 candidate is eligible; keep readiness, not a frozen B6")
        self.report = report


def quality_evidence_files(spec):
    """Bind saved grading files once at freeze; none are generation inputs."""
    path = spec.get("tuning", {}).get("quality_evidence")
    if path is None:
        return set()
    mapping = spec.get("path_map", {})
    value = json.loads(Path(remap(path, mapping)).read_text())
    files = {str(path)}
    paths = {block["quality_config"] for block in value["blocks"]}
    files.update(paths)
    configs = [json.loads(Path(remap(p, mapping)).read_text()) for p in paths]
    for config in configs:
        files.add(config.get("split_manifest"))
        files.add(config.get("frozen_protocol"))
        for method in config.get("methods", {}).values():
            files.update(method.get(k) for k in ("run", "audit", "scores", "manifest", "server_record"))
    return {p for p in files if p}


def _quality_tasks(spec):
    from adapters.mtrag import CLOUD_COLLECTION
    split = set(json.loads(Path(spec["split_manifest"]).read_text())["dev"])
    cohort = [row for row in load_rows(spec["input"]) if row["conversation_id"] in split
              and (spec.get("domain", "cloud") != "cloud" or row["Collection"] == CLOUD_COLLECTION)]
    if spec.get("limit_sessions"):
        selected = set(sorted({r["conversation_id"] for r in cohort})[:spec["limit_sessions"]])
        cohort = [r for r in cohort if r["conversation_id"] in selected]
    return {row["task_id"] for row in cohort}


def _evaluate_bound_quality(spec, by_name, value, reference, expected_tasks):
    """Bind actual job outputs, cohort and evaluator for one workload block."""
    from bench.quality_acceptance import evaluate
    original_value = value
    mapping = spec.get("path_map", {})
    value = remap(value, mapping)
    if (value.get("scope") != "dev" or value.get("reference") != reference
            or value.get("epsilon") != spec["tuning"]["quality_constraints"]["epsilon"]
            or not value.get("frozen_protocol") or set(value.get("methods", {})) != set(by_name)
            or sha256(Path(value.get("split_manifest", "")).read_bytes()).hexdigest() != sha256(Path(spec["split_manifest"]).read_bytes()).hexdigest()):
        raise ValueError("quality evidence must match all declared jobs, dev split, reference, epsilon and frozen evaluator protocol")
    for method, result in by_name.items():
        declared = value["methods"][method]
        bindings = [("run", "raw_output"), ("audit", "audit_output")]
        if result.get("run_manifest"):
            bindings.append(("manifest", "run_manifest"))
        elif declared.get("manifest"):
            raise ValueError(f"quality manifest has no recorded generation source for {method}")
        for field, result_key in bindings:
            if not result.get(result_key) or not declared.get(field) or Path(declared[field]).resolve() != Path(remap(result[result_key], mapping)).resolve():
                raise ValueError(f"quality {field} is not the actual generated artifact for {method}")
            recorded = result.get("artifact_sha256", {}).get(result_key)
            if recorded is None or sha256(Path(declared[field]).read_bytes()).hexdigest() != recorded:
                raise ValueError(f"quality {field} bytes differ from the completed generation artifact for {method}")
        payload = json.loads(Path(declared["run"]).read_text())
        turns = payload.get("report", payload)["turns"]
        if len(turns) != len(expected_tasks) or {r["task_id"] for r in turns} != expected_tasks:
            raise ValueError(f"quality run does not retain the complete declared development cohort: {method}")
        if any((row.get("online_turn") or {}).get("protocol") != result["protocol"] for row in turns):
            raise ValueError(f"quality run protocol differs from its declared job: {method}")
        if ("repetition" in result
                and payload.get("experiment_protocol", {}).get("repetition") != result["repetition"]):
            raise ValueError(f"quality run repetition differs from its declared job: {method}")
        if result.get("server_profile"):
            recorded = remap(result.get("server_execution", {}).get("record"), mapping)
            if (not recorded or Path(declared.get("server_record", "")).resolve() != Path(recorded).resolve()
                    or sha256(Path(recorded).read_bytes()).hexdigest() != result.get("artifact_sha256", {}).get("server_record")):
                raise ValueError(f"quality server record is not bound to the actual owned execution: {method}")
    report = evaluate({**original_value, "path_map": mapping})
    complete = all(not data["simulation"] and not data["generation_incomplete"]
        and not data["unbound_audit_task_ids"] and not data["unbound_score_task_ids"]
        and data["evaluator_kinds"] and set(data["evaluator_kinds"]) <= {"human_blind", "independent_model_declared"}
        and all(not dimension["missing_task_ids"] for dimension in data["coverage"].values())
        for data in report["methods"].values())
    complete &= all(not row["acceptance_blockers"] for row in report["comparisons"].values())
    unaccepted = [name for name, data in report["methods"].items()
        if not data["evaluator_kinds"] or not set(data["evaluator_kinds"]) <= {"human_blind", "independent_model_declared"}]
    gate = {name: name == reference or report["comparisons"][name]["dev_quality_eligible"] for name in by_name}
    return gate, report, complete, unaccepted


BLOCK_FIELDS = ("protocol", "concurrency", "seed", "repetition")


def _block_key(job):
    return (job["protocol"], job["concurrency"], job["seed"], job.get("repetition", 1))


def _quality_gate_by_block(spec, results, evidence, expected_tasks):
    planned = {job["job_id"]: job for job in matrix_jobs(spec)}
    actual = {job["job_id"]: job for job in results}
    if len(actual) != len(results) or set(actual) != set(planned):
        raise ValueError("block quality evidence needs every declared job, including failures")
    expected_blocks = {}
    for job_id, job in planned.items():
        recorded = actual[job_id]
        if (any(recorded.get(k) != job.get(k) for k in (*BLOCK_FIELDS, "method", "candidate_index", "split", "server_profile"))):
            raise ValueError(f"recorded job identity differs from its declared block: {job_id}")
        expected_blocks.setdefault(_block_key(job), {})[job_id] = recorded
    blocks = {}
    for entry in evidence["blocks"]:
        if set(entry) != {*BLOCK_FIELDS, "quality_config"}:
            raise ValueError("each quality block requires protocol, concurrency, seed, repetition and quality_config")
        key = _block_key(entry)
        if key in blocks:
            raise ValueError("duplicate quality block")
        blocks[key] = entry
    if set(blocks) != set(expected_blocks):
        raise ValueError("quality blocks must exactly cover all declared protocol/concurrency/seed/repetition blocks")
    reports, gate, evaluator = [], {}, None
    complete, unaccepted_jobs = True, []
    for key, jobs in expected_blocks.items():
        entry = blocks[key]
        references = [job_id for job_id, job in jobs.items()
                      if job["method"] == spec["tuning"]["quality_constraints"]["reference"]]
        if len(references) != 1:
            raise ValueError("each quality block requires exactly one actual reference job; reference candidates are ambiguous")
        value = json.loads(Path(remap(entry["quality_config"], spec.get("path_map", {}))).read_text())
        if set(value.get("methods", {})) != set(jobs) or value.get("reference") != references[0]:
            raise ValueError("quality block methods/reference must be its complete actual job IDs")
        failed = [job_id for job_id, job in jobs.items() if job["status"] != "completed"
                  or job["summary"]["timing"]["failure_fraction"] != 0
                  or job["summary"]["cohort"]["unfinished_turns"] != 0]
        block_report = {**entry, "job_ids": list(jobs), "generation_incomplete_jobs": failed}
        if failed:
            passed, block_complete = {}, False
            block_report.update(report=None, readiness_reason="generation_incomplete")
        else:
            passed, report, block_complete, rejected = _evaluate_bound_quality(
                spec, jobs, value, references[0], expected_tasks)
            protocol = json.loads(Path(remap(value["frozen_protocol"], spec.get("path_map", {}))).read_text())
            identity = {k: protocol.get(k) for k in ("evaluator", "dimensions", "epsilon")}
            if evaluator is not None and identity != evaluator:
                raise ValueError("quality blocks must share the complete evaluator protocol, dimensions and epsilon")
            evaluator = identity
            unaccepted_jobs.extend(rejected)
            block_report["report"] = report
        complete &= block_complete
        block_report["evidence_complete"] = block_complete
        reports.append(block_report)
        for job_id, job in jobs.items():
            candidates = gate.setdefault(job["method"], {})
            index = str(job["candidate_index"])
            candidates[index] = candidates.get(index, True) and block_complete and passed.get(job_id, False)
    eligible = [{"method": method["id"], "candidate_index": int(index)}
                for method in spec["methods"] if method.get("b6_eligible")
                for index, passed in gate.get(method["id"], {}).items() if passed]
    report = {"schema_version": 2, "scope": "dev", "blocks": reports, "candidate_eligibility": gate,
        "aggregation": "all_declared_blocks_must_pass; separate session-cluster intervals, no pooled repetition CI",
        "matrix_evidence_gate": {"all_required_evidence_complete": complete,
            "unaccepted_evaluator_jobs": unaccepted_jobs, "eligible_b6_candidates_on_evaluator": eligible,
            "ready_to_freeze": complete and bool(eligible)}}
    if not complete or not eligible:
        raise QualityNotReady(report)
    return gate, report


def quality_gate_from_evidence(spec, results):
    """All matrices use the same job-bound workload blocks."""
    tuning = spec["tuning"]
    if "quality_eligible" in tuning:
        raise ValueError("require_quality_evidence rejects a supplied boolean quality gate")
    if not tuning.get("quality_evidence"):
        raise ValueError("require_quality_evidence needs the saved quality blocks; no B6 is frozen")
    value = json.loads(Path(remap(tuning["quality_evidence"], spec.get("path_map", {}))).read_text())
    if value.get("schema_version") != 2:
        raise ValueError("quality evidence requires schema_version=2 workload blocks")
    return _quality_gate_by_block(spec, results, value, _quality_tasks(spec))


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
    implementation = implementation_manifest()
    expected_provenance = generation_provenance(spec, implementation=implementation)
    if any(not same_execution_provenance(r.get("experiment_provenance"), expected_provenance) for r in results):
        raise ValueError("development results belong to different or unrecorded inputs/configuration/implementation; legacy unbound results require a new run")
    for row in results:
        if row["status"] == "completed" and row["summary"].get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            path = Path(remap(row["raw_output"], spec.get("path_map", {})))
            if sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]["raw_output"]:
                raise ValueError("recompute metrics requires the content-bound original raw_output")
            indexed[row["job_id"]] = {**row, "summary": summarize_run(json.loads(path.read_text()))}
    results = list(indexed.values())
    grouped = {}
    for job in expected:
        row = indexed[job["job_id"]]
        grouped.setdefault((job["method"], job["candidate_index"]), []).append(row)
    winners = {}
    quality_report = None
    if spec.get("tuning", {}).get("require_quality_evidence"):
        quality_gate, quality_report = quality_gate_from_evidence(spec, results)
    else:
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
    return {"schema_version": 1, "selection_split": "dev", "provenance": provenance(spec, implementation=implementation),
            "winners": winners, "b6": b6, "objective": "equal_cell_weight_mean_complete_turn_seconds",
            "simulation_only": simulation, "quality_gate_applied": bool(quality_gate),
            "quality_noninferiority_established": False,
            "quality_evidence_report": quality_report,
            "equal_budget": spec.get("tuning", {}).get("equal_budget", 1)}


async def execute_job(spec, job, directory, *, native_engine_manifest=None, verified_server=None):
    if job.get("server_profile") is not None:
        profile_id = job["server_profile"]
        if (verified_server is None or verified_server.get("status") != "running"
                or verified_server.get("server_profile") != profile_id
                or verified_server.get("profile") != spec["server_profiles"][profile_id]):
            raise ValueError("server_profile requires its verified owned-server executor before generation")
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
        if "repetition" in job:
            environment["CONTEXTKV_JOB_REPETITION"] = str(job["repetition"])
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
                **job.get("retrieval", retrieval_settings(spec)),
                evidence_corpus=spec.get("evidence_corpus"),
                concurrency=job["concurrency"], seed=job["seed"], limit_sessions=spec.get("limit_sessions"),
                think_time=spec.get("think_time", 0), arrival_rate=spec.get("arrival_rate"),
                arrival_window=spec.get("arrival_window"), drain_timeout=spec.get("drain_timeout"),
                per_turn_timeout=spec.get("per_turn_timeout"))
    shared_config = matrix_config(spec)
    sampler = None
    if verified_server is not None:
        from bench.native_metrics import NativePoolSampler
        profile = verified_server["profile"]
        sampler = NativePoolSampler(shared_config.engine.base_url,
            gpu_capacity=profile["gpu_tokens"], host_enabled=profile["hicache"],
            model=shared_config.model.name,
            initial_empty=(verified_server.get("cache_state_protocol") == "fresh_process_empty_KV_no_generation_warmup"
                           and not lifecycle.get("before_job")))
    output = None
    try:
        if sampler is not None:
            await sampler.start()
        if job["kind"] == "native":
            from bench.native_baseline import _run
            native = job["native"] or {}
            declared_manifest = native.get("engine_manifest")
            if (native_engine_manifest is not None and declared_manifest is not None
                    and Path(native_engine_manifest).resolve() != Path(declared_manifest).resolve()):
                raise ValueError("runtime native manifest cannot replace a declared static manifest")
            observed_manifest = native_engine_manifest or declared_manifest
            if observed_manifest is None:
                raise ValueError("native job requires a verified engine manifest from its owned-server executor")
            args.update(baseline=native["baseline"], base_url=native["base_url"], engine_manifest=str(observed_manifest),
                        model_path=native.get("model_path"),
                        model_config=shared_config.model, sampling=shared_config.sampling,
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
    finally:
        if sampler is not None:
            await sampler.stop()
            if output is None:
                (directory / "native-pool-observation.json").write_text(json.dumps(sampler.observation(), indent=2))
    payload = json.loads(output.read_text())
    if sampler is not None:
        payload["native_pool_observation"] = sampler.observation()
    default_cache_state = "fresh_mock_instance" if payload.get("engine_kind") == "mock" else "unspecified_hardware"
    protocol = {"cache_state_protocol": lifecycle.get("cache_state_protocol", default_cache_state),
                "comparison_scope": spec.get("comparison_scope", "execution"),
                "retrieval": job.get("retrieval", retrieval_settings(spec)),
                "before_job": hook, "warmup_input": lifecycle.get("warmup_input"),
                "response_mode": spec.get("response_mode", "buffered"), "slo": spec.get("slo", {})}
    if "repetition" in job:
        protocol["repetition"] = job["repetition"]
    payload["experiment_protocol"] = protocol
    output.write_text(json.dumps(payload, indent=2))
    summary = summarize_run(payload)
    summary["experiment_protocol"] = protocol
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "summary.md").write_text(markdown_report(summary))
    result = {**job, "status": "completed", "raw_output": str(output), "summary": summary}
    audits = list(directory.glob("audit-*.jsonl"))
    if len(audits) == 1:
        result["audit_output"] = str(audits[0])
    if job["kind"] == "native":
        result["native_engine_manifest"] = {"path": str(Path(observed_manifest).resolve()),
                                           "sha256": payload["engine_manifest_sha256"]}
    if job["kind"] == "controller":
        manifest_path = output.with_name(output.name.replace("benchmark-", "manifest-", 1))
        if manifest_path.is_file():
            result["run_manifest"] = str(manifest_path)
    # Saved raw bytes support metric recomputation without new generation.
    result["artifact_sha256"] = {key: sha256(Path(result[key]).read_bytes()).hexdigest()
                                 for key in ("raw_output", "audit_output", "run_manifest") if key in result}
    return result


async def run_matrix(spec, output, *, stage="dev", frozen=None, executor=execute_job):
    if stage not in {"dev", "test", "dev_ablation"}:
        raise ValueError("stage must be dev, test or dev_ablation")
    jobs = matrix_jobs(spec, split="dev" if stage == "dev_ablation" else stage,
                       frozen=frozen, ablations_only=stage == "dev_ablation")
    implementation = implementation_manifest()
    run_provenance = generation_provenance(spec, implementation=implementation)
    if needs_development_selection(spec, stage, frozen):
        if frozen is None:
            raise ValueError("test requires a frozen development selection")
        validate_frozen_selection(frozen["provenance"], provenance(spec, implementation=implementation))
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
        (root / "implementation-manifest.json").write_text(json.dumps(implementation, indent=2) + "\n")
        for job in jobs:
            try:
                result = await executor(spec, job, root / job["job_id"])
            except Exception as exc:
                result = {**job, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                hook_log = root / job["job_id"] / "before-job.json"
                if hook_log.exists():
                    result["before_job"] = json.loads(hook_log.read_text())
            if job.get("method_aliases"):
                result["method_aliases"] = job["method_aliases"]
            result["experiment_provenance"] = run_provenance
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
    aliases = {}
    for job in jobs:
        for alias in job.get("method_aliases", []):
            entry = aliases.setdefault(alias, {"canonical_method": job["method"], "job_ids": [],
                "scope": "same physical executions; not additional samples or repetitions"})
            entry["job_ids"].append(job["job_id"])
    if aliases:
        (root / "method-aliases.json").write_text(json.dumps(aliases, indent=2) + "\n")
    comparisons = []
    reference = spec.get("reference_method", "B3")
    reference = aliases.get(reference, {}).get("canonical_method", reference)
    for candidate in results:
        matches = [r for r in results if r["method"] == reference and r["status"] == "completed"
                   and _block_key(r) == _block_key(candidate)]
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
        try:
            selected = select_development(spec, results)
        except QualityNotReady as exc:
            readiness = Path(args.output).with_suffix(".readiness.json")
            readiness.parent.mkdir(parents=True, exist_ok=True)
            with readiness.open("x") as stream:
                json.dump(exc.report, stream, indent=2)
            raise SystemExit(f"Quality readiness saved to {readiness}; no B6 freeze produced") from exc
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
