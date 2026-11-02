"""Freeze A cohorts, execute declared dev/test batches and preserve reports."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from urllib.parse import urlparse

import yaml

from bench.evidence_policy_comparison import own_p3_cell
from bench.study import dataset_rows, merge_config
from training.a_workflow import stage
from training.reader_data import digest, file_digest, rows, write_rows


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def prepare(args):
    settings = yaml.safe_load(Path(args.config).read_text())["a_final"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(settings["study_config"]).read_text())
    comparison = yaml.safe_load(Path(settings["comparison_config"]).read_text())["policy_comparison"]
    comparison["policies"]["P1"] = args.policy_asset
    config = merge_config(config, {"policy_comparison": comparison,
        "generation": {"run_config": comparison["helper_run_config"], "helper_run_config": comparison["helper_run_config"],
                       "teacher_url": comparison["teacher_url"]}})
    config["base_study_config"] = settings["study_config"]
    source_matrix = yaml.safe_load(Path(settings["public_matrix"]).read_text())
    source_matrix["datasets"]["twowiki"] = yaml.safe_load(Path(settings["twowiki_spec"]).read_text())
    matrix = deepcopy(source_matrix)
    matrix["datasets"] = {}
    cohorts, sources = {"dev": {}, "test": {}}, {}
    for name, limits in settings["cohorts"].items():
        spec = deepcopy(source_matrix["datasets"][name])
        spec.pop("dev_conversations", None)
        spec.pop("dev_seed", None)
        spec["weight"] = limits["weight"]
        materialized = {}
        for split in ("dev", "test"):
            values = dataset_rows(spec, split)
            identities = sorted({row["conversation_id"] for row in values},
                key=lambda identity: digest([settings["seed"], name, split, identity]))[:limits[split]]
            selected = [row for row in values if row["conversation_id"] in set(identities)]
            path = output/"cohorts"/name/split/"tasks.jsonl"
            write_rows(path, selected)
            materialized[split] = str(path)
            cohorts[split][name] = [row["task_id"] for row in selected]
            source_path = Path(spec["inputs"][split] if "inputs" in spec else spec["input"])
            sources[name+":"+split] = {"source": str(source_path.resolve()), "source_sha256": file_digest(source_path),
                "population_turns": len(values), "population_conversations": len({row["conversation_id"] for row in values}),
                "selected_conversations": identities, "selected_turns": len(selected), "cohort_sha256": file_digest(path),
                "split_kind": spec["split_kind"], "sample_not_full_dataset": len(selected) < len(values)}
        spec.pop("input", None)
        spec.pop("split_manifest", None)
        spec["inputs"] = materialized
        matrix["datasets"][name] = spec
    matrix["methods"] = [{"id": policy+"_"+reader, "strategy": "B7"}
        for policy in ("P0", "P1") for reader in comparison["readers"]]
    matrix["cohorts"] = cohorts
    matrix["selection"]["development_weights"] = {
        name: limits["weight"] for name, limits in settings["cohorts"].items()}
    matrix["selection"]["test_comparators"] = ["P0_R0"]
    matrix["resources"].update(settings["resources"])
    for field in ("gpu_capacity_tokens", "host_capacity_tokens"):
        matrix["resources"].pop(field, None)
    matrix["study_config"] = str(output/"study.yaml")
    (output/"study.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output/"matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False))
    write(output/"cohort_manifest.json", {"created_unix_ns": time.time_ns(), "seed": settings["seed"], "cohorts": cohorts,
        "sources": sources, "test_selection": "fixed before test generation; not chosen by answer quality",
        "development_methods": settings["development_methods"], "source_config_sha256": file_digest(args.config),
        "freeze_target_beijing": "2026-11-03 05:00", "delivery_deadline_beijing": "2026-11-03 09:00"})
    write(output/"schedule.json", {"diagnostic": {"frozen_states": 14, "own_P3_conversations": 2,
        "methods": [row["id"] for row in matrix["methods"]], "separate_from_public_quality": True},
        "public_development_methods": settings["development_methods"], "test_methods": "P0_R0 plus frozen selected candidate",
        "scoring": {name: "original 2Wiki answer/SP/evidence/joint scorer" if name == "twowiki"
                    else "existing study_scoring; frozen R0 offline judge" for name in matrix["datasets"]},
        "incomplete_or_failed": "retain raw outputs and costs; no fabricated quality; report exact coverage"})
    return {"matrix": str(output/"matrix.yaml"), "cohorts": str(output/"cohort_manifest.json")}


def diagnose(args):
    root, _, _ = bundle(args.output)
    diagnostic = root/"diagnostics"
    command = [sys.executable, "-m", "bench.evidence_policy_comparison", "both",
        "--config", str(root/"study.yaml"), "--matrix", args.diagnostic_matrix, "--dataset", args.diagnostic_dataset,
        "--records", args.records, "--goal-policy", args.goal_policy, "--limit-states", "14",
        "--limit-conversations", "2", "--output", str(diagnostic/"comparison")]
    stage(root, "A-four-frozen14-and-own-P3-two-conversations", command)
    methods = ("P0_R0", "P0_R1", "P1_R0", "P1_R1")
    for phase in ("frozen", "p3"):
        scoring = diagnostic/("score_"+phase)
        scoring.mkdir(parents=True, exist_ok=True)
        for method in methods:
            source = diagnostic/"comparison"/phase/method
            target = scoring/method
            if phase == "frozen":
                if not target.exists():
                    target.symlink_to(source.resolve(), target_is_directory=True)
            else:
                target.mkdir(exist_ok=True)
                for name in ("official.jsonl", "audit.jsonl", "references.jsonl", "results.unscored.jsonl"):
                    values = [row for path in sorted(source.glob("*/"+name)) for row in rows(path)]
                    write_rows(target/name, values)
        write(scoring/"matrix.json", {"kind": "A_policy_reader_grid", "cells": [{"method": method} for method in methods]})
        stage(root, "score-A-"+phase, [sys.executable, "-m", "bench.score_reader_diagnosis",
            "--diagnosis-dir", str(scoring), "--config", str(root/"study.yaml"),
            "--matrix", args.diagnostic_matrix, "--dataset", args.diagnostic_dataset, "--judge-gpu", "2"])
    write(diagnostic/"protocol.json", {"frozen_planned_states": 14, "P3_planned_conversations": 2,
        "methods": methods, "source_records_sha256": file_digest(args.records),
        "source_goal_policy_sha256": file_digest(args.goal_policy), "source_matrix_sha256": file_digest(args.diagnostic_matrix),
        "quality_scope": "constructed dynamic development; separate from public benchmark aggregate"})
    return {"diagnostics": str(diagnostic)}


def bundle(output, split=None):
    root = Path(output).resolve()
    if split == "test":
        frozen = json.loads((root/"frozen.json").read_text())
        return root, frozen["config"], frozen["matrix"]
    return root, yaml.safe_load((root/"study.yaml").read_text()), yaml.safe_load((root/"matrix.yaml").read_text())


async def cell(args):
    root, config, matrix = bundle(args.output, args.split)
    policy, reader = args.method.split("_")
    return await own_p3_cell(config, matrix, args.dataset, args.split,
        root/args.split/args.method/args.dataset, policy=policy, reader=reader)


async def score_cell(args):
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from bench.study_scoring import score_dataset
    root, config, matrix = bundle(args.output, args.split)
    directory = root/args.split/args.method/args.dataset
    scored = []
    parts = [directory] if args.dataset == "twowiki" else [path.parent for path in sorted(directory.glob("*/generation_manifest.json"))]
    for part in parts:
        values = rows(part/"results.unscored.jsonl")
        if not values:
            continue
        gpu = [] if args.dataset == "twowiki" else [2]
        port = urlparse(config["generation"]["teacher_url"]).port
        sampler = StudyResourceSampler(gpu, [os.getpid(), *listening_processes((port,))] if gpu else [os.getpid()])
        await sampler.start()
        start = time.monotonic()
        try:
            if args.dataset == "twowiki":
                from bench.public_multihop import score
                source_dir = matrix["datasets"]["twowiki"]["scoring"]["source_dir"]
                score(source_dir, matrix["datasets"]["twowiki"]["inputs"][args.split], part,
                    projection=matrix["datasets"]["twowiki"].get("answer_projection"))
                current = rows(part/"results.scored.jsonl")
            else:
                current = await score_dataset(config, matrix["datasets"][args.dataset], values, part)
            scored.extend(current)
        finally:
            end = time.monotonic()
            await sampler.stop()
            stamp = time.time_ns()
            write(part/"scoring"/f"physical.{stamp}.json", sampler.observation())
            write(part/"scoring"/f"complete_cost.{stamp}.json", {"window": [start, end],
                "physical": integrate_physical(sampler.samples, start, end), "model_scope": "offline fixed R0 judge" if gpu else "offline CPU original scorer"})
    write_rows(directory/"results.scored.jsonl", scored)
    return {"scored": len(scored)}


def run_batch(args):
    root, _, matrix = bundle(args.output, args.split)
    if args.split == "test":
        frozen = json.loads((root/"frozen.json").read_text())
        methods = frozen["test_methods"]
        if file_digest(root/"cohort_manifest.json") != frozen["cohort_manifest_sha256"]:
            raise ValueError("cohort manifest changed after development freeze")
        cohort = json.loads((root/"cohort_manifest.json").read_text())
        for name, spec in matrix["datasets"].items():
            if file_digest(spec["inputs"]["test"]) != cohort["sources"][name+":test"]["cohort_sha256"]:
                raise ValueError("test cohort contents changed after freeze")
        for item in frozen["policies"].values():
            if file_digest(Path(item["asset"])/"heads.pt") != item["heads_sha256"]:
                raise ValueError("policy weights changed after development freeze")
        for item in frozen["readers"].values():
            if file_digest(item["run_config"]) != item["run_config_sha256"]:
                raise ValueError("reader identity/profile changed after development freeze")
    else:
        methods = args.methods or json.loads((root/"schedule.json").read_text())["public_development_methods"]
    datasets = args.datasets or list(matrix["datasets"])
    receipts = []
    def dispatch(name, command):
        try:
            receipt = stage(root, name, command)
        except subprocess.CalledProcessError:
            receipt_path = sorted((root/"logs").glob(name+".*.json"))[-1]
            receipt = json.loads(receipt_path.read_text())
            receipt["source_receipt"] = str(receipt_path)
        receipts.append(receipt)
        write(root/args.split/"stages.json", receipts)
    for method in methods:
        for dataset in datasets:
            base = [sys.executable, "-m", "bench.a_final_batch", "--output", str(root),
                    "--method", method, "--dataset", dataset, "--split", args.split]
            if not args.score_only:
                dispatch(f"{args.split}-{method}-{dataset}-generate", [*base, "cell"])
            dispatch(f"{args.split}-{method}-{dataset}-score", [*base, "score-cell"])
    result = []
    for method in methods:
        for dataset in datasets:
            directory = root/args.split/method/dataset
            aggregate = directory/"results.scored.jsonl"
            if aggregate.exists():
                result.extend(rows(aggregate))
            else:
                result.extend(row for path in sorted(directory.glob("*/results.scored.jsonl")) for row in rows(path))
    write_rows(root/args.split/"results.scored.jsonl", result)
    write(root/args.split/"stages.json", receipts)
    failed = [receipt for receipt in receipts if receipt["returncode"] != 0]
    return {"methods": methods, "datasets": datasets, "scored": len(result), "failed_stages": len(failed),
            "status": "partial_failed" if failed else "completed"}


def freeze(args):
    from bench.study_metrics import select_development, summarize_methods
    root, config, matrix = bundle(args.output)
    result = rows(root/"dev"/"results.scored.jsonl")
    summaries = summarize_methods(matrix, result, "dev")
    if args.method is None:
        selected = select_development(matrix, result)
        method, status = selected["method"], "complete_cohort_continuous_quality_selection"
    else:
        method, selected, status = args.method, summaries[args.method], "explicit_operational_freeze_with_reported_missingness"
    policies = {}
    for name, path in config["policy_comparison"]["policies"].items():
        policies[name] = {"asset": path, "heads_sha256": file_digest(Path(path)/"heads.pt"),
                          "manifest_sha256": file_digest(Path(path)/"support-model.json")}
    reader_specs = {**config["policy_comparison"]["readers"],
                    **config["policy_comparison"].get("cell_readers", {})}
    frozen = {"selected_method": method, "selection_status": status, "selected": selected, "development_summaries": summaries,
        "test_methods": list(dict.fromkeys(("P0_R0", method))), "config": config, "matrix": matrix, "policies": policies,
        "readers": {name: {"run_config": item["run_config"], "run_config_sha256": file_digest(item["run_config"])}
                    for name, item in reader_specs.items()},
        "dev_results_sha256": file_digest(root/"dev"/"results.scored.jsonl"),
        "cohort_manifest_sha256": file_digest(root/"cohort_manifest.json"), "frozen_unix_ns": time.time_ns(),
        "test_cohorts": matrix["cohorts"]["test"], "test_selection_uses_test_answers": False,
        "claim_scope": "declared A development cohort, not a claim of strongest overall Incumbent"}
    write(root/"frozen.json", frozen)
    return {"method": method, "status": status, "test_methods": frozen["test_methods"]}


def report(args):
    from bench.study_metrics import summarize_methods, paired_comparisons
    root, _, matrix = bundle(args.output, "test")
    result = rows(root/"test"/"results.scored.jsonl")
    summary = summarize_methods(matrix, result, "test")
    comparisons = paired_comparisons(matrix, result, baseline="P0_R0", samples=500)
    write(root/"report"/"public_summaries.json", summary)
    write(root/"report"/"paired_comparisons.json", comparisons)
    write(root/"report"/"artifact_inventory.json", {"frozen": str(root/"frozen.json"),
        "cohorts": str(root/"cohort_manifest.json"), "schedule": str(root/"schedule.json"),
        "scored_results": str(root/"test"/"results.scored.jsonl"),
        "generation_and_failures": [str(path) for path in root.glob("test/*/*/*/manifest.json")],
        "costs": [str(path) for path in root.glob("test/**/batch_cost*.json")],
        "scope": "deterministic declared samples; raw metrics, missing cases, phase costs and diagnostics remain distinct"})
    return {"report_directory": str(root/"report")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "diagnose", "run", "cell", "score-cell", "freeze", "report"))
    parser.add_argument("--config", default="configs/a_final_batch.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-asset", default="models/study-support-r1-prototype22")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--method")
    parser.add_argument("--dataset")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--diagnostic-matrix")
    parser.add_argument("--diagnostic-dataset", default="dynamic_goal_dev")
    parser.add_argument("--records")
    parser.add_argument("--goal-policy")
    args = parser.parse_args(argv)
    if args.command in {"cell", "score-cell"}:
        result = asyncio.run(cell(args) if args.command == "cell" else score_cell(args))
        print(json.dumps({"result_rows": len(result)} if isinstance(result, list) else result))
    else:
        result = {"prepare": prepare, "diagnose": diagnose, "run": run_batch, "freeze": freeze, "report": report}[args.command](args)
        print(json.dumps(result, ensure_ascii=False))
        if args.command == "run" and result["failed_stages"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
