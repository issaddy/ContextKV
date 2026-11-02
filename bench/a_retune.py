"""Prepare an A V2 bundle, reuse MultiDoc receipts and declare fresh reader queues."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import yaml

from training.reader_data import digest, file_digest, rows, write_rows


ANSWER_SYSTEM_PROMPT = (
    "Use task conditions and Reading conclusions to answer the current question. "
    "Give only the shortest complete final answer followed by its allowed source citations [E_...]. "
    "Keep intermediate facts and explanations internal. If evidence is insufficient, say so briefly.")
FINAL_LINE_SYSTEM_PROMPT = (
    "Use task conditions and Reading conclusions to answer the current question. "
    "Briefly state the supporting facts, then finish with exactly one terminal line: "
    "Final answer: <shortest complete answer> [E_...]. "
    "Use only allowed source citations. If evidence is insufficient, say so briefly in that final field.")
METHODS = ["P0_R0", "P1_R0", "P1_R1", "P0_R2", "P1_R2"]


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def collect(output):
    output = Path(output).resolve()
    matrix = yaml.safe_load((output/"matrix.yaml").read_text())
    schedule = json.loads((output/"schedule.json").read_text())
    values, artifacts = [], []
    for method in schedule["public_development_methods"]:
        for name in matrix["datasets"]:
            directory = output/"dev"/method/name
            aggregate = directory/"results.scored.jsonl"
            paths = [aggregate] if aggregate.exists() else sorted(directory.glob("*/results.scored.jsonl"))
            for path in paths:
                values.extend(rows(path))
                artifacts.append({"method": method, "dataset": name, "path": str(path.resolve()),
                    "sha256": file_digest(path), "read_only_reuse": name in schedule["read_only_reused_datasets"]})
    write_rows(output/"dev/results.scored.jsonl", values)
    stages = []
    for path in sorted((output/"logs").glob("dev-*.json")):
        stages.append({**json.loads(path.read_text()), "source_receipt": str(path.resolve())})
    write(output/"dev/stages.json", stages)
    manifest = {"status": "collected_existing_artifacts", "rows": len(values), "sources": artifacts,
        "stages": len(stages), "failed_stages": sum(row["returncode"] != 0 for row in stages),
        "generation_or_scoring_executed": False, "created_unix_ns": time.time_ns()}
    write(output/"dev/collection_manifest.json", manifest)
    return manifest


def prepare(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ("study.yaml", "matrix.yaml", "schedule.json", "cohort_manifest.json")
    originals = {name: {"path": str(source/name), "sha256": file_digest(source/name)} for name in names}
    for name in names:
        shutil.copyfile(source/name, output/name)
    config = yaml.safe_load((output/"study.yaml").read_text())
    matrix = yaml.safe_load((output/"matrix.yaml").read_text())
    schedule = json.loads((output/"schedule.json").read_text())
    manifest = json.loads((output/"cohort_manifest.json").read_text())
    seed = manifest["seed"]
    upstream = Path(matrix["datasets"]["twowiki"]["corpus_path"]).resolve().parent/"cohort_manifest.json"
    upstream_manifest = json.loads(upstream.read_text())
    previous_sources = deepcopy(manifest["sources"])
    methods = ["P0_R0", "P1_R1", "P1_R2"] if args.provided_full_direct else METHODS
    reused_datasets = ["multidoc2dial", "loong"] if args.reuse_loong else ["multidoc2dial"]
    fresh_datasets = ["twowiki"] if args.reuse_loong else ["twowiki", "loong"]
    prompt = FINAL_LINE_SYSTEM_PROMPT if args.answer_projection == "final_line" else ANSWER_SYSTEM_PROMPT
    version = ("V4_all_supplied_originals_when_actual_budget_fits" if args.provided_full_direct else
        "V3_final_line_format_ablation" if args.answer_projection == "final_line" else "V2_short_answer_and_read_budget")
    if args.provided_full_direct:
        version = "V4_provided_full_direct_when_fits_reader_baseline"
    cohorts, sources, counts = {"dev": {}, "test": {}}, {}, {}
    for name, spec in matrix["datasets"].items():
        for split in ("dev", "test"):
            original = Path(spec["inputs"][split]).resolve()
            population = rows(original)
            selected = population
            if name == "twowiki" and split == "dev":
                identities = sorted({row["conversation_id"] for row in population},
                    key=lambda identity: digest([seed, name, split, identity]))[:args.dev_twowiki_count]
                selected = [row for row in population if row["conversation_id"] in set(identities)]
            target = output/"cohorts"/name/split/"tasks.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "twowiki" and split == "dev":
                write_rows(target, selected)
            else:
                shutil.copyfile(original, target)
            spec["inputs"][split] = str(target)
            cohorts[split][name] = [row["task_id"] for row in selected]
            sources[name+":"+split] = {**previous_sources[name+":"+split],
                "source": str(original), "source_sha256": file_digest(original),
                "population_turns": len(population),
                "population_conversations": len({row["conversation_id"] for row in population}),
                "selected_turns": len(selected), "selected_conversations": sorted({row["conversation_id"] for row in selected}),
                "cohort_path": str(target), "cohort_sha256": file_digest(target),
                "sample_not_full_dataset": len(selected) < len(population)
                    or previous_sources[name+":"+split]["sample_not_full_dataset"],
                "parent_source_record": previous_sources[name+":"+split]}
            counts[name+":"+split] = {"parent_rows": len(population), "selected_rows": len(selected),
                "content_unchanged": file_digest(original) == file_digest(target)}
    matrix["cohorts"] = cohorts
    matrix["study_config"] = str(output/"study.yaml")
    matrix["datasets"]["twowiki"].update(reuse_provider=True, answer_system_prompt=prompt)
    if args.provided_full_direct:
        matrix["datasets"]["twowiki"].setdefault("selection", {})["provided_full_direct_when_fits"] = True
    if args.answer_projection == "final_line":
        matrix["datasets"]["twowiki"]["answer_projection"] = "final_line"
    if args.provided_full_direct:
        matrix["datasets"]["twowiki"]["selection"] = {
            **matrix["datasets"]["twowiki"].get("selection", {}),
            "provided_full_direct_when_fits": True, "body_tokens": 8192}
        matrix["methods"] = [method for method in matrix["methods"] if method["id"] in methods]
    matrix["datasets"]["loong"]["reading"] = {
        **matrix["datasets"]["loong"].get("reading", {}), "input_target_tokens": 4096, "output_tokens": 8192}
    matrix["datasets"]["loong"]["coverage_mode"] = "all_supplied_documents"
    config["execution"]["output_tokens"] = 1024
    schedule["public_development_methods"] = methods
    schedule["retune_format_version"] = version
    schedule["retune_fresh_datasets"] = fresh_datasets
    schedule["read_only_reused_datasets"] = reused_datasets
    retune = {"created_unix_ns": time.time_ns(), "source_bundle": str(source), "source_files": originals,
        "format_version": version, "answer_projection": args.answer_projection,
        "provided_full_direct": args.provided_full_direct,
        "full_input_shared_between_P0_P1_on_fit": args.provided_full_direct,
        "provided_full_direct_when_fits": args.provided_full_direct,
        "direct_full_scope": "twowiki standalone supplied originals only; all complete source blocks; no gold selection; body8192/QA1024",
        "direct_comparison": "P0/P1 selection heads bypassed when all originals fit; reader-only direct baseline; oversized requests keep existing READ",
        "seed": seed, "selection": "sort sha256(canonical_json([seed,'twowiki','dev',conversation_id])); first count",
        "selection_uses_answers": False, "development_twowiki": {
            "selected": len(cohorts["dev"]["twowiki"]), "parent_cohort": counts["twowiki:dev"]["parent_rows"],
            "original_development_cohort": len(upstream_manifest["cohorts"]["dev"]["task_ids"]),
            "official_labeled_dev_population": upstream_manifest["population"],
            "upstream_cohort_manifest": str(upstream), "upstream_cohort_manifest_sha256": file_digest(upstream)},
        "test_twowiki_ids": cohorts["test"]["twowiki"],
        "test_twowiki_content_unchanged": counts["twowiki:test"]["content_unchanged"],
        "cohort_counts": counts, "answer_system_prompt": prompt,
        "reading_profile": {"input_target_tokens": 4096, "output_tokens": 8192, "qa_output_tokens": 1024},
        "fresh_datasets": fresh_datasets, "read_only_reused_datasets": reused_datasets,
        "old_outputs": "preserved in source bundle; fresh output directories only for declared reruns",
        "frozen_json_copied": False, "generation_or_scoring_executed": False}
    manifest.update(created_unix_ns=time.time_ns(), cohorts=cohorts, sources=sources,
        development_methods=methods, retune_protocol=retune)
    manifest["v4_provided_full_direct" if args.provided_full_direct else
             "v3_format_ablation" if args.answer_projection == "final_line" else "v2_retune"] = retune
    (output/"study.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output/"matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False))
    write(output/"schedule.json", schedule)
    write(output/"cohort_manifest.json", manifest)
    reused = []
    for method in methods:
        for dataset in reused_datasets:
            old = source/"dev"/method/dataset
            new = output/"dev"/method/dataset
            new.parent.mkdir(parents=True, exist_ok=True)
            new.symlink_to(old, target_is_directory=True)
            files = [path for path in (old/"results.scored.jsonl", old/"results.unscored.jsonl", old/"manifest.json") if path.exists()]
            reused.append({"method": method, "dataset": dataset, "read_only_reference": True,
                "source": str(old), "link": str(new), "source_present": old.exists(),
                "files": [{"path": str(path.resolve()), "sha256": file_digest(path)} for path in files],
                "cost_policy": "keep original generation_dir, measurement IDs and costs; no new model or judge call"})
    write(output/"reused_multidoc.json", [row for row in reused if row["dataset"] == "multidoc2dial"])
    write(output/"reused_datasets.json", reused)
    queues = []
    for name in ("R0", "R1", "R2"):
        reader_methods = [method for method in methods if method.endswith("_"+name)]
        command = [sys.executable, "-m", "bench.a_final_batch", "run", "--output", str(output),
            "--split", "dev", "--datasets", *fresh_datasets, "--methods", *reader_methods]
        script = output/"queues"/(name+".sh")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env bash\n"+shlex.join(command)+"\n")
        script.chmod(0o755)
        queues.append({"reader": name, "methods": reader_methods, "command": command, "script": str(script),
            "fresh_datasets": fresh_datasets, "read_only_reused_datasets": reused_datasets})
    write(output/"queues.json", {"queues": queues,
        "after_all_queues": [sys.executable, "-m", "bench.a_retune", "--output", str(output), "--collect"],
        "parallel_aggregation": "per-cell artifacts and log receipts are independent; collect rebuilds the shared summary after queues finish",
        "service_and_GPU_lifecycle": "root schedules queues; preparation starts no services or model calls"})
    write(output/"retune_manifest.json", retune)
    summary = collect(output)
    return {"output": str(output), "dev_twowiki": retune["development_twowiki"],
        "test_twowiki": len(cohorts["test"]["twowiki"]), "reused_scored_rows": summary["rows"],
        "queue_manifest": str(output/"queues.json"), "frozen": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-twowiki-count", type=int, default=12)
    parser.add_argument("--answer-projection", choices=("none", "final_line"), default="none")
    parser.add_argument("--reuse-loong", action="store_true")
    parser.add_argument("--provided-full-direct", action="store_true")
    parser.add_argument("--provided-full-direct", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args(argv)
    if not args.collect and args.source is None:
        parser.error("--source is required for bundle preparation")
    result = collect(args.output) if args.collect else prepare(args)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
