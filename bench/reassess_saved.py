"""Reassess saved answers and freeze full-queue reference reports without model calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import yaml

from bench.official import condition_metrics, load_conditioning_rules, load_jsonl
from bench.plan_result_summary import batch_ledger, file_identity, history_messages
from bench.public_multihop import score as score_twowiki
from bench.study_metrics import declared_queue_summary, paired_comparisons, _track_specs
from bench.study_scoring import _record_metrics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False)+"\n")


def condition_saved(rows, spec, checkout, output):
    """Recombine stored judge and algorithmic stages using the author's functions."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["generation_dir"]].append(row)
    rules = load_conditioning_rules(checkout)
    updated, sources = [], []
    for index, (directory, group) in enumerate(groups.items()):
        root = Path(directory)/"scoring"
        scoring = group[0].get("scoring") or {}
        if "component_completion" in (scoring.get("identity") or {}):
            paths = (Path(scoring["raw_outputs"][0]),)
        else:
            paths = (root/"official/scores.jsonl", root/"official.algorithmic.jsonl")
        records = {row["task_id"]: row for row in load_jsonl(paths[0])}
        for path in paths[1:]:
            for row in load_jsonl(path):
                records[row["task_id"]]["metrics"].update(row["metrics"])
        condition_metrics(rules, list(records.values()), ("RL_F", "RB_llm", "RB_agg"))
        target = output/f"conditioned-{index}.jsonl"
        write_rows(target, records.values())
        sources.extend(file_identity(path) for path in paths)
        for row in group:
            record = records.get(row["task_id"])
            metrics = _record_metrics(record, spec, {}) if record is not None else {}
            updated.append({**row, "metrics": {**row.get("metrics", {}), **metrics},
                "reassessment": {"operation": "author_answerability_conditioning",
                    "conditioned_records": str(target), "original_scoring": row.get("scoring")}})
    return updated, sources


def history_pairs(rows, baseline, methods):
    indexed = {(row["method"], row["dataset"], str(row["task_id"])): row for row in rows}
    pairs = []
    for (method, dataset, task), right in indexed.items():
        if method == baseline or method not in methods:
            continue
        left = indexed.get((baseline, dataset, task))
        if left is None:
            continue
        previous_left, previous_right = history_messages(left), history_messages(right)
        same_question = left.get("question") == right.get("question")
        pairs.append({"baseline": baseline, "method": method, "dataset": dataset, "task_id": task,
            "same_question": same_question,
            "same_actual_history": (previous_left == previous_right
                if previous_left is not None and previous_right is not None else None),
            "fixed_input_pair": same_question and previous_left is not None
                and previous_right is not None and previous_left == previous_right})
    return pairs


def paid_execution_index(rows, roots, destination):
    """Include steps of failed and interrupted attempts, even without a result row."""
    from profiling.native_usage_index import index

    directories = {Path(row["generation_dir"]) for row in rows if row.get("generation_dir")}
    directories.update(path.parent.parent for root in roots
        for path in Path(root).glob("**/state/sessions.sqlite3"))
    databases = sorted(str(path/"state/sessions.sqlite3") for path in directories)
    native = index(databases)
    write(destination, native)
    return {"path": str(destination), **native["totals"], "by_kind": native["by_kind"],
        "confirmed_pre_submission_steps": len(native["pre_submission_steps"]),
        "unmeasured_inventory": native["inventory_reasons"],
        "physical_batch_files": sorted({str(path) for directory in directories
            for pattern in ("batch_cost.*.json", "measurements/*/batch_cost.json")
            for path in directory.glob(pattern)})}


def reassess(matrix, rows, output, *, split, baseline, history_protocol,
             mtrag_checkout="data/mtrag-official", twowiki_source="data/2wikimultihopqa/source",
             execution_roots=None):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [row for row in rows if row["split"] == split]
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["dataset"]].append(row)
    updated, score_sources, official_manifests = [], [], {}
    for (method, dataset), group in groups.items():
        spec = matrix["datasets"][dataset]
        destination = output/"cells"/method/dataset
        if spec["primary_metric"] == "2wiki_answer_f1":
            write_rows(destination/"results.unscored.jsonl", group)
            manifest = score_twowiki(twowiki_source, spec["inputs"][split], destination,
                                     projection=spec.get("answer_projection"))
            official_manifests[f"{method}/{dataset}"] = manifest
            group = load_jsonl(destination/"results.scored.jsonl")
            score_sources.append(file_identity(spec["inputs"][split]))
        elif spec["primary_metric"] == "mtrageval_harmonic":
            group, sources = condition_saved(group, spec, mtrag_checkout, destination)
            score_sources.extend(sources)
        updated.extend(group)
    fields = ("task_id", "conversation_id", "turn", "dataset", "split", "method", "status",
              "question", "answer", "metrics", "cost", "generation_dir",
              "measurement_segment_id", "request_window")
    write_rows(output/"question_scores.jsonl", ({
        **{key: row.get(key) for key in fields},
        "source_result_path": row.get("source_result_path") or str(Path(row["generation_dir"])/"results.scored.jsonl"),
        "attempt_id": (row.get("response") or {}).get("attempt_id") or row.get("attempt_id"),
    } for row in updated))
    methods = sorted({row["method"] for row in updated})
    specs = _track_specs(matrix, split)
    summaries = {}
    for method in methods:
        selected = [row for row in updated if row["method"] == method]
        tracks = {name: declared_queue_summary([row for row in selected if row["dataset"] == name],
            spec, matrix["cohorts"][split][name]) for name, spec in specs.items()}
        values = [tracks[name]["queue_primary_score"] for name in specs]
        summaries[method] = {"datasets": tracks,
            "queue_quality": sum(spec["weight"]*tracks[name]["queue_primary_score"]
                /spec.get("normalization", 1.0) for name, spec in specs.items())
                if all(value is not None for value in values) else None,
            "queue_quality_bounds": [sum(spec["weight"]*tracks[name]["queue_score_bounds"][bound]
                /spec.get("normalization", 1.0) for name, spec in specs.items()) for bound in (0, 1)],
            "cost_batches": batch_ledger(selected)}
        if execution_roots is not None:
            summaries[method]["paid_native_usage"] = paid_execution_index(selected,
                execution_roots.get(method, ()), output/"costs"/method/"native.json")
    pairs = history_pairs(updated, baseline, methods)
    report = {"split": split, "baseline": baseline, "history_protocol": history_protocol,
        "methods": summaries, "official_rescoring": official_manifests,
        "scoring_sources": score_sources, "history_pairs": pairs,
        "common_scored_diagnostics": paired_comparisons(matrix, updated, baseline, split),
        "track_weights": {name: spec["weight"] for name, spec in specs.items()},
        "model_calls": 0, "reassessment_wall_ms": (time.monotonic()-started)*1000,
        "selection": "reference registration only; original freezes and test selections retained"}
    write(output/"report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--rows", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--history-protocol", choices=("fixed_history", "own_p3"), required=True)
    parser.add_argument("--mtrag-checkout", default="data/mtrag-official")
    parser.add_argument("--twowiki-source", default="data/2wikimultihopqa/source")
    parser.add_argument("--cost-asset", action="append", default=[])
    parser.add_argument("--execution-root", action="append", default=[], metavar="METHOD=PATH",
                        help="Include every attempt database below a declared method directory")
    args = parser.parse_args(argv)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    rows = [{**row, "source_result_path": row.get("source_result_path") or str(Path(path).resolve())}
            for path in args.rows for row in load_jsonl(path)
            if args.methods is None or row["method"] in args.methods]
    roots = defaultdict(list)
    for item in args.execution_root:
        method, path = item.split("=", 1)
        roots[method].append(path)
    report = reassess(matrix, rows, args.output, split=args.split, baseline=args.baseline,
        history_protocol=args.history_protocol, mtrag_checkout=args.mtrag_checkout,
        twowiki_source=args.twowiki_source, execution_roots=roots if args.execution_root else None)
    write(Path(args.output)/"sources.json", {"matrix": file_identity(args.matrix),
        "results": [file_identity(path) for path in args.rows],
        "paid_cost_assets": [file_identity(path) for path in args.cost_asset]})
    print(json.dumps({method: {"queue_quality": value["queue_quality"],
        "datasets": {name: {key: dataset[key] for key in ("declared", "completed", "scored",
            "queue_primary_score")} for name, dataset in value["datasets"].items()}}
        for method, value in report["methods"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
