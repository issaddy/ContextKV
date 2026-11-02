"""Summarize prespecified B3/B6/S3 on exact common saved QA task cohorts.

Only reads completed generation/scoring/cost records. It does not run models,
retry failures, choose a lambda, refit predictions or fill missing scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import yaml

from bench.plan_prediction_audit import audit_directory, summaries as prediction_summaries
from bench.study import read_jsonl, write_json, write_jsonl
from bench.study_metrics import _cost_summary, _metric, _primary, _scored, _track_specs


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def response_counts(row):
    response = row["response"]
    steps = response["step_metrics"]
    return {"READ": response["read_count"], "REDUCE": response["reduce_count"],
        "CPU_REDUCE": sum(step["kind"] == "REDUCE" and "cpu_reduce_ms" in step for step in steps),
        "model_REDUCE": sum(step["kind"] == "REDUCE" and "cpu_reduce_ms" not in step for step in steps),
        "ANSWER": int(response["receipt"]["generation_complete"]),
        "truncated_ANSWER": int(response["truncated"]),
        "idempotent_replay": int(response["idempotent_replay"])}


def history_messages(row):
    history = row.get("actual_history")
    return ([(turn["turn_id"], turn["user_text"], turn["assistant_text"]) for turn in history]
            if history is not None else None)


def completed(row):
    response = row.get("response")
    return (row["status"] == "completed" and response is not None
            and response.get("receipt") is not None and response["receipt"]["generation_complete"])


def compact(row):
    return {key: row.get(key) for key in ("method", "dataset", "split", "task_id", "conversation_id",
        "status", "question", "actual_history", "answer", "metrics", "cost", "response")}


def load_results(roots, files, methods, split):
    paths = set(Path(path).resolve() for path in files)
    for root in roots:
        for manifest in Path(root).resolve().rglob("generation_manifest.json"):
            directory = manifest.parent
            scored, raw = directory/"results.scored.jsonl", directory/"results.unscored.jsonl"
            if scored.exists() or raw.exists():
                paths.add(scored if scored.exists() else raw)
    grouped, sources = defaultdict(list), []
    for path in sorted(paths):
        sources.append(file_identity(path))
        generation_dirs = set()
        for row in read_jsonl(path):
            if row["method"] in methods and row["split"] == split:
                grouped[row["method"], row["dataset"], str(row["task_id"])].append((row, str(path)))
                if row.get("generation_dir"):
                    generation_dirs.add(row["generation_dir"])
        sources[-1]["generation_dirs"] = sorted(generation_dirs)
    unique, ambiguous = {}, []
    for key, candidates in grouped.items():
        signatures = {fingerprint(compact(row)) for row, _ in candidates}
        if len(signatures) == 1:
            row = dict(candidates[0][0])
            row["summary_sources"] = sorted({path for _, path in candidates})
            unique[key] = row
        else:
            ambiguous.append({"method": key[0], "dataset": key[1], "task_id": key[2],
                "reason": "different saved executions or scores for same task; no implicit best/latest selection",
                "source_files": [path for _, path in candidates]})
    return unique, ambiguous, sources


def group_summary(rows, spec):
    count = {key: sum(response_counts(row)[key] for row in rows) for key in (
        "READ", "REDUCE", "CPU_REDUCE", "model_REDUCE", "ANSWER", "truncated_ANSWER", "idempotent_replay")}
    primary = _primary(rows, spec)
    names = sorted({key for row in rows for key in row["metrics"]})
    raw = {}
    for name in names:
        values = [_metric(row, name, spec)[0] for row in rows]
        values = [value for value in values if value is not None]
        raw[name] = {"count": len(values), "mean": sum(values)/len(values) if values else None}
    return {"qa_count": len(rows), "primary_metric": spec["primary_metric"], "primary": primary,
        "normalized_primary": primary/spec.get("normalization", 1.0) if primary is not None else None,
        "raw_metrics": raw, "counts": count, "cost": _cost_summary(rows),
        "script_history_mismatch": {"true": sum(row.get("script_history_mismatch") is True for row in rows),
            "false": sum(row.get("script_history_mismatch") is False for row in rows),
            "unknown": sum(row.get("script_history_mismatch") is None for row in rows)}}


def summarize(matrix, index, methods, split, datasets):
    declared = matrix.get("cohorts", {}).get(split, {})
    specs = _track_specs(matrix, split)
    if datasets is not None:
        specs = {name: spec for name, spec in specs.items() if name in datasets}
    by_dataset, selected, qa, inventory = {}, [], [], {}
    for dataset, spec in specs.items():
        observed = {key[2] for key in index if key[1] == dataset}
        universe = set(map(str, declared[dataset])) if dataset in declared else observed
        method_rows = {method: {identity: index[method, dataset, identity] for identity in universe
                               if (method, dataset, identity) in index} for method in methods}
        valid = {method: {identity for identity, row in rows.items() if completed(row)}
                 for method, rows in method_rows.items()}
        graded = {method: {identity for identity in valid[method] if _scored(rows[identity], spec)}
                  for method, rows in method_rows.items()}
        common_completed = set.intersection(*(valid[method] for method in methods))
        common = set.intersection(*(graded[method] for method in methods))
        same_question, same_frozen_input = [], []
        for identity in sorted(common):
            rows = {method: method_rows[method][identity] for method in methods}
            question_equal = (all(row.get("question") is not None for row in rows.values())
                              and len({row["question"] for row in rows.values()}) == 1)
            history_known = all(row.get("actual_history") is not None for row in rows.values())
            history_equal = history_known and len({fingerprint(history_messages(row)) for row in rows.values()}) == 1
            if question_equal:
                same_question.append(identity)
            if question_equal and history_equal:
                same_frozen_input.append(identity)
            selected.extend(rows.values())
            qa.append({"dataset": dataset, "task_id": identity, "same_question": question_equal,
                "same_actual_history": history_equal if history_known else None,
                "methods": {method: {"attempt_id": row["response"]["attempt_id"],
                    "question": row["question"], "actual_history_messages_sha256": fingerprint(history_messages(row)),
                    "metrics": row["metrics"], "cost": row["cost"], "counts": response_counts(row),
                    "selection": row.get("evaluation_selection"), "source_files": row["summary_sources"],
                    "generation_dir": row.get("generation_dir"), "scorer": row.get("scoring")}
                    for method, row in rows.items()}})
        compared = {method: group_summary([method_rows[method][identity] for identity in sorted(common)], spec)
                    for method in methods}
        by_dataset[dataset] = {"declared": dataset in declared, "planned": len(universe),
            "common_completed_task_ids": sorted(common_completed), "common_scored_task_ids": sorted(common),
            "same_question_task_ids": same_question, "same_frozen_input_task_ids": same_frozen_input,
            "same_question_methods": {method: group_summary([method_rows[method][identity] for identity in same_question], spec)
                                      for method in methods},
            "same_frozen_input_methods": {method: group_summary([method_rows[method][identity] for identity in same_frozen_input], spec)
                                          for method in methods},
            "methods": compared, "coverage": {method: {"observed": len(rows), "completed": len(valid[method]),
                "scored": len(graded[method]), "missing_or_incomplete": sorted(universe-valid[method]),
                "missing_primary_score": sorted(valid[method]-graded[method])} for method, rows in method_rows.items()}}
        inventory[dataset] = {method: {"status_counts": dict(Counter(row["status"] for row in rows.values())),
            "all_completed_cost": _cost_summary([row for row in rows.values() if completed(row)])}
            for method, rows in method_rows.items()}
    aggregate = {}
    for method in methods:
        represented = {name: spec["weight"] for name, spec in specs.items()
                       if by_dataset[name]["methods"][method]["normalized_primary"] is not None}
        value = sum(weight*by_dataset[name]["methods"][method]["normalized_primary"]
                    for name, weight in represented.items())
        aggregate[method] = {"common_cohort_Q": value if len(represented) == len(specs) and specs else None,
            "represented_weight": sum(represented.values()), "declared_weights": {name: spec["weight"] for name, spec in specs.items()},
            "weighted_observed_contribution": value if represented else None,
            "all_declared_tasks_common_scored": all(track["declared"] and len(track["common_scored_task_ids"]) == track["planned"]
                                                    for track in by_dataset.values())}
        aggregate[method]["scope"] = "prespecified common subset, not a replacement for full-cohort benchmark Q"
    return {"datasets": by_dataset, "aggregate": aggregate, "all_observed_inventory": inventory}, selected, qa


def prediction_report(rows, supplied):
    attempts = {row["response"]["attempt_id"]: row for row in rows}
    predictions, sources = [], []
    if supplied:
        for path in supplied:
            predictions.extend(read_jsonl(path))
            sources.append(file_identity(path))
    else:
        for directory in sorted({row["generation_dir"] for row in rows if row.get("generation_dir")}):
            path = Path(directory)
            if not (path/"state"/"events.jsonl").exists() or not (path/"state"/"sessions.sqlite3").exists():
                sources.append({"generation_dir": directory, "unavailable_reason": "original_state_or_events_missing"})
                continue
            actual, provenance = audit_directory(path)
            predictions.extend(actual)
            sources.append(provenance)
    chosen = []
    seen = set()
    for item in predictions:
        identity = item["attempt_id"], item["step_id"], item["event_index"]
        if item["attempt_id"] in attempts and identity not in seen:
            seen.add(identity)
            row = attempts[item["attempt_id"]]
            chosen.append({**item, "method": row["method"], "dataset": row["dataset"], "task_id": row["task_id"]})
    return {"sources": sources, "methods": {method: prediction_summaries([item for item in chosen if item["method"] == method])
        for method in sorted({row["method"] for row in rows})}, "scope": "selected current phases on the exact common scored task cohort; paid and future tail removed"}, chosen


def batch_ledger(rows):
    result = []
    for directory in sorted({row["generation_dir"] for row in rows if row.get("generation_dir")}):
        for path in sorted((Path(directory)/"measurements").glob("*/batch_cost.json")):
            result.append({"generation_dir": directory, "source": file_identity(path), "batch": json.loads(path.read_text()),
                "scope": "whole measurement segment, including noncommon/failed/idle intervals; not added to common-QA attributed cost"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--results", action="append", default=[])
    parser.add_argument("--methods", nargs="+", required=True, help="Exact prespecified method IDs; no best-lambda selection")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--prediction-audit", action="append", default=[], help="Optional saved predictions.jsonl; otherwise audit local saved traces")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    index, ambiguous, sources = load_results(args.root, args.results, args.methods, args.split)
    summary, common, qa = summarize(matrix, index, args.methods, args.split, args.datasets)
    forecast, predictions = prediction_report(common, args.prediction_audit)
    summary.update(format="contextkv_common_plan_result_summary_v1", methods=args.methods, split=args.split,
        matrix=file_identity(args.matrix), sources=sources, ambiguous_tasks=ambiguous, prediction_audit=forecast,
        protocol="common completed and primary-scored task IDs across all declared methods; each method keeps its actual history; exact frozen-input cohort reported separately",
        selection="prespecified method IDs; test results never select lambda or rewrite frozen configuration",
        physical_cost_scope="saved online per-request attribution; whole segment physical ledgers retained separately",
        missing_values="missing scores, costs or predictions remain null")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"summary.json", summary)
    write_jsonl(output/"common-qa.jsonl", qa)
    write_jsonl(output/"prediction-errors.jsonl", predictions)
    write_json(output/"batch-ledger.json", batch_ledger([{"generation_dir": directory}
        for item in sources for directory in item["generation_dirs"]]))
    text = ["# B3 / B6 / S3 实际共同 QA", "", "共同 cohort 为三方法均完成且已有原始主指标评分的 task IDs；实际历史各自保留。", "",
        "| 轨道 | 方法 | 共同 QA | 原始主分 | READ | REDUCE | ANSWER | 平均 ms | GPU GiB·s | Host GiB·s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, track in summary["datasets"].items():
        for method, row in track["methods"].items():
            counts, cost = row["counts"], row["cost"]
            text.append(f"| {name} | {method} | {row['qa_count']} | {row['primary']} | {counts['READ']} | {counts['REDUCE']} | "
                        f"{counts['ANSWER']} | {cost['mean_complete_ms']} | {cost['gpu_gib_seconds']} | {cost['host_gib_seconds']} |")
        text.extend(["", f"{name}：预定 {track['planned']}；共同完成 {len(track['common_completed_task_ids'])}；"
                     f"共同评分 {len(track['common_scored_task_ids'])}；同问题 {len(track['same_question_task_ids'])}；"
                     f"同问题和冻结实际历史 {len(track['same_frozen_input_task_ids'])}。", ""])
    text.extend(["预测误差见 `summary.json.prediction_audit`：逐 kind 的 count/MAE/bias/RMSE，实测与预测不混称。",
                 "完整测量段费用含失败与空闲，另存 `batch-ledger.json`，不与请求归因积分叠加。", ""])
    (output/"report.md").write_text("\n".join(text))
    print(json.dumps({"output": str(output.resolve()), "common_qa": len(qa), "ambiguous_tasks": len(ambiguous)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
