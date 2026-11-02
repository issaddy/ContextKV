"""Score frozen reader cells and report comparisons on their common task set."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit
import uuid

import yaml

from bench.metrics import percentile
from bench.official import load_jsonl
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from bench.study_scoring import _judge_config, _settings, score_dataset


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def comparison_report(rows, primary):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], {})[row["task_id"]] = row
    scored = {method: {task for task, row in cells.items()
        if row["status"] == "completed" and primary in row["metrics"]}
        for method, cells in grouped.items()}
    common = set.intersection(*scored.values()) if scored else set()
    union = set.union(*(set(cells) for cells in grouped.values())) if grouped else set()
    summaries = {}
    for method, cells in grouped.items():
        paired = [cells[task] for task in sorted(common)]
        costs = {}
        for name in ("complete_ms", "total_input_tokens", "total_output_tokens", "gpu_gib_seconds", "host_gib_seconds"):
            values = [row["cost"].get(name) for row in paired]
            costs[name] = sum(values) if values and all(value is not None for value in values) else None
        metrics = set.intersection(*(set(row["metrics"]) for row in paired)) if paired else set()
        summaries[method] = {"planned": len(cells), "status_counts": dict(Counter(row["status"] for row in cells.values())),
            "scored": len(scored[method]), "paired_tasks": len(paired),
            "metrics": {name: statistics.fmean(row["metrics"][name] for row in paired) for name in metrics},
            "paired_cost_totals": costs,
            "paired_p95_complete_ms": percentile([row["cost"]["complete_ms"] for row in paired], .95)
                if paired and all(row["cost"].get("complete_ms") is not None for row in paired) else None}
    comparisons = []
    pairs = [(f"E{level}_R0_{binding}", f"E{level}_R1_{binding}", "reader")
        for level in range(3) for binding in ("inferred", "verified")]
    pairs += [(f"E{level}_{reader}_inferred", f"E{level}_{reader}_verified", "task_binding")
        for level in range(3) for reader in ("R0", "R1")]
    pairs += [(f"E{left}_{reader}_{binding}", f"E{left+1}_{reader}_{binding}", "evidence")
        for left in range(2) for reader in ("R0", "R1") for binding in ("inferred", "verified")]
    for before, after, intervention in pairs:
        if before in grouped and after in grouped:
            deltas = [grouped[after][task]["metrics"][primary]-grouped[before][task]["metrics"][primary]
                      for task in sorted(common)]
            comparisons.append({"before": before, "after": after, "intervention": intervention,
                "paired_tasks": len(deltas), "mean_delta": statistics.fmean(deltas) if deltas else None,
                "task_deltas": dict(zip(sorted(common), deltas, strict=True))})
    return {"scope": "frozen_actual_history_diagnosis; common completed and scored tasks across declared cells",
        "primary_metric": primary, "common_task_ids": sorted(common), "excluded_task_ids": sorted(union-common),
        "methods": summaries, "comparisons": comparisons, "public_benchmark_quality_aggregate": False,
        "uncertainty": "descriptive paired differences; no heldout significance claim"}


async def run(args):
    root = Path(args.diagnosis_dir)
    config = yaml.safe_load(Path(args.config).read_text())
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    spec = matrix["datasets"][args.dataset]
    cells = json.loads((root/"matrix.json").read_text())["cells"]
    methods = args.methods or sorted({row["method"] for row in cells})
    judge = _judge_config(config, _settings(config, spec))
    port = urlsplit(judge["base_url"]).port
    stages_path = root/"scoring-stages.json"
    all_rows = []
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else []
    for method in methods:
        directory = root/method
        rows = load_jsonl(directory/"results.unscored.jsonl")
        stage_id = uuid.uuid4().hex
        sampler = StudyResourceSampler(args.judge_gpu, listening_processes((port,)))
        await sampler.start()
        start, cpu_start = time.monotonic(), time.process_time()
        error = None
        try:
            scored = await score_dataset(config, spec, rows, directory)
        except Exception as failure:
            error = {"type": type(failure).__name__, "message": str(failure)}
            saved = directory/"results.scored.jsonl"
            scored = load_jsonl(saved) if saved.exists() else [
                {**row, "scoring": {"status": "failed", "error": error}} for row in rows]
        finally:
            end, cpu_end = time.monotonic(), time.process_time()
            await sampler.stop()
            physical_path = directory/"scoring"/f"physical.{stage_id}.json"
            write(physical_path, sampler.observation())
            stage = {"method": method, "stage_id": stage_id,
                "status": "completed" if error is None else "failed",
                "error": error,
                "window": {"start": start, "end": end, "clock": "client_monotonic"},
                "physical_cost": integrate_physical(sampler.samples, start, end),
                "controller_cpu_seconds": cpu_end-cpu_start,
                "cpu_scope": "scoring controller only; judge engine CPU process time is unmeasured",
                "physical_observation": str(physical_path.resolve()), "judge": judge,
                "cost_scope": "offline scorer and declared judge GPU/processes, separate from online generation"}
            write(directory/"scoring"/f"physical-cost.{stage_id}.json", stage)
            stages.append(stage)
            write(stages_path, stages)
        all_rows.extend(scored)
        latest = {row["method"]: row for row in stages if row["method"] in methods}
        report = comparison_report(all_rows, spec["primary_metric"])
        report.update(scoring_complete=len(latest) == len(methods) and all(
                          row["status"] == "completed" for row in latest.values()),
                      declared_methods=methods,
                      completed_methods=[name for name, row in latest.items() if row["status"] == "completed"],
                      failed_methods=[name for name, row in latest.items() if row["status"] == "failed"])
        write(root/"quality-comparison.json", report)
        print(json.dumps({"method": method, "stage_status": stage["status"], "error": error,
            "scored": sum(row["scoring"]["status"] == "scored"
            for row in scored), "requested": len(rows)}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--judge-gpu", type=int, action="append", required=True)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    if report["failed_methods"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
