"""Benchmark quality, budgeted selection and method comparisons."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


MTRAGEVAL_COMPONENTS = ("RL_F", "RB_llm", "RB_alg")
RB_AGG_ALIAS_SOURCE = "data/mtrag-official/scripts/evaluation/run_algorithmic.py:184"
MTRAGEVAL_RANKING_SOURCE = "https://ibm.github.io/mt-rag-benchmark/MTRAGEval/#-leaderboard-ranking"


def split_metric_spec(spec, split):
    """Resolve a preregistered split's metric without changing its task cohort."""
    override = spec.get("metric_by_split", {}).get(split, {})
    return {**spec, **override,
        "scoring": {**spec.get("scoring", {}), **override.get("scoring", {})}}


def _track_specs(matrix, split, datasets=None):
    """Use the selected split and declared dataset identities."""
    selected = datasets if datasets is not None else matrix.get("cohorts", {}).get(split)
    if selected is None:
        selected = [name for name, spec in matrix["datasets"].items()
                    if ("inputs" not in spec or split in spec["inputs"])
                    and not spec.get("evaluation_only")]
    return {name: {"protocol": matrix.get("protocol", "P3"),
                   **split_metric_spec(matrix["datasets"][name], split)}
            for name in selected}


def _number(value):
    if isinstance(value, (list, tuple)):
        return statistics.fmean(value) if value and all(_number(v) is not None for v in value) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    location = (len(ordered)-1)*probability
    left, right = math.floor(location), math.ceil(location)
    return ordered[left]+(ordered[right]-ordered[left])*(location-left)


def _metric(row, name, spec):
    configured = spec.get("metric_sources", {})
    if spec["primary_metric"] == "mtrageval_harmonic" and name not in configured:
        bases = (name, "RB_agg") if name == "RB_alg" else (name,)
        for suffix in ("_idk_underspecified", "_idk"):
            for base in bases:
                source = base+suffix
                if source in row["metrics"]:
                    return _number(row["metrics"][source]), source
        return None, name+"_idk"
    source = configured.get(name, name)
    if source in row["metrics"]:
        return _number(row["metrics"][source]), source
    if name == "RB_alg" and source == "RB_alg" and "RB_agg" in row["metrics"]:
        return _number(row["metrics"]["RB_agg"]), "RB_agg"
    return None, source


def _required(spec):
    return MTRAGEVAL_COMPONENTS if spec["primary_metric"] == "mtrageval_harmonic" else (spec["primary_metric"],)


def _scored(row, spec):
    return row["status"] == "completed" and all(_metric(row, name, spec)[0] is not None
                                                for name in _required(spec))


def _primary(rows, spec):
    if not rows:
        return None
    means = [statistics.fmean(_metric(row, name, spec)[0] for row in rows) for name in _required(spec)]
    return _primary_components(means, spec)


def _primary_components(means, spec):
    if spec["primary_metric"] == "mtrageval_harmonic":
        return 0.0 if any(value == 0 for value in means) else len(means)/sum(1/value for value in means)
    return means[0]


def _cost_total(rows, name):
    values = [_number(row.get("cost", {}).get(name)) for row in rows]
    return sum(values) if values and all(value is not None for value in values) else None


def _cost_summary(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    timings = [_number(row.get("cost", {}).get("complete_ms")) for row in completed]
    complete_timings = bool(timings) and all(value is not None for value in timings)
    result = {"mean_complete_ms": statistics.fmean(timings) if complete_timings else None,
        "p95_complete_ms": _quantile(timings, .95) if complete_timings else None}
    for name in ("total_input_tokens", "total_output_tokens", "online_compute_ms",
                 "auxiliary_input_tokens", "auxiliary_output_tokens", "reader_input_tokens",
                 "reader_output_tokens", "actual_prefill_tokens", "actual_device_hit_tokens",
                 "actual_host_loaded_tokens"):
        result[name] = _cost_total(rows, name)
    batches = [json.loads(path.read_text()) for directory in {
        row["generation_dir"] for row in rows if "generation_dir" in row}
        if (path := Path(directory)/"batch_cost.json").exists()]
    for tier in ("gpu", "host"):
        field = tier + "_gib_seconds"
        result[field] = sum(batch[field] for batch in batches) if batches else None
        peak = tier + "_peak_gib"
        result[peak] = max((batch[peak] for batch in batches), default=None)
    token_fields = (result["total_input_tokens"], result["total_output_tokens"])
    result["online_tokens"] = sum(token_fields) if all(v is not None for v in token_fields) else None
    result["token_accounting"] = "all_online_models"
    for name in ("gpu_capacity_tokens", "host_capacity_tokens", "gpu_peak_tokens", "host_peak_tokens",
                 ):
        values = [_number(row.get("cost", {}).get(name)) for row in rows if row["status"] != "blocked"]
        result[name] = max(values) if values and all(value is not None for value in values) else None
    return result


def declared_queue_summary(rows, spec, task_ids):
    """Delivered quality over the declared queue, with all paid requests retained.

    Confirmed failures and blocked requests deliver zero quality. Missing
    execution and missing scores retain their possible range.
    This queue measure accompanies the original benchmark and scored-subset means.
    """
    expected = set(map(str, task_ids))
    selected = [row for row in rows if str(row["task_id"]) in expected]
    by_id = {str(row["task_id"]): row for row in selected}
    if len(by_id) != len(selected):
        raise ValueError("a declared queue requires one saved execution per task")
    completed = [row for row in selected if row["status"] == "completed"]
    scored = [row for row in completed if _scored(row, spec)]
    unknown = [row for row in completed if not _scored(row, spec)]
    missing_execution = expected-by_id.keys()
    unresolved = [str(row["task_id"]) for row in selected if row["status"] not in {"completed", "failed", "blocked"}]
    unknown_execution_count = len(missing_execution)+len(unresolved)
    maximum = spec.get("normalization", 1.0)
    components = {}
    for name in _required(spec):
        values = [_metric(row, name, spec)[0] for row in completed]
        observed = [value for value in values if value is not None]
        missing = len(values)-len(observed)
        components[name] = {"observed_count": len(observed), "unscored_answers": missing,
            "bounds": [(sum(observed)+fill*(missing+unknown_execution_count))/len(expected)
                       for fill in (0, maximum)] if expected else [None, None]}

    def aggregate(index):
        values = [value["bounds"][index] for value in components.values()]
        if not expected:
            return None
        return _primary_components(values, spec)

    benchmark_metrics = {}
    for name in sorted({name for row in completed for name in row.get("metrics", {})}):
        values = [_number(row.get("metrics", {}).get(name)) for row in completed]
        values = [value for value in values if value is not None]
        if values:
            benchmark_metrics[name] = {"mean": statistics.fmean(values), "scored": len(values)}
    cost = _cost_summary(selected)
    if missing_execution:
        for name in ("total_input_tokens", "total_output_tokens", "online_compute_ms",
                     "actual_prefill_tokens", "actual_device_hit_tokens", "actual_host_loaded_tokens"):
            cost[name] = None
        cost["online_tokens"] = None
    cost["p95_delivered_ms"] = cost["p95_complete_ms"]
    timings = sorted(row["cost"]["complete_ms"] for row in completed
                     if _number(row.get("cost", {}).get("complete_ms")) is not None)
    position = (len(expected)-1)*.95
    right = math.ceil(position)
    cost["p95_complete_ms"] = (timings[math.floor(position)]
        +(timings[right]-timings[math.floor(position)])*(position-math.floor(position))
        if expected and len(timings) == len(completed) and right < len(timings) else None)
    return {"declared": len(expected), "recorded": len(selected), "completed": len(completed),
        "scored": len(scored), "completion_rate": len(completed)/len(expected) if expected else None,
        "missing_execution_task_ids": sorted(missing_execution), "unresolved_task_ids": sorted(unresolved),
        "failed_task_ids": [str(row["task_id"]) for row in selected if row["status"] == "failed"],
        "blocked_task_ids": [str(row["task_id"]) for row in selected if row["status"] == "blocked"],
        "unscored_answer_task_ids": [str(row["task_id"]) for row in unknown],
        "queue_primary_score": aggregate(0) if not unknown and not unknown_execution_count else None,
        "queue_score_bounds": [aggregate(0), aggregate(1)], "components": components,
        "scored_subset_primary": _primary(scored, spec),
        "benchmark_metrics": benchmark_metrics,
        "cost": cost,
        "quality_rule": "declared task denominator; confirmed non-delivery is zero; missing outcomes and scores remain unknown",
        "cost_rule": "all saved requests including failures; incomplete totals retain known lower bounds"}
