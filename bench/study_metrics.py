"""Benchmark quality, budgeted selection and method comparisons."""
from __future__ import annotations

from collections import Counter
import math
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


def evaluation_track_protocol(spec, split):
    """Report the registered source split separately from the metric's scale."""
    origins = {
        "native": {"dev": "public_benchmark", "test": "public_benchmark"},
        "project_conversation": {"dev": "public_benchmark", "test": "public_benchmark"},
        "project_selected_turn_reference_diagnostic": {
            "dev": "public_reference_diagnostic", "test": "public_reference_diagnostic"},
        "public_test": {"test": "public_benchmark"},
        "constructed": {"dev": "constructed_task", "test": "constructed_task"},
    }
    resolved = split_metric_spec(spec, split)
    return {"origin": origins[spec["split_kind"]][split], "split_kind": spec["split_kind"],
            "primary_metric": resolved["primary_metric"], "normalization": resolved["normalization"],
            "history_protocol": spec.get("protocol", "P3")}


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
    accounting = {}
    for name in ("total_input_tokens", "total_output_tokens", "gpu_gib_seconds", "host_gib_seconds",
                 "online_compute_ms", "auxiliary_input_tokens", "auxiliary_output_tokens",
                 "reader_input_tokens", "reader_output_tokens", "actual_prefill_tokens",
                 "actual_device_hit_tokens", "actual_host_loaded_tokens"):
        result[name] = _cost_total(rows, name)
        unknown = [row for row in rows if _number(row.get("cost", {}).get(name)) is None]
        accounting[name] = {"unknown_task_ids": [str(row["task_id"]) for row in unknown],
            "known_lower_bound": sum((_number(row.get("cost", {}).get(name))
                if _number(row.get("cost", {}).get(name)) is not None else
                _number(row.get("cost", {}).get("accounting", {}).get("known_lower_bound", {}).get(name)) or 0)
                for row in rows)}
    token_fields = (result["total_input_tokens"], result["total_output_tokens"])
    result["online_tokens"] = sum(token_fields) if all(v is not None for v in token_fields) else None
    result["token_accounting"] = "all_online_models"
    result["accounting"] = accounting
    for name in ("gpu_capacity_tokens", "host_capacity_tokens", "gpu_peak_tokens", "host_peak_tokens",
                 "gpu_peak_gib", "host_peak_gib"):
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
        for name, accounting in cost["accounting"].items():
            accounting["unknown_task_ids"].extend(sorted(missing_execution))
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
        "cost_rule": "all saved requests including failures; observed totals retain unmeasured task IDs"}


def _budget_reasons(matrix, cost):
    limits = matrix.get("resources", {})
    reasons = []
    for name in ("mean_complete_ms", "p95_complete_ms", "total_input_tokens", "total_output_tokens",
                 "gpu_gib_seconds", "host_gib_seconds"):
        if cost[name] is None:
            reasons.append("missing_cost:"+name)
    observation = matrix.get("resource_observation", {})
    bounded = {"p95_complete_ms": "p95_complete_ms", "max_online_tokens": "online_tokens",
        "max_online_compute_ms": "online_compute_ms", "max_gpu_gib_seconds": "gpu_gib_seconds",
        "max_host_gib_seconds": "host_gib_seconds", "max_gpu_gib": "gpu_peak_gib",
        "max_host_gib": "host_peak_gib", "gpu_capacity_tokens": "gpu_capacity_tokens",
        "host_capacity_tokens": "host_capacity_tokens"}
    for limit, name in bounded.items():
        if limit not in limits:
            continue
        value = cost[name] if cost[name] is not None else observation.get(name)
        if value is None:
            reasons.append("missing_resource_measurement:"+name)
        elif value > limits[limit]:
            reasons.append("budget_exceeded:"+limit)
    return reasons


def summarize_methods(matrix, rows, split="dev"):
    """Report each benchmark in its original metric; training may declare an objective."""
    rows = [row for row in rows if row["split"] == split]
    methods = [method["id"] for method in matrix["methods"]]
    methods.extend(sorted({row["method"] for row in rows}-set(methods)))
    cohorts = matrix.get("cohorts", {}).get(split, {})
    specs = _track_specs(matrix, split)
    summaries = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method and row["dataset"] in specs]
        datasets, reasons = {}, []
        for name, spec in specs.items():
            observed = [row for row in method_rows if row["dataset"] == name]
            counts = Counter(str(row["task_id"]) for row in observed)
            duplicate_ids = sorted(key for key, count in counts.items() if count > 1)
            declared = name in cohorts
            expected = {str(task_id) for task_id in cohorts[name]} if declared else set(counts)
            selected = [row for row in observed if str(row["task_id"]) in expected]
            scored = [row for row in selected if _scored(row, spec)]
            complete = (declared and bool(expected) and set(counts) == expected and not duplicate_ids
                        and len(scored) == len(expected))
            metric_names = sorted({key for row in selected for key in row["metrics"]})
            raw = {}
            for metric in metric_names:
                values = [_number(row["metrics"].get(metric)) for row in selected]
                values = [value for value in values if value is not None]
                raw[metric] = {"mean": statistics.fmean(values) if values else None, "count": len(values)}
            primary = _primary(scored, spec)
            aliases = Counter(source for row in scored for metric in _required(spec)
                              for _, source in [_metric(row, metric, spec)] if source != metric)
            datasets[name] = {"planned": len(expected) if declared else None, "observed": len(observed),
                "completed": sum(row["status"] == "completed" for row in selected), "scored": len(scored),
                "cohort_declared": declared, "complete": complete,
                "missing_task_ids": sorted(expected-set(counts)), "extra_task_ids": sorted(set(counts)-expected),
                "duplicate_task_ids": duplicate_ids, "raw_metrics": raw,
                "primary_metric": spec["primary_metric"], "primary_score": primary,
                "normalized_score": primary/spec.get("normalization", 1.0) if primary is not None else None,
                "metric_source_alias_counts": dict(aliases),
                "alias_provenance": {source: RB_AGG_ALIAS_SOURCE for source in aliases
                                     if source.startswith("RB_agg")},
                "ranking_definition": MTRAGEVAL_RANKING_SOURCE if spec["primary_metric"] == "mtrageval_harmonic" else None}
            if not declared:
                reasons.append("cohort_not_declared:"+name)
            if not complete:
                reasons.append("incomplete_cohort_or_scoring:"+name)
        cost = _cost_summary(method_rows)
        reasons.extend(_budget_reasons(matrix, cost))
        weights = (matrix.get("selection", {}).get("development_weights") if split == "dev" else None)
        if weights:
            observed_quality = (sum(weight*datasets[name]["normalized_score"] for name, weight in weights.items())
                if all(datasets[name]["normalized_score"] is not None for name in weights) else None)
        else:
            observed_quality = next(iter(datasets.values()))["normalized_score"] if len(datasets) == 1 else None
        quality = observed_quality if all(dataset["complete"] for dataset in datasets.values()) else None
        summaries[method] = {"method": method, "split": split, "quality": quality,
            "observed_quality": observed_quality,
            "quality_track": "development_selection" if split == "dev" else "final_evaluation",
            "track_weights": weights,
            "excluded_datasets": sorted(set(matrix["datasets"])-set(specs)),
            "datasets": datasets, "cost": cost, "eligible": not reasons,
            "selection_reasons": reasons}
    return summaries


def select_development(matrix, rows):
    summaries = summarize_methods(matrix, rows, "dev")
    eligible = [row for row in summaries.values() if row["eligible"]]
    if not eligible:
        raise ValueError({method: summary["selection_reasons"] for method, summary in summaries.items()})
    best_quality = max(row["quality"] for row in eligible)
    threshold = matrix.get("selection", {}).get("near_equal_quality", 0.0)
    near = [row for row in eligible if best_quality-row["quality"] <= threshold]
    selected = min(near, key=lambda row: (row["cost"]["mean_complete_ms"],
        row["cost"]["gpu_gib_seconds"], row["cost"]["host_gib_seconds"], -row["quality"], row["method"]))
    return {**selected, "best_quality": best_quality, "near_equal_quality": threshold,
        "near_equal_methods": [row["method"] for row in near], "methods": summaries,
        "selection_objective": "continuous_quality_with_declared_resource_budgets"}


def paired_comparisons(matrix, rows, baseline=None, split="test"):
    """Point comparisons using each benchmark's declared task denominator."""
    baseline = baseline or matrix["selection"]["baseline"]
    result = []
    for dataset, spec in _track_specs(matrix, split).items():
        expected = matrix["cohorts"][split][dataset]
        selected = [row for row in rows if row["dataset"] == dataset and row["split"] == split]
        methods = {row["method"] for row in selected}
        if baseline not in methods:
            continue
        left = declared_queue_summary([row for row in selected if row["method"] == baseline], spec, expected)
        for method in sorted(methods-{baseline}):
            right = declared_queue_summary([row for row in selected if row["method"] == method], spec, expected)
            quality = (right["queue_primary_score"]-left["queue_primary_score"]
                       if right["queue_primary_score"] is not None and left["queue_primary_score"] is not None else None)
            a, b = left["cost"]["p95_complete_ms"], right["cost"]["p95_complete_ms"]
            result.append({"dataset": dataset, "method": method, "baseline": baseline,
                "primary_metric": spec["primary_metric"], "score_difference": quality,
                "p95_reduction_percent": (1-b/a)*100 if a and b is not None else None,
                "baseline_completed": left["completed"], "candidate_completed": right["completed"],
                "declared": len(expected)})
    return result
