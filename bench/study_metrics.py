"""Continuous benchmark quality, complete-cohort selection and paired intervals."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics


MTRAGEVAL_COMPONENTS = ("RL_F", "RB_llm", "RB_alg")
RB_AGG_ALIAS_SOURCE = "data/mtrag-official/scripts/evaluation/run_algorithmic.py:184"
MTRAGEVAL_RANKING_SOURCE = "https://ibm.github.io/mt-rag-benchmark/MTRAGEval/#-leaderboard-ranking"


def _track_specs(matrix, split):
    development = matrix.get("selection", {}).get("development_weights")
    if split == "dev" and development is not None:
        return {name: {**matrix["datasets"][name], "weight": weight} for name, weight in development.items()}
    return matrix["datasets"]


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
    source = spec.get("metric_sources", {}).get(name, name)
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
    for name in ("total_input_tokens", "total_output_tokens", "gpu_gib_seconds", "host_gib_seconds",
                 "online_compute_ms", "auxiliary_input_tokens", "auxiliary_output_tokens"):
        result[name] = _cost_total(rows, name)
    token_fields = (result["total_input_tokens"], result["total_output_tokens"])
    result["online_tokens"] = sum(token_fields) if all(v is not None for v in token_fields) else None
    result["token_accounting"] = "all_online_models"
    for name in ("gpu_capacity_tokens", "host_capacity_tokens", "gpu_peak_tokens", "host_peak_tokens",
                 "gpu_peak_gib", "host_peak_gib"):
        values = [_number(row.get("cost", {}).get(name)) for row in rows]
        result[name] = max(values) if values and all(value is not None for value in values) else None
    return result


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
    """Retain raw metrics and compute Q only from every configured benchmark."""
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
                "alias_provenance": {"RB_agg": RB_AGG_ALIAS_SOURCE} if "RB_agg" in aliases else {},
                "ranking_definition": MTRAGEVAL_RANKING_SOURCE if spec["primary_metric"] == "mtrageval_harmonic" else None}
            if not declared:
                reasons.append("cohort_not_declared:"+name)
            if not complete:
                reasons.append("incomplete_cohort_or_scoring:"+name)
        cost = _cost_summary(method_rows)
        reasons.extend(_budget_reasons(matrix, cost))
        observed_quality = sum(spec["weight"]*datasets[name]["normalized_score"]
                      for name, spec in specs.items()) if all(
            datasets[name]["normalized_score"] is not None for name in specs) else None
        quality = observed_quality if all(dataset["complete"] for dataset in datasets.values()) else None
        summaries[method] = {"method": method, "split": split, "quality": quality,
            "observed_quality": observed_quality,
            "quality_track": "development_selection" if split == "dev" else "final_evaluation",
            "track_weights": {name: spec["weight"] for name, spec in specs.items()},
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


def _interval(values, confidence):
    alpha = (1-confidence)/2
    return [_quantile(values, alpha), _quantile(values, 1-alpha)]


def paired_comparisons(matrix, rows, baseline="B0", split="test", *, samples=None, seed=None):
    """Resample whole conversations/documents, retaining paired actual turns."""
    selection = matrix.get("selection", {})
    samples = samples if samples is not None else selection.get("bootstrap_samples", 5000)
    rng = random.Random(matrix.get("seed", 42) if seed is None else seed)
    confidence = selection.get("confidence", .95)
    rows = [row for row in rows if row["split"] == split]
    specs = _track_specs(matrix, split)
    methods = sorted({row["method"] for row in rows}-{baseline})
    output = []
    for method in methods:
        overall = []
        for dataset, spec in specs.items():
            left = {str(row["task_id"]): row for row in rows if row["method"] == baseline
                    and row["dataset"] == dataset and _scored(row, spec)}
            right = {str(row["task_id"]): row for row in rows if row["method"] == method
                     and row["dataset"] == dataset and _scored(row, spec)}
            common = sorted(left.keys() & right.keys())
            if not common:
                continue
            clusters = defaultdict(list)
            for task_id in common:
                row = left[task_id]
                identity = str(row.get("document_id", row["conversation_id"]))
                clusters[identity].append(task_id)
            keys = sorted(clusters)
            estimated = _primary([right[key] for key in common], spec)-_primary([left[key] for key in common], spec)
            draws, cost_draws = [], {name: [] for name in (
                "complete_ms", "gpu_gib_seconds", "host_gib_seconds", "online_compute_ms")}
            for _ in range(samples):
                sampled = [task_id for cluster in rng.choices(keys, k=len(keys)) for task_id in clusters[cluster]]
                draws.append(_primary([right[key] for key in sampled], spec)
                             - _primary([left[key] for key in sampled], spec))
                for name in cost_draws:
                    pairs = [(right[key].get("cost", {}).get(name), left[key].get("cost", {}).get(name)) for key in sampled]
                    if all(_number(a) is not None and _number(b) is not None for a, b in pairs):
                        cost_draws[name].append(statistics.fmean(a-b for a, b in pairs))
            cost_effects = {}
            for name, cost_values in cost_draws.items():
                pairs = [(right[key].get("cost", {}).get(name), left[key].get("cost", {}).get(name)) for key in common]
                cost_effects[name] = {"mean_difference": statistics.fmean(a-b for a, b in pairs),
                    "interval": _interval(cost_values, confidence) if len(keys) >= 2 else None} if len(cost_values) == samples else None
            result = {"method": method, "baseline": baseline, "dataset": dataset,
                "quality_track": "development_selection" if split == "dev" else "final_evaluation",
                "paired_tasks": len(common), "clusters": len(keys), "score_difference": estimated,
                "score_interval": _interval(draws, confidence) if len(keys) >= 2 else None,
                "confidence": confidence,
                "bootstrap_samples": samples, "resampling_unit": "conversation_or_document",
                "cost_differences": cost_effects,
                "unpaired_baseline_tasks": sorted(left.keys()-right.keys()),
                "unpaired_method_tasks": sorted(right.keys()-left.keys())}
            tolerance = selection.get("equal_cost_relative_tolerance", .05)
            matched = []
            for name in ("complete_ms", "online_compute_ms"):
                if cost_effects[name] is None:
                    matched.append(None)
                    continue
                baseline_mean = statistics.fmean(left[key]["cost"][name] for key in common)
                method_mean = statistics.fmean(right[key]["cost"][name] for key in common)
                matched.append(abs(method_mean-baseline_mean) <= tolerance*baseline_mean)
            result["equal_cost_match"] = all(matched) if None not in matched else None
            result["equal_cost_relative_tolerance"] = tolerance
            output.append(result)
            weight = spec["weight"]/spec.get("normalization", 1.0)
            overall.append((dataset, estimated*weight, [draw*weight for draw in draws], len(keys)))
        if len(overall) == len(specs):
            draws = [sum(component[2][index] for component in overall) for index in range(samples)]
            output.append({"method": method, "baseline": baseline, "dataset": "engineering_Q",
                "quality_track": "development_selection" if split == "dev" else "final_evaluation",
                "track_weights": {name: spec["weight"] for name, spec in specs.items()},
                "score_difference": sum(component[1] for component in overall),
                "score_interval": _interval(draws, confidence) if all(component[3] >= 2 for component in overall) else None,
                "confidence": confidence,
                "bootstrap_samples": samples, "resampling_unit": "independent_dataset_conversation_or_document"})
    return output
