"""Continuous benchmark quality, complete-cohort selection and paired intervals."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics


MTRAGEVAL_COMPONENTS = ("RL_F", "RB_llm", "RB_alg")
RB_AGG_ALIAS_SOURCE = "data/mtrag-official/scripts/evaluation/run_algorithmic.py:184"
MTRAGEVAL_RANKING_SOURCE = "https://ibm.github.io/mt-rag-benchmark/MTRAGEval/#-leaderboard-ranking"


def split_metric_spec(spec, split):
    """Resolve a preregistered split's metric without changing its task cohort."""
    override = spec.get("metric_by_split", {}).get(split, {})
    return {**spec, **override,
        "scoring": {**spec.get("scoring", {}), **override.get("scoring", {})}}


def _track_specs(matrix, split):
    development = matrix.get("selection", {}).get("development_weights")
    if split == "dev" and development is not None:
        return {name: {**split_metric_spec(matrix["datasets"][name], split), "weight": weight}
                for name, weight in development.items()}
    return {name: split_metric_spec(spec, split) for name, spec in matrix["datasets"].items()}


def evaluation_track_protocol(spec, split):
    """Report the registered source split separately from the metric's scale."""
    origins = {
        "native": {"dev": "public_benchmark", "test": "public_benchmark"},
        "project_conversation": {"dev": "public_benchmark", "test": "public_benchmark"},
        "project_selected_turn_reference_diagnostic": {
            "dev": "public_reference_diagnostic", "test": "public_reference_diagnostic"},
        "independent_family_dev_and_public_test": {"dev": "constructed_task", "test": "public_benchmark"},
        "independent_training_sources_dev_and_public_test": {"dev": "constructed_task", "test": "public_benchmark"},
    }
    resolved = split_metric_spec(spec, split)
    return {"origin": origins[spec["split_kind"]][split], "split_kind": spec["split_kind"],
            "primary_metric": resolved["primary_metric"], "normalization": resolved["normalization"],
            "registered_weight": resolved["weight"]}


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

    paid = {}
    for name in ("total_input_tokens", "total_output_tokens", "gpu_gib_seconds",
                 "host_gib_seconds", "online_compute_ms", "auxiliary_input_tokens",
                 "auxiliary_output_tokens"):
        observed = [(row, _number(row.get("cost", {}).get(name))) for row in selected]
        measured = [(row, value) for row, value in observed if value is not None]
        paid[name] = {"observed_total": sum(value for _, value in measured),
            "failed_request_total": sum(value for row, value in measured if row["status"] != "completed"),
            "measured_requests": len(measured),
            "unmeasured_task_ids": [str(row["task_id"]) for row, value in observed if value is None]}
    return {"declared": len(expected), "recorded": len(selected), "completed": len(completed),
        "scored": len(scored), "completion_rate": len(completed)/len(expected) if expected else None,
        "missing_execution_task_ids": sorted(missing_execution), "unresolved_task_ids": sorted(unresolved),
        "failed_task_ids": [str(row["task_id"]) for row in selected if row["status"] == "failed"],
        "blocked_task_ids": [str(row["task_id"]) for row in selected if row["status"] == "blocked"],
        "unscored_answer_task_ids": [str(row["task_id"]) for row in unknown],
        "queue_primary_score": aggregate(0) if not unknown and not unknown_execution_count else None,
        "queue_score_bounds": [aggregate(0), aggregate(1)], "components": components,
        "scored_subset_primary": _primary(scored, spec), "cost": _cost_summary(selected),
        "paid_cost": paid,
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
                "alias_provenance": {source: RB_AGG_ALIAS_SOURCE for source in aliases
                                     if source.startswith("RB_agg")},
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


def declared_queue_quality_interval(matrix, pairs, *, split="dev"):
    """Paired conversation bootstrap of the declared, fixed-weight quality difference."""
    specs = _track_specs(matrix, split)
    selection = matrix["selection"]
    samples, confidence = selection.get("bootstrap_samples", 5000), selection.get("confidence", .95)
    rng = random.Random(matrix["seed"])
    arms = [{(row["dataset"], str(row["task_id"])): row for row in rows} for pair in pairs for rows in pair]
    draws, cluster_counts = [0.0]*samples, {}
    for dataset, spec in specs.items():
        components = _required(spec)
        width = len(components)
        clusters = defaultdict(lambda: [0]*(1+len(arms)*width))
        for task_id in matrix["cohorts"][split][dataset]:
            key = dataset, str(task_id)
            if any(key not in arm for arm in arms):
                return None
            rows = [arm[key] for arm in arms]
            conversation = rows[0]["conversation_id"]
            if not conversation or any(row["conversation_id"] != conversation for row in rows):
                raise ValueError("paired declared tasks must belong to the same conversation")
            cluster = clusters[conversation]
            cluster[0] += 1
            for arm, row in enumerate(rows):
                if row["status"] in {"failed", "blocked"}:
                    continue
                if not _scored(row, spec):
                    return None
                for i, name in enumerate(components, 1+arm*width):
                    cluster[i] += _metric(row, name, spec)[0]
        cluster_counts[dataset] = len(clusters)
        if len(clusters) < 2:
            return None
        populations = [clusters[key] for key in sorted(clusters)]
        weight = spec["weight"]/spec["normalization"]
        for draw in range(samples):
            n, *scores = map(sum, zip(*rng.choices(populations, k=len(populations))))
            quality = [_primary_components([value/n for value in scores[start:start+width]], spec)
                       for start in range(0, len(scores), width)]
            draws[draw] += weight*statistics.fmean(right-left for left, right in zip(quality[::2], quality[1::2]))
    return {"interval": _interval(draws, confidence), "confidence": confidence,
            "bootstrap_samples": samples, "clusters_by_dataset": cluster_counts,
            "resampling_unit": "paired_conversation_within_fixed_dataset"}


def paired_comparisons(matrix, rows, baseline=None, split="test", *, samples=None, seed=None):
    """Resample whole conversations/documents, retaining paired actual turns."""
    selection = matrix.get("selection", {})
    baseline = baseline if baseline is not None else selection.get("baseline", matrix["methods"][0]["id"])
    samples = samples if samples is not None else selection.get("bootstrap_samples", 5000)
    rng = random.Random(matrix.get("seed", 42) if seed is None else seed)
    confidence = selection.get("confidence", .95)
    rows = [row for row in rows if row["split"] == split]
    specs = _track_specs(matrix, split)
    cohorts = matrix.get("cohorts", {}).get(split, {})
    methods = sorted({row["method"] for row in rows}-{baseline})
    output = []
    for method in methods:
        overall = []
        for dataset, spec in specs.items():
            declared = dataset in cohorts
            expected = ({str(task_id) for task_id in cohorts[dataset]} if declared else {
                str(row["task_id"]) for row in rows
                if row["dataset"] == dataset and row["method"] in {baseline, method}})
            left = {str(row["task_id"]): row for row in rows if row["method"] == baseline
                    and row["dataset"] == dataset and str(row["task_id"]) in expected and _scored(row, spec)}
            right = {str(row["task_id"]): row for row in rows if row["method"] == method
                     and row["dataset"] == dataset and str(row["task_id"]) in expected and _scored(row, spec)}
            common = sorted(left.keys() & right.keys())
            complete_cohort = declared and bool(expected) and set(common) == expected
            coverage = {"cohort_declared": declared, "planned_tasks": len(expected) if declared else None,
                "paired_tasks": len(common), "paired_task_ids": common, "complete_cohort": complete_cohort,
                "comparison_scope": "declared_cohort" if complete_cohort else "common_scored_subset",
                "baseline_unscored_task_ids": sorted(expected-left.keys()),
                "method_unscored_task_ids": sorted(expected-right.keys()),
                "both_unscored_task_ids": sorted(expected-(left.keys() | right.keys()))}
            if not common:
                output.append({"method": method, "baseline": baseline, "dataset": dataset,
                    **coverage, "score_difference": None, "score_interval": None})
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
                **coverage, "clusters": len(keys), "score_difference": estimated,
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
            overall.append((dataset, estimated*weight, [draw*weight for draw in draws], len(keys), complete_cohort))
        if len(overall) == len(specs):
            draws = [sum(component[2][index] for component in overall) for index in range(samples)]
            complete_cohort = all(component[4] for component in overall)
            output.append({"method": method, "baseline": baseline,
                "dataset": "engineering_Q" if complete_cohort else "engineering_Q_common_scored",
                "complete_cohort": complete_cohort,
                "comparison_scope": "declared_cohort" if complete_cohort else "common_scored_subset",
                "quality_track": "development_selection" if split == "dev" else "final_evaluation",
                "track_weights": {name: spec["weight"] for name, spec in specs.items()},
                "score_difference": sum(component[1] for component in overall),
                "score_interval": _interval(draws, confidence) if all(component[3] >= 2 for component in overall) else None,
                "confidence": confidence,
                "bootstrap_samples": samples, "resampling_unit": "independent_dataset_conversation_or_document"})
    return output
