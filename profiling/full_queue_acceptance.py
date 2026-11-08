"""Acceptance from predeclared, actual P3 queue pairs."""
from collections import defaultdict
import json
import math
import random
import statistics
from pathlib import Path

import yaml

from bench.study_metrics import _interval, declared_queue_quality_interval


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _gate_cell(cell, policy):
    reasons = []
    tracks, windows = cell["datasets"], cell["online_windows"]
    if (cell["requested_qa"] != policy["declared_qa"]
            or sum(track["declared"] for track in tracks.values()) != policy["declared_qa"]
            or cell["unexpected_execution_task_ids"]):
        reasons.append("declared_cohort_mismatch")
    if any(track["missing_execution_task_ids"] or track["unresolved_task_ids"] for track in tracks.values()):
        reasons.append("unresolved_execution")
    if not _number(cell["queue_quality"]) or set(cell["scorer_contracts"]) != set(tracks):
        reasons.append("incomplete_declared_quality")
    if cell["execution_status"] != "completed":
        reasons.append("workflow_job_did_not_finish")
    occupancy = cell.get("online_occupancy", {})
    if not all(occupancy.get(name) for name in ("hardware_only", "physical_coverage", "native_coverage")):
        reasons.append("incomplete_global_hardware_measurement")
    resource = cell["comparison_identity"]["resource_scope"]
    runtime = cell.get("reader_runtime", {})
    if (resource["hardware"] is None or resource["native_cache_config"] is None
            or not runtime.get("fingerprint") or not runtime.get("sglang_commit")):
        reasons.append("missing_actual_reader_or_cache_identity")
    paid = cell["paid_native_usage"]
    ledger = paid["online_token_ledger"]
    if ledger["reasons"] or (cell["completed_qa"] and not ledger["provider_phases"]):
        reasons.append("incomplete_online_token_ledger")
    if any(paid["native_work"][name]["unknown_calls"]
           for name in ("actual_prefill_tokens", "actual_host_loaded_tokens", "output_tokens")):
        reasons.append("incomplete_native_work")

    pools = [row["engine_pool"] for group in windows.values() for row in group]
    measured = {"max_online_tokens": ledger["total_tokens"], "p95_complete_ms": cell["p95_complete_ms"]}
    for tier in ("gpu", "host"):
        key = tier+"_capacity_tokens"
        capacities = {row[key] for row in pools}
        measured[key] = max(capacities) if capacities else None
        if capacities and capacities != {resource[key]}:
            reasons.append("actual_"+tier+"_capacity_differs_from_run_config")
        measured["max_"+tier+"_gib"] = occupancy.get(tier+"_peak_gib")
        measured["max_"+tier+"_gib_seconds"] = occupancy.get(tier+"_gib_seconds")
    budget = {name: {"limit": limit, "observed": measured[name],
                     "pass": measured[name] <= limit if _number(measured[name]) else None}
              for name, limit in policy["resource_limits"].items()}
    passed = [limit["pass"] for limit in budget.values()]
    return {"measurement_complete": not reasons,
            "resource_budget_pass": False if False in passed else None if None in passed else True,
            "measurement_reasons": reasons, "resource_limits": budget}


def _pair_contract(cell):
    reader = dict(cell.get("reader_runtime") or {})
    if reader.get("model_scope") is not None:
        reader["model_scope"] = {name: value for name, value in reader["model_scope"].items() if name != "source_path"}
    return (reader, cell["scorer_contracts"],
            {name: sorted({row["resident_model_digest"] for row in group})
             for name, group in cell["online_windows"].items()})


def _gate(point_met, interval_met, eligible, missing):
    if not eligible or point_met is None:
        return "not_evaluable"
    if not point_met:
        return "target_not_met"
    if interval_met:
        return "met"
    return missing


def evaluate_full_queue(cells):
    groups = defaultdict(list)
    for cell in cells:
        if cell.get("acceptance_protocol") is not None:
            key = json.dumps([cell["concurrency"], cell["comparison_identity"], cell["acceptance_protocol"]], sort_keys=True)
            groups[key].append(cell)
    reports = []
    for group in groups.values():
        first = group[0]
        policy = first["acceptance_protocol"]
        methods = (policy["baseline"], policy["candidate"])
        slots = defaultdict(lambda: defaultdict(list))
        cell_gates = {cell["generation_dir"]: _gate_cell(cell, policy) for cell in group}
        for cell in group:
            for repetition in cell["planned_repetitions"]:
                slots[cell["replicate_scope"], repetition]
            if cell["method"] in methods:
                slots[cell["replicate_scope"], cell["repetition"]][cell["method"]].append(cell)
        pair_issues, pairs, contracts = [], [], set()
        for slot, arms in slots.items():
            counts = {method: len(arms[method]) for method in methods}
            if set(counts.values()) != {1}:
                pair_issues.append({"replicate": slot, "method_counts": counts, "reason": "missing_or_ambiguous_queue"})
                continue
            left, right = (arms[method][0] for method in methods)
            contract = _pair_contract(left)
            if contract != _pair_contract(right):
                pair_issues.append({"replicate": slot, "reason": "reader_scorer_or_resident_A_differs"})
                continue
            assets = [{name: item["sha256"] for name, item in cell["active_assets"].items()}
                      for cell in (left, right)]
            contracts.add(json.dumps((contract, assets), sort_keys=True))
            pairs.append((left, right))
        if len(contracts) > 1:
            pair_issues.append({"reason": "reader_scorer_or_assets_differ_across_replicates"})
            pairs.clear()

        measurement_complete = bool(pairs) and not pair_issues and all(
            cell_gates[cell["generation_dir"]]["measurement_complete"] for pair in pairs for cell in pair)
        resource_results = [cell_gates[cell["generation_dir"]]["resource_budget_pass"]
                            for pair in pairs for cell in pair]
        resource_pass = (False if False in resource_results else
                         True if resource_results and not pair_issues and all(value is True for value in resource_results) else None)
        eligible = measurement_complete and resource_pass is True
        bounds, quality, ratios = [], [], []
        for left, right in pairs:
            lower, upper = right["queue_quality_bounds"]
            baseline_lower, baseline_upper = left["queue_quality_bounds"]
            bounds.append((lower-baseline_upper, upper-baseline_lower))
            if _number(left["queue_quality"]) and _number(right["queue_quality"]):
                quality.append(right["queue_quality"]-left["queue_quality"])
            if (left["full_queue_delivered"] and right["full_queue_delivered"]
                    and _number(left["p95_complete_ms"]) and left["p95_complete_ms"] > 0
                    and _number(right["p95_complete_ms"])):
                ratios.append(right["p95_complete_ms"]/left["p95_complete_ms"])
        quality_point = statistics.fmean(quality) if quality and len(quality) == len(pairs) else None
        latency_point = statistics.fmean(ratios) if ratios and len(ratios) == len(pairs) else None
        quality_ci, latency_ci = None, None
        if quality_point is not None:
            from bench.capacity_workflow import declared_execution_rows

            matrix = yaml.safe_load(Path(pairs[0][0]["queue_job"]["matrix"]).read_text())
            quality_ci = declared_queue_quality_interval(matrix, [
                tuple(declared_execution_rows(cell["queue_job"])[0] for cell in pair) for pair in pairs])
        if latency_point is not None and len(ratios) >= 2:
            rng = random.Random(policy["seed"])
            draws = [statistics.fmean(rng.choices(ratios, k=len(ratios))) for _ in range(policy["bootstrap_samples"])]
            latency_ci = _interval(draws, policy["confidence"])
        q_lower = quality_ci["interval"][0] if quality_ci is not None else None
        quality_target = policy["quality_improvement"]
        noninferiority = -policy["quality_noninferiority"]
        latency_target = policy["p95_ratio"]
        quality_gate = _gate(None if quality_point is None else quality_point >= quality_target,
            q_lower is not None and q_lower >= quality_target, eligible, "point_met_interval_inconclusive")
        latency_gate = _gate(None if quality_point is None or latency_point is None else
            quality_point >= noninferiority and latency_point <= latency_target,
            q_lower is not None and q_lower >= noninferiority and latency_ci is not None
            and latency_ci[1] <= latency_target, eligible,
            "point_met_single_queue_descriptive" if latency_ci is None else "point_met_interval_inconclusive")
        required_cells = {label: any(cell["method"] == method and cell_gates[cell["generation_dir"]]["measurement_complete"]
                                     for cell in group)
                          for label, method in first["required_experiment_cells"].items()}
        reports.append({"concurrency": first["concurrency"], "comparison_identity": first["comparison_identity"],
            "policy": policy, "required_cells_measured": required_cells, "declared_pair_slots": len(slots),
            "paired_queues": len(pairs), "pair_issues": pair_issues, "measurement_complete": measurement_complete,
            "resource_budget_pass": resource_pass, "quality_gate": quality_gate, "latency_gate": latency_gate,
            "quality_delta": quality_point, "quality_difference_bounds": list(map(statistics.fmean, zip(*bounds))) if bounds else None,
            "quality_confidence": quality_ci, "p95_ratio": latency_point, "p95_ratio_interval": latency_ci,
            "p95_resampling_unit": "independent_paired_full_arrival_queue" if latency_ci is not None else None,
            "full_delivery_pairs": len(ratios),
            "cells": cell_gates})
    return reports
