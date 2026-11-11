"""Read existing study artifacts into a final index and arithmetic comparisons.

No generation, judging, gold parsing, model loading, or remote operations occur.
Separate caller-labelled cohorts remain separate.  All quality numbers come
from existing numeric metric fields; missing scores keep their denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
import math
from pathlib import Path
import shlex
import statistics

import yaml


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def result_fields(value):
    if isinstance(value, dict):
        return {key: result_fields(item) for key, item in value.items()
                if key != "sha256" and not key.endswith(("_sha256", "_hash", "_digest", "_digests"))}
    if isinstance(value, list):
        return [result_fields(item) for item in value]
    return value


def load(path):
    path = Path(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return yaml.safe_load(path.read_text()) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text())


def compact(value, depth=3):
    if isinstance(value, dict):
        if depth == 0:
            return {"keys": list(value), "count": len(value)}
        return {str(key): compact(item, depth-1) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item, depth-1) for item in value] if len(value) <= 8 and depth > 0 else {"items": len(value)}
    if isinstance(value, str) and len(value) > 320:
        return value[:320]+"…"
    return value


def expanded(specs):
    paths, missing = [], []
    for spec in specs:
        label, pattern = spec.split("=", 1)
        found = sorted(glob.glob(pattern, recursive=True))
        paths.extend((label, Path(path).resolve()) for path in found)
        if not found:
            missing.append(dict(label=label, pattern=pattern))
    return list(dict.fromkeys(paths)), missing


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_fields(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix+str(key)
        if numeric(item):
            result[name] = float(item)
        elif isinstance(item, dict):
            result.update(metric_fields(item, name+"."))
    return result


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def mean(values):
    return statistics.fmean(values) if values else None


def total(rows, field):
    values = [row["cost"].get(field) for row in rows]
    return sum(values) if values and all(numeric(value) for value in values) else None


def import_records(specs):
    files, missing = expanded(specs)
    indexed, provenance, duplicate, conflicts, unsupported = {}, [], [], set(), []
    for cohort, path in files:
        provenance.append(dict(cohort=cohort, path=str(path)))
        payload = load(path)
        rows = payload if isinstance(payload, list) else payload.get("results", [payload])
        for ordinal, raw in enumerate(rows):
            if not isinstance(raw, dict) or not {"method", "task_id", "status"} <= raw.keys():
                unsupported.append(dict(path=str(path), ordinal=ordinal, reason="not an explicit scored/unscored study result record"))
                continue
            key = (cohort, raw["method"], raw.get("dataset"), raw.get("split"), raw["task_id"])
            row = dict(cohort=cohort, method=raw["method"], dataset=raw.get("dataset"), split=raw.get("split"),
                task_id=raw["task_id"], conversation_id=raw.get("conversation_id"), status=raw["status"],
                metrics=metric_fields(raw.get("metrics", {})), cost=raw.get("cost", {}),
                question=raw.get("question"), actual_history=raw.get("actual_history"),
                answer=raw.get("answer"),
                source_files=[str(path)])
            if key in indexed:
                old = indexed[key]
                agreement = (old["status"] == row["status"] and old["answer"] == row["answer"]
                    and all(old["metrics"][name] == row["metrics"][name] for name in old["metrics"].keys() & row["metrics"].keys()))
                if not agreement:
                    conflicts.add(key)
                    duplicate.append(dict(key=list(key), status="conflicting_versions_excluded_from_arithmetic", sources=old["source_files"]+row["source_files"]))
                    continue
                old["metrics"].update(row["metrics"])
                old["source_files"].extend(row["source_files"])
                duplicate.append(dict(key=list(key), status="same_answer_scored_fields_joined_once", sources=old["source_files"]))
            else:
                indexed[key] = row
    for key in conflicts:
        indexed.pop(key, None)
    return list(indexed.values()), dict(files=provenance, unmatched=missing, duplicate_records=duplicate, unsupported=unsupported)


def record_tables(rows, baselines):
    groups, summaries, comparisons = defaultdict(list), [], []
    for row in rows:
        groups[row["cohort"], row["dataset"], row["split"], row["method"]].append(row)
    for (cohort, dataset, split, method), values in groups.items():
        complete = [row for row in values if row["status"] in {"completed", "success"}]
        names = sorted({name for row in complete for name in row["metrics"]})
        timings = [row["cost"]["complete_ms"] for row in complete if numeric(row["cost"].get("complete_ms"))]
        summaries.append(dict(cohort=cohort, dataset=dataset, split=split, method=method,
            recorded_rows=len(values), completed_rows=len(complete), status_counts=dict(Counter(row["status"] for row in values)),
            missing_requested_records="read the linked frozen cohort/plan manifest; absent rows are not inferred successful",
            metrics={name: dict(mean=mean([row["metrics"][name] for row in complete if name in row["metrics"]]),
                scored_rows=sum(name in row["metrics"] for row in complete), completed_rows=len(complete)) for name in names},
            response_cost=dict(mean_complete_ms=mean(timings), p95_complete_ms=quantile(timings, .95), observed_timing_rows=len(timings),
                **{field: total(complete, field) for field in ("total_input_tokens", "total_output_tokens",
                    "actual_prefill_tokens", "actual_device_hit_tokens", "actual_host_loaded_tokens")}),
            physical_cost_scope="use the indexed batch/ledger physical integrals; request shares are not summed as full cost"))
        baseline = baselines.get(cohort, baselines.get("*"))
        reference = groups.get((cohort, dataset, split, baseline))
        if baseline is None or method == baseline or reference is None:
            continue
        left = {row["task_id"]: row for row in complete}
        right = {row["task_id"]: row for row in reference if row["status"] in {"completed", "success"}}
        common = sorted(left.keys() & right.keys())
        metric_names = sorted({name for key in common for name in left[key]["metrics"].keys() & right[key]["metrics"].keys()})
        differences = {name: [left[key]["metrics"][name]-right[key]["metrics"][name] for key in common
                             if name in left[key]["metrics"] and name in right[key]["metrics"]] for name in metric_names}
        latency = [left[key]["cost"]["complete_ms"]-right[key]["cost"]["complete_ms"] for key in common
            if numeric(left[key]["cost"].get("complete_ms")) and numeric(right[key]["cost"].get("complete_ms"))]
        comparisons.append(dict(cohort=cohort, dataset=dataset, split=split, candidate=method, baseline=baseline,
            common_completed_tasks=len(common), candidate_only=sorted(left.keys()-right.keys()),
            baseline_only=sorted(right.keys()-left.keys()),
            identical_question_rows=sum(left[key]["question"] is not None and left[key]["question"] == right[key]["question"] for key in common),
            identical_history_rows=sum(left[key]["actual_history"] is not None and left[key]["actual_history"] == right[key]["actual_history"] for key in common),
            metrics={name: dict(mean_candidate_minus_baseline=mean(values), paired_rows=len(values)) for name, values in differences.items()},
            mean_complete_ms_candidate_minus_baseline=mean(latency), paired_timing_rows=len(latency),
            scope="matched task IDs under the caller-declared cohort; own-P3 questions/history/DAG may differ; no gold rejudging"))
    return summaries, comparisons


def paired_tables(specs):
    paths, missing = expanded(specs)
    rows, sources = [], []
    for label, path in paths:
        sources.append(dict(label=label, path=str(path)))
        payload = load(path)
        for row in payload if isinstance(payload, list) else [payload]:
            rows.append({**row, "index_label": label, "source_file": str(path)})
    fields = ("split", "dataset", "event_id", "training_eligible", "measurement_eligible", "fixed_decode_work", "reused", "horizon_seconds",
        "cache_scope_hash", "sampling_weight", "full_window_cost_difference_ms", "target_revisit_cost_difference_ms",
        "target_cost_rule", "target_cache_work_changed", "target_revisit_step_wall_difference_ms",
        "collateral_and_initial_control_cost_difference_ms", "retention_control_ms", "physical_cost",
        "native_prefix_residency", "retained_prefix_residency", "reports")
    return rows, dict(files=sources, unmatched=missing, rows=[{name: row[name] for name in fields if name in row}
        | {"index_label": row["index_label"], "source_file": row["source_file"]} for row in rows])


def retention_validation(models, paired):
    from controller.prefix_retention import EmpiricalRetentionModel, TARGET_COST_RULE, session_components, session_group, balance_session_rows
    files, missing = expanded(models)
    dev = [row for row in paired if row["split"] == "dev" and row["training_eligible"]
           and row.get("target_cost_rule") == TARGET_COST_RULE and row.get("fixed_decode_work")]
    components = session_components(dev)
    result = []
    for name, path in files:
        raw = load(path)
        model = EmpiricalRetentionModel.from_dict(raw)
        matched, unmatched, matched_groups = [], [], set()
        for row in dev:
            try:
                forecast = model.predict(row["features"], row["cache_scope_hash"], row["horizon_seconds"])
            except KeyError:
                unmatched.append(row["event_id"])
                continue
            tokens = row["gpu_page_count"]*row["page_size"]
            within_range = (forecast.source_token_range[0] <= tokens <= forecast.source_token_range[1]
                            if forecast.source_token_range is not None else None)
            forecast = forecast.for_tokens(tokens)
            matched_groups.add(session_group(row, components))
            matched.append({**row, "predicted_full_window_ms": forecast.full_window_saved_ms,
                            "predicted_arm_ms": forecast.saved_ms,
                            "within_training_token_range": within_range})
        values = [(row["sampling_weight"], row["predicted_full_window_ms"], row["predicted_arm_ms"],
                   row["full_window_cost_difference_ms"]) for row in balance_session_rows(matched, components)]
        weight = sum(item[0] for item in values)
        policy = sum(w*actual if choice > 0 else 0 for w, _, choice, actual in values)/weight if values else None
        full = [(w, predicted, actual) for w, predicted, _, actual in values if predicted is not None]
        result.append(dict(model=name, path=str(path), fit_split=raw["training_manifest"]["split"],
            evaluation_split="dev", eligible_pairs=len(dev), matched_pairs=len(values),
            independent_samples=len(components), matched_independent_samples=len(matched_groups),
            prediction_coverage=len(values)/len(dev) if dev else None, unmatched_event_ids=unmatched,
            within_training_length_pairs=sum(row["within_training_token_range"] is True for row in matched),
            outside_training_length_pairs=sum(row["within_training_token_range"] is False for row in matched),
            full_window_prediction_pairs=len(full),
            weighted_mae_ms=sum(w*abs(predicted-actual) for w, predicted, actual in full)
                /sum(w for w, _, _ in full) if full else None,
            prediction_error_scope="complete response window",
            policy_selection_scope="accept each positive-predicted paired TTL arm independently; group-balanced mean over all matched arms; no per-window TTL selection or online policy evaluation",
            selected_positive_predictions=sum(choice > 0 for _, _, choice, _ in values),
            paired_arm_acceptance_delta_ms=policy if len(values) == len(dev) else None,
            matched_subgroup_paired_arm_delta_ms=policy,
            scope="offline dev cost ablation on existing paired data; no dev fitting; signed Native−Retain; online inspect/inference cost remains in actual joint records"))
    return dict(models=result, unmatched_model_patterns=missing)


def artifact_summary(value):
    if not isinstance(value, dict):
        return dict(type=type(value).__name__, items=len(value) if isinstance(value, list) else None)
    keys = ("format", "schema_version", "kind", "status", "scope", "source", "identity", "model_identity",
        "reader_checkpoint", "reader_profile_id", "reader_binding", "calibration_status", "calibration_split",
        "feature_schema", "training_manifest", "measurement_protocol_scope", "protocol_scopes",
        "current_read_inputs_match", "current_reduce_inputs_match", "phases",
        "coverage", "source_partition", "sample_counts", "actual_train_samples", "optimizer_steps", "training_wall_ms",
        "total_model_input_tokens", "total_supervised_tokens", "available_kinds", "missing_kinds", "validation",
        "resource_scope", "resident_model_manifest_digest", "selected_protocols", "answer_start_definition",
        "requested_qa", "per_method_requested_qa", "completed_by_method", "development_freeze", "test_execution",
        "method_spec", "test_methods", "selection_status", "cost_scope", "physical_total", "physical_cost",
        "native_tokens", "generations", "elapsed_seconds", "child_cpu_seconds", "controller_cpu_seconds")
    return dict(top_level_keys=list(value), **{key: compact(value[key]) for key in keys if key in value},
        profile_rows=len(value["rows"]) if isinstance(value.get("rows"), list) else None,
        forecast_profiles=len(value["forecasts"]) if isinstance(value.get("forecasts"), dict) else None)


def full_queue_cells(value, path, joint_scope=None, *, refresh_to=None):
    from profiling.runtime_manifest import hardware_identity
    if not isinstance(value, dict):
        return []
    if value.get("format") == "contextkv_dataset_queue_workflow_v1":
        rows, root = value["results"], path.parent
    elif value.get("format") == "contextkv_dataset_queue_result_v1":
        rows, root = [value], path.parents[2]
    else:
        return []
    plan_path = root/"capacity-plan.json"
    plan = load(plan_path)
    device_source = {name: plan["source_"+name]["path"] for name in ("config", "matrix")}
    binding = dict(frozen_A_config=device_source["config"], matrix=device_source["matrix"],
        cohort_source=(plan.get("source_cohorts") or plan["source_matrix"])["path"],
        arrival_plan=(plan["source_arrival_plan"]["path"] if plan["source_arrival_plan"] is not None else None),
        reader_profile_id=plan["reader_profile_id"],
        lambda_time=plan["lambda_time"], execution_commit=plan.get("execution_commit"))
    if joint_scope is not None:
        if not any(all(Path(worker[name]).resolve() == Path(source).resolve()
                       for name, source in device_source.items()) for worker in joint_scope["workers"]):
            raise ValueError("full-queue source configuration/matrix is not a worker in the explicit joint scope")
        binding.update(frozen_A_config=joint_scope["source_config"], matrix=joint_scope["source_matrix"])
    jobs = {job["name"]: job for job in plan["jobs"]}
    if refresh_to is not None:
        from bench.capacity_workflow import refresh_saved_full_queue

        cost = load(root/"capacity-cost.json")
        if value.get("format") == "contextkv_dataset_queue_workflow_v1":
            summarized = {row["job"] for row in rows}
            rows = [*rows, *({"job": name, "execution_status": "incomplete_saved_artifacts"}
                for name, job in jobs.items() if name not in summarized and Path(job["output"]).is_dir())]
        rows = [refresh_saved_full_queue(jobs[row["job"]], row, cost,
            refresh_to/root.parent.name/root.name/(row["job"]+".json")) for row in rows]
    cells = []
    for row in rows:
        job = jobs[row["job"]]
        factors = plan["method_factors"][row["method"]]
        missing = {name: track["missing_execution_task_ids"] for name, track in row["datasets"].items()}
        failed = {name: track["failed_task_ids"] for name, track in row["datasets"].items()}
        unscored = {name: track["unscored_answer_task_ids"] for name, track in row["datasets"].items()}
        scope = plan["resource_scope"]
        hardware = (hardware_identity(row["runtime_hardware"], scope["gpu_devices"])
                    if row.get("runtime_hardware") is not None else None)
        resource_identity = {key: scope[key] for key in ("gpu_capacity_tokens", "host_capacity_tokens")}
        resource_identity.update(gpu_device_count=len(set(scope["gpu_devices"])), hardware=hardware,
                                 native_cache_config=row.get("native_cache_config"))
        comparison = {**binding, "resource_scope": resource_identity,
            "dataset": job["dataset"], "split": job["split"], "protocol": job["protocol"],
            "cohort": plan["cohort"][job["dataset"]], "workload": plan["workload_protocols"][job["dataset"]],
            "complete_slo_seconds": job["complete_slo_seconds"]}
        acceptance = {}
        if plan.get("acceptance") is not None:
            replicate_scope = plan.get("replicate_id") or (joint_scope["manifest"]
                if joint_scope is not None else str(plan_path))
            acceptance = {"acceptance_protocol": {**plan["acceptance"],
                    "declared_qa": job["requested_qa"], "declared_sessions": job["requested_sessions"]}, "queue_job": job,
                "replicate_scope": replicate_scope, "repetition": job["repetition"],
                "required_experiment_cells": plan["experiment_cells"],
                "planned_repetitions": sorted({entry["repetition"] for entry in plan["jobs"]
                    if entry["concurrency"] == row["concurrency"] and entry["dataset"] == job["dataset"]
                    and entry["split"] == job["split"] and entry["method"] in {plan["acceptance"]["baseline"], plan["acceptance"]["candidate"]}})}
        cells.append({**row, "evaluation_kind": "dataset_queue", "source_file": str(path),
            "generation_dir": job["output"], "comparison_identity": comparison,
            "active_assets": {name: (plan["assets"].get(name) or {}).get("path") for name, active in (
                ("gain", factors["B"] == "gain"),
                ("completion_timing", factors["B"] in {"gain", "rule"}),
                ("demand_pricing", factors["retention"] == "demand"
                 and factors.get("retention_allocation") == "paired_value")) if active}
                | ({"compilation": plan["compilation_policy"]["checkout"]}
                   if factors["compilation"] == "contextpilot" else {}),
            "evaluation_factors": factors,
            "experiment_design": plan["experiment_design"],
            "device_source_identity": device_source,
            "explicit_joint_scope": {"manifest": joint_scope["manifest"], "frozen_A": joint_scope["frozen_A"]}
                if joint_scope is not None else None,
            "capacity_plan": str(plan_path), "full_workflow_cost": str(root/"capacity-cost.json"),
            "completion_rate": row["completed_qa"]/row["requested_qa"],
            "p95_response_count": row["completed_qa"], "missing_execution_task_ids": missing,
            "failed_task_ids": failed, "unscored_answer_task_ids": unscored,
            "full_queue_delivered": row["completed_qa"] == row["requested_qa"],
            "latency_scope": "complete request wall time of delivered responses; failed and blocked tasks are absent from P95",
            **acceptance})
    return cells


def queue_repetition(cell):
    return cell.get("replicate_scope"), cell.get("repetition")


def shared_asset_differences(left, right):
    left, right = left["active_assets"], right["active_assets"]
    return sorted(name for name in left.keys() & right.keys() if left[name] != right[name])


def queue_changed_factors(candidate, reference):
    current, base = candidate["evaluation_factors"], reference["evaluation_factors"]
    changes = {name for name in current.keys() | base.keys()
               if name != "role" and current.get(name) != base.get(name)}
    if "retention" in changes:
        # The allocator is part of enabling demand retention. Compare allocators
        # separately only when both cells already use the same retention mode.
        changes.discard("retention_allocation")
    return changes


def is_queue_reference(candidate, reference):
    """Select whole-system references and controlled changes from recorded factors."""
    if candidate["method"] == reference["method"]:
        return False
    current, base = candidate["evaluation_factors"], reference["evaluation_factors"]
    if (base["role"] == "main_baseline"
            or reference["method"] == candidate.get("acceptance_protocol", {}).get("baseline")):
        return True
    if current["dialogue"] != "contextkv":
        return False
    if base["role"] == "external_reference":
        return True
    changes = queue_changed_factors(candidate, reference)
    directions = {
        "B": {("rule", "gain")},
        "compilation": {("native", "contextpilot")},
        "retention": {("native", "demand")},
        "retention_allocation": {("next_use", "paired_value")},
        "fact_reuse": {(False, True)},
    }
    controlled = len(changes) == 1 or changes == {"compilation", "retention"}
    return controlled and all((base.get(name), current.get(name)) in directions.get(name, ()) for name in changes)


def full_queue_comparisons(cells):
    comparisons = []
    for row in cells:
        baselines = [other for other in cells if is_queue_reference(row, other)
                     and other["concurrency"] == row["concurrency"]
                     and all(other[key] == row[key] for key in ("dataset", "split", "protocol"))
                     and queue_repetition(other) == queue_repetition(row)]
        for reference in baselines:
            changes = sorted(queue_changed_factors(row, reference))
            reference_role = reference["evaluation_factors"]["role"]
            comparison_kind = ("whole_system" if reference_role in {"main_baseline", "external_reference"}
                else "single_factor" if len(changes) == 1 else "joint_factors")
            missing_resource_fields = [name for name in ("hardware", "native_cache_config")
                if any(cell["comparison_identity"]["resource_scope"][name] is None for cell in (row, reference))]
            different_assets = shared_asset_differences(row, reference)
            same = (not missing_resource_fields and not different_assets
                    and row["comparison_identity"] == reference["comparison_identity"])
            complete_scored = all(cell["full_queue_delivered"]
                and numeric(cell["queue_quality"])
                and not any(cell["unscored_answer_task_ids"].values()) for cell in (row, reference))
            relative_p95 = ((row["p95_complete_ms"] / reference["p95_complete_ms"] - 1)
                if same and complete_scored and numeric(row["p95_complete_ms"])
                and numeric(reference["p95_complete_ms"]) and reference["p95_complete_ms"] > 0 else None)
            comparisons.append(dict(dataset=row["dataset"], split=row["split"], protocol=row["protocol"],
                concurrency=row["concurrency"], candidate=row["method"], baseline=reference["method"],
                replicate_scope=row.get("replicate_scope"), repetition=row.get("repetition"),
                source_file=row["source_file"], baseline_source_file=reference["source_file"],
                candidate_generation_dir=row["generation_dir"], baseline_generation_dir=reference["generation_dir"],
                comparison_kind=comparison_kind, changed_factors=changes,
                comparable=same, incompatible_identity_fields=[key for key, value in row["comparison_identity"].items()
                    if value != reference["comparison_identity"][key]] + ["assets."+name for name in different_assets],
                missing_identity_fields=["resource_scope."+name for name in missing_resource_fields],
                device_source_identity_differs=row["device_source_identity"] != reference["device_source_identity"],
                differences={field: row[field]-reference[field] if same and numeric(row[field])
                    and numeric(reference[field]) and (field != "p95_complete_ms" or complete_scored) else None
                    for field in ("queue_quality", "completion_rate", "completed_qa", "p95_complete_ms")},
                candidate_completed=row["completed_qa"], baseline_completed=reference["completed_qa"],
                candidate_declared=row["requested_qa"], baseline_declared=reference["requested_qa"],
                both_full_queue_delivered=row["full_queue_delivered"] and reference["full_queue_delivered"],
                both_full_queue_scored=complete_scored,
                relative_p95_change=relative_p95,
                quality_not_lower=(row["queue_quality"] >= reference["queue_quality"]
                                   if same and complete_scored else None),
                latency_scope="delivered responses only; with incomplete delivery, P95 does not compare the complete declared queue",
                cost_scope="each cell retains its paid native index, batch costs and whole workflow cost; nested scopes are not added",
                scope="same dataset, split, history protocol, reader, resource budgets and workload; method-specific context management"))
    return comparisons


def full_queue_track_tables(cells):
    lines = []
    for origin, title in (("public_benchmark", "公开基准数据轨"),
                          ("public_reference_diagnostic", "公开参考历史诊断轨"),
                          ("constructed_task", "构造任务机制轨")):
        selected = [(cell, name, cell["datasets"][name], protocol) for cell in cells
                    if cell.get("evaluation_kind") == "dataset_queue"
                    for name, protocol in cell["dataset_protocols"].items() if protocol["origin"] == origin]
        if not selected:
            continue
        lines += ["", "## "+title, "",
            "| 方法 | 并发 | 数据集 / 分区 / 历史 / 原指标 | 交付 / 声明 | 已评分 / 已交付 | 全分母原指标 | 未知区间 | 已评分子集指标 |",
            "|---|---:|---|---:|---:|---:|---|---:|"]
        for cell, name, track, protocol in selected:
            lines.append(f"| {cell['method']} | {cell['concurrency']} | {name} / {cell['split']} / {cell['protocol']} / {protocol['primary_metric']} | "
                f"{track['completed']}/{track['declared']} | {track['scored']}/{track['completed']} | "
                f"{track['queue_primary_score']} | {track['queue_score_bounds']} | {track['scored_subset_primary']} |")
    return lines


def markdown(report):
    lines = ["# ContextKV 最终产物索引", "", f"评审基线：`{report['baseline_commit']}`。",
        "仅从指定产物读取已有评分与费用，按原指标名称作算术汇总；缺失值不补零。", "",
        "## 产物", "", "| 标签 | 文件 | 状态 / 类型 |", "|---|---|---|"]
    for item in report["artifacts"]:
        summary = item["summary"]
        lines.append(f"| {item['label']} | `{item['path']}` | {summary.get('status', summary.get('format', summary.get('kind', 'indexed')))} |")
    lines += ["", "## 已有逐题结果", "", "| cohort / 数据 / split | 方法 | 完成 / 已记录 | 已有指标均值（评分分母） | P95 ms |", "|---|---|---:|---|---:|"]
    for row in report["record_summaries"]:
        metrics = "; ".join(f"{name}={value['mean']:.6g} ({value['scored_rows']}/{value['completed_rows']})"
                            for name, value in row["metrics"].items()) or "尚无已存数值评分"
        lines.append(f"| {row['cohort']} / {row['dataset']} / {row['split']} | {row['method']} | {row['completed_rows']}/{row['recorded_rows']} | {metrics} | {row['response_cost']['p95_complete_ms']} |")
    lines += ["", "## 固定 token 的 C 配对", "", "| split | 合格 | 原文复用 | TTL s | 整窗 Native−Retain ms | 目标 ms | 其他与控制 ms |", "|---|---|---|---:|---:|---:|---:|"]
    for row in report["paired_C"]["rows"]:
        lines.append(f"| {row['split']} | {row['training_eligible']} | {row['reused']} | {row['horizon_seconds']} | {row['full_window_cost_difference_ms']} | {row['target_revisit_cost_difference_ms']} | {row['collateral_and_initial_control_cost_difference_ms']} |")
    lines += ["", "## C 模型开发消融", "", "| 资产 | dev 覆盖 | MAE ms | 独立组等权配对臂差 ms |", "|---|---:|---:|---:|"]
    for row in report["retention_dev"]["models"]:
        lines.append(f"| {row['model']} | {row['matched_pairs']}/{row['eligible_pairs']} | {row['weighted_mae_ms']} | {row['paired_arm_acceptance_delta_ms']} |")
    lines += ["", "## 各数据集完整队列", "",
        "| 数据集 / 分区 / 历史 | 方法 | 并发 | 交付 / 声明 | 归一化分数 | 分数区间 | 成功完整响应 P95 ms |",
        "|---|---|---:|---:|---:|---|---:|"]
    for row in report["capacity"]:
        if row.get("evaluation_kind") == "dataset_queue":
            lines.append(f"| {row['dataset']} / {row['split']} / {row['protocol']} | {row['method']} | {row['concurrency']} | "
                f"{row['completed_qa']}/{row['requested_qa']} | {row['queue_quality']} | {row['queue_quality_bounds']} | {row['p95_complete_ms']} |")
    lines += ["", "## 完整队列对照差", "",
        "| 数据集 / 分区 / 历史 | 候选 / 基线 | 并发 | 对照范围 / 变化因素 | 质量差（候选−基线） | 完整 P95 相对变化 | 质量不低于基线 |",
        "|---|---|---:|---|---:|---:|---|"]
    comparison_labels = {"whole_system": "整包",
                         "single_factor": "单因素", "joint_factors": "联合变化"}
    for comparison in report["capacity_comparisons"]:
        if "relative_p95_change" not in comparison:
            continue
        delta = comparison["differences"]["queue_quality"]
        relative = comparison["relative_p95_change"]
        quality = comparison["quality_not_lower"]
        factors = ", ".join(comparison["changed_factors"])
        lines.append(f"| {comparison['dataset']} / {comparison['split']} / {comparison['protocol']} | "
            f"{comparison['candidate']} / {comparison['baseline']} | {comparison['concurrency']} | "
            f"{comparison_labels[comparison['comparison_kind']]} / {factors} | "
            f"{delta if delta is not None else '未知'} | "
            f"{format(relative, '.2%') if relative is not None else '未知'} | "
            f"{('是' if quality else '否') if quality is not None else '未知'} |")
    lines += ["", "## 预登记的完整队列验收", "",
        "| 数据集 / 分区 / 历史 | 并发 | 完成测量的实验格 | 配对队列 | 费用与测量完整 | 资源预算 | 质量差及区间 | P95 比率及区间 | 质量验收 | 延迟验收 |",
        "|---|---:|---|---:|---|---|---|---|---|---|"]
    def point_interval(point, interval):
        if point is None:
            return "未知"
        suffix = f" [{interval[0]:.4f}, {interval[1]:.4f}]" if interval is not None else ""
        return f"{point:.4f}"+suffix

    for row in report["full_queue_acceptance"]:
        quality_ci = row["quality_confidence"]
        cells = row["required_cells_measured"]
        pending = [label for label, measured in cells.items() if not measured]
        coverage = f"{len(cells)-len(pending)}/{len(cells)}"
        if pending:
            coverage += "（缺 " + "、".join(pending) + "）"
        lines.append(f"| {row['dataset']} / {row['split']} / {row['protocol']} | "
            f"{row['concurrency']} | {coverage} | {row['paired_queues']}/{row['declared_pair_slots']} | "
            f"{row['measurement_complete']} | {row['resource_budget_pass']} | "
            f"{point_interval(row['quality_delta'], quality_ci['interval'] if quality_ci else None)} | "
            f"{point_interval(row['p95_ratio'], row['p95_ratio_interval'])} | "
            f"{row['quality_gate']} | {row['latency_gate']} |")
    lines += full_queue_track_tables(report["capacity"])
    lines += ["", "每个数据集保留原指标与归一化分数；不同数据集不合成为单一质量分。"]
    lines += ["", "## 完整响应 SLO", "",
        "| 方法 | 并发 | 数据集 | 完成 / 失败 / 阻断 | SLO 秒 | 达标回答 / 秒 | 观察秒数 |",
        "|---|---:|---|---|---|---|---:|"]
    for row in report["capacity"]:
        if row.get("evaluation_kind") != "dataset_queue":
            continue
        for dataset, delivery in row["delivery_metrics"].items():
            if delivery is None:
                lines.append(f"| {row['method']} | {row['concurrency']} | {dataset} | 未观测 | — | — | — |")
                continue
            lines.append(f"| {row['method']} | {row['concurrency']} | {dataset} | {delivery['statuses']} | "
                f"{delivery['slo'].get('complete_seconds', '未声明')} | "
                f"{delivery['latency_slo_goodput_requests_per_second']} | {delivery['observation_seconds']} |")
    lines += ["", "P95 的样本仅包含已交付响应。未完成与未评分任务按完整声明分母另列，存在交付缺失时不据此声称完整队列延迟改善。"]
    lines += ["", "系统比较按各数据集声明的历史协议执行；固定输入的缓存配对另行解释执行成本。",
        "逐方法相对基线差、共同任务数、相同问题/历史计数及原始成本来源见 JSON。",
        "", "## 成本", "", "物理总量引用已有去重 ledger；嵌套 job、phase 与请求份额不重复相加。",
        "提供原始采样与显式 GPU 设备时，成本命令保存为 `cost-ledger-command.json`；已有账本直接引用。", ""]
    for ledger in report["cost_ledgers"]:
        lines.extend([f"`{ledger['path']}`", "", "```json", json.dumps(ledger["physical_total"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main(argv=None):
    from profiling.full_queue_acceptance import evaluate_full_queue

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit", required=True, help="Declared baseline source commit; not inferred from the current checkout")
    parser.add_argument("--gpu-devices", help="Explicit physical device IDs for a requested cost-ledger command")
    parser.add_argument("--artifact", action="append", default=[], help="LABEL=path/glob (JSON/YAML)")
    parser.add_argument("--records", action="append", default=[], help="COHORT=explicit scored/unscored JSONL path/glob")
    parser.add_argument("--paired", action="append", default=[], help="LABEL=paired-rows.jsonl path/glob")
    parser.add_argument("--retention-model", action="append", default=[], help="LABEL=existing empirical model JSON")
    parser.add_argument("--baseline", action="append", default=[], help="COHORT=METHOD; *=METHOD supplies the default")
    parser.add_argument("--cost-ledger", action="append", default=[])
    parser.add_argument("--cost-manifest", action="append", default=[], help="Preserve this explicit glob in the ledger command")
    parser.add_argument("--physical", action="append", default=[])
    parser.add_argument("--joint-scope", help="Explicit frozen inputs.json registering the workers used by full-queue artifacts")
    parser.add_argument("--refresh-saved-queues", action="store_true",
        help="Rebuild finalized full-queue summaries from saved answers, scores, and original resource observations")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (args.cost_manifest or args.physical) and not args.gpu_devices:
        parser.error("raw cost-ledger inputs require --gpu-devices")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts, capacity, capacity_comparisons, ledgers, full_cells = [], [], [], [], []
    joint_scope = None
    if args.joint_scope:
        joint_scope = {**load(args.joint_scope), "manifest": str(Path(args.joint_scope).resolve())}
    paths, missing = expanded(args.artifact)
    for label, path in paths:
        value = load(path)
        artifacts.append(dict(label=label, path=str(path), summary=artifact_summary(value)))
        full_cells.extend(full_queue_cells(value, path, joint_scope,
            refresh_to=root/"refreshed-native" if args.refresh_saved_queues else None))
    full_cells = list({row["generation_dir"]: row for row in full_cells}.values())
    capacity.extend(full_cells)
    capacity_comparisons.extend(full_queue_comparisons(full_cells))
    for path in args.cost_ledger:
        value = load(path)
        ledgers.append({"path": str(Path(path).resolve()), "physical_total": value["physical_total"],
            "native_tokens": value.get("native_tokens"), "resource_window": value.get("resource_window")})
    rows, record_import = import_records(args.records)
    summaries, comparisons = record_tables(rows, dict(item.split("=", 1) for item in args.baseline))
    paired, paired_index = paired_tables(args.paired)
    report = dict(format="contextkv_final_study_index_v1", baseline_commit=args.baseline_commit,
        scope="explicit existing manifests and existing numeric scored fields; no generation/judging/gold reinterpretation",
        artifacts=artifacts, unmatched_artifact_patterns=missing, record_import=record_import,
        record_summaries=summaries, common_task_comparisons=comparisons,
        paired_C=paired_index, retention_dev=retention_validation(args.retention_model, paired),
        capacity=capacity, capacity_comparisons=capacity_comparisons,
        full_queue_acceptance=evaluate_full_queue(full_cells),
        cost_ledgers=ledgers)
    if args.cost_manifest or args.physical:
        command = ["python", "-m", "profiling.study_cost_ledger"]
        for path in args.cost_manifest:
            command += ["--manifest", path]
        for path in args.physical:
            command += ["--physical", path]
        command += ["--gpu-devices", args.gpu_devices, "--output", str(root/"full-cost-ledger.json")]
        write(root/"cost-ledger-command.json", dict(argv=command, shell=shlex.join(command), executed=False,
            scope="explicit requested files only; unknown sampling intervals remain unknown"))
    write(root/"final-index.json", result_fields(report))
    (root/"final-index.md").write_text(markdown(report))
    print(json.dumps(dict(artifacts=len(artifacts), result_groups=len(summaries), comparisons=len(comparisons),
        C_pairs=len(paired), capacity_cells=len(capacity), cost_ledgers=len(ledgers), output=str(root)), ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
