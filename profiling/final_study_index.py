"""Read existing study artifacts into a final index and arithmetic comparisons.

No generation, judging, gold parsing, model loading, or remote operations occur.
Separate caller-labelled cohorts remain separate.  All quality numbers come
from existing numeric metric fields; missing scores keep their denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
from itertools import product
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


def identity(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            sha.update(chunk)
    return dict(path=str(path), sha256=sha.hexdigest(), bytes=path.stat().st_size)


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
        provenance.append(dict(cohort=cohort, **identity(path)))
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
                answer_sha256=hashlib.sha256(raw["answer"].encode()).hexdigest() if isinstance(raw.get("answer"), str) else None,
                source_files=[str(path)])
            if key in indexed:
                old = indexed[key]
                agreement = (old["status"] == row["status"] and old["answer_sha256"] == row["answer_sha256"]
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
        sources.append(dict(label=label, **identity(path)))
        payload = load(path)
        for row in payload if isinstance(payload, list) else [payload]:
            rows.append({**row, "index_label": label, "source_file": str(path)})
    fields = ("split", "dataset", "event_id", "training_eligible", "measurement_eligible", "reused", "horizon_seconds",
        "cache_scope_hash", "sampling_weight", "full_window_cost_difference_ms", "target_revisit_cost_difference_ms",
        "collateral_and_initial_control_cost_difference_ms", "retention_control_ms", "physical_cost",
        "native_prefix_residency", "retained_prefix_residency", "reports", "report_digests")
    return rows, dict(files=sources, unmatched=missing, rows=[{name: row[name] for name in fields if name in row}
        | {"index_label": row["index_label"], "source_file": row["source_file"]} for row in rows])


def retention_validation(models, paired):
    from controller.prefix_retention import EmpiricalRetentionModel, session_components, session_group, balance_session_rows
    files, missing = expanded(models)
    dev = [row for row in paired if row["split"] == "dev" and row["training_eligible"]]
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
            matched_groups.add(session_group(row, components))
            matched.append({**row, "predicted_full_window_ms": forecast.full_window_saved_ms,
                            "predicted_arm_ms": forecast.saved_ms})
        values = [(row["sampling_weight"], row["predicted_full_window_ms"], row["predicted_arm_ms"],
                   row["full_window_cost_difference_ms"]) for row in balance_session_rows(matched, components)]
        weight = sum(item[0] for item in values)
        policy = sum(w*actual if choice > 0 else 0 for w, _, choice, actual in values)/weight if values else None
        full = [(w, predicted, actual) for w, predicted, _, actual in values if predicted is not None]
        result.append(dict(model=name, **identity(path), fit_split=raw["training_manifest"]["split"],
            evaluation_split="dev", eligible_pairs=len(dev), matched_pairs=len(values),
            independent_samples=len(components), matched_independent_samples=len(matched_groups),
            prediction_coverage=len(values)/len(dev) if dev else None, unmatched_event_ids=unmatched,
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


def capacity_tables(value, path):
    if isinstance(value, dict) and value.get("format") == "contextkv_two_turn_capacity_workflow_v1":
        rows = value["results"]
    elif isinstance(value, dict) and "queue_quality" in value:
        return [], []
    elif isinstance(value, dict) and {"job", "concurrency", "completed_qa"} <= value.keys():
        rows = [value]
    else:
        return [], []
    fields = ("job", "method", "concurrency", "clients", "requested_qa", "completed_qa", "submitted_qa", "failed_clients",
        "completed_qa_coverage", "throughput_qa_per_second", "makespan_seconds_including_pause", "p95_complete_ms",
        "successful_qa_tokens", "need_support", "source_unit_read_coverage", "batch_costs", "failure_cost_scope")
    indexed = [{**{field: row[field] for field in fields if field in row}, "source_file": str(path)} for row in rows]
    by_key = {(row["concurrency"], row["method"]): row for row in rows}
    comparisons = []
    for row in rows:
        reference = by_key.get((row["concurrency"], "B6_native"))
        if reference is None or row["method"] == "B6_native":
            continue
        comparisons.append(dict(concurrency=row["concurrency"], candidate=row["method"], baseline="B6_native",
            source_file=str(path), differences={field: row[field]-reference[field]
                if numeric(row.get(field)) and numeric(reference.get(field)) else None for field in (
                "completed_qa_coverage", "throughput_qa_per_second", "p95_complete_ms")},
            scope="actual own-P3 two-turn capacity; different answers/branches/DAG permitted; not fixed-token causal C pairing"))
    return indexed, comparisons


def full_queue_cells(value, path, joint_scope=None):
    from profiling.runtime_manifest import hardware_identity
    if not isinstance(value, dict):
        return []
    if value.get("format") == "contextkv_full_queue_joint_workflow_v1":
        rows, root = value["results"], path.parent
    elif {"job", "datasets", "queue_quality", "paid_native_usage"} <= value.keys():
        rows, root = [value], path.parents[2]
    else:
        return []
    plan_path = root/"capacity-plan.json"
    plan = load(plan_path)
    run = load(plan["reader_run"]["path"])
    run.pop("state_dir", None)
    run["engine"].pop("base_url", None)
    def canonical(item):
        return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
    binding = dict(frozen_A_config_sha256=plan["source_config"]["sha256"],
        matrix_sha256=plan["source_matrix"]["sha256"], cohort_sha256=canonical(plan["cohort"]),
        reader_run_sha256=canonical(run), reader_profile_id=plan["reader_profile_id"],
        lambda_time=plan["lambda_time"])
    device_source = {key: binding[key] for key in ("frozen_A_config_sha256", "matrix_sha256")}
    if joint_scope is not None:
        registered = [worker for worker in joint_scope["workers"]
            if Path(worker["config"]).resolve() == Path(plan["source_config"]["path"]).resolve()
            and Path(worker["matrix"]).resolve() == Path(plan["source_matrix"]["path"]).resolve()]
        if len(registered) != 1:
            raise ValueError("full-queue source configuration/matrix is not a worker in the explicit joint scope")
        for name in ("config", "matrix"):
            source = plan["source_"+name]
            if identity(registered[0][name])["sha256"] != source["sha256"]:
                raise ValueError("registered worker input differs from the full-queue execution source")
        binding.update(frozen_A_config_sha256=joint_scope["source_config_sha256"],
            matrix_sha256=joint_scope["source_matrix_sha256"],
            joint_scope_sha256=joint_scope["manifest"]["sha256"])
    jobs = {job["name"]: job for job in plan["jobs"]}
    cells = []
    for row in rows:
        job = jobs[row["job"]]
        missing = {name: track["missing_execution_task_ids"] for name, track in row["datasets"].items()}
        failed = {name: track["failed_task_ids"] for name, track in row["datasets"].items()}
        unscored = {name: track["unscored_answer_task_ids"] for name, track in row["datasets"].items()}
        workload = plan["workload_plans"][str(row["concurrency"])]
        scope = plan["resource_scope"]
        hardware = (hardware_identity(row["runtime_hardware"], scope["gpu_devices"])
                    if row.get("runtime_hardware") is not None else None)
        resource_identity = {key: scope[key] for key in ("gpu_capacity_tokens", "host_capacity_tokens")}
        resource_identity.update(gpu_device_count=len(set(scope["gpu_devices"])), hardware=hardware,
                                 native_cache_config=row.get("native_cache_config"))
        comparison = {**binding,
            "resource_scope": resource_identity,
            "workload_sha256": {name: item["sha256"] for name, item in workload.items()},
            "complete_slo_seconds": job["complete_slo_seconds"]}
        cells.append({**row, "evaluation_kind": "full_queue_own_P3", "source_file": str(path),
            "generation_dir": job["output"], "comparison_identity": comparison,
            "active_assets": {name: plan["assets"][name] for name, active in (
                ("gain", row["method"].startswith("S3_")), ("value", row["method"].endswith("_value"))) if active},
            "device_source_identity": device_source,
            "explicit_joint_scope": {"manifest": joint_scope["manifest"], "frozen_A": joint_scope["frozen_A"]}
                if joint_scope is not None else None,
            "capacity_plan": identity(plan_path), "full_workflow_cost": str(root/"capacity-cost.json"),
            "completion_rate": row["completed_qa"]/row["requested_qa"],
            "p95_response_count": row["completed_qa"], "missing_execution_task_ids": missing,
            "failed_task_ids": failed, "unscored_answer_task_ids": unscored,
            "full_queue_delivered": row["completed_qa"] == row["requested_qa"],
            "latency_scope": "complete request wall time of delivered responses; failed and blocked tasks are absent from P95"})
    return cells


def full_queue_comparisons(cells):
    comparisons = []
    references = {"E0_native": (), "B6_native": ("E0_native",),
        "S3_native": ("E0_native", "B6_native"),
        "B6_value": ("E0_native", "B6_native"),
        "S3_value": ("E0_native", "B6_native", "S3_native", "B6_value")}
    for row in cells:
        baselines = [other for other in cells if other["method"] in references[row["method"]]
                     and other["concurrency"] == row["concurrency"]]
        for reference in baselines:
            missing_resource_fields = [name for name in ("hardware", "native_cache_config")
                if any(cell["comparison_identity"]["resource_scope"][name] is None for cell in (row, reference))]
            different_assets = [name for name in row["active_assets"].keys() & reference["active_assets"].keys()
                if row["active_assets"][name]["sha256"] != reference["active_assets"][name]["sha256"]]
            same = (not missing_resource_fields and not different_assets
                    and row["comparison_identity"] == reference["comparison_identity"])
            complete_scored = all(cell["full_queue_delivered"]
                and numeric(cell["queue_quality"])
                and not any(cell["unscored_answer_task_ids"].values()) for cell in (row, reference))
            relative_p95 = ((row["p95_complete_ms"] / reference["p95_complete_ms"] - 1)
                if same and complete_scored and numeric(row["p95_complete_ms"])
                and numeric(reference["p95_complete_ms"]) and reference["p95_complete_ms"] > 0 else None)
            comparisons.append(dict(concurrency=row["concurrency"], candidate=row["method"], baseline=reference["method"],
                source_file=row["source_file"], baseline_source_file=reference["source_file"],
                candidate_generation_dir=row["generation_dir"], baseline_generation_dir=reference["generation_dir"],
                comparable=same, incompatible_identity_fields=[key for key, value in row["comparison_identity"].items()
                    if value != reference["comparison_identity"][key]] + ["assets."+name for name in sorted(different_assets)],
                missing_identity_fields=["resource_scope."+name for name in missing_resource_fields],
                device_source_identity_differs=row["device_source_identity"] != reference["device_source_identity"],
                differences={field: row[field]-reference[field] if same and numeric(row[field])
                    and numeric(reference[field]) else None
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
                scope=("registered E0 reference versus current method under shared reader, budgets, arrivals and own-P3 queue"
                       if reference["method"] == "E0_native" else
                       "same frozen A, reader budgets, arrivals and own-P3 queue; different actual histories permitted")))
    return comparisons


def full_queue_interactions(cells):
    """Compute the registered B/C factorial contrast on the complete queue."""
    methods = ("B6_native", "S3_native", "B6_value", "S3_value")
    groups = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        if cell["method"] in methods:
            key = (cell["concurrency"], json.dumps(cell["comparison_identity"], sort_keys=True))
            groups[key][cell["method"]].append(cell)
    interactions = []
    for (concurrency, _), group in groups.items():
        for quartet in product(*(group[method] for method in methods)):
            e1, e2, e3, e4 = quartet
            scope = e1["comparison_identity"]["resource_scope"]
            if any(scope[name] is None for name in ("hardware", "native_cache_config")):
                continue
            if (e2["active_assets"]["gain"]["sha256"] != e4["active_assets"]["gain"]["sha256"]
                    or e3["active_assets"]["value"]["sha256"] != e4["active_assets"]["value"]["sha256"]):
                continue
            complete = all(cell["full_queue_delivered"] and numeric(cell["queue_quality"])
                and not any(cell["unscored_answer_task_ids"].values()) for cell in quartet)
            contrasts = {}
            for metric in ("queue_quality", "completion_rate", "p95_complete_ms"):
                values = [cell[metric] for cell in quartet]
                known = all(numeric(value) for value in values) and (metric != "p95_complete_ms" or complete)
                contrasts[metric] = (dict(B_native=values[1]-values[0], C_fixed=values[2]-values[0],
                    B_value=values[3]-values[2], C_gain=values[3]-values[1],
                    joint=values[3]-values[0], interaction=values[3]-values[1]-values[2]+values[0])
                    if known else None)
            interactions.append(dict(concurrency=concurrency, comparison_identity=e1["comparison_identity"],
                cells={f"E{index}": dict(method=cell["method"], source_file=cell["source_file"],
                    generation_dir=cell["generation_dir"]) for index, cell in enumerate(quartet, 1)},
                contrasts=contrasts, full_queue_delivered_and_scored=complete,
                formula="interaction = E4 - E2 - E3 + E1",
                scope="own-P3 full-queue contrasts; P95 contrast uses four queue quantiles, not request-level effects"))
    return interactions


def full_queue_track_tables(cells):
    lines = []
    for origin, title in (("public_benchmark", "公开基准数据轨"),
                          ("public_reference_diagnostic", "公开参考历史诊断轨"),
                          ("constructed_task", "构造任务机制轨")):
        selected = [(cell, name, cell["datasets"][name], protocol) for cell in cells
                    if cell.get("evaluation_kind") == "full_queue_own_P3"
                    for name, protocol in cell["dataset_protocols"].items() if protocol["origin"] == origin]
        if not selected:
            continue
        lines += ["", "## "+title, "",
            "| 方法 | 并发 | 数据集 / 原指标 | 交付 / 声明 | 已评分 / 已交付 | 全分母原指标 | 未知区间 | 已评分子集指标 |",
            "|---|---:|---|---:|---:|---:|---|---:|"]
        for cell, name, track, protocol in selected:
            lines.append(f"| {cell['method']} | {cell['concurrency']} | {name} / {protocol['primary_metric']} | "
                f"{track['completed']}/{track['declared']} | {track['scored']}/{track['completed']} | "
                f"{track['queue_primary_score']} | {track['queue_score_bounds']} | {track['scored_subset_primary']} |")
    return lines


def markdown(report):
    lines = ["# ContextKV 最终产物索引", "", f"评审基线：`{report['baseline_commit']}`。",
        "仅从指定产物读取已有评分与费用，按原指标名称作算术汇总；缺失值不补零。", "",
        "## 产物", "", "| 标签 | 文件 | 状态 / 类型 | SHA256 |", "|---|---|---|---|"]
    for item in report["artifacts"]:
        summary = item["summary"]
        lines.append(f"| {item['label']} | `{item['path']}` | {summary.get('status', summary.get('format', summary.get('kind', 'indexed')))} | `{item['sha256']}` |")
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
    lines += ["", "## 实际两轮容量与联合", "", "| 方法 | 并发 / clients | 完成 / 预定 | QA/s（含停顿） | P95 ms |", "|---|---:|---:|---:|---:|"]
    for row in report["capacity"]:
        if row.get("evaluation_kind") != "full_queue_own_P3":
            lines.append(f"| {row['method']} | {row['concurrency']}/{row['clients']} | {row['completed_qa']}/{row['requested_qa']} | {row['throughput_qa_per_second']} | {row['p95_complete_ms']} |")
    lines += ["", "## 完整队列联合格", "", "| 方法 | 并发 | 交付 / 声明 | 队列质量 | 队列质量区间 | 成功完整响应 P95 ms |", "|---|---:|---:|---:|---|---:|"]
    for row in report["capacity"]:
        if row.get("evaluation_kind") == "full_queue_own_P3":
            lines.append(f"| {row['method']} | {row['concurrency']} | {row['completed_qa']}/{row['requested_qa']} | {row['queue_quality']} | {row['queue_quality_bounds']} | {row['p95_complete_ms']} |")
    lines += ["", "## 完整队列对照差", "",
        "| 候选 / 基线 | 并发 | 质量差（候选−基线） | 完整 P95 相对变化 | 质量不低于基线 |",
        "|---|---:|---:|---:|---|"]
    for comparison in report["capacity_comparisons"]:
        if "relative_p95_change" not in comparison:
            continue
        delta = comparison["differences"]["queue_quality"]
        relative = comparison["relative_p95_change"]
        quality = comparison["quality_not_lower"]
        lines.append(f"| {comparison['candidate']} / {comparison['baseline']} | {comparison['concurrency']} | "
            f"{delta if delta is not None else '未知'} | "
            f"{format(relative, '.2%') if relative is not None else '未知'} | "
            f"{('是' if quality else '否') if quality is not None else '未知'} |")
    lines += ["", "P95 相对变化为（候选 / 基线 − 1），负值表示降低；仅在身份一致、完整交付且全部质量已评分时报告。"
        "质量不低于基线仅比较现有点估计，不代表统计等质量结论；本表不判定同预算、资源收益或研究目标达成。"]
    lines += ["", "## B/C 四格贡献与交互", "",
        "| 并发 | 指标 | B（Native） | B（价值 C） | C（固定 B） | C（收益 B） | 联合 E4−E1 | 交互 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["factorial_interactions"]:
        for metric, effects in row["contrasts"].items():
            values = [effects[name] if effects is not None else "未知"
                      for name in ("B_native", "B_value", "C_fixed", "C_gain", "joint", "interaction")]
            lines.append(f"| {row['concurrency']} | {metric} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "交互＝E4−E2−E3＋E1；质量为正表示互补，延迟为负表示额外降低。"
        "P95 使用四格各自完整队列的分位数，仅为队列统计量的算术对照。"]
    lines += full_queue_track_tables(report["capacity"])
    lines += ["", "分轨保留原指标量纲；混合开发 Q 仍采用登记权重，不将构造题增益表述为公开基准增益。"]
    lines += ["", "## 完整响应 SLO", "",
        "| 方法 | 并发 | 数据集 | 完成 / 失败 / 阻断 | SLO 秒 | 达标回答 / 秒 | 观察秒数 |",
        "|---|---:|---|---|---|---|---:|"]
    for row in report["capacity"]:
        if row.get("evaluation_kind") != "full_queue_own_P3":
            continue
        for dataset, delivery in row["delivery_metrics"].items():
            if delivery is None:
                lines.append(f"| {row['method']} | {row['concurrency']} | {dataset} | 未观测 | — | — | — |")
                continue
            lines.append(f"| {row['method']} | {row['concurrency']} | {dataset} | {delivery['statuses']} | "
                f"{delivery['slo'].get('complete_seconds', '未声明')} | "
                f"{delivery['latency_slo_goodput_requests_per_second']} | {delivery['observation_seconds']} |")
    lines += ["", "P95 的样本仅包含已交付响应。未完成与未评分任务按完整声明分母另列，存在交付缺失时不据此声称完整队列延迟改善。"]
    lines += ["", "四格自身 P3 与并发容量保留不同回答、分支和 DAG；它们不替代固定输入的 C 配对。",
        "逐方法相对基线差、共同任务数、相同问题/历史计数及原始成本来源见 JSON。",
        "", "## 成本", "", "物理总量引用已有去重 ledger；嵌套 job、phase 与请求份额不重复相加。",
        "提供原始采样与显式 GPU 设备时，成本命令保存为 `cost-ledger-command.json`；已有账本直接引用。", ""]
    for ledger in report["cost_ledgers"]:
        lines.extend([f"`{ledger['path']}`", "", "```json", json.dumps(ledger["physical_total"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main(argv=None):
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (args.cost_manifest or args.physical) and not args.gpu_devices:
        parser.error("raw cost-ledger inputs require --gpu-devices")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts, capacity, capacity_comparisons, ledgers, full_cells = [], [], [], [], []
    joint_scope = None
    if args.joint_scope:
        scope = load(args.joint_scope)
        for name in ("config", "matrix"):
            if identity(scope["source_"+name])["sha256"] != scope["source_"+name+"_sha256"]:
                raise ValueError("explicit joint scope source identity differs from the frozen original input")
        joint_scope = {**scope, "manifest": identity(args.joint_scope)}
    paths, missing = expanded(args.artifact)
    for label, path in paths:
        value = load(path)
        artifacts.append(dict(label=label, **identity(path), summary=artifact_summary(value)))
        indexed, comparisons = capacity_tables(value, path)
        capacity.extend(indexed)
        capacity_comparisons.extend(comparisons)
        full_cells.extend(full_queue_cells(value, path, joint_scope))
    full_cells = list({row["generation_dir"]: row for row in full_cells}.values())
    capacity.extend(full_cells)
    capacity_comparisons.extend(full_queue_comparisons(full_cells))
    for path in args.cost_ledger:
        value = load(path)
        ledgers.append({**identity(path), "physical_total": value["physical_total"],
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
        factorial_interactions=full_queue_interactions(full_cells), cost_ledgers=ledgers)
    if args.cost_manifest or args.physical:
        command = ["python", "-m", "profiling.study_cost_ledger"]
        for path in args.cost_manifest:
            command += ["--manifest", path]
        for path in args.physical:
            command += ["--physical", path]
        command += ["--gpu-devices", args.gpu_devices, "--output", str(root/"full-cost-ledger.json")]
        write(root/"cost-ledger-command.json", dict(argv=command, shell=shlex.join(command), executed=False,
            scope="explicit requested files only; unknown sampling intervals remain unknown"))
    write(root/"final-index.json", report)
    (root/"final-index.md").write_text(markdown(report))
    print(json.dumps(dict(artifacts=len(artifacts), result_groups=len(summaries), comparisons=len(comparisons),
        C_pairs=len(paired), capacity_cells=len(capacity), cost_ledgers=len(ledgers), output=str(root)), ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
