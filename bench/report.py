"""One report schema for controller and native runs; missing telemetry stays null."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from statistics import mean

from bench.judge import DIMENSIONS, parse_citations
from bench.metrics import TurnTiming, distribution, occupancy_integral, summarize_timings, transfer_totals
from bench.official import load_jsonl
from bench.quality import paired_cluster_bootstrap
from controller.planner import COST_SELECTION_REASONS


def _dict(value):
    return asdict(value) if is_dataclass(value) else value


def bind_quality_rows(turns, quality_rows, *, require_answer=False):
    """Check external scores against the run's successful answers once at import."""
    expected = {_dict(r)["task_id"]: _dict(r) for r in turns if _dict(r)["status"] == "success"}
    ids = [r["task_id"] for r in quality_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("quality report needs one score per task; select a method and aggregate human raters first")
    if set(ids) - expected.keys():
        raise ValueError("quality scores contain tasks absent from this run's successful answers")
    for row in quality_rows:
        target = expected[row["task_id"]]
        if row.get("conversation_id") != target["conversation_id"]:
            raise ValueError("quality task conversation differs from the evaluated run")
        if require_answer and "answer" not in row:
            raise ValueError("external quality scores must retain the evaluated answer")
        if "answer" in row and row["answer"] != target["assistant_text"]:
            raise ValueError("quality scores belong to a different answer for this task")
    return expected


def receipt_summary(rows):
    """Executed prefill can exceed prompt length after re-execution; do not clamp."""
    observed_cold = [r["cold"] for r in rows if r["cold"] is not None]
    return {"count": len(rows), "prefill_to_prompt_ratio": distribution(r["ratio"] for r in rows),
            "cold_prefill_observed_turns": len(observed_cold),
            "cold_prefill_turns": sum(observed_cold) if observed_cold else None,
            "cold_prefill_fraction": sum(observed_cold) / len(observed_cold) if observed_cold else None,
            "ttft_seconds": distribution(r["ttft"] for r in rows if r["ttft"] is not None),
            "complete_turn_seconds": distribution(r["turn_seconds"] for r in rows if r["turn_seconds"] is not None)}


def decision_summary(turns):
    """Describe executed controller choices, without treating fallback as cost selection."""
    actions, reasons = defaultdict(int), defaultdict(int)
    cost_choices = fallbacks = predicted = calibrated = eligible = 0
    for row in turns:
        row = _dict(row)
        result = _dict(row.get("result") or {})
        action, reason = result.get("action"), result.get("reason")
        if row["status"] != "success" or action is None or reason is None:
            continue
        actions[action] += 1
        reasons[reason] += 1
        cost_choices += reason in COST_SELECTION_REASONS
        fallbacks += reason.endswith("_fallback")
        estimate = _dict(result.get("prediction") or {})
        if estimate:
            predicted += 1
            calibrated += estimate.get("calibrated") is True
            supported = estimate.get("selection_eligible")
            eligible += (estimate.get("calibrated") is True if supported is None else supported is True)
    return {"scope": "successful_controller_turn_responses", "observed_turns": sum(actions.values()),
            "actions": dict(actions), "reasons": dict(reasons),
            "cost_based_selection_turns": cost_choices, "fallback_turns": fallbacks,
            "selected_prediction_turns": predicted, "selected_prediction_calibrated_turns": calibrated,
            "selected_prediction_eligible_turns": eligible}


def telemetry_series(telemetry):
    """Normalize native /context/metrics snapshots without changing clock domains."""
    snapshots = list(telemetry.values()) if isinstance(telemetry, dict) else list(telemetry)
    snapshots = [row for row in snapshots if isinstance(row, dict) and "pool" in row]
    if not snapshots:
        return (), (), {"available": False}
    if len({row["engine_restart_id"] for row in snapshots}) > 1:
        return (), (), {"available": False, "reason": "engine_restarted_during_measurement"}
    start = min(row["observed_time_ns"] for row in snapshots)
    end = max(row["observed_time_ns"] for row in snapshots)
    pools, transfers = {}, {}
    def sample(time_ns, pool):
        pools[time_ns] = {"timestamp": time_ns / 1e9, "gpu_slots": pool["unique_gpu_tokens"],
                          "host_slots": pool["unique_host_tokens"]}
    for row in snapshots:
        sample(row["observed_time_ns"], row["pool"])
        for event in row.get("events", ()):
            at = event.get("observed_time_ns", 0)
            if not start <= at <= end:
                continue
            if event.get("event") == "pool_snapshot":
                sample(at, event)
            elif event.get("event") == "transfer":
                transfers[event["transfer_id"]] = event
    return tuple(pools.values()), tuple(transfers.values()), {
        "available": True, "clock": "server_monotonic", "source": snapshots[-1].get("source", "hardware"),
        "sample_scopes": sorted({s["pool_sample_scope"] for s in snapshots}),
        "events_truncated": any(s.get("events_truncated", False) for s in snapshots),
        "peak_interpretation": "observed_samples_not_allocator_instantaneous_peak"}


def summarize_run(payload, *, occupancy=(), transfers=(), quality=(), ttft_slo=None, tpot_slo=None,
                  response_mode=None, complete_slo=None):
    payload = _dict(payload)
    protocol = payload.get("experiment_protocol", {})
    recorded_slo = protocol.get("slo", {})
    response_mode = protocol.get("response_mode", "buffered") if response_mode is None else response_mode
    ttft_slo = recorded_slo.get("ttft_seconds", 2.0) if ttft_slo is None else ttft_slo
    tpot_slo = recorded_slo.get("tpot_seconds", .1) if tpot_slo is None else tpot_slo
    complete_slo = recorded_slo.get("complete_seconds") if complete_slo is None else complete_slo
    sampled_occupancy, sampled_transfers, telemetry_info = telemetry_series(payload.get("telemetry", ()))
    occupancy = occupancy or sampled_occupancy
    transfers = transfers or sampled_transfers
    report = _dict(payload.get("report", payload))
    turns = report["turns"]
    timings = []
    actual_prefill, truncated, controls, engine_ms, server_ttft, server_tpot = [], [], [], [], [], []
    timing_sources = defaultdict(int)
    latency_pairs = defaultdict(list)
    receipt_rows, receipt_sessions, receipt_branches = [], defaultdict(list), defaultdict(list)
    citations = {"observed_turns": 0, "scope_observed_turns": 0,
                 "unknown_reference_task_ids": [], "out_of_scope_task_ids": []}
    for value in turns:
        row = _dict(value)
        result = _dict(row.get("result") or {})
        timing = _dict(result.get("timing") or {})
        receipt = _dict(result.get("receipt") or {})
        finish = result.get("finish_reason", receipt.get("finish_reason"))
        is_truncated = result.get("truncated", receipt.get("truncated"))
        if is_truncated is None:
            if finish in ("length", "max_tokens"):
                is_truncated = True
            elif finish == "stop":
                is_truncated = False
        # Both streaming clients use the same monotonic chunk timestamps.
        # Legacy/nonstream completion proves response+commit, never TTFT.
        fallback_done = row["t_done"] if row["status"] == "success" else None
        item = TurnTiming(row["conversation_id"], row["task_id"], row["status"],
                          result.get("output_tokens", len(result.get("output_ids", ()))),
                          row["t_arrive"], timing.get("t_first"), timing.get("t_last_token"),
                          timing.get("t_response_done", fallback_done), timing.get("t_commit", fallback_done),
                          timing.get("t_submit"), timing.get("t_admit"), timing.get("t_retire_ack"),
                          timing.get("token_observation", "none"), is_truncated)
        timings.append(item)
        if row["status"] == "success" and item.turn_seconds is not None:
            latency_pairs[row["conversation_id"]].append({"task_id": row["task_id"], "value": item.turn_seconds})
        count = result.get("actual_prefill_tokens", receipt.get("actual_prefill_tokens"))
        if count is not None:
            actual_prefill.append(count)
        prompt_tokens = result.get("input_tokens")
        if receipt and count is not None and prompt_tokens and row["status"] == "success":
            gpu = receipt.get("actual_device_hit_tokens")
            host = receipt.get("actual_host_loaded_tokens")
            observed = {"ratio": count / prompt_tokens, "ttft": item.ttft, "turn_seconds": item.turn_seconds,
                        "cold": (gpu + host == 0 and count >= prompt_tokens) if gpu is not None and host is not None else None}
            receipt_rows.append(observed)
            receipt_sessions[row["conversation_id"]].append(observed)
            branch = result.get("branch_reason") or result.get("action") or "unspecified"
            receipt_branches[branch].append(observed)
        if row["status"] == "success" and is_truncated is not None:
            truncated.append(is_truncated)
        if row["status"] == "success" and "visible_evidence_manifest" in result:
            citations["observed_turns"] += 1
            cited = parse_citations(row["assistant_text"])
            visible = {span["reference_id"] for span in result["visible_evidence_manifest"]}
            if cited - visible:
                citations["unknown_reference_task_ids"].append(row["task_id"])
            if "citation_scope" in result:
                citations["scope_observed_turns"] += 1
                if (cited & visible) - set(result["citation_scope"]):
                    citations["out_of_scope_task_ids"].append(row["task_id"])
        if result.get("control_ms") is not None:
            controls.append(result["control_ms"])
        for field, target in (("engine_elapsed_ms", engine_ms), ("server_ttft_ms", server_ttft), ("tpot_ms", server_tpot)):
            if result.get(field) is not None:
                target.append(result[field])
        timing_sources[result.get("timing_source") or "unavailable"] += 1
    # Fixed cohorts finish at completed_at; open-loop throughput uses its specified
    # arrival window, with the drained cohort separately visible.
    start, end = report["started_at"], report["arrival_window_end"]
    summary = summarize_timings(timings, window_start=start, window_end=max(end, start + 1e-9),
                                ttft_slo=ttft_slo, tpot_slo=tpot_slo,
                                response_mode=response_mode, complete_slo=complete_slo)
    source = "simulation" if payload.get("engine_kind") == "mock" else payload.get("measurement_kind", "unvalidated_hardware_run")
    if any((_dict(_dict(row).get("result") or {})).get("measurement_kind") == "fake_http_correctness" for row in turns):
        source = "simulation"
    quality_rows = list(quality)
    expected_quality = bind_quality_rows(turns, quality_rows)
    scored = [r for r in quality_rows if r.get("scores")]
    dimensions = sorted({k for r in scored for k in r["scores"]})
    missing_quality = sorted(expected_quality.keys() - {r["task_id"] for r in scored})
    quality_simulated = source == "simulation" or any(r.get("generation_measurement_kind") == "simulation" for r in quality_rows)
    quality_status = ("not_validated" if quality_simulated or not scored else
                      "partially_evaluated" if missing_quality else "evaluated")
    pools = {**occupancy_integral(occupancy), "sample_count": len(occupancy),
             "sample_span_seconds": max(r["timestamp"] for r in occupancy) - min(r["timestamp"] for r in occupancy) if occupancy else None}
    nontruncated = sum(t.status == "success" and t.completed_at is not None
                      and t.truncated is False and t.output_tokens > 0 for t in timings)
    normalize = nontruncated and pools["sample_span_seconds"] and not telemetry_info.get("events_truncated")
    resource_efficiency = {"denominator": "nontruncated_completed_turns_including_drain",
        "completion_count": nontruncated, "measurement_span_seconds": pools["sample_span_seconds"],
        "semantic_quality_included": False,
        **{f"{tier}_slot_seconds_per_nontruncated_turn": pools[f"{tier}_slot_seconds"] / nontruncated if normalize else None
           for tier in ("gpu", "host")}}
    return {"metric_schema_version": 2, "measurement_kind": source, "performance_claims_validated": False,
            "time_origin": report["time_origin"], "timing": summary,
            "decisions": decision_summary(turns),
            "cohort": {"scheduled_sessions": report["scheduled_sessions"], "arrived_sessions": report["arrived_sessions"],
                       "unfinished_turns": report["unfinished_turns"], "timed_out_sessions": report["timed_out_sessions"],
                       "drain_seconds": max(0, report["completed_at"] - end)},
            "session_makespan_seconds": distribution(t[1] for t in report["session_makespans"]),
            "session_latency": {
                "equal_session_mean_turn_seconds": mean(mean(t["value"] for t in rows) for rows in latency_pairs.values()) if latency_pairs else None,
                "cumulative_system_wait_seconds": distribution(sum(t["value"] for t in rows) for rows in latency_pairs.values()),
                "scope": "observed_successful_turns_excluding_user_think_time"},
            "control_ms": distribution(controls), "latency_by_session": dict(latency_pairs),
            "engine_elapsed_ms": distribution(engine_ms),
            "server_metrics": {"ttft_ms": distribution(server_ttft), "tpot_ms": distribution(server_tpot),
                               "timing_sources": dict(timing_sources), "end_to_end_client_timing": False},
            "actual_prefill_tokens": {"observed_turns": len(actual_prefill), "total": sum(actual_prefill) if actual_prefill else None},
            "receipt_work": {**receipt_summary(receipt_rows),
                             "cold_definition": "no_actual_gpu_or_host_hit_and_actual_prefill_at_least_prompt",
                             "by_session": {k: receipt_summary(v) for k, v in receipt_sessions.items()},
                             "by_branch": {k: receipt_summary(v) for k, v in receipt_branches.items()}},
            "truncated_answers": {"observed_turns": len(truncated), "count": sum(truncated)},
            "citation_validity": citations,
            # Occupancy timestamps may be engine-local. Integrate only within the
            # sample clock; never extrapolate to a client completion timestamp.
            "occupancy": pools, "resource_efficiency": resource_efficiency,
            "transfers": transfer_totals(transfers) if transfers else (transfer_totals(()) if telemetry_info.get("available") else None),
            "telemetry": telemetry_info,
            "quality": {"status": quality_status, "simulation_scores": quality_simulated,
                        "quality_noninferiority_established": False,
                        "interpretation": "score_summary_only_use_quality_acceptance_for_provenance_and_scope",
                        "expected_successful_tasks": len(expected_quality), "missing_scored_task_ids": missing_quality,
                        "scored_tasks": len(scored), "failed_tasks": sum(r.get("status") == "failed" for r in quality_rows),
                        "dimensions": {key: distribution(r["scores"][key] for r in scored if r["scores"].get(key) is not None) for key in dimensions},
                        "script_history_mismatch_count": sum(r.get("script_history_mismatch") is True for r in scored)},
            "cleanup": payload.get("cleanup", {k: payload.get(k) for k in
                         ("close_error", "undispatched_native_sessions", "cleanup_started_at", "cleanup_finished_at")})}


def paired_quality(candidate, reference, *, dimension="correctness", epsilon=.01, seed=42, samples=5000):
    if dimension != "latency" and any(r.get("generation_measurement_kind") == "simulation" for r in (*candidate, *reference)):
        raise ValueError("simulation answers cannot establish quality noninferiority")
    def keyed(rows):
        result = {}
        for row in rows:
            task_id = row["task_id"]
            if task_id in result:
                raise ValueError("aggregate multiple human raters explicitly before paired comparison")
            score = (row.get("scores") or {}).get(dimension)
            if score is None:
                raise ValueError(f"missing {dimension} score: {task_id}")
            result[task_id] = (row["conversation_id"], score)
        return result
    left, right = keyed(candidate), keyed(reference)
    if left.keys() != right.keys():
        raise ValueError("quality comparison needs identical task coverage")
    a, b = defaultdict(list), defaultdict(list)
    for task_id in sorted(left):
        if left[task_id][0] != right[task_id][0]:
            raise ValueError("paired task has different conversation IDs")
        sid = left[task_id][0]
        a[sid].append(left[task_id][1])
        b[sid].append(right[task_id][1])
    return {"dimension": dimension, **paired_cluster_bootstrap(a, b, epsilon=epsilon, seed=seed, samples=samples)}


def paired_latency(candidate_summary, reference_summary, *, seed=42, samples=5000):
    # Reuse the paired cluster implementation with exact task-ID alignment;
    # negative differences mean faster. No noninferiority quality claim here.
    def rows(summary):
        return [{"conversation_id": sid, "task_id": row["task_id"], "scores": {"latency": row["value"]}}
                for sid, values in summary["latency_by_session"].items() for row in values]
    result = paired_quality(rows(candidate_summary), rows(reference_summary), dimension="latency",
                            epsilon=0, seed=seed, samples=samples)
    result.pop("decision")
    result["interpretation"] = "negative_candidate_minus_reference_is_faster"
    return result


def markdown_report(summary):
    timing = summary["timing"]
    def number(value):
        return "未测量/未配置" if value is None else f"{value:.3f}"

    rows = [
        ("完整响应，会话等权均值 / s", summary["session_latency"]["equal_session_mean_turn_seconds"]),
        ("完整响应 P95 / s", timing["turn_seconds"]["p95"]),
        ("会话累计系统等待均值 / s（不含用户停顿）", summary["session_latency"]["cumulative_system_wait_seconds"]["mean"]),
    ]
    if timing["response_mode"] == "streaming":
        rows.extend([(name, timing[key][quantile]) for name, key, quantile in (
            ("TTFT P50 / s", "ttft_seconds", "median"), ("TTFT P95 / s", "ttft_seconds", "p95"),
            ("TPOT P95 / s", "tpot_seconds", "p95"))])
    rows.append(("非截断、时延达标吞吐 / 请求每秒", timing["latency_slo_goodput_requests_per_second"]))
    for tier in ("gpu", "host"):
        rows.append((f"{tier.upper()} KV占用时间 / 非截断完成轮次（slot·s）",
                     summary["resource_efficiency"][f"{tier}_slot_seconds_per_nontruncated_turn"]))
    lines = ["# ContextKV 目标场景评价", "",
        f"指标版本：2；测量类型：{summary['measurement_kind']}；返回方式：{timing['response_mode']}。",
        f"时间起点：{summary['time_origin']}；声明SLO：{json.dumps(timing['slo'])}。", "",
        "| 主要服务指标 | 数值 |", "|---|---:|",
        *(f"| {label} | {number(value)} |" for label, value in rows), "",
        f"质量评价状态：{summary['quality']['status']}。语义质量独立评价，不计入上述goodput。"]
    quality = summary["quality"]
    if quality["scored_tasks"]:
        labels = ("正确性与必要内容覆盖", "证据支持", "可回答性", "条件/版本", "引用支持")
        lines += ["", "| 质量维度 | 已有评分均值 | 样本数 |", "|---|---:|---:|"]
        for key, label in zip(DIMENSIONS, labels):
            value = quality["dimensions"].get(key)
            if value:
                lines.append(f"| {label} | {number(value['mean'])} | {value['count']} |")
        lines += ["", "这些分数不自动构成独立质量合格或非劣结论。"]
    refs = summary["citation_validity"]
    lines += ["", f"运行状态：{json.dumps(timing['statuses'])}；未完成依赖轮次：{summary['cohort']['unfinished_turns']}；"
        f"截断：{summary['truncated_answers']['count']}/{summary['truncated_answers']['observed_turns']}条有终态观测的回答。",
        f"Goodput窗口内终态不明的完成请求：{timing['unknown_truncation_in_window']}，不计入达标分子。",
        f"引用可见性违规：{len(refs['unknown_reference_task_ids'])}/{refs['observed_turns']}；"
        f"本轮范围违规：{len(refs['out_of_scope_task_ids'])}/{refs['scope_observed_turns']}。编号检查不代表语义支持。",
        "资源效率使用整批运行及收尾的实际采样区间；缺失/截断遥测不补造效率值。"]
    if timing["response_mode"] == "buffered":
        lines.append("当前为完整返回，内部引擎TTFT/TPOT保留在JSON作诊断，不作为HTTP用户首字延迟。")
    lines += ["当前负载点的吞吐不代表最大承载能力；质量、资源与负载须共同匹配。",
              "控制开销、Prefill、缓存命中、传输与动作分布见附带JSON，用于解释机制。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality")
    parser.add_argument("--occupancy")
    parser.add_argument("--transfers")
    parser.add_argument("--reference-quality", help="paired offline quality JSONL; requires --quality and --reference-run")
    parser.add_argument("--reference-run", help="raw reference run for paired complete-turn latency intervals")
    parser.add_argument("--epsilon", type=float, default=.01)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ttft-slo-seconds", type=float)
    parser.add_argument("--tpot-slo-seconds", type=float)
    parser.add_argument("--response-mode", choices=("buffered", "streaming"), help="reuse recorded mode; legacy default is buffered")
    parser.add_argument("--complete-slo-seconds", type=float, help="declared full-response deadline; no inferred default")
    args = parser.parse_args(argv)
    optional = {key: load_jsonl(getattr(args, key)) if getattr(args, key) else () for key in ("quality", "occupancy", "transfers")}
    slo = {"ttft_slo": args.ttft_slo_seconds, "tpot_slo": args.tpot_slo_seconds,
           "response_mode": args.response_mode, "complete_slo": args.complete_slo_seconds}
    summary = summarize_run(json.loads(Path(args.input).read_text()), **optional, **slo)
    if args.reference_quality:
        if not args.quality or not args.reference_run:
            parser.error("--reference-quality requires --quality and --reference-run to bind both answers")
        reference = load_jsonl(args.reference_quality)
        reference_payload = json.loads(Path(args.reference_run).read_text())
        expected_reference = bind_quality_rows(reference_payload.get("report", reference_payload)["turns"], reference, require_answer=True)
        candidate_payload = json.loads(Path(args.input).read_text())
        bind_quality_rows(candidate_payload.get("report", candidate_payload)["turns"], optional["quality"], require_answer=True)
        dimensions = sorted({k for row in optional["quality"] if row.get("scores") for k in row["scores"]})
        summary["paired_quality"] = ({"status": "not_validated_simulation"} if summary["quality"]["simulation_scores"] else
            {"status": "incomplete_quality_coverage"} if summary["quality"]["missing_scored_task_ids"] or set(expected_reference) != {r["task_id"] for r in reference if r.get("scores")} else
            {key: paired_quality(optional["quality"], reference, dimension=key,
                                 epsilon=args.epsilon, seed=args.seed, samples=args.bootstrap_samples) for key in dimensions})
    if args.reference_run:
        reference = summarize_run(json.loads(Path(args.reference_run).read_text()), **slo)
        summary["paired_latency"] = paired_latency(summary, reference, seed=args.seed, samples=args.bootstrap_samples)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    target.with_suffix(".md").write_text(markdown_report(summary))


if __name__ == "__main__":
    main()
