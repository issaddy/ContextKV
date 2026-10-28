"""Presentation of already-computed metrics; no scoring or execution policy."""
import json

from bench.judge import DIMENSIONS


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
        lines += ["", "分数按任务适用范围与评分口径解释；跨方法结论使用匹配的任务对照。"]
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
