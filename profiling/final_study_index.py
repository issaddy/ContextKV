"""Report benchmark quality, latency, tokens and cache/resource measurements."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


METHOD_NAMES = {
    "RAG_native": "普通文档问答",
    "LlamaIndex_native": "LlamaIndex",
    "context_cache_demand": "ContextKV",
    "context_native": "上下文管理＋原生缓存",
    "context_cache_order": "上下文管理＋缓存感知排列",
    "Rule_contextpilot_joint_demand": "规则阅读决策＋缓存协同",
    "context_no_fact_reuse": "关闭跨轮事实复用",
}
REFERENCES = {
    "context_cache_demand": (
        "RAG_native", "LlamaIndex_native", "context_cache_order",
        "Rule_contextpilot_joint_demand", "context_no_fact_reuse"),
    "context_cache_order": ("context_native",),
}


def comparisons(rows):
    groups = {}
    for row in rows:
        key = row["dataset"], row["split"], row["protocol"], row["concurrency"], row["repetition"]
        groups.setdefault(key, {})[row["method"]] = row
    result = []
    for (dataset, split, protocol, concurrency, repetition), methods in groups.items():
        for candidate, references in REFERENCES.items():
            if candidate not in methods:
                continue
            right = methods[candidate]
            for reference in references:
                if reference not in methods:
                    continue
                left = methods[reference]
                quality = (right["quality"]-left["quality"]
                           if right["quality"] is not None and left["quality"] is not None else None)
                ratio = (right["p95_complete_ms"]/left["p95_complete_ms"]
                         if right["p95_complete_ms"] is not None and left["p95_complete_ms"] else None)
                result.append(dict(dataset=dataset, split=split, protocol=protocol, concurrency=concurrency, repetition=repetition,
                    candidate=candidate, reference=reference, quality_delta=quality,
                    p95_reduction_percent=(1-ratio)*100 if ratio is not None else None))
    return result


def markdown(report):
    lines = ["# ContextKV 评测结果", "",
        "| Benchmark | 历史 | 并发 | 方法 | 完成/声明 | 质量 | P95 秒 | 输入 token | 输出 token | GPU GiB·秒 | Host GiB·秒 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        cost, resources = row["cost"], row.get("online_occupancy", {})
        p95 = row["p95_complete_ms"]
        lines.append(f"| {row['dataset']} | {row['protocol']} | {row['concurrency']} | "
            f"{METHOD_NAMES.get(row['method'], row['method'])} | {row['completed_qa']}/{row['requested_qa']} | "
            f"{row['quality']} | {p95/1000 if p95 is not None else None} | "
            f"{cost.get('total_input_tokens')} | {cost.get('total_output_tokens')} | "
            f"{resources.get('gpu_gib_seconds')} | {resources.get('host_gib_seconds')} |")
    lines += ["", "| Benchmark | 历史 | 并发 | 方法 / 参照 | 质量差 | P95 降幅 % |",
              "|---|---|---:|---|---:|---:|"]
    for row in report["comparisons"]:
        lines.append(f"| {row['dataset']} | {row['protocol']} | {row['concurrency']} | "
            f"{METHOD_NAMES.get(row['candidate'], row['candidate'])} / "
            f"{METHOD_NAMES.get(row['reference'], row['reference'])} | "
            f"{row['quality_delta']} | {row['p95_reduction_percent']} |")
    lines += ["", "| Benchmark | 历史 | 并发 | 方法 | 场景维度 | 场景 | 完成/声明 | 质量 | P95 秒 |",
              "|---|---|---:|---|---|---|---:|---:|---:|"]
    for row in report["results"]:
        for dimension, groups in row.get("groups", {}).items():
            for label, group in groups.items():
                p95 = group["p95_complete_ms"]
                lines.append(f"| {row['dataset']} | {row['protocol']} | {row['concurrency']} | "
                    f"{METHOD_NAMES.get(row['method'], row['method'])} | "
                    f"{dimension} | {label} | {group['completed']}/{group['declared']} | "
                    f"{group['quality']} | {p95/1000 if p95 is not None else None} |")
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, help="NAME=workflow.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = []
    for item in args.artifact:
        _, pattern = item.split("=", 1)
        for path in sorted(glob.glob(pattern)):
            value = json.loads(Path(path).read_text())
            rows.extend(value["results"])
    report = {"results": rows, "comparisons": comparisons(rows)}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (root/"report.md").write_text(markdown(report))
    return report


if __name__ == "__main__":
    main()
