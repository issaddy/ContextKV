"""Offline format/length diagnostics for saved benchmarks, not semantic scoring."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

from bench.judge import CITATION_PARSER_VERSION, parse_citations


STRICT_ID = re.compile(r"\[(E_[0-9a-f]+)\]")
LOOSE_ID = re.compile(r"\bE_[0-9a-fA-F]+\b")
NUMERIC_CITATION = re.compile(r"\[(?:(?:Evidence|Source)\s+)?\d+(?:\s*[,;]\s*\d+)*\]", re.I)
MARKDOWN_LINK = re.compile(r"\[[^\]\n]+\]\((https?://[^\s)]+)\)")
URL = re.compile(r"https?://[^\s<>`]+")
FENCED_CODE = re.compile(r"```[^\n]*\n.*?(?:```|\Z)", re.S)


def summarize(rows):
    lengths = [row["output_tokens"] for row in rows]
    return {
        "turns": len(rows),
        "truncated": sum(row["truncated"] for row in rows),
        "format_classes": dict(Counter(row["format_class"] for row in rows)),
        "parsed_id_turns_including_groups": sum(bool(row["parsed_ids"]) for row in rows),
        "truncation_by_format": dict(Counter(
            f"{'truncated' if row['truncated'] else 'complete'}:{row['format_class']}" for row in rows)),
        "loose_id_not_strict_turns": sum(bool(row["loose_ids_not_strict"]) for row in rows),
        "numeric_citation_turns": sum(bool(row["numeric_citations"]) for row in rows),
        "any_url_turns": sum(bool(row["url_count"]) for row in rows),
        "url_only_in_fenced_code_turns": sum(row["url_count"] > 0 and not row["prose_url_count"] for row in rows),
        "actions": dict(Counter(row["action"] for row in rows)),
        "output_tokens": {"min": min(lengths), "median": statistics.median(lengths),
                          "mean": statistics.mean(lengths), "max": max(lengths)} if lengths else None,
    }


def diagnose(path):
    path = Path(path)
    data = json.loads(path.read_text())
    rows, sessions = [], defaultdict(list)
    for turn in data["report"]["turns"]:
        if turn["status"] != "success":
            continue
        answer, result = turn["assistant_text"], turn["result"]
        strict = sorted(set(STRICT_ID.findall(answer)))
        loose = sorted(set(LOOSE_ID.findall(answer)) - set(strict))
        prose = FENCED_CODE.sub("", answer)
        links = MARKDOWN_LINK.findall(prose)
        first = STRICT_ID.search(answer)
        row = {"task_id": turn["task_id"], "conversation_id": turn["conversation_id"],
            "turn": turn["turn"], "query": turn["online_turn"]["query"],
            "input_hash": result["receipt"]["input_hash"], "action": result["action"],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "truncated": result["truncated"], "stable_ids": strict,
            "parsed_ids": sorted(parse_citations(answer)),
            "loose_ids_not_strict": loose, "numeric_citations": NUMERIC_CITATION.findall(prose),
            "markdown_links": links, "url_count": len(URL.findall(answer)),
            "prose_url_count": len(URL.findall(prose)),
            "first_citation_character_fraction": first.start() / len(answer) if first else None,
            "format_class": "stable_id" if strict else "markdown_link_only" if links else "neither",
            "citation_support_verified": False}
        rows.append(row)
        sessions[turn["conversation_id"]].append(row)
    return {"benchmark_path": str(path.resolve()), "engine_kind": data["engine_kind"],
            "status_counts": dict(Counter(t["status"] for t in data["report"]["turns"])),
            "summary": summarize(rows),
            "sessions": [{"conversation_id": sid, **summarize(items)} for sid, items in sorted(sessions.items())],
            "turns": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [diagnose(path) for path in args.benchmark]
    comparisons = []
    for first, second in zip(runs, runs[1:]):
        left = {row["task_id"]: row for row in first["turns"]}
        right = {row["task_id"]: row for row in second["turns"]}
        common = sorted(left.keys() & right.keys())
        comparisons.append({"first": first["benchmark_path"], "second": second["benchmark_path"],
            "common_tasks": len(common),
            "identical_input_tasks": sum(left[k]["input_hash"] == right[k]["input_hash"] for k in common),
            "same_action_tasks": sum(left[k]["action"] == right[k]["action"] for k in common),
            "causal_per_turn_comparison": False})
    output = {"schema_version": 1, "semantic_quality_evaluated": False,
        "citation_parser_version": CITATION_PARSER_VERSION,
        "scope": "saved benchmark answer syntax and length; no full-visible evidence support verification",
        "definitions": {"stable_id": STRICT_ID.pattern, "loose_id": LOOSE_ID.pattern,
            "numeric_citation": NUMERIC_CITATION.pattern,
            "markdown_link_only": "contains Markdown HTTP(S) link outside fenced code, but no strict E_ID; not verified support",
            "neither": "neither strict E_ID nor Markdown HTTP(S) link; may still contain prose URLs",
            "url_counts": "syntactic URLs, including placeholders/localhost; never counted as evidence support"},
        "runs": runs, "comparisons": comparisons}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
