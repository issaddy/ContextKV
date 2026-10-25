"""Build a deterministic full-evidence review packet; never infer semantic scores."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from bench.judge import CITATION_PARSER_VERSION, CITATION_PATTERN, parse_citations, rule_audit


ID = re.compile(r"\[(E_[0-9a-f]+)\]")
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
COMMAND = re.compile(r"^(?:ibmcloud|pip3?|pod|carthage|brew|curl|npm|docker|terraform)\b.+")
URL = re.compile(r"https?://[^\s<>`\])]+")
DOTTED_NUMBER = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b")


def key_id(key):
    return json.dumps(key, sort_keys=True)


def source_conditions(data):
    config = data.get("config", {})
    workload = data.get("workload", {})
    policy = data.get("policy", config.get("context", {}))
    return {"baseline": data.get("baseline", "controller"),
        "protocol": data.get("protocol", next(iter(data["report"]["turns"]), {}).get("protocol")),
        "model_fingerprint": data.get("template_fingerprint", data.get("fingerprint")),
        "max_new_tokens": policy.get("max_new_tokens"),
        "concurrency": workload.get("concurrency"), "split": workload.get("split"),
        "input_sha256": data.get("input_sha256"),
        "split_manifest_sha256": data.get("split_manifest_sha256"),
        "policy": policy}


def freeze_cohort(data):
    if data.get("workload", {}).get("split") != "dev":
        raise ValueError("freeze this diagnostic cohort from an explicit development run")
    turns = sorted(data["report"]["turns"], key=lambda t: (t["conversation_id"], t["turn"]))
    return {"schema_version": 1, "source_run_id": data.get("run_id"),
        "selection": "every scheduled task in the source run; no answer-based filtering",
        "conditions": source_conditions(data),
        "tasks": [{"task_id": t["task_id"], "conversation_id": t["conversation_id"],
            "turn": t["turn"], "query": t["online_turn"]["query"],
            "retrieved_contexts": t["online_turn"]["contexts"]} for t in turns]}


def record_for_turn(turn, external_audit=None):
    result = turn.get("result") or {}
    online = turn["online_turn"]
    audit = external_audit or result
    return {"task_id": turn["task_id"], "conversation_id": turn["conversation_id"],
        "query": online["query"], "answer": turn.get("assistant_text") or "",
        "truncated": result.get("truncated"), "actual_history": online["history"],
        "visible_evidence_manifest": audit.get("visible_evidence_manifest", []),
        **({"citation_scope": audit["citation_scope"]} if "citation_scope" in audit else {}),
        "audit_evidence": audit.get("audit_evidence", []),
        "full_audit_present": "visible_evidence_manifest" in audit and "audit_evidence" in audit,
        "input_hash": result.get("input_hash", result.get("receipt", {}).get("input_hash")),
        "output_tokens": len(result["output_ids"]) if "output_ids" in result else None}


def literal_hints(answer, blocks):
    """Literal source locations help review commands/versions; absence is not falsity."""
    fragments = set(INLINE_CODE.findall(answer))
    fragments.update(line.strip().removeprefix("$ ") for line in answer.splitlines()
                     if COMMAND.match(line.strip().removeprefix("$ ")))
    terms = [("command", text) for text in fragments if COMMAND.match(text)]
    terms += [("url", text) for text in set(URL.findall(answer))]
    terms += [("dotted_number", text) for text in set(DOTTED_NUMBER.findall(answer))]
    hints = []
    for kind, text in sorted(set(terms)):
        normalized = " ".join(text.split())
        hits = [{"reference_id": block["reference_id"], "document_id": block["key"]["document_id"]}
                for block in blocks if normalized in " ".join(block["text"].split())]
        hints.append({"kind": kind, "text": text, "literal_source_hits": hits,
                      "semantic_support": None})
    return hints


def review_turn(turn, expected, external_audit=None):
    if turn is None:
        return {**{k: expected[k] for k in ("task_id", "conversation_id", "turn", "query")},
                "status": "missing", "manual_review": {"status": "pending", "claims": []}}
    record = record_for_turn(turn, external_audit)
    bodies = {key_id(b["key"]): b for b in record["audit_evidence"]}
    visible = []
    for span in record["visible_evidence_manifest"]:
        block = bodies.get(key_id(span["key"]))
        if block is not None and span["reference_id"] not in {b["reference_id"] for b in visible}:
            visible.append({**block, "reference_id": span["reference_id"]})
    citation_contexts = [{"reference_id": reference, "answer_start": match.start(),
        "answer_end": match.end(), "nearby_text": record["answer"][max(0, match.start()-350):match.end()+120],
        "format": "single" if match.group(1) == reference else "grouped",
        "support_judgment": "pending"} for match in CITATION_PATTERN.finditer(record["answer"])
        for reference in sorted(parse_citations(match.group(0)))]
    mismatch = []
    if record["query"] != expected["query"]:
        mismatch.append("query_changed")
    if turn["online_turn"]["contexts"] != expected["retrieved_contexts"]:
        mismatch.append("retrieved_contexts_changed")
    audit = rule_audit(record) if record["full_audit_present"] else None
    available = {span["reference_id"] for span in record["visible_evidence_manifest"]}
    return {**record, "turn": turn["turn"], "status": turn["status"], "error": turn.get("error"),
        "cohort_mismatches": mismatch, "rule_audit": audit,
        "visible_blocks": visible, "citation_contexts": citation_contexts,
        "diagnostic_unknown_ids": sorted({item["reference_id"] for item in citation_contexts} - available),
        "literal_hints": literal_hints(record["answer"], visible),
        "manual_review": {"status": "pending", "answerability": None, "groundedness": None,
            "scope_versions": None, "citation_support": None, "claims": [], "notes": []}}


def make_packet(data, cohort, external_audits=()):
    turns = data["report"]["turns"]
    by_id = {turn["task_id"]: turn for turn in turns}
    if len(by_id) != len(turns):
        raise ValueError("duplicate task IDs in saved run")
    external = {row["task_id"]: row for row in external_audits}
    rows = [review_turn(by_id.get(task["task_id"]), task, external.get(task["task_id"]))
            for task in cohort["tasks"]]
    expected = {task["task_id"] for task in cohort["tasks"]}
    lengths = [row["output_tokens"] for row in rows if row.get("output_tokens") is not None]
    complete = [row for row in rows if row.get("full_audit_present")]
    supported_audits = [row for row in complete if "missing_visible_evidence_body" not in row["rule_audit"]["flags"]]
    conditions = source_conditions(data)
    differences = {key: {"cohort": value, "run": conditions.get(key)}
                   for key, value in cohort["conditions"].items()
                   if key != "policy" and value != conditions.get(key)}
    summary = {"expected_tasks": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
        "citation_parser_version": CITATION_PARSER_VERSION,
        "extra_task_ids": sorted(by_id.keys() - expected),
        "changed_task_inputs": [r["task_id"] for r in rows if r.get("cohort_mismatches")],
        "conditions": conditions, "condition_differences": differences,
        "full_audit_tasks": len(complete), "full_audit_without_missing_bodies": len(supported_audits),
        "turns_with_stable_ids": sum(bool(row.get("citation_contexts")) for row in rows),
        "strict_single_id_turns": sum(bool(ID.search(row.get("answer", ""))) for row in rows),
        "grouped_only_id_turns": sum(bool(row.get("citation_contexts")) and not ID.search(row.get("answer", "")) for row in rows),
        "unknown_id_turns": sum(bool(row["rule_audit"]["unknown_citations"]) for row in complete),
        "out_of_scope_id_turns": sum(bool(row["rule_audit"]["out_of_scope_citations"]) for row in complete),
        "diagnostic_unknown_id_turns_including_groups": sum(bool(row.get("diagnostic_unknown_ids")) for row in complete),
        "truncated": sum(row.get("truncated") is True for row in rows),
        "output_tokens_mean": statistics.mean(lengths) if lengths else None,
        "semantic_quality_evaluated": False,
        "manual_review_status": "pending; literal hints and ID visibility are not semantic support"}
    return rows, summary


def compare_packets(paths):
    packets = []
    for path in map(Path, paths):
        packets.append({"path": str(path.resolve()),
            "cohort": json.loads((path/"cohort.json").read_text()),
            "summary": json.loads((path/"summary.json").read_text()),
            "rows": [json.loads(line) for line in (path/"review.jsonl").read_text().splitlines()]})
    first = packets[0]
    if any(packet["cohort"] != first["cohort"] for packet in packets[1:]):
        raise ValueError("comparison requires the same frozen cohort")
    reference = {row["task_id"]: row for row in first["rows"]}
    comparisons = []
    for packet in packets:
        visible_changed, history_changed = [], []
        for row in packet["rows"]:
            prior = reference[row["task_id"]]
            if {key_id(b["key"]) for b in row.get("visible_blocks", [])} != {
                    key_id(b["key"]) for b in prior.get("visible_blocks", [])}:
                visible_changed.append(row["task_id"])
            if row.get("actual_history") != prior.get("actual_history"):
                history_changed.append(row["task_id"])
        comparisons.append({"packet": packet["path"], "summary": packet["summary"],
            "visible_evidence_changed_vs_first": visible_changed,
            "actual_history_changed_vs_first": history_changed})
    return {"cohort_source_run_id": first["cohort"]["source_run_id"],
        "expected_tasks": len(first["cohort"]["tasks"]), "runs": comparisons,
        "semantic_noninferiority_established": False,
        "interpretation": "Inspect condition and actual-history differences: P1 uses reference history; "
            "P2 histories may differ and require conversation-level interpretation. "
            "Single/grouped citation parsing and literal hints do not establish claim support."}


def write_review_packet(output, cohort, rows, summary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cohort.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=2)+"\n")
    (output / "review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    lines = ["# Saved-answer full-evidence review", "", "Automatic checks are not semantic scores.",
             "Missing/failed tasks remain in the cohort. Read actual history for pronouns, not as factual evidence.", "",
             "For each claim: record exact answer text, cited ID(s), supporting/contradicting document ID and text,",
             "then mark supported / partly_supported / not_supported_by_cited / absent_from_visible / unresolved.",
             "Do not infer support from lexical matches; preserve product, version, negation and conditions.", ""]
    for row in rows:
        lines.extend([f"## {row['task_id']}", "", f"Status: {row['status']}", "", row["query"], ""])
        if row["status"] == "missing":
            continue
        lines.extend(["### Answer", "", row["answer"], "", "### Actual history", "",
                      json.dumps(row["actual_history"], ensure_ascii=False, indent=2), "", "### Automatic audit", "",
                      json.dumps(row["rule_audit"], ensure_ascii=False), "", "### Literal review hints (not judgments)", "",
                      json.dumps(row["literal_hints"], ensure_ascii=False, indent=2), "", "### Full visible evidence", ""])
        for block in row["visible_blocks"]:
            lines.extend([f"#### {block['reference_id']} — {block['key']['document_id']}", "",
                          f"Source: {block['source_uri']}; version: {block['key']['version']}", "",
                          block["text"], ""])
        lines.extend(["### Manual review", "", "Pending. Check answerability, claim support, scope/version, citation binding and omissions.", ""])
    (output / "review.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="saved native/controller benchmark JSON")
    source.add_argument("--compare", nargs="+", help="compare existing packet directories on their identical frozen cohort")
    parser.add_argument("--audit", help="optional full audit JSONL for controller runs")
    parser.add_argument("--cohort", help="reuse the source cohort for Q1/Q2; omit only when freezing Q0 dev")
    parser.add_argument("--output", required=True, help="new directory; preserves existing manual annotations")
    args = parser.parse_args()
    if args.compare:
        result = compare_packets(args.compare)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        (output/"comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        return
    data = json.loads(Path(args.run).read_text())
    cohort = json.loads(Path(args.cohort).read_text()) if args.cohort else freeze_cohort(data)
    audit = [json.loads(line) for line in Path(args.audit).read_text().splitlines() if line] if args.audit else ()
    rows, summary = make_packet(data, cohort, audit)
    summary["source_run"] = str(Path(args.run).resolve())
    write_review_packet(args.output, cohort, rows, summary)
    print(json.dumps({key: summary[key] for key in ("expected_tasks", "status_counts", "turns_with_stable_ids",
        "full_audit_without_missing_bodies", "truncated", "output_tokens_mean")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
