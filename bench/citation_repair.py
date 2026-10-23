"""Optional offline citation rebinding from complete grounding diagnostics.

The answer wording and evidence remain fixed. This is an experimental repair,
not an independently validated quality score or an online default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .judge import CITATION_PATTERN


def repair_record(record, *, support_threshold=.9, cited_ceiling=.2):
    if not 0 <= cited_ceiling < support_threshold <= 1:
        raise ValueError("citation thresholds require 0 <= cited < support <= 1")
    answer = record["answer"]
    changes = []
    for claim in record["claims"]:
        if (claim["finding"] != "supported_elsewhere_citation_mismatch"
                or claim["kind"] != "prose" or claim.get("contains_inline_code")
                or claim["unknown_citations"]):
            continue
        pairs = claim["pairs"]
        # Model coverage is part of the input contract. A truncated source must
        # not become either negative evidence or a replacement citation.
        if not pairs or any(pair["status"] != "scored" for pair in pairs):
            continue
        cited = [pair for pair in pairs if pair["cited"]]
        alternatives = [pair for pair in pairs if not pair["cited"]]
        if not cited or not alternatives:
            continue
        best = min(alternatives, key=lambda pair: (-pair["probability"], pair["reference_id"]))
        old_max = max(pair["probability"] for pair in cited)
        if old_max > cited_ceiling or best["probability"] < support_threshold:
            continue
        start, end = claim["start"], claim["end"]
        original = answer[start:end]
        if not 0 <= start < end <= len(answer) or original != claim["text"]:
            raise ValueError("grounding claim offsets do not match its original answer")
        replacement = CITATION_PATTERN.sub("[" + best["reference_id"] + "]", original)
        if replacement == original:
            continue
        changes.append({"start": start, "end": end, "before": original, "after": replacement,
                        "old_reference_ids": claim["citations"],
                        "new_reference_id": best["reference_id"],
                        "new_source_uri": best["source_uri"],
                        "old_max_support_probability": old_max,
                        "new_support_probability": best["probability"]})
    changes.sort(key=lambda item: item["start"])
    if any(left["end"] > right["start"] for left, right in zip(changes, changes[1:])):
        raise ValueError("grounding claim offsets overlap")
    revised = answer
    for change in reversed(changes):
        revised = revised[:change["start"]] + change["after"] + revised[change["end"]:]
    return {"task_id": record["task_id"], "conversation_id": record.get("conversation_id"),
            "original_answer": answer, "repaired_answer": revised, "changes": changes,
            "wording_changed": False, "new_evidence_added": False,
            "quality_validated": False, "policy": "visible_source_citation_rebinding_v1",
            "support_threshold": support_threshold, "cited_ceiling": cited_ceiling,
            "thresholds_calibrated": False,
            "scope": "offline model-guided repair; independent evaluation required"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support-threshold", type=float, default=.9)
    parser.add_argument("--cited-ceiling", type=float, default=.2)
    args = parser.parse_args(argv)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.grounding).open() as source, target.open("x") as output:
        for line in source:
            if line.strip():
                result = repair_record(json.loads(line), support_threshold=args.support_threshold,
                                       cited_ceiling=args.cited_ceiling)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
