"""Post-hoc coverage of MultiDoc2Dial's original annotated source spans.

Inputs are a completed official-format prediction JSONL, the task JSONL with
grounding annotations, and the exact passage corpus used for those predictions.
The annotations never enter retrieval or generation. Coverage is evidence
visibility, not proof that an answer is correct or complete.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Passage:
    parent: str
    start: int
    end: int
    text: str


def _rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_passages(path):
    """Index literal passage offsets without conflating passages and parents."""
    passages, parents = {}, defaultdict(list)
    for row in _rows(path):
        pid, parent, text, span = row["passage_id"], row["document_id"], row["text"], row["span"]
        if (not isinstance(pid, str) or not pid or not isinstance(parent, str) or not parent
                or not isinstance(text, str) or len(span) != 2 or any(type(v) is not int for v in span)
                or span[0] < 0 or span[1] < span[0] or span[1] - span[0] != len(text)):
            raise ValueError(f"invalid literal passage span: {pid!r}")
        if pid in passages:
            raise ValueError(f"duplicate passage ID: {pid}")
        passage = Passage(parent, span[0], span[1], text)
        passages[pid] = passage
        parents[parent].append(passage)
    if not passages:
        raise ValueError("passage corpus is empty")
    return passages, parents


def _covered(start, end, intervals):
    """Count the union of intervals clipped to one half-open reference span."""
    merged_end, total = start, 0
    for left, right in sorted((max(start, a), min(end, b)) for a, b in intervals if a < end and start < b):
        if right > merged_end:
            total += right - max(left, merged_end)
            merged_end = right
    return total


def _matching_intervals(grounding, passages):
    start, end, quote = grounding["start"], grounding["end"], grounding["quote"]
    intervals = []
    for passage in passages:
        left, right = max(start, passage.start), min(end, passage.end)
        if left < right and passage.text[left-passage.start:right-passage.start] == quote[left-start:right-start]:
            intervals.append((left, right))
    return intervals


def span_coverage(predictions, references, passages, parents, *, k=None):
    """Score only reference spans verified against the complete original corpus."""
    if k is not None and k < 1:
        raise ValueError("k must be positive")
    tasks, missing, no_grounding, unalignable, total_unaligned = {}, [], [], [], 0
    for task_id, retrieved in predictions.items():
        if task_id not in references:
            missing.append(task_id)
            continue
        if not references[task_id]:
            no_grounding.append(task_id)
            continue
        ids = list(dict.fromkeys(retrieved))[:k]
        absent = set(ids) - passages.keys()
        if absent:
            raise ValueError(f"passages missing from source corpus: {sorted(absent)}")
        selected = defaultdict(list)
        for pid in ids:
            selected[passages[pid].parent].append(passages[pid])
        aligned = []
        seen = set()
        skipped = 0
        for item in references[task_id]:
            parent, start, end, quote = (item.get("passage_id"), item.get("start"),
                                         item.get("end"), item.get("quote"))
            identity = (parent, start, end, quote)
            if identity in seen:
                continue
            seen.add(identity)
            if (not isinstance(parent, str) or type(start) is not int or type(end) is not int
                    or not isinstance(quote, str) or start < 0 or end <= start or len(quote) != end-start):
                skipped += 1
                continue
            size = end - start
            # Do not award coverage merely because unrelated offsets happen to overlap.
            if _covered(start, end, _matching_intervals(item, parents.get(parent, ()))) != size:
                skipped += 1
                continue
            covered = _covered(start, end, _matching_intervals(item, selected.get(parent, ())))
            aligned.append((covered, size))
        total_unaligned += skipped
        if not aligned:
            unalignable.append(task_id)
            continue
        covered_characters = sum(covered for covered, _ in aligned)
        characters = sum(size for _, size in aligned)
        tasks[task_id] = {"aligned_spans": len(aligned), "unaligned_spans": skipped,
            "covered_characters": covered_characters, "characters": characters,
            "character_recall": covered_characters / characters,
            "fully_covered_spans": sum(covered == size for covered, size in aligned),
            "all_aligned_spans_covered": all(covered == size for covered, size in aligned)}
    return {"k": k, "requested_tasks": len(predictions), "evaluated_tasks": len(tasks), "tasks": tasks,
        "missing_reference_task_ids": sorted(missing), "no_grounding_task_ids": sorted(no_grounding),
        "no_alignable_span_task_ids": sorted(unalignable),
        "aligned_spans": sum(row["aligned_spans"] for row in tasks.values()), "unaligned_spans": total_unaligned,
        "mean_character_recall": mean(row["character_recall"] for row in tasks.values()) if tasks else None,
        "fraction_all_aligned_spans_covered": mean(row["all_aligned_spans_covered"] for row in tasks.values()) if tasks else None,
        "interpretation": "Gold spans verified against the supplied original corpus; presence does not prove answer correctness or sufficiency."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="completed official-format predictions JSONL")
    parser.add_argument("--references", required=True, help="task JSONL with held-out grounding annotations")
    parser.add_argument("--corpus", required=True, help="literal passage JSONL used in the completed run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, help="optional passage cutoff; default uses all visible contexts")
    args = parser.parse_args(argv)
    predictions, references = {}, {}
    passages, parents = load_passages(args.corpus)
    for row in _rows(args.input):
        if row["task_id"] in predictions:
            raise ValueError(f"duplicate prediction task ID: {row['task_id']}")
        for context in row["contexts"]:
            pid = context["document_id"]
            if pid not in passages:
                raise ValueError(f"visible passage missing from source corpus: {pid}")
            if context.get("text") != passages[pid].text:
                raise ValueError(f"visible passage text differs from source corpus: {pid}")
        predictions[row["task_id"]] = [context["document_id"] for context in row["contexts"]]
    for row in _rows(args.references):
        if row["task_id"] in references:
            raise ValueError(f"duplicate reference task ID: {row['task_id']}")
        references[row["task_id"]] = row.get("grounding", [])
    result = span_coverage(predictions, references, passages, parents, k=args.k)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
