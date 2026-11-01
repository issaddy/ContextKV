"""Prepare and aggregate a blinded, requirement-level review of paired answers.

Reviewers judge facts and contradictions from an offline rubric; this module
does not score answer semantics or supply references to the serving system.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from statistics import mean

from bench.quality import paired_cluster_bootstrap


def prepare_review(predictions, rubric, methods, *, histories=None, seed=9172):
    """Anonymize paired answers and retain each method's own history and evidence, if supplied."""
    if len(methods) != 2 or len(set(methods)) != 2:
        raise ValueError("review requires two distinct methods")
    rows = list(predictions)
    ids = [row["task_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(rubric):
        raise ValueError("prediction and rubric task IDs must match exactly and be unique")
    if histories is not None and set(histories) != set(ids):
        raise ValueError("history and prediction task IDs must match")
    rng = random.Random(seed)
    review, key = [], {}
    for row in rows:
        task = row["task_id"]
        requirements = rubric[task]["required"]
        contradictions = rubric[task]["contradictions"]
        if (not isinstance(requirements, list) or not requirements
                or not isinstance(contradictions, list)
                or any(not isinstance(item, str) or not item.strip() for item in (*requirements, *contradictions))):
            raise ValueError(f"{task}: rubric needs nonempty requirements and string contradiction checks")
        order = list(methods)
        rng.shuffle(order)
        answers = {}
        for field in ("history", "evidence"):
            if any(field in row["modes"][item] for item in methods) and not all(
                field in row["modes"][item] for item in methods
            ):
                raise ValueError(f"{task}: both answers need method-specific {field}")
        for label, method in zip(("A", "B"), order, strict=True):
            value = row["modes"][method]
            status, answer = value["status"], value.get("answer", "")
            if not isinstance(status, str) or not isinstance(answer, str):
                raise ValueError(f"{task}: each answer needs a status and text")
            answers[label] = {"status": status, "text": answer}
            for field in ("history", "evidence"):
                if field in value:
                    if not isinstance(value[field], list):
                        raise ValueError(f"{task}: method-specific {field} must be a list")
                    if field == "history" and histories is not None:
                        raise ValueError(f"{task}: choose shared or method-specific histories")
                    answers[label][field] = value[field]
        review.append({"task_id": task, "question": row["query"],
            "history": histories[task] if histories is not None else [],
            "requirements": requirements, "contradictions": contradictions, **answers})
        changed = row.get("changed_selection")
        if "changed_selection" in row and type(changed) is not bool:
            raise ValueError(f"{task}: changed_selection must be boolean")
        group = ("changed" if changed else "control") if "changed_selection" in row else row.get("group", "all")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{task}: review group must be a nonempty string")
        key[task] = {"conversation_id": row["conversation_id"], "group": group,
                     "requirements": len(requirements), "contradictions": len(contradictions),
                     "A": order[0], "B": order[1]}
    return review, key


def score_review(key, judgments, methods):
    """Report per-task fact coverage and contradiction incidence without merging them."""
    if len(methods) != 2 or len(set(methods)) != 2:
        raise ValueError("review requires two distinct methods")
    judgments = list(judgments)
    ids = [row["task_id"] for row in judgments]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(key):
        raise ValueError("every reviewed task needs exactly one judgment")
    cases = []
    for row in judgments:
        task = row["task_id"]
        mapping = key[task]
        if {mapping["A"], mapping["B"]} != set(methods):
            raise ValueError(f"{task}: key does not contain both methods")
        scores = {}
        for label in ("A", "B"):
            entry = row[label]
            facts, conflicts = entry["requirements"], entry["contradictions"]
            if (not isinstance(facts, list) or len(facts) != mapping["requirements"]
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v not in (0, .5, 1) for v in facts)):
                raise ValueError(f"{task}: {label} requirement scores must be 0, 0.5 or 1")
            if (not isinstance(conflicts, list) or len(conflicts) != mapping["contradictions"]
                    or any(type(v) is not bool for v in conflicts)):
                raise ValueError(f"{task}: {label} contradiction checks must be true or false")
            scores[mapping[label]] = {"coverage": mean(facts),
                                      "contradiction": float(any(conflicts)) if conflicts else None}
        cases.append({"task_id": task, "conversation_id": mapping["conversation_id"],
                      "group": mapping["group"], "scores": scores})
    result = {"scope": "reviewer judgment against offline requirements; not automatic factuality or independent review",
              "methods": list(methods), "cases": cases, "groups": {}}
    for group in ("all", *sorted({case["group"] for case in cases} - {"all"})):
        cohort = cases if group == "all" else [case for case in cases if case["group"] == group]
        dimensions = {}
        for dimension in ("coverage", "contradiction"):
            assessed = [case for case in cohort if all(case["scores"][method][dimension] is not None for method in methods)]
            if not assessed:
                dimensions[dimension] = {"assessed_tasks": 0, "mean": None, "paired": None}
                continue
            values = {method: defaultdict(list) for method in methods}
            for case in assessed:
                for method in methods:
                    values[method][case["conversation_id"]].append(case["scores"][method][dimension])
            paired = paired_cluster_bootstrap(values[methods[1]], values[methods[0]])
            paired.pop("decision")
            paired.pop("epsilon")
            dimensions[dimension] = {"assessed_tasks": len(assessed),
                "mean": {method: mean(case["scores"][method][dimension] for case in assessed) for method in methods},
                "paired": paired}
        result["groups"][group] = {"tasks": len(cohort), "sessions": len({c["conversation_id"] for c in cohort}),
                                   "dimensions": dimensions}
    return result


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="create a blinded review and separate answer key")
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--freeze", type=Path, help="optional prediction manifest containing SHA-256 and row count")
    prepare.add_argument("--histories", type=Path, help="optional JSONL with task_id and actual history")
    prepare.add_argument("--rubric", type=Path, required=True)
    prepare.add_argument("--methods", nargs=2, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=9172)
    score = subparsers.add_parser("score", help="aggregate completed blinded judgments")
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--judgments", type=Path, required=True)
    score.add_argument("--methods", nargs=2, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        data = args.predictions.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if args.freeze:
            frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
            if hashlib.sha256(data).hexdigest() != frozen["sha256"] or len(rows) != frozen["rows"]:
                raise ValueError("predictions do not match their freeze manifest")
        histories = None
        if args.histories:
            history_rows = _read_jsonl(args.histories)
            histories = {row["task_id"]: row["history"] for row in history_rows}
            if len(histories) != len(history_rows):
                raise ValueError("history task IDs must be unique")
        rubric = json.loads(args.rubric.read_text(encoding="utf-8"))["cases"]
        review, key = prepare_review(rows, rubric, args.methods, histories=histories, seed=args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review), encoding="utf-8")
        (args.output / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        template = [{"task_id": row["task_id"], **{label: {"requirements": [None] * len(row["requirements"]),
            "contradictions": [None] * len(row["contradictions"])} for label in ("A", "B")}} for row in review]
        (args.output / "judgments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in template), encoding="utf-8")
        print(f"Prepared {len(review)} blinded cases; keep key.json separate until review is complete.")
    else:
        key = json.loads(args.key.read_text(encoding="utf-8"))
        result = score_review(key, _read_jsonl(args.judgments), args.methods)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({group: value for group, value in result["groups"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
