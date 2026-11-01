"""Prepare and score an offline paired review of interpreted user requests.

Reviewers score task coverage and conditions separately on a 0, 0.5, 1 scale.
The review file contains anonymous A/B interpretations; the separate key is
only needed after scoring. This is an evaluation tool, not a runtime policy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from bench.quality import paired_cluster_bootstrap


DIMENSIONS = ("task", "conditions")


def prepare_review(predictions, rubric, methods, *, seed=9172):
    if len(methods) != 2 or len(set(methods)) != 2:
        raise ValueError("review requires two distinct method names")
    rows = list(predictions)
    ids = [row["task_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(rubric):
        raise ValueError("prediction and rubric task IDs must match exactly and be unique")
    rng = random.Random(seed)
    review, key = [], {}
    for row in rows:
        task_id = row["task_id"]
        order = list(methods)
        rng.shuffle(order)
        interpretations = {}
        for label, method in zip(("A", "B"), order, strict=True):
            value = row[method]
            if not isinstance(value.get("text"), str) or not isinstance(value.get("status"), str):
                raise ValueError(f"{task_id}: interpretation needs text and status")
            interpretations[label] = {"status": value["status"], "text": value["text"]}
        review.append({"task_id": task_id, "conversation_id": row["conversation_id"],
                       "question": row["question"], "expected": rubric[task_id], **interpretations})
        key[task_id] = {"conversation_id": row["conversation_id"], "A": order[0], "B": order[1]}
    return review, key


def score_review(key, judgments, methods):
    if len(methods) != 2 or len(set(methods)) != 2:
        raise ValueError("review requires two distinct method names")
    judgments = list(judgments)
    ids = [row["task_id"] for row in judgments]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(key):
        raise ValueError("every reviewed task needs exactly one judgment")
    by_method = {method: {dimension: defaultdict(list) for dimension in (*DIMENSIONS, "combined")}
                 for method in methods}
    for row in judgments:
        task_id = row["task_id"]
        mapping = key[task_id]
        if {mapping["A"], mapping["B"]} != set(methods):
            raise ValueError(f"{task_id}: key does not contain both methods")
        for label in ("A", "B"):
            scores = row[label]
            values = []
            for dimension in DIMENSIONS:
                value = scores[dimension]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, .5, 1):
                    raise ValueError(f"{task_id}: {label}.{dimension} must be 0, 0.5 or 1")
                values.append(float(value))
                by_method[mapping[label]][dimension][mapping["conversation_id"]].append(float(value))
            by_method[mapping[label]]["combined"][mapping["conversation_id"]].append(mean(values))
    summary = {}
    for dimension in (*DIMENSIONS, "combined"):
        values = {method: by_method[method][dimension] for method in methods}
        means = {method: mean(score for session in values[method].values() for score in session)
                 for method in methods}
        paired = paired_cluster_bootstrap(values[methods[1]], values[methods[0]])
        paired.pop("decision")
        paired.pop("epsilon")
        summary[dimension] = {"mean": means, "paired": paired}
    return {"scope": "reviewer-scored task coverage and conditions; no automatic factual or answer quality judgment",
            "methods": list(methods), "tasks": len(judgments),
            "sessions": len({value["conversation_id"] for value in key.values()}), "dimensions": summary}


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="write anonymous review cases and a separate key")
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--rubric", type=Path, required=True)
    prepare.add_argument("--methods", nargs=2, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=9172)
    score = subparsers.add_parser("score", help="score completed A/B judgments")
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--judgments", type=Path, required=True)
    score.add_argument("--methods", nargs=2, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        rubric = json.loads(args.rubric.read_text(encoding="utf-8"))["cases"]
        review, key = prepare_review(_read_jsonl(args.predictions), rubric, args.methods, seed=args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "review.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in review), encoding="utf-8")
        (args.output / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        template = [{"task_id": row["task_id"],
                     **{label: {dimension: None for dimension in DIMENSIONS} for label in ("A", "B")}}
                    for row in review]
        (args.output / "judgments.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in template), encoding="utf-8")
        print(f"Prepared {len(review)} cases; keep key.json separate until the review is complete.")
    else:
        key = json.loads(args.key.read_text(encoding="utf-8"))
        report = score_review(key, _read_jsonl(args.judgments), args.methods)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
