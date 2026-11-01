"""Prepare and score QReCC question-resolution probes without leaking rewrites.

Each case contains earlier *raw* user questions and published reference answers.
Human standalone rewrites are exported separately and opened only for scoring.
This is a fixed-history interpretation diagnostic, not open-corpus QReCC retrieval
or an end-to-end evaluation using the application's own generated answers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from statistics import mean

from bench.quality import paired_cluster_bootstrap


def prepare(rows, *, seed=42, dev_per_bucket=1, test_per_bucket=2, excluded_conversations=()):
    """Freeze disjoint conversations across QReCC sources and turn buckets."""
    if min(dev_per_bucket, test_per_bucket) < 1:
        raise ValueError("each split needs at least one case per bucket")
    excluded = set(excluded_conversations)
    sessions = defaultdict(list)
    for row in rows:
        key = (row["Conversation_source"], row["Conversation_no"])
        if f"{key[0]}:{key[1]}" not in excluded:
            sessions[key].append(row)
    eligible = defaultdict(list)
    for (source, conversation), turns in sorted(sessions.items()):
        turns.sort(key=lambda r: r["Turn_no"])
        history = []
        for row in turns:
            number = row["Turn_no"]
            if number != len(history) + 1 or not row["Question"].strip():
                break
            if history and all(item["answer"].strip() for item in history) and row["Rewrite"].strip():
                bucket = min(5, number)
                case_id = f"{source}:{conversation}:{number}"
                item = {"case_id": case_id, "conversation_id": f"{source}:{conversation}",
                        "source": source, "turn": number, "history": tuple(history),
                        "question": row["Question"]}
                eligible[(source, bucket)].append((item, row["Rewrite"]))
            history.append({"question": row["Question"], "answer": row["Answer"]})
    rng, used, cases, references = random.Random(seed), set(), [], []
    # Select longer-turn strata first so shorter strata cannot exhaust them.
    for key in sorted(eligible, key=lambda item: (item[0], -item[1])):
        if key[0] not in {"nq", "quac", "trec"} or key[1] not in {2, 3, 4, 5}:
            continue
        pool = sorted(eligible[key], key=lambda x: (x[0]["conversation_id"], x[0]["turn"]))
        rng.shuffle(pool)
        selected = []
        for item, rewrite in pool:
            if item["conversation_id"] not in used:
                used.add(item["conversation_id"])
                selected.append((item, rewrite))
                if len(selected) == dev_per_bucket + test_per_bucket:
                    break
        if len(selected) != dev_per_bucket + test_per_bucket:
            raise ValueError(f"insufficient distinct conversations for {key}")
        for position, (item, rewrite) in enumerate(selected):
            cases.append({**item, "split": "dev" if position < dev_per_bucket else "test"})
            references.append({"case_id": item["case_id"], "rewrite": rewrite})
    expected = 12 * (dev_per_bucket + test_per_bucket)
    if len(cases) != expected:
        raise ValueError("the selected data lacks a required QReCC source/turn bucket")
    return cases, references, {"seed": seed, "unit": "conversation", "sources": ["nq", "quac", "trec"],
        "turn_buckets": [2, 3, 4, "5+"], "dev_cases": 12 * dev_per_bucket,
        "test_cases": 12 * test_per_bucket,
        "history": "previous raw questions and published answers; only contiguous, nonempty answered histories",
        "excluded_conversations": len(excluded),
        "purpose": "balanced fixed-history interpretation diagnostic, not natural distribution or P3"}


def term_scores(prediction, reference):
    """SQuAD-style normalized token overlap; no semantic equivalence claim."""
    def tokens(text):
        return [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in {"a", "an", "the"}]
    predicted, expected = tokens(prediction), tokens(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted) if predicted else 0.
    recall = overlap / len(expected) if expected else 0.
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}


def evaluate(cases, references, predictions):
    if (len({item["case_id"] for item in cases}) != len(cases) or
            len({item["conversation_id"] for item in cases}) != len(cases) or
            {item["split"] for item in cases} != {"dev", "test"}):
        raise ValueError("QReCC scoring needs one unique case per conversation in both splits")
    reference = {item["case_id"]: item["rewrite"] for item in references}
    if len(reference) != len(references) or set(reference) != {item["case_id"] for item in cases}:
        raise ValueError("QReCC references must match the frozen cases exactly")
    methods = defaultdict(dict)
    for row in predictions:
        method, case_id = row["method"], row["case_id"]
        if case_id not in reference or case_id in methods[method]:
            raise ValueError("unknown or duplicate QReCC prediction")
        methods[method][case_id] = row
    result = {}
    for method, rows in sorted(methods.items()):
        per_case = {}
        for item in cases:
            pred = rows.get(item["case_id"])
            query = pred.get("query", "") if pred else ""
            per_case[item["case_id"]] = {**term_scores(query, reference[item["case_id"]]),
                "status": pred.get("status", "missing") if pred else "missing"}
        result[method] = {split: {"cases": sum(item["split"] == split for item in cases),
            **{field: mean(per_case[item["case_id"]][field] for item in cases if item["split"] == split)
               for field in ("precision", "recall", "f1")},
            "statuses": dict(Counter(per_case[item["case_id"]]["status"] for item in cases if item["split"] == split))}
            for split in ("dev", "test")}
        result[method]["cases"] = per_case
    pairs = {}
    if "incremental" in result:
        for method in result.keys() - {"incremental"}:
            pairs[method] = {}
            for split in ("dev", "test"):
                selected = [item for item in cases if item["split"] == split]
                candidate = {item["conversation_id"]: [result[method]["cases"][item["case_id"]]["f1"]] for item in selected}
                baseline = {item["conversation_id"]: [result["incremental"]["cases"][item["case_id"]]["f1"]] for item in selected}
                pairs[method][split] = paired_cluster_bootstrap(candidate, baseline)
    return {"scope": "fixed-history human-rewrite token overlap; not semantic, retrieval, or answer quality",
            "methods": result, "paired_f1_vs_incremental": pairs}


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--input", required=True, help="published qrecc-test.json")
    create.add_argument("--output", required=True, help="new directory for inputs and offline labels")
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--dev-per-bucket", type=int, default=1)
    create.add_argument("--test-per-bucket", type=int, default=2)
    create.add_argument("--exclude-cases", help="earlier cases JSONL; exclude its entire conversations")
    score = commands.add_parser("score")
    score.add_argument("--cases", required=True)
    score.add_argument("--references", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
        excluded = (set(json.loads(line)["conversation_id"] for line in
                    Path(args.exclude_cases).read_text(encoding="utf-8").splitlines() if line)
                    if args.exclude_cases else ())
        cases, references, manifest = prepare(rows, seed=args.seed, dev_per_bucket=args.dev_per_bucket,
                                             test_per_bucket=args.test_per_bucket, excluded_conversations=excluded)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        write_jsonl(output / "cases.jsonl", cases)
        write_jsonl(output / "references.jsonl", references)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest))
    else:
        def read(path):
            return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        summary = evaluate(read(args.cases), read(args.references), read(args.predictions))
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({method: {split: values[split] for split in ("dev", "test")}
                          for method, values in summary["methods"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
