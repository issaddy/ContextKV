"""Prepare RECOR conversations for existing P2/P3 and retrieval evaluators.

Reference answers and document labels are written separately. P2/P3 replace
the empty assistant slots with their own committed answers; this export is not
an evaluation of RECOR's original retrieval protocol by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(cases, documents, *, revision, seed=42, dev_fraction=.2):
    """Keep original documents and user turns; put all labels in offline files."""
    if not revision or not 0 < dev_fraction < 1:
        raise ValueError("revision and a dev fraction in (0, 1) are required")
    corpus, ids = [], set()
    for row in documents:
        doc_id, text = row.get("doc_id"), row.get("content")
        if not isinstance(doc_id, str) or not doc_id or doc_id in ids or not isinstance(text, str) or not text:
            raise ValueError("RECOR documents need unique IDs and original nonempty text")
        ids.add(doc_id)
        corpus.append({"passage_id": doc_id, "document_id": doc_id, "text": text,
                       "source_uri": f"recor://{doc_id}"})
    if not corpus:
        raise ValueError("RECOR corpus is empty")
    tasks, references, qrels, sessions = [], [], [], set()
    for case in cases:
        sid, collection = str(case["id"]), str(case["task"])
        if not sid or not collection or sid in sessions:
            raise ValueError("RECOR conversation IDs must be unique and nonempty")
        sessions.add(sid)
        turns = case["turns"]
        if not turns:
            raise ValueError("RECOR conversation must contain turns")
        history = []
        for number, turn in enumerate(turns, 1):
            if turn["turn_id"] != number or not isinstance(turn["query"], str) or not turn["query"]:
                raise ValueError("RECOR turns must be consecutive with nonempty queries")
            task_id = f"{sid}:{number}"
            inputs = [*history, {"speaker": "user", "text": turn["query"]}]
            common = {"task_id": task_id, "conversation_id": sid, "turn": number, "Collection": collection}
            tasks.append({**common, "dataset": {"name": "recor", "revision": revision},
                          "input": inputs, "contexts": []})
            references.append({**common, "targets": [{"text": turn["answer"]}]})
            qrels.extend((task_id, doc_id, 1) for doc_id in dict.fromkeys(turn["gold_doc_ids"]))
            history = [*inputs, {"speaker": "assistant", "text": ""}]
    if not tasks:
        raise ValueError("RECOR benchmark is empty")
    # Freeze conversations, not turns, so no session crosses the split.
    shuffled = sorted(sessions)
    random.Random(seed).shuffle(shuffled)
    dev_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * dev_fraction))) if len(shuffled) > 1 else 0
    split = {"unit": "conversation", "seed": seed, "dev_fraction": dev_fraction,
             "dev": sorted(shuffled[:dev_count]), "test": sorted(shuffled[dev_count:])}
    missing = sorted({doc_id for _, doc_id, _ in qrels if doc_id not in ids})
    manifest = {"dataset": {"name": "recor", "revision": revision}, "conversations": len(sessions),
                "turns": len(tasks), "documents": len(corpus), "missing_gold_document_ids": missing,
                "scope": "P2/P3 and retrieval inputs; reference answers and qrels are offline only"}
    return corpus, tasks, references, qrels, split, manifest


def write_prepared(output, prepared):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    corpus, tasks, references, qrels, split, manifest = prepared
    for name, rows in (("corpus.jsonl", corpus), ("tasks.jsonl", tasks), ("references.jsonl", references)):
        with (output / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "qrels.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("query-id", "corpus-id", "score"))
        writer.writerows(qrels)
    for name, value in (("split.json", split), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="RECOR domain benchmark JSONL")
    parser.add_argument("--corpus", required=True, help="matching RECOR domain documents JSONL")
    parser.add_argument("--revision", required=True, help="revision actually downloaded from the public dataset")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=.2)
    args = parser.parse_args(argv)
    prepared = prepare(read_jsonl(args.benchmark), read_jsonl(args.corpus), revision=args.revision,
                       seed=args.seed, dev_fraction=args.dev_fraction)
    write_prepared(args.output, prepared)
    print(json.dumps(prepared[-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
