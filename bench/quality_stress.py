"""Materialize the independent fictional multi-turn QA diagnostic workload.

The serving tasks and corpus contain no reference answers or support labels.
Those are written separately and are only meant for post-generation scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def materialize(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    documents = data["documents"]
    document_ids = {row["id"] for row in documents}
    if len(document_ids) != len(documents):
        raise ValueError("document IDs must be unique")
    sessions = data["sessions"]
    if len({row["id"] for row in sessions}) != len(sessions):
        raise ValueError("conversation IDs must be unique")
    output.mkdir(parents=True, exist_ok=False)
    corpus = [{"_id": row["id"], "document_id": row["id"], "version": row["version"],
               "title": row["title"], "text": row["text"],
               "source": f"synthetic://conversation-quality-stress/{row['id']}"}
              for row in documents]
    tasks, references, labels = [], [], []
    split = {"dev": [], "test": []}
    for session in sessions:
        sid, partition = session["id"], session["split"]
        split[partition].append(sid)
        history = []
        for number, turn in enumerate(session["turns"], 1):
            if not set(turn["support"]) <= document_ids:
                raise ValueError(f"{sid} turn {number} names an unknown support passage")
            task_id = f"{sid}<::>{number}"
            question = {"speaker": "user", "text": turn["question"]}
            task = {"dataset": {"name": data["name"], "revision": data["revision"]},
                    "conversation_id": sid, "task_id": task_id, "turn": number,
                    "Collection": "synthetic/conversation-quality-stress",
                    "input": [*history, question], "contexts": []}
            tasks.append(task)
            references.append({**task, "targets": [{"text": turn["reference"]}]})
            labels.append({"task_id": task_id, "conversation_id": sid, "tags": turn["tags"],
                           "support": turn["support"], "reference": turn["reference"]})
            # P3 ignores these placeholders and consumes actual committed answers.
            history += [question, {"speaker": "assistant", "text": ""}]
    for name, rows in (("corpus.jsonl", corpus), ("tasks.jsonl", tasks),
                       ("references.jsonl", references), ("labels.jsonl", labels)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                   encoding="utf-8")
    (output / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qrels = ["query-id\tcorpus-id\tscore\n"]
    qrels += [f"{row['task_id']}\t{pid}\t1\n" for row in labels for pid in row["support"]]
    (output / "qrels.tsv").write_text("".join(qrels), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "assets/conversation_quality_stress.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)


if __name__ == "__main__":
    main()
