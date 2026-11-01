"""Prepare TopiOCQA conversations and exact author passage labels for evaluation.

The normal task file contains no gold passages or current reference answers. P3
replaces the published assistant history with answers generated in the run. A
separate oracle task file exposes the author's single gold passage for P1 only;
it is not an open-retrieval benchmark. The official Wikipedia TSV can be
converted independently, without constructing a corpus from gold labels.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
import sys


REVISION = "zenodo-6193037-dev"
EVALUATOR_COMMIT = "49b710e9746b32bad2296043e1e2fc4a1532bfdf"
EVALUATOR_SHA256 = "e9d2bfae2a01cc08422db0f793e50fb12aacf79870ea9543577dca613efd3fa9"


def prepare(questions, retrieval_labels, *, seed=42, dev_conversations=41):
    """Join author turn labels, preserving their exact passage IDs and text."""
    gold, passage_bodies = {}, {}
    for row in retrieval_labels:
        key = (row["conv_id"], row["turn_id"])
        if key in gold or len(row["positive_ctxs"]) != 1:
            raise ValueError(f"retrieval label must have one unique positive: {key}")
        passage = row["positive_ctxs"][0]
        if not str(passage.get("passage_id", "")).strip() or not passage.get("text"):
            raise ValueError(f"missing gold passage ID or text: {key}")
        passage_id = str(passage["passage_id"])
        body = (passage.get("title", ""), passage["text"])
        if passage_id in passage_bodies and passage_bodies[passage_id] != body:
            raise ValueError(f"one gold passage ID has conflicting original bodies: {passage_id}")
        passage_bodies[passage_id] = body
        gold[key] = row
    sessions = {}
    for row in questions:
        sessions.setdefault(row["Conversation_no"], []).append(row)
    if not 0 < dev_conversations < len(sessions):
        raise ValueError("development and test splits each need conversations")
    names = sorted(sessions)
    random.Random(seed).shuffle(names)
    dev = set(names[:dev_conversations])
    split = {"dev": [f"topiocqa:{n}" for n in sorted(dev)],
             "test": [f"topiocqa:{n}" for n in sorted(set(names) - dev)]}
    tasks, oracle, references, qrels = [], [], [], []
    seen = set()
    dataset = {"name": "topiocqa", "revision": REVISION}
    for conversation in sorted(sessions):
        turns = sorted(sessions[conversation], key=lambda row: row["Turn_no"])
        history, author_context = [], []
        sid = f"topiocqa:{conversation}"
        for turn, row in enumerate(turns, 1):
            key = (conversation, turn)
            if row["Turn_no"] != turn or row["Context"] != author_context:
                raise ValueError(f"noncontiguous or inconsistent author history: {key}")
            if key not in gold or key in seen:
                raise ValueError(f"missing or duplicate retrieval label: {key}")
            label, passage = gold[key], gold[key]["positive_ctxs"][0]
            if label.get("answers", [None])[0] != row["Answer"]:
                raise ValueError(f"answer differs between author files: {key}")
            seen.add(key)
            task_id = f"{sid}<::>{turn}"
            current = {"speaker": "user", "text": row["Question"]}
            task = {"dataset": dataset, "conversation_id": sid, "task_id": task_id,
                    "turn": turn, "Collection": "wikipedia/topiocqa",
                    "input": [*history, current], "contexts": []}
            passage_id = str(passage["passage_id"])
            context = {"document_id": passage_id, "text": passage["text"],
                       "title": passage.get("title", "")}
            tasks.append(task)
            oracle.append({**task, "contexts": [context]})
            answers = [row["Answer"], *(a.get("Answer", "") for a in row.get("Additional_answers", []))]
            # Keep the annotation order and duplicate answers: the author's
            # leave-one-out scoring treats them as distinct annotations.
            references.append({**task, "targets": [{"text": a} for a in answers],
                               "topic": row["Topic"], "topic_section": row.get("Topic_section", "")})
            qrels.append((task_id, passage_id))
            history.extend((current, {"speaker": "assistant", "text": row["Answer"]}))
            author_context.extend((row["Question"], row["Answer"]))
    if seen != set(gold):
        raise ValueError("retrieval labels and author questions cover different turns")
    return tasks, oracle, references, qrels, split


def _write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize(questions_path, labels_path, output, *, seed=42, dev_conversations=41):
    questions_data, labels_data = Path(questions_path).read_bytes(), Path(labels_path).read_bytes()
    tasks, oracle, references, qrels, split = prepare(json.loads(questions_data), json.loads(labels_data),
        seed=seed, dev_conversations=dev_conversations)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("tasks.jsonl", tasks), ("oracle-tasks.jsonl", oracle), ("references.jsonl", references)):
        _write_jsonl(output / name, rows)
    with (output / "qrels.tsv").open("x", encoding="utf-8") as stream:
        stream.write("query-id\tcorpus-id\tscore\n")
        for task_id, passage_id in qrels:
            stream.write(f"{task_id}\t{passage_id}\t1\n")
    (output / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"dataset": "topiocqa", "revision": REVISION, "questions_sha256": sha256(questions_data).hexdigest(),
        "retrieval_labels_sha256": sha256(labels_data).hexdigest(), "seed": seed, "turns": len(tasks),
        "dev_conversations": len(split["dev"]), "test_conversations": len(split["test"]),
        "qrels": "author's first positive passage ID, one per turn; exact passage hit is the author metric",
        "oracle": "author gold passage supplied for P1 diagnostic only; not open retrieval",
        "p3_history": "actual generated answers replace published assistant history"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def convert_wiki(tsv_path, output, *, oracle_tasks=None):
    """Stream the author's complete TSV; do not derive a corpus from labels."""
    csv.field_size_limit(sys.maxsize)
    count = 0
    expected, matched = {}, set()
    if oracle_tasks is not None:
        with Path(oracle_tasks).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                context = row["contexts"][0]
                passage_id = context["document_id"]
                body = (context["title"], context["text"])
                if passage_id in expected and expected[passage_id] != body:
                    raise ValueError(f"conflicting gold bodies: {passage_id}")
                expected[passage_id] = body
    with Path(tsv_path).open(encoding="utf-8", newline="") as source, Path(output).open("x", encoding="utf-8") as dest:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["id", "text", "title"]:
            raise ValueError("expected the official id/text/title Wikipedia TSV")
        for row in reader:
            passage_id = row["id"]
            if not passage_id or not row["text"]:
                raise ValueError(f"missing ID or body in Wikipedia TSV at record {count + 1}")
            if passage_id in expected:
                if passage_id in matched or expected[passage_id] != (row["title"], row["text"]):
                    raise ValueError(f"gold passage ID is duplicated or has different original content: {passage_id}")
                matched.add(passage_id)
            record = {"passage_id": passage_id, "document_id": passage_id, "text": row["text"],
                      "title": row["title"], "source": f"topiocqa://wikipedia/{passage_id}"}
            dest.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if expected.keys() - matched:
        raise ValueError(f"{len(expected.keys() - matched)} gold passage IDs are missing from Wikipedia TSV")
    return count


def score_official(references, predictions, evaluator_path, *, conversations=None):
    """Call the pinned author's leave-one-out scorer; missing turns remain zero."""
    evaluator_path = Path(evaluator_path)
    if sha256(evaluator_path.read_bytes()).hexdigest() != EVALUATOR_SHA256:
        raise ValueError("TopiOCQA evaluator does not match the pinned author source")
    spec = importlib.util.spec_from_file_location("topiocqa_author_evaluator", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_by_id = {row["task_id"]: row for row in references}
    if len(reference_by_id) != len(references):
        raise ValueError("reference task IDs must be unique")
    predictions_by_id = {row["task_id"]: row for row in predictions}
    if len(predictions_by_id) != len(predictions) or set(predictions_by_id) - set(reference_by_id):
        raise ValueError("prediction task IDs must be unique and belong to the references")
    per_turn, missing = {}, []
    for task_id, reference in reference_by_id.items():
        if conversations is not None and reference["conversation_id"] not in conversations:
            continue
        answers = [target["text"] for target in reference["targets"]]
        if not answers:
            raise ValueError(f"missing author annotations: {task_id}")
        if task_id not in predictions_by_id:
            per_turn[task_id] = {"em": 0., "f1": 0.}
            missing.append(task_id)
        else:
            answer = predictions_by_id[task_id]["predictions"][0]["text"]
            per_turn[task_id] = module.TopiOCQAEvaluator._compute_turn_score(answers, answer)
    if not per_turn:
        raise ValueError("no reference turns selected")
    return {"em": round(100 * sum(row["em"] for row in per_turn.values()) / len(per_turn), 1),
            "f1": round(100 * sum(row["f1"] for row in per_turn.values()) / len(per_turn), 1),
            "turns": len(per_turn), "missing_predictions": missing, "per_turn": per_turn,
            "evaluator_commit": EVALUATOR_COMMIT, "evaluator_sha256": EVALUATOR_SHA256,
            "prediction_preprocessing": "none; citations and other output text remain in the score"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--questions", required=True)
    create.add_argument("--retrieval-labels", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--dev-conversations", type=int, default=41)
    corpus = commands.add_parser("convert-wiki")
    corpus.add_argument("--tsv", required=True)
    corpus.add_argument("--output", required=True)
    corpus.add_argument("--oracle-tasks", help="verify every author gold ID against its exact title and text")
    score = commands.add_parser("score")
    score.add_argument("--references", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--evaluator", required=True, help="pinned author's evaluate_reader.py")
    score.add_argument("--split-manifest", help="optional conversation split JSON")
    score.add_argument("--split", choices=("dev", "test"))
    score.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(materialize(args.questions, args.retrieval_labels, args.output,
                                    seed=args.seed, dev_conversations=args.dev_conversations)))
    elif args.command == "convert-wiki":
        print(json.dumps({"passages": convert_wiki(args.tsv, args.output, oracle_tasks=args.oracle_tasks),
                          "author_gold_verified": args.oracle_tasks is not None}))
    else:
        if bool(args.split_manifest) != bool(args.split):
            parser.error("--split-manifest and --split must be supplied together")
        def read_jsonl(path):
            return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
        conversations = (set(json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))[args.split])
                         if args.split else None)
        result = score_official(read_jsonl(args.references), read_jsonl(args.predictions), args.evaluator,
                                conversations=conversations)
        with Path(args.output).open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps({key: result[key] for key in ("em", "f1", "turns", "missing_predictions")}))


if __name__ == "__main__":
    main()
