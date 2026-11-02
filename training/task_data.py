"""Native training partition exports and teacher labels over actual rollouts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from adapters.doc2dial import benchmark_rows, convert
from controller.context_schema import RetrievalState
from controller.schema import OnlineMessage, OnlineTurn, TurnRecord
from controller.serialization import decode, to_jsonable


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(to_jsonable(row), ensure_ascii=False)+"\n")
            count += 1
    return count


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        yield from (json.loads(line) for line in source if line.strip())


def build_task_data(config):
    settings = config.get("training", {})
    root = Path(settings.get("data_dir", "runs/study/data"))
    source = Path(settings.get("multidoc_source", "data/multidoc2dial/source"))
    corpus, supervision, manifest = convert(
        json.loads((source/"multidoc2dial_doc.json").read_text()),
        json.loads((source/"multidoc2dial_dial_train.json").read_text()),
        dataset_version="archive-f0c034c24966", source_uri="https://doc2dial.github.io/multidoc2dial/",
        split="train", source_format="multidoc2dial")
    sources = [{"source_id": row["passage_id"], "document_family": row["document_id"],
        "version": row["version"], "valid_from": None, "valid_to": None,
        "applicability": {}, "text": row["text"], "title": row["title"],
        "offsets": [0, len(row["text"])], "split": "train",
        "source_uri": row["source_uri"]} for row in corpus]
    sources_count = write_jsonl(root/"sources.jsonl", sources)
    rows = []
    for row in benchmark_rows(supervision):
        rows.append({"task_id": row["task_id"], "conversation_id": row["conversation_id"],
            "split": "train", "turn": row["turn"], "question": row["input"][-1]["text"],
            "reference_history": row["input"][:-1], "reference_answer": row["targets"][0]["text"],
            "snapshot_id": "multidoc2dial-train-archive-f0c034c24966", "grounding": row["grounding"],
            "history_origin": "author_reference", "rollout_status": "pending_actual_generation"})
    reference_count = write_jsonl(root/"original_train_turns.jsonl", rows)
    write_jsonl(root/"supervision.jsonl", supervision)
    rollout_paths = [Path(path) for path in settings.get("rollout_paths", ())]
    turns_path = root/"turns.jsonl"
    if rollout_paths:
        original = {row["task_id"]: row for row in rows}
        actual_rows = []
        for path in rollout_paths:
            for actual in read_jsonl(path):
                author = original[actual["task_id"]]
                actual_rows.append({**actual, "retrieved_grounding": actual.get("grounding", []),
                    "grounding": author["grounding"], "author_grounding": author["grounding"],
                    "reference_history": author["reference_history"], "reference_answer": author["reference_answer"],
                    "grounding_origin": "author_native_train_pending_actual_history_review"})
        actual_count = write_jsonl(turns_path, actual_rows)
    elif turns_path.exists():
        actual_rows = list(read_jsonl(turns_path))
        actual_count = len(actual_rows)
    else:
        actual_rows = []
        actual_count = write_jsonl(turns_path, actual_rows)
    conversations = {row["conversation_id"] for row in actual_rows}
    summary = {"schema_version": 1, "native_partition": "train", "reference_turns": reference_count,
        "reference_conversations": len(supervision), "sources": sources_count,
        "actual_rollout_turns": actual_count, "actual_rollout_conversations": len(conversations),
        "rollout_paths": [str(path) for path in rollout_paths],
        "paths": {name: str((root/name).resolve()) for name in (
            "sources.jsonl", "original_train_turns.jsonl", "supervision.jsonl", "turns.jsonl")},
        "stage_one_target": {"turns": 10000, "actual_conversations": 1000},
        "source_manifest": manifest}
    root.mkdir(parents=True, exist_ok=True)
    (root/"task_data_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary


def label_actual_rollouts(turns_path, output_dir, interpreter):
    output = Path(output_dir)
    states, labels, removals = {}, [], []
    for row in read_jsonl(turns_path):
        history = tuple(decode(TurnRecord, item) for item in row["actual_history"])
        messages = tuple(message for item in history for message in (
            OnlineMessage("user", item.user_text), OnlineMessage("assistant", item.assistant_text)))
        turn = OnlineTurn(row["conversation_id"], row["task_id"], row["turn"], row["snapshot_id"],
                          row["question"], messages, (), "P3")
        prior = tuple(states.get(row["conversation_id"], ()))
        memory = RetrievalState(len(history), history, prior)
        state = interpreter.interpret(turn, memory)
        states.setdefault(row["conversation_id"], []).append(state)
        labels.append({**row, "prediction": to_jsonable(interpreter.last_prediction),
            "task_state": to_jsonable(state), "label_origin": "frozen_qwen3_32b_teacher",
            "teacher_metrics": interpreter.last_metrics})
        for ref in state.working_history:
            removals.append({"task_id": row["task_id"], "split": row["split"],
                "removed_span": to_jsonable(ref), "actual_history": row["actual_history"],
                "question": row["question"], "status": "pending_paired_retrieval_answer_measurement"})
    count = write_jsonl(output/"task_labels.jsonl", labels)
    removal_count = write_jsonl(output/"history_removal_requests.jsonl", removals)
    summary = {"labels": count, "removal_requests": removal_count,
        "relation_counts": dict(Counter(row["prediction"]["relation"] for row in labels)),
        "paths": {name: str((output/name).resolve()) for name in ("task_labels.jsonl", "history_removal_requests.jsonl")}}
    (output/"task_label_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary
