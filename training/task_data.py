"""Native training partition exports and teacher labels over actual rollouts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from adapters.doc2dial import benchmark_rows, convert
from controller.serialization import to_jsonable


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


def select_cohort_rows(values, manifest_path, split):
    """Select registered states from one session-disjoint cohort partition."""
    from training.reader_data import file_digest
    members = {member["task_id"]: member for member in
        json.loads(Path(manifest_path).read_text())["members"] if member["split"] == split}
    manifest_hash = file_digest(manifest_path)
    return [{**row, "cohort_split": split, "cohort_manifest_sha256": manifest_hash}
            for row in values if row["task_id"] in members
            and row["conversation_id"] == members[row["task_id"]]["conversation_id"]]


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
        "source_manifest": manifest}
    root.mkdir(parents=True, exist_ok=True)
    (root/"task_data_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary


def prepare_cohort_task_labels(config):
    """Bind registered states to actual histories, original spans and saved labels."""
    import sqlite3
    import sys
    import yaml
    from copy import deepcopy
    from adapters.history_state import TASK_NEED_SEMANTICS_VERSION
    from controller.session_store import Attempt, load_record
    from training.reader_data import file_digest, history_digest

    settings = config["task_cohort_labeling"]
    output = Path(settings["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output/"task_labels.jsonl"
    cohort_path = Path(settings["cohort_manifest"])
    cohort = json.loads(cohort_path.read_text())
    members = {member["task_id"]: member for member in cohort["members"]}
    actual = {row["task_id"]: row for row in read_jsonl(settings["actual_turns_path"]) if row["task_id"] in members}
    if set(actual) != set(members):
        raise ValueError("actual rollout does not contain every registered labeling state")
    authored = {row["task_id"]: row for row in read_jsonl(settings["original_turns_path"]) if row["task_id"] in members}
    originals = {row["source_id"]: row for row in read_jsonl(settings["sources_path"])}
    existing_labels = settings["existing_labels_path"]
    prior = {} if existing_labels is None else {row["task_id"]: row for row in read_jsonl(existing_labels)}
    local_labels = list(read_jsonl(labels_path)) if labels_path.exists() else []
    if any(row["task_id"] not in members for row in local_labels):
        raise ValueError("existing output labels belong to a different cohort")
    prior.update((row["task_id"], row) for row in local_labels)
    database = Path(settings["state_db"]).resolve()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    selected, reused, reuse_exclusions = [], [], []
    for identity, member in members.items():
        row = actual[identity]
        if (row["conversation_id"] != member["conversation_id"] or row["turn"] != member["turn"]
                or history_digest(row["actual_history"]) != member["actual_history_sha256"]):
            raise ValueError(f"actual state differs from registered cohort: {identity}")
        if row["split"] != "train":
            raise ValueError("cohort task labels must retain the native training partition")
        author = authored[identity]
        for span in author["grounding"]:
            text = originals[span["passage_id"]]["text"]
            if text[span["start"]:span["end"]] != span["quote"]:
                raise ValueError(f"author grounding differs from original source coordinates: {identity}")
        attempt_id = row["response"]["attempt_id"]
        payload = connection.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
        attempt = load_record(connection, Attempt, payload)
        if attempt.query != row["question"] or attempt.stage != "COMMITTED":
            raise ValueError(f"source attempt differs from registered completed state: {identity}")
        blocks = {block.key: block for block in attempt.evidence}
        plan = attempt.reading_outcome.plan if attempt.reading_outcome is not None else attempt.context_plan
        if plan is not None:
            blocks.update((block.key, block) for block in plan.selected_set.source_blocks)
        source_rows = [{"source_key": to_jsonable(block.key), "source_uri": block.source_uri,
            "start": block.span[0] if block.span else 0,
            "end": block.span[1] if block.span else len(block.text), "text": block.text}
            for block in blocks.values()]
        verification_sources = [{"source_id": source_id, "text": originals[source_id]["text"],
            "version": originals[source_id]["version"], "source_uri": originals[source_id]["source_uri"]}
            for source_id in dict.fromkeys(span["passage_id"] for span in author["grounding"])]
        prepared = {**row, "retrieved_grounding": row["grounding"],
            "grounding": author["grounding"], "author_grounding": author["grounding"],
            "reference_history": author["reference_history"], "reference_answer": author["reference_answer"],
            "grounding_origin": "author_native_train_pending_actual_history_review",
            "actual_history_sha256": member["actual_history_sha256"], "cohort_split": member["split"],
            "cohort_manifest_sha256": file_digest(cohort_path), "original_sources": source_rows,
            "grounding_original_sources": verification_sources,
            "source_attempt": {"attempt_id": attempt_id, "state_db": str(database)}}
        selected.append(prepared)
        old = prior.get(identity)
        if old is not None:
            same_state = all(old[key] == row[key] for key in ("conversation_id", "turn", "question", "snapshot_id"))
            same_history = history_digest(old["actual_history"]) == member["actual_history_sha256"]
            if (same_state and same_history and old["script_history_review_status"] == "reviewed"
                    and old.get("correct_task_binding") is True and old.get("source_support_verified") is True
                    and old.get("need_semantics_version") == TASK_NEED_SEMANTICS_VERSION
                    and old.get("need_semantics_verified") is True
                    and "task_catalogue" in old):
                reused.append({**old, **prepared, "grounding_origin": old["grounding_origin"],
                               "cohort_label_origin": "reused_exact_actual_state_review"})
            else:
                reuse_exclusions.append({"task_id": identity, "reason":
                    "existing_label_not_bound_to_registered_actual_state" if not same_state or not same_history
                    else "existing_label_missing_independent_source_and_task_verification"})
    connection.close()
    turns_path = output/"turns.jsonl"
    write_jsonl(turns_path, selected)
    write_jsonl(labels_path, reused)
    write_jsonl(output/"label_reuse_exclusions.jsonl", reuse_exclusions)
    effective = deepcopy(config)
    training = effective.setdefault("training", {})
    training.update(data_dir=str(output), task_labels=str(labels_path),
        task_label_turns=str(turns_path), cohort_manifest=str(cohort_path.resolve()))
    config_path = output/"task_labeling.yaml"
    config_path.write_text(yaml.safe_dump(effective, sort_keys=False))
    manifest = {"schema_version": 1, "cohort_manifest": str(cohort_path.resolve()),
        "cohort_manifest_sha256": file_digest(cohort_path), "registered_states": len(members),
        "registered_sessions": len({member["conversation_id"] for member in members.values()}),
        "cohort_splits": dict(Counter(member["split"] for member in members.values())),
        "reused_labels": len(reused), "pending_labels": len(members)-len(reused),
        "source_files": {name: {"path": str(Path(settings[name]).resolve()), "sha256": file_digest(settings[name])}
            for name in ("actual_turns_path", "original_turns_path", "sources_path", "existing_labels_path")
            if settings[name] is not None},
        "state_db": str(database), "turns_path": str(turns_path), "labels_path": str(labels_path),
        "labeling_config": str(config_path),
        "labeling_command": [sys.executable, "-m", "training.task_labeling", "--config", str(config_path)],
        "model_requests": 0, "support_supervision": "support candidates retain cohort_split; fitting uses manifest train states only"}
    (output/"cohort_label_preparation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest
