"""Cohort-bound orchestration of the existing reader trace/review/SFT pipeline."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

from training.reader_data import file_digest, prepare_reader_data, rows, write_rows


def prepare_reader_stage(config):
    """Write executable collection and reader configs after task labels complete."""
    settings = config["reader_stage"]
    output = Path(settings["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort_path = Path(settings["cohort_manifest"]).resolve()
    cohort = json.loads(cohort_path.read_text())
    labels_dir = Path(settings["labels_dir"]).resolve()
    label_manifest = json.loads((labels_dir/"task_labels_manifest.json").read_text())
    if label_manifest["status"] not in {"completed", "incomplete"} or label_manifest["cohort_manifest_sha256"] != file_digest(cohort_path):
        raise ValueError("reader collection requires reviewed task labels bound to this registered cohort")
    reader_config = yaml.safe_load(Path(settings["reader_training_config"]).read_text())
    reader = reader_config["reader_training"]
    reader.update(output_dir=settings["reader_output_dir"], merged_model_dir=settings["reader_merged_dir"],
                  max_length=settings.get("max_length", 32768))
    data = reader["data"]
    data_root = output/"reader-data"
    data.update(output_dir=str(data_root), reviews_path=str(data_root/"reviews.jsonl"),
        export_turns=None, rollouts=[], retrieved_read_turns=0, cohort_manifest=str(cohort_path),
        require_independent_review=True, read_protocol_versions=settings.get("read_protocol_versions", [2, 3]),
        minimum_train_turns=sum(member["split"] == "train" for member in cohort["members"]))
    if settings.get("review"):
        reader["review"].update(settings["review"])
    collection_configs, commands, exclusions, available = {}, {}, [], {}
    for split in ("train", "dev"):
        labels_path = labels_dir/f"{split}_task_labels.jsonl"
        labels = rows(labels_path)
        expected = {member["task_id"] for member in cohort["members"] if member["split"] == split}
        actual = {row["task_id"] for row in labels}
        if not actual <= expected:
            raise ValueError(f"reader collection has {split} labels outside the registered cohort")
        if any(row["correct_task_binding"] is not True or row["source_support_verified"] is not True for row in labels):
            raise ValueError("reader collection needs independently accepted TaskState labels")
        available[split] = len(actual)
        exclusions.extend({"task_id": identity, "cohort_split": split, "reason": "no_accepted_task_state_label"}
                          for identity in sorted(expected-actual))
        collection = deepcopy(config)
        collection.pop("reader_stage")
        collection.setdefault("training", {})["data_dir"] = settings["source_data_dir"]
        preferences = collection.setdefault("preferences", {})
        preferences.update(settings["collection"])
        for key in ("run_config", "reader_checkpoint", "reader_profile_id"):
            if not preferences.get(key):
                raise ValueError(f"reader collection requires preferences.{key}")
        support_key = "support_teacher_url" if preferences.get("support_mode", "trained") == "teacher" else "support_asset"
        if not preferences.get(support_key):
            raise ValueError(f"reader collection requires preferences.{support_key}")
        destination = output/"collection"/split
        preferences.update(turns_path=str(labels_path), state_db=settings["source_state_db"],
            output_dir=str(destination), cohort_manifest=str(cohort_path), cohort_split=split,
            reader_data_only=True, candidate_preparation_only=False, reading=True,
            eligible_task_ids=sorted(actual))
        preferences.setdefault("reading_mode", "typed_reduce")
        preferences.setdefault("read_layout", "source_first")
        preferences.setdefault("read_packing", "source_windows")
        path = output/f"{split}_collection.yaml"
        path.write_text(yaml.safe_dump(collection, sort_keys=False))
        collection_configs[split] = str(path)
        commands[f"collect_{split}"] = [sys.executable, "-m", "training.preference_rollout", "--config", str(path)]
        data["rollouts"].append({"turns_path": str(destination/"turns.jsonl"),
            "failed_turns_path": str(destination/"failed_cases.jsonl"),
            "state_db": str(destination/"runtime"/"sessions.sqlite3"), "split": "train",
            "source_partition": "native_multidoc2dial_train", "reader_identity": preferences["reader_checkpoint"]})
    data["minimum_train_turns"] = available["train"]
    data["task_label_coverage"] = {"declared_states": len(cohort["members"]),
        "available_states": sum(available.values()), "available_splits": available, "excluded_states": exclusions}
    write_rows(output/"task_label_exclusions.jsonl", exclusions)
    training_path = output/"reader_training.yaml"
    training_path.write_text(yaml.safe_dump(reader_config, sort_keys=False))
    for phase in ("export", "review"):
        commands[phase] = [sys.executable, "-m", "training.train_reader", phase, "--config", str(training_path)]
    stage_config = output/"reader_stage.yaml"
    effective = deepcopy(config)
    effective["reader_stage"].update(output_dir=str(output), prepared_reader_config=str(training_path))
    stage_config.write_text(yaml.safe_dump(effective, sort_keys=False))
    commands["prepare_phases"] = [sys.executable, "-m", "training.a_workflow", "reader-prepare",
        "--config", str(stage_config), "--output", str(output)]
    result = {"schema_version": 1, "cohort_manifest": str(cohort_path),
        "cohort_manifest_sha256": file_digest(cohort_path), "collection_configs": collection_configs,
        "declared_states": len(cohort["members"]), "available_states": sum(available.values()),
        "available_splits": available, "excluded_states": exclusions,
        "reader_training_config": str(training_path), "reader_stage_config": str(stage_config),
        "commands": commands, "model_requests": 0,
        "recovery_entry": {split: [sys.executable, "-m", "training.preference_rollout", "--config", path,
            "--prepare-resume", "--pause-receipt", "<saved_gpu_pause_receipt.json>", "--confirmed-stopped"]
            for split, path in collection_configs.items()},
        "policy_rewards": "recollect with the trained reader after these reader stages complete"}
    (output/"reader_stage_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


def prepare_reader_phases(config):
    """Prepare phase datasets using accepted exact-prompt reviews and native sessions."""
    settings = config["reader_stage"]
    source_config = yaml.safe_load(Path(settings["prepared_reader_config"]).read_text())
    combined = prepare_reader_data(source_config)
    source = Path(source_config["reader_training"]["data"]["output_dir"])
    output = Path(settings["output_dir"])
    last_adapter = source_config["reader_training"].get("initial_adapter")
    stages = []
    for phase in ("READ", "REDUCE", "ANSWER"):
        destination = source/phase.lower()
        selected = {split: [row for row in rows(source/f"{split}.jsonl") if row["phase"] == phase]
                    for split in ("train", "validation")}
        for split, values in selected.items():
            write_rows(destination/f"{split}.jsonl", values)
        phase_manifest = {**combined, "phase": phase, "phases": {phase: sum(map(len, selected.values()))},
            "examples": sum(map(len, selected.values())),
            "training_turns": len({row["task_id"] for row in selected["train"]}),
            "data_splits": {split: len(values) for split, values in selected.items()},
            "source_data_manifest_sha256": file_digest(source/"data_manifest.json")}
        (destination/"data_manifest.json").write_text(json.dumps(phase_manifest, ensure_ascii=False, indent=2)+"\n")
        stage = {"phase": phase, "examples": phase_manifest["data_splits"], "train_command": None,
            "status": "no_observed_training_examples" if not selected["train"] else "prepared"}
        if selected["train"]:
            actual = deepcopy(source_config)
            reader = actual["reader_training"]
            reader["data"].update(output_dir=str(destination), minimum_train_turns=phase_manifest["training_turns"])
            reader["output_dir"] = str(Path(settings["reader_output_dir"])/phase.lower())
            reader["merged_model_dir"] = str(Path(settings["reader_merged_dir"])/phase.lower())
            if last_adapter:
                reader["initial_adapter"] = str(last_adapter)
            path = output/f"reader_{phase.lower()}_training.yaml"
            path.write_text(yaml.safe_dump(actual, sort_keys=False))
            stage.update(training_config=str(path.resolve()),
                train_command=[sys.executable, "-m", "training.train_reader", "train", "--config", str(path.resolve())])
            last_adapter = Path(reader["output_dir"])/"adapter"
        stages.append(stage)
    result = {"schema_version": 1, "data": combined, "stages": stages,
        "final_adapter": str(last_adapter) if last_adapter else None, "model_requests": 0,
        "cpu_reductions": "remain observed upstream operations; no model prompt or SFT target is invented"}
    (output/"reader_phases_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result
