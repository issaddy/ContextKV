"""Session-disjoint native reader cohorts for the actual trace/review pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from training.reader_data import digest, file_digest, history_digest, rows, write_rows


def prepare_stage_cohorts(config):
    """Select saved actual states using their declared session partition.

    Candidate construction stays in preference_rollout, where the native source
    directory and support model are already loaded. No reward is needed here.
    """
    settings = config["reader_stage_cohort"]
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    states, sources, sessions = {}, [], {"train": set(), "dev": set()}
    for spec in settings["sources"]:
        split = spec["split"]
        if split not in sessions:
            raise ValueError("reader cohorts require declared train/dev source partitions")
        path = Path(spec["turns_path"])
        source_rows = rows(path)
        sources.append({**spec, "sha256": file_digest(path), "states": len(source_rows)})
        for state in source_rows:
            if "actual_history" not in state:
                raise ValueError(f"reader cohort needs actual runtime history: {state['task_id']}")
            if state["split"] not in {split, "train"}:
                raise ValueError(f"source partition conflicts with declared cohort: {state['task_id']}")
            key = (split, state["task_id"])
            if key in states and states[key] != state:
                raise ValueError(f"conflicting actual state: {state['task_id']}")
            states[key] = state
            sessions[split].add(state["conversation_id"])
    declared_dev = set(sessions["dev"])
    exposure = []
    exposed_sessions = set()
    for path in settings.get("reader_train_paths", ()):
        values = rows(path)
        seen = {value["conversation_id"] for value in values}
        exposed_sessions.update(seen)
        exposure.append({"path": str(Path(path).resolve()), "sha256": file_digest(path),
                         "training_sessions": len(seen)})
    for path in settings.get("dev_session_paths", ()):
        values = rows(path)
        declared_dev.update(value["conversation_id"] for value in values)
        sources.append({"dev_session_path": str(Path(path).resolve()), "sha256": file_digest(path)})
    fraction = settings.get("validation_fraction", .03)
    split_seed = settings.get("split_seed", 42)
    deterministic_dev = {session for session in sessions["train"]
        if int(digest([split_seed, session])[:8], 16)/16**8 < fraction}
    dev_sessions = declared_dev | deterministic_dev
    excluded = []
    eligible = {"train": [], "dev": []}
    for (split, _), state in states.items():
        session = state["conversation_id"]
        target = "dev" if split == "dev" or session in dev_sessions else "train"
        if target == "dev" and session in exposed_sessions:
            excluded.append({"task_id": state["task_id"], "conversation_id": session,
                             "reason": "heldout_session_seen_by_existing_reader_training"})
            continue
        eligible[target].append(state)
    for split in eligible:
        eligible[split].sort(key=lambda state: (
            digest([settings.get("seed", 42), state["conversation_id"]]), state["turn"], state["task_id"]))
    write_rows(output/"cohort_exclusions.jsonl", excluded)
    full_counts = settings.get("full", {"train": 200, "dev": 40})
    pilot_counts = settings.get("pilot", {"train": 12, "dev": 4})
    manifests = {}
    for name, counts in (("full", full_counts), ("pilot", pilot_counts)):
        directory = output/name
        directory.mkdir(exist_ok=True)
        members, splits = [], {}
        for split in sessions:
            selected = eligible[split][:counts[split]]
            path = directory/f"{split}_turns.jsonl"
            write_rows(path, selected)
            members.extend({"task_id": state["task_id"], "conversation_id": state["conversation_id"],
                "turn": state["turn"], "split": split,
                "actual_history_sha256": history_digest(state["actual_history"])} for state in selected)
            splits[split] = {"requested_states": counts[split], "states": len(selected),
                "shortfall": max(0, counts[split]-len(selected)),
                "sessions": len({state["conversation_id"] for state in selected}),
                "turns_path": str(path.resolve()), "turns_sha256": file_digest(path)}
        manifest = {"schema_version": 1, "cohort": name, "sources": sources, "splits": splits,
            "members": members, "minimum_candidates_per_state": 4, "maximum_candidates_per_state": 6,
            "candidate_status": "pending_real_source_pool_construction_and_reader_rollout",
            "reader_training_precedes_policy_reward_recollection": True,
            "excluded_train_states_in_dev_sessions": sum(
                state["conversation_id"] in dev_sessions for (split, _), state in states.items() if split == "train"),
            "reader_training_exposure": exposure,
            "heldout_training_exposure_exclusions": len(excluded),
            "session_split": {"split_seed": split_seed, "validation_fraction": fraction,
                "declared_dev_sessions": len(declared_dev), "deterministic_dev_sessions": len(deterministic_dev)}}
        path = directory/"cohort_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        manifests[name] = str(path.resolve())
    result = {"cohorts": manifests, "sources": sources,
        "heldout_training_exposure_exclusions": len(excluded), "model_requests": 0}
    (output/"stage_cohorts_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return result
