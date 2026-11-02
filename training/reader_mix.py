"""Continue R1 from reviewed P1 runtime data plus frozen parent/curriculum data."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import yaml

from training.reader_data import (digest, export_reader_traces, file_digest, prepare_reader_data,
                                 review_reader_traces, rows, write_rows)


def parent_reader(config):
    settings = config["reader_mix"]
    manifest_path = Path(settings["reader_manifest"])
    manifest = json.loads(manifest_path.read_text())
    from training.train_reader import _asset_digest
    adapter = manifest_path.parent/"adapter"
    identity, _ = _asset_digest(adapter)
    if manifest["status"] != "trained" or manifest["reader_id"] != "contextkv-reader-r1:"+identity:
        raise ValueError("reader mix must initialize the actual trained R1 adapter")
    return manifest, adapter


def stage_configs(config, p1_config):
    settings = config["reader_mix"]
    manifest, adapter = parent_reader(config)
    new = deepcopy(p1_config)
    reader = new["reader_training"]
    if any(spec["reader_identity"] != manifest["reader_id"] or spec["split"] != "train"
           for spec in reader["data"]["rollouts"]):
        raise ValueError("new P1 traces must be actual training rollouts from the parent R1")
    reader["data"].update(output_dir=settings["new_p1_data_dir"], export_turns=None,
        retrieved_read_turns=0, validation_fraction=0,
        reviews_path=str(Path(settings["new_p1_data_dir"])/"reviews.jsonl"))
    reader["initial_adapter"] = str(adapter)
    mixed = deepcopy(new)
    final = mixed["reader_training"]
    final.update(initial_adapter=str(adapter), output_dir=settings["reader_output_dir"],
        merged_model_dir=settings["reader_merged_dir"], r1_base_url=settings["reader_base_url"])
    final["data"].update(output_dir=settings["output_dir"], minimum_train_turns=1000, export_turns=None, rollouts=[])
    return new, mixed


def mix_prepared(config, p1_config):
    settings = config["reader_mix"]
    new_config, final_config = stage_configs(config, p1_config)
    prepare_reader_data(new_config)
    parent = Path(settings["parent_data_dir"])
    curriculum = Path(settings["curriculum_data_dir"])
    new_root = Path(settings["new_p1_data_dir"])
    output = Path(settings["output_dir"])
    if output.resolve() in {parent.resolve(), curriculum.resolve(), new_root.resolve()}:
        raise ValueError("mixed data must have its own output directory")
    parent_train, validation = rows(parent/"train.jsonl"), rows(parent/"validation.jsonl")
    curriculum_train, new_train = rows(curriculum/"train.jsonl"), rows(new_root/"train.jsonl")
    heldout = {row["conversation_id"] for row in validation}
    selected, seen, excluded = {"new_P1_actual": [], "reviewed_curriculum": [], "frozen_parent": []}, {}, []
    component_files = {"frozen_parent": parent/"train.jsonl", "reviewed_curriculum": curriculum/"train.jsonl",
                       "new_P1_actual": new_root/"train.jsonl"}
    for component, values in (("new_P1_actual", new_train), ("frozen_parent", parent_train),
                              ("reviewed_curriculum", curriculum_train)):
        for row in values:
            if row["split"] != "train" or row["data_split"] != "train" or row["supervision"]["status"] != "accepted":
                raise ValueError("reader mix inputs must be prepared, accepted training examples")
            if row["conversation_id"] in heldout:
                excluded.append({"example_id": row["example_id"], "component": component,
                                 "reason": "parent_validation_conversation"})
                continue
            identity = digest({"messages": row["messages"], "target_text": row["target_text"]})
            origin = {"component": component, "original_example_id": row["example_id"],
                "source_file": str(component_files[component]), "source_row_sha256": digest(row),
                "task_id": row["task_id"], "actual_history_sha256": row["actual_history_sha256"],
                "reader_identity": row["reader_identity"], "target_origin": row.get("target_origin")}
            if identity in seen:
                seen[identity]["mix_origins"].append(origin)
                continue
            mixed_row = {**row, "example_id": f"{component}:{row['example_id']}:{identity[:24]}",
                "original_example_id": row["example_id"], "input_target_sha256": identity, "mix_origins": [origin]}
            seen[identity] = mixed_row
            selected[component].append(mixed_row)
    if not selected["new_P1_actual"]:
        raise ValueError("the next reader needs fresh accepted P1 runtime examples")
    actual_turns = {row["task_id"] for row in seen.values() if row["source_partition"] == "native_multidoc2dial_train"}
    if len(actual_turns) < 1000:
        raise ValueError(f"reader mix requires 1000 actual native train turns; selected {len(actual_turns)}")
    train = [{**row, "mix_component": "new_P1_actual", "sampling_replica": replica}
        for replica in range(settings["new_p1_repetitions"]) for row in selected["new_P1_actual"]]
    train.extend({**row, "mix_component": "reviewed_curriculum", "sampling_replica": replica}
        for replica in range(settings.get("curriculum_repetitions", 3)) for row in selected["reviewed_curriculum"])
    train.extend({**row, "mix_component": "frozen_parent", "sampling_replica": 0} for row in selected["frozen_parent"])
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"train.jsonl", train)
    shutil.copyfile(parent/"validation.jsonl", output/"validation.jsonl")
    write_rows(output/"mix_exclusions.jsonl", excluded)
    manifest, adapter = parent_reader(config)
    files = {name: {"path": str(path.resolve()), "sha256": file_digest(path)} for name, path in (
        ("frozen_parent_train", parent/"train.jsonl"), ("frozen_parent_validation", parent/"validation.jsonl"),
        ("reviewed_curriculum_train", curriculum/"train.jsonl"), ("new_P1_train", new_root/"train.jsonl"),
        ("new_P1_reviews", new_root/"reviews.jsonl"))}
    data_manifest = {"schema_version": 1, "evidence_kind": "frozen_R1_parent_reviewed_curriculum_and_fresh_P1_runtime",
        "parent_reader_id": manifest["reader_id"], "initial_adapter": str(adapter),
        "training_turns": len({row["task_id"] for row in train}), "actual_native_training_turns": len(actual_turns),
        "minimum_train_turns": 1000, "examples": len(train)+len(validation),
        "unique_training_examples": len(seen), "components_unique": {name: len(values) for name, values in selected.items()},
        "deduplication": "exact messages plus target_text content identity; original task/example IDs are provenance, not dedup keys",
        "deduplicated_source_origins": sum(len(row["mix_origins"])-1 for row in seen.values()),
        "components_sampled": dict(Counter(row["mix_component"] for row in train)),
        "new_P1_repetitions": settings["new_p1_repetitions"],
        "curriculum_repetitions": settings.get("curriculum_repetitions", 3),
        "sampling_policy": "new P1 examples first and explicitly oversampled; Trainer samples the expanded data uniformly",
        "export_turns": None, "phases": dict(Counter(row["phase"] for row in (*train, *validation))),
        "data_splits": {"train": len(train), "validation": len(validation)}, "sources": files,
        "validation_origin": "unchanged frozen parent training-partition validation; new P1 heldout conversations excluded",
        "teacher_collection": "only new P1 missing example IDs; frozen parent/curriculum are not relabelled",
        "cost_ledgers": [str(new_root/"review_calls.jsonl"), str(curriculum/"review_calls.jsonl")],
        "quality_evaluation": "separate actual P0/P1 x R0/R1 or R2 results; no gain inferred from data preparation"}
    (output/"data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2)+"\n")
    target = output/"reader_r2_training.yaml"
    target.write_text(yaml.safe_dump(final_config, sort_keys=False))
    return {"data": data_manifest, "training_config": str(target.resolve()),
        "next_command": f"python -m training.train_reader train --config {target}"}


async def run(config, p1_config, command, *, concurrency=8):
    from functools import partial
    from training.train_reader import measured_command
    new, _ = stage_configs(config, p1_config)
    result = None
    if command in {"export", "all"}:
        result = await measured_command(new, "export", partial(export_reader_traces, new))
    if command in {"review", "all"}:
        result = await measured_command(new, "review", partial(review_reader_traces, new,
            concurrency=concurrency), asynchronous=True)
    if command in {"prepare", "all"}:
        result = await measured_command(new, "prepare", partial(mix_prepared, config, p1_config))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "review", "prepare", "all"))
    parser.add_argument("--config", default="configs/reader_mix_r2.yaml")
    parser.add_argument("--p1-config", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    p1 = yaml.safe_load(Path(args.p1_config).read_text())
    print(json.dumps(asyncio.run(run(config, p1, args.command, concurrency=args.concurrency)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
