import json

import yaml

from training.reader_data import file_digest, history_digest, rows, write_rows
from training.reader_mix import mix_prepared, stage_configs
from training.train_reader import _asset_digest


def test_reader_mix_keeps_frozen_parent_and_holdout_and_prioritizes_reviewed_actual_p1(tmp_path):
    parent, curriculum, fresh, output = (tmp_path/name for name in ("parent", "curriculum", "new", "mixed"))
    adapter = tmp_path/"r1"/"adapter"
    adapter.mkdir(parents=True)
    (adapter/"adapter_config.json").write_text('{"fixture":"implementation only"}')
    reader_id = "contextkv-reader-r1:"+_asset_digest(adapter)[0]
    manifest = adapter.parent/"reader_manifest.json"
    manifest.write_text(json.dumps({"status": "trained", "reader_id": reader_id}))
    def example(identity, *, task=None, conversation="train", partition="native_multidoc2dial_train", phase="ANSWER"):
        return {"example_id": identity, "task_id": task or identity, "conversation_id": conversation,
            "split": "train", "data_split": "train", "source_partition": partition, "phase": phase,
            "supervision": {"status": "accepted"}, "target_text": "Reviewed fixture target",
            "actual_history": [], "actual_history_sha256": history_digest([]), "prompt_token_hash": identity,
            "reader_identity": reader_id, "read_protocol_version": 2 if phase == "READ" else None}
    frozen = [example(f"parent-{index}") for index in range(1000)]
    heldout = {**example("validation", conversation="heldout"), "data_split": "validation"}
    write_rows(parent/"train.jsonl", frozen)
    write_rows(parent/"validation.jsonl", [heldout])
    write_rows(curriculum/"train.jsonl", [*frozen, example("constructed", task="parent-0", partition="constructed_train")])
    new = [example("p1-read", task="new-task", phase="READ"), example("p1-answer", task="new-task"),
           example("p1-heldout", task="heldout-task", conversation="heldout")]
    write_rows(fresh/"traces.jsonl", [{key: value for key, value in row.items() if key not in {"supervision", "target_text", "data_split"}}
                                     for row in new])
    write_rows(fresh/"reviews.jsonl", [{"example_id": row["example_id"], "status": "accepted", "quality": .95,
        "source_support_verified": True, "correct_task_binding": True, "target_text": "Fresh teacher fixture",
        "target_origin": "implementation_test_fresh_actual_prompt", "actual_history_sha256": row["actual_history_sha256"],
        "prompt_token_hash": row["prompt_token_hash"], "reader_identity": reader_id} for row in new])
    config = {"reader_mix": {"reader_manifest": str(manifest), "parent_data_dir": str(parent),
        "curriculum_data_dir": str(curriculum), "new_p1_data_dir": str(fresh), "output_dir": str(output),
        "reader_output_dir": str(tmp_path/"r2"), "reader_merged_dir": str(tmp_path/"r2-merged"),
        "reader_base_url": "http://fixture-r2", "new_p1_repetitions": 4, "curriculum_repetitions": 3}}
    p1 = {"reader_training": {"data": {"export_turns": 2000, "read_protocol_versions": [2], "minimum_quality": .85,
        "validation_fraction": .03, "split_seed": 42, "minimum_train_turns": 1000,
        "rollouts": [{"reader_identity": reader_id, "split": "train"}]}}}
    before = {path: file_digest(path) for path in (parent/"train.jsonl", parent/"validation.jsonl", curriculum/"train.jsonl")}
    scoped, _ = stage_configs(config, p1)
    assert scoped["reader_training"]["data"]["export_turns"] is None
    assert scoped["reader_training"]["data"]["retrieved_read_turns"] == 0
    result = mix_prepared(config, p1)
    train = rows(output/"train.jsonl")
    assert result["data"]["actual_native_training_turns"] == 1001
    assert result["data"]["components_unique"] == {"new_P1_actual": 2, "reviewed_curriculum": 1, "frozen_parent": 1000}
    assert len(train) == 1011 and all(row["mix_component"] == "new_P1_actual" for row in train[:8])
    assert not any(row["conversation_id"] == "heldout" for row in train)
    assert (parent/"validation.jsonl").read_bytes() == (output/"validation.jsonl").read_bytes()
    final = yaml.safe_load((output/"reader_r2_training.yaml").read_text())["reader_training"]
    assert final["initial_adapter"] == str(adapter) and final["data"]["minimum_train_turns"] == 1000
    assert {path: file_digest(path) for path in before} == before
