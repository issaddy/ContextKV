import copy
import json

import pytest

from training.train_evidence_policy import (build_reader_bound_pairs, qualify_collection_rows,
    reader_binding, require_policy_reader_binding)
from training.reader_data import file_digest


def binding():
    return {"reader_checkpoint": "contextkv-reader-r1:fixture", "reader_profile_id": "source-first-fixture",
        "scorer_identity": {"id": "fixed-rubric-fixture"}}


def reward(category, quality, *, checkpoint=None, body=4096, output=1024, input_hash=None):
    history = [{"turn_id": 1, "user_text": "Windows", "assistant_text": "Actual old answer"}]
    return {"task_id": "fixture-task", "split": "train", "category": category,
        "fixture": "implementation only; no hardware experiment", "measurement_kind": "hardware",
        "history_origin": "frozen_actual_rollout", "actual_history": history, "history_id": json.dumps(history),
        "body_budget": body, "output_budget": output, "input_hash": input_hash or category,
        "reader_checkpoint": checkpoint or binding()["reader_checkpoint"],
        "reader_profile_id": binding()["reader_profile_id"], "scorer_identity": binding()["scorer_identity"],
        "task_state": {"task_id": "fixture-task", "raw_question": "Switch to Linux"},
        "response": {"receipt": {"generation_complete": True, "input_hash": input_hash or category}},
        "answer": f"Actual branch {category}", "reader_model": "qwen3_32b_nonthinking",
        "quality_components": {"necessary_points": quality, "grounding": quality,
                               "conditions_versions": quality, "task_score": quality},
        "cost": {"complete_ms": 1, "gpu_gib_seconds": 1, "host_gib_seconds": 1,
                 "total_input_tokens": 100, "total_output_tokens": 20}}


WEIGHTS = {"necessary_points": .3, "grounding": .2, "conditions_versions": .2, "task_score": .3}


def test_r1_preferences_use_actual_quality_and_keep_history_budget_scorer_fixed():
    preferred, bad = reward("complementary", .9), reward("deleted_premise", .2)
    pairs = build_reader_bound_pairs([bad, preferred], binding=binding(), weights=WEIGHTS)
    assert len(pairs) == 1
    assert pairs[0]["preferred"]["category"] == "complementary"
    assert pairs[0]["reward_difference"] == pytest.approx(.7)
    assert pairs[0]["reader_checkpoint"] == binding()["reader_checkpoint"]
    assert pairs[0]["dispreferred"]["cost"] == bad["cost"]
    assert not build_reader_bound_pairs([preferred, reward("other_budget", .1, body=2048)],
                                        binding=binding(), weights=WEIGHTS)
    assert not build_reader_bound_pairs([preferred, reward("same_input", .1, input_hash="complementary")],
                                        binding=binding(), weights=WEIGHTS)


@pytest.mark.parametrize("field,value", [("reader_checkpoint", "R0"), ("reader_profile_id", "old-profile"),
    ("scorer_identity", {"id": "different-rubric"}), ("split", "dev"), ("history_origin", "author_reference")])
def test_r0_or_unmatched_reader_rewards_cannot_be_relabelled_as_current_r1(field, value):
    changed = copy.deepcopy(reward("wrong", .3))
    changed[field] = value
    with pytest.raises(ValueError):
        build_reader_bound_pairs([reward("good", .8), changed], binding=binding(), weights=WEIGHTS)


def test_policy_asset_retains_r1_reader_binding_and_r0_is_not_a_training_manifest(tmp_path):
    path = tmp_path/"reader_manifest.json"
    path.write_text(json.dumps({"status": "trained", "reader_id": "R0"}))
    with pytest.raises(ValueError, match="trained R1"):
        reader_binding({"evidence_policy_training": {"reader_manifest": str(path)}})
    (tmp_path/"heads.pt").write_bytes(b"implementation-test-heads")
    (tmp_path/"support-model.json").write_text(json.dumps({"reader_binding": binding(),
        "policy_checkpoint": "contextkv-evidence-p1:"+file_digest(tmp_path/"heads.pt")}))
    assert require_policy_reader_binding(tmp_path, reader_checkpoint=binding()["reader_checkpoint"],
                                          reader_profile_id=binding()["reader_profile_id"]) == binding()
    with pytest.raises(ValueError, match="checkpoint/profile"):
        require_policy_reader_binding(tmp_path, reader_checkpoint="R0", reader_profile_id=binding()["reader_profile_id"])


def test_collection_audit_excludes_old_dangling_graphs_and_missing_origins_without_repair():
    def example(task, number, predecessors=(), origin=None):
        question = "Linux permit"
        ref = {"turn_id": number if origin is None else origin, "speaker": "user", "start": 0, "end": 5}
        return {"task_id": task, "conversation_id": "conversation", "turn": number, "question": question,
            "split": "train", "actual_history": [{"turn_id": index, "user_text": "Old question", "assistant_text": "Actual old answer"}
                for index in range(1, number)], "script_history_review_status": "reviewed", "script_history_mismatch": False,
            "verified_grounding": [{"passage_id": "source", "start": 0, "end": 5, "quote": "Linux"}],
            "task_state": {"raw_question": question, "current_turn_id": number, "working_history": [],
                "conditions": [], "version_intent": {"time_refs": []}, "needs": [{"need_id": task+":n0",
                    "condition_ids": [], "predecessor_need_ids": list(predecessors), "anchor_spans": [ref], "subject_refs": []}]}}
    examples = [example("one", 1), example("two", 2, ("never-materialized:n3",)), example("three", 3, origin=0)]
    before = copy.deepcopy(examples)
    accepted, excluded = qualify_collection_rows(examples, [{"source_id": "source", "text": "Linux permit"}])
    assert accepted == [examples[0]]
    assert [row["reason"] for row in excluded] == ["dangling_need_dependency", "missing_actual_span_origin"]
    assert examples == before
