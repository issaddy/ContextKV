"""Causal projection contracts with synthetic source labels and distinct text."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from adapters.dialogue_examples import main, project_conversation
from adapters.doc2dial import convert


def exported(source_format="v1"):
    text = "Original source 🙂."
    document = {"domain": "manual", "doc_id": "doc", "title": "Title", "doc_text": text,
                "spans": {"s": {"id_sp": "s", "start_sp": 0, "end_sp": len(text), "text_sp": text}}}
    turns = []
    for turn_id, role in ((5, "agent"), (10, "user"), (11, "user"),
                          (20, "agent"), (22, "agent"), (31, "user")):
        turn = {"turn_id": turn_id, "role": role, "utterance": f"TEXT_{turn_id}_ONLY",
                "da": f"GOLD_ACT_{turn_id}_ONLY"}
        if source_format == "v1":
            turn["references"] = [{"sp_id": "s", "label": f"GOLD_REF_{turn_id}_ONLY"}]
        else:
            turn["reference"] = {"s": [f"GOLD_REF_{turn_id}_ONLY", "precondition_fuzzy"]}
        turns.append(turn)
    # An empty reference is preserved; it does not become a reject/clarify label.
    turns[-2]["references" if source_format == "v1" else "reference"] = [] if source_format == "v1" else {}
    documents = {"doc_data": {"manual": {"doc": document}}}
    dialogues = {"dial_data": {"manual": {"doc": [
        {"domain": "manual", "doc_id": "doc", "dial_id": "dialogue", "turns": turns}]}}}
    _, rows, _ = convert(documents, dialogues, dataset_version="original-release",
                         source_uri="dataset://author", split="train", source_format=source_format)
    return rows[0]


@pytest.mark.parametrize("source_format", ["v1", "v0.9"])
def test_strict_prefix_keeps_roles_and_ids_but_no_target_future_or_any_gold(source_format):
    conversation = exported(source_format)
    before = deepcopy(conversation)
    examples = list(project_conversation(conversation))
    assert [e["address"]["target_turn_id"] for e in examples] == [5, 20, 22]
    assert [len(e["input"]["turns"]) for e in examples] == [0, 3, 4]
    for example, position in zip(examples, (0, 3, 4)):
        expected = [{key: turn[key] for key in ("turn_id", "role", "utterance")}
                    for turn in conversation["turns"][:position]]
        assert example["input"] == {"turns": expected}
        serialized = json.dumps(example["input"])
        for turn in conversation["turns"][position:]:
            assert turn["utterance"] not in serialized
        assert "GOLD_" not in serialized and "Original source" not in serialized
        assert "grounding" not in serialized and "passage_id" not in serialized
        assert example["supervision"] == conversation["turns"][position]
        assert example["address"]["conversation_id"] == conversation["conversation_id"]
        assert example["address"]["source_dialogue_id"] == "dialogue"
        assert example["task_scope"] == {"kind": "offline_given_document",
            "passage_id": conversation["passage_id"], "domain": "manual", "source_document_id": "doc"}
        for key in ("dataset", "dataset_version", "source_format", "split"):
            assert example[key] == conversation[key]
        assert not {"action", "clarify", "quality", "answerable"}.intersection(example["supervision"])
    assert [t["role"] for t in examples[-1]["input"]["turns"]] == ["agent", "user", "user", "agent"]
    assert conversation == before


def test_yielded_records_do_not_mutate_source_or_later_prefixes():
    conversation = exported()
    before = deepcopy(conversation)
    iterator = project_conversation(conversation)
    next(iterator)["supervision"]["grounding"][0]["labels"].append("not-author-label")
    second = next(iterator)
    second["input"]["turns"][0]["utterance"] = "corrupted"
    last = next(iterator)
    assert last["input"]["turns"][0]["utterance"] == "TEXT_5_ONLY"
    assert conversation == before


def test_unsupported_original_role_fails_before_yielding_partial_conversation():
    conversation = exported()
    conversation["turns"][-1]["role"] = "system"
    with pytest.raises(ValueError, match="unsupported original role"):
        next(project_conversation(conversation))


def test_empty_and_user_only_dialogues_create_no_synthetic_answer():
    conversation = exported()
    conversation["turns"] = [turn for turn in conversation["turns"] if turn["role"] == "user"]
    assert list(project_conversation(conversation)) == []
    conversation["turns"] = []
    assert list(project_conversation(conversation)) == []


def test_unknown_source_format_is_not_guessed():
    conversation = exported()
    conversation["source_format"] = "auto"
    with pytest.raises(ValueError, match="source_format"):
        list(project_conversation(conversation))


def test_cli_streams_mixed_release_provenance_counts_and_refuses_overwrite(tmp_path):
    rows = [exported("v1"), exported("v0.9")]
    rows[1]["dataset_version"] = "older-release"
    rows[1]["split"] = "development"
    source = tmp_path / "supervision.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "examples"
    argv = ["--supervision", str(source), "--output", str(output)]
    manifest = main(argv)
    assert manifest["source"]["sha256"] == sha256(source.read_bytes()).hexdigest()
    assert manifest["status"] == "completed"
    assert manifest["conversations"] == 2 and manifest["turns"] == 12 and manifest["examples"] == 6
    assert manifest["empty_prefix_examples"] == manifest["consecutive_agent_targets"] == 2
    assert manifest["terminal_user_conversations"] == 2
    assert manifest["partitions"] == [
        {"dataset": "doc2dial", "dataset_version": "original-release", "source_format": "v1", "split": "train",
         "conversations": 1, "examples": 3},
        {"dataset": "doc2dial", "dataset_version": "older-release", "source_format": "v0.9", "split": "development",
         "conversations": 1, "examples": 3}]
    actual = [json.loads(line) for line in (output / "examples.jsonl").read_text().splitlines()]
    assert actual == [example for row in rows for example in project_conversation(row)]
    saved = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == saved
