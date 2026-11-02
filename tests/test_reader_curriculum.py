from copy import deepcopy
from dataclasses import asdict
import json

import yaml

from controller.evidence import make_evidence
from controller.schema import Message
from controller.templates import ByteTemplate
from training.reader_curriculum import (audit_curriculum, build_curriculum, changed_literal,
    numeric_pairs, quantities)
from training.reader_data import file_digest, history_digest, rows, write_rows


class Tokens:
    def encode(self, text, **kwargs):
        return tuple(text.encode())
    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode()), "offset_mapping": [
            (index, index+1) for index, char in enumerate(text) for _ in char.encode()]}


def parent_example(template):
    blocks = (make_evidence("Linux allowance is 12 credits.\nA second eligibility paragraph.", "allowance", corpus_id="native", version="v1"),
              make_evidence("Filing notes contain no additional allowance.\nGeneral procedural notes.", "notes", corpus_id="native", version="v1"))
    messages = (Message("system", "Answer from originals."), Message("user",
        "\n".join(block.text for block in blocks)+"\nWhat is the Linux allowance?"))
    prompt = template.render(messages)
    return {"example_id": "native-attempt:ANSWER", "task_id": "native-task", "conversation_id": "native-conversation",
        "attempt_id": "native-attempt", "turn": 1, "split": "train", "data_split": "train", "phase": "ANSWER",
        "source_partition": "native_multidoc2dial_train", "actual_history": [], "actual_history_sha256": history_digest([]),
        "reader_identity": "fixture-r0", "snapshot_id": "native-train-snapshot", "question": "What is the Linux allowance?",
        "messages": [asdict(message) for message in messages], "prompt_token_hash": prompt.input_hash,
        "prompt_tokens": len(prompt.input_ids), "task_state": None, "context_plan": None,
        "actual_output": "Raw previous output", "actual_output_ids": [], "read_protocol_version": None,
        "trace_origin": {"fixture": "implementation-only native-shaped source"},
        "target_text": "The allowance is 12 credits.", "target_origin": "implementation_test_review",
        "quality": .95, "supervision": {"status": "accepted", "source_support_verified": True, "correct_task_binding": True},
        "original_sources": [{"source_key": asdict(block.key), "start": 80, "end": 80+len(block.text),
            "text": block.text, "content_sha256": block.key.content_sha256, "source_uri": block.source_uri} for block in blocks]}


def fixture_config(tmp_path):
    template = ByteTemplate()
    template.tokenizer = Tokens()
    parent = tmp_path/"parent"
    parent.mkdir()
    write_rows(parent/"train.jsonl", [parent_example(template)])
    write_rows(parent/"validation.jsonl", [])
    training = tmp_path/"reader.yaml"
    training.write_text(yaml.safe_dump({"reader_training": {"data": {"output_dir": str(parent),
        "minimum_quality": .85, "read_protocol_versions": [2], "validation_fraction": 0,
        "split_seed": 42, "minimum_train_turns": 1000}}}))
    config = {"reader_curriculum": {"source_data_dir": str(parent), "output_dir": str(tmp_path/"round2"),
        "parent_training_config": str(training), "reader_output_dir": "models/round2",
        "reader_merged_dir": "models/round2-merged", "initial_adapter": "models/r1/adapter",
        "maximum_numeric_pairs": 8, "numeric_delta": 7, "maximum_constructed_cases": 0}}
    return config, template


def test_numeric_candidates_require_both_reviewed_target_and_unique_original_quantity(tmp_path):
    _, template = fixture_config(tmp_path)
    parent = parent_example(template)
    pairs, excluded = numeric_pairs([parent])
    assert len(pairs) == 1 and not excluded
    pair = pairs[0]
    offset = parent["original_sources"][0]["text"].index("12")
    assert pair["mutation"]["start"] == 80+offset and pair["mutation"]["end"] == 82+offset
    assert pair["mutation"]["replacement"] == "19"
    assert pair["original_source"]["text"] != pair["mutated_source"]["text"]
    assert changed_literal(quantities("$1,234.50")[0], 7) == "1,241.50"
    assert quantities("$12k") == []
    no_target = {**parent, "target_text": "There is an allowance."}
    ambiguous = deepcopy(parent)
    ambiguous["original_sources"].append(deepcopy(parent["original_sources"][0]))
    _, excluded = numeric_pairs([no_target, ambiguous])
    assert [row["reason"] for row in excluded] == ["no_shared_explicit_quantity", "ambiguous_retrieved_quantity"]


def test_curriculum_constructs_separate_fresh_read_pairs_and_position_views_without_touching_parent(tmp_path):
    config, template = fixture_config(tmp_path)
    parent_path = tmp_path/"parent"/"train.jsonl"
    before = file_digest(parent_path)
    manifest = build_curriculum(config, template=template)
    root = tmp_path/"round2"
    assert manifest["numeric_pairs"] == 1 and manifest["position_inputs"] == 2
    assert manifest["public_dev_test_sources_read"] == []
    assert file_digest(parent_path) == before
    traces = rows(root/"traces.jsonl")
    assert all(row["phase"] == "READ" and row["actual_output"] == "" for row in traces)
    assert all("target_text" not in row and "supervision" not in row for row in traces)
    pair = rows(root/"pairs.prepared.jsonl")[0]
    assert pair["control_example_ids"] and pair["changed_example_ids"]
    by_id = {row["example_id"]: row for row in traces}
    assert by_id[pair["control_example_ids"][0]]["prompt_token_hash"] != by_id[pair["changed_example_ids"][0]]["prompt_token_hash"]


def review_rows(root, *, changed_value):
    reviews = []
    for trace in rows(root/"traces.jsonl"):
        value = changed_value if trace["source_mutations"] else "12"
        target = {"claims": [{"need_id": "question", "entity": "Linux", "relation": "allowance",
            "text": f"The Linux allowance is {value} credits.", "value": value, "unit": "credits"}]}
        reviews.append({"example_id": trace["example_id"], "task_id": trace["task_id"], "status": "accepted",
            "target_text": json.dumps(target), "target_origin": "implementation_test_fresh_review", "quality": .95,
            "source_support_verified": True, "correct_task_binding": True,
            "actual_history_sha256": trace["actual_history_sha256"], "prompt_token_hash": trace["prompt_token_hash"],
            "reader_identity": trace["reader_identity"]})
    write_rows(root/"reviews.jsonl", reviews)


def test_audit_requires_new_target_quantity_and_retains_unchanged_failure_counts(tmp_path):
    config, template = fixture_config(tmp_path)
    build_curriculum(config, template=template)
    root = tmp_path/"round2"
    review_rows(root, changed_value="19")
    result = audit_curriculum(config)
    assert result["pair_checks"] == {"source_sensitive_change_verified": 1}
    assert result["curriculum_added_examples"] == 4
    assert result["training_turns"] == 1
    assert result["position_checks"] == {"target_numeric_invariance_verified": 2}
    review_rows(root, changed_value="12")
    result = audit_curriculum(config)
    assert result["pair_checks"] == {"target_unchanged": 1}
    assert result["curriculum_added_examples"] == 2
    assert not any(row["source_mutations"] for row in rows(root/"train.jsonl") if "source_mutations" in row)


def test_constructed_condition_version_and_multihop_inputs_keep_actual_history_and_native_spans(tmp_path):
    config, template = fixture_config(tmp_path)
    config["reader_curriculum"]["maximum_constructed_cases"] = 1
    result = build_curriculum(config, template=template)
    assert result["constructed_pairs"] == {"condition_swap": 1, "version_swap": 1, "multihop_terminal": 1, "record_presence": 1}
    traces = [row for row in rows(tmp_path/"round2"/"traces.jsonl")
              if row["source_partition"] == "constructed_train_reader_curriculum"]
    assert len(traces) == 12
    assert all(row["task_id"] == "native-task" and row["actual_history"] == [] for row in traces)
    assert all(row["actual_output"] == "" and "target_text" not in row for row in traces)
    assert all(source["source_key"]["corpus_id"] == "constructed_train_reader_curriculum"
               for row in traces for source in row["original_sources"])
    assert all(source["native_parent_span"]["source_key"]["corpus_id"] == "native"
               for row in traces for source in row["original_sources"])
    hops = [row for row in traces if row["curriculum_kind"] == "multihop_terminal"]
    assert [len(row["input_claims_oracle"]) for row in hops] == [0, 1, 2, 0, 1, 2]
    assert [claim["claim_id"] for claim in hops[2]["input_claims_oracle"]] == [
        claim["claim_id"] for claim in hops[5]["input_claims_oracle"]]
    for source in traces[0]["original_sources"]:
        assert "text" not in source["native_parent_span"] and "text" not in source["declared_training_literal"]
    for pair in rows(tmp_path/"round2"/"pairs.prepared.jsonl"):
        if "constructed_control_sources" in pair:
            for source in pair["constructed_control_sources"]:
                literal = source["constructed_literal"]
                assert source["text"][literal["start"]:literal["end"]] == literal["text"]


def test_record_presence_audit_rejects_ghost_records_and_unneeded_mass_followups(tmp_path):
    config, template = fixture_config(tmp_path)
    config["reader_curriculum"]["maximum_constructed_cases"] = 1
    build_curriculum(config, template=template)
    root = tmp_path/"round2"
    review_rows(root, changed_value="19")
    by_id = {row["example_id"]: row for row in rows(root/"traces.jsonl")}
    reviewed = rows(root/"reviews.jsonl")
    for review in reviewed:
        trace = by_id[review["example_id"]]
        if "curriculum_kind" not in trace or trace["curriculum_kind"] != "record_presence":
            continue
        claims = []
        if trace["source_mutations"]:
            claims = [{"need_id": trace["read_task"]["need_ids"][0], "entity": "SR-C0000-1",
                "text": "Shipment record SR-C0000-1", "value": "1", "unit": "1", "followups": []}]
        review["target_text"] = json.dumps({"claims": claims, "support": [], "conflicts": [], "open_needs": []})
    write_rows(root/"reviews.jsonl", reviewed)
    audit_curriculum(config)
    pair = next(pair for pair in rows(root/"pairs.audited.jsonl") if pair["kind"] == "record_presence")
    assert pair["pair_check"] == "source_sensitive_change_verified"
    changed = next(review for review in reviewed if review["example_id"] in pair["changed_example_ids"])
    target = json.loads(changed["target_text"])
    target["claims"][0]["followups"] = [{"missing_relation": "mass"}]
    changed["target_text"] = json.dumps(target)
    write_rows(root/"reviews.jsonl", reviewed)
    audit_curriculum(config)
    pair = next(pair for pair in rows(root/"pairs.audited.jsonl") if pair["kind"] == "record_presence")
    assert pair["pair_check"] == "spurious_record_window_gap"
