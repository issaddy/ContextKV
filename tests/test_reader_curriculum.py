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
    assert result["constructed_read_inputs"] == len(traces)
    assert all(row["task_id"] == "native-task" and row["actual_history"] == [] for row in traces)
    assert all(row["actual_output"] == "" and "target_text" not in row for row in traces)
    assert all(source["source_key"]["corpus_id"] == "constructed_train_reader_curriculum"
               for row in traces for source in row["original_sources"])
    assert all(source["native_parent_span"]["source_key"]["corpus_id"] == "native"
               for row in traces for source in row["original_sources"])
    hops = [row for row in traces if row["curriculum_kind"] == "multihop_terminal"]
    for changed in (False, True):
        variant = [row for row in hops if bool(row["source_mutations"]) == changed]
        depths = [len(row["input_claims_oracle"]) for row in variant]
        assert set(depths) == {0, 1, 2} and depths == sorted(depths)
    for depth in range(3):
        incoming = [row["input_claims_oracle"] for row in hops if len(row["input_claims_oracle"]) == depth]
        assert all(claims == incoming[0] for claims in incoming)
    for source in traces[0]["original_sources"]:
        assert "text" not in source["native_parent_span"] and "text" not in source["declared_training_literal"]
    for pair in rows(tmp_path/"round2"/"pairs.prepared.jsonl"):
        if "constructed_control_sources" in pair:
            for side in ("control", "changed"):
                selected = [row for row in traces if row["example_id"] in pair[f"all_{side}_example_ids"]]
                assert selected and all(row["generation_metrics"]["status"] == "pending_actual_teacher_target" for row in selected)
                source_spans = {}
                for trace in selected:
                    assert trace["prompt_tokens"] <= trace["read_task"]["input_budget"]
                    originals = {json.dumps(source["source_key"], sort_keys=True): source for source in trace["original_sources"]}
                    units = {unit["unit_id"]: unit for unit in trace["context_plan"]["selected_set"]["selected_units"]}
                    assert trace["read_task"]["input_unit_ids"]
                    for identity in trace["read_task"]["input_unit_ids"]:
                        for span in units[identity]["source_spans"]:
                            original = originals[json.dumps(span["source_key"], sort_keys=True)]
                            assert original["text"][span["start"]:span["end"]] == units[identity]["text"]
                            source_spans.setdefault(span["source_key"]["document_id"], set()).update(range(span["start"], span["end"]))
                for source in pair[f"constructed_{side}_sources"]:
                    literal = source["constructed_literal"]
                    assert source["text"][literal["start"]:literal["end"]] == literal["text"]
                    assert all(character.isspace() or offset in source_spans[source["document_id"]]
                               for offset, character in enumerate(source["text"]))


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


def test_pair_audit_requires_fresh_accepted_targets_for_every_packed_read_including_empty_windows(tmp_path):
    import re
    config, template = fixture_config(tmp_path)
    config["reader_curriculum"]["maximum_constructed_cases"] = 1
    build_curriculum(config, template=template)
    root = tmp_path/"round2"
    review_rows(root, changed_value="19")
    reviews = rows(root/"reviews.jsonl")
    traces = {row["example_id"]: row for row in rows(root/"traces.jsonl")}
    pair = next(row for row in rows(root/"pairs.prepared.jsonl") if row["kind"] == "condition_swap")
    all_ids = set((*pair["all_control_example_ids"], *pair["all_changed_example_ids"]))
    empty_control = None
    for review in reviews:
        if review["example_id"] not in all_ids:
            continue
        trace = traces[review["example_id"]]
        input_ids = set(trace["read_task"]["input_unit_ids"])
        actual_window = "\n".join(unit["text"] for unit in trace["context_plan"]["selected_set"]["selected_units"]
                                  if unit["unit_id"] in input_ids)
        value = re.search(r"Ledger-C0000 policy version 2024 on Windows: filing allowance is (\d+) credits", actual_window)
        claims = [{"need_id": trace["read_task"]["need_ids"][0], "entity": "Ledger-C0000",
            "text": f"The Windows allowance is {value[1]} credits.", "value": value[1], "unit": "credits",
            "followups": []}] if value else []
        if not claims and review["example_id"] in pair["all_control_example_ids"]:
            empty_control = review["example_id"]
        review["target_text"] = json.dumps({"claims": claims, "support": [], "conflicts": [], "open_needs": []})
    assert empty_control is not None
    write_rows(root/"reviews.jsonl", reviews)
    audit_curriculum(config)
    result = next(row for row in rows(root/"pairs.audited.jsonl") if row["pair_id"] == pair["pair_id"])
    assert result["pair_check"] == "source_sensitive_change_verified"
    assert set(result["fresh_step_review_statuses"].values()) == {"accepted"}
    write_rows(root/"reviews.jsonl", [review for review in reviews if review["example_id"] != empty_control])
    audit_curriculum(config)
    result = next(row for row in rows(root/"pairs.audited.jsonl") if row["pair_id"] == pair["pair_id"])
    assert result["pair_check"] == "incomplete_accepted_control_steps"
    assert result["fresh_step_review_statuses"][empty_control] == "missing_review"
    assert not any(row["example_id"] in all_ids for row in rows(root/"train.jsonl"))


def test_computed_answer_pairs_bind_full_scope_oracle_and_partial_representative_excerpt(tmp_path):
    from training.reader_computed_curriculum import computed_answer_pairs
    config, template = fixture_config(tmp_path)
    parent = parent_example(template)
    traces, pairs = computed_answer_pairs([parent], template, maximum_cases=1)
    assert len(traces) == 2 and len(pairs) == 1
    assert all(trace["phase"] == "ANSWER" and trace["actual_output"] == "" and "target_text" not in trace for trace in traces)
    assert all(trace["actual_history"] == parent["actual_history"] and trace["task_id"] == parent["task_id"] for trace in traces)
    assert all(trace["offline_scope_audit"]["distinct_dispatches"] == 5
        and trace["offline_scope_audit"]["representative_scope_is_partial"] for trace in traces)
    assert all(len(trace["offline_scope_audit"]["representative_unit_ids"]) == 1 for trace in traces)
    assert [trace["offline_scope_audit"]["mass_g"] for trace in traces] == ["105000", "116000"]
    assert all("Computed results for the complete reading scope:" in trace["messages"][-1]["content"] for trace in traces)
    assert traces[0]["prompt_token_hash"] != traces[1]["prompt_token_hash"]
    mutation = traces[1]["source_mutations"][0]
    original = next(source for source in traces[0]["original_sources"] if source["source_key"] == mutation["source_key"])
    changed = next(source for source in traces[1]["original_sources"] if source["source_key"] == mutation["mutated_source_key"])
    assert original["text"][mutation["start"]:mutation["end"]] == "29"
    assert changed["text"] == original["text"][:mutation["start"]]+"40"+original["text"][mutation["end"]:]
    assert "typed offline construction oracle" in traces[0]["trace_origin"]["computed_claims"]
    assert all(source["native_parent_span"]["source_key"]["corpus_id"] == "native" for source in traces[0]["original_sources"])


def test_computed_curriculum_audit_rejects_locally_correct_representative_totals(tmp_path):
    config, template = fixture_config(tmp_path)
    config["reader_curriculum"]["maximum_computed_cases"] = 1
    manifest = build_curriculum(config, template=template)
    assert manifest["computed_answer_inputs"] == 2
    root = tmp_path/"round2"
    review_rows(root, changed_value="19")
    reviews = rows(root/"reviews.jsonl")
    by_id = {trace["example_id"]: trace for trace in rows(root/"traces.jsonl")}
    for review in reviews:
        trace = by_id[review["example_id"]]
        if trace["phase"] == "ANSWER":
            mass = 116 if trace["source_mutations"] else 105
            review["target_text"] = f"The complete scope has 5 distinct dispatch records, with total mass {mass} kg."
    write_rows(root/"reviews.jsonl", reviews)
    audit_curriculum(config)
    pair = next(row for row in rows(root/"pairs.audited.jsonl") if row["kind"] == "computed_full_scope_answer")
    assert pair["pair_check"] == "source_sensitive_change_verified"
    changed = next(review for review in reviews if review["example_id"] in pair["changed_example_ids"])
    changed["target_text"] = "The complete scope has 1 distinct dispatch record, with total mass 13 kg."
    write_rows(root/"reviews.jsonl", reviews)
    audit_curriculum(config)
    pair = next(row for row in rows(root/"pairs.audited.jsonl") if row["kind"] == "computed_full_scope_answer")
    assert pair["pair_check"] == "computed_scope_totals_not_reproduced"
    assert not any(row.get("curriculum_kind") == "computed_full_scope_answer" for row in rows(root/"train.jsonl"))
