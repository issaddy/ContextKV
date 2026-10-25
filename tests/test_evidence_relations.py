"""Declared corpus relations survive both candidate forms, without rewriting bodies."""
from dataclasses import replace
import json

from adapters.corpus import CorpusIndex
from controller import candidates as candidate_module
from controller.candidates import CandidatePolicy, build_candidates
from controller.evidence import assign_references, make_evidence
from controller.schema import ExecutionLedger, Message, SessionSnapshot, TurnRecord
from controller.templates import (SYSTEM_PROMPT, ByteTemplate, current_user_message,
                                  current_source_guide, render_evidence_relations, verify_evidence_spans)


def relations(message):
    prefix = "Declared evidence metadata:\n"
    return json.loads(message.content.split(prefix, 1)[1].split("\n", 1)[0])


def legacy_snapshot(blocks):
    """A persisted pre-annotation execution; no new format is applied retroactively."""
    template = ByteTemplate()
    refs = assign_references(b.key for b in blocks)
    user, spans = current_user_message("Earlier question", blocks, refs, message_index=1)
    messages = (Message("system", SYSTEM_PROMPT), user)
    rendered = template.render(messages)
    answer = "The actual earlier answer."
    output = template.tokenizer.encode(answer)
    keys = frozenset(b.key for b in blocks)
    execution = ExecutionLedger(1, "old-engine-session", 1,
        messages + (Message("assistant", answer),), rendered.input_ids, output,
        keys, template.fingerprint.template_hash, template.fingerprint.model_revision,
        "snapshot", spans, refs)
    return SessionSnapshot("s", version=1, raw_history_version=1,
        raw_history=(TurnRecord(1, "Earlier question", answer, output, evidence_keys=tuple(sorted(keys))),),
        archive=tuple(blocks), execution=execution, reference_map=refs, corpus_snapshot_id="snapshot")


def test_same_bodies_with_swapped_declared_edges_produce_different_actual_inputs():
    def index(swapped):
        return CorpusIndex.from_records([
            {"_id": "a", "text": "The request count is 4.", "dependencies": ["d" if swapped else "c"]},
            {"_id": "b", "text": "The request count is 8.", "dependencies": ["c" if swapped else "d"]},
            {"_id": "c", "text": "Only batch requests."},
            {"_id": "d", "text": "Only interactive requests."},
        ], corpus_id="manual", snapshot_id="snapshot")

    left, right = index(False), index(True)
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot, "What request count applies?", left.resolve(("a", "b")))[0]
    second = build_candidates(snapshot, "What request count applies?", right.resolve(("a", "b")))[0]
    assert first.required_blocks == second.required_blocks and first.reference_map == second.reference_map
    assert first.evidence_spans == second.evidence_spans
    assert first.input_ids != second.input_ids and first.input_hash != second.input_hash
    refs = {key.passage_id: ref for ref, key in first.reference_map}
    assert relations(first.messages[-1]) == [
        {"id": refs["a"], "dependencies": [refs["c"]]},
        {"id": refs["b"], "dependencies": [refs["d"]]},
    ]
    assert relations(second.messages[-1]) == [
        {"id": refs["a"], "dependencies": [refs["d"]]},
        {"id": refs["b"], "dependencies": [refs["c"]]},
    ]
    for candidate, corpus in ((first, left), (second, right)):
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
                                    corpus.blocks, candidate.reference_map) == candidate.required_blocks


def test_append_from_pre_annotation_ledger_gets_current_relations_once(monkeypatch):
    condition = make_evidence("Complete original condition.", "condition")
    row = replace(make_evidence("Complete original value.", "value", dependencies=(condition.key,)),
                  parent_section='Limits "A"\n子节')
    blocks = (row, condition)
    snapshot = legacy_snapshot(blocks)
    assert "Declared evidence metadata:" not in snapshot.execution.messages[1].content
    original = candidate_module.render_evidence_relations
    calls = []

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(candidate_module, "render_evidence_relations", counted)
    append, rebuild = build_candidates(snapshot, "Current question", blocks)
    assert calls == [True]
    assert append.feasible and rebuild.feasible
    assert append.messages[:-1] == snapshot.execution.messages
    assert append.evidence_spans == snapshot.execution.evidence_spans
    assert row.text not in append.messages[-1].content
    assert relations(append.messages[-1]) == relations(rebuild.messages[-1])
    ref = {key: reference for reference, key in append.reference_map}
    assert relations(append.messages[-1]) == [{"id": ref[row.key], "dependencies": [ref[condition.key]],
                                             "parent_section": row.parent_section}]
    assert rebuild.messages[1:3] == (Message("user", "Earlier question"),
                                     Message("assistant", "The actual earlier answer."))


def test_cycles_preserve_exact_revision_edges_without_inferred_relation_types():
    index = CorpusIndex.from_records([
        {"_id": passage, "version": version, "text": f"{passage} body for {version}",
         "dependencies": [other]}
        for version in ("v1", "v2") for passage, other in (("a", "b"), ("b", "a"))
    ], corpus_id="manual", snapshot_id="snapshot")
    selected = index.resolve(block.key for block in index.blocks if block.key.passage_id == "a")
    candidate = build_candidates(SessionSnapshot("s"), "Compare versions v1 and v2", selected)[0]
    assert candidate.feasible and len(candidate.required_blocks) == 4
    ref_to_key = dict(candidate.reference_map)
    rows = relations(candidate.messages[-1])
    assert len(rows) == 4 and all(set(row) == {"id", "dependencies"} for row in rows)
    for row in rows:
        owner = ref_to_key[row["id"]]
        dependency, = (ref_to_key[ref] for ref in row["dependencies"])
        assert owner.version == dependency.version and owner.passage_id != dependency.passage_id


def test_revision_change_still_rebuilds_and_never_relabels_old_source():
    old_condition = make_evidence("Old condition.", "condition", version="v1")
    old_value = make_evidence("Old value.", "value", version="v1", dependencies=(old_condition.key,))
    snapshot = legacy_snapshot((old_condition, old_value))
    new_condition = make_evidence("New condition.", "condition", version="v2")
    new_value = make_evidence("New value.", "value", version="v2", dependencies=(new_condition.key,))
    append, rebuild = build_candidates(snapshot, "Current values", (new_value, new_condition))
    assert not append.feasible and append.reason == "invalid_visible_version"
    assert rebuild.feasible and rebuild.required_blocks == {new_value.key, new_condition.key}
    refs = {key: ref for ref, key in rebuild.reference_map}
    assert relations(rebuild.messages[-1]) == [{"id": refs[new_value.key],
                                              "dependencies": [refs[new_condition.key]]}]
    assert old_condition.text not in rebuild.messages[-1].content
    assert append.messages[:-1] == snapshot.execution.messages


def test_absent_metadata_keeps_existing_message_bytes_tokens_spans_and_hash():
    block = make_evidence("The complete original body.\n", "plain")
    snapshot = legacy_snapshot((block,))
    append, rebuild = build_candidates(snapshot, "Next question", (block,))
    template = ByteTemplate()
    assert render_evidence_relations((block,), snapshot.reference_map) == ""
    guide = current_source_guide("Next question", (block,), snapshot.reference_map)
    current, spans = current_user_message("Next question", (), snapshot.reference_map,
        message_index=len(snapshot.execution.messages), reused=(block.key,), source_guide=guide)
    expected_append = snapshot.execution.messages + (current,)
    current_rebuild, rebuild_spans = current_user_message("Next question", (block,), snapshot.reference_map,
                                                        message_index=3, source_guide=guide)
    expected_rebuild = (Message("system", SYSTEM_PROMPT), Message("user", "Earlier question"),
                        Message("assistant", "The actual earlier answer."), current_rebuild)
    for candidate, messages, expected_spans in (
            (append, expected_append, snapshot.execution.evidence_spans + spans),
            (rebuild, expected_rebuild, rebuild_spans)):
        rendered = template.render(messages)
        assert candidate.messages == messages and candidate.evidence_spans == expected_spans
        assert candidate.input_ids == rendered.input_ids and candidate.input_hash == rendered.input_hash


def test_annotation_tokens_count_toward_existing_context_budget():
    condition = make_evidence("Original condition.", "condition")
    value = make_evidence("Original value.", "value", dependencies=(condition.key,))
    snapshot = SessionSnapshot("s")
    plain = build_candidates(snapshot, "q", (replace(value, dependencies=()), condition))[0]
    policy = CandidatePolicy(max_total_model_tokens=len(plain.input_ids) + 1, max_new_tokens=1)
    plain_at_limit = build_candidates(snapshot, "q", (replace(value, dependencies=()), condition), policy)[0]
    with_metadata = build_candidates(snapshot, "q", (value, condition), policy)[0]
    assert plain_at_limit.feasible
    assert len(with_metadata.input_ids) > len(plain_at_limit.input_ids)
    assert not with_metadata.feasible and with_metadata.reason == "shared_context_policy_overflow"
    assert with_metadata.required_blocks == plain_at_limit.required_blocks
