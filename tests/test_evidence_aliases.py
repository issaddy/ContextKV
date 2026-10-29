"""Exact body sharing preserves logical sources, current scope and durable execution."""
from dataclasses import replace

import pytest

from adapters.contextpilot import committed_contexts
from controller.candidates import build_candidates
from controller.evidence import EvidenceError, EvidencePolicy, make_evidence
from bench.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import EvidenceSpan, SessionSnapshot
from controller.serialization import dumps, loads, to_jsonable
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import longest_common_prefix, verify_evidence_spans
from test_candidates import committed


BODY = "The upload limit is 10 MB. The original plan conditions still apply.\n" * 20


def sources():
    return (make_evidence(BODY, "a", title="First source", source_uri="https://source/a"),
            make_evidence(BODY, "b", title="Second source", source_uri="https://source/b"))


def verify(candidate, blocks):
    assert verify_evidence_spans(candidate.messages, candidate.evidence_spans, blocks,
                                 candidate.reference_map) == candidate.visible_blocks


def test_rebuild_shares_only_the_body_while_preserving_sources_and_dependencies():
    a, b = sources()
    condition = make_evidence("Only for the regional free plan.", "condition")
    b = replace(b, dependencies=(condition.key,))
    candidate, = build_candidates(SessionSnapshot("s", archive=(condition,)), "Upload limit?", (a, b),
                                   measure_evidence_redundancy=True)
    verify(candidate, (a, b, condition))
    assert candidate.required_blocks == {a.key, b.key, condition.key}
    assert candidate.messages[-1].content.count(BODY) == 1
    root, alias, _ = candidate.evidence_spans
    assert alias.body_reference_id == root.reference_id
    assert (alias.message_index, alias.start, alias.end) == (root.message_index, root.start, root.end)
    assert alias.source_uri == b.source_uri and alias.provenance == b.provenance
    assert all(b.source_uri in candidate.messages[-1].content for b in (a, b))
    assert '"dependencies"' in candidate.messages[-1].content
    assert candidate.evidence_body_tokens == len((BODY + condition.text).encode())


def test_append_reuses_visible_body_under_new_source_scope_without_a_chain():
    a, b = sources()
    snapshot = SessionSnapshot("s")
    first, = build_candidates(snapshot, "First", (a,))
    snapshot = committed(first, snapshot, "First", (a,))
    append, rebuild = build_candidates(snapshot, "Current", (b,), measure_evidence_redundancy=True)
    verify(append, (a, b))
    verify(rebuild, (a, b))
    root, alias = append.evidence_spans
    assert append.messages[:-1] == snapshot.execution.messages
    assert append.required_blocks == rebuild.required_blocks == {b.key}
    scope = append.messages[-1].content.split("Citations allowed for this answer: ")[1].split("\n")[0]
    assert alias.reference_id in scope and root.reference_id not in scope
    assert alias.body_reference_id == root.reference_id
    assert append.evidence_body_tokens == rebuild.evidence_body_tokens == len(BODY.encode())
    assert append.nonrequired_evidence_body_tokens == 0
    assert append.theoretical_lcp == longest_common_prefix(
        append.input_ids, snapshot.execution.last_input_ids + snapshot.execution.last_output_ids)
    second = committed(append, snapshot, "Current", (b,))
    assert committed_contexts(second)[:2] == ((a.key,), (b.key,))
    third = make_evidence(BODY, "c")
    again = build_candidates(second, "Third", (third,))[0]
    verify(again, (a, b, third))
    assert again.evidence_spans[-1].body_reference_id == root.reference_id
    assert sum(m.content.count(BODY) for m in again.messages) == 1


def test_archive_or_quoted_answer_does_not_authorize_body_alias():
    a, b = sources()
    other = make_evidence("An unrelated original.", "other")
    snapshot = SessionSnapshot("s", archive=(a,))
    first, = build_candidates(snapshot, "First", (other,))
    snapshot = committed(first, snapshot, "First", (other,), answer=BODY)
    for candidate in build_candidates(snapshot, "Current", (b,)):
        verify(candidate, (a, b, other))
        assert candidate.evidence_spans[-1].body_reference_id is None
        assert BODY in candidate.messages[-1].content


def test_equal_body_versions_still_require_explicit_comparison():
    old = make_evidence(BODY, "api")
    new = make_evidence(BODY, "api", version="v2")
    snapshot = SessionSnapshot("s")
    snapshot = committed(build_candidates(snapshot, "First", (old,))[0], snapshot, "First", (old,))
    append, rebuild = build_candidates(snapshot, "Current version", (new,))
    assert not append.feasible and append.reason == "invalid_visible_version" and rebuild.feasible
    for candidate in build_candidates(snapshot, "Compare api v1 and v2", (old, new)):
        verify(candidate, (old, new))
        assert candidate.required_blocks == {old.key, new.key} and candidate.feasible
        assert candidate.evidence_spans[-1].body_reference_id is not None


@pytest.mark.parametrize("left,right", [("short", "short"), (BODY, BODY + " ")])
def test_tiny_bodies_and_nonidentical_bytes_are_rendered_normally(left, right):
    a, b = make_evidence(left, "a"), make_evidence(right, "b")
    candidate, = build_candidates(SessionSnapshot("s"), "Question", (a, b))
    verify(candidate, (a, b))
    assert all(span.body_reference_id is None for span in candidate.evidence_spans)
    assert "Evidence alias" not in candidate.messages[-1].content


def test_alias_proof_requires_a_complete_physical_root_and_own_declaration():
    blocks = sources()
    candidate, = build_candidates(SessionSnapshot("s"), "Question", blocks)
    root, alias = candidate.evidence_spans
    with pytest.raises(EvidenceError, match="referenced complete physical body"):
        verify_evidence_spans(candidate.messages, (alias,), blocks, candidate.reference_map)
    with pytest.raises(EvidenceError, match="identity/provenance declaration"):
        index, start, end = alias.alias_declaration
        verify_evidence_spans(candidate.messages, (root, replace(alias, alias_declaration=(index,start+1,end))),
                              blocks, candidate.reference_map)
    with pytest.raises(EvidenceError, match="complete original body"):
        verify_evidence_spans(candidate.messages, (root, replace(alias, end=alias.end-1)),
                              blocks, candidate.reference_map)
    assert loads(EvidenceSpan, dumps(alias)) == alias
    legacy = to_jsonable(root)
    legacy.pop("body_reference_id")
    legacy.pop("alias_declaration")
    assert loads(EvidenceSpan, dumps(legacy)) == root


@pytest.mark.asyncio
async def test_aliases_survive_commit_reopen_replay_and_rebuild(tmp_path):
    path = tmp_path / "session.sqlite3"
    blocks = sources()
    store = SessionStore(path)
    controller = Controller(store, MockEngine(), knobs=PlannerKnobs(strategy="always_append"))
    try:
        await controller.serve_turn("s", "First", blocks[:1], request_id="first")
        result = await controller.serve_turn("s", "Second", blocks[1:], request_id="second")
        assert result.action == "APPEND"
        await controller.drain()
    finally:
        await controller.aclose()
        store.close()
    store = SessionStore(path)
    controller = Controller(store, MockEngine(), knobs=PlannerKnobs(strategy="always_append"))
    try:
        replay = await controller.serve_turn("s", "Second", blocks[1:], request_id="second")
        assert replay.idempotent_replay
        assert replay.visible_evidence_manifest[-1].body_reference_id is not None
        third = await controller.serve_turn("s", "Third", blocks[1:], request_id="third")
        assert third.action == "REBUILD"  # Engine restart requires fresh execution.
        assert all(s.body_reference_id is None for s in third.visible_evidence_manifest)
    finally:
        await controller.aclose()
        store.close()


def test_qwen3_alias_reduces_real_input_and_preserves_append_prefix(qwen3_tokenizer_template):
    template = qwen3_tokenizer_template
    a, b = sources()
    first, = build_candidates(SessionSnapshot("s"), "First", (a,), template=template)
    snapshot = committed(first, SessionSnapshot("s"), "First", (a,))
    # Use the real template identity/output IDs, not the byte-only fixture defaults.
    snapshot = replace(snapshot, execution=replace(snapshot.execution,
        template_hash=template.fingerprint.template_hash, model_revision=template.fingerprint.model_revision,
        last_output_ids=tuple(template.tokenizer.encode("Complete original answer.", add_special_tokens=False))))
    append, _ = build_candidates(snapshot, "Current", (b,), template=template)
    expanded = make_evidence(BODY + "\n", "b")
    unshared, _ = build_candidates(snapshot, "Current", (expanded,), template=template)
    assert append.feasible and len(append.input_ids) < len(unshared.input_ids)
    assert append.theoretical_lcp == unshared.theoretical_lcp
    assert BODY in template.tokenizer.decode(append.input_ids[:append.theoretical_lcp])
    limited = EvidencePolicy(max_total_model_tokens=len(append.input_ids) + 64, max_new_tokens=64)
    assert build_candidates(snapshot, "Current", (b,), limited, template)[0].feasible
    assert not build_candidates(snapshot, "Current", (expanded,), limited, template)[0].feasible
