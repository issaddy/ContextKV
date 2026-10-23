from dataclasses import FrozenInstanceError, replace
import copy

import pytest

from controller.candidates import build_candidates
from controller.evidence import (EvidenceError, EvidencePolicy, assign_references,
                                 make_evidence, select_shared_required_evidence)
from controller.schema import ExecutionLedger, Message, SessionSnapshot, TurnRecord
from controller.templates import SYSTEM_PROMPT, ByteTemplate, longest_common_prefix, token_hash, verify_evidence_spans


def committed(candidate, snapshot, query, blocks, answer="Complete original answer."):
    template = ByteTemplate()
    output = template.tokenizer.encode(answer)
    history = snapshot.raw_history + (TurnRecord(len(snapshot.raw_history)+1, query, answer,
                                      output, evidence_keys=tuple(sorted(candidate.required_blocks))),)
    archive = {b.key:b for b in (*snapshot.archive,*blocks)}
    ledger = ExecutionLedger(1,"engine-session",snapshot.raw_history_version+1,
        candidate.messages+(Message("assistant",answer),),candidate.input_ids,output,
        candidate.visible_blocks,template.fingerprint.template_hash,
        template.fingerprint.model_revision,snapshot.corpus_snapshot_id,
        candidate.evidence_spans,candidate.reference_map)
    return replace(snapshot,version=snapshot.version+1,raw_history=history,
        raw_history_version=snapshot.raw_history_version+1,execution=ledger,
        archive=tuple(archive.values()),reference_map=candidate.reference_map)


def test_first_turn_rebuild_and_pure_reordered_repeat():
    a,b = make_evidence("BODY_ALPHA","a"),make_evidence("BODY_BETA","b")
    snap = SessionSnapshot("s")
    first = build_candidates(snap,"first",(a,b))
    assert [c.action for c in first] == ["REBUILD"]
    snap = committed(first[0],snap,"first",(a,b))
    frozen = copy.deepcopy(snap)
    append,rebuild = build_candidates(snap,"second",(b,a))
    assert snap == frozen
    assert append.required_blocks == rebuild.required_blocks == {a.key,b.key}
    assert append.messages[-1].content.count("BODY_ALPHA") == 0
    assert sum(m.content.count("BODY_ALPHA") for m in append.messages) == 1
    assert append.theoretical_lcp == longest_common_prefix(
        append.input_ids,snap.execution.last_input_ids+snap.execution.last_output_ids)
    assert append.input_hash == token_hash(append.input_ids)
    assert append.reference_map == first[0].reference_map
    with pytest.raises(FrozenInstanceError):
        append.messages[0].content = "mutable"


def test_historical_revisit_is_restored_after_rebuild():
    a,b,c = (make_evidence(f"BODY_{x}",x) for x in "ABC")
    snap = SessionSnapshot("s")
    first = build_candidates(snap,"first",(a,))[0]
    snap = committed(first,snap,"first",(a,))
    second = build_candidates(snap,"second",(b,))[-1]
    snap = committed(second,snap,"second",(b,))
    ref = next(ref for ref,key in snap.reference_map if key == a.key)
    append,rebuild = build_candidates(snap,f"Revisit [{ref}]",(c,))
    for candidate in (append,rebuild):
        assert a.key in candidate.required_blocks <= candidate.visible_blocks
        assert "BODY_A" in candidate.messages[-1].content


def test_raw_history_is_complete_and_old_evidence_removed_on_rebuild():
    a,b = make_evidence("OLD_FULL_EVIDENCE","a"),make_evidence("CURRENT_EVIDENCE","b")
    query,answer = "exact old question\nwith details", "complete answer\nwith all conditions"
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap,query,(a,))[0],snap,query,(a,),answer)
    rebuild = build_candidates(snap,"next",(b,))[-1]
    assert rebuild.messages[1:3] == (Message("user",query),Message("assistant",answer))
    assert "OLD_FULL_EVIDENCE" not in "".join(m.content for m in rebuild.messages)


def test_changed_version_or_content_makes_append_invalid():
    a = make_evidence("old config","api")
    new = make_evidence("new config","api",version="v2")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap,"first",(a,))[0],snap,"first",(a,))
    append,rebuild = build_candidates(snap,"latest configuration",(new,))
    assert not append.feasible and append.reason == "invalid_visible_version"
    assert rebuild.feasible and rebuild.visible_blocks == {new.key}
    same_version_changed = make_evidence("changed without bump","api")
    assert not build_candidates(snap,"latest",(same_version_changed,))[0].feasible


def test_explicit_version_comparison_restores_both_versions():
    old,new = make_evidence("old","api"),make_evidence("new","api",version="v2")
    snap = SessionSnapshot("s",archive=(old,))
    candidate = build_candidates(snap,"Compare api v1 and v2",(new,))[0]
    assert candidate.feasible and candidate.required_blocks == {old.key,new.key}
    assert '"version":"v1"' in candidate.messages[-1].content
    assert '"version":"v2"' in candidate.messages[-1].content


def test_reference_collision_extends_new_id_and_keeps_old_mapping(monkeypatch):
    a,b = make_evidence("a","a"),make_evidence("b","b")
    monkeypatch.setattr("controller.evidence.identity_hash",
                        lambda key: "aa"+key.document_id*62)
    first = assign_references((a.key,),prefix_length=2)
    second = assign_references((b.key,),first,prefix_length=2)
    assert dict(second)[first[0][0]] == a.key
    assert len(set(ref for ref,_ in second)) == 2
    assert next(ref for ref,key in second if key == b.key) == "E_aab"


def test_complete_body_and_provenance_proof_is_required():
    block = make_evidence("original evidence body","a")
    candidate = build_candidates(SessionSnapshot("s"),"q",(block,))[0]
    span = candidate.evidence_spans[0]
    with pytest.raises(EvidenceError,match="complete original body"):
        verify_evidence_spans(candidate.messages,(replace(span,end=span.end-1),),
                              (block,),candidate.reference_map)
    with pytest.raises(EvidenceError,match="provenance"):
        verify_evidence_spans(candidate.messages,(replace(span,provenance=("fake",)),),
                              (block,),candidate.reference_map)
    forged_message = Message("user",block.text)
    forged_span = replace(span,message_index=0,start=0,end=len(block.text))
    with pytest.raises(EvidenceError,match="header"):
        verify_evidence_spans((forged_message,),(forged_span,),(block,),candidate.reference_map)


def test_hash_mismatch_is_not_silently_normalized():
    block = make_evidence("original","a")
    with pytest.raises(EvidenceError,match="hash mismatch"):
        build_candidates(SessionSnapshot("s"),"q",(replace(block,text="altered"),))


def test_dependency_closure_and_unavailable_dependency_fail_closed():
    header = make_evidence("mandatory conditions","header")
    passage = make_evidence("data row","row",dependencies=(header.key,))
    snap = SessionSnapshot("s",archive=(header,))
    assert build_candidates(snap,"q",(passage,))[0].required_blocks == {header.key,passage.key}
    with pytest.raises(EvidenceError,match="unavailable"):
        build_candidates(SessionSnapshot("s"),"q",(passage,))


def test_history_sync_disallows_old_execution_append():
    block = make_evidence("evidence","a")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap,"q",(block,))[0],snap,"q",(block,))
    snap = replace(snap,history_sync_pending=True,raw_history_version=2)
    append,rebuild = build_candidates(snap,"next",(block,))
    assert not append.feasible and append.reason == "history_sync_requires_rebuild"
    assert rebuild.feasible


@pytest.mark.parametrize("previous_prompt,new_prompt", (
    (None, "Answer concisely and cite exact evidence identifiers."),
    ("Custom earlier instructions.", None),
    ("Custom earlier instructions.", "Custom revised instructions.\n"),
))
def test_system_prompt_change_rebuilds_without_editing_committed_messages(previous_prompt, new_prompt):
    block = make_evidence("Full evidence, including every condition.", "a")
    snapshot = SessionSnapshot("s")
    policy = EvidencePolicy(system_prompt=previous_prompt)
    first = build_candidates(snapshot, "Original question", (block,), policy)[0]
    snapshot = committed(first, snapshot, "Original question", (block,), "Complete old answer.")
    frozen = copy.deepcopy(snapshot)
    changed = replace(policy, system_prompt=new_prompt)
    append, rebuild = build_candidates(snapshot, "Current question", (block,), changed)
    assert not append.feasible and append.reason == "system_prompt_changed"
    assert append.messages[:-1] == snapshot.execution.messages
    assert snapshot == frozen
    assert rebuild.feasible
    assert rebuild.messages[0] == Message("system", SYSTEM_PROMPT if new_prompt is None else new_prompt)
    assert rebuild.messages[1:3] == (Message("user", "Original question"), Message("assistant", "Complete old answer."))
    assert rebuild.required_blocks == append.required_blocks == {block.key}
    assert rebuild.visible_blocks == {block.key}
    assert block.text in rebuild.messages[-1].content
    assert rebuild.reference_map == first.reference_map
    assert rebuild.fingerprint == first.fingerprint
    snapshot = committed(rebuild, snapshot, "Current question", (block,))
    assert build_candidates(snapshot, "Next question", (block,), changed)[0].feasible


def test_explicit_default_system_prompt_does_not_force_rebuild():
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot, "Original question", ())[0]
    assert first.messages[0] == Message("system", SYSTEM_PROMPT)
    snapshot = committed(first, snapshot, "Original question", ())
    append = build_candidates(snapshot, "Next question", (), EvidencePolicy(system_prompt=SYSTEM_PROMPT))[0]
    assert append.feasible


def test_shared_budget_does_not_silently_trim_history_or_evidence():
    a = make_evidence("A"*100,"a")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap,"q",(a,))[0],snap,"q",(a,),"X"*1000)
    candidates = build_candidates(snap,"q2",(a,),EvidencePolicy(max_total_model_tokens=100,max_new_tokens=20))
    assert all(not c.feasible and c.reason == "shared_context_policy_overflow" for c in candidates)
    assert all("X"*1000 in [m.content for m in c.messages] for c in candidates)


def test_bare_reference_hint_is_not_visible_body():
    block = make_evidence("body","a")
    candidate = build_candidates(SessionSnapshot("s"),"q",(block,))[0]
    assert not verify_evidence_spans((Message("user","[E_hint]"),),(),
                                     (block,),candidate.reference_map)


@pytest.mark.parametrize("omitted_body", ("BODY_OLD", "BODY_NEW"))
def test_template_cannot_drop_committed_or_new_evidence_payload(omitted_body):
    old, new = make_evidence("BODY_OLD", "a"), make_evidence("BODY_NEW", "b")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap,"q",(old,))[0],snap,"q",(old,))

    class DroppingTemplate(ByteTemplate):
        def render(self, messages):
            return super().render(tuple(replace(message,
                content=message.content.replace(omitted_body, "")) for message in messages))

    with pytest.raises(EvidenceError, match="template omitted or transformed"):
        build_candidates(snap, "next", (new,), template=DroppingTemplate())


def test_current_candidate_cannot_read_future_or_truth():
    block = make_evidence("body","a")
    with pytest.raises(EvidenceError,match="unknown evidence reference"):
        select_shared_required_evidence("[E_future]",(block,),SessionSnapshot("s"))


def test_snapshot_copies_mutable_containers():
    original = [make_evidence("body","a")]
    snap = SessionSnapshot("s",archive=original)
    original.clear()
    assert len(snap.archive) == 1


def test_configuration_anchor_does_not_restore_superseded_version_from_archive():
    old = make_evidence("old timeout","api",anchors=("connect_timeout",))
    new = make_evidence("new timeout","api",version="v2",anchors=("connect_timeout",))
    candidate = build_candidates(SessionSnapshot("s",archive=(old,)),
                                 "What is connect_timeout?",(new,))[0]
    assert candidate.required_blocks == {new.key}


def test_shared_configuration_anchor_does_not_restore_unrelated_product():
    old = make_evidence("Product alpha connect_timeout is 10.", "alpha",
                        anchors=("connect_timeout",))
    current = make_evidence("Product beta connect_timeout is 30.", "beta",
                            anchors=("connect_timeout",))
    snapshot = SessionSnapshot("s", archive=(old,))
    selected = build_candidates(snapshot, "What is connect_timeout?", (current,))[-1]
    assert selected.required_blocks == {current.key}
    explicit = build_candidates(snapshot, "What is alpha connect_timeout?", (current,))[-1]
    assert explicit.required_blocks == {old.key, current.key}


def test_ambiguous_historical_anchor_requires_document_scope():
    a = make_evidence("timeout 10", "alpha", anchors=("connect_timeout",))
    b = make_evidence("timeout 30", "beta", anchors=("connect_timeout",))
    current = make_evidence("Available deployment regions", "regions")
    snapshot = SessionSnapshot("s", archive=(a, b))
    candidate = build_candidates(snapshot, "What is connect_timeout?", (current,))[-1]
    assert candidate.required_blocks == {current.key}


def test_anchor_respects_named_historical_version():
    old = make_evidence("timeout 10", "api", version="v1", anchors=("connect_timeout",))
    new = make_evidence("timeout 30", "api", version="v2", anchors=("connect_timeout",))
    current = make_evidence("Deployment regions", "regions")
    candidate = build_candidates(SessionSnapshot("s", archive=(old, new)),
                                 "What is api v2 connect_timeout?", (current,))[-1]
    assert candidate.feasible and candidate.required_blocks == {current.key, new.key}


def test_version_token_does_not_match_longer_version_substring():
    old = make_evidence("old","api",version="v1")
    new = make_evidence("new","api",version="v10")
    candidate = build_candidates(SessionSnapshot("s",archive=(old,)),"api v10",(new,))[0]
    assert candidate.required_blocks == {new.key}


def test_natural_revisit_restores_recent_condition_after_rebuild():
    condition = make_evidence("仅限高级套餐，限制为三次重试。", "limits")
    other = make_evidence("账户头像可以更换。", "profile")
    current = make_evidence("服务也支持导出。", "export")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "重试怎样配置？", (condition,))[0],
                     snap, "重试怎样配置？", (condition,))
    snap = committed(build_candidates(snap, "怎样更换头像？", (other,))[-1],
                     snap, "怎样更换头像？", (other,))
    for layout in ("stable_key", "contextpilot"):
        candidates = build_candidates(snap, "前面提到的限制是什么？", (current,),
                                      EvidencePolicy(layout=layout))
        for candidate in candidates:
            assert condition.key in candidate.required_blocks <= candidate.visible_blocks
            assert condition.text in candidate.messages[-1].content
            assert dict(candidate.evidence_resolution)[condition.key].startswith("natural_revisit:")
    explicit_only = build_candidates(snap, "前面提到的限制是什么？", (current,),
                                    EvidencePolicy(revisit_mode="explicit"))[-1]
    assert condition.key not in explicit_only.required_blocks


def test_natural_revisit_does_not_guess_between_two_conditions():
    blocks = (make_evidence("First service has a limit of 3", "one"),
              make_evidence("Second service has a limit of 7", "two"))
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "What are the limits?", blocks)[0],
                     snap, "What are the limits?", blocks)
    current = make_evidence("Current service data", "three")
    candidate = build_candidates(snap, "What about that limit?", (current,))[-1]
    assert candidate.required_blocks == {current.key}


def test_topic_without_revisit_cue_does_not_restore_old_material():
    old = make_evidence("Timeout is 30 seconds", "old")
    new = make_evidence("Timeout is 50 seconds", "new")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "q", (old,))[0], snap, "q", (old,))
    assert build_candidates(snap, "What is the timeout?", (new,))[-1].required_blocks == {new.key}


def test_natural_revisit_does_not_restore_superseded_version():
    old = make_evidence("Timeout is 30 seconds", "api")
    new = make_evidence("Timeout is 50 seconds", "api", version="v2")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "q", (old,))[0], snap, "q", (old,))
    candidate = build_candidates(snap, "Is that timeout still applicable?", (new,))[-1]
    assert candidate.required_blocks == {new.key} and candidate.feasible


@pytest.mark.parametrize("query", ("旧版本是否也适用？", "Does the previous version also support this?"))
def test_explicit_prior_revision_revisit_uses_committed_identity(query):
    old = make_evidence("Only enterprise plans", "api", version="v1")
    new = make_evidence("All plans", "api", version="v2")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "q", (old,))[0], snap, "q", (old,))
    snap = committed(build_candidates(snap, "latest", (new,))[-1], snap, "latest", (new,))
    candidate = build_candidates(snap, query, (new,))[-1]
    assert candidate.feasible and candidate.required_blocks == {old.key, new.key}
    assert dict(candidate.evidence_resolution)[old.key].endswith(":prior_revision")


def test_contextpilot_layout_is_independent_pure_and_keeps_complete_history():
    z, a, m = (make_evidence(f"body for {name}", name) for name in ("z", "a", "m"))
    policy = EvidencePolicy(layout="contextpilot")
    snap = SessionSnapshot("s")
    first = build_candidates(snap, "first question", (z, a), policy)[0]
    stable = build_candidates(snap, "first question", (z, a))[0]
    assert [span.key for span in first.evidence_spans] == [z.key, a.key]
    assert [span.key for span in stable.evidence_spans] == [a.key, z.key]
    snap = committed(first, snap, "first question", (z, a), "full answer with conditions")
    before = copy.deepcopy(snap)
    append, rebuild = build_candidates(snap, "second question", (a, m, z), policy)
    assert snap == before
    assert append.required_blocks == rebuild.required_blocks == {a.key, z.key, m.key}
    assert [span.key for span in rebuild.evidence_spans] == [z.key, a.key, m.key]
    assert rebuild.messages[1:3] == (Message("user", "first question"),
                                   Message("assistant", "full answer with conditions"))
    assert "Refer to evidence [E_" in append.messages[-1].content
    assert z.text not in append.messages[-1].content
    assert rebuild.reference_map == append.reference_map


def test_contextpilot_tail_annotation_preserves_original_retrieval_priority():
    a, b, c = (make_evidence("body " + name, name) for name in ("a", "b", "c"))
    policy = EvidencePolicy(layout="contextpilot")
    snap = SessionSnapshot("s")
    first = build_candidates(snap, "old question", (b, a), policy)[0]
    snap = committed(first, snap, "old question", (b, a), "Complete answer and condition.")
    old_messages = snap.execution.messages
    append, rebuild = build_candidates(snap, "new question", (a, c, b), policy)
    inverse = {key: ref for ref, key in rebuild.reference_map}
    annotation = "Retrieval priority (most relevant first): " + " > ".join(
        f"[{inverse[block.key]}]" for block in (a, c, b)) + "."
    stable = build_candidates(snap, "new question", (a, c, b))[-1]
    assert "Retrieval priority" not in stable.messages[-1].content
    assert append.required_blocks == rebuild.required_blocks == stable.required_blocks
    assert append.messages[:-1] == old_messages
    assert rebuild.messages[1:3] == stable.messages[1:3]
    assert [span.key for span in rebuild.evidence_spans] == [b.key, a.key, c.key]
    for candidate in (append, rebuild):
        text = candidate.messages[-1].content
        assert annotation in text
        assert text.rindex("</evidence_body>") < text.index(annotation) < text.index("Question:")
    different_question = build_candidates(snap, "another question", (a, c, b), policy)[-1]
    shared_ids = rebuild.input_ids[:longest_common_prefix(rebuild.input_ids, different_question.input_ids)]
    assert annotation in ByteTemplate().tokenizer.decode(shared_ids)


def test_contextpilot_annotations_never_replace_evicted_visible_body():
    a, b = make_evidence("BODY A", "a"), make_evidence("BODY B", "b")
    snap = SessionSnapshot("s")
    policy = EvidencePolicy(layout="contextpilot")
    snap = committed(build_candidates(snap, "first", (a,), policy)[0], snap, "first", (a,))
    snap = committed(build_candidates(snap, "second", (b,), policy)[-1], snap, "second", (b,))
    append = build_candidates(snap, "revisit", (a,), policy)[0]
    assert a.text in append.messages[-1].content
    assert "Refer to evidence" not in append.messages[-1].content


def test_contextpilot_distance_matches_documented_overlap_position_metric():
    from adapters.contextpilot import context_distance
    a, b, c = (make_evidence(name, name).key for name in ("a", "b", "c"))
    assert context_distance((a, b), (a, b)) == 0
    assert context_distance((a, b), (b, a)) == pytest.approx(.001)
    assert context_distance((a, b), (a, c)) == .5
    assert context_distance((), (a,)) == 1


@pytest.mark.parametrize("layout", ("stable_key", "contextpilot"))
def test_evidence_before_question_preserves_real_token_prefix_across_queries(layout):
    block = make_evidence("Shared body " * 40, "shared")
    policy = EvidencePolicy(layout=layout)
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot, "Which option is supported?", (block,), policy)[0]
    second = build_candidates(snapshot, "How do I enable the option?", (block,), policy)[0]
    content = first.messages[-1].content
    assert content.index(block.text) < content.index("Question:")
    assert longest_common_prefix(first.input_ids, second.input_ids) > len(block.text)
    assert first.input_hash != second.input_hash
