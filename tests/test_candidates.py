from dataclasses import FrozenInstanceError, replace
import copy

import pytest

from controller.candidates import build_candidates
from controller.evidence import (EvidenceError, EvidencePolicy, assign_references,
                                 make_evidence, select_shared_required_evidence)
from controller.schema import ExecutionLedger, Message, QueryContext, SessionSnapshot, TurnRecord
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


def test_current_citation_scope_excludes_history_only_ids_for_both_actions():
    old,new = make_evidence("Old evidence body","old"),make_evidence("Current body","new")
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot,"First question",(old,))[0]
    old_ref = next(ref for ref,key in first.reference_map if key == old.key)
    answer = f"Original answer cites [{old_ref}]."
    snapshot = committed(first,snapshot,"First question",(old,),answer)
    candidates = build_candidates(snapshot,"New question",(new,))
    for plan in candidates:
        new_ref = next(ref for ref,key in plan.reference_map if key == new.key)
        scope = plan.messages[-1].content.split("Citations allowed for this answer: ")[1].split("\n")[0]
        assert new_ref in scope and old_ref not in scope
        assert any(message.content == answer for message in plan.messages)
    assert old.key in candidates[0].visible_blocks and old.key not in candidates[1].visible_blocks
    # An explicit revisit restores the body and citation eligibility together.
    for plan in build_candidates(snapshot,f"Revisit [{old_ref}]",(new,)):
        assert old.key in plan.required_blocks <= plan.visible_blocks
        assert old_ref in plan.messages[-1].content.split("Citations allowed for this answer: ")[1]


def test_current_source_navigation_is_shared_verbatim_and_keeps_cached_prefix():
    import json
    from controller.templates import current_source_guide
    block = make_evidence("General physical activity.\nAccess to safe places to play is important for physical activity.\n"
                          "Additional conditions remain in the full source.", "source", title="Community design")
    old = make_evidence("Obsolete topic and citation.", "old")
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot, "Original question", (old, block))[0]
    snapshot = committed(first, snapshot, "Original question", (old, block))
    query = "How to keep children safe during physical activity?"
    append, rebuild = build_candidates(snapshot, query, (block,))
    assert append.messages[:-1] == snapshot.execution.messages
    guide = current_source_guide(query, (block,), append.reference_map)
    assert guide in append.messages[-1].content and guide in rebuild.messages[-1].content
    row, = json.loads(guide.split("\n", 1)[1])
    assert row == {"id": row["id"], "source": block.source_uri, "title": "Community design"}
    assert row["id"] == next(ref for ref, key in append.reference_map if key == block.key)
    assert old.key not in append.required_blocks and old.key in append.visible_blocks
    for plan in (append, rebuild):
        assert any(m.content[s.start:s.end] == block.text for s in plan.evidence_spans
                   for m in (plan.messages[s.message_index],) if s.key == block.key)


@pytest.mark.parametrize("costs,action", [((10, 30), "APPEND"), ((30, 10), "REBUILD")])
def test_disjoint_evidence_keeps_both_options_and_selects_by_actual_cost(costs, action):
    from controller.planner import choose_plan
    from test_planner import MeasuredModel, batch
    old, new = make_evidence("Old complete body", "old"), make_evidence("New complete body", "new")
    snapshot = SessionSnapshot("s")
    snapshot = committed(build_candidates(snapshot, "First", (old,))[0], snapshot, "First", (old,))
    append, rebuild = build_candidates(snapshot, "Independent new question", (new,))
    assert append.feasible and append.required_blocks == rebuild.required_blocks == {new.key}
    assert append.theoretical_lcp > rebuild.theoretical_lcp
    assert old.key not in append.required_blocks and old.key in append.visible_blocks
    assert rebuild.feasible and rebuild.messages[2].content == snapshot.raw_history[0].assistant_text
    quotes = replace(batch(append, rebuild), fingerprint=append.fingerprint)
    assert choose_plan((append, rebuild), quotes, MeasuredModel(*costs), snapshot).candidate.action == action
    append, rebuild = build_candidates(snapshot, "Followup", (old, new))
    assert append.feasible and rebuild.feasible


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
    candidate = build_candidates(snap,"Compare api v1 and v2",(old,new))[0]
    assert candidate.feasible and candidate.required_blocks == {old.key,new.key}
    assert '"source_revision":"v1"' in candidate.messages[-1].content
    assert '"source_revision":"v2"' in candidate.messages[-1].content


@pytest.mark.parametrize("document_id,title,query,current_document", (
    ("api", "", "What is the rapid v1 retry interval?", "rapid"),
    ("support-guide", "Support", "How are unsupported v1 settings handled?", "unsupported"),
))
def test_explicit_version_revisit_does_not_match_part_of_another_document_name(
        document_id, title, query, current_document):
    old = make_evidence("Unrelated archived source", document_id, title=title, version="v1")
    current = make_evidence("Current requested source", current_document, version="v1")
    snapshot = SessionSnapshot("s", archive=(old,))
    candidate = build_candidates(snapshot, query, (current,),
                                 EvidencePolicy())[0]
    assert candidate.required_blocks == candidate.visible_blocks == {current.key}
    assert dict(candidate.evidence_resolution) == {current.key: "current_retrieval"}


def test_unrelated_version_name_does_not_create_false_conflict_during_rebuild():
    old_api = make_evidence("OLD_API_SOURCE", "api", version="v1")
    new_api = make_evidence("CURRENT_API_SOURCE", "api", version="v2")
    rapid = make_evidence("RAPID_SOURCE", "rapid", version="v1")
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot, "Initial API question", (old_api,))[0]
    snapshot = committed(first, snapshot, "Initial API question", (old_api,))
    append, rebuild = build_candidates(snapshot, "What is the rapid v1 retry interval?",
                                       (new_api, rapid), EvidencePolicy())
    assert append.required_blocks == rebuild.required_blocks == {new_api.key, rapid.key}
    assert not append.feasible and append.reason == "invalid_visible_version"
    assert rebuild.feasible and rebuild.visible_blocks == {new_api.key, rapid.key}
    assert "OLD_API_SOURCE" not in "".join(message.content for message in rebuild.messages)
    assert rebuild.messages[1:3] == (
        Message("user", "Initial API question"), Message("assistant", "Complete original answer."))


@pytest.mark.parametrize("document_id,title,query", (
    ("api", "", "Explain API v1"),
    ("internal-doc", "接口说明", "解释接口说明 v1 的参数"),
    ("docs/client-api.md", "", "Explain docs/client-api.md v1"),
))
def test_explicit_document_version_revisit_preserves_literal_names(document_id, title, query):
    old = make_evidence("Archived requested source", document_id, title=title, version="v1")
    unrelated = make_evidence("Other retrieved source", "other", version="v2")
    candidate = build_candidates(SessionSnapshot("s", archive=(old,)), query, (unrelated,),
                                 EvidencePolicy(explicit_keys=(old.key,)))[0]
    assert candidate.feasible and candidate.required_blocks == {old.key, unrelated.key}
    assert dict(candidate.evidence_resolution)[old.key] == "explicit_key"


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


@pytest.mark.parametrize("with_execution", [False, True])
def test_template_reasoning_transform_cannot_silently_drop_raw_assistant_history(with_execution):
    old, new = make_evidence("Old original source", "a"), make_evidence("Current original source", "b")
    snapshot = SessionSnapshot("s")
    answer = "Original literal prefix</think>Final text with conditions."
    snapshot = committed(build_candidates(snapshot, "Prior question", (old,))[0], snapshot,
                         "Prior question", (old,), answer)
    if not with_execution:
        snapshot = replace(snapshot, execution=None)
    original = copy.deepcopy(snapshot)

    class ReasoningTemplate(ByteTemplate):
        render_calls = 0

        def render(self, messages):
            self.render_calls += 1
            # Model-template reasoning handling is valid in isolation, but the
            # resulting candidate cannot claim to retain this entire raw QA.
            transformed = tuple(replace(m, content=m.content.split("</think>")[-1])
                                if m.role == "assistant" else m for m in messages)
            return super().render(transformed)

    template = ReasoningTemplate()
    with pytest.raises(EvidenceError, match=r"original message .*\(assistant\)"):
        build_candidates(snapshot, "Current question", (new,), template=template)
    assert snapshot == original
    assert template.render_calls == (2 if with_execution else 1)  # no validation re-render


def test_rendered_candidate_requires_history_occurrences_in_order_but_allows_template_wrappers():
    from controller.templates import verify_rendered_evidence

    messages = (Message("system", "Policy"), Message("user", "First question"),
                Message("assistant", "Same answer"), Message("user", "Second question"),
                Message("assistant", "Same answer"), Message("user", "Current question"))
    rendered = "\n".join(f"<role={m.role}>\n{m.content}\n</role>" for m in messages)
    verify_rendered_evidence(messages, (), rendered+"\n<assistant><think>\n\n</think>\n")
    with pytest.raises(EvidenceError, match="template omitted or transformed"):
        verify_rendered_evidence(messages, (), rendered.replace("Same answer", "", 1))
    with pytest.raises(EvidenceError, match="template omitted or transformed"):
        verify_rendered_evidence(messages, (), rendered.replace("Second question", "changed question"))


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
    assert explicit.required_blocks == {current.key}  # Names do not grant historical citation rights.


def test_ambiguous_historical_anchor_requires_document_scope():
    a = make_evidence("timeout 10", "alpha", anchors=("connect_timeout",))
    b = make_evidence("timeout 30", "beta", anchors=("connect_timeout",))
    current = make_evidence("Available deployment regions", "regions")
    snapshot = SessionSnapshot("s", archive=(a, b))
    candidate = build_candidates(snapshot, "What is connect_timeout?", (current,))[-1]
    assert candidate.required_blocks == {current.key}


@pytest.mark.parametrize("anchor", ["--timeout", "request_limit", "cache_size"])
@pytest.mark.parametrize("relation", [None, "standalone", "followup"])
def test_shared_identifier_never_restores_an_unidentified_historical_object(anchor, relation):
    from controller.schema import QueryContext

    old = make_evidence(f"Atlas {anchor} is 10.", "Atlas", anchors=(anchor,))
    current = make_evidence("Beacon configuration reference.", "Beacon")
    snapshot = SessionSnapshot("s")
    snapshot = committed(build_candidates(snapshot, "Atlas settings?", (old,))[0],
                         snapshot, "Atlas settings?", (old,))
    query = f"What is Beacon {anchor}?"
    context = None if relation is None else QueryContext(query, relation, (1,))
    for candidate in build_candidates(snapshot, query, (current,), query_context=context):
        assert candidate.required_blocks == {current.key}


def test_anchor_respects_selected_revisit_turns_and_current_document_identity():
    from controller.schema import QueryContext

    atlas = make_evidence("Atlas --timeout is 10.", "Atlas", anchors=("--timeout",))
    beacon = make_evidence("Beacon --timeout is 30.", "Beacon", anchors=("--timeout",))
    snapshot = SessionSnapshot("s")
    for block in (atlas, beacon):
        query = f"Describe {block.key.document_id}."
        snapshot = committed(build_candidates(snapshot, query, (block,))[-1], snapshot, query, (block,))
    current = make_evidence("Current deployment information.", "regions")
    context = QueryContext("What is --timeout?", "revisit", (1,))
    for candidate in build_candidates(snapshot, "Return to the first setting", (current,), query_context=context):
        assert candidate.required_blocks == {atlas.key, current.key}
    header = make_evidence("Settings for this source.", "Atlas", passage_id="header")
    candidate = build_candidates(snapshot, "What is --timeout?", (header,))[-1]
    assert candidate.required_blocks == {header.key}
    revised_header = replace(header, key=replace(header.key, version="v2"))
    candidate = build_candidates(snapshot, "What is Atlas --timeout?", (revised_header,))[-1]
    assert candidate.required_blocks == {revised_header.key}


def test_anchor_respects_named_historical_version():
    old = make_evidence("timeout 10", "api", version="v1", anchors=("connect_timeout",))
    new = make_evidence("timeout 30", "api", version="v2", anchors=("connect_timeout",))
    current = make_evidence("Deployment regions", "regions")
    candidate = build_candidates(SessionSnapshot("s", archive=(old, new)),
                                 "What is api v2 connect_timeout?", (current,), EvidencePolicy(explicit_keys=(new.key,)))[-1]
    assert candidate.feasible and candidate.required_blocks == {current.key, new.key}


def test_selected_history_does_not_expand_anchors_to_other_turns_of_the_same_document():
    from controller.schema import QueryContext

    first = make_evidence("Timeout value: --timeout is 10.", "Atlas", passage_id="setting",
                          anchors=("--timeout",))
    second = make_evidence("Timeout policy overview.", "Atlas", passage_id="overview")
    snapshot = SessionSnapshot("s")
    for block in (first, second):
        snapshot = committed(build_candidates(snapshot, "Describe this source", (block,))[-1],
                             snapshot, "Describe this source", (block,))
    current = make_evidence("Deployment information.", "regions")
    intent = QueryContext("What is --timeout?", "revisit", (2,))
    for candidate in build_candidates(snapshot, "Return to that source", (current,), query_context=intent):
        assert candidate.required_blocks == {second.key, current.key}


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
                                      EvidencePolicy(layout=layout),
                                      query_context=QueryContext("回访限制", "revisit", (1,)))
        for candidate in candidates:
            assert condition.key in candidate.required_blocks <= candidate.visible_blocks
            assert condition.text in candidate.messages[-1].content
            assert dict(candidate.evidence_resolution)[condition.key].startswith("selected_revisit:")
    explicit_only = build_candidates(snap, "前面提到的限制是什么？", (current,),
                                    EvidencePolicy())[-1]
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
    candidate = build_candidates(snap, query, (new,),
        query_context=QueryContext(query, "revisit", (1,)))[-1]
    assert candidate.feasible and candidate.required_blocks == {old.key, new.key}
    assert dict(candidate.evidence_resolution)[old.key]== "selected_revisit:turn=1"


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


def test_compact_source_metadata_preserves_execution_and_original_spans():
    block = make_evidence("Complete original source.\nIncluding its conditions.", "document",
                          title="Original title", provenance="Recorded ingestion provenance")
    snapshot = SessionSnapshot("s")
    previous = build_candidates(snapshot, "Original", (block,))[0]
    snapshot = committed(previous, snapshot, "Original", (block,))
    append, rebuild = build_candidates(snapshot, "Continue", (block,))
    assert append.messages[:-1] == snapshot.execution.messages
    assert '"provenance"' not in rebuild.messages[-1].content
    assert '"source_revision":"v1"' in rebuild.messages[-1].content
    for candidate in (append, rebuild):
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
                                     (block,), candidate.reference_map) == {block.key}
