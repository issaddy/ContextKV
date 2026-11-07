from context_helpers import context_plan
from dataclasses import replace

from controller.candidates import build_plan_candidates
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceSet,
    EvidenceClaim, EvidenceUnit, Need, ReadingOutcome, SourceSpan, TaskState, TextRef)
from controller.evidence import make_evidence
from controller.schema import (SessionSnapshot, TurnRecord)
from controller.templates import ByteTemplate, verify_evidence_spans


def test_support_plateau_keeps_specific_source_and_complete_low_relevance_bridge():
    from adapters.study_retrieval import SupportCoverageUtility
    from controller.context_schema import EvidencePool, SupportBundle, SupportScores
    from controller.evidence_planner import BudgetedEvidencePlanner, SearchProfile, best_need_supports

    # Actual 000026.json error: both sources scored (1,1,complete), but the
    # shorter generic rule displaced the exact foreign-license exception.
    question = ("If I have a driver license from another country and I have a NY State learner permit, "
                "can I drive without a supervising driver?")
    state = TaskState("dmv", question, needs=(Need("n"),))
    generic = EvidenceUnit("generic", (), "A learner permit requires a supervising driver.", retrieval_score=4.40625)
    exact = EvidenceUnit("exact", (), question+" Yes. Your valid foreign driver license allows you to drive "
        "without a supervising driver, except in a DMV road test area.", retrieval_score=9.625)
    generic_support = SupportBundle("generic", "n", ("generic",), 1, 1, disposition="complete")
    exact_support = SupportBundle("exact", "n", ("exact",), 1, 1, disposition="complete")
    utility = SupportCoverageUtility()
    planner = BudgetedEvidencePlanner(ByteTemplate().tokenizer, utility,
        profile=SearchProfile(beam_width=1, max_actions=2),
        measure_bodies=lambda groups: tuple(sum(len(unit.text) for unit in units) for units in groups))
    selected = planner.search_sets(state, EvidencePool((generic, exact)),
        SupportScores((generic_support, exact_support)), ContextBudget())
    assert selected[0].quality_prediction == 1
    assert tuple(unit.unit_id for unit in selected[0].selected_units) == ("exact",)
    assert planner.trace[0]["frontier"] == [selected[0].set_id]

    # A bridge may have little standalone relevance to the original question.
    # The complete bundle retains it and its dependency; a high-score header
    # cannot inflate relevance, and the anchor alone is still only partial.
    header = EvidenceUnit("header", (), "License applicability", retrieval_score=100)
    bridge = EvidenceUnit("bridge", (), "The foreign license remains valid.",
        dependencies=("header",), retrieval_score=-7)
    partial = replace(exact_support, disposition="partial")
    complete = SupportBundle("complementary", "n", ("exact", "bridge"), 1, 1, disposition="complete")
    selected = planner.search_sets(state, EvidencePool((generic, exact, bridge, header)),
        SupportScores((generic_support, partial), (complete,)), ContextBudget())
    assert selected[0].quality_prediction == 1
    assert {unit.unit_id for unit in selected[0].selected_units} == {"exact", "bridge", "header"}
    assert selected[0].need_support == (complete,)
    assert best_need_supports(state, selected[0].selected_units, (complete,))[1] == 9.625


def test_numeric_support_score_does_not_turn_partial_contract_into_answer():
    from adapters.study_retrieval import SupportCoverageUtility
    from controller.context_schema import SupportBundle

    state = TaskState("task", "Which rule applies?", needs=(Need("n"),))
    unit = EvidenceUnit("u", (), "A partial prerequisite.", retrieval_score=10)
    support = SupportBundle("partial", "n", ("u",), 1, 1, disposition="partial")
    prediction = SupportCoverageUtility().predict_set(state, (unit,), (support,))
    assert prediction.quality == 1
    assert prediction.disposition == "partial"


def test_compiler_keeps_source_char_offsets_separate_from_prompt_offsets():
    from controller.context_budget import actual_source_coverage

    plan, block = context_plan()
    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    candidates = build_plan_candidates(snapshot, "How?", plan, template)
    assert len({c.candidate_id for c in candidates}) == 3
    for candidate in candidates:
        assert candidate.output_budget == 1024
        span = candidate.evidence_spans[0]
        assert span.source_start == 3 and span.source_end == 21
        assert candidate.messages[span.message_index].content[span.start:span.end] == block.text[3:21]
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
                                    (block,), candidate.reference_map) == {block.key}
        required = plan.selected_set.selected_units
        assert actual_source_coverage(candidate, required)
        assert actual_source_coverage(candidate, required[0].source_spans)
        assert not actual_source_coverage(replace(candidate, evidence_spans=()), required)
        middle = span.source_start+5
        left = replace(span, end=span.start+5, source_end=middle)
        right = replace(span, start=span.start+5, source_start=middle)
        assert actual_source_coverage(replace(candidate, evidence_spans=(left, right)), required)
        assert not actual_source_coverage(replace(candidate,
            evidence_spans=(left, replace(right, start=right.start+1, source_start=middle+1))), required)
        changed = replace(required[0].source_spans[0], source_key=replace(block.key, version="v2"))
        assert not actual_source_coverage(candidate, (changed,))


def test_rebuild_selects_original_history_spans_and_preserves_actual_qa():
    plan, _ = context_plan()
    history = TurnRecord(1, "Windows installation", "irrelevant "*10000)
    state = replace(plan.task_state, raw_question="Change to Linux", current_turn_id=2,
        working_history=(TextRef(1, "user", 8, 20),))
    snapshot = SessionSnapshot("s", raw_history=(history,))
    candidate = build_plan_candidates(snapshot, state.raw_question,
                                      replace(plan, task_state=state), ByteTemplate())[0]
    assert candidate.feasible
    assert candidate.messages[1].content.endswith(history.user_text[8:20])
    assert "irrelevant" not in "".join(m.content for m in candidate.messages)
    assert snapshot.raw_history == (history,)


def test_source_aliases_preserve_two_versions_of_identical_text():
    plan, block = context_plan()
    alias = make_evidence(block.text, "manual", version="v2")
    unit = plan.selected_set.selected_units[0]
    other = replace(unit, unit_id="u2", source_spans=(
        SourceSpan("span:u2", alias.key, 3, 21),))
    plan = replace(plan, selected_set=replace(plan.selected_set,
        selected_units=(unit, other), source_blocks=(block, alias)))
    candidate = build_plan_candidates(SessionSnapshot("s"), "Compare?", plan, ByteTemplate())[0]
    assert candidate.messages[-1].content.count(unit.text) == 1
    assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
                                (block, alias), candidate.reference_map) == {block.key, alias.key}


def test_support_teacher_uses_complete_span_selection_and_records_actual_request(tmp_path):
    import json
    import httpx
    from jsonschema import Draft202012Validator
    from adapters.support_model import SupportInput
    from adapters.teacher_support import TeacherSupportBackend
    from controller.schema import Message

    plan, _ = context_plan()
    first = replace(plan.selected_set.selected_units[0], retrieval_score=.440746)
    other = make_evidence('An independent supporting rule.', 'rules')
    second = EvidenceUnit('rule', (SourceSpan('rule:span', other.key, 0, len(other.text)),),
                          other.text, retrieval_score=.290210)
    scores = {'applicability': .9, 'direct_support': .7, 'joint_support': .8, 'bridge_value': .6,
              'disposition': 'partial', 'missing_relation': 'The applicable filing limit.'}
    output = {'predictions': {'0': {**scores, 'span_selection': {'s0': True}},
                             '1': {**scores, 'span_selection': {'s0': False, 's1': True}}}}
    observed = {}

    def respond(request):
        observed.update(json.loads(request.content))
        return httpx.Response(200, json={'text': json.dumps(output),
            'meta_info': {'completion_tokens': 51, 'finish_reason': {'type': 'stop'}}})

    template, path = ByteTemplate(), tmp_path/'support-calls.jsonl'
    backend = TeacherSupportBackend('http://support-fixture', template, record_path=path)
    backend.client.close()
    backend.client = httpx.Client(base_url='http://support-fixture', transport=httpx.MockTransport(respond))
    try:
        predictions = backend.predict((SupportInput('task', 'need', 'Which rule?', (first,)),
            SupportInput('task', 'need', 'Which rule?', (first, second))))
    finally:
        backend.close()
    assert [prediction.source_spans for prediction in predictions] == [first.source_spans, second.source_spans]
    assert predictions[1].missing_relation == scores['missing_relation']
    schema = json.loads(observed['sampling_params']['json_schema'])
    validator = Draft202012Validator(schema)
    validator.validate(output)
    for invalid in ({}, {'s0': True, 's1': True}, ['s0', 's0']):
        assert not validator.is_valid({'predictions': {**output['predictions'],
            '0': {**scores, 'span_selection': invalid}}})
    record = json.loads(path.read_text())
    payload = json.loads(record['messages'][-1]['content'])
    assert all('retrieval_score' not in unit for unit in payload['units'])
    assert record['response_schema'] == schema
    assert record['output_limit'] == observed['sampling_params']['max_new_tokens']
    assert list(template.render(tuple(Message(**row) for row in record['messages'])).input_ids) == observed['input_ids']
    assert backend.usage['calls'] == 1 and backend.usage['examples'] == 2 and backend.usage['output_tokens'] == 51


def test_compiled_instruction_and_output_reservations_use_real_tokens():
    plan, _ = context_plan()
    tiny = replace(plan, budget=ContextBudget(window_tokens=2048, output_tokens=1024))
    candidates = build_plan_candidates(SessionSnapshot("s"), "How?", tiny, ByteTemplate())
    assert all(not c.feasible and c.reason == "context_window_budget" for c in candidates)


def test_large_read_unit_keeps_budgeted_original_and_root_citation():
    from controller.claim_graph import ClaimGraph
    from controller.evidence_planner import render_selection
    from controller.templates import candidate_request

    prefix, fact = "背景。"*2000, "条目 A weighs 26 kg."
    block = make_evidence(prefix+fact+"\nBackground notes."*2000, "inventory")
    span = SourceSpan("inventory:whole", block.key, 0, len(block.text))
    unit = EvidenceUnit("inventory", (span,), block.text)
    background = make_evidence("General shipping background.", "background")
    background_unit = EvidenceUnit("background", (SourceSpan("background", background.key,
        0, len(background.text)),), background.text, retrieval_score=10)
    state = TaskState("task", "Total weight?", needs=(Need("weight", operation="sum"),))
    leaf = EvidenceClaim("read:0:claim:0", "weight", "extract", text=fact, value=26, unit="kg",
                         source_spans=(span,))
    root = EvidenceClaim("reduce:weight:claim:0", "weight", "derived", value=26,
                         unit="kg", operation="sum", parent_claim_ids=(leaf.claim_id,))
    graph = ClaimGraph((leaf, root))
    plan = ContextPlan("plan", state, EvidenceSet("set", (unit, background_unit),
        source_blocks=(block, background)),
        "all_supplied_documents", answer_plan=graph.answer_plan(state),
        budget=ContextBudget(body_tokens=700))
    outcome = ReadingOutcome(plan, claims=(leaf, root))
    template = ByteTemplate()
    assert len(template.tokenizer.encode(render_selection((unit,)))) > plan.budget.body_tokens
    for candidate in build_plan_candidates(SessionSnapshot("s"), state.raw_question, plan,
                                            template, outcome=outcome):
        assert candidate.required_blocks == {block.key}
        source = candidate.evidence_spans[0]
        assert len(prefix) == source.source_start < source.source_end < len(block.text)
        assert fact in candidate.messages[-1].content
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
                                    (block, background), candidate.reference_map) == {block.key}
        allowed = candidate_request(candidate, 1024)["citation_ids"]
        assert allowed and f'["{allowed[0]}",0,{len(block.text)}]' in candidate.messages[-1].content
        assert leaf.claim_id not in candidate.messages[-1].content
        assert candidate.visible_blocks == frozenset()


def test_derived_answer_excerpt_budget_is_shared_across_original_sources_before_background():
    from controller.claim_graph import ClaimGraph

    blocks = tuple(make_evidence(f"Item {index} weighs {index+1} kg. " + "Filing background. "*1000,
        f"register-{index}") for index in range(4))
    units = tuple(EvidenceUnit(f"u{index}", (SourceSpan(f"span:{index}", block.key, 0, len(block.text)),),
        block.text) for index, block in enumerate(blocks))
    leaves = tuple(EvidenceClaim(f"read:{index}:claim:0", "mass", "extract",
        text=f"Item {index} weighs {index+1} kg.", value=index+1, unit="kg", source_spans=unit.source_spans)
        for index, unit in enumerate(units))
    root = EvidenceClaim("reduce:mass:claim:0", "mass", "derived", value=10, unit="kg", operation="sum",
        parent_claim_ids=tuple(claim.claim_id for claim in leaves))
    state = TaskState("t", "What is their combined mass?", needs=(Need("mass", operation="sum"),))
    plan = ContextPlan("p", state, EvidenceSet("all", units, source_blocks=blocks), "all_supplied_documents",
        answer_plan=ClaimGraph((*leaves, root)).answer_plan(state), budget=ContextBudget(body_tokens=4200))
    for candidate in build_plan_candidates(SessionSnapshot("s"), state.raw_question, plan, ByteTemplate(),
            outcome=ReadingOutcome(plan, claims=(*leaves, root))):
        assert candidate.required_blocks == {block.key for block in blocks}
        assert all(claim.text in candidate.messages[-1].content for claim in leaves)
        assert sum(span.source_end-span.source_start for span in candidate.evidence_spans) < 4200
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans,
            blocks, candidate.reference_map) == candidate.required_blocks
