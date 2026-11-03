from dataclasses import replace

from controller.candidates import build_plan_candidates
from controller.context_budget import append_legal
from controller.context_schema import (Condition, ContextBudget, ContextPlan, EvidenceSet,
    EvidenceClaim, EvidenceUnit, Need, ReadingOutcome, SourceSpan, TaskState, TextRef)
from controller.cost_model import CostModel
from controller.evidence import make_evidence
from controller.planner import PlannerKnobs, choose_plan
from controller.schema import (CacheQuote, CacheQuoteBatch, CostEstimate, ExecutionLedger,
    Message, QueryContext, SessionSnapshot, TurnRecord)
from controller.serialization import dumps, loads
from controller.templates import ByteTemplate, verify_evidence_spans


def context_plan(question="How?", *, set_id="set", quality=.7):
    block = make_evidence("前文。Linux instructions。后文", "manual")
    start, end = 3, 21
    unit = EvidenceUnit("u", (SourceSpan("span:u", block.key, start, end),),
                        block.text[start:end])
    state = TaskState("task", question, needs=(Need("need"),), current_turn_id=1)
    selected = EvidenceSet(set_id, (unit,), quality_prediction=quality, source_blocks=(block,))
    return ContextPlan("plan:"+set_id, state, selected), block


def quotes(candidates):
    return CacheQuoteBatch("quotes", 1, 1, "restart", "cache", 100000, 100000,
        tuple(CacheQuote(c.candidate_id, c.input_hash, len(c.input_ids), len(c.input_ids)-1)
              for c in candidates), fingerprint=candidates[0].fingerprint)


class CandidateCosts(CostModel):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def predict_native(self, candidate, quote, *args, **kwargs):
        return CostEstimate(candidate.candidate_id, self.values[candidate.evidence_set_id],
                            calibrated=True)


def test_semantic_plan_round_trip_preserves_typed_original_blocks():
    plan, _ = context_plan()
    assert loads(ContextPlan, dumps(plan)) == plan


def test_compiler_keeps_source_char_offsets_separate_from_prompt_offsets():
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


def test_fixed_action_compares_all_sets_with_unique_ids():
    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    plans = (context_plan(set_id="low", quality=.3)[0],
             context_plan(set_id="best", quality=.9)[0],
             context_plan(set_id="middle", quality=.6)[0])
    candidates = tuple(c for p in plans for c in build_plan_candidates(snapshot, "How?", p, template))
    selected = choose_plan(candidates, None, CostModel(), snapshot,
                           PlannerKnobs(strategy="always_rebuild"))
    assert len({c.candidate_id for c in candidates}) == 9
    assert selected.candidate.evidence_set_id == "best"


def test_multiple_rebuild_sets_compete_on_quality_and_complete_cost():
    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    plans = (context_plan(set_id="fast", quality=.6)[0],
             context_plan(set_id="best", quality=.9)[0])
    candidates = tuple(c for p in plans for c in build_plan_candidates(snapshot, "How?", p, template))
    model = CandidateCosts({"fast": 100, "best": 200})
    selected = choose_plan(candidates, quotes(candidates), model, snapshot,
                           PlannerKnobs(lambda_time=.001))
    assert selected.candidate.evidence_set_id == "best"
    assert len(selected.predicted) == 6
    selected = choose_plan(candidates, quotes(candidates), model, snapshot,
                           PlannerKnobs(lambda_time=.01))
    assert selected.candidate.evidence_set_id == "fast"


def test_append_requires_task_condition_version_and_scope_continuation():
    plan, block = context_plan()
    state = plan.task_state
    turn = TurnRecord(1, state.raw_question, "answer", query_context=QueryContext(
        state.raw_question, "task_state", task_state=state))
    template = ByteTemplate()
    ledger = ExecutionLedger(1, "epoch", 1, (Message("system", "system"),), (), (),
        frozenset((block.key,)), template.fingerprint.template_hash,
        template.fingerprint.model_revision, "demo-v1")
    snapshot = SessionSnapshot("s", raw_history=(turn,), execution=ledger)
    state = replace(state, current_turn_id=2, working_history=(
        TextRef(1, "user", 0, len(turn.user_text)), TextRef(1, "assistant", 0, len(turn.assistant_text))))
    spans = plan.selected_set.selected_units[0].source_spans
    assert append_legal(snapshot, state, frozenset((block.key,)), spans) == (True, "")
    corrected = replace(state, conditions=(Condition("platform", relation_type="PLATFORM"),))
    assert append_legal(snapshot, corrected, frozenset((block.key,)), spans)[1] == "conditions_changed"
    assert append_legal(snapshot, state, frozenset(), spans)[1] == "source_scope_shrank"


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


def test_explicit_empty_answer_system_is_preserved():
    from controller.candidates import CandidatePolicy
    plan, _ = context_plan()
    plan = replace(plan, answer_system_prompt="A different task policy")
    candidate = build_plan_candidates(SessionSnapshot("s"), "How?", plan, ByteTemplate(),
        policy=CandidatePolicy(system_prompt=""))[0]
    assert candidate.messages[0].content == ""


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
