from dataclasses import replace

import pytest

from bench.evidence_diagnosis import (ApplicableScopeReview, OracleAnnotations, RequirementWitnesses,
                                     WitnessGroup, diagnose_evidence, review_binding)
from controller.context_schema import ContextPlan, EvidencePool, EvidenceSet, EvidenceUnit, SourceSpan, TaskState
from controller.evidence import make_evidence
from controller.evidence_planner import render_selection
from controller.schema import TurnRecord
from controller.templates import ByteTokenizer


def original_unit(name, block, start, end, *, dependencies=()):
    base = block.span[0] if block.span else 0
    return EvidenceUnit(name, (SourceSpan("span:"+name, block.key, base+start, base+end),),
        block.text[start:end], dependencies=dependencies, original_passage_ids=(block.key.passage_id,))


def audited_case(*, omit_exception=False):
    body = ("background "*70+"\nPolicy version 2024. Linux only.\n"
            "The applicant must hold a valid permit.\nSubmit form L.\n"
            "Exception: temporary permits do not qualify.\n"+"tail "*40)
    block = make_evidence(body, "policy", corpus_id="diagnostic", version="2024")
    texts = ("Policy version 2024. Linux only.", "The applicant must hold a valid permit.",
             "Submit form L.", "Exception: temporary permits do not qualify.")
    spans = tuple(SourceSpan(f"review:{index}", block.key, body.index(text), body.index(text)+len(text))
                  for index, text in enumerate(texts))
    units = tuple(original_unit(f"original:{index}", block, span.start, span.end,
        dependencies=("original:0",) if index in {1, 2, 3} else ()) for index, span in enumerate(spans))
    wrong = make_evidence(texts[-1], "policy-old", corpus_id="diagnostic", version="2019")
    wrong_unit = original_unit("old-exception", wrong, 0, len(wrong.text))
    pool = EvidencePool((units[0], units[1], units[2], wrong_unit) if omit_exception else units,
        (block, wrong) if omit_exception else (block,))
    e0 = ContextPlan("E0", TaskState("case", "For Linux under 2024, how do I apply?"),
        EvidenceSet("E0-selection", (units[2],), source_blocks=(block,)))
    history = (TurnRecord(1, "Use Windows", "Windows follows the earlier policy."),)
    requirements = (RequirementWitnesses("eligibility", (WitnessGroup("permit-and-exception",
        (spans[0], spans[1], spans[3]), "review.jsonl#eligibility"),)),
        RequirementWitnesses("procedure", (WitnessGroup("form-with-current-condition",
            (spans[0], spans[2]), "review.jsonl#procedure"),)))
    annotation = OracleAnnotations(**review_binding(e0, history), requirements=requirements,
                                   review_origin="specified-unit-test-review")
    scope = ApplicableScopeReview("applicable-2024-linux", (block.key,), (block.key,),
                                 ("review.jsonl",), "test-reviewer", True)
    return e0, pool, annotation, history, EvidencePool(units, (block,)), scope


def diagnose(case, budgets=(100000,)):
    e0, pool, annotation, history, full, scope = case
    return diagnose_evidence(e0, pool, annotation, actual_history=history,
        full_applicable_pool=full, scope_review=scope, body_budgets=budgets, tokenizer=ByteTokenizer())


def test_e1_does_not_substitute_wrong_versions_or_split_conjunctive_witnesses():
    case = audited_case(omit_exception=True)
    result = diagnose(case)
    assert result.e1[0].completed_requirements == ("procedure",)
    assert result.e1[0].omissions == (("eligibility", "witness_not_in_scope"),)
    assert set(result.e1[0].evidence_set.selected_units) <= set(case[1].units)
    assert result.e2[0].missing_requirements == ()
    assert all(unit.source_spans[0].source_key.version == "2024" for unit in result.e2[0].evidence_set.selected_units)
    # A single premise receives no partial credit for eligibility.
    assert "eligibility" not in result.e0[0].completed_requirements


def test_e1_keeps_original_dependency_units_and_e2_materializes_every_reviewed_context_span():
    case = audited_case()
    result = diagnose(case)
    assert result.e1[0].missing_requirements == result.e2[0].missing_requirements == ()
    e1 = result.e1[0].evidence_set
    assert tuple(e1.selected_units) == case[1].units
    e2 = result.e2[0].evidence_set
    block = case[4].source_blocks[0]
    for unit in e2.selected_units:
        span = unit.source_spans[0]
        assert unit.text == block.text[span.start:span.end]
    literal = "\n".join(unit.text for unit in e2.selected_units)
    assert "Linux only" in literal and "valid permit" in literal and "temporary permits do not qualify" in literal
    assert "background" not in literal and "tail tail" not in literal
    assert e2.source_blocks == (block,)
    assert result.provenance["quality_delta"] is None


def test_e2_clips_source_boundaries_while_e1_keeps_larger_candidate_units():
    e0, _, annotation, history, full, scope = audited_case()
    block = full.source_blocks[0]
    whole = original_unit("whole-original-passage", block, 0, len(block.text))
    pool = EvidencePool((whole,), (block,))
    result = diagnose((e0, pool, annotation, history, pool, scope))
    assert result.e1[0].evidence_set.selected_units == (whole,)
    assert result.e2[0].evidence_set.token_cost < result.e1[0].evidence_set.token_cost
    assert result.e2[0].missing_requirements == ()


def test_actual_token_budget_curve_keeps_only_complete_necessary_points():
    case = audited_case()
    full = diagnose(case)
    e2_cost = full.e2[0].evidence_set.token_cost
    curve = diagnose(case, budgets=(0, e2_cost-1, e2_cost))
    assert curve.e2[0].completed_requirements == ()
    assert len(curve.e2[1].completed_requirements) < 2
    assert curve.e2[-1].missing_requirements == ()
    assert curve.e2[-1].evidence_set.token_cost == e2_cost
    for point in (*curve.e1, *curve.e2):
        assert point.evidence_set.token_cost <= point.body_budget
        assert point.evidence_set.token_cost == len(ByteTokenizer().encode(
            render_selection(point.evidence_set.selected_units), add_special_tokens=False))


def test_alternative_witnesses_use_cheapest_complete_group_and_share_overlaps():
    block = make_evidence("条件甲乙丙丁戊己庚辛壬癸"+" long alternative"*40, "unicode", corpus_id="diagnostic")
    block = replace(block, span=(100, 100+len(block.text)))
    unit = original_unit("unicode-whole", block, 0, len(block.text))
    pool = EvidencePool((unit,), (block,))
    plan = ContextPlan("unicode", TaskState("unicode-task", "两项要求"), EvidenceSet("initial", ()))
    first = WitnessGroup("first-short", (SourceSpan("a", block.key, 100, 106),), "review#a")
    overlapping = WitnessGroup("second-overlap", (SourceSpan("b", block.key, 103, 110),), "review#b")
    long = WitnessGroup("first-long", (SourceSpan("long", block.key, 100, 100+len(block.text)),), "review#long")
    annotations = OracleAnnotations(**review_binding(plan, ()), requirements=(
        RequirementWitnesses("one", (first, long)), RequirementWitnesses("two", (overlapping,))), review_origin="review")
    review = ApplicableScopeReview("unicode", (block.key,), (block.key,), ("review",), "reviewer", True)
    result = diagnose_evidence(plan, pool, annotations, actual_history=(), full_applicable_pool=pool,
        scope_review=review, body_budgets=(100000,), tokenizer=ByteTokenizer())
    selected = result.e2[0].evidence_set.selected_units
    assert len(selected) == 1 and selected[0].text == block.text[:10]
    assert (selected[0].source_spans[0].start, selected[0].source_spans[0].end) == (100, 110)
    assert result.e2[0].completed_requirements == ("one", "two")


def test_review_bindings_and_full_scope_claim_are_explicit():
    case = audited_case()
    e0, pool, annotation, history, full, scope = case
    with pytest.raises(ValueError, match="different actual-history"):
        diagnose((e0, pool, annotation, (), full, scope))
    with pytest.raises(ValueError, match="every supplied source"):
        diagnose((e0, pool, annotation, history, full, replace(scope, reviewed_source_keys=())))
    partial = diagnose((e0, pool, annotation, history, full, replace(scope, complete=False)))
    assert partial.provenance["full_applicable_scope_reviewed"] is False
    with pytest.raises(ValueError, match="explicit review records"):
        diagnose((e0, pool, annotation, history, full, replace(scope, review_records=())))
