"""Offline E0/E1/E2 witness-group evidence diagnosis on frozen actual history.

Witnesses are reviewed conjunctions of original SourceSpans.  E1 retains the
candidate pool's units and dependencies; E2 materializes literal reviewed span
boundaries.  No answer-quality label is inferred from witness coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from itertools import product

from controller.context_schema import (ContextPlan, EvidenceKey, EvidencePool, EvidenceSet,
    EvidenceUnit, SourceSpan)
from controller.evidence_planner import close_dependencies, render_selection
from controller.serialization import to_jsonable


def _digest(value):
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def review_binding(plan, actual_history):
    return {"task_id": plan.task_state.task_id, "task_state_digest": _digest(plan.task_state),
            "actual_history_digest": _digest(actual_history)}


@dataclass(frozen=True)
class WitnessGroup:
    witness_id: str
    spans: tuple[SourceSpan, ...]
    review_record: str


@dataclass(frozen=True)
class RequirementWitnesses:
    requirement_id: str
    alternatives: tuple[WitnessGroup, ...]


@dataclass(frozen=True)
class OracleAnnotations:
    task_id: str
    task_state_digest: str
    actual_history_digest: str
    requirements: tuple[RequirementWitnesses, ...]
    review_origin: str


@dataclass(frozen=True)
class ApplicableScopeReview:
    scope_id: str
    applicable_source_keys: tuple[EvidenceKey, ...]
    reviewed_source_keys: tuple[EvidenceKey, ...]
    review_records: tuple[str, ...]
    reviewer: str
    complete: bool


@dataclass(frozen=True)
class OraclePoint:
    body_budget: int
    evidence_set: EvidenceSet
    completed_requirements: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    witnesses: tuple[tuple[str, tuple[str, ...]], ...]
    omissions: tuple[tuple[str, str], ...]
    budget_feasible: bool


@dataclass(frozen=True)
class EvidenceDiagnosis:
    e0: tuple[OraclePoint, ...]
    e1: tuple[OraclePoint, ...]
    e2: tuple[OraclePoint, ...]
    provenance: dict


def _covers_span(units, span):
    intervals = sorted((source.start, source.end) for unit in units for source in unit.source_spans
                       if source.source_key == span.source_key)
    cursor = span.start
    for left, right in intervals:
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= span.end:
            return True
    return False


def _witnesses(units, requirements):
    return {requirement.requirement_id: tuple(witness.witness_id for witness in requirement.alternatives
        if all(_covers_span(units, span) for span in witness.spans)) for requirement in requirements}


def _minimal_unit_sets(sets):
    ordered = sorted(set(sets), key=lambda ids: (len(ids), tuple(sorted(ids))))
    return tuple(ids for ids in ordered if not any(other < ids for other in ordered))


def _span_unit_options(span, pool):
    holders = [(unit.unit_id, source.start, source.end) for unit in pool.units for source in unit.source_spans
               if source.source_key == span.source_key and source.start < span.end and span.start < source.end]
    found = set()
    def extend(cursor, selected):
        if cursor >= span.end:
            found.add(frozenset(selected))
            return
        for unit_id, left, right in holders:
            if left <= cursor < right:
                extend(right, selected | {unit_id})
    extend(span.start, set())
    return _minimal_unit_sets(found)


def _pool_witness_options(requirements, pool):
    by_id = {unit.unit_id: unit for unit in pool.units}
    options = {}
    for requirement in requirements:
        choices = []
        for witness in requirement.alternatives:
            alternatives = [_span_unit_options(span, pool) for span in witness.spans]
            for combination in product(*alternatives):
                ids = set().union(*combination)
                closed = close_dependencies(sorted(ids), by_id)
                choices.append(frozenset(unit.unit_id for unit in closed))
        options[requirement.requirement_id] = _minimal_unit_sets(choices)
    return options


def _union_candidates(options):
    states = {frozenset()}
    for choices in options.values():
        states |= {state | choice for state in states for choice in choices}
    return states


def _point(budget, selected, requirements, possible, *, selection_reason="body_budget"):
    witnesses = _witnesses(selected.selected_units, requirements)
    complete = tuple(requirement.requirement_id for requirement in requirements
                     if witnesses[requirement.requirement_id])
    missing = tuple(requirement.requirement_id for requirement in requirements
                    if not witnesses[requirement.requirement_id])
    omissions = tuple((identity, "no_reviewed_witness" if not next(
        requirement.alternatives for requirement in requirements if requirement.requirement_id == identity)
        else "witness_not_in_scope" if not possible[identity] else selection_reason) for identity in missing)
    return OraclePoint(budget, selected, complete, missing,
        tuple((identity, witnesses[identity]) for identity in complete), omissions, selected.token_cost <= budget)


def _selection(name, units, pool, tokenizer):
    keys = {span.source_key for unit in units for span in unit.source_spans}
    return EvidenceSet(name, tuple(units), token_cost=len(tokenizer.encode(
        render_selection(tuple(units)), add_special_tokens=False)),
        source_blocks=tuple(block for block in pool.source_blocks if block.key in keys))


def _curve(name, budgets, selections, requirements, possible):
    ranked = [(selected, _witnesses(selected.selected_units, requirements)) for selected in selections]
    curve = []
    for budget in budgets:
        selected, _ = min(((selected, witnesses) for selected, witnesses in ranked if selected.token_cost <= budget),
            key=lambda item: (-sum(bool(value) for value in item[1].values()), item[0].token_cost,
                              tuple(unit.unit_id for unit in item[0].selected_units)))
        curve.append(_point(budget, replace(selected, set_id=f"{name}:{budget}"), requirements, possible))
    return tuple(curve)


def _span_key(span):
    return (span.source_key, span.start, span.end)


def _literal_units(span_keys, pool, tokenizer):
    blocks = {block.key: block for block in pool.source_blocks}
    grouped = {}
    for key, left, right in span_keys:
        grouped.setdefault(key, []).append((left, right))
    result = []
    for key in sorted(grouped):
        ranges = []
        for left, right in sorted(grouped[key]):
            if ranges and left <= ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], right))
            else:
                ranges.append((left, right))
        block = blocks[key]
        base = block.span[0] if block.span else 0
        for left, right in ranges:
            if not base <= left < right <= base+len(block.text):
                raise ValueError("reviewed witness coordinates exceed their original source body")
            unit_id = f"{key.corpus_id}:{key.document_id}:{key.passage_id}:{key.version}:{left}:{right}"
            source = SourceSpan("span:"+unit_id, key, left, right)
            parent = next((unit for unit in pool.units if any(source.source_key == key and
                source.start <= left < source.end for source in unit.source_spans)), None)
            text = block.text[left-base:right-base]
            # Witness annotations explicitly include prerequisites, conditions,
            # exceptions and version context. Only their literal spans enter E2.
            body_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            unit = (replace(parent, unit_id=unit_id, source_spans=(source,), text=text,
                heading=block.title, dependencies=(), body_tokens=body_tokens) if parent is not None else EvidenceUnit(
                unit_id, (source,), text, heading=block.title, body_tokens=body_tokens,
                original_passage_ids=(key.passage_id,)))
            result.append(unit)
    return tuple(result)


def diagnose_evidence(e0: ContextPlan, candidate_pool: EvidencePool, annotations: OracleAnnotations, *,
                      actual_history, full_applicable_pool: EvidencePool,
                      scope_review: ApplicableScopeReview, body_budgets, tokenizer):
    """Exact witness-union search; source applicability is independently reviewed."""
    binding = review_binding(e0, actual_history)
    if any(getattr(annotations, field) != value for field, value in binding.items()):
        raise ValueError("oracle witness review belongs to a different actual-history task state")
    full_keys = {block.key for block in full_applicable_pool.source_blocks}
    if full_keys != set(scope_review.applicable_source_keys) or not scope_review.review_records:
        raise ValueError("E2 requires the applicable source scope and its explicit review records")
    reviewed = set(scope_review.reviewed_source_keys)
    if not reviewed <= full_keys or scope_review.complete and reviewed != full_keys:
        raise ValueError("complete applicable-scope review must include every supplied source")
    requirements = annotations.requirements
    if len({requirement.requirement_id for requirement in requirements}) != len(requirements):
        raise ValueError("necessary answer points have unique requirement IDs")
    for requirement in requirements:
        for witness in requirement.alternatives:
            if not witness.spans or not witness.review_record or any(
                    span.source_key not in reviewed for span in witness.spans):
                raise ValueError("each witness is a reviewed nonempty conjunction in the audited source scope")
    budgets = tuple(sorted(set(body_budgets)))
    options1 = _pool_witness_options(requirements, candidate_pool)
    by_id = {unit.unit_id: unit for unit in candidate_pool.units}
    sets1 = [_selection("E1", tuple(by_id[identity] for identity in sorted(ids)), candidate_pool, tokenizer)
             for ids in _union_candidates(options1)]
    # Atom coordinates retain version and complete source identity. Adjacent or
    # overlapping selected intervals merge only inside the same original body.
    options2 = {requirement.requirement_id: tuple(frozenset(_span_key(span) for span in witness.spans)
        for witness in requirement.alternatives) for requirement in requirements}
    sets2 = [_selection("E2", _literal_units(ids, full_applicable_pool, tokenizer), full_applicable_pool, tokenizer)
             for ids in _union_candidates(options2)]
    possible0 = {requirement.requirement_id: bool(requirement.alternatives) for requirement in requirements}
    e0_set = _selection(e0.selected_set.set_id, e0.selected_set.selected_units,
        replace(candidate_pool, source_blocks=e0.selected_set.source_blocks), tokenizer)
    return EvidenceDiagnosis(tuple(_point(budget, e0_set, requirements, possible0,
        selection_reason="not_in_E0_selection") for budget in budgets),
        _curve("E1", budgets, sets1, requirements, options1),
        _curve("E2", budgets, sets2, requirements, options2),
        {"binding": binding, "usage": "offline_diagnosis_and_training_supervision_only",
         "review_origin": annotations.review_origin, "scope_review": to_jsonable(scope_review),
         "full_applicable_scope_reviewed": scope_review.complete,
         "minimum_scope": "exact_over_reviewed_witness_unions",
         "e1_granularity": "original_candidate_units_and_dependency_closure",
         "e2_granularity": "literal_reviewed_source_spans_with_same_source_interval_merging",
         "selection_objective": "maximum_complete_necessary_points_then_minimum_real_token_count",
         "quality_delta": None, "requirement_count": len(requirements),
         "candidate_states": {"E1": len(sets1), "E2": len(sets2)}})
