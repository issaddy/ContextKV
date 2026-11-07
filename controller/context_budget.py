"""Compile task-selected original history and account complete model windows."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

from .schema import Message
from .serialization import to_jsonable


class AnswerWindowExceeded(ValueError):
    """A proposal cannot reserve its actual terminal ANSWER envelope."""


def text_ref_sources(history, *, current_turn_id=None, current_user_text=None):
    sources = {(row.turn_id, speaker): text for row in history for speaker, text in (
        ("user", row.user_text), ("assistant", row.assistant_text))}
    if current_turn_id is not None:
        sources[current_turn_id, "user"] = current_user_text
    return sources


def text_ref_text(ref, sources, *, field="TextRef"):
    owner = ref.turn_id, ref.speaker
    if owner not in sources:
        raise ValueError(f"{field} names absent actual-QA message {owner}; available={tuple(sources)}")
    text = sources[owner]
    if not 0 <= ref.start < ref.end <= len(text):
        raise ValueError(f"{field} span [{ref.start}:{ref.end}] is outside actual-QA message {owner} length {len(text)}")
    return text[ref.start:ref.end]


def resolve_text(ref, state, snapshot):
    return text_ref_text(ref, text_ref_sources(snapshot.raw_history,
        current_turn_id=state.current_turn_id, current_user_text=state.raw_question))


def condition_references(condition):
    return tuple(ref for ref in (condition.key_span, *condition.value_spans, condition.predicate_span)
                 if ref is not None)


def resolved_condition(condition, resolve):
    return {"relation": condition.relation_type,
        "key": resolve(condition.key_span) if condition.key_span else "",
        "values": [resolve(ref) for ref in condition.value_spans],
        "predicate": resolve(condition.predicate_span) if condition.predicate_span else ""}


def resolved_task_view(state, history, *, need_ids=None):
    sources = text_ref_sources(history, current_turn_id=state.current_turn_id,
                               current_user_text=state.raw_question)
    needs = tuple(need for need in state.needs if need_ids is None or need.need_id in need_ids)
    active = {identity for need in needs for identity in need.condition_ids}
    conditions = [{"id": condition.condition_id,
                   **resolved_condition(condition, lambda ref: text_ref_text(ref, sources))}
                  for condition in state.conditions if condition.condition_id in active]
    version = to_jsonable(state.version_intent)
    if state.version_intent.time_refs:
        version["time_text"] = [text_ref_text(ref, sources) for ref in state.version_intent.time_refs]
    return {"task_id": state.task_id, "conditions": conditions, "needs": to_jsonable(needs),
            "version_intent": version}


def working_history_messages(state, snapshot):
    return tuple(Message(ref.speaker,
        f"[turn={ref.turn_id} speaker={ref.speaker} chars={ref.start}:{ref.end}]\n"
        + resolve_text(ref, state, snapshot)) for ref in state.working_history)


def task_instruction(state, snapshot):
    return json.dumps(resolved_task_view(state, snapshot.raw_history), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class WindowCost:
    input_tokens: int
    output_tokens: int
    history_tokens: int
    source_tokens: int
    window_tokens: int

    @property
    def feasible(self):
        return self.input_tokens + self.output_tokens <= self.window_tokens


@dataclass(frozen=True)
class CompletionReserve:
    """Known current allocation and the minimum registered work after it."""
    action_input_tokens: int
    action_output_tokens: int
    future_input_tokens: int
    future_output_tokens: int
    future_steps: int

    @property
    def input_tokens(self):
        return self.action_input_tokens + self.future_input_tokens

    @property
    def output_tokens(self):
        return self.action_output_tokens + self.future_output_tokens

    @property
    def steps(self):
        return 1 + self.future_steps


def completion_reserve(plan, results, task, candidate=None, *, read_all=False):
    """Reserve the final envelope and already declared mandatory continuation.

    Only the current compiled input is exact. Future READs contribute source
    work and their output limits, before later facts and prompts exist.
    """
    from .reading import numeric_selection_step_id

    current_input = len(candidate.input_ids) if candidate is not None else 0
    current_output = candidate.output_budget if candidate is not None else 0
    if task is not None and task.kind == "ANSWER":
        return CompletionReserve(current_input, current_output, 0, 0, 0)

    full_scope = plan.coverage_scope == "all_supplied_documents"
    typed = plan.reading_mode == "typed_reduce"
    numeric = {need.need_id: need.operation for need in plan.task_state.needs
               if typed and need.operation in {"count", "sum", "group"}}
    exhaustive = set(numeric) | {need.need_id for need in plan.task_state.needs
                                 if need.scope == "all_supplied_documents"}
    completed = {result.step_id for result in results}
    reductions = tuple(result.reduction for result in results if result.reduction is not None)
    completed.update(item.step_id for item in reductions)
    current_id = task.step_id if task is not None else None
    outstanding = tuple(step for step in plan.steps if step.step_id not in completed
                        and step.step_id != current_id)
    reads = tuple(step for step in outstanding if step.kind == "READ"
                  and (read_all or full_scope or exhaustive.intersection(step.need_ids)))
    if full_scope and typed:
        numeric.update((need.need_id, "union") for need in plan.task_state.needs
                       if need.operation == "enumerate")
    required_ops = {*numeric.values(), "numeric_select"} if numeric else set()
    scheduled = tuple(step for step in outstanding if step.kind == "REDUCE"
                      and (read_all or step.operation in required_ops))
    scheduled_ops = {(need, step.operation) for step in (*scheduled, *((task,) if task is not None else ()))
                     if step.kind == "REDUCE" for need in step.need_ids}
    finished_ops = {(need, item.operation) for item in reductions for need in item.need_ids}
    selectors = sum(numeric_selection_step_id(plan, need) not in completed | {None}
                    and (need, "numeric_select") not in scheduled_ops for need in numeric)
    aggregates = sum((need, operation) not in scheduled_ops and
                     ((need, operation) not in finished_ops
                      or any(need in step.need_ids for step in reads)
                      or (task is not None and task.kind == "READ" and need in task.need_ids))
                     for need, operation in numeric.items())
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    body_tokens = sum(sum({units[key].text: units[key].body_tokens
                           for key in step.input_unit_ids}.values()) for step in reads)
    cpu_ops = {"union", "count", "sum", "group"} if typed else set()
    output_tokens = (sum(step.output_budget for step in (*reads, *scheduled)
                         if step.operation not in cpu_ops or step.kind == "READ")
                     + selectors*plan.budget.read_output_tokens)
    return CompletionReserve(current_input, current_output,
        plan.answer_allocation.fixed_input_tokens+body_tokens, plan.budget.output_tokens+output_tokens,
        1+len(reads)+len(scheduled)+selectors+aggregates)


def measure_window(template, messages, budget, rendered, *, history=(), source_text=""):
    encode = template.tokenizer.encode
    return WindowCost(len(rendered.input_ids), budget.output_tokens,
        sum(len(encode(m.content, add_special_tokens=False)) for m in history),
        len(encode(source_text, add_special_tokens=False)), budget.window_tokens)


def allocate_answer_window(plan, template, snapshot):
    """Reserve the actual final envelope before admitting reading output."""
    from .candidates import build_plan_candidates
    from .context_schema import (AnswerPlan, AnswerPoint, AnswerWindowAllocation,
                                 EvidenceClaim, ReadingOutcome)
    from .schema import SessionSnapshot

    empty = replace(plan, selected_set=replace(plan.selected_set, selected_units=()),
        coverage_scope="selected_sources", answer_plan=AnswerPlan(tuple(
            AnswerPoint(need.need_id) for need in plan.task_state.needs)))
    snapshot = snapshot or SessionSnapshot("allocation")
    candidate = build_plan_candidates(snapshot,
        plan.task_state.raw_question, empty, template, layouts=("existing",))[0]
    if candidate.reason == "working_history_budget":
        raise AnswerWindowExceeded("actual working history exceeds the ANSWER history allocation")
    fixed = len(candidate.input_ids)
    spans = tuple(dict.fromkeys(span for unit in plan.selected_set.selected_units for span in unit.source_spans))
    claims = tuple(EvidenceClaim(f"allocation:{index}", need.need_id, "extract",
        conditions=need.condition_ids, source_spans=spans)
        for index, need in enumerate(plan.task_state.needs) if spans)
    source_plan = replace(empty, answer_plan=AnswerPlan(tuple(AnswerPoint(need.need_id,
        (f"allocation:{index}",) if spans else (), spans)
        for index, need in enumerate(plan.task_state.needs))))
    source_candidate = build_plan_candidates(snapshot, plan.task_state.raw_question, source_plan,
        template, layouts=("existing",), outcome=ReadingOutcome(source_plan, claims=claims))[0]
    metadata = len(source_candidate.input_ids)-fixed
    remaining = plan.budget.window_tokens-plan.budget.output_tokens-fixed-metadata
    if remaining <= 0:
        raise AnswerWindowExceeded("system, task, history, complete source metadata and ANSWER output exhaust the conclusion window")
    count = len(plan.task_state.needs)
    shares = tuple((need.need_id, remaining//count+(index < remaining%count))
                   for index, need in enumerate(plan.task_state.needs))
    return AnswerWindowAllocation(fixed, remaining, plan.budget.read_output_tokens, shares, metadata)


def split_conclusion_tokens(shares, count):
    """Partition a retained target; merging partitions adds these same shares."""
    return tuple(tuple((need, tokens//count+(index < tokens%count)) for need, tokens in shares)
                 for index in range(count))


def merge_conclusion_tokens(partitions):
    totals = {}
    for shares in partitions:
        for need, tokens in shares:
            totals[need] = totals.get(need, 0)+tokens
    return tuple(totals.items())


def bind_answer_window(plan, template, snapshot, system_prompt=None, *, preserve_step_ids=()):
    """Bind one final envelope and partition retained targets without repacking sources."""
    system_prompt = plan.answer_system_prompt if system_prompt is None else system_prompt
    bound = replace(plan, answer_system_prompt=system_prompt)
    allocation = allocate_answer_window(bound, template, snapshot)
    if (plan.answer_allocation == allocation and plan.answer_system_prompt == system_prompt
            and all(task.retained_conclusion_tokens for task in plan.steps if task.kind == "READ")):
        return plan
    plan = bound
    reads = tuple(task for task in plan.steps if task.kind == "READ")
    preserved = set(preserve_step_ids)
    assigned = {task.step_id: list(task.retained_conclusion_tokens) if task.step_id in preserved else []
                for task in reads}
    for need, tokens in allocation.need_tokens:
        spent = sum(dict(task.retained_conclusion_tokens).get(need, 0)
                    for task in reads if task.step_id in preserved)
        owners = tuple(task for task in reads if need in task.need_ids and task.step_id not in preserved)
        tokens = max(0, tokens-spent)
        for task, share in zip(owners, split_conclusion_tokens(((need, tokens),), len(owners))):
            assigned[task.step_id].extend(share)
    reducible = {need.need_id for need in plan.task_state.needs
                 if need.operation in {"enumerate", "compare", "count", "sum", "group", "infer"}}
    steps = []
    for task in plan.steps:
        if task.step_id not in preserved:
            if task.kind == "READ":
                task = replace(task, retained_conclusion_tokens=tuple(assigned[task.step_id]),
                    requires_reduction=task.operation == "summarize" or bool(reducible.intersection(task.need_ids)))
            elif task.kind == "REDUCE":
                task = replace(task, retained_conclusion_tokens=tuple((need, tokens)
                    for need, tokens in allocation.need_tokens if need in task.need_ids))
        steps.append(task)
    return replace(plan, answer_allocation=allocation, steps=tuple(steps))


def reduction_output_budget(plan, need_id):
    """Response space includes lineage; final content targets are stored on each task."""
    return plan.budget.read_output_tokens


def _covers(interval, ranges):
    start, end = interval
    for left, right in sorted(ranges):
        if left > start:
            break
        if right > start:
            start = right
        if start >= end:
            return True
    return start >= end


def actual_source_coverage(candidate, required):
    """Whether actual visible original intervals cover every SourceSpan/unit.

    This is source geometry, not task sufficiency or completed computation.
    Source references and visible-block flags alone do not establish coverage.
    """
    from .context_schema import EvidenceUnit

    required_spans = tuple(span for item in required for span in (
        item.source_spans if isinstance(item, EvidenceUnit) else (item,)))
    ranges = {}
    for span in candidate.evidence_spans:
        if span.source_start is not None and span.source_end is not None:
            ranges.setdefault(span.key, []).append((span.source_start, span.source_end))
    return all(_covers((span.start, span.end), ranges.get(span.source_key, ())) for span in required_spans)


def completed_source_ranges(plan, results):
    """Project completed extraction onto original source intervals."""
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    ranges = {}
    for result in results:
        for identity in result.covered_units:
            for span in units[identity].source_spans:
                ranges.setdefault(span.source_key, []).append((span.start, span.end))
    return ranges


def completed_source_coverage(plan, results, required):
    """Check completed extraction intervals independently of unit aliases."""
    ranges = completed_source_ranges(plan, results)
    return all(_covers((span.start, span.end), ranges.get(span.source_key, ()))
               for unit in required for span in unit.source_spans)


def direct_answer_reason(state, candidate, *, require_complete_evidence=True):
    """Distinguish skipping READ from delivering an answer with evidence gaps.

    Skipping READ requires sufficient evidence for every Need. An ANSWER may
    report gaps in a selected-source scope. Explicit full-source scopes and
    statistical views keep their completion requirements in either mode.
    """
    return next(iter(answer_evidence_gaps(state, candidate,
        require_complete_evidence=require_complete_evidence).values()), None)


def answer_evidence_gaps(state, candidate=None, *, require_complete_evidence=True):
    """Return unresolved Needs for the facts alone or the actual ANSWER input."""
    from .reading import numeric_selection_step_id

    plan = state.plan
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    tasks = {task.step_id: task for task in plan.steps}
    points = {point.need_id: point for point in plan.answer_plan.points}
    results = tuple(state.results)
    raw = candidate is not None and candidate.representation == "raw"
    gaps = {}
    for need in plan.task_state.needs:
        full_scope = plan.coverage_scope == "all_supplied_documents" or need.scope == "all_supplied_documents"
        numeric = need.operation in {"count", "sum", "group"}
        if not require_complete_evidence and not full_scope and not numeric:
            continue
        required = tuple(units[identity] for identity in plan.coverage_units) if full_scope else tuple(
            unit for unit in units.values() if not unit.need_ids or need.need_id in unit.need_ids)
        read_results = tuple(result for result in results if result.step_id in tasks
            and tasks[result.step_id].kind == "READ" and need.need_id in tasks[result.step_id].need_ids)
        facts_complete = bool(required) and completed_source_coverage(plan, read_results, required)
        point = points.get(need.need_id)
        if numeric:
            if not facts_complete or raw:
                gaps[need.need_id] = "numeric_fact_scope_incomplete"
                continue
            selector_id = numeric_selection_step_id(plan, need.need_id)
            selection = next((result.reduction for result in reversed(results) if result.reduction is not None
                and result.reduction.step_id == selector_id), None) if selector_id is not None else None
            basis = (set(selection.basis_claim_ids) if selection is not None else
                {claim.claim_id for result in read_results for claim in result.claims
                 if claim.kind == "extract" and claim.serves(need.need_id)})
            if selector_id is not None and selection is None:
                gaps[need.need_id] = "numeric_selection_incomplete"
                continue
            if not any(result.reduction is not None and need.need_id in result.reduction.need_ids
                    and result.reduction.operation == need.operation
                    and set(result.reduction.basis_claim_ids) == basis
                    and point is not None and set(result.reduction.frontier_claim_ids) <= set(point.claim_ids)
                    for result in results):
                gaps[need.need_id] = "numeric_operation_incomplete"
            continue
        supported = any(need.need_id in result.support for result in read_results)
        reduced = any(result.reduction is not None and need.need_id in result.reduction.need_ids
                      and result.reduction.operation != "numeric_select" for result in results)
        conclusion_complete = (point is not None and bool(point.claim_ids) and (supported or reduced)
            and not any(gap.need_id == need.need_id for gap in state.open_needs)
            and (not full_scope or facts_complete))
        if conclusion_complete and not raw:
            continue
        support = next((item for item in plan.selected_set.need_support if item.need_id == need.need_id), None)
        source_sufficient = full_scope or (support.disposition == "complete" if support is not None
                                         else plan.selected_set.disposition == "answer")
        if (not source_sufficient or not required or candidate is None
                or not actual_source_coverage(candidate, required)):
            gaps[need.need_id] = "answer_evidence_incomplete"
    return gaps


def evidence_interval_covered(snapshot, source):
    archive = {block.key: block for block in snapshot.archive}
    ranges = []
    for span in snapshot.execution.evidence_spans:
        if span.key != source.source_key:
            continue
        block = archive[span.key]
        base = block.span[0] if block.span else 0
        ranges.append((base, base+len(block.text)) if span.source_start is None else
                      (span.source_start, span.source_end))
    return _covers((source.start, source.end), ranges)


def append_legal(snapshot, state, required_keys, source_spans):
    if snapshot.execution is None:
        return False, "initial_rebuild"
    if not snapshot.execution.visible_blocks <= required_keys:
        return False, "source_scope_shrank"
    previous = next((turn.query_context.task_state for turn in reversed(snapshot.raw_history)
                     if turn.query_context and turn.query_context.task_state), None)
    if previous is None:
        return False, "task_state_initialization"
    if previous.task_id != state.task_id:
        return False, "task_changed"
    if previous.conditions != state.conditions:
        return False, "conditions_changed"
    if previous.version_intent != state.version_intent:
        return False, "version_intent_changed"
    last_turn = snapshot.raw_history[-1]
    retained = [(ref.turn_id, ref.speaker, ref.start, ref.end) for ref in previous.working_history]
    retained.extend(((last_turn.turn_id, "user", 0, len(last_turn.user_text)),
                     (last_turn.turn_id, "assistant", 0, len(last_turn.assistant_text))))
    for turn_id, speaker, start, end in retained:
        ranges = [(ref.start, ref.end) for ref in state.working_history
                  if ref.turn_id == turn_id and ref.speaker == speaker]
        if not _covers((start, end), ranges):
            return False, "working_history_shrank"
    archive = {block.key: block for block in snapshot.archive}
    for span in snapshot.execution.evidence_spans:
        block = archive[span.key]
        base = block.span[0] if block.span else 0
        interval = ((base, base+len(block.text)) if span.source_start is None else
                    (span.source_start, span.source_end))
        if not _covers(interval, [(s.start, s.end) for s in source_spans if s.source_key == span.key]):
            return False, "source_span_scope_shrank"
    return True, ""
