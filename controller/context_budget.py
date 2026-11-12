"""Compile task-selected original history and account complete model windows."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

from .schema import Message, history_messages
from .serialization import to_jsonable


COMPLETION_RESERVE_SCOPE = "known_inputs_and_allocated_future_outputs_v1"


class AnswerWindowExceeded(ValueError):
    """A proposal cannot reserve its actual terminal ANSWER envelope."""


def text_ref_sources(history, *, current_turn_id=None, current_user_text=None):
    sources = {(row.turn_id, message.role): message.content for row in history
               for message in history_messages((row,))}
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


def completion_reserve(plan, results, task, candidate=None, *, read_all=False,
                       template=None, snapshot=None, claims=(), read_inputs=None):
    """Reserve the final envelope and already declared mandatory continuation.

    Known REDUCE and final fact inputs use the reader's actual compiler. Missing
    future facts use declared output/conclusion allocations. Without template
    and snapshot this returns structural work only, not a complete online budget.
    """
    from .reading import numeric_selection_step_ids

    current_input = len(candidate.input_ids) if candidate is not None else 0
    current_output = candidate.output_budget if candidate is not None else 0
    if task is not None and task.kind == "ANSWER":
        return CompletionReserve(current_input, current_output, 0, 0, 0)

    full_scope = plan.coverage_scope == "all_supplied_documents"
    typed = plan.reading_mode == "typed_reduce"
    terminal = {gap.need_id for gap in plan.terminal_gaps}
    numeric = {need.need_id: need.operation for need in plan.task_state.needs
               if typed and need.operation in {"count", "sum", "group"} and need.need_id not in terminal}
    exhaustive = set(numeric) | {need.need_id for need in plan.task_state.needs
                                 if need.scope == "all_supplied_documents" and need.need_id not in terminal}
    completed = {result.step_id for result in results}
    reductions = tuple(result.reduction for result in results if result.reduction is not None)
    completed.update(item.step_id for item in reductions)
    current_id = task.step_id if task is not None else None
    outstanding = tuple(step for step in plan.steps if step.step_id not in completed
                        and step.step_id != current_id)
    reads = tuple(step for step in outstanding if step.kind == "READ" and set(step.need_ids)-terminal
                  and (read_all or full_scope or exhaustive.intersection(step.need_ids)))
    if full_scope and typed:
        numeric.update((need.need_id, "union") for need in plan.task_state.needs
                       if need.operation == "enumerate" and need.need_id not in terminal)
    required_ops = {*numeric.values(), "numeric_select"} if numeric else set()
    scheduled = tuple(step for step in outstanding if step.kind == "REDUCE" and set(step.need_ids)-terminal
                      and (read_all or step.operation in required_ops))
    scheduled_ops = {(need, step.operation) for step in (*scheduled, *((task,) if task is not None else ()))
                     if step.kind == "REDUCE" for need in step.need_ids}
    finished_ops = {(need, item.operation) for item in reductions for need in item.need_ids}
    selector_ids = numeric_selection_step_ids(plan)
    pending_selectors = {selector_ids[need] for need in numeric.keys() & selector_ids.keys()
                        if (need, "numeric_select") not in scheduled_ops} - completed
    selectors = len(pending_selectors)
    aggregates = sum((need, operation) not in scheduled_ops and
                     ((need, operation) not in finished_ops
                      or any(need in step.need_ids for step in reads)
                      or (task is not None and task.kind == "READ" and need in task.need_ids))
                     for need, operation in numeric.items())
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    read_inputs = read_inputs or {}
    read_tokens = sum(len(read_inputs[step.step_id].input_ids) if step.step_id in read_inputs else
        sum({units[key].text: units[key].body_tokens for key in step.input_unit_ids}.values()) for step in reads)
    cpu_ops = {"union", "count", "sum", "group"} if typed else set()
    output_tokens = (sum(read_inputs[step.step_id].output_budget if step.step_id in read_inputs
                        else step.output_budget for step in (*reads, *scheduled)
                         if step.operation not in cpu_ops or step.kind == "READ")
                     + selectors*plan.budget.read_output_tokens)
    future_inputs = plan.answer_allocation.fixed_input_tokens+read_tokens
    if template is not None and snapshot is not None:
        future_inputs = _completion_content_inputs(plan, results, task, reads, scheduled,
            pending_selectors, current_output, template, snapshot, claims, cpu_ops, read_inputs)
    return CompletionReserve(current_input, current_output,
        future_inputs, plan.budget.output_tokens+output_tokens,
        1+len(reads)+len(scheduled)+selectors+aggregates)


def _completion_content_inputs(plan, results, current, reads, scheduled, pending_selectors,
                               current_output, template, snapshot, claims, cpu_ops, read_inputs):
    """Compile known content and project each unfinished packet through its DAG.

    A READ packet can be consumed once by a reducer and again by ANSWER: these
    are separate model inputs. Unknown descendants share their ancestors' final
    Need allocation; that allocation is not a claim that the model will consume
    every parent. Known parents remain until actual lineage or a deterministic
    computation establishes the new frontier. Numeric selection retains facts.
    """
    from .candidates import build_plan_candidates
    from .action_proposals import bind_read_parents
    from .claim_graph import ClaimGraph, claim_frontier
    from .context_schema import ReadingOutcome, ReadResult, ReductionState
    from .execution import aggregate_numeric_view, reduction_basis_ids
    from .reading import numeric_selection_tasks, step_messages, unresolved_needs

    known = {claim.claim_id: claim for result in results for claim in result.claims}
    known.update((claim.claim_id, claim) for claim in claims)
    reductions = (*scheduled, *(task for task in numeric_selection_tasks(
        plan, results, tuple(known.values()), ready_only=False) if task.step_id in pending_selectors))
    steps = {task.step_id: task for task in (*plan.steps, *reductions)}
    if current is not None:
        steps[current.step_id] = current
    generated = tuple(task for task in (*reads, *reductions, *((current,) if current is not None else ()))
        if task.kind == "READ" or (task.kind == "REDUCE"
            and task.operation not in cpu_ops | {"numeric_select"}))
    capacities = {task.step_id: (current_output if task is current else
        read_inputs[task.step_id].output_budget if task.step_id in read_inputs else task.output_budget)
        for task in generated}

    def ancestors(task, *, all_producers=False):
        pending, found, visited = list(task.predecessor_step_ids), set(), set()
        while pending:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            if identity in capacities:
                found.add(identity)
                if not all_producers:
                    continue
            if identity in steps:
                pending.extend(steps[identity].predecessor_step_ids)
        return found

    projected_results = list(results)
    cpu_tasks = [task for task in (*((current,) if current is not None else ()), *scheduled)
                 if task.kind == "REDUCE" and task.operation in cpu_ops]
    while cpu_tasks:
        completed = {result.step_id for result in projected_results}
        completed.update(result.reduction.step_id for result in projected_results if result.reduction is not None)
        ready = tuple(task for task in cpu_tasks if set(task.predecessor_step_ids) <= completed
                      and set(task.input_claim_ids) <= known.keys())
        if not ready:
            break
        for task in ready:
            parents = tuple(known[identity] for identity in task.input_claim_ids)
            conclusion = aggregate_numeric_view(task, parents, plan, projected_results)
            known[conclusion.claim_id] = conclusion
            frontier = claim_frontier(parents, (conclusion,))
            reduction = ReductionState(task.step_id, task.need_ids, task.operation,
                reduction_basis_ids(task, parents, projected_results),
                tuple(claim.claim_id for claim in frontier), "merged")
            projected_results.append(ReadResult(task.step_id, claims=(conclusion,), reduction=reduction))
            cpu_tasks.remove(task)
    facts = tuple(known.values())
    graph = ClaimGraph(facts)
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    blocks = {block.key: block for block in (*snapshot.archive, *plan.selected_set.source_blocks)}
    observed = ReadingOutcome(plan, results=tuple(projected_results), claims=facts,
                              open_needs=unresolved_needs(projected_results))
    read_tokens, pending_messages = 0, []
    for task in reads:
        if task.step_id in read_inputs:
            read_tokens += len(read_inputs[task.step_id].input_ids)
            continue
        task = bind_read_parents(task, observed)
        parents = tuple(known[identity] for identity in task.input_claim_ids if identity in known)
        visible = replace(task, input_claim_ids=tuple(claim.claim_id for claim in parents), output_schema={})
        messages = step_messages(plan, visible, tuple(units[identity] for identity in task.input_unit_ids),
                                 parents, snapshot)
        pending_messages.append(messages)
        read_tokens += sum(capacities[identity] for identity in ancestors(task))
    reduction_inputs = 0
    for task in reductions:
        if task.operation in cpu_ops:
            continue
        parents = tuple(known[identity] for identity in task.input_claim_ids if identity in known)
        visible = replace(task, input_claim_ids=tuple(claim.claim_id for claim in parents), output_schema={})
        messages = step_messages(plan, visible, (), parents, snapshot)
        pending_inputs = ancestors(task)
        if task.operation == "numeric_select":
            # Selection reads original excerpts as well as normalized facts.
            # These excerpts already exist even when their future claims do not.
            payload = json.loads(messages[1].content)
            spans = {(span.source_key, span.start, span.end) for claim in parents for span in claim.source_spans}
            for identity in steps:
                if identity not in pending_inputs:
                    continue
                source = steps[identity]
                for unit_id in source.input_unit_ids:
                    for span in units[unit_id].source_spans:
                        key = (span.source_key, span.start, span.end)
                        if key in spans:
                            continue
                        block = blocks[span.source_key]
                        base = block.span[0] if block.span else 0
                        payload["sources"][f"s{len(spans)}"] = block.text[span.start-base:span.end-base]
                        spans.add(key)
            messages = (messages[0], Message("user", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
        pending_messages.append(messages)
        reduction_inputs += sum(capacities[identity] for identity in pending_inputs)

    # Divide a shared packet's one output allowance across its Need allocations;
    # multiple views of the same extraction do not become multiple final bodies.
    terminal = {}
    for task in generated:
        targets = (task.retained_conclusion_tokens or tuple((need, 1) for need in task.need_ids))
        total = sum(value for _, value in targets)
        available = min(capacities[task.step_id], total) if task.retained_conclusion_tokens else capacities[task.step_id]
        remaining = available
        shares = {}
        for index, (need, target) in enumerate(targets):
            share = remaining if index == len(targets)-1 else available*target//total if total else 0
            shares[need] = share
            remaining -= share
        terminal[task.step_id] = shares
    for task in sorted(generated, key=lambda task: len(ancestors(task, all_producers=True))):
        if task.kind != "REDUCE":
            continue
        for need in task.need_ids:
            inherited = sum(terminal[identity].pop(need, 0) for identity in ancestors(task, all_producers=True))
            terminal[task.step_id][need] = max(terminal[task.step_id].get(need, 0), inherited)

    answer = graph.answer_plan(plan.task_state, plan.answer_plan.disposition, projected_results)

    def answer_size(answer_plan):
        bound = replace(plan, answer_plan=answer_plan)
        outcome = ReadingOutcome(bound, results=tuple(projected_results), claims=facts,
                                 open_needs=unresolved_needs(projected_results))
        candidate, = build_plan_candidates(snapshot, plan.task_state.raw_question, bound, template,
            outcome=outcome, representation="facts", layouts=("existing",), actions=("REBUILD",))
        return len(candidate.input_ids)

    final_input = answer_size(answer)
    empty_input = answer_size(replace(answer, points=tuple(replace(point, claim_ids=(), source_spans=())
                                                         for point in answer.points)))
    future_content = sum(sum(shares.values()) for shares in terminal.values())
    available = max(0, plan.answer_allocation.conclusion_tokens-max(0, final_input-empty_input))
    known_sources = {(span.source_key, span.start, span.end) for point in answer.points for span in point.source_spans}
    new_sources = any((span.source_key, span.start, span.end) not in known_sources
        for task in generated if task.kind == "READ" for identity in task.input_unit_ids
        for span in units[identity].source_spans)
    # The one existing source-metadata allocation also covers provenance that
    # has not appeared in a fact yet. It is shared by the entire final frontier.
    metadata = plan.answer_allocation.source_metadata_tokens if new_sources else 0
    final_room = max(0, plan.budget.window_tokens-plan.budget.output_tokens-final_input)
    length_reader = getattr(template, "token_lengths", None)
    compiled_inputs = (sum(length_reader(tuple(pending_messages))) if length_reader is not None else
        sum(len(prompt.input_ids) for prompt in template.render_many(tuple(pending_messages))))
    return read_tokens+reduction_inputs+compiled_inputs+final_input+min(min(future_content, available)+metadata, final_room)


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


def required_source_units(plan, need):
    full_scope = plan.coverage_scope == "all_supplied_documents" or need.scope == "all_supplied_documents"
    coverage = set(plan.coverage_units) if full_scope else None
    return tuple(unit for unit in plan.selected_set.selected_units
                 if (coverage is None or unit.unit_id in coverage)
                 and (not unit.need_ids or need.need_id in unit.need_ids))


def need_read_results(plan, results, need_id):
    reads = {task.step_id for task in plan.steps if task.kind == "READ" and need_id in task.need_ids}
    return tuple(result for result in results if result.step_id in reads)


def need_read_complete(plan, results, need):
    required = required_source_units(plan, need)
    return bool(required) and completed_source_coverage(plan, need_read_results(plan, results, need.need_id), required)


def incomplete_full_coverage(plan, results, *, allow_terminal_gaps=False):
    terminal = {gap.need_id for gap in plan.terminal_gaps} if allow_terminal_gaps else set()
    return any(not completed_source_coverage(plan, need_read_results(plan, results, need.need_id),
                                              required_source_units(plan, need))
               for need in plan.task_state.needs
               if need.need_id not in terminal and
               (plan.coverage_scope == "all_supplied_documents" or need.scope == "all_supplied_documents"))


def direct_answer_reason(state, candidate, *, require_complete_evidence=True):
    """Distinguish skipping READ from delivering an answer with evidence gaps.

    Skipping READ requires sufficient evidence for every Need. An ANSWER may
    report unresolved points after its source scope has been read. Statistical
    views still require their declared selection and computation.
    """
    return next(iter(answer_evidence_gaps(state, candidate,
        require_complete_evidence=require_complete_evidence).values()), None)


def answer_evidence_gaps(state, candidate=None, *, require_complete_evidence=True):
    """Return unresolved Needs for the facts alone or the actual ANSWER input."""
    from .reading import numeric_selection_step_ids

    plan = state.plan
    selector_ids = numeric_selection_step_ids(plan)
    points = {point.need_id: point for point in plan.answer_plan.points}
    results = tuple(state.results)
    open_need_ids = {gap.need_id for gap in state.open_needs}
    raw = candidate is not None and candidate.representation == "raw"
    terminal = {gap.need_id for gap in plan.terminal_gaps}
    gaps = {}
    for need in plan.task_state.needs:
        if need.need_id in terminal:
            # This is an explicit partial/IDK answer, never sufficient evidence
            # for a correctness claim or permission to skip necessary reading.
            if require_complete_evidence or raw:
                gaps[need.need_id] = "terminal_evidence_gap"
            continue
        full_scope = plan.coverage_scope == "all_supplied_documents" or need.scope == "all_supplied_documents"
        numeric = need.operation in {"count", "sum", "group"}
        if not require_complete_evidence and not full_scope and not numeric:
            continue
        required = required_source_units(plan, need)
        read_results = need_read_results(plan, results, need.need_id)
        facts_complete = bool(required) and completed_source_coverage(plan, read_results, required)
        point = points.get(need.need_id)
        if numeric:
            if not facts_complete or raw:
                gaps[need.need_id] = "numeric_fact_scope_incomplete"
                continue
            if need.operation in {"count", "sum"} and need.need_id in open_need_ids:
                gaps[need.need_id] = "numeric_fields_incomplete"
                continue
            selector_id = selector_ids.get(need.need_id)
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
        if not require_complete_evidence and full_scope and facts_complete:
            continue
        supported = any(need.need_id in result.support for result in read_results)
        reduced = any(result.reduction is not None and need.need_id in result.reduction.need_ids
                      and result.reduction.operation != "numeric_select" for result in results)
        conclusion_complete = (point is not None and bool(point.claim_ids) and (supported or reduced)
            and need.need_id not in open_need_ids
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
    retained.extend((last_turn.turn_id, message.role, 0, len(message.content))
                    for message in history_messages((last_turn,)))
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
