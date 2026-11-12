"""Bounded, durable READ recovery over the original supplied source intervals.

No model call occurs here. Every successor is an ordinary quoted READ admitted
by the rolling controller's existing complete-response budget. Paid failures
remain StepRecords and never contribute facts or source coverage.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import re

from .context_schema import NumericReadScope, ReadTask
from .context_budget import completed_source_coverage
from .reading import (_period_key, _repair_target_matches, normalized_fact_text as _norm,
                      numeric_gap_field as _field, unresolved_needs)
from .serialization import dumps


MAX_SPLIT_DEPTH = 2
MAX_FIELD_REPAIRS = 2
MAX_REPAIR_UNITS = 4
REPAIR_CHAR_WINDOW = 1536


def _id(*values):
    return sha256(dumps(values).encode()).hexdigest()[:24]


def gap_key(gap):
    return gap.need_id, gap.missing_relation, gap.bound_entities, gap.temporal_scope


def _slices(unit, plan, tokenizer, *, targeted=None, split=False):
    """Derive source geometry from the original block, never from token offsets."""
    blocks = {block.key: block for block in plan.selected_set.source_blocks}
    pieces, headers = [], {}
    for span in unit.source_spans:
        block = blocks.get(span.source_key)
        if block is None:
            # Some unit catalogs are self-contained. Only a one-span exact
            # raw-text match has sufficient geometry to split safely.
            if len(unit.source_spans) != 1 or len(unit.text) != span.end-span.start:
                return ()
            text = unit.text
        else:
            base = block.span[0] if block.span else 0
            text = block.text[span.start-base:span.end-base]
        intervals = [(0, len(text))]
        first_line = text.split("\n", 1)[0]
        header_end = 0
        if "\n" in text and ("|" in first_line or "\t" in first_line or "单位" in first_line
                              or re.search(r"\bunits?\b", first_line, re.I)):
            header_end = min(256, len(first_line)+1)
        elif "\n" not in text:
            marker = re.search(r"[:：]", text[:128])
            header_end = marker.end() if marker is not None else 0
        if split and len(text) >= 64:
            midpoint = len(text)//2
            boundaries = [match.end() for match in re.finditer(r"[\n；;。]", text)
                          if len(text)//4 <= match.end() <= 3*len(text)//4]
            if boundaries:
                cut = min(boundaries, key=lambda value: abs(value-midpoint))
                intervals = [(0, cut), (cut, len(text))]
            # A character midpoint can sever one entity/field/value record.
            # If there is no record/sentence boundary, leave it unsplittable
            # and let bounded recovery report the unresolved point explicitly.
        elif targeted is not None and len(text) > REPAIR_CHAR_WINDOW:
            target = _field(targeted)
            at = text.casefold().find(target.casefold()) if target else -1
            if at < 0:
                at = next((text.find(entity) for entity in targeted.bound_entities if entity in text), 0)
            left = max(0, at-REPAIR_CHAR_WINDOW//3)
            right = min(len(text), left+REPAIR_CHAR_WINDOW)
            left = max(0, right-REPAIR_CHAR_WINDOW)
            intervals = [(left, right)]
        for left, right in intervals:
            if left == right:
                continue
            identity = "repair-unit:"+_id(unit.unit_id, span.source_key, span.start+left, span.start+right)
            source = replace(span, span_id=identity+":source", start=span.start+left, end=span.start+right)
            body = text[left:right]
            dependencies = unit.dependencies
            if left >= header_end > 0:
                header_id = "repair-header:"+_id(unit.unit_id, span.source_key, span.start, header_end)
                header_span = replace(span, span_id=header_id+":source", end=span.start+header_end)
                header = text[:header_end]
                headers[header_id] = replace(unit, unit_id=header_id, source_spans=(header_span,), text=header,
                    body_tokens=len(tokenizer.encode(header, add_special_tokens=False)))
                dependencies = tuple(dict.fromkeys((*dependencies, header_id)))
            pieces.append(replace(unit, unit_id=identity, source_spans=(source,), text=body,
                dependencies=dependencies, body_tokens=len(tokenizer.encode(body, add_special_tokens=False))))
    return (*pieces, *headers.values())


def _closed_inputs(units, catalog):
    from .evidence_planner import close_dependencies
    return tuple(unit.unit_id for unit in close_dependencies(tuple(unit.unit_id for unit in units), catalog))


def _attach(state, tasks, units=()):
    catalog = {unit.unit_id: unit for unit in (*state.plan.selected_set.selected_units, *units)}
    steps = {task.step_id: task for task in (*state.plan.steps, *tasks)}
    selected = replace(state.plan.selected_set, selected_units=tuple(catalog.values()))
    return replace(state, plan=replace(state.plan, selected_set=selected, steps=tuple(steps.values())))


def _replace_records(state, updates):
    records = {record.step_id: record for record in state.steps}
    records.update((record.step_id, record) for record in updates)
    results = {result.step_id: result for result in state.results}
    results.update((record.step_id, record.result) for record in updates)
    observed = tuple(results.values())
    return replace(state, steps=tuple(records.values()), results=observed,
                   open_needs=unresolved_needs(observed))


def terminalize(state, gaps, reason):
    """Explicit unresolved output, without converting missing facts into zeros."""
    terminal = {gap_key(gap): gap for gap in state.plan.terminal_gaps}
    for gap in gaps:
        final = replace(gap, missing_relation=f"{reason}: {gap.missing_relation}")
        terminal[gap_key(final)] = final
    return replace(state, plan=replace(state.plan, terminal_gaps=tuple(terminal.values()),
        answer_plan=replace(state.plan.answer_plan, disposition="partial")))


def recover_failed_reads(state, tokenizer):
    """Schedule at most two levels of smaller packets for durable failures."""
    for record in tuple(state.steps):
        if not record.metrics.get("read_recovery_pending") or record.metrics.get("read_recovery_children"):
            continue
        parent = next((row for row in state.steps
            if record.step_id in row.metrics.get("read_recovery_children", ())), None)
        depth = 0 if parent is None else parent.metrics.get("read_recovery_depth", 0)+1
        by_id = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
        originals = tuple(by_id[key] for key in record.task.input_unit_ids)
        dependencies = {key for unit in originals for key in unit.dependencies}
        bodies = tuple(unit for unit in originals if unit.unit_id not in dependencies) or originals
        new_units = ()
        packs = ()
        if depth < MAX_SPLIT_DEPTH:
            if len(bodies) > 1:
                midpoint = len(bodies)//2
                packs = (bodies[:midpoint], bodies[midpoint:])
            elif bodies:
                new_units = _slices(bodies[0], state.plan, tokenizer, split=True)
                new_dependencies = {key for unit in new_units for key in unit.dependencies}
                new_bodies = tuple(unit for unit in new_units if unit.unit_id not in new_dependencies)
                if len(new_bodies) > 1:
                    midpoint = len(new_bodies)//2
                    packs = (new_bodies[:midpoint], new_bodies[midpoint:])
        children = []
        catalog = {**by_id, **{unit.unit_id: unit for unit in new_units}}
        for index, pack in enumerate(packs):
            inputs = _closed_inputs(pack, catalog)
            children.append(replace(record.task, step_id="policy:split:"+_id(record.step_id, index, inputs),
                input_unit_ids=inputs, output_schema={}, predecessor_step_ids=(record.step_id,)))
        metrics = {**record.metrics, "read_recovery_depth": depth,
            "read_recovery_children": tuple(child.step_id for child in children),
            "read_recovery_pending": bool(children)}
        state = _replace_records(state, (replace(record, metrics=metrics),))
        if children:
            state = _attach(state, children, new_units)
        else:
            state = terminalize(state, record.result.open_needs, "read_recovery_exhausted")
    return state


def _repairs_gap(gap, result, units=()):
    if any(item.need_id == gap.need_id and _norm(_field(item)) == _norm(_field(gap))
           and (not item.bound_entities or not gap.bound_entities
                or set(map(_norm, item.bound_entities)) & set(map(_norm, gap.bound_entities)))
           and (item.temporal_scope in {"", "current"} or gap.temporal_scope in {"", "current"}
                or _period_key(item.temporal_scope) == _period_key(gap.temporal_scope))
           for item in result.open_needs):
        return False
    texts = {span.span_id: unit.text for unit in units for span in unit.source_spans}
    claims = tuple(claim for claim in result.claims if claim.serves(gap.need_id)
        and claim.value is not None and _repair_target_matches(claim, gap,
            "\n\n".join(dict.fromkeys(texts[span.span_id] for span in claim.source_spans
                                      if span.span_id in texts))))
    return bool(claims) and all(any(_norm(entity) == _norm(claim.entity) for claim in claims)
                                for entity in gap.bound_entities)


def reconcile_recovery(state):
    """Close only repaired field gaps or fully covered split parents."""
    results = {result.step_id: result for result in state.results}
    resolved = set()
    for task in state.plan.steps:
        result = results.get(task.step_id)
        if result is not None and task.operation == "repair_extract":
            resolved.update(gap_key(gap) for gap in task.repair_gaps
                if _repairs_gap(gap, result, state.plan.selected_set.selected_units))
    updates = []
    by_id = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
    records = {record.step_id: record for record in state.steps}
    for record in state.steps:
        children = record.metrics.get("read_recovery_children", ())
        descendants, pending = set(children), list(children)
        while pending:
            child = records.get(pending.pop())
            for identity in child.metrics.get("read_recovery_children", ()) if child is not None else ():
                if identity not in descendants:
                    descendants.add(identity)
                    pending.append(identity)
        repaired = bool(children) and completed_source_coverage(state.plan,
            tuple(result for result in state.results if result.step_id in descendants),
            tuple(by_id[identity] for identity in record.task.input_unit_ids))
        gaps = tuple(gap for gap in record.result.open_needs
            if gap_key(gap) not in resolved and not (repaired and gap.missing_relation.startswith("READ recovery required:")))
        if gaps != record.result.open_needs or repaired and record.metrics.get("read_recovery_pending"):
            updates.append(replace(record, result=replace(record.result, open_needs=gaps), metrics={
                **record.metrics, "read_recovery_pending": False if repaired else record.metrics.get("read_recovery_pending", False)}))
    return _replace_records(state, updates) if updates else state


def schedule_field_repair(state, tokenizer):
    """Repair a remaining numeric gap in its observed source, once per turn.

    Exhaustive reading must finish first. Each Need admits at most two local
    repair calls; an unresolved field then becomes an explicit terminal gap.
    """
    state = reconcile_recovery(state)
    completed = {result.step_id for result in state.results}
    terminal = {gap.need_id for gap in state.plan.terminal_gaps}
    numeric = {need.need_id for need in state.plan.task_state.needs if need.operation in {"count", "sum"}}
    gaps = tuple(gap for gap in state.open_needs if gap.need_id in numeric-terminal
                 and not gap.missing_relation.startswith("READ recovery required:"))
    for need_id in dict.fromkeys(gap.need_id for gap in gaps):
        if any(task.kind == "READ" and need_id in task.need_ids and task.step_id not in completed
               for task in state.plan.steps):
            continue
        repairs = tuple(task for task in state.plan.steps if task.operation == "repair_extract" and need_id in task.need_ids)
        targets = tuple(gap for gap in gaps if gap.need_id == need_id)
        if len(repairs) >= MAX_FIELD_REPAIRS:
            state = terminalize(state, targets, "field_recovery_exhausted")
            continue
        units = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
        origins = {identity for result in state.results if any(gap_key(gap) in {gap_key(item) for item in targets}
            for gap in result.open_needs) for identity in result.covered_units}
        candidates = tuple(units[identity] for identity in origins)
        if not candidates:
            state = terminalize(state, targets, "field_source_unavailable")
            continue
        def score(unit):
            return sum(4*(_field(gap) in unit.text)+sum(entity in unit.text for entity in gap.bound_entities)
                       for gap in targets)
        selected, new_units = [], {}
        for unit in sorted(candidates, key=lambda unit: (-score(unit), unit.unit_id)):
            sliced = _slices(unit, state.plan, tokenizer, targeted=targets[0])
            new_units.update((item.unit_id, item) for item in sliced)
            dependencies = {key for item in sliced for key in item.dependencies}
            selected.extend(item for item in sliced if item.unit_id not in dependencies)
            if len(selected) >= MAX_REPAIR_UNITS:
                break
        selected = tuple(selected[:MAX_REPAIR_UNITS])
        if not selected:
            state = terminalize(state, targets, "field_source_unavailable")
            continue
        parents = tuple(dict.fromkeys(identity for gap in targets for identity in gap.parent_claim_ids))
        bound_need = next(need for need in state.plan.task_state.needs if need.need_id == need_id)
        inputs = _closed_inputs(selected, {**units, **new_units})
        task = ReadTask("policy:field:"+_id(need_id, targets, len(repairs), tuple(unit.unit_id for unit in selected)),
            "READ", (need_id,), inputs, input_claim_ids=parents,
            input_budget=state.plan.budget.read_input_tokens, output_budget=state.plan.budget.read_output_tokens,
            operation="repair_extract", repair_gaps=targets, condition_ids=bound_need.condition_ids,
            shared_numeric_scopes=(NumericReadScope("repair", (need_id,), bound_need.condition_ids,
                operations=(bound_need.operation,)),))
        return _attach(state, (task,), tuple(new_units.values()))
    return state
