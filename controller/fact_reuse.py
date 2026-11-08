"""Reuse completed source facts and their extraction contracts across actual QA."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json

from .claim_graph import ClaimGraph
from .context_budget import _covers, bind_answer_window, resolved_condition, text_ref_sources, text_ref_text
from .context_schema import (NumericReadScope, ReadResult, ReadTask,
                             ReadingOutcome)
from .serialization import dumps, to_jsonable


@dataclass(frozen=True)
class ReadReuse:
    need_id: str
    read_id: str
    mode: str


def _resolve(ref, state, history):
    return text_ref_text(ref, text_ref_sources(history, current_turn_id=state.current_turn_id,
                                              current_user_text=state.raw_question))


def condition_view(condition, state, history):
    return resolved_condition(condition, lambda ref: _resolve(ref, state, history))


def _condition_view(identity, state, history):
    conditions = {condition.condition_id: condition for condition in state.conditions}
    return condition_view(conditions[identity], state, history) if identity in conditions else json.loads(identity)


def _valid_units(units, blocks):
    for unit in units:
        for span in unit.source_spans:
            if span.source_key not in blocks:
                return False
            block = blocks[span.source_key]
            base = block.span[0] if block.span else 0
            if not base <= span.start < span.end <= base+len(block.text):
                return False
            if block.text[span.start-base:span.end-base] != unit.text:
                return False
    return True


@dataclass(frozen=True)
class ReadContract:
    attempt_id: str
    outcome: ReadingOutcome
    step_id: str
    need_id: str

    @property
    def read_id(self):
        return f"{self.attempt_id}:{self.step_id}:{self.need_id}"

    @property
    def record(self):
        return next(record for record in self.outcome.steps if record.step_id == self.step_id)

    @property
    def units(self):
        units = {unit.unit_id: unit for unit in self.outcome.plan.selected_set.selected_units}
        return tuple(units[identity] for identity in self.record.task.input_unit_ids)

    @property
    def atoms(self):
        graph = ClaimGraph(self.outcome.claims)
        atoms = {claim.claim_id: claim for claim in self.record.result.claims
                 if claim.kind == "extract" and claim.serves(self.need_id)}
        parents = [parent for atom in atoms.values() for parent in atom.parent_claim_ids]
        parents.extend(parent for gap in self.open_needs for parent in gap.parent_claim_ids)
        for parent in parents:
            atoms.update((claim.claim_id, claim) for claim in graph.leaf_claims(parent) if claim.kind == "extract")
        return tuple(atoms.values())

    @property
    def open_needs(self):
        return tuple(gap for gap in self.record.result.open_needs if gap.need_id == self.need_id)

    def visible(self, history):
        state, task = self.outcome.plan.task_state, self.record.task
        conditions = {condition.condition_id: condition for condition in state.conditions}
        return {"source_ranges": to_jsonable(tuple(span for unit in self.units for span in unit.source_spans)),
            "operation": task.operation, "coverage_scope": self.outcome.plan.coverage_scope,
            "needs": [{"request": [_resolve(ref, state, history) for ref in need.anchor_spans],
                "subjects": [_resolve(ref, state, history) for ref in need.subject_refs],
                "operation": need.operation, "scope": need.scope,
                "conditions": [condition_view(conditions[identity], state, history) for identity in need.condition_ids]}
                for need in state.needs if need.need_id == self.need_id],
            "facts": [{"text": atom.text, "entity": atom.entity, "relation": atom.relation,
                "value": atom.value, "unit": atom.unit, "version": atom.version,
                "conditions": [_condition_view(identity, state, history) for identity in atom.conditions],
                "sources": to_jsonable(atom.source_spans)} for atom in self.atoms],
            "open_needs": [{"missing_relation": need.missing_relation, "bound_entities": need.bound_entities,
                            "temporal_scope": need.temporal_scope} for need in self.open_needs]}


def read_contracts(memory):
    """Only complete physical READs with unchanged literal sources are offered."""
    if not memory.committed_readings:
        return ()
    blocks = memory.source_index._blocks
    scope = (None if memory.source_scope is None else
             {block.key for block in memory.source_index.source_fetch(memory.source_scope)})
    contracts = []
    for attempt_id, outcome in memory.committed_readings:
        for record in outcome.steps:
            if (outcome.plan.reading_mode != "typed_reduce" or record.task.kind != "READ"
                    or record.task.operation == "summarize" or record.result is None or record.receipt is None
                    or not record.receipt.generation_complete):
                continue
            for need_id in record.task.need_ids:
                contract = ReadContract(attempt_id, outcome, record.step_id, need_id)
                if (contract.units and set(record.task.input_unit_ids) == set(record.result.covered_units)
                        and _valid_units(contract.units, blocks)
                        and (scope is None or all(span.source_key in scope
                            for span in (*tuple(span for unit in contract.units for span in unit.source_spans),
                                         *tuple(span for atom in contract.atoms for span in atom.source_spans))))):
                    contracts.append(contract)
    return tuple(contracts)


def _applicable(contract, need, state, memory):
    source = memory.source_index
    if not _valid_units(contract.units, source._blocks):
        return False
    keys = {span.source_key for item in (*contract.units, *contract.atoms) for span in item.source_spans}
    scope = state.effective_source_scope
    if scope is not None and not keys.issubset(scope):
        return False
    roots = state.source_scope if scope is None and state.source_scope else tuple(keys)
    applicable = source.applicable_scope(roots, replace(state, needs=(need,)), memory)
    return keys.issubset(block.key for block in applicable.source_blocks)


def build_reuse_seed(plan, bindings, contracts, memory):
    """Bind source atoms to this plan; completed logical nodes carry no old calls."""
    state = plan.task_state
    needs = {need.need_id: need for need in state.needs}
    by_read = {contract.read_id: contract for contract in contracts}
    current_conditions = {dumps(condition_view(condition, state, memory.committed_history)): condition.condition_id
                          for condition in state.conditions}
    graph, results, tasks, coverage = ClaimGraph(), [], [], []
    material = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    blocks = {block.key: block for block in plan.selected_set.source_blocks}
    for binding in bindings:
        if binding.mode == "none":
            continue
        if binding.mode not in {"facts", "complete"}:
            raise ValueError("unknown READ reuse decision")
        need, contract = needs[binding.need_id], by_read[binding.read_id]
        if not _applicable(contract, need, state, memory):
            continue
        prior_state = contract.outcome.plan.task_state
        origin = f"history:{contract.attempt_id}:"
        identity = sha256(dumps((contract.read_id, need.need_id)).encode()).hexdigest()[:20]
        step_id = f"policy:reuse:{contract.attempt_id}:{contract.step_id}:{identity}"
        original_graph = ClaimGraph(contract.outcome.claims)
        atoms = []
        for atom in contract.atoms:
            translated = []
            for condition_id in atom.conditions:
                literal = dumps(_condition_view(condition_id, prior_state, memory.committed_history))
                translated.append(current_conditions[literal] if literal in current_conditions else literal)
            reused = replace(atom, claim_id=f"{origin}{atom.claim_id}", need_id=need.need_id,
                shared_need_ids=(), conditions=tuple(translated), parent_claim_ids=(),
                source_spans=original_graph.source_spans(atom.claim_id))
            if reused.claim_id in graph.claims:
                previous = graph.claims[reused.claim_id]
                reused = replace(previous, shared_need_ids=tuple(dict.fromkeys((
                    *previous.shared_need_ids, *((need.need_id,) if need.need_id != previous.need_id else ())))))
                graph.claims[reused.claim_id] = reused
            else:
                graph.add(reused)
            atoms.append(reused)
        atom_ids = {atom.claim_id: reused.claim_id for atom, reused in zip(contract.atoms, atoms, strict=True)}
        open_needs = tuple(replace(gap, need_id=need.need_id,
            parent_claim_ids=tuple(dict.fromkeys(atom_ids[leaf.claim_id]
                for parent in gap.parent_claim_ids for leaf in original_graph.leaf_claims(parent)
                if leaf.kind == "extract"))) for gap in contract.open_needs)
        unit_ids = tuple(unit.unit_id for unit in contract.units) if binding.mode == "complete" else ()
        result = ReadResult(step_id, tuple(atoms), unit_ids,
            (need.need_id,) if binding.mode == "complete" and not open_needs
                and contract.need_id in contract.record.result.support else (), open_needs=open_needs)
        results.append(result)
        numeric = need.operation in {"count", "sum"}
        tasks.append(ReadTask(step_id, "READ", (need.need_id,), unit_ids,
            input_budget=0, output_budget=0, operation="extract",
            shared_numeric_scopes=(NumericReadScope(f"reuse:{identity}", (need.need_id,),
                need.condition_ids, (need.operation,)),) if numeric else (), condition_ids=need.condition_ids,
            requires_reduction=need.operation in {"count", "sum", "group", "infer", "compare", "enumerate"}))
        for unit in contract.units:
            previous = material.get(unit.unit_id)
            material[unit.unit_id] = replace(unit, need_ids=tuple(dict.fromkeys((
                *(previous.need_ids if previous is not None else ()), need.need_id))))
        source_keys = {span.source_key for unit in contract.units for span in unit.source_spans}
        source_keys.update(span.source_key for atom in atoms for span in atom.source_spans)
        blocks.update((key, memory.source_index._blocks[key]) for key in source_keys)
        coverage.extend((plan.coverage_scope, span.source_key.document_id, unit.unit_id, step_id)
            for unit in contract.units if unit.unit_id in unit_ids for span in unit.source_spans)
    selected = replace(plan.selected_set, selected_units=tuple(material.values()), source_blocks=tuple(blocks.values()))
    results = tuple(replace(result, claims=tuple(graph.claims[claim.claim_id] for claim in result.claims))
                    for result in results)
    updated = replace(plan, selected_set=selected, steps=(*plan.steps, *tasks),
        answer_plan=graph.answer_plan(state, plan.selected_set.disposition, results))
    from .reading import unresolved_needs
    return ReadingOutcome(updated, results, claims=tuple(graph.claims.values()),
                          coverage=tuple(dict.fromkeys(coverage)), open_needs=unresolved_needs(results))


def complete_reuse_ranges(seed):
    by_unit = {unit.unit_id: unit for unit in seed.plan.selected_set.selected_units}
    by_step = {task.step_id: task for task in seed.plan.steps}
    ranges = {}
    for result in seed.results:
        for need_id in by_step[result.step_id].need_ids:
            ranges.setdefault(need_id, []).extend(span for identity in result.covered_units
                for span in by_unit[identity].source_spans)
    return {need_id: tuple(dict.fromkeys(spans)) for need_id, spans in ranges.items()}


def apply_reuse_to_plan(plan, seed, template, snapshot, *, output_tokens_per_fact=64):
    """Repack only missing initial source/Need ranges, retaining logical reuse."""
    from .evidence_planner import close_dependencies
    from .reading import build_read_plan

    completed_ranges = complete_reuse_ranges(seed)
    ranges = {need_id: {} for need_id in completed_ranges}
    for need_id, spans in completed_ranges.items():
        for span in spans:
            ranges[need_id].setdefault(span.source_key, []).append((span.start, span.end))
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    units.update((unit.unit_id, unit) for unit in seed.plan.selected_set.selected_units if unit.unit_id not in units)
    claims = {claim.claim_id: claim for claim in seed.claims}
    completed = {result.step_id for result in seed.results}
    imported = tuple(task for task in seed.plan.steps if task.kind == "READ" and task.step_id in completed)
    groups = {}
    for task in plan.steps:
        if task.kind != "READ" or task.step_id in completed:
            continue
        for identity in task.input_unit_ids:
            unit = units[identity]
            missing = tuple(need_id for need_id in task.need_ids if not all(
                _covers((span.start, span.end), ranges.get(need_id, {}).get(span.source_key, ()))
                for span in unit.source_spans))
            if missing:
                key = (missing, task.input_claim_ids, task.predecessor_step_ids)
                groups.setdefault(key, {})[identity] = unit
    reads = []
    for (need_ids, parents, predecessors), roots in groups.items():
        material = close_dependencies(tuple(roots), units)
        state = replace(plan.task_state, needs=tuple(need for need in plan.task_state.needs if need.need_id in need_ids))
        identity = sha256(dumps((state, parents, predecessors, plan.reading_mode,
            plan.read_layout, plan.read_packing, tuple(replace(unit, retrieval_score=0.0, need_ids=())
                for unit in material))).encode()).hexdigest()[:20]
        pending = build_read_plan(state, replace(plan.selected_set, selected_units=material),
            plan.coverage_scope, plan.budget, template=template, snapshot=snapshot,
            input_claims=tuple(claims[parent] for parent in parents), reading_mode=plan.reading_mode,
            output_tokens_per_fact=output_tokens_per_fact,
            separate_needs=plan.separate_needs, numeric_identity=plan.numeric_identity,
            read_layout=plan.read_layout, read_packing=plan.read_packing,
            answer_system_prompt=plan.answer_system_prompt, _step_prefix=f"policy:read:{identity}:")
        units.update((unit.unit_id, unit) for unit in pending.selected_set.selected_units)
        reads.extend(replace(task, predecessor_step_ids=predecessors) for task in pending.steps)
    if len(reads)+1 > plan.budget.max_steps:
        raise ValueError("remaining source coverage exceeds the request step allocation")
    blocks = {block.key: block for block in (*plan.selected_set.source_blocks, *seed.plan.selected_set.source_blocks)}
    selected = replace(plan.selected_set, selected_units=tuple(units.values()), source_blocks=tuple(blocks.values()))
    points = {point.need_id: point for point in plan.answer_plan.points}
    points.update((point.need_id, point) for point in seed.plan.answer_plan.points
                  if point.claim_ids or point.need_id not in points)
    answer = replace(seed.plan.answer_plan, points=tuple(points[need.need_id]
        for need in plan.task_state.needs if need.need_id in points))
    updated = bind_answer_window(replace(plan, selected_set=selected, steps=(*imported, *reads),
        answer_plan=answer, answer_allocation=None), template, snapshot,
        preserve_step_ids=tuple(completed))
    return replace(seed, plan=updated)
