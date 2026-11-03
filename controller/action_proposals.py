"""Shared single-action proposals over the existing observed reading state."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from .context_budget import bind_answer_window
from .context_schema import OpenNeed, ReadTask
from .plan_policy import complete_plan
from .serialization import dumps


def gap_key(need):
    return need.need_id, need.missing_relation, need.bound_entities, need.temporal_scope


@dataclass(frozen=True)
class GapRetrievalRequest:
    open_needs: tuple[OpenNeed, ...]
    query: str
    retriever_identity: dict
    limits: dict


def gap_query(state, open_needs):
    from adapters.evidence_pool import SourceDirectory
    return "\n\n".join(SourceDirectory.open_need_query(state, need) for need in open_needs)



def build_observed_action_plans(frozen, snapshot, template, *, reader_checkpoint,
                                gap_request=None, answer_system_prompt=None, include_gap_step=False):
    """Register at most one ready READ, one current-basis REDUCE and one GAP."""
    base = bind_answer_window(frozen.plan, template, snapshot, answer_system_prompt,
                             preserve_step_ids=tuple(record.step_id for record in frozen.steps))
    covered = {unit for result in frozen.results for unit in result.covered_units}
    if base.coverage_scope == "all_supplied_documents" and not set(base.coverage_units) <= covered:
        return (), ({"family": "direct_answer", "reason": "mandatory_source_coverage_incomplete"},)
    historic = tuple(record.task for record in frozen.steps)
    identity = hashlib.sha256(dumps((base.task_state, frozen.results, frozen.claims,
        frozen.coverage, frozen.open_needs, base.selected_set, snapshot.raw_history)).encode()).hexdigest()[:16]
    completed = {result.step_id for result in frozen.results}
    completed.update(result.reduction.step_id for result in frozen.results if result.reduction is not None)
    branches = []

    def add(kind, tasks):
        plan = replace(base, plan_id=base.plan_id+":observed:"+identity+":"+kind, steps=(*historic, *tasks))
        branches.append(complete_plan(plan, (), semantic_kind=kind, reader_checkpoint=reader_checkpoint))

    add("direct_answer", ())
    ready = tuple(task for task in base.steps if task.kind == "READ" and task.step_id not in completed
                  and set(task.predecessor_step_ids) <= completed)
    if ready:
        add("read_once", ready[:1])
    for need in base.task_state.needs:
        if need.operation not in {"enumerate", "count", "sum", "group", "compare", "infer"}:
            continue
        claims = tuple(claim for claim in frozen.claims if claim.kind == "extract" and claim.serves(need.need_id))
        if not claims:
            continue
        inputs = tuple(claim.claim_id for claim in claims)
        operation = "union" if need.operation == "enumerate" else need.operation
        if any(result.reduction is not None and need.need_id in result.reduction.need_ids
               and result.reduction.operation == operation
               and set(inputs) <= set(result.reduction.basis_claim_ids) for result in frozen.results):
            continue
        task = ReadTask("policy:observed:reduce:"+need.need_id+":"+identity, "REDUCE", (need.need_id,),
            input_claim_ids=inputs, reduction_basis_claim_ids=inputs,
            predecessor_step_ids=tuple(dict.fromkeys(claim.rsplit(":claim:", 1)[0] for claim in inputs)),
            input_budget=base.budget.window_tokens-base.budget.read_output_tokens,
            output_budget=base.budget.read_output_tokens,
            operation=operation,
            retained_conclusion_tokens=tuple((identity, tokens) for identity, tokens
                in base.answer_allocation.need_tokens if identity == need.need_id))
        add("reduce_once", (task,))
        break
    if gap_request is not None:
        if not gap_request.open_needs or not set(gap_request.open_needs) <= set(frozen.open_needs):
            raise ValueError("registered GAP relations must belong to the frozen observed state")
        if gap_request.query != gap_query(base.task_state, gap_request.open_needs):
            raise ValueError("registered GAP query differs from the frozen task")
        if not gap_request.retriever_identity or any(gap_request.limits[name] < 1
                                                   for name in ("max_units", "max_passages")):
            raise ValueError("registered GAP requires retriever identity and positive limits")
        parents = tuple(dict.fromkeys(parent for need in gap_request.open_needs for parent in need.parent_claim_ids))
        task = ReadTask("policy:gap:"+identity, "GAP", tuple(dict.fromkeys(need.need_id for need in gap_request.open_needs)),
            input_claim_ids=parents, predecessor_step_ids=tuple(record.step_id for record in frozen.steps),
            input_budget=0, output_budget=0, operation="retrieve")
        add("gap_retrieval", (task,) if include_gap_step else ())
    return tuple(branches), ()

