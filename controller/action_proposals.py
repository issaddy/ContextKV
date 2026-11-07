"""Shared single-action proposals over the existing observed reading state."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from .context_budget import bind_answer_window
from .context_schema import OpenNeed, ReadTask
from .plan_policy import complete_plan
from .reading import numeric_reduction_inputs, numeric_selection_step_id, numeric_selection_task
from .serialization import dumps


FINITE_ACTION_DOMAIN = "two_reads_shared_outputs_terminal_representations_v1"
ALTERNATE_ANSWERS = ("raw", "facts", "reduced")


def finite_action_domain(output_budgets, output_tokens_per_fact):
    return {"protocol": FINITE_ACTION_DOMAIN, "read_choices": 2,
            "output_budgets": sorted(set(output_budgets)), "output_tokens_per_fact": output_tokens_per_fact}


def observed_step_outputs(plan, task, template, *, output_budgets, output_tokens_per_fact, finite):
    from .reading import read_output_reservation

    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    dependencies = {identity for unit in units.values() for identity in unit.dependencies}
    reserved = read_output_reservation(plan.task_state, task,
        tuple(units[identity] for identity in task.input_unit_ids), template.tokenizer,
        dependency_ids=dependencies, output_tokens_per_fact=output_tokens_per_fact, reading_mode=plan.reading_mode)
    outputs = tuple(value for value in output_budgets if reserved <= value <= task.output_budget)
    if finite and outputs:
        # READ has a fact reservation. REDUCE packs were sized for the full output allowance.
        outputs = (max(outputs) if task.kind == "REDUCE" else min(outputs),)
    return reserved, outputs


def observed_answer_options(state, *, joint, finite):
    from .context_budget import answer_evidence_gaps

    options = {"layouts": ("existing",), "actions": ("REBUILD",)} if finite else {}
    if joint:
        representations = ["hybrid"]
        if not answer_evidence_gaps(state):
            roots = {identity for point in state.plan.answer_plan.points for identity in point.claim_ids}
            derived = any(claim.claim_id in roots and claim.kind == "derived" for claim in state.claims)
            representations.append("reduced" if derived else "facts")
        if not any(need.operation in {"count", "sum", "group"} for need in state.plan.task_state.needs):
            representations.append("raw")
        options["representations"] = tuple(representations)
    return options


def eligible_observed_answers(state, candidates):
    from .context_budget import answer_evidence_gaps
    from .request_compiler import unique_answer_inputs

    return unique_answer_inputs(tuple(candidate for candidate in candidates
        if candidate.representation not in ALTERNATE_ANSWERS or not answer_evidence_gaps(state, candidate)))


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


def gap_scope_available(state):
    """A directed search can add material only outside completed source intervals."""
    from .context_budget import _covers, completed_source_ranges

    plan = state.plan
    if plan.coverage_scope == "all_supplied_documents":
        return False
    scope = plan.task_state.effective_source_scope
    if scope is None:
        return True
    blocks = {block.key: block for block in plan.selected_set.source_blocks}
    ranges = completed_source_ranges(plan, state.results)
    return any(key not in blocks or not _covers(blocks[key].span or (0, len(blocks[key].text)), ranges.get(key, ()))
               for key in scope)


def claim_predecessor_steps(results, claim_ids):
    """Resolve current producers from results, including logical reuse steps."""
    producers = {claim.claim_id: result.step_id for result in results for claim in result.claims}
    return tuple(dict.fromkeys(producers[identity] for identity in claim_ids if identity in producers))


def bind_read_parents(task, state):
    """Give the next READ the facts that establish its unresolved relations."""
    parents = tuple(dict.fromkeys((*task.input_claim_ids,
        *(parent for need in state.open_needs if need.need_id in task.need_ids
          for parent in need.parent_claim_ids))))
    return replace(task, input_claim_ids=parents)


def gap_retrieval_plan(state, request):
    """The retriever excludes completed source ranges, not unread candidates."""
    from .evidence_planner import _passages, close_dependencies

    plan = state.plan
    covered = tuple(dict.fromkeys(identity for result in state.results for identity in result.covered_units))
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    read_units = close_dependencies(covered, units)
    cap = len(_passages(read_units)) + request.limits["max_passages"]
    if plan.budget.max_passages is not None:
        cap = min(cap, plan.budget.max_passages)
    return replace(plan, selected_set=replace(plan.selected_set, selected_units=read_units),
                   budget=replace(plan.budget, max_passages=cap))



def observed_reduction_task(frozen):
    """Choose the next typed reduction or compress a multi-fact extract basis."""
    base = frozen.plan
    if base.reading_mode == "coverage_only":
        return None
    shared_needs = {need_id for task in base.steps if task.kind == "READ"
                    for scope in task.shared_numeric_scopes for need_id in scope.need_ids}
    for need in base.task_state.needs:
        if need.need_id in shared_needs:
            selector = numeric_selection_task(base, frozen.results, frozen.claims, need_id=need.need_id)
            if selector is not None:
                return selector
    typed = {"enumerate", "count", "sum", "group", "compare", "infer"}
    needs = sorted(base.task_state.needs, key=lambda need: need.operation not in typed)
    for need in needs:
        summarize = need.operation not in typed
        shared_numeric = need.need_id in shared_needs
        if shared_numeric:
            selector_id = numeric_selection_step_id(base, need.need_id)
            selection = next((result for result in reversed(frozen.results)
                if result.step_id == selector_id or
                (result.reduction is not None and result.reduction.step_id == selector_id)), None)
            if selection is None:
                continue
            claims = numeric_reduction_inputs(base, need.need_id, frozen.results)
            same_selection = {record.step_id for record in frozen.steps
                              if selection.step_id in record.task.predecessor_step_ids}
        else:
            claims = tuple(claim for claim in frozen.claims if claim.kind == "extract" and claim.serves(need.need_id))
            read_ids = {task.step_id for task in base.steps if task.kind == "READ" and need.need_id in task.need_ids}
            empty_group = (need.operation == "group" and bool(read_ids)
                           and read_ids <= {result.step_id for result in frozen.results})
            if len(claims) < (2 if summarize else 1) and not empty_group:
                continue
        inputs = tuple(claim.claim_id for claim in claims)
        operation = ("summarize" if summarize else "union"
                     if need.operation == "enumerate" else need.operation)
        if any(result.reduction is not None and need.need_id in result.reduction.need_ids
               and result.reduction.operation == operation
               and (result.step_id in same_selection if shared_numeric
                    else set(inputs) <= set(result.reduction.basis_claim_ids))
               for result in frozen.results):
            continue
        predecessors = ((selection.step_id,) if shared_numeric
                        else claim_predecessor_steps(frozen.results, inputs))
        identity = hashlib.sha256(dumps((need.need_id, operation, inputs, predecessors)).encode()).hexdigest()[:16]
        return ReadTask("policy:observed:reduce:"+need.need_id+":"+identity, "REDUCE", (need.need_id,),
            input_claim_ids=inputs, reduction_basis_claim_ids=inputs,
            predecessor_step_ids=predecessors,
            input_budget=base.budget.window_tokens-base.budget.read_output_tokens,
            output_budget=base.budget.read_output_tokens,
            operation=operation,
            retained_conclusion_tokens=tuple((identity, tokens) for identity, tokens
                in base.answer_allocation.need_tokens if identity == need.need_id)
                if base.answer_allocation is not None else ())
    return None



def build_observed_action_plans(frozen, snapshot, template, *, reader_checkpoint,
                                gap_request=None, answer_system_prompt=None, include_gap_step=False,
                                read_choices=1):
    """Register a bounded set of ready READs, one current-basis REDUCE and one GAP."""
    base = bind_answer_window(frozen.plan, template, snapshot, answer_system_prompt,
                             preserve_step_ids=tuple(record.step_id for record in frozen.steps))
    result_ids = {result.step_id for result in frozen.results}
    historic = {task.step_id: task for task in base.steps if task.kind == "READ" and task.step_id in result_ids}
    historic.update((record.step_id, record.task) for record in frozen.steps)
    historic = tuple(historic.values())
    identity = hashlib.sha256(dumps((base.task_state, frozen.results, frozen.claims,
        frozen.coverage, frozen.open_needs, base.selected_set, snapshot.raw_history)).encode()).hexdigest()[:16]
    completed = {result.step_id for result in frozen.results}
    completed.update(result.reduction.step_id for result in frozen.results if result.reduction is not None)
    branches = []
    unmet = tuple(need.need_id for need in base.task_state.needs
                  if not any(claim.serves(need.need_id) for claim in frozen.claims))

    def add(kind, tasks, suffix=""):
        plan = replace(base, plan_id=base.plan_id+":observed:"+identity+":"+kind+suffix, steps=(*historic, *tasks))
        branches.append(complete_plan(plan, (), semantic_kind=kind, reader_checkpoint=reader_checkpoint,
                                      unmet_answer_points=unmet))

    add("direct_answer", ())
    ready = tuple(task for task in base.steps if task.kind == "READ" and task.step_id not in completed
                  and set(task.predecessor_step_ids) <= completed)
    if read_choices > 1:
        from .demand import DemandState
        demand = DemandState.from_outcome(frozen)
        ready = tuple(sorted(ready, key=lambda task: -sum(need.priority for need in demand.needs
                                                          if need.need_id in task.need_ids)))
    for index, task in enumerate(ready[:read_choices]):
        add("read_once", (bind_read_parents(task, frozen),), ":"+task.step_id if index else "")
    reduction = observed_reduction_task(replace(frozen, plan=base))
    if reduction is not None:
        if reduction.operation not in {"union", "count", "sum", "group"}:
            from .execution import reduction_schedule
            by_id = {claim.claim_id: claim for claim in frozen.claims}
            pending, _, _ = reduction_schedule(base, reduction,
                tuple(by_id[identity] for identity in reduction.input_claim_ids),
                frozen.results, snapshot, template)
            reduction, _ = pending
        add("reduce_once", (reduction,))
    if gap_request is not None and gap_scope_available(frozen):
        if not gap_request.open_needs or not set(gap_request.open_needs) <= set(frozen.open_needs):
            raise ValueError("registered GAP relations must belong to the frozen observed state")
        if gap_request.query != gap_query(base.task_state, gap_request.open_needs):
            raise ValueError("registered GAP query differs from the frozen task")
        if not gap_request.retriever_identity or any(gap_request.limits[name] < 1
                                                   for name in ("max_units", "max_passages")):
            raise ValueError("registered GAP requires retriever identity and positive limits")
        parents = tuple(dict.fromkeys(parent for need in gap_request.open_needs for parent in need.parent_claim_ids))
        predecessors = tuple(dict.fromkeys((*(record.step_id for record in frozen.steps),
                                            *claim_predecessor_steps(frozen.results, parents))))
        task = ReadTask("policy:gap:"+identity, "GAP", tuple(dict.fromkeys(need.need_id for need in gap_request.open_needs)),
            input_claim_ids=parents, predecessor_step_ids=predecessors,
            input_budget=0, output_budget=0, operation="retrieve")
        add("gap_retrieval", (task,) if include_gap_step else ())
    return tuple(branches), ()



BOUNDED_CONTINUATION = "infer_bridge_ready_or_one_gap_one_terminal_read_v4"


def continuation_gap(state, *, used_gap):
    """Only an observed infer relation can justify the one continuation GAP."""
    if used_gap or not gap_scope_available(state):
        return ()
    infer = {need.need_id for need in state.plan.task_state.needs if need.operation == "infer"}
    return tuple(need for need in state.open_needs if need.need_id in infer)


def validate_action_counts(kind, counts, *, complete=True):
    """Bounded branches contain the first action and at most GAP + terminal READ."""
    first = {"direct_answer": None, "read_once": "READ", "reduce_once": "REDUCE",
             "gap_retrieval": "GAP_RETRIEVAL"}[kind]
    limits = {"READ": int(first == "READ") + int(first is not None),
              "REDUCE": int(first == "REDUCE"), "GAP_RETRIEVAL": int(first is not None), "ANSWER": 1}
    if set(counts) != set(limits) or any(type(counts[k]) is not int or not 0 <= counts[k] <= v for k, v in limits.items()):
        raise ValueError("action counts exceed bounded continuation protocol")
    if complete and (counts["ANSWER"] != 1 or (first and counts[first] < 1)):
        raise ValueError("registered first action and terminal ANSWER must execute")



def continuation_read(state, catalog):
    """Prefer one existing ready source for an observed missing infer relation."""
    needs = {need.need_id for need in continuation_gap(state, used_gap=False)}
    completed = {result.step_id for result in state.results}
    covered = {unit for result in state.results for unit in result.covered_units}
    task = next((task for task in catalog.steps if task.kind == "READ"
        and task.step_id not in completed and needs.intersection(task.need_ids)
        and set(task.predecessor_step_ids) <= completed
        and not set(task.input_unit_ids) <= covered), None)
    return bind_read_parents(task, state) if task is not None else None
