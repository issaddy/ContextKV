"""Project current evidence obligations onto the same finite native request set."""
from dataclasses import dataclass

from .context_budget import answer_evidence_gaps
from .context_schema import OpenNeed, SourceSpan


@dataclass(frozen=True)
class NeedDemand:
    need_id: str
    reason: str
    source_spans: tuple[SourceSpan, ...]
    missing_fields: tuple[OpenNeed, ...]
    predecessor_need_ids: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class DemandState:
    needs: tuple[NeedDemand, ...]
    remaining_read_ids: tuple[str, ...]

    @classmethod
    def from_outcome(cls, state):
        gaps = answer_evidence_gaps(state)
        plan = state.plan
        exhaustive = set(plan.coverage_units)
        predecessors = {identity for need in plan.task_state.needs if need.need_id in gaps
                        for identity in need.predecessor_need_ids}
        needs = []
        for need in plan.task_state.needs:
            if need.need_id not in gaps:
                continue
            full = plan.coverage_scope == "all_supplied_documents" or need.scope == "all_supplied_documents"
            sources = tuple(dict.fromkeys(span for unit in plan.selected_set.selected_units
                if (unit.unit_id in exhaustive if full else not unit.need_ids or need.need_id in unit.need_ids)
                for span in unit.source_spans))
            fields = tuple(gap for gap in state.open_needs if gap.need_id == need.need_id)
            priority = 1 + int(full or need.operation in {"count", "sum", "group"}) + int(need.need_id in predecessors)
            needs.append(NeedDemand(need.need_id, gaps[need.need_id], sources, fields,
                need.predecessor_need_ids, priority))
        pending = {need.need_id for need in needs}
        complete = {result.step_id for result in state.results}
        reads = tuple(task.step_id for task in plan.steps if task.kind == "READ"
            and task.step_id not in complete and pending.intersection(task.need_ids))
        return cls(tuple(needs), reads)

    def after(self, task):
        if task.kind == "ANSWER":
            return DemandState((), ())
        # A selected READ only promises an attempt; actual facts resolve Needs.
        return DemandState(self.needs, tuple(identity for identity in self.remaining_read_ids
                                             if identity != task.step_id))


@dataclass(frozen=True)
class DemandLease:
    owner_id: str
    source_step_id: str
    target_step_id: str
    at_risk_tokens: int


@dataclass(frozen=True)
class JointOption:
    demand: DemandState
    retain: tuple[DemandLease, ...] = ()
    release: tuple[str, ...] = ()
    protected_tokens: int = 0
    uncovered_risk_tokens: int = 0

    def trace(self):
        return {"needs": tuple((need.need_id, need.reason) for need in self.demand.needs),
                "remaining_read_ids": self.demand.remaining_read_ids,
                "retain": self.retain, "release": self.release,
                "protected_tokens": self.protected_tokens, "uncovered_risk_tokens": self.uncovered_risk_tokens}


def joint_options(plans, state, *, allocate=None):
    from .plan_policy import _completed_steps

    demand = DemandState.from_outcome(state)
    # Complete plans carry observed producers before the first new action.
    actions = []
    for plan in plans:
        completed = _completed_steps(plan, state)
        step = next(step for step in plan.steps if step.task.step_id not in completed)
        variants = tuple(row for row in step.variants if row.feasible)
        actions.extend((step.task, candidate) for candidate in variants or (None,))
    return {(candidate.candidate_id if candidate is not None else task.step_id):
            (allocate(demand.after(task), task, candidate, actions) if allocate is not None
             else JointOption(demand.after(task)))
            for task, candidate in actions}
