"""Project current evidence obligations onto the same finite native request set."""
from dataclasses import dataclass, replace

from .context_budget import answer_evidence_gaps, required_source_units
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
        predecessors = {identity for need in plan.task_state.needs if need.need_id in gaps
                        for identity in need.predecessor_need_ids}
        needs = []
        for need in plan.task_state.needs:
            if need.need_id not in gaps:
                continue
            full = plan.coverage_scope == "all_supplied_documents" or need.scope == "all_supplied_documents"
            sources = tuple(dict.fromkeys(span for unit in required_source_units(plan, need)
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
    boundary_tokens: int
    at_risk_tokens: int
    page_size: int
    cache_scope_hash: str
    ttl_seconds: float
    action: str = "new"
    consumer_arrival_ms: float | None = None


@dataclass(frozen=True)
class JointOption:
    demand: DemandState
    retain: tuple[DemandLease, ...] = ()
    release: tuple[str, ...] = ()
    protected_tokens: int = 0
    uncovered_risk_tokens: int = 0
    retention_evidence: str | None = None
    keep: tuple[str, ...] = ()
    arrival_basis: str | None = None
    retention_allocation_value_ms: float | None = None
    retention_control_ms: float | None = 0.0

    def trace(self):
        return {"needs": tuple((need.need_id, need.reason) for need in self.demand.needs),
                "remaining_read_ids": self.demand.remaining_read_ids,
                "retain": self.retain, "release": self.release, "keep": self.keep,
                "protected_tokens": self.protected_tokens, "uncovered_risk_tokens": self.uncovered_risk_tokens,
                "retention_evidence": self.retention_evidence,
                "arrival_basis": self.arrival_basis,
                "retention_allocation_value_ms": self.retention_allocation_value_ms,
                "retention_control_ms": self.retention_control_ms}


def joint_options(plans, state, *, allocate=None):
    from .plan_policy import _completed_steps

    # Project demand and consumers from each candidate's remaining catalogue.
    options = {}
    for plan in plans:
        projected = replace(state, plan=plan.context_plan)
        completed = _completed_steps(plan, projected)
        remaining = tuple(step for step in plan.steps if step.task.step_id not in completed)
        if not remaining:
            continue
        first = remaining[0]
        variants = tuple(row for row in first.variants if row.feasible)
        actions = tuple((first.task, candidate) for candidate in variants or (None,))
        consumers = tuple((step.task, candidate) for step in plan.route_steps
            if step.task.step_id not in completed
            for candidate in tuple(row for row in step.variants if row.feasible) or (None,))
        demand = DemandState.from_outcome(projected)
        for task, candidate in actions:
            key = candidate.candidate_id if candidate is not None else task.step_id
            options[key] = (allocate(demand.after(task), task, candidate, consumers)
                if allocate is not None else JointOption(demand.after(task)))
    return options
