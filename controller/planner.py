"""Feasible candidates compete on full cost, then their KV working set."""
from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Iterable, Mapping

from .cost_model import CostModel
from .schema import CacheQuoteBatch, NoFeasiblePlan, Plan, PromptCandidate, SessionSnapshot


COST_SELECTION_REASONS = frozenset({"cost_saving", "expected_cost", "native_kv_tie_break", "context_size_tie_break", "quality_complete_cost"})


@dataclass(frozen=True)
class PlannerKnobs:
    strategy: str = "cost"
    fixed_token_threshold: int = 8192
    concurrency: int = 1
    # Old manifests carried these options even for current strategies.
    period: InitVar[int | None] = None
    warm_ratio: InitVar[float | None] = None
    redundancy_threshold: InitVar[float | None] = None
    lambda_time: float = 1.0
    lambda_gpu: float = 0.0
    lambda_host: float = 0.0
    max_complete_ms: float | None = None

    def __post_init__(self, period, warm_ratio, redundancy_threshold) -> None:
        supported = {"cost", "always_append", "always_rebuild", "fixed_threshold"}
        if self.strategy not in supported:
            raise ValueError("unknown planner strategy")
        if self.fixed_token_threshold < 0:
            raise ValueError("planner thresholds must be nonnegative")


def _fixed(append, rebuild, strategy, knobs):
    if strategy == "always_rebuild":
        return rebuild
    if strategy == "always_append":
        return append
    return rebuild if len(append.input_ids) > knobs.fixed_token_threshold else append


def _resource_tie_break(append, rebuild, quotes, observed, estimates):
    """Break an equal-cost tie; never reserve or credit future retirement.

    For a ready GPU prefix D, pinning its evictable part I makes the native
    FULL gate's candidate-dependent input demand n-D+I. The common requested
    output budget/page overhead cancels between A/R. A resident prefix is not
    free capacity unless another request already holds its hard lock.
    With a queue, host gap, pending transfer or older quote, rank logical input
    lengths instead; neither rank claims future physical occupancy/admission.
    """
    use_native_pin = quotes.native_waiting_requests == () and all(
        q.evictable_gpu_prefix_tokens is not None
        and not (q.host_only_load_tokens or q.inflight_h2d_tokens or q.inflight_d2h_tokens)
        for q in (observed[append.candidate_id], observed[rebuild.candidate_id]))

    def rank(candidate):
        n = len(candidate.input_ids)
        q = observed[candidate.candidate_id]
        demand = n-q.ready_gpu_prefix_tokens+q.evictable_gpu_prefix_tokens if use_native_pin else n
        # With identical resource/cost ranks, retaining the current epoch avoids
        # an unnecessary lifecycle operation. No action gets a general margin.
        return demand, n, estimates[candidate.candidate_id].cost, candidate.action == "REBUILD"

    return (min((append, rebuild), key=rank),
            "native_kv_tie_break" if use_native_pin else "context_size_tie_break")


def choose_plan(candidates: Iterable[PromptCandidate], quotes: CacheQuoteBatch | None,
                model: CostModel, session: SessionSnapshot,
                knobs: PlannerKnobs | None = None, *, concurrency: int | None = None,
                control_ms: float | Mapping[str, float] = 0.0,
                queue_ms: float = 0.0) -> Plan | NoFeasiblePlan:
    """Select compiled candidates using fixed rules or an actual cost quote batch."""
    knobs = knobs or PlannerKnobs()
    candidates = tuple(candidates)
    # build_candidates owns snapshot, evidence and action construction.
    valid = [c for c in candidates if c.feasible]
    if not valid:
        return NoFeasiblePlan()
    if quotes is not None:
        by_id = {q.candidate_id:q for q in quotes.quotes}
        if len(by_id) != len(quotes.quotes) or any(c.candidate_id not in by_id for c in valid):
            raise ValueError("missing or duplicate candidate quotes")
        for candidate in valid:
            quote = by_id[candidate.candidate_id]
            if quote.input_hash != candidate.input_hash or quote.input_len != len(candidate.input_ids):
                raise ValueError("quote does not describe candidate input")
            if candidate.fingerprint != quotes.fingerprint:
                raise ValueError("quote/generation fingerprint mismatch")
    forced = session.must_rebuild or session.force_new_epoch_for_recovery or session.history_sync_pending
    if forced:
        valid = [c for c in valid if c.action == "REBUILD"]
        if not valid:
            return NoFeasiblePlan("required_rebuild_unavailable")
    if len(valid) == 1:
        return Plan(valid[0], reason="only_feasible_candidate")
    if quotes is None and knobs.strategy == "cost":
        raise ValueError("cost comparison requires a quote batch")
    def fixed_choice(strategy):
        groups = {}
        for candidate in valid:
            group = (candidate.context_plan_id, candidate.evidence_set_id,
                     candidate.working_history_id, candidate.layout_id)
            groups.setdefault(group, {})[candidate.action] = candidate
        selected = []
        for actions in groups.values():
            if len(actions) == 1:
                selected.append(next(iter(actions.values())))
            else:
                selected.append(_fixed(actions.get("APPEND"), actions.get("REBUILD"),
                                       strategy, knobs))
        return min(selected, key=lambda c: (-c.quality_prediction, len(c.input_ids), c.candidate_id))

    if knobs.strategy != "cost":
        chosen = fixed_choice("always_rebuild" if forced else knobs.strategy)
        return Plan(chosen, reason="forced_rebuild" if forced else knobs.strategy,
                    quote_seq=quotes.observed_seq if quotes else None)
    estimates = tuple(model.predict_native(c,by_id[c.candidate_id],session,
                      batch=quotes,concurrency=concurrency or knobs.concurrency,
                      control_ms=(control_ms[c.candidate_id] if c.candidate_id in control_ms else
                                  control_ms[c.action]) if isinstance(control_ms, Mapping) else control_ms,
                      queue_ms=queue_ms) for c in valid)
    costed = [candidate for candidate, estimate in zip(valid, estimates, strict=True)
              if (estimate.calibrated if estimate.selection_eligible is None else estimate.selection_eligible)]
    if not costed:
        return NoFeasiblePlan("uncalibrated_cost")
    by_estimate = {e.candidate_id:e for e in estimates}
    eligible = [c for c in costed if knobs.max_complete_ms is None or
                by_estimate[c.candidate_id].cost <= knobs.max_complete_ms]
    if not eligible:
        return NoFeasiblePlan("complete_latency_budget")

    def objective(candidate):
        return (candidate.quality_prediction - knobs.lambda_time*by_estimate[candidate.candidate_id].cost
                - knobs.lambda_gpu*candidate.predicted_gpu_gib_ms
                - knobs.lambda_host*candidate.predicted_host_gib_ms)

    best = max(objective(c) for c in eligible)
    tied = [c for c in eligible if objective(c) == best]
    chosen = min(tied, key=lambda c: (len(c.input_ids), c.action == "REBUILD", c.candidate_id))
    reason = "quality_complete_cost" if any(c.context_plan_id for c in valid) else "cost_saving"
    if len(tied) == 2 and {c.action for c in tied} == {"APPEND", "REBUILD"}:
        append = next(c for c in tied if c.action == "APPEND")
        rebuild = next(c for c in tied if c.action == "REBUILD")
        chosen, reason = _resource_tie_break(append, rebuild, quotes, by_id, by_estimate)
    saving = max(by_estimate[c.candidate_id].cost for c in costed) - by_estimate[chosen.candidate_id].cost
    if len(costed) == 2 and {c.action for c in costed} == {"APPEND", "REBUILD"}:
        saving = (next(by_estimate[c.candidate_id].cost for c in costed if c.action == "APPEND")
                  - next(by_estimate[c.candidate_id].cost for c in costed if c.action == "REBUILD"))
    return Plan(chosen, predicted=estimates, quote_seq=quotes.observed_seq,
                reason=reason,
                predicted_saving_ms=saving)
