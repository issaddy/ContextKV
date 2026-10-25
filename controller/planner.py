"""Feasible candidates compete on full cost, then their KV working set."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cost_model import CostModel
from .schema import CacheQuoteBatch, NoFeasiblePlan, Plan, PromptCandidate, SessionSnapshot


COST_SELECTION_REASONS = frozenset({"cost_saving", "expected_cost", "hysteresis", "native_kv_tie_break", "context_size_tie_break"})


@dataclass(frozen=True)
class PlannerKnobs:
    strategy: str = "cost"
    # Retained for cost-gap diagnostics and existing configuration records.
    # Per-arm timing error is not a switching cost or a paired regret bound.
    delta_abs: float = 0.0
    delta_rel: float = .05
    z: float = 1.0
    # Net savings already include the measured execution/control costs and an
    # uncertainty margin. A fixed dwell time can veto avoiding native KV waits.
    # Retain an explicit dwell option for reproducing older policies only.
    min_stable_turns: int = 0
    fixed_token_threshold: int = 8192
    fallback_strategy: str = "fixed_threshold"
    concurrency: int = 1
    period: int = 4
    warm_ratio: float = .5
    redundancy_threshold: float = .4

    def __post_init__(self) -> None:
        supported = {"cost", "always_append", "always_rebuild", "fixed_threshold", "stable_layout", "normal_rag", "fixed_period", "warm_idle", "contextpilot", "redundancy_threshold"}
        if self.strategy not in supported or self.fallback_strategy not in supported - {"cost", "warm_idle"}:
            raise ValueError("unknown planner strategy or recursive fallback")
        if min(self.delta_abs,self.delta_rel,self.z,self.min_stable_turns,self.fixed_token_threshold) < 0:
            raise ValueError("planner thresholds must be nonnegative")
        if self.period < 1 or not 0 <= self.warm_ratio <= 1:
            raise ValueError("period must be positive and warm_ratio must be in [0, 1]")
        if not 0 <= self.redundancy_threshold <= 1:
            raise ValueError("redundancy_threshold must be in [0, 1]")


def _fixed(append, rebuild, strategy, knobs, session, quotes=None):
    if strategy in {"always_rebuild", "normal_rag"}:
        return rebuild
    if strategy in {"always_append", "stable_layout", "contextpilot"}:
        return append
    if strategy == "fixed_period":
        return rebuild if session.turns_since_rebuild + 1 >= knobs.period else append
    if strategy == "redundancy_threshold":
        if append.evidence_body_tokens is None or append.nonrequired_evidence_body_tokens is None:
            raise ValueError("redundancy_threshold requires measured evidence body tokens")
        ratio = append.nonrequired_evidence_body_tokens / max(1, append.evidence_body_tokens)
        return rebuild if ratio > knobs.redundancy_threshold else append
    if strategy == "warm_idle":
        if quotes is None:
            return _fixed(append,rebuild,knobs.fallback_strategy,knobs,session)
        quote = next(q for q in quotes.quotes if q.candidate_id == append.candidate_id)
        # GPU-ready only: pending transfers and merely recoverable host copies
        # cannot inflate the observed warm signal.  This is an adaptation of a
        # warm/idle gate, not a reproduction of ReCAP's context graph.
        ratio = quote.ready_gpu_prefix_tokens / max(1, quote.match_limit)
        return append if ratio >= knobs.warm_ratio else rebuild
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
                control_ms: float = 0.0, queue_ms: float = 0.0) -> Plan | NoFeasiblePlan:
    knobs = knobs or PlannerKnobs()
    candidates = tuple(candidates)
    if any(c.snapshot_version != session.version for c in candidates):
        raise ValueError("candidates do not share current snapshot")
    if candidates and len({c.required_blocks for c in candidates}) != 1:
        raise ValueError("candidates must use the same required evidence")
    if any(c.corpus_snapshot_id != session.corpus_snapshot_id for c in candidates):
        raise ValueError("candidates do not share current corpus snapshot")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate identity")
    if any(c.action not in {"APPEND", "REBUILD"} for c in candidates):
        raise ValueError("only APPEND and REBUILD are supported")
    valid = [c for c in candidates if c.feasible and c.required_blocks <= c.visible_blocks]
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
    by_action = {c.action:c for c in valid}
    if len(by_action) != len(valid):
        raise ValueError("duplicate text action")
    append, rebuild = by_action.get("APPEND"), by_action.get("REBUILD")
    if session.must_rebuild or session.force_new_epoch_for_recovery or session.history_sync_pending:
        return Plan(rebuild, reason="forced_rebuild") if rebuild else NoFeasiblePlan("required_rebuild_unavailable")
    if len(valid) == 1:
        return Plan(valid[0], reason="only_feasible_candidate")
    if append is None or rebuild is None:
        raise ValueError("only APPEND and REBUILD are supported")
    if knobs.strategy != "cost":
        chosen = _fixed(append,rebuild,knobs.strategy,knobs,session,quotes)
        reason = "warm_quote_unavailable_fallback" if knobs.strategy == "warm_idle" and quotes is None else knobs.strategy
        return Plan(chosen, reason=reason,quote_seq=quotes.observed_seq if quotes else None)
    if quotes is None:
        return Plan(_fixed(append,rebuild,knobs.fallback_strategy,knobs,session), reason="quote_unavailable_fallback")
    estimates = tuple(model.predict_native(c,by_id[c.candidate_id],session,
                      batch=quotes,concurrency=concurrency or knobs.concurrency,
                      control_ms=control_ms,queue_ms=queue_ms) for c in valid)
    if not all(e.calibrated if e.selection_eligible is None else e.selection_eligible for e in estimates):
        return Plan(_fixed(append,rebuild,knobs.fallback_strategy,knobs,session),
                    predicted=estimates, quote_seq=quotes.observed_seq,
                    reason="uncalibrated_cost_fallback")
    by_estimate = {e.candidate_id:e for e in estimates}
    a, r = by_estimate[append.candidate_id], by_estimate[rebuild.candidate_id]
    saving = a.cost-r.cost
    margin = max(knobs.delta_abs,knobs.delta_rel*min(a.cost,r.cost),knobs.z*model.paired_error(a,r))
    if saving:
        chosen = rebuild if saving > 0 else append
        reason = "cost_saving" if abs(saving) > margin else "expected_cost"
    else:
        chosen, reason = _resource_tie_break(append, rebuild, quotes, by_id, by_estimate)
    if chosen is rebuild and session.turns_since_rebuild < knobs.min_stable_turns:
        chosen, reason = append, "hysteresis"  # explicit legacy dwell only
    return Plan(chosen, predicted=estimates, quote_seq=quotes.observed_seq,
                reason=reason,
                switch_margin_ms=margin, predicted_saving_ms=saving)
