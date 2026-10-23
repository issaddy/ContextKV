"""Hard constraints precede measured-cost switching and hysteresis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cost_model import CostModel
from .schema import CacheQuoteBatch, NoFeasiblePlan, Plan, PromptCandidate, SessionSnapshot


@dataclass(frozen=True)
class PlannerKnobs:
    strategy: str = "cost"
    delta_abs: float = 0.0
    delta_rel: float = .05
    z: float = 1.0
    min_stable_turns: int = 2
    fixed_token_threshold: int = 8192
    fallback_strategy: str = "fixed_threshold"
    concurrency: int = 1
    period: int = 4
    warm_ratio: float = .5

    def __post_init__(self) -> None:
        supported = {"cost", "always_append", "always_rebuild", "fixed_threshold", "stable_layout", "normal_rag", "fixed_period", "warm_idle", "contextpilot"}
        if self.strategy not in supported or self.fallback_strategy not in supported - {"cost", "warm_idle"}:
            raise ValueError("unknown planner strategy or recursive fallback")
        if min(self.delta_abs,self.delta_rel,self.z,self.min_stable_turns,self.fixed_token_threshold) < 0:
            raise ValueError("planner thresholds must be nonnegative")
        if self.period < 1 or not 0 <= self.warm_ratio <= 1:
            raise ValueError("period must be positive and warm_ratio must be in [0, 1]")


def _fixed(append, rebuild, strategy, knobs, session, quotes=None):
    if strategy in {"always_rebuild", "normal_rag"}:
        return rebuild
    if strategy in {"always_append", "stable_layout", "contextpilot"}:
        return append
    if strategy == "fixed_period":
        return rebuild if session.turns_since_rebuild + 1 >= knobs.period else append
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
    if not all(e.calibrated for e in estimates):
        return Plan(_fixed(append,rebuild,knobs.fallback_strategy,knobs,session),
                    predicted=estimates, quote_seq=quotes.observed_seq,
                    reason="uncalibrated_cost_fallback")
    by_estimate = {e.candidate_id:e for e in estimates}
    a, r = by_estimate[append.candidate_id], by_estimate[rebuild.candidate_id]
    saving = a.cost-r.cost
    margin = max(knobs.delta_abs,knobs.delta_rel*a.cost,knobs.z*model.paired_error(a,r))
    stable = session.turns_since_rebuild >= knobs.min_stable_turns
    chosen = rebuild if stable and saving > margin else append
    return Plan(chosen, predicted=estimates, quote_seq=quotes.observed_seq,
                reason="cost_saving" if chosen is rebuild else "hysteresis",
                switch_margin_ms=margin, predicted_saving_ms=saving)
