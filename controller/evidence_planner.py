"""Beam search over learned single/pair/triple actions with source closure."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import time

from .context_schema import (ContextBudget, EvidencePool, EvidenceSet, EvidenceUnit,
    OpenNeed, SupportScores, TaskState)


@dataclass(frozen=True)
class SearchProfile:
    beam_width: int = 4
    max_actions: int = 12
    retained_sets: int = 3
    gap_search_rounds: int = 1


def render_unit_union(units: tuple[EvidenceUnit, ...]) -> str:
    """Render each identical body once, retaining all source/span identities."""
    grouped = {}
    for unit in units:
        grouped.setdefault(unit.text, []).append(unit)
    rows = []
    for text, aliases in grouped.items():
        metadata = [{"unit_id": unit.unit_id, "heading": unit.heading,
            "sources": [{"span_id": span.span_id,
                         "corpus": span.source_key.corpus_id,
                         "document": span.source_key.document_id,
                         "passage": span.source_key.passage_id,
                         "version": span.source_key.version,
                         "start": span.start, "end": span.end}
                        for span in unit.source_spans]} for unit in aliases]
        rows.append(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))+"\n"+text)
    return "\n\n".join(rows)


def close_dependencies(selected, by_id):
    pending = list(selected)
    closed, seen = [], set()
    while pending:
        unit_id = pending.pop(0)
        if unit_id in seen:
            continue
        unit = by_id[unit_id]
        seen.add(unit_id)
        closed.append(unit)
        pending.extend(unit.dependencies)
    return tuple(closed)


def _passages(units):
    return {pid for unit in units for pid in (unit.original_passage_ids or tuple(
        json.dumps([s.source_key.corpus_id, s.source_key.document_id,
                    s.source_key.passage_id, s.source_key.version]) for s in unit.source_spans))}


def unresolved_needs(state: TaskState, evidence_set: EvidenceSet):
    return tuple(OpenNeed(need.need_id,
                         (support.missing_relation if support else "") or need.operation,
                         temporal_scope=state.version_intent.mode)
                 for need in state.needs
                 for support in [next((s for s in evidence_set.need_support
                                       if s.need_id == need.need_id), None)]
                 if support is None or support.disposition != "complete")


class BudgetedEvidencePlanner:
    def __init__(self, tokenizer, utility_model, *, profile=SearchProfile(),
                 measure_context=None, measure_body=None, passage_exclusions=frozenset()):
        self.tokenizer = tokenizer
        self.utility_model = utility_model
        self.profile = profile
        self.measure_context = measure_context
        self.measure_body = measure_body
        self.passage_exclusions = frozenset(passage_exclusions)
        self.trace = ()

    def search_sets(self, state: TaskState, pool: EvidencePool, scores: SupportScores,
                    budget: ContextBudget) -> tuple[EvidenceSet, ...]:
        started = time.perf_counter()
        by_id = {unit.unit_id: unit for unit in pool.units}
        supports = (*scores.singles, *scores.bundles)
        actions = tuple(dict.fromkeys(s.unit_ids for s in supports))
        frontier = [frozenset()]
        visited, retained, trace = set(), {}, []
        candidate_cache = {}

        def evaluate(ids):
            ordered = tuple(sorted(ids, key=lambda uid: (-by_id[uid].retrieval_score, uid)))
            units = close_dependencies(ordered, by_id)
            closed_ids = frozenset(u.unit_id for u in units)
            if closed_ids in candidate_cache:
                return candidate_cache[closed_ids]
            body_tokens = (self.measure_body(units) if self.measure_body is not None else
                           len(self.tokenizer.encode(render_unit_union(units), add_special_tokens=False)))
            if body_tokens > budget.body_tokens:
                return None
            if budget.max_passages is not None and len(_passages(units)-self.passage_exclusions) > budget.max_passages:
                return None
            if self.measure_context is not None:
                input_tokens = self.measure_context(state, units)
                if input_tokens + budget.output_tokens > budget.window_tokens:
                    return None
            available = tuple(s for s in supports if set(s.unit_ids) <= closed_ids)
            best = tuple(max((s for s in available if s.need_id == need.need_id),
                            key=lambda s: s.joint_support*s.applicability)
                         for need in state.needs if any(s.need_id == need.need_id for s in available))
            prediction = self.utility_model.predict_set(state, units, available)
            set_id = json.dumps([state.task_id, sorted(closed_ids)], ensure_ascii=False,
                                separators=(",", ":"))
            source_keys = {span.source_key for unit in units for span in unit.source_spans}
            blocks = tuple(block for block in pool.source_blocks if block.key in source_keys)
            candidate = EvidenceSet(set_id, units, tuple(u.unit_id for u in units if u.unit_id not in ids),
                best, prediction.quality, body_tokens, prediction.disposition,
                source_blocks=blocks)
            candidate_cache[closed_ids] = candidate
            return candidate

        for depth in range(self.profile.max_actions):
            proposals = {}
            for ids in frontier:
                for action in actions:
                    expanded = ids | frozenset(action)
                    if expanded == ids or expanded in visited:
                        continue
                    visited.add(expanded)
                    candidate = evaluate(expanded)
                    if candidate is not None:
                        closed = frozenset(u.unit_id for u in candidate.selected_units)
                        proposals[closed] = candidate
                        retained[closed] = candidate
            ordered = sorted(proposals.items(), key=lambda item: (
                -item[1].quality_prediction, item[1].token_cost, item[1].set_id))
            trace.append({"depth": depth, "proposed": len(proposals),
                          "frontier": [c.set_id for _, c in ordered[:self.profile.beam_width]]})
            frontier = [ids for ids, _ in ordered[:self.profile.beam_width]]
            if not frontier:
                break
        if not retained:
            prediction = self.utility_model.predict_set(state, (), ())
            retained[frozenset()] = EvidenceSet(json.dumps([state.task_id, []]), (),
                quality_prediction=prediction.quality, disposition=prediction.disposition)
        # Keep the quality/length frontier, then add distinct high quality sets
        # when dominance leaves fewer than the requested number of candidates.
        ranked = sorted(retained.values(), key=lambda c: (
            -c.quality_prediction, c.token_cost, c.set_id))
        pareto = [candidate for candidate in ranked if not any(
            other.quality_prediction >= candidate.quality_prediction
            and other.token_cost <= candidate.token_cost
            and (other.quality_prediction > candidate.quality_prediction
                 or other.token_cost < candidate.token_cost) for other in ranked)]
        ordered = pareto + [c for c in ranked if c not in pareto]
        elapsed = (time.perf_counter()-started)*1000
        self.trace = tuple(trace)
        return tuple(replace(c, search_ms=elapsed) for c in ordered[:min(3, self.profile.retained_sets)])

    def search_with_gaps(self, state, pool, scores, budget, *, score_model, retrieve_gap):
        sets = self.search_sets(state, pool, scores, budget)
        traces = [*pool.trace]
        for round_no in range(self.profile.gap_search_rounds):
            gaps = unresolved_needs(state, sets[0])
            if not gaps:
                break
            additions = tuple(unit for gap in gaps for unit in retrieve_gap(state, gap))
            merged = {u.unit_id: u for u in (*pool.units, *additions)}
            traces.append({"gap_round": round_no, "open_needs": [g.need_id for g in gaps],
                           "added_units": [u.unit_id for u in additions]})
            pool = replace(pool, units=tuple(merged.values()), trace=tuple(traces))
            scores = score_model.score_support(state, pool)
            sets = self.search_sets(state, pool, scores, budget)
        return pool, scores, sets


def search_sets(state, pool, scores, budget, *, tokenizer, utility_model,
                profile=SearchProfile(), measure_context=None, measure_body=None,
                passage_exclusions=frozenset()):
    return BudgetedEvidencePlanner(tokenizer, utility_model, profile=profile,
        measure_context=measure_context, measure_body=measure_body,
        passage_exclusions=passage_exclusions).search_sets(state, pool, scores, budget)


render_selection = render_unit_union
