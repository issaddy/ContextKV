"""Learned applicability, single/group support and evidence-set utility.

The scoring schedule is independent of the model implementation. Production
loading uses the trained BGE cross encoder and graph heads; tests can provide a
backend with explicit predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import time
from typing import Callable, Protocol

from controller.context_schema import (EvidencePool, EvidenceUnit, Need, SourceSpan,
    SupportBundle, SupportScores, TaskState)


@dataclass(frozen=True)
class SupportInput:
    task_id: str
    need_id: str
    query: str
    units: tuple[EvidenceUnit, ...]
    dependencies: tuple[EvidenceUnit, ...] = ()


@dataclass(frozen=True)
class SupportPrediction:
    applicability: float
    direct_support: float
    joint_support: float
    bridge_value: float
    disposition: str
    source_spans: tuple[SourceSpan, ...] = ()
    missing_relation: str = ""


@dataclass(frozen=True)
class SetPrediction:
    quality: float
    disposition: str = "answer"


@dataclass(frozen=True)
class SupportProfile:
    single_candidates_per_need: int = 24
    bridge_candidates_per_need: int = 6
    triple_seed_pairs: int = 2
    triple_successors_per_pair: int = 3


class SupportBackend(Protocol):
    def predict(self, examples: tuple[SupportInput, ...]) -> tuple[SupportPrediction | None, ...]: ...
    def predict_set(self, state: TaskState, units: tuple[EvidenceUnit, ...],
                    supports: tuple[SupportBundle, ...]) -> SetPrediction: ...


def query_view(state: TaskState, need: Need, resolve: Callable, *, open_needs=()) -> str:
    history = [{"turn_id": ref.turn_id, "speaker": ref.speaker,
                "start": ref.start, "end": ref.end, "text": resolve(ref)}
               for ref in state.working_history]
    conditions = [{"id": c.condition_id, "relation": c.relation_type,
                   "key": resolve(c.key_span) if c.key_span else "",
                   "values": [resolve(ref) for ref in c.value_spans],
                   "predicate": resolve(c.predicate_span) if c.predicate_span else ""}
                  for c in state.conditions if c.condition_id in need.condition_ids]
    payload = {"question": state.raw_question, "need_id": need.need_id,
        "operation": need.operation, "scope": need.scope, "anchors": [resolve(ref) for ref in need.anchor_spans],
        "subjects": [resolve(ref) for ref in need.subject_refs], "working_history": history,
        "conditions": conditions, "version_intent": {
            "mode": state.version_intent.mode,
            "document_families": state.version_intent.document_families,
            "version_ids": state.version_intent.version_ids,
            "resolved_version_ids": state.version_intent.resolved_version_ids,
            "temporal_relation": state.version_intent.temporal_relation,
            "time_refs": [resolve(ref) for ref in state.version_intent.time_refs],
            "source_turns": state.version_intent.source_turns}}
    pending = [item for item in open_needs if item.need_id == need.need_id]
    if pending:
        payload["open_needs"] = [{"need_id": item.need_id, "missing_relation": item.missing_relation,
            "bound_entities": item.bound_entities, "temporal_scope": item.temporal_scope,
            "parent_claim_ids": item.parent_claim_ids} for item in pending]
    return json.dumps(payload, ensure_ascii=False)


def _dependencies(units, by_id):
    selected = {u.unit_id for u in units}
    pending = [dependency for unit in units for dependency in unit.dependencies]
    result = []
    while pending:
        unit_id = pending.pop(0)
        if unit_id in selected:
            continue
        unit = by_id[unit_id]
        selected.add(unit_id)
        result.append(unit)
        pending.extend(unit.dependencies)
    return tuple(result)


class SupportModel:
    def __init__(self, backend: SupportBackend, *, history_resolver: Callable,
                 profile: SupportProfile = SupportProfile()):
        self.backend = backend
        self.resolve = history_resolver
        self.profile = profile
        self.last_schedule = ()

    @classmethod
    def from_pretrained(cls, path, *, history_resolver, device="cpu", batch_size=8,
                        profile=SupportProfile()):
        from training.support_model import TrainedSupportBackend
        return cls(TrainedSupportBackend.load(path, device=device, batch_size=batch_size),
                   history_resolver=history_resolver, profile=profile)

    def score_support(self, state: TaskState, pool: EvidencePool, *, open_needs=()) -> SupportScores:
        started = time.perf_counter()
        by_id = {unit.unit_id: unit for unit in pool.units}
        singles, groups, schedule = [], [], []
        for need in state.needs:
            query = query_view(state, need, self.resolve, open_needs=open_needs)
            ranked = sorted(pool.units, key=lambda u: (
                need.need_id not in u.need_ids, -u.retrieval_score, u.unit_id))
            selected = ranked[:self.profile.single_candidates_per_need]

            def evaluate(batches):
                examples = tuple(SupportInput(state.task_id, need.need_id, query,
                    tuple(batch), _dependencies(batch, by_id)) for batch in batches)
                predictions = self.backend.predict(examples)
                result = []
                for example, prediction in zip(examples, predictions, strict=True):
                    if prediction is None:
                        continue
                    ids = tuple(unit.unit_id for unit in example.units)
                    bundle_id = json.dumps([state.task_id, need.need_id, ids], separators=(",", ":"))
                    support = (prediction.direct_support if len(ids) == 1
                               else prediction.joint_support)
                    missing = prediction.missing_relation
                    if not missing and prediction.disposition == "partial":
                        missing = " ".join(self.resolve(ref) for ref in need.anchor_spans) or state.raw_question
                    result.append(SupportBundle(bundle_id, need.need_id, ids,
                        prediction.applicability, support, prediction.source_spans,
                        prediction.disposition, prediction.bridge_value,
                        missing))
                schedule.append({"need_id": need.need_id,
                    "arity": len(examples[0].units) if examples else 0,
                    "count": len(examples), "scored": len(result)})
                return result

            single_scores = evaluate([(unit,) for unit in selected])
            singles.extend(single_scores)
            if any(score.disposition == "complete" and score.applicability >= .5
                   for score in single_scores):
                continue
            bridges = sorted(single_scores, key=lambda score: (
                -score.applicability * max(score.bridge_value, score.joint_support), score.unit_ids))[
                    :self.profile.bridge_candidates_per_need]
            bridge_units = [by_id[score.unit_ids[0]] for score in bridges]
            pair_scores = evaluate(list(combinations(bridge_units, 2)))
            groups.extend(pair_scores)
            expandable = [score for score in pair_scores
                          if need.operation == "infer" or score.missing_relation]
            seed_pairs = sorted(expandable, key=lambda score: (
                -score.applicability * score.joint_support, score.unit_ids))[
                    :self.profile.triple_seed_pairs]
            triples, seen = [], set()
            for pair in seed_pairs:
                successors = [unit for unit in selected if unit.unit_id not in pair.unit_ids]
                for unit in successors[:self.profile.triple_successors_per_pair]:
                    ids = tuple(sorted((*pair.unit_ids, unit.unit_id)))
                    if ids not in seen:
                        seen.add(ids)
                        triples.append(tuple(by_id[uid] for uid in ids))
            groups.extend(evaluate(triples))
        self.last_schedule = tuple(schedule)
        return SupportScores(tuple(singles), tuple(groups), (time.perf_counter()-started)*1000)

    def predict_set(self, state, units, supports):
        return self.backend.predict_set(state, units, supports)

    def predict_candidate(self, state, units, supports, *, action, layout,
                          old_body_tokens, history_tokens, input_tokens, output_budget):
        context = (float(action == "APPEND"), float(layout == "existing"),
            float(layout == "need_order"), float(layout == "stable_evidence"),
            old_body_tokens/32768, history_tokens/8192, input_tokens/32768, output_budget/8192)
        return self.backend.predict_set(state, units, supports, context=context)

    def release_task(self, task_id):
        self.backend.release_task(task_id)

    def close(self):
        self.backend.close()

    @property
    def manifest(self):
        return {"backend": self.backend.manifest,
                "schedule": self.profile.__dict__, "last_schedule": self.last_schedule}
