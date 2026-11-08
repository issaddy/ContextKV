"""Claim lineage connects derived results to the exact original source spans."""
from __future__ import annotations

from .context_schema import AnswerPlan, AnswerPoint, EvidenceClaim


class ClaimGraph:
    def __init__(self, claims=()):
        self.claims: dict[str, EvidenceClaim] = {}
        for claim in claims:
            self.add(claim)

    def add(self, claim: EvidenceClaim):
        if claim.claim_id in self.claims:
            if self.claims[claim.claim_id] != claim:
                raise ValueError(f"claim ID reused: {claim.claim_id}")
            return
        for parent in claim.parent_claim_ids:
            self.claims[parent]
        if claim.kind == "derived" and not claim.parent_claim_ids and not claim.source_spans:
            raise ValueError("a derived claim requires its inputs")
        if claim.kind == "extract" and not claim.source_spans:
            raise ValueError("an extracted claim requires original spans")
        self.claims[claim.claim_id] = claim

    def leaf_claims(self, claim_id):
        leaves, visited, todo = [], set(), [claim_id]
        while todo:
            identity = todo.pop()
            if identity in visited:
                continue
            visited.add(identity)
            claim = self.claims[identity]
            if claim.kind == "extract" or not claim.parent_claim_ids:
                leaves.append(claim)
            todo.extend(reversed(claim.parent_claim_ids))
        return tuple(leaves)

    def source_spans(self, claim_id):
        spans = {}
        for leaf in self.leaf_claims(claim_id):
            for span in leaf.source_spans:
                spans[span.span_id] = span
        for span in self.claims[claim_id].source_spans:
            spans.setdefault(span.span_id, span)
        return tuple(spans.values())

    def answer_plan(self, state, disposition="answer", results=()):
        reductions = tuple(result.reduction for result in results if result.reduction is not None)
        completed_claims = set()
        todo = [identity for reduction in reductions for identity in reduction.frontier_claim_ids]
        while todo:
            identity = todo.pop()
            if identity not in completed_claims:
                completed_claims.add(identity)
                todo.extend(self.claims[identity].parent_claim_ids)
        pending = {claim.claim_id for result in results for claim in result.claims
                   if claim.kind == "derived" and claim.claim_id not in completed_claims}
        points = []
        for need in state.needs:
            need_reductions = tuple(item for item in reductions if need.need_id in item.need_ids)
            managed = {identity for item in need_reductions
                       for identity in (*item.basis_claim_ids, *item.frontier_claim_ids)}
            active = tuple(item for index, item in enumerate(need_reductions)
                if not any(item.operation == later.operation
                           and set(item.basis_claim_ids) <= set(later.basis_claim_ids)
                           for later in need_reductions[index+1:]))
            frontier = {identity for item in active
                        for identity in item.frontier_claim_ids}
            current = tuple(claim for claim in self.claims.values()
                if claim.serves(need.need_id) and (claim.claim_id in frontier
                    or claim.claim_id not in managed | pending))
            selected = claim_frontier(current, ())
            spans = {span.span_id: span for claim in selected
                     for span in self.source_spans(claim.claim_id)}
            points.append(AnswerPoint(need.need_id, tuple(c.claim_id for c in selected),
                                      tuple(spans.values())))
        return AnswerPlan(tuple(points), disposition)

    def conclusion_table(self, answer_plan, references):
        """Encode every answer root and its complete source path with shared scopes."""
        by_key = {key: reference for reference, key in references}
        roots = tuple(dict.fromkeys(identity for point in answer_plan.points
                                    for identity in point.claim_ids))
        aliases = {identity: f"c{index}" for index, identity in enumerate(roots)}
        spans, paths, scopes, facts = {}, {}, {}, []
        for identity in roots:
            claim = self.claims[identity]
            source_ids = []
            for span in self.source_spans(identity):
                coordinates = by_key[span.source_key], span.start, span.end
                source_ids.append(spans.setdefault(coordinates, f"s{len(spans)}"))
            path = tuple(dict.fromkeys(source_ids))
            path_id = paths.setdefault(path, f"p{len(paths)}")
            scope = (claim.need_id, claim.conditions, claim.version, claim.operation, claim.kind,
                     claim.shared_need_ids)
            scope_id = scopes.setdefault(scope, f"g{len(scopes)}")
            facts.append([aliases[identity], scope_id, claim.text, claim.entity,
                          claim.relation, claim.value, claim.unit, path_id])
        return {"fact_columns": ["id", "scope", "text", "entity", "relation", "value", "unit", "sources"],
            "scope_columns": ["need_id", "conditions", "version", "operation", "kind", "shared_need_ids"],
            "scopes": {alias: value for value, alias in scopes.items()},
            "source_columns": ["citation_id", "start", "end"],
            "sources": {alias: value for value, alias in spans.items()},
            "source_paths": {alias: value for value, alias in paths.items()},
            "facts": facts,
            "answer_plan": {"disposition": answer_plan.disposition,
                "points": [{"need_id": point.need_id,
                    "claim_ids": [aliases[identity] for identity in point.claim_ids]}
                    for point in answer_plan.points]}}


def claim_frontier(inputs, outputs):
    """Retain independent conclusions and inputs not consumed by a conclusion."""
    claims = (*inputs, *outputs)
    parents = {parent for claim in claims if claim.kind == "derived" for parent in claim.parent_claim_ids}
    return tuple(claim for claim in claims if claim.claim_id not in parents)
