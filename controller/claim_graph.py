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
        if claim.kind == "derived" and not claim.parent_claim_ids:
            raise ValueError("a derived claim requires its inputs")
        if claim.kind == "extract" and not claim.source_spans:
            raise ValueError("an extracted claim requires original spans")
        self.claims[claim.claim_id] = claim

    def leaf_claims(self, claim_id):
        claim = self.claims[claim_id]
        if not claim.parent_claim_ids:
            return (claim,)
        leaves = {claim.claim_id: claim} if claim.kind == "extract" else {}
        for parent in claim.parent_claim_ids:
            for leaf in self.leaf_claims(parent):
                leaves[leaf.claim_id] = leaf
        return tuple(leaves.values())

    def source_spans(self, claim_id):
        spans = {}
        for leaf in self.leaf_claims(claim_id):
            for span in leaf.source_spans:
                spans[span.span_id] = span
        return tuple(spans.values())

    def answer_plan(self, state, disposition="answer", results=()):
        reductions = tuple(result.reduction for result in results if result.reduction is not None)
        active = tuple(reduction for index, reduction in enumerate(reductions)
            if not any(reduction.need_ids == later.need_ids and reduction.operation == later.operation
                and set(reduction.basis_claim_ids) <= set(later.basis_claim_ids)
                for later in reductions[index+1:]))
        managed = {identity for reduction in reductions
                   for identity in (*reduction.basis_claim_ids, *reduction.frontier_claim_ids)}
        frontier = {identity for reduction in active for identity in reduction.frontier_claim_ids}
        completed_claims = set()
        todo = [identity for reduction in reductions for identity in reduction.frontier_claim_ids]
        while todo:
            identity = todo.pop()
            if identity not in completed_claims:
                completed_claims.add(identity)
                todo.extend(self.claims[identity].parent_claim_ids)
        pending = {claim.claim_id for result in results for claim in result.claims
                   if claim.kind == "derived" and claim.claim_id not in completed_claims}
        parents = {parent for claim in self.claims.values() if claim.claim_id not in pending
                   for parent in claim.parent_claim_ids}
        points = []
        for need in state.needs:
            selected = tuple(claim for claim in self.claims.values()
                if claim.need_id == need.need_id and (claim.claim_id in frontier
                    or claim.claim_id not in parents | managed | pending))
            spans = {span.span_id: span for claim in selected
                     for span in self.source_spans(claim.claim_id)}
            points.append(AnswerPoint(need.need_id, tuple(c.claim_id for c in selected),
                                      tuple(spans.values())))
        return AnswerPlan(tuple(points), disposition)


def claim_frontier(inputs, outputs):
    """Retain independent conclusions and inputs not consumed by a conclusion."""
    claims = (*inputs, *outputs)
    parents = {parent for claim in claims for parent in claim.parent_claim_ids}
    return tuple(claim for claim in claims if claim.claim_id not in parents)
