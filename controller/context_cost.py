"""Native quotes for the known READ inputs of competing semantic plans."""
from __future__ import annotations

from dataclasses import replace

from .reading import step_generation_kind, step_messages
from .schema import PromptCandidate
from .serialization import dumps


async def quote_read_costs(plans, snapshot, template, engine, cost_model, *, completed_steps=()):
    """Quote pending READs whose complete inputs are already known."""
    candidates, ownership, unquoted = [], {}, []
    completed = {record.step_id for record in completed_steps}
    claims = {claim.claim_id: claim for record in completed_steps for claim in record.result.claims}
    for plan in plans:
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        for task in plan.steps:
            if task.kind != "READ" or task.step_id in completed:
                continue
            missing = tuple(cid for cid in task.input_claim_ids if cid not in claims)
            if missing:
                unquoted.append({"plan_id": plan.plan_id, "step_id": task.step_id,
                                 "unknown_parent_claim_ids": missing})
                continue
            source = tuple(units[uid] for uid in task.input_unit_ids)
            messages = step_messages(plan, task, source,
                tuple(claims[cid] for cid in task.input_claim_ids), snapshot=snapshot)
            rendered = template.render(messages)
            candidate_id = dumps([plan.plan_id, task.step_id, "native_read_quote"])
            candidate = PromptCandidate(candidate_id, "REBUILD", snapshot.version,
                messages, rendered.input_ids, rendered.input_hash, frozenset(), frozenset(),
                fingerprint=template.fingerprint, corpus_snapshot_id=snapshot.corpus_snapshot_id,
                context_plan_id=plan.plan_id, evidence_set_id=plan.selected_set.set_id,
                output_budget=task.output_budget,
                generation_kind=step_generation_kind(task.kind, task.operation, plan.reading_mode))
            candidates.append(candidate)
            ownership[candidate_id] = plan.plan_id
    if not candidates:
        updated = tuple(replace(plan, predicted_cost_ms=0.0,
            cost_prediction_eligible=not any(item["plan_id"] == plan.plan_id for item in unquoted)) for plan in plans)
        return updated, {"steps": [], "unquoted": unquoted,
                         "predicted_read_ms": {p.plan_id: 0.0 for p in updated}}
    quotes = await engine.quote_batch(tuple(candidates), template.fingerprint)
    by_id = {quote.candidate_id: quote for quote in quotes.quotes}
    estimates = tuple(cost_model.predict_native(candidate, by_id[candidate.candidate_id],
        None, batch=quotes, concurrency=max(1, (quotes.active_requests or 0)+1)) for candidate in candidates)
    updated = []
    for plan in plans:
        selected = [e for e in estimates if ownership[e.candidate_id] == plan.plan_id]
        eligible = (not any(item["plan_id"] == plan.plan_id for item in unquoted)
            and all(e.calibrated if e.selection_eligible is None else e.selection_eligible for e in selected))
        updated.append(replace(plan, predicted_cost_ms=sum(e.cost for e in selected),
                               cost_prediction_eligible=eligible))
    return tuple(updated), {"quotes": quotes, "steps": estimates, "unquoted": unquoted,
        "prediction_scope": "known READ inputs at the current native cache state; content-dependent REDUCE and later OpenNeeds are measured when executed",
        "predicted_read_ms": {p.plan_id: p.predicted_cost_ms for p in updated}}
