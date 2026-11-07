"""Native quotes for the known READ inputs of competing semantic plans."""
from __future__ import annotations

from dataclasses import replace

from .request_compiler import RequestCompiler


async def quote_read_costs(plans, snapshot, template, engine, cost_model, *, completed_steps=(),
                           initial_outcomes=(), request_compiler=None):
    """Quote pending READs whose complete inputs are already known."""
    plans = tuple(plans)
    compiler = request_compiler or RequestCompiler(template)
    seeds = {outcome.plan.plan_id: outcome for outcome in initial_outcomes}
    candidates, groups, unquoted = [], [], []
    scope = await engine.info()
    for plan in plans:
        seed = seeds.get(plan.plan_id)
        completed = {record.step_id for record in completed_steps}
        if seed is not None:
            completed.update(result.step_id for result in seed.results)
        claims = {claim.claim_id: claim for claim in seed.claims} if seed is not None else {}
        claims.update((claim.claim_id, claim) for record in completed_steps for claim in record.result.claims)
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
            variants = compiler.compile_read(plan, task, source,
                tuple(claims[cid] for cid in task.input_claim_ids), snapshot,
                outputs=(task.output_budget,), cache_scope=scope)
            candidates.extend(variants)
            groups.append((plan.plan_id, task.step_id, variants))
    if not candidates:
        updated = tuple(replace(plan, predicted_cost_ms=0.0,
            cost_prediction_eligible=not any(item["plan_id"] == plan.plan_id for item in unquoted)) for plan in plans)
        return updated, {"steps": [], "unquoted": unquoted,
                         "predicted_read_ms": {p.plan_id: 0.0 for p in updated}}
    quotes = await engine.quote_batch(tuple(candidates), template.fingerprint)
    by_id = {quote.candidate_id: quote for quote in quotes.quotes}
    selected = tuple((plan_id, step_id, compiler.choose_variant(variants, quotes))
                     for plan_id, step_id, variants in groups)
    estimates = tuple(cost_model.predict_native(candidate, by_id[candidate.candidate_id],
        None, batch=quotes, concurrency=max(1, (quotes.active_requests or 0)+1)) for _, _, candidate in selected)
    ownership = {candidate.candidate_id: plan_id for plan_id, _, candidate in selected}
    updated = []
    for plan in plans:
        plan_estimates = [e for e in estimates if ownership[e.candidate_id] == plan.plan_id]
        eligible = (not any(item["plan_id"] == plan.plan_id for item in unquoted)
            and all(e.calibrated if e.selection_eligible is None else e.selection_eligible for e in plan_estimates))
        updated.append(replace(plan, predicted_cost_ms=sum(e.cost for e in plan_estimates),
                               cost_prediction_eligible=eligible))
    return tuple(updated), {"quotes": quotes, "steps": estimates, "unquoted": unquoted,
        "selected_requests": tuple({"plan_id": plan_id, "step_id": step_id,
            "candidate_id": candidate.candidate_id, "input_hash": candidate.input_hash,
            "read_unit_order": candidate.read_unit_order} for plan_id, step_id, candidate in selected),
        "prediction_scope": "known READ inputs at the current native cache state; content-dependent REDUCE and later OpenNeeds are measured when executed",
        "predicted_read_ms": {p.plan_id: p.predicted_cost_ms for p in updated}}
