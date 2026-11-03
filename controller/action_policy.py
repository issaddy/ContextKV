"""One observed action: learned final quality and complete response cost differences."""
from __future__ import annotations

from dataclasses import replace

from .plan_policy import (CompleteCost, Decision, NextStep, NoCompletePlan, PlanEvaluation,
                          _completed_steps, plan_features)
from .reading import step_generation_kind
from .evidence_planner import _passages


def regression_cost(values):
    return CompleteCost(values["complete_ms"], values["gpu_gib_seconds"]*1000,
        values["host_gib_seconds"]*1000, values["total_input_tokens"], values["total_output_tokens"])


def choose_observed_action(policy, plans, state, native, reusable_prefixes):
    """Price one action plus ANSWER; sunk costs constrain the budget, not its marginal value."""
    plans = tuple(plans)
    if len(plans) > policy.limits.max_plan_candidates:
        raise ValueError("observed actions exceed the finite candidate allocation")
    model = policy.gain_model
    if (native.reader_checkpoint, native.reader_profile_id) != (model.reader_checkpoint, model.reader_profile_id):
        raise ValueError("observed action asset belongs to a different reader checkpoint/profile")
    quotes = {quote.candidate_id: quote for quote in native.quotes.quotes}
    evaluations, decisions = [], []
    policy.last_option_diagnostics = {}
    for plan in plans:
        if plan.reader_checkpoint != native.reader_checkpoint:
            raise ValueError("observed action belongs to a different reader")
        if (state.plan.budget.max_passages is not None and len(_passages(
                plan.context_plan.selected_set.selected_units)) > state.plan.budget.max_passages):
            evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "original_passage_budget"))
            continue
        if (not model.supports_semantic_kind(plan.semantic_kind)
                or not model.supports_semantic_kind(plan.semantic_kind, costs=True)):
            evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "observed_action_not_trained"))
            continue
        ordered = policy._schedule(plan, _completed_steps(plan, state))
        kinds = {"direct_answer": ("ANSWER",), "read_once": ("READ", "ANSWER"),
                 "reduce_once": ("REDUCE", "ANSWER"), "gap_retrieval": ("GAP", "ANSWER")}
        if tuple(step.task.kind for step in ordered) != kinds[plan.semantic_kind]:
            raise ValueError("observed action plan must contain its one new action then ANSWER")
        first = ordered[0]
        cpu = (first.task.kind == "REDUCE" and plan.context_plan.reading_mode == "typed_reduce"
               and first.task.operation in {"union", "count", "sum", "group"})
        if cpu or first.task.kind == "GAP":
            variants = (None,)
        else:
            # The paired collector currently declares REBUILD/existing terminal inputs.
            variants = tuple(candidate for candidate in first.variants if candidate.feasible
                and (first.task.kind != "ANSWER" or (candidate.action, candidate.layout_id) == ("REBUILD", "existing")))
        if not variants:
            evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "next_step_not_compiled"))
        for candidate in variants:
            task = first.task
            output = task.output_budget if candidate is None else candidate.output_budget
            reason = None
            current = CompleteCost(0, 0, 0, 0, 0)
            peaks = [native.paid_cost]
            if candidate is not None:
                quote = quotes[candidate.candidate_id]
                if (quote.input_hash != candidate.input_hash or quote.input_len != len(candidate.input_ids)
                        or candidate.fingerprint != native.quotes.fingerprint):
                    raise ValueError("plan quote does not bind the actual compiled input")
                if candidate.generation_kind != step_generation_kind(task.kind, task.operation,
                                                                    plan.context_plan.reading_mode):
                    raise ValueError("compiled candidate generation kind differs from its plan step")
                if not policy._within_step_budget(plan, first, len(candidate.input_ids), output):
                    reason = "output_reservation_or_step_context_budget"
                current = CompleteCost(0, 0, 0, len(candidate.input_ids), output)
                resource = native.resources.get(candidate.candidate_id)
                if resource is not None:
                    if resource.quote_seq != native.quotes.observed_seq:
                        raise ValueError("resource forecast and native quote use different observations")
                    if not set(resource.prefix_ids) <= set(reusable_prefixes.get(candidate.candidate_id, ())):
                        raise ValueError("resource forecast names unavailable reusable prefixes")
                    peaks.append(CompleteCost(0, 0, 0, 0, 0, resource.gpu_peak_gib, resource.host_peak_gib))
                elif policy.limits.max_gpu_peak_gib is not None or policy.limits.max_host_peak_gib is not None:
                    reason = "unmeasured_current_resource_peak"
                admission = (native.paid_observation or {}).get("native_admission")
                if admission is not None:
                    gate = len(candidate.input_ids)+min(output, admission["native_max_new_tokens_clip"])+admission["page_size"]
                    if gate >= admission["gpu_capacity_tokens"]:
                        reason = "native_hard_capacity_exceeded"
            elif cpu:
                profiles = native.future.get((plan.plan_id, task.step_id), ())
                if profiles:
                    peaks.extend(option.cost for option in profiles)
                elif policy.limits.max_gpu_peak_gib is not None or policy.limits.max_host_peak_gib is not None:
                    reason = "unmeasured_current_resource_peak"
            elif policy.limits.max_gpu_peak_gib is not None or policy.limits.max_host_peak_gib is not None:
                reason = "unmeasured_gap_resource_peak"
            if task.kind != "ANSWER":
                current += CompleteCost(0, 0, 0, 0, plan.context_plan.budget.output_tokens)
                forecasts = native.future.get((plan.plan_id, ordered[-1].task.step_id), ())
                if forecasts:
                    peaks.extend(option.cost for option in forecasts)
                elif policy.limits.max_gpu_peak_gib is not None or policy.limits.max_host_peak_gib is not None:
                    reason = "unmeasured_future_resource_peak"
            if len(state.steps)+len(ordered) > state.plan.budget.max_steps:
                reason = "cumulative_step_budget"
            reason = reason or policy._constraint(native.paid_cost+current, state.plan.budget)
            features = plan_features(plan, state, output_budgets={task.step_id: output})
            predicted = model.predict_costs(features)
            branch = regression_cost(predicted["branch_complete_cost"])
            branch = replace(branch, gpu_peak_gib=max(cost.gpu_peak_gib for cost in peaks),
                             host_peak_gib=max(cost.host_peak_gib for cost in peaks))
            cumulative = native.paid_cost+branch
            reason = reason or policy._constraint(cumulative, state.plan.budget)
            gain = 0.0 if plan.semantic_kind == "direct_answer" else model.predict(features)
            delta = regression_cost(predicted["incremental_complete_cost"])
            objective = gain-policy._penalty(delta, state.plan.budget)
            key = candidate.candidate_id if candidate is not None else task.step_id
            policy.last_option_diagnostics[key] = {"plan_id": plan.plan_id, "features": features,
                "cost_source": "observed_action_paired_run_regression", "predicted_costs": predicted,
                "native_quote_role": "actual_known_input_admission_and_audit", "reason": reason or "eligible"}
            evaluation = PlanEvaluation(plan.plan_id, gain, objective, cumulative, reason is None,
                reason or "reader_bound_observed_action_regression", ((task.step_id,
                    "GAP" if task.kind == "GAP" else "CPU" if cpu else candidate.action, output),),
                ("observed_action_regression:"+model.asset_id,), model.asset_id)
            evaluations.append(evaluation)
            if reason is None:
                action = "GAP" if task.kind == "GAP" else "CPU" if cpu else candidate.action
                prefixes = () if candidate is None else tuple(reusable_prefixes.get(candidate.candidate_id, ()))
                decisions.append((evaluation, Decision(NextStep(plan.plan_id, replace(task, output_budget=output),
                    candidate, prefixes), action, output)))
    policy.last_evaluations = tuple(evaluations)
    if not decisions:
        raise NoCompletePlan("no observed action satisfies its measured training scope and cumulative budgets")
    return max(decisions, key=lambda item: (item[0].objective, -item[0].cost.latency_ms,
        -item[0].cost.gpu_gib_ms, -item[0].cost.host_gib_ms, item[0].plan_id))[1]
