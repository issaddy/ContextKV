"""One observed action: learned final quality and complete response cost differences."""
from __future__ import annotations

from dataclasses import replace

from .plan_policy import (NATIVE_COST_SCHEMA, OBSERVED_FEATURE_SCHEMA, OBSERVED_FEATURE_NAMES, CompleteCost, Decision, NextStep, NoCompletePlan, PlanEvaluation,
                          _completed_steps, plan_cost_features, plan_features)
from .reading import step_generation_kind
from .action_proposals import BOUNDED_CONTINUATION
from .evidence_planner import _passages


def regression_cost(values):
    return CompleteCost(values["complete_ms"], values["gpu_gib_seconds"]*1000,
        values["host_gib_seconds"]*1000, values["total_input_tokens"], values["total_output_tokens"])


def native_admission_reason(native, candidate):
    admission = (native.paid_observation or {}).get("native_admission")
    if admission is not None:
        output = min(candidate.output_budget, admission["native_max_new_tokens_clip"])
        if len(candidate.input_ids)+output+admission["page_size"] >= admission["gpu_capacity_tokens"]:
            return "native_hard_capacity_exceeded"
    return None


def choose_observed_action(policy, plans, state, native, reusable_prefixes, *, read_control=None):
    """Price one action plus ANSWER; sunk costs constrain the budget, not its marginal value."""
    plans = tuple(plans)
    if len(plans) > policy.limits.max_plan_candidates:
        raise ValueError("observed actions exceed the finite candidate allocation")
    model = policy.gain_model
    if (native.reader_checkpoint, native.reader_profile_id) != (model.reader_checkpoint, model.reader_profile_id):
        raise ValueError("observed action asset belongs to a different reader checkpoint/profile")
    if model.provenance.get("continuation_protocol") != BOUNDED_CONTINUATION:
        raise ValueError("observed action asset uses a different bounded continuation protocol")
    if (model.provenance.get("observed_feature_schema") != OBSERVED_FEATURE_SCHEMA
            or tuple(model.feature_names) != OBSERVED_FEATURE_NAMES
            or model.cost_provenance.get("observed_feature_schema") != OBSERVED_FEATURE_SCHEMA):
        raise ValueError("observed action asset requires the current pre-action budget feature schema")
    quotes = {quote.candidate_id: quote for quote in native.quotes.quotes}
    native_cost = model.cost_provenance.get("feature_schema") == NATIVE_COST_SCHEMA
    direct_prediction = None
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
                reason = native_admission_reason(native, candidate) or reason
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
            features = plan_features(plan, state, output_budgets={task.step_id: output}, paid_cost=native.paid_cost)
            cost_features = (plan_cost_features(features, native.quotes, quote if candidate is not None else None)
                             if native_cost else features)
            predicted = model.predict_costs(cost_features)
            if native_cost and plan.semantic_kind == "direct_answer":
                direct_prediction = predicted["branch_complete_cost"]
            branch = regression_cost(predicted["branch_complete_cost"])
            control = read_control if task.kind == "READ" else None
            control_ms = None if control is None else control["predicted_ms"]
            if control_ms is not None:
                if model.cost_provenance["prefix_control_scope"] != "native_excludes_prefix_control":
                    raise ValueError("READ control addition requires native-only branch cost labels")
                branch = replace(branch, latency_ms=branch.latency_ms+control_ms)
            branch = replace(branch, gpu_peak_gib=max(cost.gpu_peak_gib for cost in peaks),
                             host_peak_gib=max(cost.host_peak_gib for cost in peaks))
            cumulative = native.paid_cost+branch
            reason = reason or policy._constraint(cumulative, state.plan.budget)
            gain = 0.0 if plan.semantic_kind == "direct_answer" else model.predict(features)
            # All native costs share the same DIRECT offset, applied once below.
            cost = branch if native_cost else regression_cost(predicted["incremental_complete_cost"])
            if not native_cost and control_ms is not None:
                cost = replace(cost, latency_ms=cost.latency_ms+control_ms)
            objective = gain-policy._penalty(cost, state.plan.budget)
            key = candidate.candidate_id if candidate is not None else task.step_id
            policy.last_option_diagnostics[key] = {"plan_id": plan.plan_id, "features": features,
                "cost_features": cost_features,
                "cost_source": "observed_action_paired_run_regression", "predicted_costs": predicted,
                "native_quote_role": "pre_action_cost_features" if native_cost else "actual_known_input_admission_and_audit",
                "first_read_prefix_control": control,
                "reason": reason or "eligible"}
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
    if native_cost and direct_prediction is not None:
        offset = policy._penalty(regression_cost(direct_prediction), state.plan.budget)
        evaluations = [replace(row, objective=row.objective+offset) if row.objective is not None else row
                       for row in evaluations]
        decisions = [(replace(row, objective=row.objective+offset), decision) for row, decision in decisions]
        for diagnostic in policy.last_option_diagnostics.values():
            predicted = diagnostic["predicted_costs"]
            predicted["incremental_complete_cost"] = {
                name: value-direct_prediction[name] for name, value in predicted["branch_complete_cost"].items()}
    policy.last_evaluations = tuple(evaluations)
    if not decisions:
        raise NoCompletePlan("no observed action satisfies its measured training scope and cumulative budgets")
    return max(decisions, key=lambda item: (item[0].objective, -item[0].cost.latency_ms,
        -item[0].cost.gpu_gib_ms, -item[0].cost.host_gib_ms, item[0].plan_id))[1]
