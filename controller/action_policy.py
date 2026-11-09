"""One observed action: learned final quality and complete response cost differences."""
from __future__ import annotations

from dataclasses import replace

from .admission import quoted_native_work
from .completion_timing import timing_point
from .plan_policy import (NATIVE_COST_SCHEMA, OBSERVED_FEATURE_SCHEMA, OBSERVED_FEATURE_NAMES, CompleteCost, Decision, NextStep, NoCompletePlan, PlanEvaluation,
                          _completed_steps, plan_cost_features, plan_features)
from .reading import step_generation_kind
from .action_proposals import ALTERNATE_ANSWERS, BOUNDED_CONTINUATION
from .evidence_planner import _passages
from .context_budget import answer_evidence_gaps, completion_reserve, direct_answer_reason


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


def observed_quality_gain(model, kind, representation, features):
    if not model.supports_semantic_kind(kind):
        return None
    if kind != "direct_answer":
        return model.predict(features)
    if representation not in ALTERNATE_ANSWERS:
        return 0.0
    record = model.provenance.get("representation_gains", {}).get(representation)
    if record is not None:
        return record["delta_q"]
    # These inputs retain the verified complete conclusion frontier of hybrid.
    return 0.0 if representation in {"facts", "reduced"} else None


def route_completion_reserve(plan, state, task, candidate, batch):
    """Account the whole remaining organization, with exact known READ inputs.

    Future facts are not invented: uncompiled reductions and the final answer
    retain the existing reservation. Quotes describe the current native state,
    not guaranteed hits at a later execution.
    """
    reserve = completion_reserve(plan.context_plan, state.results, task, candidate,
                                 read_all=task is not None and task.kind == "READ")
    if task is None or task.kind != "READ" or not plan.route_steps:
        return reserve, ()
    quotes = {row.candidate_id: row for row in batch.quotes}
    units = {unit.unit_id: unit for unit in plan.context_plan.selected_set.selected_units}
    done = {result.step_id for result in state.results} | {task.step_id}
    added_input = added_output = 0
    future_quotes = []
    for step in plan.route_steps:
        if step.task.kind != "READ" or step.task.step_id in done:
            continue
        choices = [(compiled, quotes[compiled.candidate_id]) for compiled in step.variants
                   if compiled.feasible]
        if not choices:
            continue
        compiled, quote = min(choices, key=lambda pair: (
            sum(quoted_native_work(pair[1])), len(pair[0].input_ids), pair[0].candidate_id))
        if quote.input_hash != compiled.input_hash or quote.input_len != len(compiled.input_ids):
            raise ValueError("remaining READ quote does not bind its compiled input")
        body = sum({units[key].text: units[key].body_tokens for key in step.task.input_unit_ids}.values())
        added_input += len(compiled.input_ids)-body
        added_output += compiled.output_budget-step.task.output_budget
        future_quotes.append(quote)
    return replace(reserve, future_input_tokens=reserve.future_input_tokens+added_input,
                   future_output_tokens=reserve.future_output_tokens+added_output), tuple(future_quotes)


def choose_observed_action(policy, plans, state, native, reusable_prefixes, *, read_control=None,
                           gap_query_chars=0, joint_options=None, timing_context=None):
    """Price one action plus ANSWER; sunk costs constrain the budget, not its marginal value."""
    plans = tuple(plans)
    if len(plans) > policy.limits.max_plan_candidates:
        raise ValueError("observed actions exceed the finite candidate allocation")
    model = policy.gain_model
    if model is not None:
        if (native.reader_checkpoint, native.reader_profile_id) != (model.reader_checkpoint, model.reader_profile_id):
            raise ValueError("observed action asset belongs to a different reader checkpoint/profile")
        if model.provenance.get("continuation_protocol") != BOUNDED_CONTINUATION:
            raise ValueError("observed action asset uses a different bounded continuation protocol")
        if (model.provenance.get("observed_feature_schema") != OBSERVED_FEATURE_SCHEMA
                or tuple(model.feature_names) != OBSERVED_FEATURE_NAMES
                or model.cost_provenance.get("observed_feature_schema") != OBSERVED_FEATURE_SCHEMA):
            raise ValueError("observed action asset requires the current pre-action budget feature schema")
    quotes = {quote.candidate_id: quote for quote in native.quotes.quotes}
    native_cost = model is not None and model.cost_provenance.get("feature_schema") == NATIVE_COST_SCHEMA
    joint = joint_options is not None
    time_limit = state.plan.budget.max_complete_ms
    unmet = answer_evidence_gaps(state) if joint else {}
    direct_prediction = None
    direct_cost = None
    evaluations, decisions = [], []
    policy.last_option_diagnostics = {}
    for plan in plans:
        if plan.reader_checkpoint != native.reader_checkpoint:
            raise ValueError("observed action belongs to a different reader")
        if (state.plan.budget.max_passages is not None and len(_passages(
                plan.context_plan.selected_set.selected_units)) > state.plan.budget.max_passages):
            evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "original_passage_budget"))
            continue
        if model is not None and (not model.supports_semantic_kind(plan.semantic_kind)
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
                and (first.task.kind != "ANSWER" or candidate.action == "REBUILD"
                     and candidate.layout_id in {"existing", "existing:contextpilot"}))
        if not variants:
            evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "next_step_not_compiled"))
        for candidate in variants:
            task = first.task
            output = task.output_budget if candidate is None else candidate.output_budget
            key = candidate.candidate_id if candidate is not None else task.step_id
            delta = joint_options.get(key) if joint else None
            reason = (direct_answer_reason(state, candidate, require_complete_evidence=False)
                      if task.kind == "ANSWER" else None)
            owner_control_ms = delta.retention_control_ms if delta is not None else 0.0
            if model is not None and owner_control_ms is None and delta is not None and delta.retain:
                reason = reason or "unmeasured_retention_control"
            peaks = [native.paid_cost]
            quote = None
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
                forecasts = native.future.get((plan.plan_id, ordered[-1].task.step_id), ())
                if forecasts:
                    peaks.extend(option.cost for option in forecasts)
                elif policy.limits.max_gpu_peak_gib is not None or policy.limits.max_host_peak_gib is not None:
                    reason = "unmeasured_future_resource_peak"
            reserve, future_quotes = route_completion_reserve(plan, state, task, candidate, native.quotes)
            if len(state.steps)+reserve.steps > state.plan.budget.max_steps:
                reason = "cumulative_step_budget"
            empirical_ms = None
            if time_limit is not None and policy.completion_timing is not None and reason is None:
                point = timing_point(task, reserve, native.quotes, quote, native.paid_observation,
                    timing_context, plan.semantic_kind,
                    gap_query_chars=gap_query_chars if task.kind == "GAP" else 0)
                empirical_ms = policy.completion_timing.reserve_ms(point)
            if model is None:
                # This comparator uses coverage and quoted work, with no fitted quality claim.
                complete = task.kind == "ANSWER" and not answer_evidence_gaps(state, candidate)
                useful = task.kind != "ANSWER" and bool(unmet.keys() & set(task.need_ids))
                rank = 2 if complete else 1 if useful else 0 if task.kind == "ANSWER" else -1
                current_work = sum(quoted_native_work(quote)) if candidate is not None else 0
                future_work = reserve.future_input_tokens-sum(row.input_len for row in future_quotes)
                future_work += sum(sum(quoted_native_work(row)) for row in future_quotes)
                work = current_work+future_work+reserve.output_tokens
                cost = native.paid_cost+CompleteCost((empirical_ms or 0)+(owner_control_ms or 0), 0, 0, reserve.input_tokens, reserve.output_tokens,
                    max(peak.gpu_peak_gib for peak in peaks), max(peak.host_peak_gib for peak in peaks))
                reason = reason or policy._constraint(cost, state.plan.budget)
                if time_limit is not None and empirical_ms is None:
                    reason = reason or "uncalibrated_completion_latency"
                if time_limit is not None and owner_control_ms is None:
                    reason = reason or "unmeasured_retention_control"
                if policy.limits.max_gpu_gib_ms is not None or policy.limits.max_host_gib_ms is not None:
                    reason = reason or "unmeasured_completion_residency"
                evaluation = PlanEvaluation(plan.plan_id, None, rank, cost, reason is None,
                    reason or "coverage_and_quoted_work_rule")
                evaluations.append(evaluation)
                policy.last_option_diagnostics[key] = {"cost_source": "coverage_and_quoted_work_rule",
                    "completion_reserve": reserve, "quoted_work_proxy": work, "completion_latency_ms": empirical_ms,
                    "semantic_kind": plan.semantic_kind, "retention_owner_control_ms": owner_control_ms,
                    "joint_context": delta.trace() if delta is not None else None, "reason": reason or "eligible"}
                if reason is None:
                    action = "GAP" if task.kind == "GAP" else "CPU" if cpu else candidate.action
                    prefixes = () if candidate is None else tuple(reusable_prefixes.get(key, ()))
                    decisions.append((evaluation, Decision(NextStep(plan.plan_id, replace(task, output_budget=output),
                        candidate, prefixes), action, output), work, delta))
                continue
            features = plan_features(plan, state, output_budgets={task.step_id: output}, paid_cost=native.paid_cost,
                support_mode=model is not None and (model.provenance.get("evidence_selection_contract") or {}).get("mode") == "support")
            representation = candidate.representation if task.kind == "ANSWER" else "existing"
            gain = observed_quality_gain(model, plan.semantic_kind, representation, features)
            if gain is None:
                policy.last_option_diagnostics[key] = {"plan_id": plan.plan_id,
                    "representation": representation, "reason": "unmeasured_answer_representation"}
                evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False,
                    "unmeasured_answer_representation"))
                continue
            cost_features = (plan_cost_features(features, native.quotes, quote,
                             reserve, gap_query_chars=gap_query_chars if task.kind == "GAP" else 0,
                             future_quotes=future_quotes)
                             if native_cost else features)
            predicted = model.predict_costs(cost_features)
            branch = regression_cost(predicted["branch_complete_cost"])
            control = read_control if task.kind == "READ" else None
            control_ms = None if control is None else control["predicted_ms"]
            read_count = (1+sum(step.task.kind == "READ" and step.task.step_id != task.step_id
                and step.task.step_id not in {result.step_id for result in state.results}
                for step in plan.route_steps)) if task.kind == "READ" else 0
            if control_ms is not None:
                if model.cost_provenance["prefix_control_scope"] != "native_excludes_prefix_control":
                    raise ValueError("READ control addition requires native-only branch cost labels")
                control_ms *= read_count
                branch = replace(branch, latency_ms=branch.latency_ms+control_ms)
            if owner_control_ms:
                if model.cost_provenance.get("prefix_control_scope") != "native_excludes_prefix_control":
                    raise ValueError("retention control addition requires native-only branch cost labels")
                # Demand's post-READ callback only inspects/records sources;
                # this separate selected retain/renew call is paid once here.
                branch = replace(branch, latency_ms=branch.latency_ms+owner_control_ms)
            branch = replace(branch, gpu_peak_gib=max(cost.gpu_peak_gib for cost in peaks),
                             host_peak_gib=max(cost.host_peak_gib for cost in peaks))
            margin = model.cost_provenance.get("completion_reserve_calibration", {}).get(
                plan.semantic_kind, {}).get("latency_margin_ms")
            latency_ms = branch.latency_ms+(margin or 0)
            if empirical_ms is not None:
                latency_ms = max(latency_ms, empirical_ms) if margin is not None else empirical_ms
            completion_cost = replace(branch, input_tokens=reserve.input_tokens, output_tokens=reserve.output_tokens,
                                      latency_ms=latency_ms)
            cumulative = native.paid_cost+completion_cost
            reason = reason or policy._constraint(cumulative, state.plan.budget)
            if native_cost and time_limit is not None and margin is None and empirical_ms is None:
                reason = reason or "uncalibrated_completion_latency"
            if (native_cost and plan.semantic_kind == "direct_answer" and representation not in ALTERNATE_ANSWERS
                    and reason is None):
                penalty = policy._penalty(branch, state.plan.budget)
                if direct_cost is None or penalty < direct_cost:
                    direct_prediction, direct_cost = predicted["branch_complete_cost"], penalty
            # All native costs share the same DIRECT offset, applied once below.
            cost = branch if native_cost else regression_cost(predicted["incremental_complete_cost"])
            if not native_cost and control_ms is not None:
                cost = replace(cost, latency_ms=cost.latency_ms+control_ms)
            if not native_cost and owner_control_ms:
                cost = replace(cost, latency_ms=cost.latency_ms+owner_control_ms)
            # Native quote features already contain the matched-prefix saving.
            # Allocation net value is auditable, but cannot be another rebate
            # without a branch cost baseline that predicts future eviction.
            objective = gain-policy._penalty(cost, state.plan.budget)
            policy.last_option_diagnostics[key] = {"plan_id": plan.plan_id, "features": features,
                "semantic_kind": plan.semantic_kind, "empirical_completion_latency_ms": empirical_ms,
                "cost_features": cost_features,
                "cost_source": "observed_action_paired_run_regression", "predicted_costs": predicted,
                "rollout_expected_cost": branch, "completion_reserve": reserve,
                "completion_cost": completion_cost, "latency_residual_margin_ms": margin,
                "completion_latency_ms": latency_ms if margin is not None or empirical_ms is not None else None,
                "native_quote_role": "pre_action_cost_features" if native_cost else "actual_known_input_admission_and_audit",
                "read_callback_control": control,
                "remaining_read_prefix_control_ms": control_ms, "remaining_read_control_count": read_count,
                "retention_owner_control_ms": owner_control_ms,
                "joint_context": delta.trace() if delta is not None else None, "reason": reason or "eligible"}
            evaluation = PlanEvaluation(plan.plan_id, gain, objective, cumulative, reason is None,
                reason or "reader_bound_observed_action_regression", ((task.step_id,
                    "GAP" if task.kind == "GAP" else "CPU" if cpu else candidate.action, output),),
                ("observed_action_regression:"+model.asset_id,), model.asset_id)
            evaluations.append(evaluation)
            if reason is None:
                action = "GAP" if task.kind == "GAP" else "CPU" if cpu else candidate.action
                prefixes = () if candidate is None else tuple(reusable_prefixes.get(candidate.candidate_id, ()))
                work = sum(quoted_native_work(quote)) if joint and candidate is not None else 0
                decisions.append((evaluation, Decision(NextStep(plan.plan_id, replace(task, output_budget=output),
                    candidate, prefixes), action, output), work, delta))
    if native_cost and direct_prediction is not None:
        offset = policy._penalty(regression_cost(direct_prediction), state.plan.budget)
        evaluations = [replace(row, objective=row.objective+offset) if row.objective is not None else row
                       for row in evaluations]
        decisions = [(replace(row, objective=row.objective+offset), decision, work, delta)
                     for row, decision, work, delta in decisions]
        for diagnostic in policy.last_option_diagnostics.values():
            predicted = diagnostic.get("predicted_costs")
            if predicted is None:
                continue
            predicted["incremental_complete_cost"] = {
                name: value-direct_prediction[name] for name, value in predicted["branch_complete_cost"].items()}
    policy.last_evaluations = tuple(evaluations)
    if not decisions:
        raise NoCompletePlan("no observed action satisfies its measured training scope and cumulative budgets")
    if joint:
        incomplete = {decision.next_step.candidate.candidate_id for _, decision, _, _ in decisions
            if decision.next_step.task.kind == "ANSWER"
            and answer_evidence_gaps(state, decision.next_step.candidate)}
        if any(decision.next_step.task.kind == "ANSWER" and
               decision.next_step.candidate.candidate_id not in incomplete for _, decision, _, _ in decisions):
            decisions = [item for item in decisions if item[1].next_step.task.kind != "ANSWER" or
                         item[1].next_step.candidate.candidate_id not in incomplete]
            for key in incomplete:
                policy.last_option_diagnostics[key]["reason"] = "complete_answer_available"
    return max(decisions, key=lambda item: (item[0].objective, -item[2], -item[0].cost.latency_ms,
        -item[3].uncovered_risk_tokens if item[3] is not None else 0,
        -item[3].protected_tokens if item[3] is not None else 0,
        -item[0].cost.gpu_gib_ms, -item[0].cost.host_gib_ms, item[0].plan_id))[1]
