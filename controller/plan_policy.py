"""Finite complete-plan gain selection with two-step execution lookahead.

The logical state is the existing ReadingOutcome.  Native quotes price known
inputs; future inputs require an explicitly supplied profile forecast.  The
policy is enabled only with a reader-bound gain model trained on actual answers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from typing import Mapping, NamedTuple, Protocol

from .context_schema import ContextPlan, ReadTask, ReadingOutcome
from .context_budget import completed_source_coverage, direct_answer_reason
from .cost_model import CostModel
from .evidence_planner import _passages
from .reading import step_generation_kind
from .schema import CacheQuoteBatch, PromptCandidate, SessionSnapshot


LEGACY_PLAN_KINDS = ("direct_answer", "complementary_evidence", "multihop", "scope_scan")
PLAN_KINDS = (*LEGACY_PLAN_KINDS, "read_once", "reduce_once", "gap_retrieval")
STRUCTURAL_FEATURE_NAMES = ("needs", "conditions", "version_comparison", "open_needs",
    "unmet_answer_points", "existing_claims", "completed_steps", "read_steps", "reduce_steps",
    "evidence_units", "source_passages", "dependencies", "source_versions", "required_coverage",
    "planned_coverage", "input_budget_tokens", "output_budget_tokens")
ACTION_SIZE_FEATURE_NAMES = ("planned_read_chars", "reduce_input_claim_chars", "existing_claim_chars",
                             "working_history_chars", "answer_body_tokens")
FEATURE_NAMES = (*PLAN_KINDS, *STRUCTURAL_FEATURE_NAMES, *ACTION_SIZE_FEATURE_NAMES)


OBSERVED_FEATURE_SCHEMA = "pre_action_budget_support_state_v2"
BUDGET_FEATURE_NAMES = ("paid_total_tokens", "paid_complete_ms", "paid_steps",
    "total_tokens_cap", "total_tokens_unbounded", "remaining_total_tokens",
    "hard_steps_cap", "remaining_hard_steps", "complete_ms_cap",
    "complete_ms_unbounded", "remaining_complete_ms")
SUPPORT_FEATURE_NAMES = ("infer_needs", "numeric_needs", "need_predecessors",
    "observed_terminal_needs", "observed_bridge_gaps", "retrieval_support_needs",
    "retrieval_support_mean", "retrieval_bridge_mean")
OBSERVED_FEATURE_NAMES = (*FEATURE_NAMES, *BUDGET_FEATURE_NAMES, *SUPPORT_FEATURE_NAMES)


class NoCompletePlan(ValueError):
    """All evaluated complete routes were infeasible; evaluations are retained."""


@dataclass(frozen=True)
class PlanStep:
    task: ReadTask
    variants: tuple[PromptCandidate, ...] = ()
    output_reservation: int = 0


@dataclass(frozen=True)
class PlanCandidate:
    plan_id: str
    semantic_kind: str
    context_plan: ContextPlan
    steps: tuple[PlanStep, ...]
    reader_checkpoint: str
    unmet_answer_points: tuple[str, ...] = ()
    required_coverage: tuple[str, ...] = ()
    answer_representation: str = "existing"


@dataclass(frozen=True)
class CompleteCost:
    latency_ms: float
    gpu_gib_ms: float
    host_gib_ms: float
    input_tokens: int
    output_tokens: int
    gpu_peak_gib: float = 0.0
    host_peak_gib: float = 0.0

    def __add__(self, other):
        return CompleteCost(self.latency_ms+other.latency_ms, self.gpu_gib_ms+other.gpu_gib_ms,
            self.host_gib_ms+other.host_gib_ms, self.input_tokens+other.input_tokens,
            self.output_tokens+other.output_tokens, max(self.gpu_peak_gib, other.gpu_peak_gib),
            max(self.host_peak_gib, other.host_peak_gib))


ZERO_COST = CompleteCost(0.0, 0.0, 0.0, 0, 0)


@dataclass(frozen=True)
class ResourceForecast:
    """C's physical-occupancy prediction for this exact native quote/candidate."""
    gpu_gib_ms: float
    host_gib_ms: float
    gpu_peak_gib: float
    host_peak_gib: float
    calibration_id: str
    quote_seq: int
    prefix_ids: tuple[str, ...]
    calibrated: bool


@dataclass(frozen=True)
class StepForecast:
    context_action: str
    output_budget: int
    cost: CompleteCost
    calibration_id: str
    calibrated: bool
    scope: str = "future_input_profile"
    candidate: PromptCandidate | None = None
    prefix_ids: tuple[str, ...] = ()
    selection_eligible: bool | None = None


def native_selection_cost(estimate, timing_model, candidate, batch, admission, profile):
    """Distinguish empirical calibration from bound statistical selection."""
    original = estimate.calibrated if estimate.selection_eligible is None else estimate.selection_eligible
    if original:
        return estimate.cost, {"selection_eligible": True, "kind": "native_quote",
                               "calibrated": estimate.calibrated}
    metadata = timing_model.calibration_metadata
    if (timing_model.calibration_scope_compatible is not True or not timing_model.sampling_scope_compatible
            or metadata.get("generation_kind") != candidate.generation_kind
            or metadata.get("output_prior_system_prompt") != candidate.messages[0].content
            or candidate.fingerprint != batch.fingerprint or profile is None):
        return None, {"selection_eligible": False, "kind": "native_statistical_scope_mismatch", "calibrated": False}
    capacity, page, clip = admission["gpu_capacity_tokens"], admission["page_size"], admission["native_max_new_tokens_clip"]
    output = candidate.output_budget or timing_model.max_new_tokens
    gate = len(candidate.input_ids)+min(output, clip)+page
    if gate >= capacity or not 0 <= batch.active_requests <= 8:
        return None, {"selection_eligible": False, "kind": "native_hard_capacity_exceeded",
                      "native_gate_tokens": gate, "gpu_capacity_tokens": capacity, "calibrated": False}
    values = [value for value in (estimate.cost, estimate.risk_cost)
              if math.isfinite(value) and value > 0]
    if candidate.generation_kind != "ANSWER":
        values.append(profile["phase_envelope_latency_ms"])
    if not values:
        return None, {"selection_eligible": False, "kind": "native_nonfinite_service_prediction", "calibrated": False}
    service = max(values)
    pending = batch.active_requests+batch.queued_requests
    queue = pending*max(service, admission["profile_service_envelope_ms"])
    total = service+queue
    return total, {"selection_eligible": True, "kind": "native_statistical_extrapolation", "calibrated": False,
        "native_mean_ms": estimate.cost, "native_risk_ms": estimate.risk_cost,
        "service_envelope_ms": service, "statistical_queue_ms": queue,
        "queue_basis": "observed_idle_no_queue" if pending == 0 else "observed pending requests times empirical service upper envelope; statistical serial waiting proxy",
        "native_active_requests": batch.active_requests, "native_queued_requests": batch.queued_requests,
        "native_gate_tokens": gate, "gpu_capacity_tokens": capacity, "remaining_prediction_ms": total}


@dataclass(frozen=True)
class NativePlanningState:
    reader_checkpoint: str
    reader_profile_id: str
    snapshot: SessionSnapshot
    quotes: CacheQuoteBatch
    cost_model: CostModel
    resources: Mapping[str, ResourceForecast]
    future: Mapping[tuple[str, str], tuple[StepForecast, ...]]
    paid_cost: CompleteCost
    paid_observation: dict | None = None


@dataclass(frozen=True)
class PolicyLimits:
    max_plan_candidates: int = 4
    max_gpu_gib_ms: float | None = None
    max_host_gib_ms: float | None = None
    max_gpu_peak_gib: float | None = None
    max_host_peak_gib: float | None = None


class GainModel(Protocol):
    reader_checkpoint: str
    reader_profile_id: str
    asset_id: str

    def predict(self, features: Mapping[str, float]) -> float: ...


@dataclass(frozen=True)
class PlanEvaluation:
    plan_id: str
    predicted_delta_q: float | None
    objective: float | None
    cost: CompleteCost | None
    feasible: bool
    reason: str
    lookahead: tuple[tuple[str, str, int], ...] = ()
    forecast_sources: tuple[str, ...] = ()
    gain_asset: str = ""


@dataclass(frozen=True)
class NextStep:
    plan_id: str
    task: ReadTask
    candidate: PromptCandidate | None
    reusable_prefix_ids: tuple[str, ...]


class Decision(NamedTuple):
    next_step: NextStep
    context_action: str
    output_budget: int


def _completed_steps(plan, state):
    """A repeated step ID is reusable only for the same actual reading task."""
    if (plan.context_plan.task_state != state.plan.task_state or
            plan.context_plan.reading_mode != state.plan.reading_mode or
            plan.context_plan.read_layout != state.plan.read_layout or
            plan.context_plan.read_packing != state.plan.read_packing):
        return set()
    executed = {task.step_id: task for task in state.plan.steps}
    executed.update({record.step_id: record.task for record in state.steps})
    results = {result.step_id for result in state.results}
    old_units = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
    new_units = {unit.unit_id: unit for unit in plan.context_plan.selected_set.selected_units}
    completed = set()
    for step in plan.steps:
        task = step.task
        old = executed.get(task.step_id)
        if task.step_id not in results or old is None:
            continue
        # A new window allocation does not undo completed reading.
        if replace(old, input_budget=0, output_budget=0, output_schema={}, retained_conclusion_tokens=()) != replace(
                task, input_budget=0, output_budget=0, output_schema={}, retained_conclusion_tokens=()):
            continue
        if any(identity not in old_units or identity not in new_units or
               old_units[identity].source_spans != new_units[identity].source_spans or
               old_units[identity].text != new_units[identity].text for identity in task.input_unit_ids):
            continue
        completed.add(task.step_id)
    reduction_owners = {result.reduction.step_id: result.step_id for result in state.results
                        if result.reduction is not None}
    predecessors = {step.task.step_id: {reduction_owners.get(identity, identity)
                                      for identity in step.task.predecessor_step_ids} for step in plan.steps}
    while any(not predecessors[identity] <= completed for identity in completed):
        completed = {identity for identity in completed if predecessors[identity] <= completed}
    return completed | {logical for logical, owner in reduction_owners.items() if owner in completed}


def plan_features(plan: PlanCandidate, state: ReadingOutcome, *, output_budgets=None, paid_cost: CompleteCost | None = None):
    """Structure and current state only; oracle scores never enter features."""
    completed = _completed_steps(plan, state)
    remaining = [step.task for step in plan.steps if step.task.step_id not in completed]
    units = plan.context_plan.selected_set.selected_units
    by_unit = {unit.unit_id: unit for unit in units}
    by_claim = {claim.claim_id: claim for claim in state.claims}

    def claim_chars(claim):
        return sum(len(value) for value in (claim.text, claim.entity, claim.relation, claim.unit,
                   str(claim.value) if claim.value is not None else ""))
    coverage = set(plan.required_coverage)
    if state.plan.coverage_scope == "all_supplied_documents":
        coverage.update(state.plan.coverage_units)
    budgets = output_budgets or {}
    values = {name: float(plan.semantic_kind == name) for name in PLAN_KINDS}
    values.update(needs=len(state.plan.task_state.needs), conditions=len(state.plan.task_state.conditions),
        version_comparison=float(state.plan.task_state.version_intent.mode == "compare"),
        open_needs=len(state.open_needs), unmet_answer_points=len(plan.unmet_answer_points),
        existing_claims=len(state.claims), completed_steps=len(completed),
        read_steps=sum(task.kind == "READ" for task in remaining),
        reduce_steps=sum(task.kind == "REDUCE" for task in remaining), evidence_units=len(units),
        source_passages=len(_passages(units)), dependencies=sum(len(unit.dependencies) for unit in units),
        source_versions=len({span.source_key.version for unit in units for span in unit.source_spans}),
        required_coverage=len(coverage), planned_coverage=len({unit_id for task in remaining
            if task.kind == "READ" for unit_id in task.input_unit_ids}),
        input_budget_tokens=sum(task.input_budget for task in remaining),
        output_budget_tokens=sum(budgets.get(task.step_id,
            budgets.get("ANSWER", task.output_budget) if task.kind == "ANSWER" else task.output_budget)
            for task in remaining),
        planned_read_chars=sum(len(by_unit[identity].text) for task in remaining if task.kind == "READ"
                               for identity in task.input_unit_ids),
        reduce_input_claim_chars=sum(claim_chars(by_claim[identity]) for task in remaining if task.kind == "REDUCE"
                                     for identity in task.input_claim_ids if identity in by_claim),
        existing_claim_chars=sum(claim_chars(claim) for claim in state.claims),
        working_history_chars=sum(ref.end-ref.start for ref in state.plan.task_state.working_history),
        answer_body_tokens=plan.context_plan.budget.body_tokens)
    if paid_cost is not None:
        import math
        if any(not math.isfinite(value) or value < 0 for value in
               (paid_cost.input_tokens, paid_cost.output_tokens, paid_cost.latency_ms)):
            raise ValueError("budget features require measured nonnegative paid cost")
        budget = state.plan.budget
        needs = state.plan.task_state.needs
        active = {need.need_id for need in needs}
        open_ids = {need.need_id for need in state.open_needs}
        terminal = {identity for result in state.results for identity in result.support}
        support = [item for item in plan.context_plan.selected_set.need_support if item.need_id in active]
        values.update(infer_needs=sum(need.operation == "infer" for need in needs),
            numeric_needs=sum(need.operation in {"count", "sum", "group"} for need in needs),
            need_predecessors=sum(len(need.predecessor_need_ids) for need in needs),
            observed_terminal_needs=len((terminal & active)-open_ids),
            observed_bridge_gaps=sum(need.need_id in active and bool(need.parent_claim_ids)
                                     for need in state.open_needs),
            retrieval_support_needs=len(support),
            retrieval_support_mean=sum(item.applicability*item.joint_support for item in support)/len(support)
                if support else 0,
            retrieval_bridge_mean=sum(item.applicability*item.bridge_value for item in support)/len(support)
                if support else 0)
        paid_tokens = paid_cost.input_tokens + paid_cost.output_tokens
        values.update(paid_total_tokens=paid_tokens, paid_complete_ms=paid_cost.latency_ms,
            paid_steps=len(state.steps), hard_steps_cap=budget.max_steps,
            remaining_hard_steps=budget.max_steps-len(state.steps))
        for stem, cap, spent in (("total_tokens", budget.max_total_tokens, paid_tokens),
                                  ("complete_ms", budget.max_complete_ms, paid_cost.latency_ms)):
            # Unbounded coordinates are masked by an explicit flag, never measured.
            values[stem+"_unbounded"] = float(cap is None)
            values[stem+"_cap"] = 0 if cap is None else cap
            values["remaining_"+stem] = 0 if cap is None else cap-spent
    names = OBSERVED_FEATURE_NAMES if paid_cost is not None else FEATURE_NAMES
    return {name: float(values[name]) for name in names}


NATIVE_COST_SCHEMA = "pre_action_native_nonnegative_work_v3"
NATIVE_COST_PLAN_FEATURES = ("direct_answer", "read_once", "reduce_once", "gap_retrieval",
                             "existing_claim_chars", "output_budget_tokens")
NATIVE_COST_FEATURE_NAMES = (*NATIVE_COST_PLAN_FEATURES, "first_input_tokens", "first_prefill_tokens",
                             "first_host_load_tokens", "active_requests", "queued_requests",
                             "future_input_tokens", "future_output_tokens", "future_steps", "gap_query_chars")


def plan_cost_features(features, batch, first_quote, reserve, *, gap_query_chars=0):
    """Use branch work known before the first action and actual Native pressure."""
    from .admission import quoted_native_work

    input_tokens, prefill_tokens, host_load = 0, 0, 0
    if first_quote is not None:
        input_tokens = first_quote.input_len
        prefill_tokens, host_load = quoted_native_work(first_quote)
    values = {**{name: features[name] for name in NATIVE_COST_PLAN_FEATURES},
        "first_input_tokens": input_tokens, "first_prefill_tokens": prefill_tokens,
        "first_host_load_tokens": host_load, "active_requests": batch.active_requests,
        "queued_requests": batch.queued_requests, "future_input_tokens": reserve.future_input_tokens,
        "future_output_tokens": reserve.future_output_tokens, "future_steps": reserve.future_steps,
        "gap_query_chars": gap_query_chars}
    return values


def complete_plan(context_plan, answer_variants, *, semantic_kind, reader_checkpoint,
                  step_variants=None, unmet_answer_points=()):
    """Adapt A's existing complete ContextPlan/DAG; append its terminal ANSWER."""
    variants = step_variants or {}
    tasks = tuple(PlanStep(task, tuple(variants.get(task.step_id, ()))) for task in context_plan.steps)
    answer = ReadTask("answer:"+context_plan.plan_id, "ANSWER",
        tuple(need.need_id for need in context_plan.task_state.needs),
        predecessor_step_ids=tuple(task.step_id for task in context_plan.steps),
        input_budget=context_plan.budget.window_tokens-context_plan.budget.output_tokens,
        output_budget=context_plan.budget.output_tokens, operation="answer")
    return PlanCandidate(context_plan.plan_id, semantic_kind, context_plan,
        (*tasks, PlanStep(answer, tuple(answer_variants))), reader_checkpoint,
        tuple(unmet_answer_points), context_plan.coverage_units if
        context_plan.coverage_scope == "all_supplied_documents" else ())


class PlanPolicy:
    def __init__(self, gain_model: GainModel | None, *, limits=PolicyLimits(), rule_baseline=False,
                 completion_timing=None):
        if gain_model is None and not rule_baseline:
            raise ValueError("plan-gain policy is not enabled without a reader-bound trained gain asset")
        self.gain_model, self.limits = gain_model, limits
        self.completion_timing = completion_timing
        self.last_evaluations = ()
        self.last_option_diagnostics = {}

    def _penalty(self, cost, budget):
        return (budget.lambda_time*cost.latency_ms+budget.lambda_gpu*cost.gpu_gib_ms
                +budget.lambda_host*cost.host_gib_ms)

    def _schedule(self, plan, completed):
        remaining = list(plan.steps)
        result, available = [], set(completed)
        while remaining:
            ready = next((step for step in remaining if step.task.step_id in available or
                          set(step.task.predecessor_step_ids) <= available), None)
            if ready is None:
                raise ValueError(f"{plan.plan_id} is not a complete executable DAG")
            remaining.remove(ready)
            if ready.task.step_id not in available:
                result.append(ready)
                available.add(ready.task.step_id)
        return result

    def _options(self, plan, step, completed, native, reusable_prefixes, *, future_execution=False):
        if future_execution or not step.variants or not set(step.task.predecessor_step_ids) <= completed:
            return tuple(option for option in native.future.get((plan.plan_id, step.task.step_id), ())
                if self._within_step_budget(plan, step, option.cost.input_tokens, option.output_budget))
        quotes = {quote.candidate_id: quote for quote in native.quotes.quotes}
        result = []
        for candidate in step.variants:
            diagnostic = {"candidate_id": candidate.candidate_id, "plan_id": plan.plan_id,
                          "step_id": step.task.step_id}
            self.last_option_diagnostics[candidate.candidate_id] = diagnostic
            if not candidate.feasible:
                diagnostic["reason"] = "compiled_input_output_infeasible"
                continue
            quote = quotes[candidate.candidate_id]
            if (quote.input_hash != candidate.input_hash or quote.input_len != len(candidate.input_ids)
                    or candidate.fingerprint != native.quotes.fingerprint):
                raise ValueError("plan quote does not bind the actual compiled input")
            resource = native.resources.get(candidate.candidate_id)
            if resource is None:
                diagnostic["reason"] = "physical_profile_unavailable"
                continue
            if resource.quote_seq != native.quotes.observed_seq:
                raise ValueError("resource forecast and native quote use different observations")
            if not set(resource.prefix_ids) <= set(reusable_prefixes.get(candidate.candidate_id, ())):
                raise ValueError("resource forecast names unavailable reusable prefixes")
            output = candidate.output_budget or step.task.output_budget
            expected_kind = step_generation_kind(step.task.kind, step.task.operation, plan.context_plan.reading_mode)
            if candidate.generation_kind != expected_kind:
                raise ValueError("compiled candidate generation kind differs from its plan step")
            if not self._within_step_budget(plan, step, len(candidate.input_ids), output):
                diagnostic["reason"] = "output_reservation_or_step_context_budget"
                continue
            timing_model = native.cost_model.step_models.get(expected_kind, native.cost_model)
            metadata = timing_model.calibration_metadata
            if (metadata.get("generation_kind") != expected_kind or
                    metadata.get("output_prior_system_prompt") != candidate.messages[0].content):
                diagnostic["reason"] = "timing_kind_or_prompt_scope_mismatch"
                continue
            control_adjustment = 0.0
            scope = "native_quote"
            if expected_kind == "ANSWER":
                prior = timing_model.step_output_priors.get(f"ANSWER:{output}")
                if (prior is None or "mean_remaining_control_ms" not in prior or
                        "rolling_proposals_compilation_start" not in metadata.get("step_control_scope", ())):
                    diagnostic["reason"] = "answer_remaining_control_or_start_scope_missing"
                    continue
                # The actual selection head is already in paid_cost. Replace
                # this profile's full control term by its measured remaining
                # prepare/commit tail; READ starts inside its executor instead.
                control_adjustment = prior["mean_remaining_control_ms"]-prior["mean_control_ms"]
                scope = "native_quote_actual_remaining_answer_control"
            # Preparation and completed reading are already in paid_cost.  Each
            # native forecast here prices only its own remaining execution.
            measured = native.cost_model.predict_native(replace(candidate, predicted_read_ms=0.0,
                auxiliary_ms=0.0), quote, native.snapshot, batch=native.quotes,
                concurrency=native.quotes.active_requests+1, control_ms=control_adjustment)
            profiles = (native.paid_observation or {}).get("execution_profile_predictions", {})
            profile = profiles.get(candidate.candidate_id)
            selected_cost, selection = native_selection_cost(measured, timing_model, candidate, native.quotes,
                (native.paid_observation or {}).get("native_admission", {}), profile)
            eligible = selection["selection_eligible"] and resource.calibrated
            if selection["kind"] == "native_statistical_extrapolation":
                scope = "native_statistical_extrapolation"
            gpu, host = resource.gpu_gib_ms, resource.host_gib_ms
            if selection["kind"] == "native_statistical_extrapolation":
                if expected_kind == "ANSWER":
                    rates = profile["conditional_remaining_resources"]
                    gpu, host = rates["mean_gpu_gib"]*selected_cost, rates["mean_host_gib"]*selected_cost
                else:
                    gpu = max(gpu, profile["mean_gpu_gib"]*selected_cost)
                    host = max(host, profile["mean_host_gib"]*selected_cost)
            diagnostic.update(reason="eligible" if eligible and resource.calibrated
                              else "native_timing_prediction_not_selection_eligible",
                              estimate=measured, native_scope=timing_model.calibration_scope_report,
                              sampling_scope=timing_model.sampling_scope_report, selection=selection)
            if selected_cost is None:
                continue
            result.append(StepForecast(candidate.action, output,
                CompleteCost(selected_cost, gpu, host,
                    len(candidate.input_ids), output, resource.gpu_peak_gib, resource.host_peak_gib),
                resource.calibration_id, bool(measured.calibrated and resource.calibrated), scope,
                candidate, resource.prefix_ids, bool(eligible)))
        return tuple(result)

    def _within_step_budget(self, plan, step, inputs, output):
        budget = plan.context_plan.budget
        output_valid = (output == budget.output_tokens if step.task.kind == "ANSWER" else
                        output <= min(step.task.output_budget, budget.read_output_tokens))
        return (output_valid and output >= step.output_reservation
                and inputs <= step.task.input_budget and inputs+output <= budget.window_tokens)

    def _constraint(self, cost, budget):
        limits = self.limits
        checks = ((budget.max_complete_ms, cost.latency_ms, "complete_latency_budget"),
            (budget.max_total_tokens, cost.input_tokens+cost.output_tokens, "cumulative_token_budget"),
            (limits.max_gpu_gib_ms, cost.gpu_gib_ms, "gpu_integral_budget"),
            (limits.max_host_gib_ms, cost.host_gib_ms, "host_integral_budget"),
            (limits.max_gpu_peak_gib, cost.gpu_peak_gib, "gpu_capacity_budget"),
            (limits.max_host_peak_gib, cost.host_peak_gib, "host_capacity_budget"))
        return next((reason for limit, value, reason in checks if limit is not None and value > limit), None)

    def choose_observed_actions(self, plans, state, native, reusable_prefixes, *, read_control=None,
                                gap_query_chars=0, joint_options=None, timing_context=None):
        from .action_policy import choose_observed_action
        return choose_observed_action(self, plans, state, native, reusable_prefixes, read_control=read_control,
            gap_query_chars=gap_query_chars, joint_options=joint_options, timing_context=timing_context)

    def choose(self, plans, state: ReadingOutcome, native: NativePlanningState, reusable_prefixes, *,
               mandatory_coverage=False):
        self.last_option_diagnostics = {}
        plans = tuple(plans)
        if len(plans) > self.limits.max_plan_candidates:
            raise ValueError("A must supply the configured finite number of complete plans")
        if (native.reader_checkpoint != self.gain_model.reader_checkpoint or
                native.reader_profile_id != self.gain_model.reader_profile_id):
            raise ValueError("plan gain asset belongs to a different reader checkpoint/profile")
        required = set(state.plan.coverage_units) if state.plan.coverage_scope == "all_supplied_documents" else set()
        evaluations, decisions = [], []
        for plan in plans:
            if plan.reader_checkpoint != native.reader_checkpoint:
                raise ValueError("candidate plan belongs to a different reader checkpoint")
            if plan.semantic_kind not in PLAN_KINDS or sum(step.task.kind == "ANSWER" for step in plan.steps) != 1:
                raise ValueError("a semantic plan must have a supported kind and one terminal ANSWER")
            completed = _completed_steps(plan, state)
            tasks = {step.task.step_id: step.task for step in plan.steps}
            if len(tasks) != len(plan.steps):
                raise ValueError("a semantic plan has unique execution step IDs")
            answer = next(task for task in tasks.values() if task.kind == "ANSWER")
            pending, ancestors = list(answer.predecessor_step_ids), set()
            while pending:
                step_id = pending.pop()
                if step_id not in ancestors:
                    ancestors.add(step_id)
                    if step_id in tasks:
                        pending.extend(tasks[step_id].predecessor_step_ids)
            if not set(tasks)-{answer.step_id} <= ancestors:
                raise ValueError("every planned reading/reduction step must lead to the terminal ANSWER")
            ordered = self._schedule(plan, completed)
            if not ordered or ordered[-1].task.kind != "ANSWER":
                raise ValueError("a semantic plan must complete with ANSWER")
            planned_coverage = {unit_id for step in ordered if step.task.kind == "READ"
                                for unit_id in step.task.input_unit_ids}
            reason = None
            material = {unit.unit_id: unit for unit in (*state.plan.selected_set.selected_units,
                                                       *plan.context_plan.selected_set.selected_units)}
            required_units = tuple(material[identity]
                for identity in (required | set(plan.required_coverage))-planned_coverage)
            if ordered[0].task.kind != "ANSWER" and not completed_source_coverage(
                    state.plan, state.results, required_units):
                reason = "required_scope_uncovered"
            elif plan.semantic_kind == "direct_answer" and any(step.task.kind != "ANSWER" for step in ordered):
                raise ValueError("direct_answer contains only its terminal ANSWER")
            elif (state.plan.budget.max_passages is not None and len(_passages(
                    plan.context_plan.selected_set.selected_units)) > state.plan.budget.max_passages):
                reason = "original_passage_budget"
            if reason:
                evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, reason))
                continue
            # selected_set retains the source archive across READs. Context limits
            # apply to each compiled READ/ANSWER window, including its output.
            options = [tuple(option for option in self._options(plan, step, completed, native, reusable_prefixes,
                future_execution=index > 0) if (option.calibrated if option.selection_eligible is None
                                               else option.selection_eligible)) for index, step in enumerate(ordered)]
            cpu_step = (ordered[0].task.kind == "REDUCE" and plan.context_plan.reading_mode == "typed_reduce"
                and ordered[0].task.operation in {"union", "count", "sum", "group"})
            if not options[0] or any(option.candidate is None and not (
                    cpu_step and option.scope in {"cpu_profile", "cpu_empirical_bin_prediction"}) for option in options[0]):
                evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "next_step_not_compiled"))
                continue
            if any(not choices for choices in options):
                evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "unmeasured_complete_cost"))
                continue
            if len(state.steps)+len(ordered) > state.plan.budget.max_steps:
                evaluations.append(PlanEvaluation(plan.plan_id, None, None, None, False, "cumulative_step_budget"))
                continue
            # Branch only over the next two executions. The complete remaining
            # tail is still charged using each step's supplied profile forecast.
            depth = min(2, len(ordered))
            tail = tuple(min(choices, key=lambda option: self._penalty(option.cost, state.plan.budget))
                         for choices in options[depth:])
            for prefix in product(*options[:depth]):
                route = (*prefix, *tail)
                cost = native.paid_cost
                for option in route:
                    cost += option.cost
                reason = self._constraint(cost, state.plan.budget)
                if ordered[0].task.kind == "ANSWER":
                    reason = direct_answer_reason(
                        state, route[0].candidate, require_complete_evidence=False) or reason
                budgets = {step.task.step_id: option.output_budget for step, option in zip(ordered, route, strict=True)}
                gain = (0.0 if mandatory_coverage or plan.semantic_kind == "direct_answer" else self.gain_model.predict(
                    plan_features(plan, state, output_budgets=budgets)))
                objective = gain-self._penalty(cost, state.plan.budget)
                evaluation = PlanEvaluation(plan.plan_id, gain, objective, cost, reason is None,
                    reason or ("mandatory_source_coverage" if mandatory_coverage else "reader_bound_complete_plan_gain"), tuple((step.task.step_id,
                        option.context_action, option.output_budget) for step, option in zip(ordered[:depth], prefix, strict=True)),
                    tuple(option.scope+":"+option.calibration_id for option in route), self.gain_model.asset_id)
                evaluations.append(evaluation)
                if reason is None:
                    first = route[0]
                    task = replace(ordered[0].task, output_budget=first.output_budget)
                    decisions.append((evaluation, Decision(NextStep(plan.plan_id, task,
                        first.candidate, first.prefix_ids), first.context_action, first.output_budget)))
        self.last_evaluations = tuple(evaluations)
        if not decisions:
            raise NoCompletePlan("no complete plan satisfies coverage, budgets and measured execution forecasts")
        return max(decisions, key=lambda item: (item[0].objective, -item[0].cost.latency_ms,
            -item[0].cost.gpu_gib_ms, -item[0].cost.host_gib_ms, item[0].plan_id))[1]
