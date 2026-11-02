"""Execute finite complete plans, observing and reselecting after each step.

Profiles and paid resource observations are explicit inputs.  This module never
imports oracle evidence, estimates quality from support scores, or invents a
native quote for an input whose facts have not been generated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time

from .claim_graph import ClaimGraph
from .context_schema import ReadTask, ReadingOutcome
from .plan_policy import (PlanPolicy, PlanStep,
                          _completed_steps, complete_plan)
from .reading import output_schema, read_output_reservation, step_generation_kind, step_messages
from .schema import Plan, PromptCandidate
from .serialization import dumps


ZERO_AUX = {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []}


def namespace_plan(plan):
    """Content identity prevents read:0 from aliasing another plan's input."""
    if plan.steps and all(task.step_id.startswith("policy:") for task in plan.steps):
        return plan
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    mapped, tasks = {}, []
    for task in plan.steps:
        if task.step_id.startswith("policy:"):
            mapped[task.step_id] = task.step_id
            tasks.append(task)
            continue
        claims = []
        for claim_id in task.input_claim_ids:
            prefix, separator, suffix = claim_id.rpartition(":claim:")
            claims.append(mapped.get(prefix, prefix)+separator+suffix if separator else claim_id)
        identity = hashlib.sha256(dumps((plan.task_state, plan.reading_mode, plan.read_layout,
            plan.read_packing, task.kind, task.operation, task.need_ids,
            tuple(replace(units[key], retrieval_score=0.0, need_ids=()) for key in task.input_unit_ids), tuple(claims),
            tuple(mapped[key] for key in task.predecessor_step_ids))).encode()).hexdigest()[:24]
        mapped[task.step_id] = "policy:"+identity
        tasks.append(replace(task, step_id=mapped[task.step_id], output_schema={},
            input_claim_ids=tuple(claims),
            predecessor_step_ids=tuple(mapped[key] for key in task.predecessor_step_ids)))
    return replace(plan, steps=tuple(tasks))


def _reduction_tasks(plan, state):
    """Declare the complete reduction tail before its READ facts exist."""
    if plan.reading_mode == "coverage_only":
        return ()
    read_ids = tuple(task.step_id for task in plan.steps if task.kind == "READ")
    if not read_ids:
        return ()
    summarize = all(task.operation == "summarize" for task in plan.steps if task.kind == "READ")
    tasks = []
    for need in plan.task_state.needs:
        if not summarize and need.operation not in {"enumerate", "compare", "count", "sum", "group", "infer"}:
            continue
        if any(task.kind == "REDUCE" and need.need_id in task.need_ids
               and task.predecessor_step_ids == read_ids for task in plan.steps):
            continue
        operation = "summarize" if summarize else "union" if need.operation == "enumerate" else need.operation
        identity = hashlib.sha256(dumps((read_ids, need.need_id, operation)).encode()).hexdigest()[:24]
        claims = tuple(claim.claim_id for claim in state.claims
                       if claim.need_id == need.need_id and claim.kind == "extract"
                       and claim.claim_id.rsplit(":claim:", 1)[0] in read_ids)
        if set(read_ids) <= {result.step_id for result in state.results} and not claims:
            continue
        tasks.append(ReadTask("policy:reduce:"+identity, "REDUCE", (need.need_id,),
            input_claim_ids=claims, predecessor_step_ids=read_ids,
            input_budget=plan.budget.window_tokens-plan.budget.read_output_tokens,
            output_budget=plan.budget.read_output_tokens, operation=operation))
    return tuple(tasks)


def _with_observed_sources(plan, state):
    read_units = {identity for record in state.steps for identity in record.task.input_unit_ids}
    previous = tuple(unit for unit in state.plan.selected_set.selected_units if unit.unit_id in read_units)
    units = {unit.unit_id: unit for unit in (*previous, *plan.selected_set.selected_units)}
    for unit in previous:
        if replace(units[unit.unit_id], retrieval_score=unit.retrieval_score, need_ids=unit.need_ids) != unit:
            raise ValueError("a candidate changes the original material of an observed unit")
    blocks = {block.key: block for block in (*state.plan.selected_set.source_blocks,
                                            *plan.selected_set.source_blocks)}
    if state.plan.coverage_scope == "all_supplied_documents":
        plan = replace(plan, coverage_scope=state.plan.coverage_scope, coverage_units=state.plan.coverage_units)
    disposition = (("partial" if state.claims else "retrieval_gap") if state.open_needs
                   else plan.selected_set.disposition)
    return replace(plan, selected_set=replace(plan.selected_set, selected_units=tuple(units.values()),
        source_blocks=tuple(blocks.values())), answer_plan=ClaimGraph(state.claims).answer_plan(
            plan.task_state, disposition) if state.claims else replace(plan.answer_plan, disposition=disposition))


def compile_step(plan, task, state, snapshot, template, *, outputs):
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    claims = {claim.claim_id: claim for claim in state.claims}
    selected_units = tuple(units[identity] for identity in task.input_unit_ids)
    selected_claims = tuple(claims[identity] for identity in task.input_claim_ids)
    task = replace(task, output_schema=output_schema(task, selected_units, selected_claims,
        reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity))
    messages = step_messages(plan, task, selected_units, selected_claims, snapshot)
    rendered = template.render(messages)
    kind = step_generation_kind(task.kind, task.operation, plan.reading_mode)
    return tuple(PromptCandidate(dumps([plan.plan_id, task.step_id, output]), "REBUILD", snapshot.version,
        messages, rendered.input_ids, rendered.input_hash, frozenset(), frozenset(),
        feasible=len(rendered.input_ids) <= task.input_budget and len(rendered.input_ids)+output <= plan.budget.window_tokens,
        fingerprint=template.fingerprint, corpus_snapshot_id=snapshot.corpus_snapshot_id,
        context_plan_id=plan.plan_id, evidence_set_id=plan.selected_set.set_id,
        output_budget=output, generation_kind=kind) for output in outputs)


@dataclass(frozen=True)
class RollingResult:
    outcome: ReadingOutcome
    plan: Plan
    quotes: object
    decisions: tuple[dict, ...]
    answer_phase_started: float


class RollingPlanRuntime:
    """Native provider supplies measured paid costs and bound physical profiles.

    `native.observe(plans, state, snapshot, engine, cost_model, arrived,
    auxiliary_usage)` returns NativePlanningState and real reusable-prefix IDs.
    `replan` may replace the finite proposals after observed facts/open needs;
    it must return existing ContextPlan values and report its auxiliary usage.
    """
    def __init__(self, gain_model, native, *, reader_checkpoint, reader_profile_id,
                 output_budgets, limits=None, replan=None, output_tokens_per_fact=64):
        from .plan_policy import PolicyLimits
        self.policy = PlanPolicy(gain_model, limits=limits or PolicyLimits())
        if (reader_checkpoint, reader_profile_id) != (gain_model.reader_checkpoint, gain_model.reader_profile_id):
            raise ValueError("rolling policy gain asset does not bind the configured reader/profile")
        self.reader_checkpoint, self.reader_profile_id = reader_checkpoint, reader_profile_id
        self.native, self.output_budgets, self.replan = native, tuple(output_budgets), replan
        self.output_tokens_per_fact = output_tokens_per_fact
        if not self.output_budgets or any(value < 1 for value in self.output_budgets):
            raise ValueError("rolling READ output budgets must be declared positive choices")

    async def start(self):
        await self.native.start()

    def bind_resident_models(self, manifest):
        self.native.bind_resident_models(manifest)

    async def aclose(self):
        await self.native.aclose()

    def proposals(self, context_plans, state, snapshot, template, compile_answers):
        plans = []
        for original in context_plans:
            plan = _with_observed_sources(original, state)
            from .evidence_planner import render_selection
            plan = replace(plan, selected_set=replace(plan.selected_set, token_cost=len(template.tokenizer.encode(
                render_selection(plan.selected_set.selected_units), add_special_tokens=False))))
            plan = replace(plan, steps=(*plan.steps, *_reduction_tasks(plan, state)))
            if plan.coverage_scope == "all_supplied_documents":
                kind = "scope_scan"
            elif any(task.input_claim_ids or task.predecessor_step_ids for task in plan.steps if task.kind == "READ"):
                kind = "multihop"
            else:
                kind = "complementary_evidence" if plan.steps else "direct_answer"
            candidate = complete_plan(plan, (), semantic_kind=kind, reader_checkpoint=self.reader_checkpoint)
            completed = _completed_steps(candidate, state)
            ready = set(completed)
            steps = []
            units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
            dependency_ids = {identity for unit in units.values() for identity in unit.dependencies}
            for step in candidate.steps:
                task = step.task
                variants = ()
                reservation = read_output_reservation(plan.task_state, task,
                    tuple(units[identity] for identity in task.input_unit_ids), template.tokenizer,
                    dependency_ids=dependency_ids, output_tokens_per_fact=self.output_tokens_per_fact,
                    reading_mode=plan.reading_mode)
                if task.step_id not in completed and set(task.predecessor_step_ids) <= ready:
                    if task.kind == "ANSWER":
                        variants = compile_answers(snapshot, (plan,), replace(state, plan=plan))
                    elif not (plan.reading_mode == "typed_reduce" and task.kind == "REDUCE"
                              and task.operation in {"union", "count", "sum", "group"}):
                        outputs = tuple(value for value in self.output_budgets
                                        if reservation <= value <= task.output_budget)
                        variants = compile_step(plan, task, state, snapshot, template, outputs=outputs)
                steps.append(PlanStep(task, variants, reservation))
            plans.append(replace(candidate, steps=tuple(steps), unmet_answer_points=tuple(
                need.need_id for need in plan.task_state.needs
                if not any(claim.need_id == need.need_id for claim in state.claims))))
        # Direct ANSWER is an explicit stop action from the same observed state.
        # Full-range feasibility is checked against state.plan.coverage_units.
        if not any(plan.semantic_kind == "direct_answer" for plan in plans):
            source = plans[0].context_plan
            direct = replace(source, plan_id="direct:"+source.plan_id, steps=())
            plans.insert(0, complete_plan(direct, compile_answers(snapshot, (direct,), replace(state, plan=direct)),
                semantic_kind="direct_answer", reader_checkpoint=self.reader_checkpoint))
        return tuple(plans)

    async def run(self, attempt_id, snapshot, context_plans, executor, engine, template,
                  cost_model, compile_answers, *, arrived, auxiliary_usage, events, open_need_provider=None):
        context_plans = tuple(namespace_plan(plan) for plan in context_plans)
        if len(context_plans)+int(all(plan.steps for plan in context_plans)) > self.policy.limits.max_plan_candidates:
            raise ValueError("rolling execution reserves one finite candidate for direct ANSWER")
        saved = tuple(executor.store.execution_step(attempt_id, record.step_id)
                      for record in executor.store.execution_steps(attempt_id))
        records = tuple(record for stage, record, _ in saved if stage == "COMPLETE")
        graph = ClaimGraph(claim for record in records for claim in record.result.claims)
        coverage = tuple((context_plans[0].coverage_scope, span.source_key.document_id,
                          unit.unit_id, record.step_id)
            for record in records for unit in context_plans[0].selected_set.selected_units
            if unit.unit_id in record.task.input_unit_ids for span in unit.source_spans)
        progress = executor.store.reading_progress(attempt_id)
        expanded = set(progress.expanded) if progress is not None else set()
        state = ReadingOutcome(context_plans[0], tuple(record.result for record in records), records,
            tuple(graph.claims.values()), coverage,
            tuple(need for record in records for need in record.result.open_needs),
            auxiliary_usage=progress.auxiliary_usage if progress is not None else dict(ZERO_AUX))
        if saved:
            arrivals = {record.metrics["request_arrived_monotonic"] for _, record, _ in saved}
            if len(arrivals) != 1 or None in arrivals:
                raise ValueError("resumed rolling cost requires its original observed request arrival")
            arrived = arrivals.pop()
            executor.request_arrived = arrived
            for stage, record, _ in saved:
                if stage != "COMPLETE":
                    variants = compile_step(state.plan, record.task, state, snapshot, template,
                                            outputs=(record.task.output_budget,))
                    state = await executor.execute_one(attempt_id, snapshot, state.plan, record.task, state,
                        initial_auxiliary_usage=auxiliary_usage, candidate=variants[0])
        decisions = []
        while True:
            selection_started = time.monotonic()
            plans = self.proposals(context_plans, state, snapshot, template, compile_answers)
            if len(plans) > self.policy.limits.max_plan_candidates:
                raise ValueError("rolling proposals exceed the configured finite candidate set")
            if any(plan.context_plan.task_state != state.plan.task_state for plan in plans):
                raise ValueError("semantic alternatives must share the current actual task binding")
            native, reusable = await self.native.observe(plans, state, snapshot, engine, cost_model,
                arrived, auxiliary_usage, selection_started=selection_started)
            decision = self.policy.choose(plans, state, native, reusable)
            audit = {"observed_at_ms": (time.monotonic()-arrived)*1000,
                "state_steps": tuple(record.step_id for record in state.steps),
                "paid_cost": native.paid_cost, "evaluations": self.policy.last_evaluations,
                "decision": decision, "quote": native.quotes,
                "physical_paid_observation": native.paid_observation,
                "current_resource_forecasts": native.resources,
                "future_step_forecasts": tuple({"plan_id": key[0], "step_id": key[1], "choices": value}
                    for key, value in native.future.items())}
            decisions.append(audit)
            events.emit("rolling_plan_decision", attempt_id=attempt_id, **audit)
            chosen = next(plan for plan in plans if plan.plan_id == decision.next_step.plan_id)
            plan = chosen.context_plan
            if decision.next_step.task.kind == "ANSWER":
                required = set(state.plan.coverage_units) if state.plan.coverage_scope == "all_supplied_documents" else set()
                covered = {identity for result in state.results for identity in result.covered_units}
                if not required <= covered:
                    raise ValueError("terminal ANSWER has not completed the user's required scope")
                spent = native.paid_cost.input_tokens+native.paid_cost.output_tokens
                plan = replace(plan, online_tokens_spent=spent)
                outcome = replace(state, plan=plan)
                candidate = decision.next_step.candidate
                quote = next(row for row in native.quotes.quotes if row.candidate_id == candidate.candidate_id)
                estimate = native.cost_model.predict_native(replace(candidate, predicted_read_ms=0.0,
                    auxiliary_ms=0.0), quote, snapshot, batch=native.quotes,
                    concurrency=native.quotes.active_requests+1)
                selected = Plan(candidate, predicted=(estimate,), quote_seq=native.quotes.observed_seq,
                                reason="rolling_plan_gain")
                return RollingResult(outcome, selected, native.quotes, tuple(decisions), selection_started)
            state = await executor.execute_one(attempt_id, snapshot, plan, decision.next_step.task, state,
                initial_auxiliary_usage=auxiliary_usage, candidate=decision.next_step.candidate)
            if self.replan is not None:
                context_plans, usage = await self.replan(context_plans, state, snapshot)
                context_plans = tuple(namespace_plan(plan) for plan in context_plans)
                from .execution import merge_auxiliary_usage
                from .session_store import ReadingProgress
                state = replace(state, auxiliary_usage=merge_auxiliary_usage(state.auxiliary_usage, usage))
                executor.store.save_reading_progress(attempt_id, state.plan,
                    ReadingProgress(tuple(sorted(expanded)), (), state.auxiliary_usage))
            elif open_need_provider is not None and state.plan.coverage_scope != "all_supplied_documents":
                needs = tuple(need for need in state.open_needs
                    if (need.need_id, need.missing_relation, need.bound_entities) not in expanded)
                if needs:
                    from .context_schema import EvidencePool
                    from .evidence_planner import close_dependencies
                    from .execution import merge_auxiliary_usage, require_online_tokens
                    from .reading import build_read_plan
                    from .session_store import ReadingProgress
                    require_online_tokens(plan.budget.max_total_tokens, state.plan.online_tokens_spent, 1, "OpenNeed")
                    tick = time.monotonic()
                    # A's gap interface counts selected units as already read.
                    # A rolling branch may still have unexecuted initial
                    # windows; only completed material consumes that allowance.
                    read_ids = tuple(dict.fromkeys(identity for result in state.results
                                                   for identity in result.covered_units))
                    by_id = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
                    read_units = close_dependencies(read_ids, by_id)
                    from .evidence_planner import render_selection
                    read_base = replace(state.plan, selected_set=replace(state.plan.selected_set,
                        selected_units=read_units, token_cost=len(template.tokenizer.encode(
                            render_selection(read_units), add_special_tokens=False))))
                    pool = await open_need_provider(read_base, needs, state.claims)
                    if not isinstance(pool, EvidencePool):
                        raise TypeError("rolling gap proposals require EvidencePool with real auxiliary usage")
                    elapsed = (time.monotonic()-tick)*1000
                    expanded.update((need.need_id, need.missing_relation, need.bound_entities) for need in needs)
                    state = replace(state, auxiliary_usage=merge_auxiliary_usage(state.auxiliary_usage,
                                                                                pool.auxiliary_usage))
                    phases = pool.auxiliary_ms or (("control", elapsed),)
                    state = replace(state, plan=replace(state.plan, auxiliary_ms=(*state.plan.auxiliary_ms,
                        *((f"gap:{len(expanded)}:{key}", value) for key, value in phases))))
                    executor.store.save_reading_progress(attempt_id, state.plan,
                        ReadingProgress(tuple(sorted(expanded)), (), state.auxiliary_usage))
                    previous = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
                    all_units = {**previous, **{unit.unit_id: unit for unit in pool.units}}
                    covered = {identity for result in state.results for identity in result.covered_units}
                    fresh = tuple(unit.unit_id for unit in pool.units if unit.unit_id not in covered)
                    closure = close_dependencies(fresh, all_units)
                    if closure:
                        blocks = {block.key: block for block in (*state.plan.selected_set.source_blocks,
                                                                *pool.source_blocks)}
                        selected = replace(pool.selected_set or state.plan.selected_set,
                            selected_units=closure, source_blocks=tuple(blocks.values()))
                        parents = tuple(dict.fromkeys(parent for need in needs for parent in need.parent_claim_ids))
                        claims = {claim.claim_id: claim for claim in state.claims}
                        gap = build_read_plan(state.plan.task_state, selected, state.plan.coverage_scope,
                            state.plan.budget, template=template, snapshot=snapshot,
                            input_claims=tuple(claims[parent] for parent in parents),
                            reading_mode=state.plan.reading_mode, separate_needs=state.plan.separate_needs,
                            numeric_identity=state.plan.numeric_identity, read_layout=state.plan.read_layout,
                            read_packing=state.plan.read_packing)
                        predecessors = tuple(dict.fromkeys(parent.rsplit(":claim:", 1)[0] for parent in parents))
                        gap = replace(gap, plan_id=f"gap:{len(expanded)}:{state.plan.plan_id}",
                            steps=tuple(record.task for record in state.steps)+tuple(replace(task,
                                predecessor_step_ids=predecessors) for task in gap.steps),
                            selected_set=replace(selected, selected_units=tuple({**previous,
                                **{unit.unit_id: unit for unit in gap.selected_set.selected_units}}.values())),
                            auxiliary_ms=state.plan.auxiliary_ms)
                        gap = namespace_plan(gap)
                        context_plans = (gap, *tuple(item for item in context_plans
                            if item.plan_id != state.plan.plan_id)[:self.policy.limits.max_plan_candidates-2])
