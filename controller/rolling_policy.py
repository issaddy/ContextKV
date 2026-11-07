"""Execute finite complete plans, observing and reselecting after each step.

Profiles and paid resource observations are explicit inputs.  This module never
imports oracle evidence, estimates quality from support scores, or invents a
native quote for an input whose facts have not been generated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import time

from .claim_graph import ClaimGraph
from .context_schema import EvidencePool, ReadResult, ReadTask, ReadingOutcome, StepRecord
from .plan_gain_protocol import input_protocol
from .plan_policy import (CompleteCost, NoCompletePlan, PlanCandidate, PlanPolicy, PlanStep,
                          _completed_steps, complete_plan)
from .reading import (advance_read_plan, read_output_reservation, unresolved_needs)
from .schema import Plan
from .serialization import decode, dumps, to_jsonable


ZERO_AUX = {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []}


def _gap_retriever_identity(manifest):
    from .resident_models import resident_model_digest

    return {"implementation": "StudyEvidenceProvider.extend_for_plan",
            "resident_model_manifest_digest": resident_model_digest(manifest)}


def namespace_plan(plan):
    """Content identity prevents read:0 from aliasing another plan's input."""
    if plan.steps and all(task.step_id.startswith("policy:") or task.operation == "numeric_select"
                          for task in plan.steps):
        return plan
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    mapped, tasks = {}, []
    for task in plan.steps:
        if task.step_id.startswith("policy:") or task.operation == "numeric_select":
            mapped[task.step_id] = task.step_id
            tasks.append(task)
            continue
        claims = []
        for claim_id in task.input_claim_ids:
            prefix, separator, suffix = claim_id.rpartition(":claim:")
            claims.append(mapped.get(prefix, prefix)+separator+suffix if separator else claim_id)
        identity = hashlib.sha256(dumps((plan.task_state, plan.reading_mode, plan.read_layout,
            plan.read_packing, task.kind, task.operation, task.need_ids, task.shared_numeric_scopes,
            tuple(replace(units[key], retrieval_score=0.0, need_ids=()) for key in task.input_unit_ids), tuple(claims),
            tuple(mapped[key] for key in task.predecessor_step_ids))).encode()).hexdigest()[:24]
        mapped[task.step_id] = "policy:"+identity
        tasks.append(replace(task, step_id=mapped[task.step_id], output_schema={},
            input_claim_ids=tuple(claims),
            predecessor_step_ids=tuple(mapped[key] for key in task.predecessor_step_ids)))
    return replace(plan, steps=tuple(tasks))


def _reduction_tasks(plan, state):
    """Declare reductions, waiting for the selected frontier of shared numeric views."""
    from .reading import numeric_reduction_inputs, numeric_selection_step_id, numeric_selection_task

    if plan.reading_mode == "coverage_only":
        return ()
    read_ids = tuple(task.step_id for task in plan.steps if task.kind == "READ")
    if not read_ids:
        return ()
    summarize = all(task.operation == "summarize" for task in plan.steps if task.kind == "READ")
    completed = {result.step_id for result in state.results}
    completed.update(result.reduction.step_id for result in state.results if result.reduction is not None)
    shared_needs = {need_id for task in plan.steps if task.kind == "READ"
                    for scope in task.shared_numeric_scopes for need_id in scope.need_ids}
    planned_ids = {task.step_id for task in plan.steps}
    tasks = []
    for need in plan.task_state.needs:
        if not summarize and need.operation not in {"enumerate", "compare", "count", "sum", "group", "infer"}:
            continue
        shared_numeric = need.need_id in shared_needs
        if shared_numeric:
            selector_id = numeric_selection_step_id(plan, need.need_id)
            selector = numeric_selection_task(plan, state.results, state.claims, need_id=need.need_id)
            if selector is not None and selector.step_id not in planned_ids:
                tasks.append(selector)
            selection = next((result for result in reversed(state.results)
                if result.step_id == selector_id or
                (result.reduction is not None and result.reduction.step_id == selector_id)), None)
            if selection is None:
                continue
            tasks.extend(record.task for record in state.steps
                if record.step_id not in planned_ids and record.task.operation == "numeric_select"
                and (record.step_id == selection.step_id or record.step_id.startswith(selector_id+":part:")))
        predecessors = (selection.step_id,) if shared_numeric else read_ids
        if any(task.kind == "REDUCE" and need.need_id in task.need_ids
               and task.predecessor_step_ids == predecessors for task in plan.steps):
            continue
        operation = "summarize" if summarize else "union" if need.operation == "enumerate" else need.operation
        identity = hashlib.sha256(dumps((read_ids, need.need_id, operation)).encode()).hexdigest()[:24]
        if shared_numeric:
            claims = tuple(claim.claim_id for claim in numeric_reduction_inputs(plan, need.need_id, state.results))
        else:
            extracted = {claim.claim_id for result in state.results if result.step_id in read_ids
                         for claim in result.claims}
            claims = tuple(claim.claim_id for claim in state.claims
                           if claim.serves(need.need_id) and claim.kind == "extract"
                           and claim.claim_id in extracted)
        if not shared_numeric and set(read_ids) <= completed and not claims and need.operation != "group":
            continue
        from .context_budget import reduction_output_budget
        output_budget = reduction_output_budget(plan, need.need_id)
        tasks.append(ReadTask("policy:reduce:"+identity, "REDUCE", (need.need_id,),
            input_claim_ids=claims, predecessor_step_ids=predecessors,
            input_budget=plan.budget.window_tokens-output_budget,
            output_budget=output_budget, operation=operation,
            retained_conclusion_tokens=tuple((identity, tokens) for identity, tokens
                in plan.answer_allocation.need_tokens if identity == need.need_id)
                if plan.answer_allocation is not None else ()))
    return tuple(tasks)


def _with_observed_sources(plan, state):
    read_units = {identity for result in state.results for identity in result.covered_units}
    read_units.update(identity for record in state.steps for identity in record.task.input_unit_ids)
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
            plan.task_state, disposition, state.results) if state.claims else replace(plan.answer_plan, disposition=disposition))


def compile_step(plan, task, state, snapshot, template, *, outputs, request_compiler=None, cache_scope=None):
    from .request_compiler import RequestCompiler

    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    claims = {claim.claim_id: claim for claim in state.claims}
    selected_units = tuple(units[identity] for identity in task.input_unit_ids)
    selected_claims = tuple(claims[identity] for identity in task.input_claim_ids)
    compiler = request_compiler or RequestCompiler(template)
    return compiler.compile_read(plan, task, selected_units, selected_claims, snapshot,
                                 outputs=outputs, cache_scope=cache_scope)


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
                 output_budgets, limits=None, replan=None, output_tokens_per_fact=64,
                 mode="whole_plan", observation_path=None, gap_retriever_identity=None, gap_limits=None):
        from .plan_policy import PolicyLimits
        self.policy = PlanPolicy(gain_model, limits=limits or PolicyLimits())
        if (reader_checkpoint, reader_profile_id) != (gain_model.reader_checkpoint, gain_model.reader_profile_id):
            raise ValueError("rolling policy gain asset does not bind the configured reader/profile")
        self.reader_checkpoint, self.reader_profile_id = reader_checkpoint, reader_profile_id
        self.native, self.output_budgets, self.replan = native, tuple(output_budgets), replan
        self.output_tokens_per_fact = output_tokens_per_fact
        if mode not in {"whole_plan", "observed_actions"}:
            raise ValueError("rolling mode must be whole_plan or observed_actions")
        if mode == "observed_actions" and replan is not None:
            raise ValueError("observed actions use selected GAP retrieval instead of automatic replanning")
        if mode == "observed_actions":
            from .action_proposals import BOUNDED_CONTINUATION
            if (gain_model.provenance.get("continuation_protocol") != BOUNDED_CONTINUATION
                    or gain_model.provenance.get("continuation_gap_limits") != gap_limits):
                raise ValueError("rolling gain asset uses a different bounded continuation protocol")
        self.mode = mode
        self.observation_path = None if observation_path is None else Path(observation_path)
        self.gap_retriever_identity, self.gap_limits = gap_retriever_identity, gap_limits
        if not self.output_budgets or any(value < 1 for value in self.output_budgets):
            raise ValueError("rolling READ output budgets must be declared positive choices")

    async def start(self):
        await self.native.start()

    def bind_resident_models(self, manifest):
        if self.mode == "observed_actions":
            from .resident_models import evidence_selection_contract
            if evidence_selection_contract(manifest) != self.policy.gain_model.provenance.get(
                    "evidence_selection_contract"):
                raise ValueError("rolling gain asset uses a different evidence selection contract")
        self.native.bind_resident_models(manifest)
        if self.mode == "observed_actions" and self.gap_limits is not None:
            self.gap_retriever_identity = _gap_retriever_identity(manifest)

    async def aclose(self):
        await self.native.aclose()

    @staticmethod
    def _mandatory_coverage(state):
        from .context_budget import completed_source_coverage
        units = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
        return (state.plan.coverage_scope == "all_supplied_documents" and not completed_source_coverage(
            state.plan, state.results, tuple(units[identity] for identity in state.plan.coverage_units)))

    def _gap_request(self, state, expanded, provider):
        from .action_proposals import GapRetrievalRequest, gap_key, gap_query
        if (provider is None or state.plan.coverage_scope == "all_supplied_documents"
                or self.gap_retriever_identity is None or self.gap_limits is None):
            return None
        needs = tuple(need for need in state.open_needs if gap_key(need) not in expanded)
        return (GapRetrievalRequest(needs, gap_query(state.plan.task_state, needs),
                self.gap_retriever_identity, self.gap_limits) if needs else None)

    def _observe_state(self, attempt_id, executor, state, snapshot, native, gap_request, mandatory, *, metadata=None):
        if self.observation_path is None:
            return
        attempt = executor.store.get_attempt(attempt_id)
        packet = {"schema_version": 1, "attempt_id": attempt_id, "request_id": attempt.request_id,
            "working_state": state, "snapshot": snapshot, "paid_cost": native.paid_cost,
            "input_protocol": input_protocol(executor.template.fingerprint),
            "native_condition": native.quotes, "paid_observation": native.paid_observation,
            "reader_checkpoint": native.reader_checkpoint, "reader_profile_id": native.reader_profile_id,
            "gap_request": gap_request, "decision_mode": "mandatory_coverage" if mandatory else self.mode}
        packet["observation_id"] = f"{attempt_id}:{native.quotes.engine_restart_id}:{native.quotes.observed_seq}:{len(state.steps)}"
        packet.update(metadata or {})
        self.observation_path.parent.mkdir(parents=True, exist_ok=True)
        with self.observation_path.open("a") as stream:
            stream.write(dumps(packet)+"\n")

    def action_proposals(self, catalog, state, snapshot, template, compile_answers, gap_request, *,
                         request_compiler=None, cache_scope=None):
        from .action_proposals import build_observed_action_plans
        base = _with_observed_sources(catalog, state)
        plans, _ = build_observed_action_plans(replace(state, plan=base), snapshot, template,
            reader_checkpoint=self.reader_checkpoint, gap_request=gap_request, include_gap_step=True)
        result = []
        for plan in plans:
            completed = _completed_steps(plan, state)
            units = {unit.unit_id: unit for unit in plan.context_plan.selected_set.selected_units}
            dependencies = {identity for unit in units.values() for identity in unit.dependencies}
            steps = []
            for step in plan.steps:
                task, variants = step.task, ()
                reservation = read_output_reservation(plan.context_plan.task_state, task,
                    tuple(units[identity] for identity in task.input_unit_ids), template.tokenizer,
                    dependency_ids=dependencies, output_tokens_per_fact=self.output_tokens_per_fact,
                    reading_mode=plan.context_plan.reading_mode)
                if task.step_id not in completed and set(task.predecessor_step_ids) <= completed:
                    if task.kind == "ANSWER":
                        variants = compile_answers(snapshot, (plan.context_plan,), replace(state, plan=plan.context_plan))
                    elif task.kind != "GAP" and not (task.kind == "REDUCE"
                            and plan.context_plan.reading_mode == "typed_reduce"
                            and task.operation in {"union", "count", "sum", "group"}):
                        outputs = tuple(value for value in self.output_budgets if reservation <= value <= task.output_budget)
                        variants = compile_step(plan.context_plan, task, state, snapshot, template, outputs=outputs,
                            request_compiler=request_compiler, cache_scope=cache_scope)
                steps.append(PlanStep(task, variants, reservation))
            result.append(replace(plan, steps=tuple(steps), unmet_answer_points=tuple(
                need.need_id for need in base.task_state.needs if not any(claim.serves(need.need_id) for claim in state.claims))))
        return tuple(result)

    @staticmethod
    def _gap_usage(executor, attempt_id):
        from .execution import merge_auxiliary_usage
        usage = [record.metrics["auxiliary_usage"] for stage, record in
            executor.store.execution_steps_with_stage(attempt_id)
            if record.task.kind == "GAP" and stage in {"RESULT_READY", "COMPLETE"}]
        return merge_auxiliary_usage(dict(ZERO_AUX), *usage)

    async def _finish_gap(self, record, pool, state, snapshot, template, executor, expanded, events):
        from .action_proposals import GapRetrievalRequest, gap_retrieval_plan
        from .context_budget import bind_answer_window
        from .evidence_planner import _passages, close_dependencies
        from .execution import ReadingBudgetExceeded
        from .reading import build_read_plan
        from .session_store import ReadingProgress
        request = decode(GapRetrievalRequest, record.metrics["gap_request"])
        read_base = gap_retrieval_plan(state, request)
        prior_passages = _passages(read_base.selected_set.selected_units)
        if (len(pool.units) > request.limits["max_units"]
                or len(_passages(pool.units)-prior_passages) > request.limits["max_passages"]):
            raise ReadingBudgetExceeded("selected GAP result exceeds its registered retrieval limits")
        old = state.plan
        units = {unit.unit_id: unit for unit in (*old.selected_set.selected_units, *pool.units)}
        closure = close_dependencies(tuple(unit.unit_id for unit in pool.units), units)
        read_ids = {identity for result in state.results for identity in result.covered_units}
        novel = tuple(unit for unit in closure if unit.unit_id not in read_ids)
        if old.budget.max_passages is not None and len(_passages(tuple(units.values()))) > old.budget.max_passages:
            raise ReadingBudgetExceeded("selected GAP exceeds the cumulative original passage allocation")
        blocks = tuple({block.key: block for block in (*old.selected_set.source_blocks, *pool.source_blocks)}.values())
        supports = old.selected_set.need_support
        if pool.selected_set is not None:
            needs = {need.need_id for need in request.open_needs}
            supports = tuple(value for value in supports if value.need_id not in needs)+tuple(
                value for value in pool.selected_set.need_support if value.need_id in needs)
        selected = replace(pool.selected_set or old.selected_set, selected_units=tuple(units.values()),
                           source_blocks=blocks, need_support=supports)
        completed = {result.step_id for result in state.results}
        pending = tuple(task for task in old.steps if task.step_id not in completed | {record.step_id})
        fresh = ()
        added_coverage = ()
        if novel:
            claims = {claim.claim_id: claim for claim in state.claims}
            parents = tuple(dict.fromkeys(parent for need in request.open_needs for parent in need.parent_claim_ids))
            gap = build_read_plan(old.task_state, replace(selected, selected_units=closure), old.coverage_scope,
                old.budget, template=template, snapshot=snapshot, input_claims=tuple(claims[parent] for parent in parents),
                reading_mode=old.reading_mode, separate_needs=old.separate_needs,
                numeric_identity=old.numeric_identity, read_layout=old.read_layout, read_packing=old.read_packing,
                answer_system_prompt=old.answer_system_prompt)
            fresh = tuple(replace(task, step_id="gapread:"+record.step_id+":"+task.step_id, input_claim_ids=parents,
                predecessor_step_ids=tuple(dict.fromkeys((record.step_id, *(parent.rsplit(":claim:", 1)[0]
                    for parent in parents))))) for task in gap.steps)
            added_coverage = gap.coverage_units
            selected = replace(selected, selected_units=tuple({**{unit.unit_id: unit for unit in old.selected_set.selected_units},
                **{unit.unit_id: unit for unit in gap.selected_set.selected_units}}.values()))
        complete = record
        records = (*state.steps, complete)
        imported = tuple(task for task in old.steps if task.kind == "READ" and task.step_id in completed
                         and task.step_id not in {item.step_id for item in records})
        plan = replace(old, selected_set=selected, steps=(*imported, *tuple(item.task for item in records), *fresh, *pending),
            coverage_units=tuple(dict.fromkeys((*old.coverage_units, *added_coverage))), answer_allocation=None,
            auxiliary_ms=(*tuple(value for value in old.auxiliary_ms if value[0] != "gap:"+record.step_id),
                          ("gap:"+record.step_id, record.result.elapsed_ms)))
        plan = namespace_plan(bind_answer_window(plan, template, snapshot,
            preserve_step_ids=tuple(item.step_id for item in records)))
        # Preserve the returned tool material in the existing step ledger for restart recovery.
        usage = self._gap_usage(executor, record.attempt_id)
        state = replace(state, plan=plan, steps=(*state.steps, complete),
                        results=(*state.results, complete.result), auxiliary_usage=usage)
        state = replace(state, plan=_with_observed_sources(plan, state))
        executor.store.save_reading_progress(record.attempt_id, state.plan,
            ReadingProgress(tuple(sorted(expanded)), (), usage), completed_step=complete)
        events.emit("rolling_gap_complete", attempt_id=record.attempt_id, step=complete,
                    returned_unit_ids=tuple(unit.unit_id for unit in pool.units))
        return state

    async def _run_gap(self, attempt_id, task, request, state, snapshot, template, executor, expanded,
                       provider, events, arrived):
        from .action_proposals import gap_key, gap_retrieval_plan
        from .evidence_planner import _passages
        from .execution import require_online_tokens
        from .session_store import ReadingProgress
        tick = time.monotonic()
        require_online_tokens(state.plan.budget.max_total_tokens, state.plan.online_tokens_spent, 1, "selected GAP")
        record = StepRecord(attempt_id, task.step_id, task, result=ReadResult(task.step_id),
            metrics={"gap_request": to_jsonable(request), "request_arrived_monotonic": arrived,
                     "phase_started_monotonic": tick})
        executor.store.save_execution_step(record, "SUBMITTED")
        expanded.update(gap_key(need) for need in request.open_needs)
        executor.store.save_reading_progress(attempt_id, state.plan,
            ReadingProgress(tuple(sorted(expanded)), (), state.auxiliary_usage))
        events.emit("rolling_gap_selected", attempt_id=attempt_id, request=request, step_id=task.step_id)
        read_base = gap_retrieval_plan(state, request)
        pool = await provider(read_base, request.open_needs, state.claims)
        ended = time.monotonic()
        record = replace(record, result=ReadResult(task.step_id, raw_output=dumps(pool), elapsed_ms=(ended-tick)*1000),
            metrics={**record.metrics, "auxiliary_usage": pool.auxiliary_usage,
                "phase_window": {"start": tick, "end": ended, "clock": "client_monotonic", "scope": "gap_retrieval"}})
        executor.store.save_execution_step(record, "RESULT_READY")
        prior_passages = _passages(read_base.selected_set.selected_units)
        if (len(pool.units) > request.limits["max_units"]
                or len(_passages(pool.units)-prior_passages) > request.limits["max_passages"]):
            from .execution import ReadingBudgetExceeded

            raise ReadingBudgetExceeded("actual GAP material exceeds its registered retrieval limits")
        return await self._finish_gap(record, pool, state, snapshot, template, executor, expanded, events)

    def proposals(self, context_plans, state, snapshot, template, compile_answers, *,
                  request_compiler=None, cache_scope=None):
        plans = []
        for original in context_plans:
            plan = _with_observed_sources(original, state)
            from .context_budget import AnswerWindowExceeded, bind_answer_window
            try:
                plan = bind_answer_window(plan, template, snapshot,
                    preserve_step_ids=tuple(result.step_id for result in state.results))
            except AnswerWindowExceeded:
                # Window exhaustion belongs to this alternative. Keep evaluating
                # other proposals, while propagating malformed source/task errors.
                continue
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
                        variants = compile_step(plan, task, state, snapshot, template, outputs=outputs,
                            request_compiler=request_compiler, cache_scope=cache_scope)
                steps.append(PlanStep(task, variants, reservation))
            plans.append(replace(candidate, steps=tuple(steps), unmet_answer_points=tuple(
                need.need_id for need in plan.task_state.needs
                if not any(claim.serves(need.need_id) for claim in state.claims))))
        # Direct ANSWER is an explicit stop action from the same observed state.
        # Full-range feasibility is checked against state.plan.coverage_units.
        if not plans:
            raise NoCompletePlan("no proposal fits its ANSWER window allocation")
        if not any(plan.semantic_kind == "direct_answer" for plan in plans):
            source = plans[0].context_plan
            direct = replace(source, plan_id="direct:"+source.plan_id, steps=())
            plans.insert(0, complete_plan(direct, compile_answers(snapshot, (direct,), replace(state, plan=direct)),
                semantic_kind="direct_answer", reader_checkpoint=self.reader_checkpoint))
        return tuple(plans)

    def _required_scope_step(self, state):
        plan = state.plan
        full_scope = plan.coverage_scope == "all_supplied_documents"
        numeric_operations = {"count", "sum", "group"}
        if not full_scope:
            if plan.reading_mode != "typed_reduce" or not any(
                    need.operation in numeric_operations for need in plan.task_state.needs):
                return None
            completed = {result.step_id for result in state.results}
            if any(task.kind == "READ" and task.step_id not in completed for task in plan.steps):
                return None
        ready = advance_read_plan(plan, state.results)
        if self._mandatory_coverage(state):
            task = next((task for task in ready if task.kind == "READ"), None)
            if task is None:
                raise NoCompletePlan("no ready READ can complete the required source scope")
            return task
        required_operations = numeric_operations | {"numeric_select"}
        if full_scope:
            required_operations.add("union")
        return next((task for task in ready if task.kind == "REDUCE"
                     and plan.reading_mode == "typed_reduce"
                     and task.operation in required_operations), None)

    async def run(self, attempt_id, snapshot, context_plans, executor, engine, template,
                  cost_model, compile_answers, *, arrived, auxiliary_usage, events, open_need_provider=None,
                  initial_outcomes=()):
        from .execution import seed_reading_outcome

        context_plans = tuple(namespace_plan(plan) for plan in context_plans)
        if len(context_plans)+int(all(plan.steps for plan in context_plans)) > self.policy.limits.max_plan_candidates:
            raise ValueError("rolling execution reserves one finite candidate for direct ANSWER")
        saved = tuple(executor.store.execution_step(attempt_id, record.step_id)
                      for record in executor.store.execution_steps(attempt_id))
        if self.mode == "observed_actions" and saved:
            catalog = executor.store.context_plan(attempt_id)
            context_plans = (namespace_plan(catalog),)
        if saved:
            arrivals = {record.metrics["request_arrived_monotonic"] for _, record, _ in saved}
            if len(arrivals) != 1 or None in arrivals:
                raise ValueError("resumed rolling cost requires its original observed request arrival")
            arrived = arrivals.pop()
            executor.request_arrived = arrived
            catalog = executor.store.context_plan(attempt_id)
            finished = []
            for stage, record, generation in saved:
                if stage == "COMPLETE" and record.engine_epoch is not None:
                    record = await executor.finish_completed_step(catalog, record)
                finished.append((stage, record, generation))
            saved = tuple(finished)
        records = tuple(record for stage, record, _ in saved if stage == "COMPLETE")
        supplied = next((outcome for outcome in initial_outcomes
                         if outcome.plan.plan_id == context_plans[0].plan_id), None)
        seed = seed_reading_outcome(executor.store, attempt_id, context_plans[0], supplied)
        graph = ClaimGraph(seed.claims)
        for record in records:
            for claim in record.result.claims:
                graph.add(claim)
        results = (*seed.results, *(record.result for record in records))
        coverage = (*seed.coverage, *((context_plans[0].coverage_scope, span.source_key.document_id,
                          unit.unit_id, record.step_id)
            for record in records for unit in context_plans[0].selected_set.selected_units
            if unit.unit_id in record.task.input_unit_ids for span in unit.source_spans))
        progress = executor.store.reading_progress(attempt_id)
        expanded = set(progress.expanded) if progress is not None else set()
        state = ReadingOutcome(seed.plan, results, records,
            tuple(graph.claims.values()), tuple(dict.fromkeys(coverage)), unresolved_needs(results),
            auxiliary_usage=progress.auxiliary_usage if progress is not None else dict(ZERO_AUX))
        state = replace(state, plan=_with_observed_sources(state.plan, state))
        executor.store.save_context_plan(attempt_id, state.plan, initial_outcome=state)
        if self.mode == "observed_actions":
            state = replace(state, auxiliary_usage=self._gap_usage(executor, attempt_id))
        if saved:
            for stage, record, _ in saved:
                if stage != "COMPLETE":
                    if self.mode == "observed_actions" and record.task.kind == "GAP":
                        if stage == "RESULT_READY":
                            import json
                            pool = decode(EvidencePool, json.loads(record.result.raw_output))
                            state = await self._finish_gap(record, pool, state, snapshot, template,
                                executor, expanded, events)
                            context_plans = (state.plan,)
                            continue
                        raise ValueError("selected GAP was already submitted; it will not be paid again")
                    variants = compile_step(state.plan, record.task, state, snapshot, template,
                        outputs=(record.task.output_budget,), request_compiler=executor.request_compiler,
                        cache_scope=await engine.info())
                    state = await executor.execute_one(attempt_id, snapshot, state.plan, record.task, state,
                        initial_auxiliary_usage=auxiliary_usage, candidate=variants[0])
        decisions = []
        while True:
            selection_started = time.monotonic()
            cache_scope = await engine.info()
            observed = self.mode == "observed_actions"
            required_step = self._required_scope_step(state) if observed else None
            if (required_step is not None and required_step.kind == "READ"
                    and not any(need.operation in {"count", "sum", "group"} for need in state.plan.task_state.needs)):
                from .context_budget import direct_answer_reason
                direct_candidates = compile_answers(snapshot, (state.plan,), state)
                if any(candidate.feasible and direct_answer_reason(state, candidate) is None
                       for candidate in direct_candidates):
                    required_step = None
            if required_step is not None:
                from .action_policy import native_admission_reason
                from .execution import reduction_partitions
                from .session_store import ReadingProgress

                plan = state.plan
                if required_step.operation == "numeric_select":
                    claims = {claim.claim_id: claim for claim in state.claims}
                    parts = reduction_partitions(plan, required_step,
                        tuple(claims[identity] for identity in required_step.input_claim_ids), snapshot, template)
                    tasks = {task.step_id: task for task in plan.steps}
                    tasks.update((task.step_id, task) for task, _ in parts)
                    plan = replace(plan, steps=tuple(tasks.values()))
                    state = replace(state, plan=plan)
                    completed = {result.step_id for result in state.results}
                    required_step = next(task for task, _ in parts if task.step_id not in completed)
                cpu = required_step.kind == "REDUCE" and required_step.operation != "numeric_select"
                variants = () if cpu else compile_step(plan, required_step, state, snapshot, template,
                    outputs=(required_step.output_budget,), request_compiler=executor.request_compiler,
                    cache_scope=cache_scope)
                if variants and not any(candidate.feasible for candidate in variants):
                    raise NoCompletePlan("required source-scope step exceeds its model window")
                proposal = PlanCandidate(plan.plan_id, "scope_scan", plan,
                    (PlanStep(required_step, variants),), self.reader_checkpoint)
                native, _ = await self.native.observe((proposal,), state, snapshot, engine, cost_model,
                    arrived, auxiliary_usage, selection_started=selection_started)
                candidate = None if cpu else executor.request_compiler.choose_variant(variants, native.quotes)
                reserved = CompleteCost(0, 0, 0, 0 if cpu else len(candidate.input_ids),
                                        plan.budget.output_tokens+(0 if cpu else candidate.output_budget))
                reason = self.policy._constraint(native.paid_cost+reserved, plan.budget)
                if candidate is not None:
                    reason = reason or native_admission_reason(native, candidate)
                if reason is not None:
                    raise NoCompletePlan(f"required source-scope step exceeds {reason}")
                self._observe_state(attempt_id, executor, state, snapshot, native, None, True)
                audit = {"decision_mode": "mandatory_coverage", "step_id": required_step.step_id,
                    "task": required_step, "quote": native.quotes, "paid_cost": native.paid_cost,
                    "physical_paid_observation": native.paid_observation}
                decisions.append(audit)
                events.emit("rolling_required_scope_step", attempt_id=attempt_id, **audit)
                state = await executor.execute_one(attempt_id, snapshot, plan, required_step, state,
                    initial_auxiliary_usage=auxiliary_usage, candidate=candidate, quote=native.quotes,
                    quote_ms=(native.paid_observation or {}).get("native_quote_ms"))
                context_plans = (state.plan,)
                executor.store.save_reading_progress(attempt_id, state.plan,
                    ReadingProgress(tuple(sorted(expanded)), (), state.auxiliary_usage))
                continue
            mandatory = self._mandatory_coverage(state)
            gap_request = self._gap_request(state, expanded, open_need_provider)
            plans = (self.action_proposals(context_plans[0], state, snapshot, template,
                compile_answers, gap_request, request_compiler=executor.request_compiler,
                cache_scope=cache_scope) if observed else
                self.proposals(context_plans, state, snapshot, template, compile_answers,
                    request_compiler=executor.request_compiler, cache_scope=cache_scope))
            if len(plans) > self.policy.limits.max_plan_candidates:
                raise ValueError("rolling proposals exceed the configured finite candidate set")
            if any(plan.context_plan.task_state != state.plan.task_state for plan in plans):
                raise ValueError("semantic alternatives must share the current actual task binding")
            native, reusable = await self.native.observe(plans, state, snapshot, engine, cost_model,
                arrived, auxiliary_usage, selection_started=selection_started)
            self._observe_state(attempt_id, executor, state, snapshot, native, gap_request, mandatory)
            control_forecast = getattr(executor.observe_prefix, "predicted_control_ms", None)
            read_control = ({"predicted_ms": control_forecast(), "source": "actual_prefix_control_ms",
                             "observation_count": executor.observe_prefix.control_observation_count,
                             "scope": "first_READ_only"}
                if control_forecast is not None and state.plan.read_layout == "source_first" else None)
            try:
                decision = (self.policy.choose_observed_actions(plans, state, native, reusable, read_control=read_control)
                    if observed else self.policy.choose(plans, state, native, reusable))
            except NoCompletePlan:
                events.emit("rolling_plan_no_complete", attempt_id=attempt_id,
                    observed_at_ms=(time.monotonic()-arrived)*1000,
                    state_steps=tuple(record.step_id for record in state.steps),
                    paid_cost=native.paid_cost, evaluations=self.policy.last_evaluations,
                    option_diagnostics=self.policy.last_option_diagnostics,
                    quote=native.quotes, physical_paid_observation=native.paid_observation,
                    current_resource_forecasts=native.resources,
                    future_step_forecasts=tuple({"plan_id": key[0], "step_id": key[1], "choices": value}
                        for key, value in native.future.items()),
                    candidate_steps=tuple({"plan_id": plan.plan_id, "steps": tuple({
                        "task": step.task, "output_reservation": step.output_reservation,
                        "variants": tuple({"candidate_id": candidate.candidate_id,
                            "generation_kind": candidate.generation_kind,
                            "input_tokens": len(candidate.input_ids), "output_budget": candidate.output_budget,
                            "feasible": candidate.feasible} for candidate in step.variants)}
                        for step in plan.steps)} for plan in plans))
                raise
            audit = {"observed_at_ms": (time.monotonic()-arrived)*1000,
                "decision_mode": "mandatory_coverage" if mandatory else self.mode,
                "gap_request": gap_request,
                "state_steps": tuple(record.step_id for record in state.steps),
                "paid_cost": native.paid_cost, "evaluations": self.policy.last_evaluations,
                "option_diagnostics": self.policy.last_option_diagnostics,
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
                from .context_budget import direct_answer_reason
                reason = direct_answer_reason(state, decision.next_step.candidate)
                if reason is not None:
                    raise ValueError(f"terminal ANSWER has not completed its evidence: {reason}")
                spent = native.paid_cost.input_tokens+native.paid_cost.output_tokens
                plan = replace(plan, online_tokens_spent=spent)
                outcome = replace(state, plan=plan)
                candidate = decision.next_step.candidate
                estimates = ()
                if not observed:
                    quote = next(row for row in native.quotes.quotes if row.candidate_id == candidate.candidate_id)
                    estimates = (native.cost_model.predict_native(replace(candidate, predicted_read_ms=0.0,
                        auxiliary_ms=0.0), quote, snapshot, batch=native.quotes,
                        concurrency=native.quotes.active_requests+1),)
                selected = Plan(candidate, predicted=estimates, quote_seq=native.quotes.observed_seq,
                    reason="rolling_observed_action_gain_cost" if observed else "rolling_plan_gain")
                return RollingResult(outcome, selected, native.quotes, tuple(decisions), selection_started)
            if self.mode == "observed_actions" and decision.next_step.task.kind == "GAP":
                catalog = _with_observed_sources(context_plans[0], state)
                catalog = replace(catalog, online_tokens_spent=native.paid_cost.input_tokens+native.paid_cost.output_tokens)
                state = await self._run_gap(attempt_id, decision.next_step.task, gap_request,
                    replace(state, plan=catalog), snapshot, template, executor, expanded,
                    open_need_provider, events, arrived)
                context_plans = (state.plan,)
                continue
            execution_plan = _with_observed_sources(context_plans[0], state) if self.mode == "observed_actions" else plan
            state = await executor.execute_one(attempt_id, snapshot, execution_plan, decision.next_step.task, state,
                initial_auxiliary_usage=auxiliary_usage, candidate=decision.next_step.candidate, quote=native.quotes,
                quote_ms=(native.paid_observation or {}).get("native_quote_ms"))
            if self.mode == "observed_actions":
                from .session_store import ReadingProgress
                context_plans = (state.plan,)
                executor.store.save_reading_progress(attempt_id, state.plan,
                    ReadingProgress(tuple(sorted(expanded)), (), state.auxiliary_usage))
                continue
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
                    if (need.need_id, need.missing_relation, need.bound_entities, need.temporal_scope) not in expanded)
                if needs:
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
                    expanded.update((need.need_id, need.missing_relation, need.bound_entities,
                                     need.temporal_scope) for need in needs)
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
                            read_packing=state.plan.read_packing,
                            answer_system_prompt=state.plan.answer_system_prompt)
                        predecessors = tuple(dict.fromkeys(parent.rsplit(":claim:", 1)[0] for parent in parents))
                        gap = replace(gap, plan_id=f"gap:{len(expanded)}:{state.plan.plan_id}",
                            steps=tuple(record.task for record in state.steps)+tuple(replace(task,
                                predecessor_step_ids=predecessors) for task in gap.steps),
                            selected_set=replace(selected, selected_units=tuple({**previous,
                                **{unit.unit_id: unit for unit in gap.selected_set.selected_units}}.values())),
                            auxiliary_ms=state.plan.auxiliary_ms)
                        from .context_budget import bind_answer_window
                        gap = bind_answer_window(replace(gap, answer_allocation=None), template, snapshot,
                            preserve_step_ids=tuple(record.step_id for record in state.steps))
                        gap = namespace_plan(gap)
                        context_plans = (gap, *tuple(item for item in context_plans
                            if item.plan_id != state.plan.plan_id)[:self.policy.limits.max_plan_candidates-2])


class FixedReadingObserver:
    """Capture real fixed-plan decision states without choosing an action."""
    mode = "fixed_reading"

    def __init__(self, settings, paid, fingerprint):
        self.paid, self.fingerprint = paid, fingerprint
        self.reader_checkpoint = settings["reader_checkpoint"]
        self.reader_profile_id = settings["reader_profile_id"]
        self.observation_path = Path(settings["observation_path"])
        self.output_budgets = tuple(settings["output_budgets"])
        self.output_tokens_per_fact = settings.get("output_tokens_per_fact", 64)
        self.gap_limits = settings["gap_limits"]
        self.gap_retriever_identity = None
        self.cohort = {"cohort_split": settings["cohort_split"],
            "cohort_manifest_sha256": hashlib.sha256(Path(settings["cohort_manifest"]).read_bytes()).hexdigest()}

    def bind_resident_models(self, manifest):
        self.gap_retriever_identity = _gap_retriever_identity(manifest)

    async def start(self):
        await self.paid.start()

    async def aclose(self):
        await self.paid.aclose()

    async def observe(self, attempt_id, executor, state, snapshot, engine, cost_model, compile_answers,
                      *, arrived, auxiliary_usage, expanded, startup_ms):
        from .plan_policy import NativePlanningState
        started = time.monotonic()
        from .context_budget import direct_answer_reason
        direct = compile_answers(snapshot, (state.plan,), state)
        direct_eligible = any(candidate.feasible and direct_answer_reason(state, candidate) is None
                              for candidate in direct)
        mandatory = RollingPlanRuntime._mandatory_coverage(state) and not direct_eligible
        gap = RollingPlanRuntime._gap_request(self, state, expanded, executor.extend_open_needs)
        plans = (() if mandatory else RollingPlanRuntime.action_proposals(self, state.plan, state,
            snapshot, executor.template, compile_answers, gap, request_compiler=executor.request_compiler,
            cache_scope=await engine.info()))
        candidates = tuple(candidate for plan in plans for step in plan.steps for candidate in step.variants
                           if candidate.feasible)
        quote_started = time.monotonic()
        quotes = await engine.quote_batch(candidates, self.fingerprint)
        quote_ended = time.monotonic()
        paid = await self.paid.observe(arrived, state, auxiliary_usage)
        native = NativePlanningState(self.reader_checkpoint, self.reader_profile_id, snapshot, quotes,
            cost_model, {}, {}, paid, self.paid.last_observation)
        RollingPlanRuntime._observe_state(self, attempt_id, executor, state, snapshot, native, gap, mandatory,
            metadata={**self.cohort, "decision_seq": len(state.steps),
                "resident_model_manifest_digest": (self.gap_retriever_identity["resident_model_manifest_digest"]
                    if self.gap_retriever_identity is not None else None),
                "after_step_id": state.steps[-1].step_id if state.steps else None,
                "after_step_kind": state.steps[-1].task.kind if state.steps else None,
                # Offline registration permits a forced DIRECT on a partial state.
                # The live ANSWER separately requires sufficient evidence.
                "legal_actions": [plan.semantic_kind for plan in plans],
                "observation_overhead": {"clock": "client_monotonic", "start": started,
                    "pre_arrival_startup_ms": startup_ms,
                    "quote_start": quote_started, "quote_end": quote_ended,
                    "paid_window_end": self.paid.last_observation["end"],
                    "handling": "pricing startup precedes arrivals and is in initialization/startup cost; decision capture through paid_window_end is in paid_cost; serialization is in subsequent paid windows and full request cost"}})
