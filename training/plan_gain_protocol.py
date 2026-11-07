"""Predeclared counterfactual plans executed through the actual Controller.

The fixed experimental choice produces reader supervision.  It does not invoke
PlanPolicy, predict gain, or claim calibrated execution costs.
"""
from __future__ import annotations

from dataclasses import replace
import time

from adapters.history_state import TaskInterpretation, TaskModelResult, TaskPrediction, validate_task_refs
from controller.action_proposals import (GapRetrievalRequest as GapRetrievalRequest,
    build_observed_action_plans as build_observed_action_plans, gap_query as gap_query,
    ALTERNATE_ANSWERS, BOUNDED_CONTINUATION, bind_read_parents, continuation_gap, continuation_read,
    gap_key, gap_retrieval_plan, observed_step_outputs)
from controller.context_schema import EvidencePool, ReadingOutcome, ReadTask
from controller.context_budget import bind_answer_window, direct_answer_reason
from controller.plan_policy import CompleteCost, complete_plan
from controller.reading import build_read_plan
from controller.rolling_policy import (RollingResult, ZERO_AUX, _reduction_tasks,
                                       compile_step, namespace_plan)
from controller.schema import Plan
from controller.serialization import decode, to_jsonable


class EmptyPlannedPremise(ValueError):
    pass


class PlannedReductionConflict(ValueError):
    pass


class ObservedActionFailure(ValueError):
    pass



def observed_state_binding(state, snapshot, *, paid_cost, native_condition,
                           reader_checkpoint, reader_profile_id, paid_observation=None):
    from training.train_plan_gain import actual_history, working_state_key
    binding = {"continuation_protocol": BOUNDED_CONTINUATION, "state_key": working_state_key(state, snapshot), "actual_history": actual_history(snapshot),
        "reader_checkpoint": reader_checkpoint, "reader_profile_id": reader_profile_id,
        "paid_cost": to_jsonable(paid_cost), "native_condition": to_jsonable(native_condition)}
    if paid_observation is not None:
        binding["paid_observation"] = to_jsonable(paid_observation)
    return binding




class AuditedTaskInterpreter:
    """Use an audited TaskState unchanged; all subsequent retrieval is real."""
    def __init__(self, history_index):
        self.history_index = history_index
        self.model = self
        self.usage = {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}
        self.row = None
        self.manifest = {"kind": "frozen_audited_actual_history_task_state", "inference_calls": 0}

    def bind(self, row, state):
        self.row, self.state = row, state

    def interpret(self, turn, memory):
        actual = [{"turn_id": row.turn_id, "user_text": row.user_text, "assistant_text": row.assistant_text}
                  for row in memory.committed_history]
        expected = [{name: row[name] for name in ("turn_id", "user_text", "assistant_text")}
                    for row in self.row["actual_history"]]
        if (actual != expected or turn.query != self.row["question"] or turn.turn != self.row["turn"]
                or self.state.raw_question != turn.query or self.state.current_turn_id != turn.turn):
            raise ValueError("audited task binding differs from the actual frozen request/history")
        # validate_task_refs reads the existing task view; no second domain is created.
        class Refs:
            history_spans = self.state.working_history
            conditions = self.state.conditions
            needs = self.state.needs
            version_intent = self.state.version_intent
        validate_task_refs(Refs, turn, memory)
        return self.state

    def interpret_result(self, turn, memory):
        return TaskInterpretation(self.interpret(turn, memory), TaskModelResult(TaskPrediction("NEW")),
                                  {"task_model_ms": 0.0})


def read_geometry(base, budget, template, snapshot, *, separate_needs=None):
    plan = build_read_plan(base.task_state, base.selected_set, base.coverage_scope, budget,
        template=template, snapshot=snapshot, output_tokens_per_fact=64,
        reading_mode=base.reading_mode,
        separate_needs=base.separate_needs if separate_needs is None else separate_needs,
        numeric_identity=base.numeric_identity, read_layout=base.read_layout, read_packing=base.read_packing,
        answer_system_prompt=base.answer_system_prompt)
    return replace(plan, answer_plan=base.answer_plan)


def multihop_geometry(base, budget, template, snapshot):
    """Use actual current Need dependency edges to declare a complete hop DAG."""
    current = {need.need_id: need for need in base.task_state.needs}
    edges = {(need.need_id, parent) for need in current.values()
             for parent in need.predecessor_need_ids if parent in current}
    if not edges:
        return None
    isolated = read_geometry(base, budget, template, snapshot, separate_needs=True)
    reads = {need: tuple(task for task in isolated.steps if need in task.need_ids) for need in current}
    reductions = {}
    for need in current.values():
        if isolated.reading_mode == "coverage_only" or need.operation not in {
                "enumerate", "count", "sum", "group", "compare", "infer"}:
            continue
        reductions[need.need_id] = ReadTask("node-reduce:"+need.need_id, "REDUCE", (need.need_id,),
            predecessor_step_ids=tuple(task.step_id for task in reads[need.need_id]),
            input_budget=budget.window_tokens-budget.read_output_tokens,
            output_budget=budget.read_output_tokens,
            operation="union" if need.operation == "enumerate" else need.operation)
    pending, done, tasks = list(current), set(), []
    while pending:
        available = [key for key in pending if not {parent for child, parent in edges if child == key}-done]
        if not available:
            raise ValueError("audited current Need dependency graph is cyclic")
        for key in available:
            parents = tuple(identity for child, parent in sorted(edges) if child == key
                for identity in ((reductions[parent].step_id,) if parent in reductions else
                                 tuple(task.step_id for task in reads[parent])))
            tasks.extend(replace(task, predecessor_step_ids=tuple(dict.fromkeys(
                (*task.predecessor_step_ids, *parents)))) for task in reads[key])
            if key in reductions:
                tasks.append(reductions[key])
            done.add(key)
            pending.remove(key)
    return replace(isolated, steps=tuple(tasks))


def complete_predeclared(plan, *, semantic_kind, reader_checkpoint, initial_state=None):
    plan = namespace_plan(plan)
    empty = (ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX)) if initial_state is None
             else replace(initial_state, plan=plan))
    reduced = {need for task in plan.steps if task.kind == "REDUCE" for need in task.need_ids}
    tail = tuple(task for task in _reduction_tasks(plan, empty) if not reduced.intersection(task.need_ids))
    plan = replace(plan, steps=(*plan.steps, *tail))
    return complete_plan(plan, (), semantic_kind=semantic_kind, reader_checkpoint=reader_checkpoint)


def build_counterfactual_plans(base, snapshot, template, *, reader_checkpoint, read_outputs,
                               scope_material=None, answer_system_prompt=None, initial_state=None):
    """Each output budget repacks the original material before registration."""
    if not base.selected_set.selected_units:
        return (), ({"family": "complementary_evidence", "reason": "no_selected_original_material"},)
    base = bind_answer_window(base, template, snapshot, answer_system_prompt)
    completed = set() if initial_state is None else {result.step_id for result in initial_state.results}
    direct = replace(base, plan_id=base.plan_id+":gain-direct",
                     steps=tuple(task for task in base.steps if task.step_id in completed),
                     coverage_scope="selected_sources", coverage_units=())
    proposals = [complete_plan(direct, (), semantic_kind="direct_answer", reader_checkpoint=reader_checkpoint)]
    unavailable = []
    for output in read_outputs:
        budget = replace(base.budget, read_output_tokens=output)
        for family in ("complementary_evidence", "multihop", "scope_scan"):
            if family == "scope_scan" and scope_material is None:
                unavailable.append({"family": family, "read_output": output, "reason": "no_explicit_applicable_source_scope"})
                continue
            try:
                if family == "multihop":
                    plan = multihop_geometry(base, budget, template, snapshot)
                    if plan is None:
                        unavailable.append({"family": family, "read_output": output,
                            "reason": "no_current_need_dependency_edge"})
                        continue
                else:
                    source = (replace(base, selected_set=scope_material, coverage_scope="all_supplied_documents")
                              if family == "scope_scan" else replace(base, coverage_scope="selected_sources"))
                    plan = read_geometry(source, budget, template, snapshot)
                if initial_state is not None:
                    from controller.fact_reuse import apply_reuse_to_plan
                    plan = apply_reuse_to_plan(plan, initial_state, template, snapshot).plan
            except ValueError as error:
                # These are pre-generation geometry failures; retain the exact
                # message and original material rather than making another plan.
                if not str(error).startswith(("full coverage exceeds", "need-isolated full coverage exceeds",
                        "one READ fact and task exceed", "READ dependency closure needs finer")):
                    raise
                unavailable.append({"family": family, "read_output": output,
                    "reason": "geometry_infeasible", "error": str(error)})
                continue
            plan = replace(plan, plan_id=f"{base.plan_id}:gain:{family}:{output}", layouts=base.layouts)
            proposals.append(complete_predeclared(plan, semantic_kind=family, reader_checkpoint=reader_checkpoint,
                initial_state=initial_state))
    return tuple(proposals), tuple(unavailable)


class FixedCounterfactualRuntime:
    """Execute one pre-registered semantic plan; bind only generated fact IDs."""
    def __init__(self, planned, *, action="REBUILD", layout="existing", initial_state=None,
                 paid_cost: CompleteCost | None = None, gap_request=None, gap_provider=None,
                 initial_snapshot=None, action_domain=None):
        self.planned, self.action, self.layout = planned, action, layout
        self.native = self
        self.decisions = []
        self.initial_state, self.paid_cost = initial_state, paid_cost
        self.initial_snapshot = initial_snapshot
        self.action_domain = action_domain
        self.gap_request, self.gap_provider = gap_request, gap_provider
        self.gap_result = None
        self.pre_action_quote = None
        self.pre_action_input = None
        self.new_auxiliary_usage = dict(ZERO_AUX)
        self.new_step_ids = ()
        self.new_action_counts = {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 0}

    def bind_resident_models(self, manifest):
        self.resident_model_manifest = manifest

    async def start(self):
        return

    async def aclose(self):
        return

    async def run(self, attempt_id, snapshot, context_plans, executor, engine, template,
                  cost_model, compile_answers, *, arrived, auxiliary_usage, events, open_need_provider=None,
                  initial_outcomes=()):
        if context_plans != (self.planned.context_plan,):
            raise ValueError("experimental branch differs from its pre-registered complete plan")
        if self.initial_snapshot is not None:
            snapshot = replace(snapshot, archive=self.initial_snapshot.archive,
                               reference_map=self.initial_snapshot.reference_map)
        plan = self.planned.context_plan
        supplied = next((seed for seed in initial_outcomes if seed.plan.plan_id == plan.plan_id), None)
        initial = self.initial_state if self.initial_state is not None else supplied
        state = (ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX)) if initial is None
                 else replace(initial, plan=plan))
        expanded = {gap_key(need) for record in state.steps if record.task.kind == "GAP"
                    for need in decode(GapRetrievalRequest, record.metrics["gap_request"]).open_needs}
        completed = {result.step_id for result in state.results}
        pending = [step.task for step in self.planned.steps if step.task.kind != "ANSWER"
                   and step.task.step_id not in completed]
        observed = self.paid_cost is not None
        if observed:
            from controller.execution import auxiliary_tokens, ReadingBudgetExceeded
            measured = sum(len(record.input_ids)+len(record.output_ids) for record in state.steps)
            prior_aux = auxiliary_tokens(state.auxiliary_usage)
            paid_tokens = self.paid_cost.input_tokens+self.paid_cost.output_tokens
            initial_auxiliary = paid_tokens-measured-prior_aux
            if initial_auxiliary < 0:
                raise ValueError("observed paid tokens omit already completed reading/auxiliary work")
            auxiliary_usage = {"input_tokens": initial_auxiliary, "output_tokens": 0,
                               "compute_ms": 0, "models": []}
            plan = replace(plan, online_tokens_spent=paid_tokens)
            state = replace(state, plan=plan)
            if plan.budget.max_complete_ms is not None and self.paid_cost.latency_ms >= plan.budget.max_complete_ms:
                raise ReadingBudgetExceeded("observed paid latency exhausts the complete request allocation")
            expected = 1 if self.planned.semantic_kind in {"read_once", "reduce_once"} else 0
            if len(pending) != expected:
                raise ValueError("observed branch must contain exactly its registered new action")
        async def retrieve(request):
            nonlocal plan, state
            from controller.execution import ReadingBudgetExceeded, require_online_tokens
            if (observed and plan.budget.max_complete_ms is not None and
                    self.paid_cost.latency_ms+(time.monotonic()-arrived)*1000 > plan.budget.max_complete_ms):
                raise ReadingBudgetExceeded("paid state plus bounded GAP exceeds complete latency")
            if len(state.steps)+self.new_action_counts["GAP_RETRIEVAL"]+2 > plan.budget.max_steps:
                raise ReadingBudgetExceeded("bounded GAP leaves no terminal ANSWER allocation")
            require_online_tokens(plan.budget.max_total_tokens, plan.online_tokens_spent,
                plan.answer_allocation.fixed_input_tokens+plan.budget.output_tokens, "bounded GAP")
            self.gap_request = request
            from controller.evidence_planner import _passages, close_dependencies
            from controller.execution import auxiliary_tokens, merge_auxiliary_usage
            if self.pre_action_quote is None:
                self.pre_action_quote = {"batch": await engine.quote_batch((), template.fingerprint), "candidate_id": None}
            tick = time.monotonic()
            self.new_action_counts["GAP_RETRIEVAL"] = 1
            read_base = gap_retrieval_plan(state, request)
            pool = await self.gap_provider(read_base, request, state.claims)
            elapsed = (time.monotonic()-tick)*1000
            if not isinstance(pool, EvidencePool):
                raise TypeError("GAP action requires measured EvidencePool material")
            self.new_auxiliary_usage = pool.auxiliary_usage
            self.gap_result = {"request": self.gap_request, "pool": pool, "elapsed_ms": elapsed}
            if auxiliary_tokens(pool.auxiliary_usage) is None:
                raise ObservedActionFailure("actual GAP auxiliary token cost is unmeasured")
            prior_passages = _passages(read_base.selected_set.selected_units)
            if (len(pool.units) > self.gap_request.limits["max_units"]
                    or len(_passages(pool.units)-prior_passages) > self.gap_request.limits["max_passages"]):
                raise ObservedActionFailure("actual GAP material exceeds its registered retrieval limits")
            units = {unit.unit_id: unit for unit in (*plan.selected_set.selected_units, *pool.units)}
            closure = close_dependencies(tuple(units), units)
            if plan.budget.max_passages is not None and len(_passages(closure)) > plan.budget.max_passages:
                raise ObservedActionFailure("actual GAP material exceeds the cumulative passage allocation")
            blocks = tuple({block.key: block for block in (*plan.selected_set.source_blocks,
                                                           *pool.source_blocks)}.values())
            selected = replace(pool.selected_set or plan.selected_set, selected_units=closure, source_blocks=blocks)
            plan = bind_answer_window(replace(plan, selected_set=selected, answer_allocation=None), template,
                                      snapshot, preserve_step_ids=tuple(record.step_id for record in state.steps))
            usage = merge_auxiliary_usage(state.auxiliary_usage, pool.auxiliary_usage)
            state = replace(state, plan=plan, auxiliary_usage=usage)
            plan = replace(plan, online_tokens_spent=plan.online_tokens_spent+auxiliary_tokens(pool.auxiliary_usage))
            state = replace(state, plan=plan)
            require_online_tokens(plan.budget.max_total_tokens, plan.online_tokens_spent,
                plan.answer_allocation.fixed_input_tokens+plan.budget.output_tokens, "bounded GAP and ANSWER")
            events.emit("plan_gain_experimental_gap", attempt_id=attempt_id, **self.gap_result)
            return pool
        if self.planned.semantic_kind == "gap_retrieval":
            pool = await retrieve(self.gap_request)
        records = []
        self.decisions = records
        continued = False
        while pending or (observed and self.planned.semantic_kind != "direct_answer" and not continued):
            if not pending:
                continued = True
                ready_terminal = continuation_read(state, self.initial_state.plan)
                paid_steps = len(state.steps)+self.new_action_counts["GAP_RETRIEVAL"]
                remaining_steps = plan.budget.max_steps-paid_steps
                if ready_terminal is not None:
                    if remaining_steps < 2:  # terminal READ plus committed ANSWER
                        break
                    task = replace(ready_terminal, step_id="bounded-terminal:"+ready_terminal.step_id,
                        predecessor_step_ids=())
                    plan = replace(plan, steps=(*plan.steps, task))
                    state = replace(state, plan=plan)
                else:
                    gaps = tuple(need for need in continuation_gap(state, used_gap=False)
                                 if gap_key(need) not in expanded)
                    used_gap = self.new_action_counts["GAP_RETRIEVAL"]
                    if gaps and not used_gap and self.gap_provider is not None and self.gap_request is not None:
                        if remaining_steps < 3:  # GAP, terminal READ, ANSWER
                            break
                        request = replace(self.gap_request, open_needs=gaps, query=gap_query(plan.task_state, gaps))
                        pool = await retrieve(request)
                    elif used_gap and gaps:
                        if remaining_steps < 2:  # registered GAP already paid
                            break
                        pool = self.gap_result["pool"]
                    else:
                        break
                    covered = {unit for result in state.results for unit in result.covered_units}
                    novel = tuple(unit for unit in pool.units if unit.unit_id not in covered)
                    if not novel:
                        break
                    from controller.evidence_planner import close_dependencies
                    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
                    terminal_units = close_dependencies(tuple(unit.unit_id for unit in novel), units)
                    parents = tuple(dict.fromkeys(parent for need in gaps
                                                  for parent in need.parent_claim_ids))
                    by_id = {claim.claim_id: claim for claim in state.claims}
                    gap_plan = build_read_plan(plan.task_state, replace(plan.selected_set, selected_units=terminal_units),
                        "selected_sources", plan.budget, template=template, snapshot=snapshot,
                        input_claims=tuple(by_id[parent] for parent in parents), reading_mode=plan.reading_mode,
                        separate_needs=plan.separate_needs, numeric_identity=plan.numeric_identity,
                        read_layout=plan.read_layout, read_packing=plan.read_packing,
                        answer_system_prompt=plan.answer_system_prompt)
                    reads = tuple(task for task in gap_plan.steps if task.kind == "READ")
                    if not reads:
                        break
                    task = replace(bind_read_parents(reads[0], state),
                        step_id="bounded-terminal:"+reads[0].step_id, predecessor_step_ids=())
                    selected = replace(plan.selected_set, selected_units=tuple({
                        **{unit.unit_id: unit for unit in plan.selected_set.selected_units},
                        **{unit.unit_id: unit for unit in gap_plan.selected_set.selected_units}}.values()))
                    plan = replace(plan, selected_set=selected, steps=(*plan.steps, task))
                    state = replace(state, plan=plan)
                pending.append(task)
            completed = {result.step_id for result in state.results}
            ready = [task for task in pending if set(task.predecessor_step_ids) <= completed]
            if not ready:
                raise ValueError("pre-registered experimental DAG has no executable next step")
            from controller.execution import ReadingBudgetExceeded
            if (observed and plan.budget.max_complete_ms is not None and
                    self.paid_cost.latency_ms+(time.monotonic()-arrived)*1000 > plan.budget.max_complete_ms):
                raise ReadingBudgetExceeded("paid state plus bounded continuation exceeds complete latency")
            task = ready[0]
            pending.remove(task)
            if observed:
                # Single-action factories already bind the actual shared/selected
                # basis. READ aliases and extract-only filters would discard valid
                # reassessment inputs or selector-derived CPU inputs.
                by_id = {claim.claim_id: claim for claim in state.claims}
                parents = tuple(by_id[identity] for identity in task.input_claim_ids)
            else:
                parents = tuple(claim for result in state.results if result.step_id in task.predecessor_step_ids
                                for claim in result.claims)
                if task.kind == "REDUCE":
                    parents = tuple(claim for claim in parents if any(claim.serves(need) for need in task.need_ids)
                                    and claim.kind == "extract")
                if task.predecessor_step_ids and not parents and any(
                        previous.task.step_id in task.predecessor_step_ids and previous.task.kind != "GAP"
                        for previous in self.planned.steps):
                    raise EmptyPlannedPremise("planned predecessor produced no factual input: "+task.step_id)
                task = replace(task, input_claim_ids=tuple(claim.claim_id for claim in parents)
                               if task.predecessor_step_ids else task.input_claim_ids)
            plan = replace(plan, steps=tuple(task if old.step_id == task.step_id else old for old in plan.steps))
            cpu = plan.reading_mode == "typed_reduce" and task.kind == "REDUCE" and task.operation in {"count", "sum", "group", "union"}
            cache_scope = None if cpu else await engine.info()
            outputs = (task.output_budget,)
            if self.action_domain is not None and not cpu and not self.new_step_ids and not continued:
                outputs = observed_step_outputs(plan, task, template,
                    output_budgets=self.action_domain["output_budgets"],
                    output_tokens_per_fact=self.action_domain["output_tokens_per_fact"], finite=True)[1]
            variants = () if cpu else compile_step(plan, task, state, snapshot, template,
                outputs=outputs, request_compiler=executor.request_compiler, cache_scope=cache_scope)
            if self.action_domain is not None and task.kind == "REDUCE":
                variants = variants[:1]
            variants = tuple(candidate for candidate in variants if candidate.feasible)
            if not cpu and not variants:
                from controller.execution import ReadingBudgetExceeded
                raise ReadingBudgetExceeded("actual generated facts exceed the pre-registered execution window")
            quote_started = time.monotonic()
            quote = None if cpu else await engine.quote_batch(variants, template.fingerprint)
            quote_ms = None if cpu else (time.monotonic()-quote_started)*1000
            candidate = None if cpu else executor.request_compiler.choose_variant(variants, quote)
            if candidate is not None:
                task = replace(task, output_budget=candidate.output_budget)
            if not self.new_step_ids and self.pre_action_quote is None:
                self.pre_action_quote = {
                    "batch": await engine.quote_batch((), template.fingerprint) if cpu else quote,
                    "candidate_id": None if cpu else candidate.candidate_id}
                self.pre_action_input = candidate
            record = {"step_id": task.step_id, "fact_binding": task.input_claim_ids,
                "actual_quote": to_jsonable(quote), "compiled_input": candidate,
                "selection": "predeclared_semantic_branch_actual_compiler_choice",
                "gain_prediction": None, "calibration_claim": None}
            records.append(record)
            events.emit("plan_gain_experimental_step", attempt_id=attempt_id, **record)
            try:
                state = await executor.execute_one(attempt_id, snapshot, plan, task, state,
                    initial_auxiliary_usage=auxiliary_usage, candidate=candidate, quote=quote, quote_ms=quote_ms)
            except ValueError as error:
                if not str(error).startswith(("conflicting quantities for ", "sum requires compatible units:")):
                    raise
                raise PlannedReductionConflict(str(error)) from error
            plan = state.plan
            self.new_step_ids = (*self.new_step_ids, task.step_id)
            self.new_action_counts[task.kind] += 1
        if self.action_domain is None and plan.coverage_scope == "all_supplied_documents" and not set(plan.coverage_units) <= {
                unit for result in state.results for unit in result.covered_units}:
            raise ValueError("predeclared scope scan did not complete its original scope")
        phase_started = time.monotonic()
        if (observed and plan.budget.max_complete_ms is not None
                and self.paid_cost.latency_ms+(phase_started-arrived)*1000 > plan.budget.max_complete_ms):
            raise ReadingBudgetExceeded("observed paid state plus action exhausts the complete request allocation")
        representation = self.planned.answer_representation
        options = (dict(layouts=(self.layout,), actions=(self.action,), representations=(representation,))
                   if self.action_domain is not None else {})
        candidates = compile_answers(snapshot, (plan,), state, **options)
        matching = [candidate for candidate in candidates if candidate.feasible
                    and candidate.action == self.action
                    and candidate.layout_id in {self.layout, self.layout+":contextpilot"}]
        if self.action_domain is not None:
            matching = [candidate for candidate in matching if direct_answer_reason(state, candidate,
                require_complete_evidence=representation in ALTERNATE_ANSWERS) is None]
        if not matching:
            raise ReadingBudgetExceeded("predeclared terminal action/layout has no feasible actual input")
        quotes = await engine.quote_batch(tuple(matching), template.fingerprint)
        candidate = executor.request_compiler.choose_variant(matching, quotes)
        if self.planned.semantic_kind == "direct_answer":
            self.pre_action_quote = {"batch": quotes, "candidate_id": candidate.candidate_id}
            self.pre_action_input = candidate
        events.emit("plan_gain_experimental_answer", attempt_id=attempt_id,
            planned_plan_id=self.planned.plan_id, actual_input_hash=candidate.input_hash,
            actual_quote=quotes, compiled_input=candidate,
            selection="predeclared_semantic_branch_actual_compiler_choice", gain_prediction=None)
        self.decisions = records
        self.new_action_counts["ANSWER"] = 1
        return RollingResult(replace(state, answer_representation=representation), Plan(candidate, quote_seq=quotes.observed_seq,
            reason="actual_counterfactual_fixed_plan"), quotes, tuple(records), phase_started)
