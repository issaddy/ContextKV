"""Predeclared counterfactual plans executed through the actual Controller.

The fixed experimental choice produces reader supervision.  It does not invoke
PlanPolicy, predict gain, or claim calibrated execution costs.
"""
from __future__ import annotations

from dataclasses import replace
import time

from adapters.history_state import validate_task_refs
from controller.context_schema import ReadingOutcome, ReadTask
from controller.plan_policy import complete_plan
from controller.reading import build_read_plan
from controller.rolling_policy import (RollingResult, ZERO_AUX, _reduction_tasks,
                                       compile_step, namespace_plan)
from controller.schema import Plan
from controller.serialization import to_jsonable


class EmptyPlannedPremise(ValueError):
    pass


class PlannedReductionConflict(ValueError):
    pass


class AuditedTaskInterpreter:
    """Use an audited TaskState unchanged; all subsequent retrieval is real."""
    def __init__(self, history_index):
        self.history_index = history_index
        self.model = self
        self.usage = {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}
        self.last_metrics = {"task_model_ms": 0.0}
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


def read_geometry(base, budget, template, snapshot, *, separate_needs=None):
    plan = build_read_plan(base.task_state, base.selected_set, base.coverage_scope, budget,
        template=template, snapshot=snapshot, output_tokens_per_fact=64,
        reading_mode=base.reading_mode,
        separate_needs=base.separate_needs if separate_needs is None else separate_needs,
        numeric_identity=base.numeric_identity, read_layout=base.read_layout, read_packing=base.read_packing)
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


def complete_predeclared(plan, *, semantic_kind, reader_checkpoint):
    plan = namespace_plan(plan)
    empty = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
    reduced = {need for task in plan.steps if task.kind == "REDUCE" for need in task.need_ids}
    tail = tuple(task for task in _reduction_tasks(plan, empty) if not reduced.intersection(task.need_ids))
    plan = replace(plan, steps=(*plan.steps, *tail))
    return complete_plan(plan, (), semantic_kind=semantic_kind, reader_checkpoint=reader_checkpoint)


def build_counterfactual_plans(base, snapshot, template, *, reader_checkpoint, read_outputs,
                               scope_material=None):
    """Each output budget repacks the original material before registration."""
    if not base.selected_set.selected_units:
        return (), ({"family": "complementary_evidence", "reason": "no_selected_original_material"},)
    direct = replace(base, plan_id=base.plan_id+":gain-direct", steps=(),
                     coverage_scope="selected_sources", coverage_units=())
    proposals = [complete_predeclared(direct, semantic_kind="direct_answer", reader_checkpoint=reader_checkpoint)]
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
            proposals.append(complete_predeclared(plan, semantic_kind=family, reader_checkpoint=reader_checkpoint))
    return tuple(proposals), tuple(unavailable)


class FixedCounterfactualRuntime:
    """Execute one pre-registered semantic plan; bind only generated fact IDs."""
    def __init__(self, planned, *, action="REBUILD", layout="existing"):
        self.planned, self.action, self.layout = planned, action, layout
        self.native = self
        self.decisions = []

    def bind_resident_models(self, manifest):
        self.resident_model_manifest = manifest

    async def start(self):
        return

    async def aclose(self):
        return

    async def run(self, attempt_id, snapshot, context_plans, executor, engine, template,
                  cost_model, compile_answers, *, arrived, auxiliary_usage, events, open_need_provider=None):
        if context_plans != (self.planned.context_plan,):
            raise ValueError("experimental branch differs from its pre-registered complete plan")
        plan = self.planned.context_plan
        state = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
        pending = [step.task for step in self.planned.steps if step.task.kind != "ANSWER"]
        records = []
        while pending:
            completed = {result.step_id for result in state.results}
            ready = [task for task in pending if set(task.predecessor_step_ids) <= completed]
            if not ready:
                raise ValueError("pre-registered experimental DAG has no executable next step")
            task = ready[0]
            pending.remove(task)
            parents = tuple(claim for result in state.results if result.step_id in task.predecessor_step_ids
                            for claim in result.claims)
            if task.kind == "REDUCE":
                parents = tuple(claim for claim in parents if claim.need_id in task.need_ids
                                and claim.kind == "extract")
            if task.predecessor_step_ids and not parents:
                raise EmptyPlannedPremise("planned predecessor produced no factual input: "+task.step_id)
            task = replace(task, input_claim_ids=tuple(claim.claim_id for claim in parents))
            plan = replace(plan, steps=tuple(task if old.step_id == task.step_id else old for old in plan.steps))
            cpu = plan.reading_mode == "typed_reduce" and task.kind == "REDUCE" and task.operation in {"count", "sum", "group", "union"}
            variants = () if cpu else compile_step(plan, task, state, snapshot, template, outputs=(task.output_budget,))
            if variants and not variants[0].feasible:
                from controller.execution import ReadingBudgetExceeded
                raise ReadingBudgetExceeded("actual generated facts exceed the pre-registered execution window")
            quote = None if cpu else await engine.quote_batch(variants, template.fingerprint)
            record = {"step_id": task.step_id, "fact_binding": task.input_claim_ids,
                "actual_quote": to_jsonable(quote), "selection": "predeclared_experimental_branch",
                "gain_prediction": None, "calibration_claim": None}
            records.append(record)
            events.emit("plan_gain_experimental_step", attempt_id=attempt_id, **record)
            try:
                state = await executor.execute_one(attempt_id, snapshot, plan, task, state,
                    initial_auxiliary_usage=auxiliary_usage, candidate=None if cpu else variants[0])
            except ValueError as error:
                if not str(error).startswith(("conflicting quantities for ", "sum requires compatible units:")):
                    raise
                raise PlannedReductionConflict(str(error)) from error
            plan = state.plan
        if plan.coverage_scope == "all_supplied_documents" and not set(plan.coverage_units) <= {
                unit for result in state.results for unit in result.covered_units}:
            raise ValueError("predeclared scope scan did not complete its original scope")
        phase_started = time.monotonic()
        candidates = compile_answers(snapshot, (plan,), state)
        matching = [candidate for candidate in candidates if candidate.feasible
                    and candidate.action == self.action and candidate.layout_id == self.layout]
        if len(matching) != 1:
            raise ValueError("predeclared terminal action/layout has no unique feasible actual input")
        candidate = matching[0]
        quotes = await engine.quote_batch((candidate,), template.fingerprint)
        events.emit("plan_gain_experimental_answer", attempt_id=attempt_id,
            planned_plan_id=self.planned.plan_id, actual_input_hash=candidate.input_hash,
            actual_quote=quotes, selection="predeclared_experimental_branch", gain_prediction=None)
        self.decisions = records
        return RollingResult(state, Plan(candidate, quote_seq=quotes.observed_seq,
            reason="actual_counterfactual_fixed_plan"), quotes, tuple(records), phase_started)
