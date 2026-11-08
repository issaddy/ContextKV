"""Execute durable reading steps inside one outer user attempt."""
from __future__ import annotations

from dataclasses import replace
import json
import time

from .action_proposals import claim_predecessor_steps
from .admission import quote_observation, quote_realization
from .claim_graph import ClaimGraph, claim_frontier
from .context_budget import bind_answer_window, completion_reserve
from .context_schema import ReadResult, ReadingOutcome, ReductionState, StepRecord
from .engine_client import EngineError
from .reading import (advance_read_plan, build_read_plan, output_schema, parse_read_result,
                      step_generation_kind, step_messages, unresolved_needs)
from .request_compiler import read_task_hash
from .reduction import reduce_claims
from .schema import ContextEpoch
from .serialization import to_jsonable
from .session_store import ReadingProgress
from .templates import candidate_request, step_request


def reduction_basis_ids(task, inputs, results):
    if task.operation in {"count", "sum"}:
        for result in reversed(results):
            selection = result.reduction
            if (selection is not None and selection.operation == "numeric_select"
                    and {selection.step_id, result.step_id}.intersection(task.predecessor_step_ids)):
                return selection.basis_claim_ids
    return tuple(claim.claim_id for claim in inputs)


def aggregate_numeric_view(task, inputs, plan, results):
    claim = reduce_claims(task.operation, inputs,
        claim_id=f"{task.step_id}:claim:0", need_id=task.need_ids[0])
    if not inputs and task.operation == "group":
        claim = replace(claim, source_spans=tuple(dict.fromkeys(span
            for unit in plan.selected_set.selected_units for span in unit.source_spans)))
    elif not inputs and task.operation in {"count", "sum"}:
        selections = [result.reduction for result in results if result.reduction is not None
                      and result.reduction.operation == "numeric_select"
                      and {result.reduction.step_id, result.step_id}.intersection(task.predecessor_step_ids)]
        if selections:
            # The empty view is proved by the completed selection over the read scope.
            claim = replace(claim, parent_claim_ids=selections[-1].basis_claim_ids,
                source_spans=tuple(dict.fromkeys(span for unit in plan.selected_set.selected_units
                                                 for span in unit.source_spans)))
    return claim


def completed_reduction_record(record, task, basis, results=(), *, frontier=None, mode=None):
    """Bind the input basis and effective frontier before publishing COMPLETE."""
    selected = task.operation == "numeric_select"
    if frontier is None:
        frontier = record.result.claims if selected else claim_frontier(basis, record.result.claims)
    return replace(record, result=replace(record.result, reduction=ReductionState(
        task.step_id, task.need_ids, task.operation, reduction_basis_ids(task, basis, results),
        tuple(claim.claim_id for claim in frontier), mode or ("selected" if selected else "merged"))))


class ReadingBudgetExceeded(ValueError):
    pass


class OnlineTokenBudgetUnmeasured(ValueError):
    pass


def reduction_partitions(plan, task, claims, snapshot, template):
    """Compile finite input packs before a caller quotes and executes each one."""
    from .context_budget import split_conclusion_tokens

    claims = tuple(claims)

    def fits(current, inputs):
        if len(template.render(step_messages(plan, current, (), inputs, snapshot)).input_ids) > current.input_budget:
            return False
        if current.operation != "numeric_select":
            return True
        schema = output_schema(current, claims=inputs)["properties"]
        outputs = ({"format_version": schema["format_version"]["const"], "views": {
            alias: {parent: decision.get("const", value) for parent, decision in view["properties"].items()}
            for alias, view in schema["views"]["properties"].items()}} for value in (False, True))
        return max(len(template.tokenizer.encode(json.dumps(output, separators=(",", ":")),
            add_special_tokens=False)) for output in outputs) <= current.output_budget

    if fits(task, claims):
        return ((task, claims),)
    packs, pack = [], []
    for claim in claims:
        child = replace(task, step_id=f"{task.step_id}:part:0:{len(packs)}", reduction_basis_claim_ids=())
        fits_pack = fits(child, (*pack, claim))
        if pack and not fits_pack:
            packs.append(tuple(pack))
            pack = []
            child = replace(task, step_id=f"{task.step_id}:part:0:{len(packs)}", reduction_basis_claim_ids=())
            fits_pack = fits(child, (claim,))
        if not fits_pack:
            raise ReadingBudgetExceeded("one claim exceeds the reduction input/output allocation")
        pack.append(claim)
    if pack:
        packs.append(tuple(pack))
    targets = split_conclusion_tokens(task.retained_conclusion_tokens, len(packs))
    parts = []
    for index, pack in enumerate(packs):
        child = replace(task, step_id=f"{task.step_id}:part:0:{index}",
            input_claim_ids=tuple(claim.claim_id for claim in pack),
            reduction_basis_claim_ids=tuple(claim.claim_id for claim in claims)
                if task.operation != "numeric_select" or index == len(packs)-1 else (),
            retained_conclusion_tokens=targets[index])
        parts.append((child, pack))
    return tuple(parts)


def reduction_schedule(plan, task, basis, results, snapshot, template):
    """Return one finite model call, or the complete logical frontier.

    Every merge consumes at least two whole bundles. N initial packs therefore
    permit at most N-1 merges, including when a caller reobserves after each call.
    """
    from .context_budget import merge_conclusion_tokens

    observed = {result.step_id: result for result in results}
    parts = reduction_partitions(plan, task, basis, snapshot, template)
    bundles = []
    for child, pack in parts:
        if child.step_id not in observed:
            return (child, pack), (), None
        result = observed[child.step_id]
        frontier = (result.claims if task.operation == "numeric_select"
                    else claim_frontier(pack, result.claims))
        bundles.append((frontier, (child.step_id,), child.retained_conclusion_tokens))
    if task.operation == "numeric_select":
        return None, tuple(claim for bundle in bundles for claim in bundle[0]), "selected"
    level = 1
    while len(bundles) > 1:
        groups, group = [], []
        for bundle in bundles:
            child = replace(task, step_id=f"{task.step_id}:merge:{level}:{len(groups)}",
                            reduction_basis_claim_ids=())
            trial = tuple(claim for item in (*group, bundle) for claim in item[0])
            size = len(template.render(step_messages(plan, child, (), trial, snapshot)).input_ids)
            if group and size > task.input_budget:
                groups.append(tuple(group))
                group = []
            group.append(bundle)
        if group:
            groups.append(tuple(group))
        if len(groups) == len(bundles):
            break
        merged = []
        for index, group in enumerate(groups):
            if len(group) == 1:
                merged.append(group[0])
                continue
            inputs = tuple(claim for item in group for claim in item[0])
            child = replace(task,
                step_id=task.step_id if len(groups) == 1 else f"{task.step_id}:merge:{level}:{index}",
                input_claim_ids=tuple(claim.claim_id for claim in inputs),
                predecessor_step_ids=tuple(identity for item in group for identity in item[1]),
                reduction_basis_claim_ids=tuple(claim.claim_id for claim in basis),
                retained_conclusion_tokens=merge_conclusion_tokens(item[2] for item in group))
            if child.step_id not in observed:
                return (child, inputs), (), None
            result = observed[child.step_id]
            merged.append((claim_frontier(inputs, result.claims), (child.step_id,),
                           child.retained_conclusion_tokens))
        bundles = merged
        level += 1
    return None, tuple(claim for bundle in bundles for claim in bundle[0]), (
        "merged" if len(bundles) == 1 else "forest")


def logical_reduction_task(plan, task, basis, results):
    """Recover the existing logical reduction from its finite call identity."""
    identity = task.step_id.split(":part:0:", 1)[0].split(":merge:", 1)[0]
    shares = task.retained_conclusion_tokens
    if task.operation == "numeric_select":
        shares = ()
    elif plan.answer_allocation is not None:
        shares = tuple((need, tokens) for need, tokens in plan.answer_allocation.need_tokens
                       if need in task.need_ids)
    parents = tuple(claim.claim_id for claim in basis)
    return replace(task, step_id=identity, input_claim_ids=parents,
        predecessor_step_ids=claim_predecessor_steps(results, parents),
        retained_conclusion_tokens=shares)


def auxiliary_tokens(usage):
    if usage is None or usage["input_tokens"] is None or usage["output_tokens"] is None:
        return None
    return usage["input_tokens"]+usage["output_tokens"]


def require_online_tokens(limit, spent, reserve, phase):
    if limit is None:
        return
    if spent is None:
        raise OnlineTokenBudgetUnmeasured(f"{phase}: online token budget requires measured auxiliary and reader usage")
    if spent+reserve > limit:
        raise ReadingBudgetExceeded(f"{phase}: online token allocation {limit} cannot admit {reserve} after {spent} spent")


def require_complete_time(limit, arrived, phase):
    if limit is not None and arrived is not None and (time.monotonic()-arrived)*1000 >= limit:
        raise ReadingBudgetExceeded(f"{phase}: complete-time allocation {limit} ms exhausted")


def require_completion(plan, results, spent, steps_paid, task, candidate=None, *, read_all=False, arrived=None):
    phase = task.kind if task is not None else "GAP"
    reserve = completion_reserve(plan, results, task, candidate, read_all=read_all)
    if steps_paid+reserve.steps > plan.budget.max_steps:
        raise ReadingBudgetExceeded(f"{phase}: remaining required steps exceed the complete request allocation")
    require_online_tokens(plan.budget.max_total_tokens, spent, reserve.input_tokens+reserve.output_tokens, phase)
    require_complete_time(plan.budget.max_complete_ms, arrived, phase)
    return reserve


def step_token_spend(stage, record):
    if stage in {"SUBMITTED", "FAILED"}:
        return None
    if stage in {"RESULT_READY", "COMPLETE"}:
        return len(record.input_ids)+len(record.output_ids)
    return 0


def merge_auxiliary_usage(*records):
    """Sum observed calls; a missing component leaves that total unmeasured."""
    fields = ("input_tokens", "output_tokens", "compute_ms")
    result = {name: (sum(record[name] for record in records)
                    if all(record is not None and record[name] is not None for record in records)
                    else None) for name in fields}
    result["models"] = [model for record in records if record is not None
                        for model in record.get("models", ())]
    result["components"] = list(records)
    return result


def seed_reading_outcome(store, attempt_id, plan, initial_outcome=None):
    """Restore unpaid reused facts; this attempt's physical ledger is overlaid separately."""
    stored = store.get_attempt(attempt_id).reading_outcome
    seed = stored if stored is not None else initial_outcome
    if seed is None:
        return ReadingOutcome(plan)
    paid = store.execution_steps(attempt_id)
    paid_ids = {record.step_id for record in paid}
    paid_claims = {claim.claim_id for record in paid if record.result is not None for claim in record.result.claims}
    results = tuple(result for result in seed.results if result.step_id not in paid_ids)
    claims = tuple(claim for claim in seed.claims if claim.claim_id not in paid_claims)
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    coverage = tuple(dict.fromkeys((plan.coverage_scope, span.source_key.document_id, identity, result.step_id)
        for result in results for identity in result.covered_units for span in units[identity].source_spans))
    graph = ClaimGraph(claims)
    answer = graph.answer_plan(plan.task_state, plan.selected_set.disposition, results)
    prior = {point.need_id: point for point in seed.plan.answer_plan.points}
    answer = replace(answer, points=tuple(point if point.claim_ids or point.need_id not in prior else
        replace(point, source_spans=prior[point.need_id].source_spans) for point in answer.points))
    return ReadingOutcome(replace(plan, answer_plan=answer), results=results, claims=claims,
        coverage=coverage, open_needs=unresolved_needs(results))


class ReadingExecutor:
    def __init__(self, store, engine, template, *, extend_open_needs=None, events=None,
                 request_arrived=None, observe_prefix=None, observe_state=None, request_compiler=None):
        from .request_compiler import RequestCompiler

        self.store, self.engine, self.template = store, engine, template
        self.request_compiler = request_compiler or RequestCompiler(template)
        self.extend_open_needs = extend_open_needs
        self.events = events
        self.request_arrived = request_arrived
        self.observe_prefix = observe_prefix
        self.observe_state = observe_state

    async def execute_one(self, attempt_id, snapshot, plan, task, state, *,
                          initial_auxiliary_usage=None, candidate=None, quote=None, quote_ms=None):
        """Execute the chosen logical step and return its observed working state.

        Selection, gap proposals and terminal ANSWER belong to the rolling
        caller.  This path never executes an unselected successor or splits an
        oversized REDUCE into unpriced model calls.
        """
        if task.kind not in {"READ", "REDUCE"}:
            raise ValueError("execute_one requires a selected READ or REDUCE")
        if task.step_id in {record.step_id for record in state.steps}:
            raise ValueError("the selected logical step has already completed")
        completed = {result.step_id for result in state.results}
        completed.update(result.reduction.step_id for result in state.results if result.reduction is not None)
        if not set(task.predecessor_step_ids) <= completed:
            raise ValueError("the selected logical step has unfinished predecessors")
        self.store.save_context_plan(attempt_id, plan)
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        if candidate is not None and task.kind == "READ":
            if set(candidate.read_unit_order) != set(task.input_unit_ids) or len(candidate.read_unit_order) != len(task.input_unit_ids):
                raise ValueError("selected READ order differs from the declared units")
            task = replace(task, input_unit_ids=candidate.read_unit_order)
        selected_units = tuple(units[identity] for identity in task.input_unit_ids)
        claims = {claim.claim_id: claim for claim in state.claims}
        selected_claims = tuple(claims[identity] for identity in task.input_claim_ids)
        task = replace(task, output_schema=output_schema(task, selected_units, selected_claims,
            reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity))
        cpu = (plan.reading_mode == "typed_reduce" and task.kind == "REDUCE"
               and task.operation in {"union", "count", "sum", "group"})
        persisted = self.store.execution_step(attempt_id, task.step_id)
        reserve = None
        if persisted is None or persisted[0] in {"PLANNED", "PREPARED"}:
            usage = 0 if initial_auxiliary_usage is None else auxiliary_tokens(initial_auxiliary_usage)
            gap = auxiliary_tokens(state.auxiliary_usage)
            spent = (usage+gap+sum(len(row.input_ids)+len(row.output_ids) for row in state.steps)
                     if usage is not None and gap is not None else None)
            reserve = require_completion(plan, state.results, spent, len(state.steps), task, candidate,
                                         arrived=self.request_arrived)
        if cpu:
            tick = time.monotonic()
            native_workload = await self.engine.metrics()
            try:
                claim = aggregate_numeric_view(task, selected_claims, plan, state.results)
            except ValueError as exc:
                self.store.save_execution_step(StepRecord(attempt_id, task.step_id, task,
                    metrics={"cpu_reduce_ms": (time.monotonic()-tick)*1000,
                             "completion_reserve": to_jsonable(reserve), "error": str(exc)}), "FAILED")
                raise
            ended = time.monotonic()
            result = ReadResult(task.step_id, (claim,), elapsed_ms=(ended-tick)*1000)
            record = StepRecord(attempt_id, task.step_id, task, result=result,
                                metrics={"cpu_reduce_ms": result.elapsed_ms,
                                         "completion_reserve": to_jsonable(reserve),
                                         "native_workload": native_workload,
                                         "phase_window": {"start": tick, "end": ended,
                                             "clock": "client_monotonic", "scope": "step_wall"},
                                         "request_arrived_monotonic": self.request_arrived})
        else:
            if candidate is None or candidate.read_task_hash != read_task_hash(task):
                raise ValueError("compiled READ or REDUCE differs from the selected logical step")
            if candidate.action != "REBUILD":
                raise ValueError("internal reading steps use independent native REBUILD epochs")
            empty_on_limit = (task.kind == "READ" and plan.coverage_scope == "selected_sources"
                and not any(need.operation in {"count", "sum", "group"} for need in plan.task_state.needs))
            record = await self._generate_step(attempt_id, snapshot, plan, task, selected_units, selected_claims,
                results=state.results, candidate=candidate, quote=quote, quote_ms=quote_ms,
                reserve=reserve, empty_on_limit=empty_on_limit)
        if cpu:
            record = completed_reduction_record(record, task, selected_claims, state.results)
            self.store.save_execution_step(record, "COMPLETE")
        graph = ClaimGraph(state.claims)
        for claim in record.result.claims:
            graph.add(claim)
        coverage = (*state.coverage, *((plan.coverage_scope, span.source_key.document_id,
            unit.unit_id, task.step_id) for unit in selected_units
            if unit.unit_id in record.result.covered_units for span in unit.source_spans))
        results = (*state.results, record.result)
        unresolved = unresolved_needs(results)
        disposition = ("partial" if graph.claims else "retrieval_gap") if unresolved else plan.selected_set.disposition
        answer_plan = graph.answer_plan(plan.task_state, disposition, results)
        prior = {point.need_id: point for point in plan.answer_plan.points}
        answer_plan = replace(answer_plan, points=tuple(point if point.claim_ids or point.need_id not in prior
            else replace(point, source_spans=prior[point.need_id].source_spans) for point in answer_plan.points))
        executed = {row.step_id: row.task for row in (*state.steps, record)}
        tasks = {row.step_id: row for row in plan.steps}
        tasks.update(executed)
        usage = (0 if initial_auxiliary_usage is None else auxiliary_tokens(initial_auxiliary_usage))
        gap = auxiliary_tokens(state.auxiliary_usage)
        spent = (usage+gap+sum(len(row.input_ids)+len(row.output_ids) for row in (*state.steps, record))
                 if usage is not None and gap is not None else None)
        plan = replace(plan, steps=tuple(tasks.values()), answer_plan=answer_plan, online_tokens_spent=spent)
        self.store.save_context_plan(attempt_id, plan)
        return ReadingOutcome(plan, results, (*state.steps, record), tuple(graph.claims.values()),
            tuple(dict.fromkeys(coverage)), unresolved, auxiliary_usage=state.auxiliary_usage)

    async def _observe_prefix(self, plan, task, units, record):
        if self.observe_prefix is None or task.kind != "READ" or plan.read_layout != "source_first":
            return record
        started = time.monotonic()
        receipt = await self.observe_prefix(plan, task, units, record)
        elapsed_ms = (time.monotonic()-started)*1000
        observe_control = getattr(self.observe_prefix, "observe_control", None)
        if observe_control is not None:
            observe_control(elapsed_ms, receipt)
        record = replace(record, metrics={**record.metrics, "evidence_prefix": receipt,
            "prefix_control_ms": elapsed_ms})
        self.store.save_execution_step(record, "COMPLETE")
        return record

    async def _retire_step(self, record):
        epoch = record.engine_epoch
        operation_id = f"retire:{epoch.engine_session_id}"
        ack = await self.engine.retire(epoch.engine_session_id, epoch.session_generation,
            operation_id, epoch.engine_restart_id, "reading_step_complete")
        if not ack.acknowledged:
            raise EngineError("READ_RETIRE_PENDING", record.step_id)
        self.store.ack_retire(operation_id)

    async def finish_completed_step(self, plan, record):
        """Finish prefix observation and retirement using the paid, saved input."""
        by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        units = tuple(by_id[identity] for identity in record.task.input_unit_ids)
        record = await self._observe_prefix(plan, record.task, units, record)
        await self._retire_step(record)
        return record

    async def _generate_step(self, attempt_id, snapshot, plan, task, units, claims, *, results=(),
                             candidate, text_callback=False, quote=None, quote_ms=None, reserve=None,
                             empty_on_limit=False):
        step_started = time.monotonic()
        identity = f"{attempt_id}:{task.step_id}"
        messages = candidate.messages
        if len(candidate.input_ids) > task.input_budget:
            raise ReadingBudgetExceeded(f"{task.step_id} input exceeds {task.input_budget}")
        if len(candidate.input_ids) + task.output_budget > plan.budget.window_tokens:
            raise ReadingBudgetExceeded(f"{task.step_id} exceeds the model window")
        persisted = self.store.execution_step(attempt_id, task.step_id)
        generation = None
        if persisted is None:
            info = await self.engine.info()
            epoch = ContextEpoch(snapshot.session_id, 0, f"step-{identity}", 0,
                                 reason=task.kind.lower(), engine_restart_id=info.engine_restart_id)
            record = StepRecord(attempt_id, task.step_id, task, epoch, messages, candidate.input_ids)
            record = replace(record, metrics={"generation_kind":
                step_generation_kind(task.kind, task.operation, plan.reading_mode),
                "external_dialogue": text_callback,
                "read_protocol_version": (None if text_callback else
                    task.output_schema["properties"]["format_version"]["const"]),
                "completion_reserve": to_jsonable(reserve),
                "request_arrived_monotonic": self.request_arrived,
                "phase_started_monotonic": step_started})
            self.store.save_execution_step(record, "PLANNED")
            stage = "PLANNED"
        else:
            stage, record, generation = persisted
            epoch = record.engine_epoch
            if record.task != task or record.input_ids != candidate.input_ids:
                raise ValueError("persisted step input differs from the reading plan")
            if stage == "COMPLETE":
                return record if text_callback else await self.finish_completed_step(plan, record)
        if stage == "PLANNED":
            info = await self.engine.info()
            prepared_at = time.monotonic()
            ack = await self.engine.prepare(epoch.engine_session_id, f"prepare:{identity}",
                epoch.engine_restart_id,
                **({"business_session_id": snapshot.session_id}
                   if info.business_session_identity_supported else {}))
            epoch = replace(epoch, session_generation=ack.session_generation)
            record = replace(record, engine_epoch=epoch,
                             metrics={**record.metrics, "prepare_ms": (time.monotonic()-prepared_at)*1000})
            self.store.save_execution_step(record, "PREPARED")
            stage = "PREPARED"
        if stage == "SUBMITTED":
            try:
                status = await self.engine.status(engine_session_id=epoch.engine_session_id,
                    attempt_id=identity, expected_restart_id=epoch.engine_restart_id)
            except EngineError as exc:
                if exc.code == "STALE_RESTART":
                    self.store.save_execution_step(replace(record, metrics={
                        **record.metrics, "terminal_state": exc.code}), "FAILED")
                raise
            if status.result is None and (not status.terminal or status.result_pending):
                raise EngineError("READ_STEP_OUTCOME_PENDING", status.state)
            if status.result is None:
                self.store.save_execution_step(replace(record, receipt=status.receipt, metrics={
                    **record.metrics, "terminal_state": status.state, "output_tokens": None}), "FAILED")
                raise EngineError("READ_STEP_RESULT_UNAVAILABLE" if status.state == "COMPLETED"
                                  else "READ_STEP_FAILED", status.state)
            generation = status.result
        elif stage == "PREPARED":
            if quote is None:
                quote_started = time.monotonic()
                quote = await self.engine.quote_batch((candidate,), self.template.fingerprint)
                quote_ms = (time.monotonic()-quote_started)*1000
            if quote.engine_restart_id != epoch.engine_restart_id:
                raise EngineError("READ_STEP_QUOTE_RESTART")
            matching = tuple(row for row in quote.quotes if row.candidate_id == candidate.candidate_id)
            if (len(matching) != 1 or matching[0].input_hash != candidate.input_hash
                    or matching[0].input_len != len(candidate.input_ids)
                    or quote.fingerprint != candidate.fingerprint):
                raise EngineError("READ_STEP_QUOTE_INPUT_MISMATCH")
            record = replace(record, metrics={**record.metrics,
                "quote_ms": quote_ms,
                "cache_quote": to_jsonable(quote),
                "active_requests": quote.active_requests, "queued_requests": quote.queued_requests,
                "cache_config_hash": quote.cache_config_hash})
            self.store.save_execution_step(record, "PREPARED")
            submitted = time.monotonic()
            record = replace(record, metrics={**record.metrics,
                "native_quote_work": quote_observation(quote, matching[0], submitted),
                "submitted_from_arrival_ms": (submitted-self.request_arrived)*1000
                    if self.request_arrived is not None else None})
            self.store.save_execution_step(record, "SUBMITTED")
            request = (candidate_request(candidate, task.output_budget) if text_callback
                       else step_request(candidate, task))
            generation = await self.engine.generate(**request,
                session_id=epoch.engine_session_id, expected_generation=epoch.session_generation,
                expected_restart_id=epoch.engine_restart_id, attempt_id=identity,
                fingerprint=self.template.fingerprint)
        if generation is not None:
            metrics = {**record.metrics,
                "elapsed_ms": generation.elapsed_ms, "ttft_ms": generation.ttft_ms,
                "last_token_ms": generation.last_token_ms,
                "queue_ms": generation.queue_ms, "prefill_ms": generation.prefill_ms,
                "decode_ms": generation.decode_ms,
                "forward_to_first_token_ms": generation.forward_to_first_token_ms,
                "server_elapsed_ms": generation.server_elapsed_ms,
                "actual_device_hit_tokens": generation.receipt.actual_device_hit_tokens,
                "actual_host_loaded_tokens": generation.receipt.actual_host_loaded_tokens,
                "actual_prefill_tokens": generation.receipt.actual_prefill_tokens,
                "finish_reason": generation.finish_reason}
            if (metrics.get("submitted_from_arrival_ms") is not None
                    and generation.ttft_ms is not None):
                metrics["first_token_from_arrival_ms"] = metrics["submitted_from_arrival_ms"]+generation.ttft_ms
            record = replace(record, output_ids=generation.output_ids,
                receipt=generation.receipt, metrics=metrics,
                result=ReadResult(task.step_id, raw_output=generation.text,
                    input_tokens=len(candidate.input_ids), output_tokens=len(generation.output_ids)))
            self.store.save_execution_step(record, "RESULT_READY", generation)
            receipt = record.receipt
            if (receipt.rid, receipt.attempt_id, receipt.engine_session_id,
                    receipt.session_generation, receipt.engine_restart_id, receipt.input_hash) != (
                    identity, identity, epoch.engine_session_id, epoch.session_generation,
                    epoch.engine_restart_id, candidate.input_hash):
                raise EngineError("READ_STEP_RECEIPT_MISMATCH")
            if not generation.generation_complete or not receipt.generation_complete:
                raise EngineError("READ_STEP_INCOMPLETE")
            native_work = quote_realization(record.metrics.get("native_quote_work"), receipt)
            observed = self.request_compiler.observe_executed(candidate, receipt,
                cache_config_hash=record.metrics["cache_config_hash"],
                engine_restart_id=record.engine_epoch.engine_restart_id)
            record = replace(record, metrics={**record.metrics, "source_view": observed, "native_quote_work": native_work})
            self.store.save_execution_step(record, "RESULT_READY", generation)
            parse_started = time.monotonic()
            if not text_callback and generation.finish_reason in {"length", "max_tokens"}:
                if not empty_on_limit:
                    raise EngineError("READ_STEP_OUTPUT_TRUNCATED", task.step_id)
                result = replace(record.result, elapsed_ms=generation.elapsed_ms or 0)
                record = replace(record, metrics={**record.metrics, "output_budget_exhausted": True})
            else:
                result = (record.result if text_callback else
                    parse_read_result(task, generation.text, units, claims,
                        elapsed_ms=generation.elapsed_ms or 0, input_tokens=len(candidate.input_ids),
                        output_tokens=len(generation.output_ids)))
            record = replace(record, result=result,
                             metrics={**record.metrics, "parse_ms": (time.monotonic()-parse_started)*1000})
            if task.kind == "REDUCE":
                by_id = {claim.claim_id: claim for previous in results for claim in previous.claims}
                by_id.update((claim.claim_id, claim) for claim in claims)
                basis = (tuple(by_id[identity] for identity in task.reduction_basis_claim_ids)
                         if task.reduction_basis_claim_ids else claims)
                logical = logical_reduction_task(plan, task, basis, results)
                pending, frontier, mode = reduction_schedule(
                    plan, logical, basis, (*results, result), snapshot, self.template)
                if pending is None:
                    record = completed_reduction_record(record, logical, basis, results,
                                                        frontier=frontier, mode=mode)
                else:
                    frontier = (result.claims if task.operation == "numeric_select"
                                else claim_frontier(claims, result.claims))
                    record = completed_reduction_record(record, task, claims, results, frontier=frontier)
            self.store.save_execution_step(record, "COMPLETE")
            record = await self._observe_prefix(plan, task, units, record)
            retire_started = time.monotonic()
            await self._retire_step(record)
            step_ended = time.monotonic()
            phase_started = record.metrics.get("phase_started_monotonic")
            record = replace(record, metrics={**record.metrics,
                "retire_ms": (time.monotonic()-retire_started)*1000,
                "step_wall_ms": (step_ended-phase_started)*1000 if phase_started is not None else None,
                "phase_window": {"start": phase_started, "end": step_ended,
                    "clock": "client_monotonic", "scope": "step_wall"} if phase_started is not None else None})
            self.store.save_execution_step(record, "COMPLETE")
            if self.events is not None:
                self.events.emit("reading_step_complete", attempt_id=attempt_id, step=record)
        return record

    async def execute_reads(self, attempt_id, snapshot, plan, *, initial_auxiliary_usage=None,
                            initial_outcome=None):
        from .evidence_planner import _passages, close_dependencies
        if (plan.budget.max_passages is not None
                and len(_passages(plan.selected_set.selected_units)) > plan.budget.max_passages):
            raise ReadingBudgetExceeded("initial plan exceeds the original passage allocation")
        seed = seed_reading_outcome(self.store, attempt_id, plan, initial_outcome)
        plan = (bind_answer_window(seed.plan, self.template, snapshot)
                if seed.plan.answer_allocation is None else seed.plan)
        self.store.save_context_plan(attempt_id, plan)
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        graph, results, records, coverage = ClaimGraph(seed.claims), list(seed.results), [], list(seed.coverage)
        result_indices = {result.step_id: index for index, result in enumerate(results)}
        record_indices = {}
        covered_unit_ids = {identity for result in results for identity in result.covered_units}
        progress = self.store.reading_progress(attempt_id)
        expanded = set(progress.expanded) if progress is not None else set()
        blocked_open_needs = {(need.need_id, need.missing_relation, need.bound_entities, need.temporal_scope): need
                             for need in progress.blocked_open_needs} if progress is not None else {}
        gap_auxiliary_usage = (progress.auxiliary_usage if progress is not None else
            {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []})
        initial_tokens = (0 if initial_auxiliary_usage is None else auxiliary_tokens(initial_auxiliary_usage))
        reader_spend = {record.step_id: step_token_spend(stage, record)
                       for stage, record in self.store.execution_steps_with_stage(attempt_id)}
        recovering = bool(reader_spend)
        started = time.monotonic()
        arrived = self.request_arrived if self.request_arrived is not None else started

        def paid_steps():
            gaps = {name.split(":", 2)[1] for name, _ in plan.auxiliary_ms if name.startswith("gap:")}
            return len(records)+len(gaps)

        def refresh_usage():
            nonlocal plan
            extra_tokens = auxiliary_tokens(gap_auxiliary_usage)
            spent = (initial_tokens+extra_tokens+sum(reader_spend.values())
                if initial_tokens is not None and extra_tokens is not None
                and all(value is not None for value in reader_spend.values()) else None)
            plan = replace(plan, online_tokens_spent=spent)
            self.store.save_context_plan(attempt_id, plan)
            return spent

        refresh_usage()

        def accept(record, selected_units=()):
            nonlocal plan
            result = record.result
            for claim in result.claims:
                graph.add(claim)
            if record.step_id in result_indices:
                results[result_indices[record.step_id]] = result
            else:
                result_indices[record.step_id] = len(results)
                results.append(result)
            if record.step_id in record_indices:
                index = record_indices[record.step_id]
                records[index] = record
            else:
                record_indices[record.step_id] = len(records)
                records.append(record)
            for unit in selected_units:
                if unit.unit_id in covered_unit_ids:
                    continue
                covered_unit_ids.add(unit.unit_id)
                for document_id in dict.fromkeys(span.source_key.document_id for span in unit.source_spans):
                    coverage.append((plan.coverage_scope, document_id, unit.unit_id, record.step_id))
            tasks = {step.step_id: step for step in plan.steps}
            tasks[record.step_id] = record.task
            plan = replace(plan, steps=tuple(tasks.values()))

        async def run_model(task, selected_units, selected_claims):
            persisted = self.store.execution_step(attempt_id, task.step_id)
            if persisted is not None:
                task = persisted[1].task
                selected_units = tuple(units[identity] for identity in task.input_unit_ids)
            task = replace(task, input_claim_ids=tuple(claim.claim_id for claim in selected_claims),
                output_schema=output_schema(task, selected_units, selected_claims,
                    reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity))
            variants = self.request_compiler.compile_read(plan, task, selected_units, selected_claims,
                snapshot, outputs=(task.output_budget,), cache_scope=await self.engine.info())
            quote, quote_ms = None, None
            if persisted is None and len(variants) > 1:
                quote_started = time.monotonic()
                quote = await self.engine.quote_batch(variants, self.template.fingerprint)
                quote_ms = (time.monotonic()-quote_started)*1000
                candidate = self.request_compiler.choose_variant(variants, quote)
            else:
                candidate = variants[0]
            task = replace(task, input_unit_ids=candidate.read_unit_order,
                output_schema=output_schema(task, tuple(units[identity] for identity in candidate.read_unit_order),
                    selected_claims, reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity))
            selected_units = tuple(units[identity] for identity in task.input_unit_ids)
            reserve = (require_completion(plan, results, refresh_usage(), paid_steps(), task, candidate,
                                          read_all=True, arrived=arrived)
                       if persisted is None or persisted[0] in {"PLANNED", "PREPARED"} else None)
            try:
                record = await self._generate_step(attempt_id, snapshot, plan, task,
                    selected_units, selected_claims, results=tuple(results), candidate=candidate,
                    quote=quote, quote_ms=quote_ms, reserve=reserve)
            finally:
                saved = self.store.execution_step(attempt_id, task.step_id)
                if saved is not None:
                    reader_spend[task.step_id] = step_token_spend(saved[0], saved[1])
                    refresh_usage()
            accept(record, selected_units)
            return record.result

        def complete_reduction(task, basis, frontier, mode):
            record = completed_reduction_record(records[-1], task, basis, results,
                                                frontier=frontier, mode=mode)
            if record == records[-1]:
                return
            records[-1], results[-1] = record, record.result
            self.store.save_execution_step(record, "COMPLETE")

        async def run_reduction(task, selected_claims):
            while True:
                pending, _, _ = reduction_schedule(
                    plan, task, selected_claims, results, snapshot, self.template)
                if pending is None:
                    return
                child, inputs = pending
                await run_model(child, (), inputs)

        async def observe_current():
            if self.observe_state is not None and not recovering:
                state = ReadingOutcome(plan, tuple(results), tuple(records), tuple(graph.claims.values()),
                    tuple(coverage), unresolved_needs(results), auxiliary_usage=gap_auxiliary_usage)
                await self.observe_state(state, expanded)

        await observe_current()
        while True:
            ready = advance_read_plan(plan, tuple(results))
            if not ready or all(task.kind == "REDUCE" for task in ready):
                open_needs = tuple(need for need in unresolved_needs(results)
                    if (need.need_id, need.missing_relation, need.bound_entities, need.temporal_scope) not in expanded)
                if open_needs and self.extend_open_needs is not None:
                    reserve = require_completion(plan, results, refresh_usage(), paid_steps(), None,
                                                 read_all=True, arrived=arrived)
                    gap_started = time.monotonic()
                    extension_pool = await self.extend_open_needs(plan, open_needs, tuple(graph.claims.values()))
                    gap_elapsed = (time.monotonic()-gap_started)*1000
                    from .context_schema import EvidencePool
                    if isinstance(extension_pool, EvidencePool):
                        new_units = extension_pool.units
                        gap_auxiliary_usage = merge_auxiliary_usage(gap_auxiliary_usage,
                                                                    extension_pool.auxiliary_usage)
                        phases = extension_pool.auxiliary_ms or (("control", gap_elapsed),)
                        source_blocks = tuple({block.key: block for block in
                            (*plan.selected_set.source_blocks, *extension_pool.source_blocks)}.values())
                    else:
                        new_units = tuple(extension_pool)
                        gap_auxiliary_usage = merge_auxiliary_usage(gap_auxiliary_usage, None)
                        phases = (("control", gap_elapsed),)
                        source_blocks = plan.selected_set.source_blocks
                    phase_id = len(expanded)
                    plan = replace(plan, auxiliary_ms=(*plan.auxiliary_ms,
                        *((f"gap:{phase_id}:{name}", value) for name, value in phases)))
                    expanded.update((need.need_id, need.missing_relation, need.bound_entities,
                                     need.temporal_scope) for need in open_needs)
                    refresh_usage()
                    self.store.save_reading_progress(attempt_id, plan, ReadingProgress(
                        tuple(sorted(expanded)), tuple(blocked_open_needs.values()), gap_auxiliary_usage))
                    require_online_tokens(plan.budget.max_total_tokens, plan.online_tokens_spent,
                        reserve.input_tokens+reserve.output_tokens, "OpenNeed result")
                    combined = {**{unit.unit_id: unit for unit in new_units}, **units}
                    closure = close_dependencies(tuple(unit.unit_id for unit in new_units
                                                        if unit.unit_id not in units), combined)
                    novel = tuple(unit for unit in closure if unit.unit_id not in units)
                    if (plan.budget.max_passages is not None and len(_passages(
                            (*units.values(), *novel))) > plan.budget.max_passages):
                        blocked_open_needs.update(((need.need_id, need.missing_relation,
                            need.bound_entities, need.temporal_scope), need) for need in open_needs)
                        if self.events is not None:
                            self.events.emit("reading_gap_budget_stop", attempt_id=attempt_id,
                                reason="original_passage_budget", open_needs=open_needs,
                                max_passages=plan.budget.max_passages)
                        novel = ()
                    if novel:
                        parents = tuple(dict.fromkeys(parent for need in open_needs
                                                       for parent in need.parent_claim_ids))
                        predecessors = claim_predecessor_steps(results, parents)
                        extension = build_read_plan(plan.task_state,
                            replace(plan.selected_set, selected_units=closure, source_blocks=source_blocks), plan.coverage_scope,
                            plan.budget, template=self.template, snapshot=snapshot,
                            input_claims=tuple(graph.claims[parent] for parent in parents),
                            reading_mode=plan.reading_mode, separate_needs=plan.separate_needs,
                            numeric_identity=plan.numeric_identity, read_layout=plan.read_layout,
                            read_packing=plan.read_packing, answer_system_prompt=plan.answer_system_prompt)
                        prefix = f"hop:{len(expanded)}:"
                        extension_units = {unit.unit_id: unit for unit in extension.selected_set.selected_units}
                        steps = []
                        for step in extension.steps:
                            step = replace(step, step_id=prefix+step.step_id,
                                input_claim_ids=parents, predecessor_step_ids=predecessors)
                            steps.append(replace(step, output_schema=output_schema(step,
                                tuple(extension_units[unit_id] for unit_id in step.input_unit_ids),
                                tuple(graph.claims[parent] for parent in parents),
                                reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity)))
                        steps = tuple(steps)
                        units.update((unit.unit_id, unit) for unit in extension.selected_set.selected_units)
                        selected_set = replace(plan.selected_set, selected_units=tuple(units.values()),
                                               source_blocks=source_blocks)
                        if isinstance(extension_pool, EvidencePool) and extension_pool.selected_set is not None:
                            updated_set = extension_pool.selected_set
                            gaps = {need.need_id for need in open_needs}
                            support = tuple(item for item in selected_set.need_support if item.need_id not in gaps)
                            support += tuple(item for item in updated_set.need_support if item.need_id in gaps)
                            selected_set = replace(selected_set, need_support=support,
                                quality_prediction=updated_set.quality_prediction,
                                disposition=updated_set.disposition)
                        plan = replace(plan, steps=plan.steps+steps,
                            coverage_units=tuple(dict.fromkeys((*plan.coverage_units, *extension.coverage_units))),
                            selected_set=selected_set)
                        plan = bind_answer_window(replace(plan, answer_allocation=None), self.template,
                            snapshot, preserve_step_ids=tuple(record.step_id for record in records))
                        self.store.save_reading_progress(attempt_id, plan, ReadingProgress(
                            tuple(sorted(expanded)), tuple(blocked_open_needs.values()), gap_auxiliary_usage))
                        continue
                    self.store.save_reading_progress(attempt_id, plan, ReadingProgress(
                        tuple(sorted(expanded)), tuple(blocked_open_needs.values()), gap_auxiliary_usage))
                if not ready:
                    break
            for task in ready:
                if task.step_id in {result.step_id for result in results}:
                    continue
                selected_units = tuple(units[unit_id] for unit_id in task.input_unit_ids)
                selected_claims = tuple(graph.claims[claim_id] for claim_id in task.input_claim_ids)
                persisted = self.store.execution_step(attempt_id, task.step_id)
                if persisted is not None and persisted[0] == "COMPLETE":
                    if persisted[1].engine_epoch is None:
                        accept(persisted[1], selected_units)
                        if task.reduction_basis_claim_ids and persisted[1].result.reduction is None:
                            basis = tuple(graph.claims[identity] for identity in task.reduction_basis_claim_ids)
                            complete_reduction(task, basis,
                                claim_frontier(basis, persisted[1].result.claims), "merged")
                    elif task.reduction_basis_claim_ids and persisted[1].result.reduction is None:
                        basis = tuple(graph.claims[identity] for identity in task.reduction_basis_claim_ids)
                        await run_reduction(task, basis)
                    else:
                        await run_model(task, selected_units, selected_claims)
                    continue
                if (plan.reading_mode == "typed_reduce" and task.kind == "REDUCE"
                        and task.operation in {"union", "count", "sum", "group"}):
                    reserve = require_completion(plan, results, refresh_usage(), paid_steps(), task,
                                                 read_all=True, arrived=arrived)
                    tick = time.monotonic()
                    native_workload = await self.engine.metrics()
                    try:
                        claim = aggregate_numeric_view(task, selected_claims, plan, results)
                    except ValueError as exc:
                        self.store.save_execution_step(StepRecord(attempt_id, task.step_id, task,
                            metrics={"cpu_reduce_ms": (time.monotonic()-tick)*1000,
                                     "completion_reserve": to_jsonable(reserve), "error": str(exc)}), "FAILED")
                        raise
                    ended = time.monotonic()
                    result = ReadResult(task.step_id, (claim,), elapsed_ms=(ended-tick)*1000)
                    record = StepRecord(attempt_id, task.step_id, task, result=result,
                                        metrics={"cpu_reduce_ms": result.elapsed_ms,
                                            "completion_reserve": to_jsonable(reserve),
                                            "native_workload": native_workload,
                                            "phase_window": {"start": tick, "end": ended,
                                                "clock": "client_monotonic", "scope": "step_wall"}})
                    record = completed_reduction_record(record, task, selected_claims, results)
                    self.store.save_execution_step(record, "COMPLETE")
                    accept(record, selected_units)
                elif task.kind == "REDUCE":
                    await run_reduction(task, selected_claims)
                else:
                    await run_model(task, selected_units, selected_claims)
                await observe_current()
            self.store.save_context_plan(attempt_id, plan)
        if plan.coverage_scope == "all_supplied_documents":
            from .context_budget import completed_source_coverage
            if not completed_source_coverage(plan, results, tuple(units[identity] for identity in plan.coverage_units)):
                raise ValueError("full-document coverage is incomplete")
        unresolved = unresolved_needs(results)
        disposition = plan.selected_set.disposition
        if unresolved:
            disposition = "partial" if graph.claims else "retrieval_gap"
        answer_plan = graph.answer_plan(plan.task_state, disposition, results)
        support_spans = {need.need_id: tuple({span.span_id: span
            for support in plan.selected_set.need_support if support.need_id == need.need_id
            for span in support.source_spans}.values()) for need in plan.task_state.needs
            if any(support.need_id == need.need_id for support in plan.selected_set.need_support)}
        planned_points = {point.need_id: point for point in plan.answer_plan.points}
        points = []
        for point in answer_plan.points:
            if point.claim_ids:
                points.append(point)
            elif point.need_id in support_spans:
                points.append(replace(point, source_spans=support_spans[point.need_id]))
            elif point.need_id in planned_points:
                points.append(replace(point, source_spans=planned_points[point.need_id].source_spans))
            else:
                points.append(point)
        plan = replace(plan, answer_plan=replace(answer_plan, points=tuple(points)))
        self.store.save_context_plan(attempt_id, plan)
        return ReadingOutcome(plan, tuple(results), tuple(records), tuple(graph.claims.values()),
                              tuple(coverage), unresolved, auxiliary_usage=gap_auxiliary_usage)
