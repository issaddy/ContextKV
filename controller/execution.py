"""Execute durable reading steps inside one outer user attempt."""
from __future__ import annotations

from dataclasses import replace
import time

from .claim_graph import ClaimGraph, claim_frontier
from .context_schema import ReadResult, ReadingOutcome, ReductionState, StepRecord
from .engine_client import EngineError
from .reading import (advance_read_plan, build_read_plan, output_schema, parse_read_result,
                      step_generation_kind, step_messages, unresolved_needs)
from .reduction import reduce_claims
from .schema import ContextEpoch, PromptCandidate
from .serialization import to_jsonable
from .session_store import ReadingProgress
from .templates import step_request


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
    if not inputs and task.operation in {"count", "sum"}:
        selections = [result.reduction for result in results if result.reduction is not None
                      and result.reduction.operation == "numeric_select"
                      and {result.reduction.step_id, result.step_id}.intersection(task.predecessor_step_ids)]
        if selections:
            # The empty view is proved by the completed selection over the read scope.
            claim = replace(claim, parent_claim_ids=selections[-1].basis_claim_ids,
                source_spans=tuple(dict.fromkeys(span for unit in plan.selected_set.selected_units
                                                 for span in unit.source_spans)))
    return claim


class ReadingBudgetExceeded(ValueError):
    pass


class OnlineTokenBudgetUnmeasured(ValueError):
    pass


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


def step_token_spend(stage, record):
    if stage == "SUBMITTED":
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


class ReadingExecutor:
    def __init__(self, store, engine, template, *, extend_open_needs=None, events=None,
                 request_arrived=None, observe_prefix=None, observe_state=None):
        self.store, self.engine, self.template = store, engine, template
        self.extend_open_needs = extend_open_needs
        self.events = events
        self.request_arrived = request_arrived
        self.observe_prefix = observe_prefix
        self.observe_state = observe_state

    async def execute_one(self, attempt_id, snapshot, plan, task, state, *,
                          initial_auxiliary_usage=None, candidate=None):
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
        if len(state.steps)+2 > plan.budget.max_steps:
            raise ReadingBudgetExceeded("selected step leaves no terminal ANSWER allocation")
        self.store.save_context_plan(attempt_id, plan)
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        selected_units = tuple(units[identity] for identity in task.input_unit_ids)
        claims = {claim.claim_id: claim for claim in state.claims}
        selected_claims = tuple(claims[identity] for identity in task.input_claim_ids)
        task = replace(task, output_schema=output_schema(task, selected_units, selected_claims,
            reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity))
        cpu = (plan.reading_mode == "typed_reduce" and task.kind == "REDUCE"
               and task.operation in {"union", "count", "sum", "group"})
        if cpu:
            tick = time.monotonic()
            native_workload = await self.engine.metrics()
            claim = aggregate_numeric_view(task, selected_claims, plan, state.results)
            ended = time.monotonic()
            result = ReadResult(task.step_id, (claim,), elapsed_ms=(ended-tick)*1000)
            record = StepRecord(attempt_id, task.step_id, task, result=result,
                                metrics={"cpu_reduce_ms": result.elapsed_ms,
                                         "native_workload": native_workload,
                                         "phase_window": {"start": tick, "end": ended,
                                             "clock": "client_monotonic", "scope": "step_wall"},
                                         "request_arrived_monotonic": self.request_arrived})
            self.store.save_execution_step(record, "COMPLETE")
        else:
            rendered = self.template.render(step_messages(plan, task, selected_units, selected_claims, snapshot))
            if candidate is None or (candidate.input_ids, candidate.output_budget) != (
                    rendered.input_ids, task.output_budget):
                raise ValueError("selected quote input/output differs from the executed logical step")
            if candidate.action != "REBUILD":
                raise ValueError("internal reading steps use independent native REBUILD epochs")
            usage = (0 if initial_auxiliary_usage is None else auxiliary_tokens(initial_auxiliary_usage))
            gap = auxiliary_tokens(state.auxiliary_usage)
            tokens = (usage+gap+sum(len(row.input_ids)+len(row.output_ids) for row in state.steps)
                      if usage is not None and gap is not None else None)
            require_online_tokens(plan.budget.max_total_tokens, tokens,
                len(rendered.input_ids)+task.output_budget, task.kind)
            record = await self._generate_step(attempt_id, snapshot, plan, task, selected_units, selected_claims)
        if task.kind == "REDUCE":
            frontier = (record.result.claims if task.operation == "numeric_select"
                        else claim_frontier(selected_claims, record.result.claims))
            record = replace(record, result=replace(record.result, reduction=ReductionState(
                task.step_id, task.need_ids, task.operation, reduction_basis_ids(task, selected_claims, state.results),
                tuple(claim.claim_id for claim in frontier), "merged")))
            self.store.save_execution_step(record, "COMPLETE")
        graph = ClaimGraph(state.claims)
        for claim in record.result.claims:
            graph.add(claim)
        coverage = (*state.coverage, *((plan.coverage_scope, span.source_key.document_id,
            unit.unit_id, task.step_id) for unit in selected_units for span in unit.source_spans))
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

    async def _observe_prefix(self, plan, task, units, rendered, record):
        if self.observe_prefix is None or task.kind != "READ" or plan.read_layout != "source_first":
            return record
        started = time.monotonic()
        receipt = await self.observe_prefix(plan, task, units, rendered, record)
        record = replace(record, metrics={**record.metrics, "evidence_prefix": receipt,
            "prefix_control_ms": (time.monotonic()-started)*1000})
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

    async def _generate_step(self, attempt_id, snapshot, plan, task, units, claims):
        step_started = time.monotonic()
        identity = f"{attempt_id}:{task.step_id}"
        messages = step_messages(plan, task, units, claims, snapshot)
        rendered = self.template.render(messages)
        if len(rendered.input_ids) > task.input_budget:
            raise ReadingBudgetExceeded(f"{task.step_id} input exceeds {task.input_budget}")
        if len(rendered.input_ids) + task.output_budget > plan.budget.window_tokens:
            raise ReadingBudgetExceeded(f"{task.step_id} exceeds the model window")
        persisted = self.store.execution_step(attempt_id, task.step_id)
        generation = None
        if persisted is None:
            info = await self.engine.info()
            epoch = ContextEpoch(snapshot.session_id, 0, f"step-{identity}", 0,
                                 reason=task.kind.lower(), engine_restart_id=info.engine_restart_id)
            record = StepRecord(attempt_id, task.step_id, task, epoch, messages, rendered.input_ids)
            record = replace(record, metrics={"generation_kind":
                step_generation_kind(task.kind, task.operation, plan.reading_mode),
                "read_protocol_version": task.output_schema["properties"]["format_version"]["const"],
                "request_arrived_monotonic": self.request_arrived,
                "phase_started_monotonic": step_started})
            self.store.save_execution_step(record, "PLANNED")
            stage = "PLANNED"
        else:
            stage, record, generation = persisted
            epoch = record.engine_epoch
            if record.task != task or record.input_ids != rendered.input_ids:
                raise ValueError("persisted step input differs from the reading plan")
            if stage == "COMPLETE":
                record = await self._observe_prefix(plan, task, units, rendered, record)
                await self._retire_step(record)
                return record
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
            status = await self.engine.status(engine_session_id=epoch.engine_session_id,
                attempt_id=identity, expected_restart_id=epoch.engine_restart_id)
            if status.result is None:
                if status.terminal and status.state in {"FAILED", "CANCELLED", "REJECTED", "RETIRED"}:
                    raise EngineError("READ_STEP_FAILED", status.state)
                raise EngineError("READ_STEP_OUTCOME_PENDING", status.state)
            generation = status.result
        elif stage == "PREPARED":
            candidate = PromptCandidate(identity, "REBUILD", snapshot.version,
                messages, rendered.input_ids, rendered.input_hash, frozenset(), frozenset(),
                fingerprint=self.template.fingerprint, corpus_snapshot_id=snapshot.corpus_snapshot_id,
                context_plan_id=plan.plan_id, evidence_set_id=plan.selected_set.set_id,
                output_budget=task.output_budget,
                generation_kind=step_generation_kind(task.kind, task.operation, plan.reading_mode))
            quote_started = time.monotonic()
            quote = await self.engine.quote_batch((candidate,), self.template.fingerprint)
            record = replace(record, metrics={**record.metrics,
                "quote_ms": (time.monotonic()-quote_started)*1000,
                "cache_quote": to_jsonable(quote),
                "active_requests": quote.active_requests, "queued_requests": quote.queued_requests,
                "cache_config_hash": quote.cache_config_hash})
            self.store.save_execution_step(record, "PREPARED")
            if quote.engine_restart_id != epoch.engine_restart_id:
                raise EngineError("READ_STEP_QUOTE_RESTART")
            submitted = time.monotonic()
            record = replace(record, metrics={**record.metrics,
                "submitted_from_arrival_ms": (submitted-self.request_arrived)*1000
                    if self.request_arrived is not None else None})
            self.store.save_execution_step(record, "SUBMITTED")
            generation = await self.engine.generate(**step_request(rendered, task),
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
                    input_tokens=len(rendered.input_ids), output_tokens=len(generation.output_ids)))
            self.store.save_execution_step(record, "RESULT_READY", generation)
            receipt = generation.receipt
            if (receipt.rid, receipt.attempt_id, receipt.engine_session_id,
                    receipt.session_generation, receipt.engine_restart_id, receipt.input_hash) != (
                    identity, identity, epoch.engine_session_id, epoch.session_generation,
                    epoch.engine_restart_id, rendered.input_hash):
                raise EngineError("READ_STEP_RECEIPT_MISMATCH")
            if not generation.generation_complete or not receipt.generation_complete:
                raise EngineError("READ_STEP_INCOMPLETE")
            if generation.finish_reason in {"length", "max_tokens"}:
                raise EngineError("READ_STEP_OUTPUT_TRUNCATED", task.step_id)
            parse_started = time.monotonic()
            result = parse_read_result(task, generation.text, units, claims,
                elapsed_ms=generation.elapsed_ms or 0, input_tokens=len(rendered.input_ids),
                output_tokens=len(generation.output_ids))
            record = replace(record, result=result,
                             metrics={**record.metrics, "parse_ms": (time.monotonic()-parse_started)*1000})
            self.store.save_execution_step(record, "COMPLETE")
            record = await self._observe_prefix(plan, task, units, rendered, record)
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

    async def execute_reads(self, attempt_id, snapshot, plan, *, initial_auxiliary_usage=None):
        from .evidence_planner import _passages, close_dependencies
        if self.observe_state is not None and self.store.execution_steps(attempt_id):
            raise ValueError("fixed state observation requires a newly executed attempt, not saved post-step state")
        if (plan.budget.max_passages is not None
                and len(_passages(plan.selected_set.selected_units)) > plan.budget.max_passages):
            raise ReadingBudgetExceeded("initial plan exceeds the original passage allocation")
        self.store.save_context_plan(attempt_id, plan)
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        graph, results, records, coverage = ClaimGraph(), [], [], []
        record_indices = {}
        covered_unit_ids = set()
        progress = self.store.reading_progress(attempt_id)
        expanded = set(progress.expanded) if progress is not None else set()
        blocked_open_needs = {(need.need_id, need.missing_relation, need.bound_entities, need.temporal_scope): need
                             for need in progress.blocked_open_needs} if progress is not None else {}
        gap_auxiliary_usage = (progress.auxiliary_usage if progress is not None else
            {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []})
        initial_tokens = (0 if initial_auxiliary_usage is None else auxiliary_tokens(initial_auxiliary_usage))
        reader_spend = {record.step_id: step_token_spend(stage, record)
                       for stage, record in self.store.execution_steps_with_stage(attempt_id)}
        started = time.monotonic()

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
            if record.step_id in record_indices:
                index = record_indices[record.step_id]
                results[index], records[index] = result, record
            else:
                record_indices[record.step_id] = len(records)
                results.append(result)
                records.append(record)
            for unit in selected_units:
                if unit.unit_id in covered_unit_ids:
                    continue
                covered_unit_ids.add(unit.unit_id)
                for document_id in dict.fromkeys(span.source_key.document_id for span in unit.source_spans):
                    coverage.append((plan.coverage_scope, document_id, unit.unit_id, record.step_id))
            if record.step_id not in {step.step_id for step in plan.steps}:
                plan = replace(plan, steps=plan.steps+(record.task,))

        async def run_model(task, selected_units, selected_claims):
            task = replace(task, input_claim_ids=tuple(claim.claim_id for claim in selected_claims),
                output_schema=output_schema(task, selected_units, selected_claims,
                    reading_mode=plan.reading_mode, numeric_identity=plan.numeric_identity))
            rendered = self.template.render(step_messages(plan, task, selected_units,
                                                          selected_claims, snapshot))
            if len(records) + 1 >= plan.budget.max_steps:
                raise ReadingBudgetExceeded("reading exhausted the request step allocation")
            persisted = self.store.execution_step(attempt_id, task.step_id)
            if persisted is None or persisted[0] in {"PLANNED", "PREPARED"}:
                require_online_tokens(plan.budget.max_total_tokens, refresh_usage(),
                    len(rendered.input_ids)+task.output_budget, task.kind)
            try:
                record = await self._generate_step(attempt_id, snapshot, plan, task,
                                                   selected_units, selected_claims)
            finally:
                saved = self.store.execution_step(attempt_id, task.step_id)
                if saved is not None:
                    reader_spend[task.step_id] = step_token_spend(saved[0], saved[1])
                    refresh_usage()
            accept(record, selected_units)
            return record.result

        def complete_reduction(task, basis, frontier, mode):
            completion = ReductionState(task.step_id, task.need_ids, task.operation,
                reduction_basis_ids(task, basis, results),
                tuple(claim.claim_id for claim in frontier), mode)
            record = records[-1]
            record = replace(record, result=replace(record.result, reduction=completion))
            records[-1], results[-1] = record, record.result
            self.store.save_execution_step(record, "COMPLETE")

        async def run_reduction(task, selected_claims):
            """Reduce source partitions, then merge whole result bundles.

            A merge consumes at least two bundles. N initial calls therefore
            admit at most N-1 merge calls; unmerged roots remain in the answer.
            """
            basis = selected_claims

            def size(current_task, claims):
                return len(self.template.render(step_messages(
                    plan, current_task, (), claims, snapshot)).input_ids)

            if size(task, basis) <= task.input_budget:
                result = await run_model(task, (), basis)
                frontier = (result.claims if task.operation == "numeric_select"
                            else claim_frontier(basis, result.claims))
                complete_reduction(task, basis, frontier, "selected" if task.operation == "numeric_select" else "merged")
                return
            packs, pack = [], []
            for claim in basis:
                child = replace(task, step_id=f"{task.step_id}:part:0:{len(packs)}",
                                reduction_basis_claim_ids=())
                if pack and size(child, (*pack, claim)) > task.input_budget:
                    packs.append(tuple(pack))
                    pack = []
                    child = replace(task, step_id=f"{task.step_id}:part:0:{len(packs)}",
                                    reduction_basis_claim_ids=())
                if size(child, (*pack, claim)) > task.input_budget:
                    raise ReadingBudgetExceeded("one claim exceeds the reduction input allocation")
                pack.append(claim)
            if pack:
                packs.append(tuple(pack))
            from .context_budget import merge_conclusion_tokens, split_conclusion_tokens
            targets = split_conclusion_tokens(task.retained_conclusion_tokens, len(packs))
            bundles = []
            for index, pack in enumerate(packs):
                child = replace(task, step_id=f"{task.step_id}:part:0:{index}",
                                reduction_basis_claim_ids=(), retained_conclusion_tokens=targets[index])
                result = await run_model(child, (), pack)
                frontier = (result.claims if task.operation == "numeric_select"
                            else claim_frontier(pack, result.claims))
                bundles.append((frontier, (result.step_id,), targets[index]))
            if task.operation == "numeric_select":
                complete_reduction(task, basis, tuple(claim for bundle in bundles for claim in bundle[0]), "selected")
                return
            level = 1
            while len(bundles) > 1:
                groups, group = [], []
                for bundle in bundles:
                    child = replace(task, step_id=f"{task.step_id}:merge:{level}:{len(groups)}",
                                    reduction_basis_claim_ids=())
                    trial = tuple(claim for item in (*group, bundle) for claim in item[0])
                    if group and size(child, trial) > task.input_budget:
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
                    claims = tuple(claim for item in group for claim in item[0])
                    predecessors = tuple(identity for item in group for identity in item[1])
                    child = replace(task,
                        step_id=task.step_id if len(groups) == 1 else f"{task.step_id}:merge:{level}:{index}",
                        predecessor_step_ids=predecessors,
                        reduction_basis_claim_ids=task.reduction_basis_claim_ids if len(groups) == 1 else (),
                        retained_conclusion_tokens=merge_conclusion_tokens(item[2] for item in group))
                    result = await run_model(child, (), claims)
                    merged.append((claim_frontier(claims, result.claims), (result.step_id,),
                                   child.retained_conclusion_tokens))
                bundles = merged
                level += 1
            frontier = tuple(claim for claims, _, _ in bundles for claim in claims)
            complete_reduction(task, basis, frontier, "merged" if len(bundles) == 1 else "forest")

        async def observe_current():
            if self.observe_state is not None:
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
                    require_online_tokens(plan.budget.max_total_tokens, refresh_usage(), 1, "OpenNeed")
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
                    require_online_tokens(plan.budget.max_total_tokens, plan.online_tokens_spent, 1, "OpenNeed result")
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
                                input_claim_ids=parents, predecessor_step_ids=tuple(dict.fromkeys(
                                    parent.rsplit(":claim:", 1)[0] for parent in parents)))
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
                        from .context_budget import bind_answer_window
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
                if len(records) + 1 >= plan.budget.max_steps:
                    raise ReadingBudgetExceeded("reading exhausted the request step allocation")
                if (plan.budget.max_complete_ms is not None
                        and (time.monotonic()-started)*1000 > plan.budget.max_complete_ms):
                    raise ReadingBudgetExceeded("reading exhausted the complete-time allocation")
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
                    tick = time.monotonic()
                    native_workload = await self.engine.metrics()
                    try:
                        claim = aggregate_numeric_view(task, selected_claims, plan, results)
                    except ValueError as exc:
                        self.store.save_execution_step(StepRecord(attempt_id, task.step_id, task,
                            metrics={"cpu_reduce_ms": (time.monotonic()-tick)*1000,
                                     "error": str(exc)}), "FAILED")
                        raise
                    ended = time.monotonic()
                    result = ReadResult(task.step_id, (claim,), elapsed_ms=(ended-tick)*1000)
                    record = StepRecord(attempt_id, task.step_id, task, result=result,
                                        metrics={"cpu_reduce_ms": result.elapsed_ms,
                                            "native_workload": native_workload,
                                            "phase_window": {"start": tick, "end": ended,
                                                "clock": "client_monotonic", "scope": "step_wall"}})
                    self.store.save_execution_step(record, "COMPLETE")
                    accept(record, selected_units)
                    complete_reduction(task, selected_claims, (claim,), "merged")
                elif task.kind == "REDUCE":
                    await run_reduction(task, selected_claims)
                else:
                    await run_model(task, selected_units, selected_claims)
                await observe_current()
            self.store.save_context_plan(attempt_id, plan)
        covered = {row[2] for row in coverage}
        if plan.coverage_scope == "all_supplied_documents" and covered != set(plan.coverage_units):
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
