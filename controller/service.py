"""Serial turns, pure planning, guarded generation and idempotent durable commit."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import logging
import time
import uuid
import statistics
from typing import Awaitable, Callable
from weakref import WeakValueDictionary

from .candidates import CandidatePolicy
from .admission import quote_observation, quote_realization
from .cost_model import CostModel
from .engine_client import Engine, EngineError, EngineInfo, PrepareAck, UnknownOutcomeError
from .evidence import conclusion_source_manifest
from .events import EventLog
from .planner import PlannerKnobs, choose_plan
from .schema import (ClientTiming, CompletionReceipt, CostEstimate, EvidenceBlock, EvidenceKey,
                     EvidenceSpan, GenerationResult, NoFeasiblePlan, Plan, PromptCandidate, QueryContext, TurnRecord)
from .serialization import dumps
from .session_store import (Attempt, IdempotencyConflict, PendingAttempt, SessionStore, StoreError, VersionConflict)
from .templates import ByteTemplate, candidate_request
from .context_schema import ContextBudget, ContextPlan, EvidenceSet, ReadTask, ReadingOutcome, TaskState
from .request_compiler import RequestCompiler, unique_answer_inputs
from .auxiliary_spend import capture_auxiliary_spend, reconcile_auxiliary_usage
from .execution import merge_auxiliary_usage


class ContextOverflow(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceSelection:
    """Material returned by a configured retriever inside the turn transaction."""
    evidence: tuple[EvidenceBlock, ...]
    retrieval_priority: tuple[EvidenceKey, ...]
    query_context: QueryContext | None = None
    context_plans: tuple[ContextPlan, ...] = ()
    open_need_provider: Callable | None = None
    auxiliary_usage: dict | None = None
    initial_outcomes: tuple[ReadingOutcome, ...] = ()


@dataclass(frozen=True)
class TurnResponse:
    session_id: str
    request_id: str
    attempt_id: str
    assistant_text: str
    action: str
    reason: str
    epoch_no: int
    ledger_version: int
    output_ids: tuple[int, ...]
    input_tokens: int
    output_tokens: int
    actual_device_hit_tokens: int
    actual_host_loaded_tokens: int
    actual_prefill_tokens: int
    complete_ms: float | None = None
    ttft_ms: float | None = None
    idempotent_replay: bool = False
    timing: ClientTiming | None = None
    receipt: CompletionReceipt | None = None
    prediction: CostEstimate | None = None
    control_ms: float | None = None
    engine_elapsed_ms: float | None = None
    server_ttft_ms: float | None = None
    tpot_ms: float | None = None
    timing_source: str | None = None
    truncated: bool = False
    branch_reason: str = ""
    retrieval_ms: float | None = None
    visible_evidence_manifest: tuple[EvidenceSpan, ...] = ()
    citation_scope: tuple[str, ...] = ()
    read_count: int = 0
    reduce_count: int = 0
    total_input_tokens: int | None = 0
    total_output_tokens: int | None = 0
    auxiliary_model_ms: float | None = 0.0
    internal_first_token_ms: float | None = None
    step_metrics: tuple[dict, ...] = ()
    reading_ms: float = 0.0
    total_device_hit_tokens: int = 0
    total_host_loaded_tokens: int = 0
    total_prefill_tokens: int = 0
    reader_input_tokens: int = 0
    reader_output_tokens: int = 0
    auxiliary_input_tokens: int | None = 0
    auxiliary_output_tokens: int | None = 0
    auxiliary_usage: dict | None = None
    online_token_budget_limit: int | None = None
    online_token_budget_verified: bool | None = None
    answer_phase_window: dict | None = None
    answer_native_quote_work: dict | None = None
    conclusion_source_manifest: tuple[dict, ...] = ()


class Controller:
    def __init__(self, store: SessionStore, engine: Engine, *, template=None,
                 policy: CandidatePolicy | None = None, knobs: PlannerKnobs | None = None,
                 cost_model: CostModel | None = None, events: EventLog | None = None,
                 epoch_retirement: bool = True,
                 online_calibration: bool = False, extend_open_needs=None, prefix_settings=None,
                 rolling_policy=None, state_observer=None, request_compiler=None,
                 dialogue_mode="contextkv", dialogue_history_tokens=8192):
        self.store, self.engine = store, engine
        if dialogue_mode not in {"contextkv", "llamaindex"}:
            raise ValueError("dialogue_mode must be contextkv or llamaindex")
        self.dialogue_mode, self.dialogue_history_tokens = dialogue_mode, dialogue_history_tokens
        self.template = template or ByteTemplate()
        self.policy = policy or CandidatePolicy()
        self.knobs = knobs or PlannerKnobs()
        self.cost_model = cost_model or CostModel(max_new_tokens=self.policy.max_new_tokens)
        self.events = events or EventLog()
        self._retire_lock = asyncio.Lock()
        self.epoch_retirement = epoch_retirement
        self.online_calibration = online_calibration
        self.extend_open_needs = extend_open_needs
        self.rolling_policy = rolling_policy
        self.state_observer = state_observer
        self.request_compiler = request_compiler or RequestCompiler(self.template)
        self.observe_prefix = None
        if prefix_settings is not None and prefix_settings["mode"] != "native":
            from .prefix_retention import PrefixReadObserver
            self.observe_prefix = PrefixReadObserver(store, engine, prefix_settings)
        self._active_generations = 0
        self._postplan_control_ms = {action: deque(maxlen=32) for action in ("APPEND", "REBUILD")}
        self._calibration_identity = None
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._retire_task: asyncio.Task | None = None
        self._retire_requested = False

    def _lock(self, session_id: str):
        # Holders and waiting coroutines keep the lock alive. Idle sessions do
        # not need an in-memory lock; their state remains in the durable store.
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def bind_resident_models(self, manifest):
        """Bind actual loaded retrieval models before enabling resource pricing."""
        if self.rolling_policy is not None:
            self.rolling_policy.bind_resident_models(manifest)
        if self.state_observer is not None:
            self.state_observer.bind_resident_models(manifest)

    async def _flush_retirements(self, session_id: str | None = None) -> int:
        acknowledged = 0
        async with self._retire_lock:
            for op in self.store.outbox(session_id):
                self.events.emit("epoch_retire_requested", operation=op)
                try:
                    ack = await self.engine.retire(op.engine_session_id, op.expected_generation,
                                                   op.operation_id, op.engine_restart_id, op.reason)
                    if (ack.engine_session_id, ack.session_generation, ack.engine_restart_id, ack.operation_id) != (
                            op.engine_session_id, op.expected_generation, op.engine_restart_id, op.operation_id):
                        raise EngineError("RETIRE_ACK_MISMATCH")
                    if not ack.acknowledged or ack.state != "RETIRED":
                        raise EngineError("RETIRE_PENDING")
                    self.store.ack_retire(op.operation_id)
                    self.events.emit("epoch_retire_acked", operation=op, acknowledgement=ack,
                                     physical_free_tokens=0)
                    acknowledged += 1
                except EngineError as exc:
                    if exc.code == "STALE_RESTART":
                        self.store.ack_retire(op.operation_id)
                        self.events.emit("epoch_retire_acked", operation=op, reason="engine_restart_invalidated_reference", physical_free_tokens=0)
                        acknowledged += 1
                    elif isinstance(exc, UnknownOutcomeError) or exc.code == "RETIRE_PENDING":
                        self.store.retire_failed(op.operation_id, str(exc))
                        self.events.emit("epoch_retire_retry", operation=op, error=str(exc))
                    else:
                        raise
        return acknowledged

    def _retire_background(self):
        self._retire_requested = True
        if self._retire_task is not None:
            if (not self._retire_task.done() or self._retire_task.cancelled()
                    or self._retire_task.exception() is not None):
                return
        self._retire_task = asyncio.create_task(self._flush_retire_requests())
        self._retire_task.add_done_callback(self._retire_finished)

    def _retire_finished(self, task):
        if not task.cancelled() and (error := task.exception()) is not None:
            self.events.emit("epoch_retire_worker_failed", error=f"{type(error).__name__}: {error}")

    async def _flush_retire_requests(self):
        # Coalesce turn notifications while one retirement is slow. A new
        # commit during flush still triggers a fresh durable-outbox snapshot.
        while self._retire_requested:
            self._retire_requested = False
            await self._flush_retirements()

    @property
    def retirement_error(self) -> str | None:
        """A failed maintenance task must not look healthy before its next drain."""
        task = self._retire_task
        if task is None or not task.done():
            return None
        if task.cancelled():
            return "retirement task cancelled"
        error = task.exception()
        return f"{type(error).__name__}: {error}" if error is not None else None

    async def drain(self):
        if self._retire_task is not None:
            task = self._retire_task
            try:
                await task
            finally:
                if task.done() and self._retire_task is task:
                    self._retire_task = None
        await self._flush_retirements()

    async def start(self):
        """Start bound pricing before admitting or timestamping requests.

        External-arrival callers await this after model binding and before
        dispatch. PhysicalPaidCost owns the existing lock and idempotence.
        """
        started = time.monotonic()
        if self.rolling_policy is not None:
            await self.rolling_policy.start()
        if self.state_observer is not None:
            await self.state_observer.start()
        return (time.monotonic()-started)*1000

    async def aclose(self):
        try:
            await self.drain()
        finally:
            try:
                if self.rolling_policy is not None:
                    await self.rolling_policy.aclose()
                if self.state_observer is not None:
                    await self.state_observer.aclose()
            finally:
                await self.engine.aclose()

    async def _prepare_epoch(self, attempt: Attempt, info: EngineInfo) -> PrepareAck:
        epoch, operation_id = attempt.epoch, f"prepare:{attempt.attempt_id}"
        ack = await self.engine.prepare(epoch.engine_session_id, operation_id, epoch.engine_restart_id,
            **({"business_session_id": attempt.session_id}
               if info.business_session_identity_supported else {}))
        if ((ack.engine_session_id, ack.engine_restart_id, ack.operation_id)
                != (epoch.engine_session_id, epoch.engine_restart_id, operation_id)
                or ack.session_generation <= 0
                or (info.business_session_identity_supported
                    and ack.business_session_id != attempt.session_id)):
            raise EngineError("PREPARE_ACK_MISMATCH")
        return ack

    def _validate(self, attempt: Attempt, result: GenerationResult):
        candidate, epoch = attempt.plan.candidate, attempt.epoch
        receipt = result.receipt
        if not result.generation_complete or not receipt.generation_complete:
            raise EngineError("INCOMPLETE_GENERATION")
        if result.finish_reason not in ("stop", "length", "eos", "max_tokens"):
            raise EngineError("INVALID_FINISH_REASON", result.finish_reason)
        physical_id = (f"{attempt.attempt_id}:{attempt.final_step_id}" if attempt.final_step_id
                       else attempt.attempt_id)
        if (receipt.rid, receipt.attempt_id, receipt.engine_session_id, receipt.session_generation,
                receipt.engine_restart_id, receipt.input_hash, receipt.finish_reason) != (
                physical_id, physical_id, epoch.engine_session_id, epoch.session_generation,
                epoch.engine_restart_id, candidate.input_hash, result.finish_reason):
            raise EngineError("RECEIPT_IDENTITY_MISMATCH")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in result.output_ids):
            raise EngineError("INVALID_OUTPUT_IDS")
        if len(result.output_ids) > (candidate.output_budget or self.policy.max_new_tokens):
            raise EngineError("OUTPUT_BUDGET_EXCEEDED")
        if attempt.final_step_id is None and not attempt.cache_config_hash:
            raise EngineError("MISSING_ANSWER_CACHE_SCOPE")

    def _observe_answer_source(self, attempt: Attempt):
        # The upstream dialogue final step was observed by ReadingExecutor.
        if attempt.final_step_id is not None:
            return
        source_view = self.request_compiler.observe_executed(attempt.plan.candidate, attempt.result.receipt,
            cache_config_hash=attempt.cache_config_hash, engine_restart_id=attempt.epoch.engine_restart_id)
        self.events.emit("answer_source_executed", attempt_id=attempt.attempt_id, source_view=source_view)

    def _response(self, attempt: Attempt, *, elapsed_ms=None, replay=False,
                  timing: ClientTiming | None = None, control_ms=None, answer_phase_window=None) -> TurnResponse:
        result, plan, epoch = attempt.result, attempt.plan, attempt.epoch
        receipt = result.receipt
        prediction = next((e for e in plan.predicted if e.candidate_id == plan.candidate.candidate_id), None)
        ttft = (timing.t_first-timing.t_arrive)*1000 if timing and timing.t_first is not None else None
        all_steps = attempt.reading_outcome.steps if attempt.reading_outcome else ()
        steps = tuple(step for step in all_steps if step.step_id != attempt.final_step_id)
        metrics = tuple({"step_id": step.step_id, "kind": step.task.kind,
                         "input_tokens": len(step.input_ids), "output_tokens": len(step.output_ids),
                         **step.metrics} for step in all_steps)
        internal_first = next((step.metrics["first_token_from_arrival_ms"] for step in steps
                               if step.metrics.get("first_token_from_arrival_ms") is not None), None)
        reader_input = len(plan.candidate.input_ids)+sum(len(step.input_ids) for step in steps)
        reader_output = len(result.output_ids)+sum(len(step.output_ids) for step in steps)
        auxiliary_input = attempt.auxiliary_usage["input_tokens"] if attempt.auxiliary_usage is not None else 0
        auxiliary_output = attempt.auxiliary_usage["output_tokens"] if attempt.auxiliary_usage is not None else 0
        total_input = reader_input+auxiliary_input if auxiliary_input is not None else None
        total_output = reader_output+auxiliary_output if auxiliary_output is not None else None
        token_limit = attempt.context_plan.budget.max_total_tokens if attempt.context_plan else None
        budget_verified = (total_input+total_output <= token_limit
            if token_limit is not None and total_input is not None and total_output is not None else None)
        sources = conclusion_source_manifest(attempt.reading_outcome, plan.candidate.reference_map)
        if sources:
            urls = {block.key: block.source_uri for block in (
                *attempt.reading_outcome.plan.selected_set.source_blocks, *attempt.evidence)}
            sources = tuple({**row, "source_uri": urls.get(row["key"])} for row in sources)
        return TurnResponse(attempt.session_id, attempt.request_id, attempt.attempt_id, result.text,
                            plan.candidate.action, plan.reason, epoch.epoch_no, attempt.expected_version + 1,
                            result.output_ids, len(plan.candidate.input_ids), len(result.output_ids),
                            receipt.actual_device_hit_tokens, receipt.actual_host_loaded_tokens,
                            receipt.actual_prefill_tokens, elapsed_ms, ttft, replay,
                            timing, receipt, prediction, control_ms, result.elapsed_ms,
                            result.server_ttft_ms, result.tpot_ms, result.timing_source,
                            result.finish_reason in {"length", "max_tokens"}, attempt.branch_reason,
                            retrieval_ms=attempt.retrieval_ms,
                            visible_evidence_manifest=plan.candidate.evidence_spans,
                            citation_scope=tuple(ref for ref,key in plan.candidate.reference_map
                                                 if key in plan.candidate.required_blocks),
                            read_count=sum(step.task.kind == "READ" for step in steps),
                            reduce_count=sum(step.task.kind == "REDUCE" for step in steps),
                            total_input_tokens=total_input,
                            total_output_tokens=total_output,
                            auxiliary_model_ms=(attempt.auxiliary_usage["compute_ms"] if attempt.auxiliary_usage is not None
                                else sum(value for _, value in attempt.context_plan.auxiliary_ms)
                                if attempt.context_plan else 0),
                            internal_first_token_ms=internal_first, step_metrics=metrics,
                            reading_ms=attempt.reading_elapsed_ms,
                            total_device_hit_tokens=receipt.actual_device_hit_tokens+sum(
                                step.metrics.get("actual_device_hit_tokens", 0) for step in steps),
                            total_host_loaded_tokens=receipt.actual_host_loaded_tokens+sum(
                                step.metrics.get("actual_host_loaded_tokens", 0) for step in steps),
                            total_prefill_tokens=receipt.actual_prefill_tokens+sum(
                                step.metrics.get("actual_prefill_tokens", 0) for step in steps),
                            reader_input_tokens=reader_input, reader_output_tokens=reader_output,
                            auxiliary_input_tokens=auxiliary_input, auxiliary_output_tokens=auxiliary_output,
                            auxiliary_usage=attempt.auxiliary_usage,
                            online_token_budget_limit=token_limit, online_token_budget_verified=budget_verified,
                            answer_phase_window=answer_phase_window,
                            answer_native_quote_work=quote_realization(attempt.answer_quote, receipt),
                            conclusion_source_manifest=sources)

    async def _retrieve_with_cost(self, attempt_id, phase, provider, *args):
        started = time.monotonic()
        with capture_auxiliary_spend(self.store, attempt_id, phase) as observation:
            try:
                selection = await provider(*args)
                observation["provider_usage"] = selection.auxiliary_usage
            except BaseException:
                current = self.store.get_attempt(attempt_id)
                usage = reconcile_auxiliary_usage(observation)
                if phase == "gap" and current.auxiliary_usage is not None:
                    usage = merge_auxiliary_usage(current.auxiliary_usage, usage)
                elapsed = (time.monotonic()-started)*1000
                self.store.save_retrieval(attempt_id, evidence=current.evidence,
                    retrieval_priority=current.retrieval_priority, query_context=current.query_context,
                    retrieval_ms=elapsed if phase == "retrieval" else current.retrieval_ms,
                    auxiliary_usage=usage)
                self.events.emit("retrieval_failed", attempt_id=attempt_id, phase=phase, retrieval_ms=elapsed)
                raise
        if observation["calls"]:
            selection = replace(selection, auxiliary_usage=reconcile_auxiliary_usage(
                observation, selection.auxiliary_usage))
        return selection

    async def _serve_dialogue_attempt(self, attempt, snapshot, query, evidence, provider, budget,
                                      arrived, migrating):
        """Run upstream dialogue inside the current durable business attempt."""
        from adapters.llamaindex_dialogue import run_dialogue
        from .execution import ReadingExecutor, auxiliary_tokens, require_online_tokens
        from .templates import SYSTEM_PROMPT

        info = await self.engine.info()
        if info.fingerprint != self.template.fingerprint:
            raise EngineError("FINGERPRINT_MISMATCH")
        budget = budget or ContextBudget(window_tokens=self.policy.max_total_model_tokens,
            output_tokens=self.policy.max_new_tokens, history_tokens=self.dialogue_history_tokens)
        if (budget.window_tokens, budget.output_tokens) != (self.policy.max_total_model_tokens,
                                                           self.policy.max_new_tokens):
            raise ValueError("dialogue and controller must use the same model allocation")
        plan = ContextPlan("llamaindex:"+attempt.attempt_id,
            TaskState(attempt.request_id, query, current_turn_id=len(snapshot.raw_history)+1, query=query),
            EvidenceSet("llamaindex", (), source_blocks=evidence), budget=budget,
            reading_mode="llamaindex", cost_prediction_eligible=False)
        self.store.save_context_plan(attempt.attempt_id, plan)
        executor = ReadingExecutor(self.store, self.engine, self.template,
            events=self.events, request_arrived=arrived, request_compiler=self.request_compiler)
        retrieval_ms = attempt.retrieval_ms or 0.0
        auxiliary_usage = attempt.auxiliary_usage
        records, candidates = [], []
        seen_sources = set()
        started = time.monotonic()

        async def retrieve(condensed_query):
            nonlocal evidence, retrieval_ms, auxiliary_usage, attempt
            if attempt.query_context is not None:
                if attempt.query_context.query != condensed_query:
                    raise ValueError("recovered condensation differs from the paid retrieval")
                evidence = attempt.evidence
                return evidence
            if provider is not None:
                tick = time.monotonic()
                selection = await self._retrieve_with_cost(attempt.attempt_id, "retrieval", provider,
                                                           snapshot, condensed_query)
                retrieval_ms += (time.monotonic()-tick)*1000
                evidence = tuple(selection.evidence)
                auxiliary_usage = (selection.auxiliary_usage if selection.auxiliary_usage is not None else
                    {"input_tokens": None, "output_tokens": None, "compute_ms": None, "models": []})
            attempt = self.store.save_retrieval(attempt.attempt_id, evidence=evidence,
                retrieval_priority=tuple(block.key for block in evidence),
                query_context=QueryContext(condensed_query, "llamaindex_condense"),
                retrieval_ms=retrieval_ms, auxiliary_usage=auxiliary_usage)
            return evidence

        async def complete(call):
            if len(records) >= budget.max_steps:
                raise ContextOverflow("LlamaIndex exhausted the common request step allocation")
            if budget.max_complete_ms is not None and (time.monotonic()-arrived)*1000 > budget.max_complete_ms:
                raise ContextOverflow("LlamaIndex exhausted the common complete-time allocation")
            rendered = self.template.render(call.messages, with_offsets=bool(call.evidence_spans))
            current_sources = frozenset(span.key for span in call.evidence_spans)
            seen_sources.update(current_sources)
            candidate = PromptCandidate(call.step_id, "REBUILD", snapshot.version,
                call.messages, rendered.input_ids, rendered.input_hash, current_sources,
                frozenset(seen_sources), evidence_spans=call.evidence_spans,
                reference_map=call.reference_map, corpus_snapshot_id=snapshot.corpus_snapshot_id,
                fingerprint=self.template.fingerprint, output_budget=call.output_budget,
                generation_kind="LLAMAINDEX_"+call.stage.upper(), cost_prediction_eligible=False,
                layout_id="llamaindex_native")
            candidate = self.request_compiler.bind_geometry(candidate, rendered)
            spent = sum(len(record.input_ids)+len(record.output_ids) for record in records)
            auxiliary = 0 if auxiliary_usage is None else auxiliary_tokens(auxiliary_usage)
            require_online_tokens(budget.max_total_tokens,
                spent+auxiliary if auxiliary is not None else None,
                len(rendered.input_ids)+call.output_budget, candidate.generation_kind)
            task = ReadTask(call.step_id, candidate.generation_kind,
                input_budget=budget.window_tokens-call.output_budget, output_budget=call.output_budget)
            record = await executor._generate_step(attempt.attempt_id, snapshot, plan, task, (), (),
                candidate=candidate, text_callback=True)
            records.append(record)
            candidates.append(candidate)
            return record.result.raw_output

        dialogue = await run_dialogue(query, snapshot.raw_history, retrieve_callback=retrieve,
            complete_callback=complete, tokenizer=self.template.tokenizer,
            context_window=budget.window_tokens, output_tokens=budget.output_tokens,
            history_tokens=budget.history_tokens,
            system_prompt=self.policy.system_prompt or SYSTEM_PROMPT)
        if not records or dialogue.calls[-1].stage == "condense":
            raise ContextOverflow("LlamaIndex returned no physical answer call")
        final, candidate = records[-1], candidates[-1]
        generation = self.store.execution_step(attempt.attempt_id, final.step_id)[2]
        if dialogue.answer != generation.text:
            raise ValueError("upstream final answer differs from the paid physical output")
        plan = replace(plan, selected_set=replace(plan.selected_set, source_blocks=dialogue.source_blocks),
            steps=tuple(record.task for record in records))
        outcome = ReadingOutcome(plan, tuple(record.result for record in records[:-1]), tuple(records[:-1]))
        previous_epoch = snapshot.current_epoch
        epoch = replace(final.engine_epoch, epoch_no=previous_epoch.epoch_no+1 if previous_epoch else 1,
            parent_epoch_no=previous_epoch.epoch_no if previous_epoch else None,
            reason="corpus_snapshot_change" if migrating else "llamaindex_native")
        attempt = replace(self.store.get_attempt(attempt.attempt_id), stage="RESULT_READY",
            evidence=dialogue.source_blocks, plan=Plan(candidate, reason="llamaindex_native"),
            epoch=epoch, result=generation, new_epoch=True, final_step_id=final.step_id,
            branch_reason=epoch.reason, context_plan=plan, reading_outcome=outcome,
            reading_elapsed_ms=sum(record.metrics.get("step_wall_ms", 0.0) or 0.0
                                   for record in records[:-1]),
            retrieval_ms=retrieval_ms, auxiliary_usage=auxiliary_usage)
        attempt = self.store.promote_completed_step(attempt)
        self._validate(attempt, generation)
        snapshot = self.store.commit_turn_and_outbox(attempt.attempt_id)
        committed_at = time.monotonic()
        answer_window = {"start": started, "end": committed_at, "commit_at": committed_at,
            "clock": "client_monotonic", "scope": "answer_until_commit",
            "start_definition": "external_dialogue_start", "end_definition": "external_qa_commit"}
        try:
            self.events.emit("ledger_committed", business_session_id=attempt.session_id,
                attempt_id=attempt.attempt_id, ledger_version=snapshot.version, epoch=epoch.epoch_no,
                answer_phase_window=answer_window)
            self.events.emit("llamaindex_dialogue_committed", business_session_id=attempt.session_id,
                attempt_id=attempt.attempt_id, ledger_version=snapshot.version,
                framework_version=dialogue.framework_version, condensed_query=dialogue.condensed_query,
                call_stages=tuple(call.stage for call in dialogue.calls))
        except Exception as error:
            logging.getLogger(__name__).error("Diagnostic output failed after commit: %s", error)
        self._retire_background()
        submitted = arrived+final.metrics["submitted_from_arrival_ms"]/1000
        timing = ClientTiming(arrived, submitted,
            submitted+generation.ttft_ms/1000 if generation.ttft_ms is not None else None,
            submitted+generation.last_token_ms/1000 if generation.last_token_ms is not None else None,
            committed_at, committed_at, token_observation="chunk" if generation.ttft_ms is not None else "none")
        return self._response(attempt, elapsed_ms=(committed_at-arrived)*1000,
            timing=timing, answer_phase_window=answer_window)

    async def serve_turn(self, session_id: str, query: str, evidence: tuple[EvidenceBlock, ...] = (), *,
                         request_id: str | None = None, corpus_snapshot_id: str = "demo-v1",
                         reference_history: tuple[TurnRecord, ...] | None = None,
                         retrieval_priority: tuple[EvidenceKey, ...] | None = None,
                         query_context: QueryContext | None = None,
                         context_plans: tuple[ContextPlan, ...] = (),
                         open_need_provider: Callable | None = None,
                         auxiliary_usage: dict | None = None,
                         evidence_provider: Callable[..., Awaitable[EvidenceSelection]] | None = None,
                         source_request_identity: tuple[str, str] | None = None,
                         extend_for_plan: Callable | None = None,
                         from_corpus_snapshot_id: str | None = None,
                         request_arrived: float | None = None,
                         dialogue_budget: ContextBudget | None = None) -> TurnResponse:
        if not session_id or not query.strip():
            raise ValueError("session_id and query must be nonempty")
        observation_startup_ms = await self.start()
        arrived = time.monotonic() if request_arrived is None else request_arrived
        evidence = tuple(evidence)
        context_plans = tuple(context_plans)
        if evidence_provider is not None and (evidence or reference_history is not None or retrieval_priority is not None
                                             or query_context is not None):
            raise ValueError("retrieved queries cannot also supply evidence, reference history or retrieval priority")
        if from_corpus_snapshot_id is not None and (evidence_provider is None or not from_corpus_snapshot_id.strip()):
            raise ValueError("a source snapshot transition requires a retrieved query and nonempty prior snapshot")
        request_id = request_id or uuid.uuid4().hex
        # Query identity is stable before retrieval. A retry returns the original
        # attempt rather than re-retrieving against newer history or rankings.
        content = (("retrieved_query", query, corpus_snapshot_id) if evidence_provider is not None else
                   (query, evidence, corpus_snapshot_id, reference_history, retrieval_priority))
        if source_request_identity is not None:
            content = (*content, source_request_identity)
        if self.dialogue_mode != "contextkv":
            content = (*content, self.dialogue_mode)
        if query_context is not None:
            content = (*content, query_context)
        if context_plans:
            content = (*content, context_plans)
        request_hash = hashlib.sha256(dumps(content).encode()).hexdigest()
        async with self._lock(session_id):
            entered = time.monotonic()
            self.events.emit("turn_received", business_session_id=session_id, request_id=request_id)
            previous = self.store.find_request(session_id, request_id)
            resume_context_plan = None
            if previous:
                matches = previous.request_hash == request_hash
                if not matches and evidence_provider is not None and source_request_identity is None:
                    # A committed retry belongs to the snapshot it originally
                    # read, even when this worker now serves a newer index.
                    original_snapshot = (previous.plan.candidate.corpus_snapshot_id if previous.plan is not None
                                         else self.store.get_snapshot(session_id).corpus_snapshot_id)
                    original = ("retrieved_query", query, original_snapshot)
                    if self.dialogue_mode != "contextkv":
                        original = (*original, self.dialogue_mode)
                    matches = previous.request_hash == hashlib.sha256(dumps(original).encode()).hexdigest()
                if not matches:
                    raise IdempotencyConflict("request_id reused with different content")
                if previous.stage == "COMMITTED":
                    return self._response(previous, replay=True)
                if previous.stage == "FAILED":
                    raise StoreError("Previous attempt failed; use a new request_id for an explicit retry")
                if previous.stage == "STARTED":
                    resume_context_plan = self.store.context_plan(previous.attempt_id)
            if self.store.pending_attempt(session_id) and resume_context_plan is None:
                await self._recover(session_id)
                previous = self.store.find_request(session_id, request_id)
                if previous and previous.stage == "COMMITTED":
                    return self._response(previous, replay=True)
                if previous and previous.stage == "FAILED":
                    raise StoreError("Recovered terminal failure; use a new request_id")
            snap = (self.store.get_snapshot(session_id) if from_corpus_snapshot_id is not None
                    else self.store.read_snapshot(session_id, corpus_snapshot_id))
            if from_corpus_snapshot_id is not None and from_corpus_snapshot_id != snap.corpus_snapshot_id:
                raise VersionConflict("Session corpus snapshot changed before the requested transition")
            migrating = snap.corpus_snapshot_id != corpus_snapshot_id
            if migrating and from_corpus_snapshot_id is None:
                raise ValueError("Session corpus snapshot differs; explicitly provide its current snapshot to transition")
            if reference_history is not None:
                snap = self.store.sync_history(session_id, reference_history, snap.version)
                self.events.emit("reference_history_sync", business_session_id=session_id,
                                 raw_history_version=snap.raw_history_version, pending=snap.history_sync_pending)
            working_snap = (replace(snap, corpus_snapshot_id=corpus_snapshot_id, must_rebuild=True)
                            if migrating else snap)
            attempt = previous if resume_context_plan is not None else self.store.begin_attempt(
                session_id, request_id, request_hash, query, snap.version)
            if attempt.stage == "COMMITTED":
                return self._response(attempt, replay=True)
            if attempt.stage == "FAILED":
                raise StoreError("Previous attempt failed; use a new request_id")
            retrieval_ms = None
            retrieval_segment_ms = 0.0
            reading_elapsed_ms = 0.0
            reading_segment_ms = 0.0
            reading_outcome = None
            initial_outcomes = ()
            try:
                if self.dialogue_mode == "llamaindex":
                    return await self._serve_dialogue_attempt(attempt, working_snap, query, evidence,
                        evidence_provider, dialogue_budget, arrived, migrating)
                if resume_context_plan is not None:
                    context_plans = (resume_context_plan,)
                    evidence = resume_context_plan.selected_set.source_blocks
                    query_context = QueryContext(resume_context_plan.task_state.query or query,
                        "task_state", task_state=resume_context_plan.task_state)
                    retrieval_ms = attempt.retrieval_ms
                    auxiliary_usage = attempt.auxiliary_usage
                    initial_outcomes = ((attempt.reading_outcome,) if attempt.reading_outcome is not None else ())
                elif evidence_provider is not None:
                    retrieval_started = time.monotonic()
                    selection = await self._retrieve_with_cost(attempt.attempt_id, "retrieval",
                                                               evidence_provider, working_snap)
                    retrieval_ms = (time.monotonic()-retrieval_started)*1000
                    retrieval_segment_ms = retrieval_ms
                    evidence = tuple(selection.evidence)
                    retrieval_priority = tuple(selection.retrieval_priority)
                    query_context = selection.query_context
                    context_plans = selection.context_plans
                    initial_outcomes = selection.initial_outcomes
                    open_need_provider = selection.open_need_provider
                    auxiliary_usage = (selection.auxiliary_usage if selection.auxiliary_usage is not None
                        else {"input_tokens": None, "output_tokens": None, "compute_ms": None, "models": []})
                    attempt = self.store.save_retrieval(attempt.attempt_id, evidence=evidence,
                        retrieval_priority=retrieval_priority, query_context=query_context,
                        retrieval_ms=retrieval_ms, auxiliary_usage=auxiliary_usage)
                elif auxiliary_usage is not None:
                    attempt = self.store.save_retrieval(attempt.attempt_id, evidence=evidence,
                        retrieval_priority=retrieval_priority, query_context=query_context,
                        retrieval_ms=retrieval_ms, auxiliary_usage=auxiliary_usage)
                info = await self.engine.info()
                if info.fingerprint != self.template.fingerprint:
                    raise EngineError("FINGERPRINT_MISMATCH")
                identity = (info.engine_restart_id, info.cache_config_hash)
                if (identity != self._calibration_identity
                        and (self.cost_model.prefill_points or self.cost_model.decode_points
                             or self.cost_model.step_models)):
                    scope = self.cost_model.check_calibration_scope(info)
                    self.events.emit("cost_calibration_scope", **scope)
                    self._calibration_identity = identity
                recovery = snap.current_epoch is not None and snap.current_epoch.engine_restart_id != info.engine_restart_id
                planning_snap = (replace(working_snap, force_new_epoch_for_recovery=True, must_rebuild=True)
                                 if recovery else working_snap)
                if resume_context_plan is not None and open_need_provider is None and extend_for_plan is not None:
                    open_need_provider = partial(extend_for_plan, planning_snap)
                gap_provider = open_need_provider or self.extend_open_needs
                if gap_provider is not None:
                    gap_provider = partial(self._retrieve_with_cost, attempt.attempt_id, "gap", gap_provider)
                if context_plans:
                    from .context_budget import bind_answer_window
                    recorded_ids = tuple(dict.fromkeys((*(record.step_id for record in
                        self.store.execution_steps(attempt.attempt_id)), *(result.step_id
                        for outcome in initial_outcomes for result in outcome.results))))
                    context_plans = tuple(bind_answer_window(context_plan, self.template, planning_snap,
                        self.policy.system_prompt, preserve_step_ids=recorded_ids)
                        for context_plan in context_plans)
                    bound = {context_plan.plan_id: context_plan for context_plan in context_plans}
                    initial_outcomes = tuple(replace(outcome, plan=bound[outcome.plan.plan_id])
                                             for outcome in initial_outcomes)
                if context_plans and self.knobs.strategy == "cost" and self.rolling_policy is None:
                    from .context_cost import quote_read_costs
                    completed_steps = tuple(record for stage, record in
                        self.store.execution_steps_with_stage(attempt.attempt_id) if stage == "COMPLETE")
                    context_plans, read_cost_trace = await quote_read_costs(context_plans,
                        planning_snap, self.template, self.engine, self.cost_model, completed_steps=completed_steps,
                        initial_outcomes=initial_outcomes, request_compiler=self.request_compiler)
                    self.events.emit("reading_plan_cost_quotes", attempt_id=attempt.attempt_id,
                                     trace=read_cost_trace)
                measure_redundancy = self.knobs.strategy == "redundancy_threshold"
                def compile_candidates(snapshot, plans, outcome=None, **compile_options):
                    if plans:
                        return tuple(candidate for context_plan in plans for candidate in
                            self.request_compiler.compile_answers(snapshot, query, context_plan,
                                policy=self.policy, outcome=outcome or next((seed for seed in initial_outcomes
                                    if seed.plan.plan_id == context_plan.plan_id), None), cache_scope=info,
                                **compile_options))
                    return self.request_compiler.compile_answers(snapshot, query, retrieved=evidence,
                        policy=self.policy, cache_scope=info,
                        retrieval_priority=retrieval_priority, query_context=query_context,
                        measure_evidence_redundancy=measure_redundancy)

                def fixed_source_order(candidates, quotes):
                    if self.request_compiler.index is None or self.knobs.strategy == "cost":
                        return candidates
                    groups = {}
                    for candidate in candidates:
                        if candidate.feasible:
                            key = (candidate.context_plan_id, candidate.action, candidate.working_history_id,
                                   candidate.layout_id.removesuffix(":contextpilot"))
                            groups.setdefault(key, []).append(candidate)
                    return tuple(self.request_compiler.choose_variant(group, quotes) for group in groups.values())

                rolling_result = None
                answer_phase_started = time.monotonic()
                if self.rolling_policy is not None:
                    if not context_plans:
                        raise ValueError("rolling plan policy requires explicit semantic ContextPlans")
                    from .execution import ReadingExecutor
                    executor = ReadingExecutor(self.store, self.engine, self.template,
                        events=self.events, request_arrived=arrived, observe_prefix=self.observe_prefix,
                        request_compiler=self.request_compiler)
                    reading_started = time.monotonic()
                    try:
                        rolling_result = await self.rolling_policy.run(attempt.attempt_id, planning_snap,
                            context_plans, executor, self.engine, self.template, self.cost_model,
                            compile_candidates, arrived=arrived, auxiliary_usage=auxiliary_usage, events=self.events,
                            open_need_provider=gap_provider,
                            initial_outcomes=initial_outcomes)
                    finally:
                        reading_segment_ms = (time.monotonic()-reading_started)*1000
                        reading_elapsed_ms = self.store.add_reading_elapsed(attempt.attempt_id, reading_segment_ms)
                    reading_outcome = rolling_result.outcome
                    context_plans = (reading_outcome.plan,)
                    answer_phase_started = rolling_result.answer_phase_started
                candidates = ((rolling_result.plan.candidate,) if rolling_result is not None else
                              unique_answer_inputs(compile_candidates(planning_snap, context_plans)))
                valid = tuple(c for c in candidates if c.feasible)
                if not valid:
                    raise ContextOverflow("shared_context_policy_overflow: required evidence and full QA exceed budget")
                needs_quote = (self.knobs.strategy in {"cost", "warm_idle"}
                               or self.request_compiler.index is not None) and len(valid) > 1
                quotes = (rolling_result.quotes if rolling_result is not None else
                          await self.engine.quote_batch(valid, self.template.fingerprint) if needs_quote else None)
                if rolling_result is None and quotes is not None and quotes.engine_restart_id != info.engine_restart_id:
                    # At most one re-quote after a restart; native admission remains authoritative.
                    info = await self.engine.info()
                    planning_snap = replace(working_snap, force_new_epoch_for_recovery=True, must_rebuild=True)
                    candidates = unique_answer_inputs(compile_candidates(planning_snap, context_plans))
                    valid = tuple(c for c in candidates if c.feasible)
                    quotes = await self.engine.quote_batch(valid, self.template.fingerprint)
                    if quotes.engine_restart_id != info.engine_restart_id:
                        raise EngineError("UNSTABLE_RESTART")
                if rolling_result is None and quotes is not None:
                    candidates = fixed_source_order(candidates, quotes)
                # The planner validates this batch once before selecting a candidate.
                pressure = (max(1, quotes.active_requests + (quotes.queued_requests or 0) + 1) if quotes is not None
                            and quotes.active_requests is not None else max(1, self._active_generations + 1))
                spent_control_ms = (time.monotonic()-entered)*1000 - retrieval_segment_ms
                observed = [value for values in self._postplan_control_ms.values() for value in values]
                default_postplan = statistics.mean(observed) if observed else 0.0
                # Preparation/serialization costs differ between retaining and
                # replacing an epoch. Charge prior completed observations to
                # their action instead of cancelling one common mean in A/R.
                planning_control_ms = {action: spent_control_ms + (
                    statistics.mean(values) if values else default_postplan)
                    for action, values in self._postplan_control_ms.items()}
                plan_started = time.monotonic()
                plan = (rolling_result.plan if rolling_result is not None else
                        choose_plan(candidates, quotes, self.cost_model, planning_snap, self.knobs,
                                   concurrency=pressure, control_ms=planning_control_ms))
                if isinstance(plan, NoFeasiblePlan):
                    raise ContextOverflow(plan.reason)
                selected_context_plan = None
                if context_plans:
                    from .execution import ReadingExecutor, require_complete_time, require_online_tokens
                    selected_context_plan = next(context_plan for context_plan in context_plans
                        if context_plan.plan_id == plan.candidate.context_plan_id)
                    if rolling_result is None:
                        from .context_budget import direct_answer_reason
                        seed = next((outcome for outcome in initial_outcomes
                                     if outcome.plan.plan_id == selected_context_plan.plan_id),
                                    ReadingOutcome(selected_context_plan))
                        seed = replace(seed, plan=selected_context_plan)
                        self.store.save_context_plan(attempt.attempt_id, selected_context_plan,
                                                     initial_outcome=seed)
                        reading_started = time.monotonic()
                        async def observe_fixed_state(state, expanded):
                            await self.state_observer.observe(attempt.attempt_id, executor, state, planning_snap,
                                self.engine, self.cost_model, compile_candidates, arrived=arrived,
                                auxiliary_usage=auxiliary_usage, expanded=expanded, startup_ms=observation_startup_ms)
                        executor = ReadingExecutor(self.store, self.engine, self.template,
                            extend_open_needs=gap_provider, events=self.events,
                            request_arrived=arrived, observe_prefix=self.observe_prefix,
                            request_compiler=self.request_compiler,
                            observe_state=observe_fixed_state if self.state_observer is not None else None)
                        try:
                            if direct_answer_reason(seed, plan.candidate) is None:
                                from .execution import auxiliary_tokens
                                spent = (auxiliary_tokens(auxiliary_usage) if auxiliary_usage is not None else 0)
                                reading_outcome = replace(seed, plan=replace(selected_context_plan,
                                    online_tokens_spent=spent), auxiliary_usage={"input_tokens": 0,
                                        "output_tokens": 0, "compute_ms": 0.0, "models": []})
                                if self.state_observer is not None:
                                    await observe_fixed_state(reading_outcome, set())
                            else:
                                reading_outcome = await executor.execute_reads(attempt.attempt_id, planning_snap,
                                    selected_context_plan, initial_auxiliary_usage=auxiliary_usage,
                                    initial_outcome=seed)
                        finally:
                            reading_segment_ms = (time.monotonic()-reading_started)*1000
                            reading_elapsed_ms = self.store.add_reading_elapsed(attempt.attempt_id,
                                reading_segment_ms)
                    gap_auxiliary_ms = sum(value for name, value in reading_outcome.plan.auxiliary_ms
                                           if name.startswith("gap:"))
                    selected_context_plan = replace(reading_outcome.plan,
                        predicted_cost_ms=reading_elapsed_ms-gap_auxiliary_ms)
                    reading_outcome = replace(reading_outcome, plan=selected_context_plan)
                    auxiliary_usage = merge_auxiliary_usage(auxiliary_usage if auxiliary_usage is not None
                        else {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []},
                        reading_outcome.auxiliary_usage)
                    self.store.save_context_plan(attempt.attempt_id, selected_context_plan)
                    info = await self.engine.info()
                    if rolling_result is not None and quotes.engine_restart_id != info.engine_restart_id:
                        raise EngineError("ROLLING_QUOTE_RESTART_CHANGED")
                    if (planning_snap.current_epoch is not None
                            and planning_snap.current_epoch.engine_restart_id != info.engine_restart_id):
                        planning_snap = replace(planning_snap, force_new_epoch_for_recovery=True,
                                                must_rebuild=True)
                    if rolling_result is None:
                        answer_phase_started = time.monotonic()
                        candidates = unique_answer_inputs(compile_candidates(
                            planning_snap, (selected_context_plan,), reading_outcome))
                        valid = tuple(candidate for candidate in candidates if candidate.feasible)
                        if not valid:
                            raise ContextOverflow("ANSWER plan exceeds the context allocation")
                        quotes = await self.engine.quote_batch(valid, self.template.fingerprint)
                        if quotes.engine_restart_id != info.engine_restart_id:
                            raise EngineError("UNSTABLE_RESTART")
                        candidates = fixed_source_order(candidates, quotes)
                        pressure = (max(1, quotes.active_requests+(quotes.queued_requests or 0)+1)
                            if quotes.active_requests is not None else max(1, self._active_generations+1))
                        plan = choose_plan(candidates, quotes, self.cost_model, planning_snap, self.knobs,
                                           concurrency=pressure, control_ms=planning_control_ms)
                        if isinstance(plan, NoFeasiblePlan):
                            raise ContextOverflow(plan.reason)
                    require_online_tokens(selected_context_plan.budget.max_total_tokens,
                        selected_context_plan.online_tokens_spent,
                        len(plan.candidate.input_ids)+plan.candidate.output_budget, "ANSWER")
                    require_complete_time(selected_context_plan.budget.max_complete_ms, arrived, "ANSWER")
                    query_context = QueryContext(selected_context_plan.task_state.query or query,
                        "task_state", task_state=selected_context_plan.task_state)
                    selected_sources = {span.source_key for unit in selected_context_plan.selected_set.selected_units
                                        for span in unit.source_spans}
                    blocks = {block.key: block for block in (*planning_snap.archive, *evidence,
                        *selected_context_plan.selected_set.source_blocks)}
                    evidence = tuple(blocks[key] for key in sorted(selected_sources | plan.candidate.required_blocks))
                    points = tuple(replace(point, citation_ids=tuple(ref for ref, key in plan.candidate.reference_map
                        if key in plan.candidate.required_blocks
                        and any(span.source_key == key for span in point.source_spans)))
                        for point in selected_context_plan.answer_plan.points)
                    selected_context_plan = replace(selected_context_plan,
                        answer_plan=replace(selected_context_plan.answer_plan, points=points))
                    reading_outcome = replace(reading_outcome, plan=selected_context_plan)
                    self.store.save_context_plan(attempt.attempt_id, selected_context_plan)
                append_failure = next((c.reason for c in candidates if c.action == "APPEND" and not c.feasible), None)
                reason = ("corpus_snapshot_change" if migrating else
                          "reference_history_sync" if snap.history_sync_pending else
                          "recovery" if planning_snap.force_new_epoch_for_recovery else
                          "initial" if snap.current_epoch is None else
                          (append_failure or plan.reason) if plan.candidate.action == "REBUILD" else "append")
                planned = replace(attempt, stage="PLANNED", evidence=evidence,
                                  plan=plan, branch_reason=reason,
                                  retrieval_priority=retrieval_priority, retrieval_ms=retrieval_ms,
                                  query_context=query_context, context_plan=selected_context_plan,
                                  reading_outcome=reading_outcome, reading_elapsed_ms=reading_elapsed_ms,
                                  auxiliary_usage=auxiliary_usage, cache_config_hash=info.cache_config_hash)
                # No-C ablation keeps the native session across text rebuilds.
                # History protocol changes and failed attempts still need safe recovery.
                new_epoch = (snap.current_epoch is None or snap.history_sync_pending
                             or (plan.candidate.action == "REBUILD" and self.epoch_retirement)
                             or planning_snap.force_new_epoch_for_recovery or migrating)
                if new_epoch:
                    attempt = self.store.plan_new_epoch(planned, info.engine_restart_id, reason)
                    ack = await self._prepare_epoch(attempt, info)
                    epoch = replace(attempt.epoch, session_generation=ack.session_generation)
                else:
                    attempt = self.store.save_attempt(planned)
                    epoch = snap.current_epoch
                generation = candidate_request(plan.candidate, self.policy.max_new_tokens)
                submitted = time.monotonic()
                if quotes is not None:
                    quote = next(row for row in quotes.quotes if row.candidate_id == plan.candidate.candidate_id)
                    attempt = replace(attempt, answer_quote=quote_observation(quotes, quote, submitted))
                attempt = self.store.submit_attempt(attempt, epoch)
                self._active_generations += 1
                try:
                    result = await self.engine.generate(**generation, session_id=epoch.engine_session_id,
                                                    expected_generation=epoch.session_generation,
                                                    expected_restart_id=epoch.engine_restart_id,
                                                    attempt_id=attempt.attempt_id, fingerprint=self.template.fingerprint)
                finally:
                    self._active_generations -= 1
                response_done = time.monotonic()
                # Persist before validation so malformed/partial results remain auditable.
                attempt = self.store.save_attempt(replace(attempt, stage="RESULT_READY", result=result))
                self._validate(attempt, result)
                self._observe_answer_source(attempt)
                snapshot = self.store.commit_turn_and_outbox(attempt.attempt_id)
            except BaseException as exc:
                current = self.store.get_attempt(attempt.attempt_id)
                reading_pending = (current.stage == "STARTED"
                    and self.store.has_submitted_model_step(attempt.attempt_id))
                if reading_pending:
                    self.events.emit("reading_recovery_required", business_session_id=session_id,
                                     attempt_id=attempt.attempt_id, error=str(exc))
                elif current.stage in ("STARTED", "PLANNED", "PREPARED"):
                    self.store.fail_attempt(attempt.attempt_id, str(exc))
                    self._retire_background()
                else:
                    # PREPARING can have a lost ACK. SUBMITTED can have a lost result.
                    # RESULT_READY can have a failed CAS. None permit a blind retry.
                    self.events.emit("attempt_recovery_required", business_session_id=session_id,
                                     attempt_id=attempt.attempt_id, stage=current.stage, error=str(exc))
                self.events.emit("attempt_failed", business_session_id=session_id, attempt_id=attempt.attempt_id, error=str(exc))
                raise
            committed_at = time.monotonic()
            timing = ClientTiming(arrived, submitted,
                submitted+result.ttft_ms/1000 if result.ttft_ms is not None else None,
                submitted+result.last_token_ms/1000 if result.last_token_ms is not None else None,
                response_done, committed_at, token_observation="chunk" if result.ttft_ms is not None else "none")
            control_ms = ((submitted-entered) + (committed_at-response_done))*1000 - retrieval_segment_ms - reading_segment_ms
            answer_phase_window = None
            try:
                self._postplan_control_ms[plan.candidate.action].append(
                    ((submitted-plan_started)+(committed_at-response_done))*1000
                    -(reading_segment_ms if rolling_result is None else 0.0))
                self.cost_model.observe_output(len(result.output_ids))
                selected_estimate = next((e for e in plan.predicted if e.candidate_id == plan.candidate.candidate_id), None)
                if (self.online_calibration and selected_estimate is not None and selected_estimate.calibrated
                        and result.server_elapsed_ms is not None and info.engine_kind == "sglang"):
                    observed_total = (result.server_elapsed_ms+control_ms
                        +plan.candidate.predicted_read_ms+plan.candidate.auxiliary_ms)
                    self.cost_model.observe(selected_estimate, observed_total)
                answer_phase_window = {"start": answer_phase_started, "end": time.monotonic(),
                                     "commit_at": committed_at,
                                     "clock": "client_monotonic", "scope": "answer_until_commit",
                                     "start_definition": ("rolling_proposals_compilation_start" if rolling_result
                                                          else "final_answer_compilation_start"),
                                     "end_definition": "response_tail_control_complete_before_audit_write"}
                self.events.emit("ledger_committed", business_session_id=session_id, attempt_id=attempt.attempt_id,
                                 ledger_version=snapshot.version, epoch=epoch.epoch_no,
                                 answer_phase_window=answer_phase_window)
            except Exception as error:
                logging.getLogger(__name__).error("Diagnostic output failed after commit: %s", error)
            self._retire_background()
            return self._response(replace(attempt, stage="COMMITTED"),
                                  elapsed_ms=(committed_at-arrived)*1000, timing=timing, control_ms=control_ms,
                                  answer_phase_window=answer_phase_window)

    async def recover(self, session_id: str, *, abandon: bool = False):
        async with self._lock(session_id):
            return await self._recover(session_id, abandon=abandon)

    async def _recover(self, session_id: str, *, abandon: bool = False):
        attempt = self.store.pending_attempt(session_id)
        if attempt is None:
            return None
        if attempt.stage == "RESULT_READY":
            try:
                self._validate(attempt, attempt.result)
            except EngineError:
                if not abandon:
                    raise PendingAttempt("Stored result is invalid; explicit abandonment required")
                self.store.fail_attempt(attempt.attempt_id, "invalid_completed_result", force_recovery=True)
                await self._flush_retirements()
                return None
            self._observe_answer_source(attempt)
            committed = self.store.commit_turn_and_outbox(attempt.attempt_id)
            self.cost_model.observe_output(len(attempt.result.output_ids))
            self.events.emit("attempt_recovered", attempt_id=attempt.attempt_id, ledger_version=committed.version)
            self._retire_background()
            return self._response(replace(attempt, stage="COMMITTED"), replay=True)
        if attempt.stage in ("STARTED", "PLANNED", "PREPARING", "PREPARED"):
            if not abandon:
                raise PendingAttempt("Unfinished preparation: confirm old controller stopped, then recover with abandon=True")
            if attempt.epoch and attempt.epoch.session_generation == 0:
                try:
                    info = await self.engine.info()
                    ack = await self._prepare_epoch(attempt, info)
                    attempt = self.store.save_attempt(replace(attempt, epoch=replace(attempt.epoch, session_generation=ack.session_generation)))
                except EngineError as exc:
                    if exc.code != "STALE_RESTART":
                        raise
            self.store.fail_attempt(attempt.attempt_id, "explicit_pre_submit_abandonment")
            await self._flush_retirements()
            return None
        try:
            status = await self.engine.status(engine_session_id=attempt.epoch.engine_session_id,
                                             attempt_id=attempt.attempt_id,
                                             expected_restart_id=attempt.epoch.engine_restart_id)
        except EngineError as exc:
            if exc.code != "STALE_RESTART":
                raise
            self.store.fail_attempt(attempt.attempt_id, "engine_restart", force_recovery=True, retire_candidate=False)
            return None
        if status.result is not None:
            self.store.save_attempt(replace(attempt, stage="RESULT_READY", result=status.result))
            return await self._recover(session_id, abandon=abandon)
        if status.inflight or status.result_pending or not status.terminal:
            raise PendingAttempt("Engine outcome is unresolved; wait for completion/cancellation or restart before retry")
        reason = ("terminal_answer_result_unavailable" if status.state == "COMPLETED"
                  else f"terminal_engine_state:{status.state}")
        self.store.fail_attempt(attempt.attempt_id, reason, force_recovery=True,
                                terminal_receipt=status.receipt)
        self.events.emit("attempt_failed", business_session_id=session_id, attempt_id=attempt.attempt_id,
                         error=reason)
        await self._flush_retirements()
        return None

    async def close_session(self, session_id: str):
        async with self._lock(session_id):
            self.store.close_session(session_id)
        await self._flush_retirements(session_id)
