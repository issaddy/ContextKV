"""Serial turns, pure planning, guarded generation and idempotent durable commit."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import time
import uuid
import statistics
from typing import Awaitable, Callable
from weakref import WeakValueDictionary

from .candidates import CandidatePolicy, build_candidates
from .cost_model import CostModel
from .engine_client import Engine, EngineError, EngineInfo, PrepareAck
from .epoch import EpochCoordinator
from .events import EventLog
from .planner import PlannerKnobs, choose_plan
from .schema import (ClientTiming, CompletionReceipt, CostEstimate, EvidenceBlock, EvidenceKey,
                     EvidenceSpan, GenerationResult, NoFeasiblePlan, QueryContext, SessionSnapshot, TurnRecord)
from .serialization import dumps
from .sampling import SamplingRecipe, request_sampling
from .session_store import (Attempt, IdempotencyConflict, PendingAttempt, SessionStore, StoreError)
from .templates import ByteTemplate


class ContextOverflow(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceSelection:
    """Material returned by a configured retriever inside the turn transaction."""
    evidence: tuple[EvidenceBlock, ...]
    retrieval_priority: tuple[EvidenceKey, ...]
    query_context: QueryContext | None = None


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


class Controller:
    def __init__(self, store: SessionStore, engine: Engine, *, template=None,
                 policy: CandidatePolicy | None = None, knobs: PlannerKnobs | None = None,
                 cost_model: CostModel | None = None, events: EventLog | None = None,
                 shadow: bool = False, epoch_retirement: bool = True,
                 online_calibration: bool = False, sampling: SamplingRecipe | None = None):
        self.store, self.engine = store, engine
        self.template = template or ByteTemplate()
        self.policy = policy or CandidatePolicy()
        self.knobs = knobs or PlannerKnobs()
        self.cost_model = cost_model or CostModel(max_new_tokens=self.policy.max_new_tokens)
        self.events = events or EventLog()
        self.epochs = EpochCoordinator(store, engine, self.events)
        self.shadow = shadow
        self.epoch_retirement = epoch_retirement
        self.online_calibration = online_calibration
        self._sampling_params = request_sampling(sampling)
        self._active_generations = 0
        self._postplan_control_ms = {action: deque(maxlen=32) for action in ("APPEND", "REBUILD")}
        self._calibration_identity = None
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._background: set[asyncio.Task] = set()
        self._retire_requested = False

    def _lock(self, session_id: str):
        # Holders and waiting coroutines keep the lock alive. Idle sessions do
        # not need an in-memory lock; their state remains in the durable store.
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def _retire_background(self):
        self._retire_requested = True
        if any(not task.done() for task in self._background):
            return
        task = asyncio.create_task(self._flush_retire_requests())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _flush_retire_requests(self):
        # Coalesce turn notifications while one retirement is slow. A new
        # commit during flush still triggers a fresh durable-outbox snapshot.
        while self._retire_requested:
            self._retire_requested = False
            await self.epochs.flush()

    async def drain(self):
        if self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)
        await self.epochs.flush()

    async def aclose(self):
        await self.drain()
        if hasattr(self.engine, "aclose"):
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
        if (receipt.rid, receipt.attempt_id, receipt.engine_session_id, receipt.session_generation,
                receipt.engine_restart_id, receipt.input_hash, receipt.finish_reason) != (
                attempt.attempt_id, attempt.attempt_id, epoch.engine_session_id, epoch.session_generation,
                epoch.engine_restart_id, candidate.input_hash, result.finish_reason):
            raise EngineError("RECEIPT_IDENTITY_MISMATCH")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in result.output_ids):
            raise EngineError("INVALID_OUTPUT_IDS")
        if len(result.output_ids) > self.policy.max_new_tokens:
            raise EngineError("OUTPUT_BUDGET_EXCEEDED")
        if any(value < 0 for value in (receipt.actual_device_hit_tokens, receipt.actual_host_loaded_tokens,
                                      receipt.actual_prefill_tokens, receipt.inserted_cache_tokens)):
            raise EngineError("INVALID_CACHE_METRICS")

    def _response(self, attempt: Attempt, *, elapsed_ms=None, replay=False,
                  timing: ClientTiming | None = None, control_ms=None) -> TurnResponse:
        result, plan, epoch = attempt.result, attempt.plan, attempt.epoch
        receipt = result.receipt
        prediction = next((e for e in plan.predicted if e.candidate_id == plan.candidate.candidate_id), None)
        ttft = (timing.t_first-timing.t_arrive)*1000 if timing and timing.t_first is not None else None
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
                                                 if key in plan.candidate.required_blocks))

    async def serve_turn(self, session_id: str, query: str, evidence: tuple[EvidenceBlock, ...] = (), *,
                         request_id: str | None = None, corpus_snapshot_id: str = "demo-v1",
                         reference_history: tuple[TurnRecord, ...] | None = None,
                         retrieval_priority: tuple[EvidenceKey, ...] | None = None,
                         query_context: QueryContext | None = None,
                         evidence_provider: Callable[[SessionSnapshot], Awaitable[EvidenceSelection]] | None = None) -> TurnResponse:
        if not session_id or not query.strip():
            raise ValueError("session_id and query must be nonempty")
        evidence = tuple(evidence)
        if evidence_provider is not None and (evidence or reference_history is not None or retrieval_priority is not None
                                             or query_context is not None):
            raise ValueError("retrieved queries cannot also supply evidence, reference history or retrieval priority")
        request_id = request_id or uuid.uuid4().hex
        # Query identity is stable before retrieval. A retry returns the original
        # attempt rather than re-retrieving against newer history or rankings.
        content = (("retrieved_query", query, corpus_snapshot_id) if evidence_provider is not None else
                   (query, evidence, corpus_snapshot_id, reference_history, retrieval_priority))
        if query_context is not None:
            intent = {"query": query_context.query, "relation": query_context.relation,
                      "history_turns": query_context.history_turns}
            if query_context.task_frame is not None:
                frame = asdict(query_context.task_frame)
                conditions = query_context.task_frame.conditions
                # Preserve explicit v9 request identities after reading their
                # empty/composite conditions into the new immutable structure.
                if not conditions:
                    frame["conditions"] = ""
                elif len(conditions) == 1 and conditions[0][0] == "legacy":
                    frame["conditions"] = conditions[0][1]
                intent["task_frame"] = frame
            content = (*content, intent)
        request_hash = hashlib.sha256(dumps(content).encode()).hexdigest()
        arrived = time.monotonic()
        async with self._lock(session_id):
            entered = time.monotonic()
            self.events.emit("turn_received", business_session_id=session_id, request_id=request_id)
            previous = self.store.find_request(session_id, request_id)
            if previous:
                if previous.request_hash != request_hash:
                    raise IdempotencyConflict("request_id reused with different content")
                if previous.stage == "COMMITTED":
                    return self._response(previous, replay=True)
                if previous.stage == "FAILED":
                    raise StoreError("Previous attempt failed; use a new request_id for an explicit retry")
            if self.store.pending_attempt(session_id):
                await self._recover(session_id)
                previous = self.store.find_request(session_id, request_id)
                if previous and previous.stage == "COMMITTED":
                    return self._response(previous, replay=True)
                if previous and previous.stage == "FAILED":
                    raise StoreError("Recovered terminal failure; use a new request_id")
            snap = self.store.read_snapshot(session_id, corpus_snapshot_id)
            if snap.corpus_snapshot_id != corpus_snapshot_id:
                raise ValueError("Session corpus snapshot is fixed; start a new session for another snapshot")
            if reference_history is not None:
                snap = self.store.sync_history(session_id, reference_history, snap.version)
                self.events.emit("reference_history_sync", business_session_id=session_id,
                                 raw_history_version=snap.raw_history_version, pending=snap.history_sync_pending)
            attempt = self.store.begin_attempt(session_id, request_id, request_hash, query, snap.version)
            if attempt.stage == "COMMITTED":
                return self._response(attempt, replay=True)
            if attempt.stage == "FAILED":
                raise StoreError("Previous attempt failed; use a new request_id")
            retrieval_ms = None
            try:
                if evidence_provider is not None:
                    retrieval_started = time.monotonic()
                    try:
                        selection = await evidence_provider(snap)
                    except BaseException as exc:
                        self.events.emit("retrieval_failed", business_session_id=session_id,
                                         attempt_id=attempt.attempt_id, error=str(exc),
                                         retrieval_ms=(time.monotonic()-retrieval_started)*1000)
                        raise
                    retrieval_ms = (time.monotonic()-retrieval_started)*1000
                    evidence = tuple(selection.evidence)
                    retrieval_priority = tuple(selection.retrieval_priority)
                    query_context = selection.query_context
                    self.events.emit("retrieval_finished", business_session_id=session_id,
                                     attempt_id=attempt.attempt_id, retrieval_ms=retrieval_ms,
                                     retrieval_priority=retrieval_priority,
                                     selected_evidence_keys=tuple(b.key for b in evidence))
                info = await self.engine.info()
                if info.fingerprint != self.template.fingerprint:
                    raise EngineError("FINGERPRINT_MISMATCH")
                recovery = snap.current_epoch is not None and snap.current_epoch.engine_restart_id != info.engine_restart_id
                planning_snap = replace(snap, force_new_epoch_for_recovery=True, must_rebuild=True) if recovery else snap
                measure_redundancy = "redundancy_threshold" in {self.knobs.strategy, self.knobs.fallback_strategy}
                candidates = build_candidates(planning_snap, query, evidence, self.policy, self.template,
                                               retrieval_priority=retrieval_priority,
                                               query_context=query_context,
                                               measure_evidence_redundancy=measure_redundancy)
                for candidate in candidates:
                    self.events.emit("candidate_built", business_session_id=session_id, attempt_id=attempt.attempt_id,
                                     candidate_id=candidate.candidate_id, action=candidate.action,
                                     input_hash=candidate.input_hash, input_tokens=len(candidate.input_ids),
                                     theoretical_lcp=candidate.theoretical_lcp, feasible=candidate.feasible,
                                     evidence_spans=candidate.evidence_spans, build_ms=candidate.build_ms,
                                     evidence_body_tokens=candidate.evidence_body_tokens,
                                     nonrequired_evidence_body_tokens=candidate.nonrequired_evidence_body_tokens)
                valid = tuple(c for c in candidates if c.feasible)
                if not valid:
                    raise ContextOverflow("shared_context_policy_overflow: required evidence and full QA exceed budget")
                needs_quote = self.knobs.strategy in {"cost", "warm_idle"} and len(valid) > 1
                quotes = await self.engine.quote_batch(valid, self.template.fingerprint) if needs_quote else None
                if quotes is not None:
                    self.events.emit("quote_observed", business_session_id=session_id, attempt_id=attempt.attempt_id, quote=quotes)
                if quotes is not None and quotes.engine_restart_id != info.engine_restart_id:
                    # At most one re-quote after a restart; native admission remains authoritative.
                    info = await self.engine.info()
                    planning_snap = replace(snap, force_new_epoch_for_recovery=True, must_rebuild=True)
                    candidates = build_candidates(planning_snap, query, evidence, self.policy, self.template,
                                                   retrieval_priority=retrieval_priority,
                                                   query_context=query_context,
                                                   measure_evidence_redundancy=measure_redundancy)
                    valid = tuple(c for c in candidates if c.feasible)
                    quotes = await self.engine.quote_batch(valid, self.template.fingerprint)
                    if quotes.engine_restart_id != info.engine_restart_id:
                        raise EngineError("UNSTABLE_RESTART")
                identity = (info.engine_restart_id, info.cache_config_hash)
                if (identity != self._calibration_identity
                        and (self.cost_model.prefill_points or self.cost_model.decode_points)):
                    scope = self.cost_model.check_calibration_scope(info)
                    self.events.emit("cost_calibration_scope", **scope)
                    self._calibration_identity = identity
                # The planner validates this batch once before selecting a candidate.
                pressure = (max(1, quotes.active_requests + (quotes.queued_requests or 0) + 1) if quotes is not None
                            and quotes.active_requests is not None else max(1, self._active_generations + 1))
                spent_control_ms = (time.monotonic()-entered)*1000 - (retrieval_ms or 0)
                observed = [value for values in self._postplan_control_ms.values() for value in values]
                default_postplan = statistics.mean(observed) if observed else 0.0
                # Preparation/serialization costs differ between retaining and
                # replacing an epoch. Charge prior completed observations to
                # their action instead of cancelling one common mean in A/R.
                planning_control_ms = {action: spent_control_ms + (
                    statistics.mean(values) if values else default_postplan)
                    for action, values in self._postplan_control_ms.items()}
                plan_started = time.monotonic()
                plan = choose_plan(candidates, quotes, self.cost_model, planning_snap, self.knobs,
                                   concurrency=pressure, control_ms=planning_control_ms)
                if isinstance(plan, NoFeasiblePlan):
                    raise ContextOverflow(plan.reason)
                if self.shadow:
                    self.events.emit("shadow_plan", business_session_id=session_id, attempt_id=attempt.attempt_id, plan=plan)
                    plan = choose_plan(candidates, quotes, self.cost_model, planning_snap,
                                       replace(self.knobs, strategy=self.knobs.fallback_strategy),
                                       concurrency=pressure, control_ms=planning_control_ms)
                self.events.emit("plan_selected", business_session_id=session_id, attempt_id=attempt.attempt_id,
                                 action=plan.candidate.action, reason=plan.reason, predicted=plan.predicted,
                                 switch_margin_ms=plan.switch_margin_ms, predicted_saving_ms=plan.predicted_saving_ms,
                                 turns_since_rebuild=planning_snap.turns_since_rebuild,
                                 min_stable_turns=self.knobs.min_stable_turns,
                                 execution_mode=plan.execution_mode, pressure_bucket=pressure,
                                 observed_queued_requests=quotes.queued_requests if quotes else None,
                                 prior_completed_length_samples=len(self.cost_model._observed_output_lengths))
                append_failure = next((c.reason for c in candidates if c.action == "APPEND" and not c.feasible), None)
                reason = ("reference_history_sync" if snap.history_sync_pending else
                          "recovery" if planning_snap.force_new_epoch_for_recovery else
                          "initial" if snap.current_epoch is None else
                          (append_failure or plan.reason) if plan.candidate.action == "REBUILD" else "append")
                planned = replace(attempt, stage="PLANNED", evidence=evidence,
                                  plan=plan, branch_reason=reason,
                                  retrieval_priority=retrieval_priority, retrieval_ms=retrieval_ms,
                                  query_context=query_context)
                # No-C ablation keeps the native session across text rebuilds.
                # History protocol changes and failed attempts still need safe recovery.
                new_epoch = (snap.current_epoch is None or snap.history_sync_pending
                             or (plan.candidate.action == "REBUILD" and self.epoch_retirement)
                             or planning_snap.force_new_epoch_for_recovery)
                if new_epoch:
                    attempt = self.store.plan_new_epoch(planned, info.engine_restart_id, reason)
                    ack = await self._prepare_epoch(attempt, info)
                    epoch = replace(attempt.epoch, session_generation=ack.session_generation)
                else:
                    attempt = self.store.save_attempt(planned)
                    epoch = snap.current_epoch
                attempt = self.store.submit_attempt(attempt, epoch)
                self.events.emit("epoch_prepared", business_session_id=session_id, attempt_id=attempt.attempt_id, epoch=epoch)
                self.events.emit("request_submitted", business_session_id=session_id, attempt_id=attempt.attempt_id,
                                 sampling_params={**self._sampling_params, "max_new_tokens": self.policy.max_new_tokens},
                                 constrained_citation_ids=tuple(ref for ref, key in plan.candidate.reference_map
                                                               if key in plan.candidate.required_blocks))
                submitted = time.monotonic()
                self._active_generations += 1
                try:
                    result = await self.engine.generate(input_ids=plan.candidate.input_ids, session_id=epoch.engine_session_id,
                                                    expected_generation=epoch.session_generation,
                                                    expected_restart_id=epoch.engine_restart_id,
                                                    attempt_id=attempt.attempt_id, fingerprint=self.template.fingerprint,
                                                    max_new_tokens=self.policy.max_new_tokens,
                                                    citation_ids=tuple(ref for ref, key in plan.candidate.reference_map
                                                                       if key in plan.candidate.required_blocks))
                finally:
                    self._active_generations -= 1
                response_done = time.monotonic()
                # Persist before validation so malformed/partial results remain auditable.
                attempt = self.store.save_attempt(replace(attempt, stage="RESULT_READY", result=result))
                self._validate(attempt, result)
                snapshot = self.store.commit_turn_and_outbox(attempt.attempt_id)
                committed_at = time.monotonic()
                timing = ClientTiming(arrived, submitted,
                    submitted+result.ttft_ms/1000 if result.ttft_ms is not None else None,
                    submitted+result.last_token_ms/1000 if result.last_token_ms is not None else None,
                    response_done, committed_at, token_observation="chunk" if result.ttft_ms is not None else "none")
                control_ms = ((submitted-entered) + (committed_at-response_done))*1000 - (retrieval_ms or 0)
                self._postplan_control_ms[plan.candidate.action].append(
                    ((submitted-plan_started)+(committed_at-response_done))*1000)
                self.cost_model.observe_output(len(result.output_ids))
                selected_estimate = next((e for e in plan.predicted if e.candidate_id == plan.candidate.candidate_id), None)
                if (self.online_calibration and selected_estimate is not None and selected_estimate.calibrated
                        and result.server_elapsed_ms is not None and info.engine_kind == "sglang"):
                    self.cost_model.observe(selected_estimate, result.server_elapsed_ms + control_ms)
                self.events.emit("generation_finished", business_session_id=session_id, attempt_id=attempt.attempt_id,
                                 receipt=result.receipt, output_tokens=len(result.output_ids),
                                 client_timing=timing, server_ttft_ms=result.server_ttft_ms,
                                 decode_ms=result.decode_ms, control_ms=control_ms,
                                 timing_source=result.timing_source)
                self.events.emit("ledger_committed", business_session_id=session_id, attempt_id=attempt.attempt_id,
                                 ledger_version=snapshot.version, epoch=epoch.epoch_no)
            except BaseException as exc:
                current = self.store.get_attempt(attempt.attempt_id)
                if current.stage in ("STARTED", "PLANNED", "PREPARED"):
                    self.store.fail_attempt(attempt.attempt_id, str(exc))
                    self._retire_background()
                else:
                    # PREPARING can have a lost ACK. SUBMITTED can have a lost result.
                    # RESULT_READY can have a failed CAS. None permit a blind retry.
                    self.events.emit("attempt_recovery_required", business_session_id=session_id,
                                     attempt_id=attempt.attempt_id, stage=current.stage, error=str(exc))
                self.events.emit("attempt_failed", business_session_id=session_id, attempt_id=attempt.attempt_id, error=str(exc))
                raise
            self._retire_background()
            return self._response(replace(attempt, stage="COMMITTED"),
                                  elapsed_ms=(committed_at-arrived)*1000, timing=timing, control_ms=control_ms)

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
                await self.epochs.flush()
                return None
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
            await self.epochs.flush()
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
        if status.inflight or not status.terminal:
            raise PendingAttempt("Engine outcome is unresolved; wait for completion/cancellation or restart before retry")
        if not abandon and status.state not in ("FAILED", "CANCELLED", "RETIRED", "REJECTED"):
            raise PendingAttempt("Completed result is unavailable; explicit abandonment required")
        self.store.fail_attempt(attempt.attempt_id, f"terminal_engine_state:{status.state}", force_recovery=True)
        await self.epochs.flush()
        return None

    async def close_session(self, session_id: str):
        async with self._lock(session_id):
            self.store.close_session(session_id)
        await self.epochs.flush(session_id)
