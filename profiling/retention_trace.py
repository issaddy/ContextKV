"""Source-first conversation capture and fixed-input C replay/fit.

The replay executes every recorded model input and CPU reduction in its original
DAG. New outputs are measured, not forced to match the captured reader output.
It measures C costs; replay output is not a new answer-quality experiment.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from controller.config import create_controller, create_model_template, load_config
from controller.context_schema import ContextBudget, ContextPlan, EvidenceClaim, EvidenceSet, Need, ReadTask, TaskState, TextRef
from controller.engine_client import EngineError, SGLangEngineClient
from controller.prefix_retention import (EmpiricalRetentionModel, PrefixReadObserver, PrefixRuntimeClient, TARGET_DEMAND_RULE,
    RETENTION_FEATURE_NAMES, source_token_ranges, source_demand_event_key)
from controller.reading import (build_read_plan, parse_read_result, stable_prefix_input_ids,
                                stable_prefix_messages, stable_source_payload, step_messages)
from controller.reduction import reduce_claims
from controller.sampling import bound_recipe, request_sampling
from controller.schema import ConfigFingerprint, PromptCandidate, SessionSnapshot, TurnRecord
from controller.serialization import decode, to_jsonable
from controller.session_store import SessionStore, load_record
from controller.templates import candidate_request, step_request, token_hash
from engine_patches.contextkv_runtime import executed_prefix_hashes
from profiling.retention_capture_cost import CaptureRunAccounting, load_resident_provider


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


class SourceLedger(SessionStore):
    """Use SessionStore's typed single-step reader without modifying its source."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self._mutex = threading.RLock()
        self.db = sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True,
                                 check_same_thread=False)
        self.db.row_factory = sqlite3.Row

    def committed_attempts(self, selected=(), *, limit=None):
        if selected:
            identities = selected if limit is None else selected[:limit]
            return tuple(self.get_attempt(identity) for identity in identities)
        suffix, parameters = ("", ()) if limit is None else (" LIMIT ?", (limit,))
        return tuple(self.get_attempt(row[0]) for row in self.db.execute(
            "SELECT a.attempt_id FROM attempts a "
            "WHERE a.stage='COMMITTED' ORDER BY a.rowid"+suffix, parameters))

    def before(self, attempt):
        return before_snapshot(self, attempt)


def before_snapshot(store, attempt):
    if attempt.expected_version == 0:
        return SessionSnapshot(attempt.session_id, corpus_snapshot_id=attempt.plan.candidate.corpus_snapshot_id)
    row = store.db.execute("SELECT snapshot FROM ledger_history WHERE session_id=? AND version=?",
                          (attempt.session_id, attempt.expected_version)).fetchone()
    return load_record(store.db, SessionSnapshot, row[0])


def recorded_request(call):
    return {key: value for key, value in call["request"].items() if key in {
        "input_ids", "max_new_tokens", "ignore_eos", "citation_ids", "response_schema"}}


def source_records(path, *, store=None, attempt_ids=None):
    if path is None:
        return {}
    records = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" in row:
                if row["status"] != "completed":
                    continue
            else:
                required = {"task_id", "conversation_id", "question", "actual_history", "snapshot_id",
                            "rollout_policy", "answer", "response", "split"}
                if not required <= row.keys() or row["rollout_policy"] != "dense5_rebuild_qwen3_32b_nonthinking":
                    raise ValueError("source without status is not the actual native training rollout contract")
            identity = row["response"]["attempt_id"]
            if attempt_ids is not None and identity not in attempt_ids:
                continue
            if "status" not in row:
                if store is None:
                    raise ValueError("native training rollout status must be derived from its actual SQLite Attempt")
                actual = store.get_attempt(identity)
                receipt = actual.result.receipt if actual.result is not None else None
                if actual.stage != "COMMITTED" or receipt is None or not receipt.generation_complete:
                    raise ValueError("native training rollout lacks a COMMITTED Attempt and complete native receipt")
                if actual.query != row["question"] or actual.result.text != row["answer"]:
                    raise ValueError("native training rollout text differs from its linked SQLite Attempt")
                corpus_ids = sorted({block.key.corpus_id for block in actual.evidence})
                row = {**row, "status": "completed", "dataset": row["dataset"] if "dataset" in row else "+".join(corpus_ids),
                    "source_status_derivation": dict(contract="training.task_rollout.run_actual_rollouts",
                        state_database=str(store.path), source_record_path=str(Path(path).resolve()),
                        source_record_sha256=digest(row), stage=actual.stage,
                        native_receipt=to_jsonable(receipt),
                        native_fingerprint=to_jsonable(actual.plan.candidate.fingerprint),
                        corpus_ids=corpus_ids)}
            records[identity] = row
            if attempt_ids is not None and records.keys() >= attempt_ids:
                break
    return records


def causal_pages(event, page_size, scope):
    hashes = executed_prefix_hashes(event["prefix_input_ids"], page_size)
    evidence, system = set(), set()
    pages = {}
    for end, hashed in hashes.items():
        page = scope+":"+hashed
        pages[page] = end
        if end <= event["fixed_system_prefix_tokens"]:
            system.add(page)
        if any(end > first and end-page_size < last for first, last in event["evidence_token_ranges"]):
            evidence.add(page)
    return pages, evidence, system


def reuse_pages(target, future, receipt, *, page_size, scope):
    """Distinguish demanded evidence pages from actual GPU/Host cache use."""
    target_pages, target_evidence, _ = causal_pages(target, page_size, scope)
    future_pages, future_evidence, future_system = causal_pages(future, page_size, scope)
    shared = target_pages.keys() & future_pages.keys()
    evidence = shared & target_evidence & future_evidence
    device = {page for page, end in future_pages.items() if end <= receipt["actual_device_hit_tokens"]}
    cached = {page for page, end in future_pages.items()
        if end <= receipt["actual_device_hit_tokens"]+receipt["actual_host_loaded_tokens"]}
    used = evidence & cached
    def category(pages, matched):
        if pages and pages == target_evidence:
            return "full_retained_evidence"
        return "partial_evidence" if pages else "system_or_metadata_only" if matched else "none"
    return dict(demand_category=category(evidence, shared), actual_use_category=category(used, shared & cached),
        shared_causal_page_ids=sorted(shared), shared_evidence_page_ids=sorted(evidence),
        actual_evidence_page_ids=sorted(used), actual_device_evidence_page_ids=sorted(used & device),
        actual_host_evidence_page_ids=sorted(used-device), shared_system_page_ids=sorted(shared & future_system),
        target_retained_evidence_pages=len(target_evidence),
        actual_evidence_token_hits=sum(max(0, min(future_pages[page], last)-max(future_pages[page]-page_size, first))
            for page in used for first, last in future["evidence_token_ranges"]),
        system_pages_are_evidence=False)


def annotated_generation(plan, step, template):
    rendered = template.render(step.input_messages)
    if tuple(rendered.input_ids) != tuple(step.input_ids) or rendered.input_hash != step.receipt.input_hash:
        raise ValueError("source StepRecord prompt, token path and native receipt differ")
    result = dict(step_id=step.step_id, kind=step.task.kind, operation=step.task.operation,
        predecessor_step_ids=list(step.task.predecessor_step_ids),
        input_ids=list(step.input_ids), input_hash=step.receipt.input_hash,
        original_output_ids=list(step.output_ids), original_receipt=to_jsonable(step.receipt),
        original_epoch=to_jsonable(step.engine_epoch), request=to_jsonable(step_request(rendered, step.task)),
        system_prompt=step.input_messages[0].content, context_action="REBUILD",
        original_cache_quote=step.metrics["cache_quote"], phase_window=step.metrics.get("phase_window"),
        prefix_input_ids=[], sources=[], task_state=to_jsonable(plan.task_state),
        fingerprint=step.metrics["cache_quote"]["fingerprint"],
        original_cache_config_hash=step.metrics["cache_quote"]["cache_config_hash"])
    if result["fingerprint"] != asdict(template.fingerprint):
        raise ValueError("source native quote differs from the declared reader/tokenizer")
    if step.task.kind == "READ":
        by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        units = tuple(by_id[identity] for identity in step.task.input_unit_ids)
        if step.input_messages[:2] != stable_prefix_messages(plan, step.task, units):
            raise ValueError("source READ is not the current canonical source_first prefix; recapture it natively")
        prefix = stable_prefix_input_ids(template, plan, step.task, units, rendered)
        ranges, system_tokens = source_token_ranges(template, rendered, step.input_messages, units)
        sources, _ = stable_source_payload(units)
        result.update(prefix_input_ids=list(prefix), evidence_token_ranges=ranges, fixed_system_prefix_tokens=system_tokens, sources=[
            {"source_unit": item["source_unit"], "source_spans": item["source_spans"],
             "document_family": item["document_family"], "applicability": to_jsonable(item["applicability"])} for item in sources])
    return result


def trace_source_metadata(metadata):
    fields = (
        "task_id", "conversation_id", "turn", "dataset", "split", "source_partition",
        "cohort_split", "cohort_manifest_sha256", "method", "status", "scope",
        "history_origin", "snapshot_id", "rollout_policy", "measurement_segment_id",
        "measurement_kind", "request_window", "workload_arrival", "record_number",
        "generation_dir", "source_status_derivation", "max_new_tokens",
        "capture_origin", "source_attempt_id", "source_example_id", "source_trace_origin",
        "source_mutations", "quality_claim", "source_trace_files", "source_bank_files",
        "probe", "packing",
    )
    return {**{key: metadata[key] for key in fields if key in metadata},
        "response": {key: metadata["response"][key] for key in ("answer_phase_window", "timing")
                     if key in metadata.get("response", {})}}


def export_attempt(store, attempt, template, sampling, metadata, *, recorded_calls=None):
    if attempt.stage != "COMMITTED" or attempt.reading_outcome is None:
        raise ValueError("retention trace requires a committed real reading request")
    plan = attempt.reading_outcome.plan
    if plan.read_layout != "source_first":
        raise ValueError("task_first is not stable-evidence training data; use capture to execute source_first first")
    parent_claims = {claim.claim_id: claim for claim in attempt.reading_outcome.claims}
    calls = {row["receipt"]["attempt_id"]: row for row in recorded_calls or ()}
    timing = metadata.get("response", {}).get("timing")
    arrival = timing["t_arrive"] if timing is not None else attempt.created_ns/1e9
    events = []
    for raw in store.execution_steps(attempt.attempt_id):
        stage, step, generation = store.execution_step(attempt.attempt_id, raw.step_id)
        if stage != "COMPLETE":
            raise ValueError("source trace has an incomplete internal step")
        if not step.input_ids:
            events.append(dict(step_id=step.step_id, kind="CPU_REDUCE", operation=step.task.operation,
                need_id=step.task.need_ids[0], claim_id=step.result.claims[0].claim_id,
                predecessor_step_ids=list(step.task.predecessor_step_ids),
                input_claims=to_jsonable(tuple(parent_claims[key] for key in step.task.input_claim_ids)),
                original_result=to_jsonable(step.result), source_started_s=None))
            continue
        event = annotated_generation(plan, step, template)
        event.update(read_task=to_jsonable(step.task),
                     original_output_text=generation.text,
                     input_claims=to_jsonable(tuple(parent_claims[key] for key in step.task.input_claim_ids)))
        event["prefix_observation"] = step.metrics.get("evidence_prefix")
        recorded = calls.get(step.receipt.attempt_id)
        if recorded:
            event["request"] = recorded_request(recorded)
        event["sampling"] = recorded["sampling"] if recorded else dict(sampling)
        event["sampling_evidence"] = "captured_generation_wire" if recorded else "declared_original_run_config"
        event["source_started_s"] = recorded["started_s"] if recorded else arrival+step.metrics["submitted_from_arrival_ms"]/1000
        event["source_completed_s"] = recorded["completed_s"] if recorded else event["source_started_s"]+generation.elapsed_ms/1000
        events.append(event)
    final = attempt.plan.candidate
    request = candidate_request(final, metadata["max_new_tokens"])
    if final.input_hash != attempt.result.receipt.input_hash or token_hash(final.input_ids) != final.input_hash:
        raise ValueError("final ANSWER path differs from its actual receipt")
    recorded = calls.get(attempt.result.receipt.attempt_id)
    final_start = recorded["started_s"] if recorded else timing["t_submit"]
    final_end = recorded["completed_s"] if recorded else timing["t_response_done"]
    events.append(dict(step_id="ANSWER", kind="ANSWER", operation="answer", predecessor_step_ids=[
        event["step_id"] for event in events], input_ids=list(final.input_ids), input_hash=final.input_hash,
        original_output_ids=list(attempt.result.output_ids), original_receipt=to_jsonable(attempt.result.receipt),
        original_output_text=attempt.result.text,
        original_epoch=to_jsonable(attempt.epoch), request=to_jsonable(request),
        system_prompt=final.messages[0].content, context_action=final.action,
        sampling=recorded["sampling"] if recorded else dict(sampling),
        sampling_evidence="captured_generation_wire" if recorded else "declared_original_run_config",
        fingerprint=asdict(template.fingerprint),
        prefix_input_ids=[], sources=[], source_started_s=final_start, source_completed_s=final_end))
    if recorded:
        events[-1]["request"] = recorded_request(recorded)
    previous_end = arrival
    for event in events:
        if event["kind"] == "CPU_REDUCE":
            event["control_gap_seconds"] = 0
        else:
            event["control_gap_seconds"] = max(0, event["source_started_s"]-previous_end)
            previous_end = event["source_completed_s"]
    return dict(attempt_id=attempt.attempt_id, session_id=attempt.session_id, request_id=attempt.request_id,
        query=attempt.query, arrival_s=arrival, context_plan=to_jsonable(plan),
        actual_history=to_jsonable(before_snapshot(store, attempt).raw_history), events=events,
        dataset=metadata.get("dataset"), split=metadata.get("split"), source_metadata=trace_source_metadata(metadata))


class RecordingEngine:
    """Capture the actual client sampling recipe and every submitted input."""
    def __init__(self, engine):
        self.engine, self.calls = engine, []
        self.pending_quotes = {}
    def __getattr__(self, name):
        return getattr(self.engine, name)
    async def quote_batch(self, candidates, fingerprint):
        batch = await self.engine.quote_batch(candidates, fingerprint)
        self.pending_quotes.update({quote.input_hash: (batch, quote.candidate_id) for quote in batch.quotes})
        return batch
    async def generate(self, **kwargs):
        hashed = token_hash(kwargs["input_ids"])
        if hashed not in self.pending_quotes:
            candidate = PromptCandidate("capture:"+kwargs["attempt_id"], "REBUILD", 0, (),
                tuple(kwargs["input_ids"]), hashed, frozenset(), frozenset(),
                fingerprint=self.fingerprint, output_budget=kwargs["max_new_tokens"])
            await self.quote_batch((candidate,), self.fingerprint)
            quote_origin = "capture_pre_generation_quote"
        else:
            quote_origin = "actual_controller_quote"
        batch, candidate_id = self.pending_quotes[hashed]
        self.pending_quotes.clear()
        row = dict(started_s=time.monotonic(), sampling=dict(self.engine._sampling_params),
                   request=to_jsonable(kwargs), state="SUBMITTED", cache_quote=to_jsonable(batch),
                   quote_candidate_id=candidate_id, quote_origin=quote_origin)
        self.calls.append(row)
        try:
            result = await self.engine.generate(**kwargs)
        except BaseException as error:
            row.update(completed_s=time.monotonic(), state="FAILED_UNRESOLVED",
                       error=dict(type=type(error).__name__, message=str(error)))
            raise
        row.update(completed_s=time.monotonic(), state="COMPLETE",
                   receipt=to_jsonable(result.receipt), result=to_jsonable(result))
        return result


class CapturePrefixObserver:
    """Record post-READ native state, without retaining any prefix."""
    def __init__(self, engine, template):
        self.runtime, self.template = PrefixRuntimeClient(engine), template
        self.accesses = defaultdict(list)

    async def __call__(self, plan, task, units, rendered, record):
        ids = stable_prefix_input_ids(self.template, plan, task, units, rendered)
        sources, _ = stable_source_payload(units)
        sources = [{"source_unit": item["source_unit"], "source_spans": item["source_spans"],
                    "document_family": item["document_family"], "applicability": to_jsonable(item["applicability"])} for item in sources]
        inspected = await self.runtime.inspect(receipt=record.receipt, prefix_input_ids=ids, sources=sources)
        key = digest([record.engine_epoch.business_session_id, inspected["cache_scope_hash"], inspected["prefix_hash"]])
        now = time.monotonic_ns()
        features = PrefixReadObserver._features(plan, sources, self.accesses[key], now)
        features.update(PrefixReadObserver._native_features(inspected))
        self.accesses[key].append(dict(observed_ns=now, payload={"turn_id": plan.task_state.current_turn_id}))
        return dict(state="OBSERVED", inspection=inspected, raw_features=features,
                    observed_ns=now, policy="capture_observation_only", training_origin="canonical_native_READ")


def make_trace(turns, *, config, source_db, captured, case_outcomes=()):
    model_events = [event for turn in turns for event in turn["events"] if event["kind"] != "CPU_REDUCE"]
    hashes = [event["input_hash"] for event in model_events]
    for event in model_events:
        if token_hash(event["input_ids"]) != event["input_hash"]:
            raise ValueError("trace input token hash mismatch")
    eligible = bool(turns) and all(turn["split"] in {"train", "dev"} and turn["dataset"]
        and turn["source_metadata"].get("scope", "study") == "study" for turn in turns)
    return dict(format="contextkv_source_first_trace_v2", source="hardware",
        scope="fixed_recorded_reading_dag", training_eligible=eligible,
        layout="source_first", captured_generation_wire=captured,
        unexported_pressure_intervals=[{**case.get("request_window", case.get("native_window")),
            "reason": case["reason"]}
            for case in case_outcomes if "request_window" in case or "native_window" in case],
        source_database=str(Path(source_db).resolve()), config=to_jsonable(config),
        fingerprint=model_events[0]["fingerprint"] if model_events else None,
        turns=turns, input_hashes=hashes, input_dag_digest=digest([
            [turn["attempt_id"], [(event["step_id"], event.get("input_hash"), event["predecessor_step_ids"])
                                  for event in turn["events"]]] for turn in turns]),
        quality_scope="none; replay is cache/cost attribution, with frozen actual inputs",
        output_scope="original output budgets and sampling; new output lengths may differ")


async def completed_capture_failure(service, recorder, cursor, error):
    """Retire completed capped READs, retaining their actual wire and cost."""
    if error.code != "READ_STEP_OUTPUT_TRUNCATED":
        raise error
    calls = recorder.calls[cursor:]
    if not calls or any(not call.get("receipt", {}).get("generation_complete") for call in calls):
        raise error
    cleanup = []
    for call in calls:
        request = call["request"]
        ack = await service.engine.retire(request["session_id"], request["expected_generation"],
            "capture-completed-failure:"+request["attempt_id"], request["expected_restart_id"],
            "completed_truncated_source_capture")
        if not ack.acknowledged:
            raise ValueError("completed capture failure retirement pending")
        cleanup.append(to_jsonable(ack))
    return dict(status="completed_native_failed_phase", reason=str(error), cleanup=cleanup,
                native_window=dict(start=min(call["started_s"] for call in calls),
                                   end=max(call["completed_s"] for call in calls)),
                actual_native_receipts=[call["receipt"] for call in calls])


def replan_actual_retrieval(attempt, snapshot, service, args):
    """New canonical READ from actual archived retrieval, never an old READ label."""
    from adapters.corpus import CorpusIndex
    from adapters.evidence_pool import SourceDirectory
    if not attempt.evidence:
        return None
    corpus = CorpusIndex(attempt.evidence, corpus_id=attempt.evidence[0].key.corpus_id,
                         snapshot_id=snapshot.corpus_snapshot_id)
    directory = SourceDirectory(corpus, None, service.template.tokenizer)
    pool = directory.full_scope(tuple(block.key for block in attempt.evidence))
    evidence = EvidenceSet("actual-capture:"+attempt.attempt_id, pool.units, source_blocks=pool.source_blocks)
    state = attempt.query_context.task_state if attempt.query_context is not None else None
    if state is None:
        history = snapshot.raw_history
        refs = tuple(TextRef(turn.turn_id, role, 0, len(text)) for turn in history
            for role, text in (("user", turn.user_text), ("assistant", turn.assistant_text)))
        turn_id = history[-1].turn_id+1 if history else 0
        state = TaskState(attempt.attempt_id, attempt.query, working_history=refs,
            needs=(Need("question", anchor_spans=(TextRef(turn_id, "user", 0, len(attempt.query)),)),),
            current_turn_id=turn_id)
    budget = ContextBudget(window_tokens=service.policy.max_total_model_tokens, body_tokens=8192,
        read_input_tokens=args.read_input_budget, read_output_tokens=args.read_output_budget, max_steps=512)
    return build_read_plan(state, evidence, "all_supplied_documents", budget,
        template=service.template, snapshot=snapshot, read_layout="source_first", read_packing=args.read_packing)


async def capture(args):
    config = load_config(args.config)
    config = replace(config, engine=replace(config.engine, base_url=args.base_url),
        state_dir=str(Path(args.output_dir)/"state"), prefix_retention={"mode": "native"})
    async with CaptureRunAccounting(config, args.output_dir, args.gpu_device,
            resource_gpu_devices=args.resource_gpu_devices,
            resource_process_roots=args.resource_process_roots) as accounting:
        service = create_controller(config)
        resident_provider = None
        if args.study_config is not None:
            resident_provider, accounting.resident_model_manifest = await asyncio.to_thread(
                load_resident_provider, args.study_config, service.template)
            accounting.record_resident_models(args.study_config, accounting.resident_model_manifest)
        service.observe_prefix = CapturePrefixObserver(service.engine, service.template)
        recorder = RecordingEngine(service.engine)
        service.engine = recorder
        accounting.calls = recorder.calls
        source = SourceLedger(args.state_db)
        selected = source.committed_attempts(args.attempt_id, limit=args.max_capture_turns)
        metadata = source_records(args.records, store=source, attempt_ids={old.attempt_id for old in selected})
        turns, cases = [], []
        accounting.case_outcomes = cases
        try:
            for old in selected:
                before = source.before(old)
                current = service.store.read_snapshot(old.session_id, before.corpus_snapshot_id)
                service.store.sync_history(old.session_id, before.raw_history, current.version)
                snapshot = service.store.get_snapshot(old.session_id)
                original = old.reading_outcome.plan if old.reading_outcome is not None else old.context_plan
                if original is None:
                    plan = replan_actual_retrieval(old, snapshot, service, args)
                    if plan is None:
                        cases.append(dict(source_attempt_id=old.attempt_id,
                            status="source_plan_unavailable", reason="actual attempt has no retrieved original sources"))
                        continue
                else:
                    plan = build_read_plan(original.task_state, original.selected_set, original.coverage_scope,
                        original.budget, template=service.template, snapshot=snapshot,
                        reading_mode=original.reading_mode, separate_needs=original.separate_needs,
                        numeric_identity=original.numeric_identity, read_layout="source_first", read_packing=args.read_packing)
                accounting.current_source = dict(source_attempt_id=old.attempt_id, request_id=old.request_id,
                    dataset=metadata.get(old.attempt_id, {}).get("dataset"), split=metadata.get(old.attempt_id, {}).get("split"))
                cursor = len(recorder.calls)
                try:
                    response = await service.serve_turn(old.session_id, old.query, context_plans=(plan,),
                        corpus_snapshot_id=before.corpus_snapshot_id, request_id="source-first:"+old.request_id)
                except EngineError as error:
                    failure = await completed_capture_failure(service, recorder, cursor, error)
                    cases.append({**accounting.current_source, **failure})
                    write_json(Path(args.output_dir)/"capture-cases.json", cases)
                    continue
                actual = service.store.get_attempt(response.attempt_id)
                meta = {**metadata.get(old.attempt_id, {}), "response": to_jsonable(response),
                        "max_new_tokens": config.context.max_new_tokens,
                        "capture_origin": "replanned_actual_retrieval" if original is None else "persisted_actual_plan_recaptured_current_protocol",
                        "source_attempt_id": old.attempt_id, "scope": "study" if old.attempt_id in metadata else "unpartitioned_capture"}
                turn = export_attempt(service.store, actual, service.template, request_sampling(config.sampling),
                                      meta, recorded_calls=recorder.calls[cursor:])
                turns.append(turn)
                cases.append({**accounting.current_source, "status": "committed", "attempt_id": actual.attempt_id})
                write_json(Path(args.output_dir)/"capture-cases.json", cases)
                write_json(Path(args.output_dir)/"capture-trace.json", make_trace(turns, config=config,
                    source_db=args.state_db, captured=True, case_outcomes=cases))
                print(json.dumps({"captured": len(turns), "events": len(turn["events"]), "request_id": old.request_id}), flush=True)
        finally:
            if turns:
                write_json(Path(args.output_dir)/"capture-trace.json", make_trace(turns, config=config,
                    source_db=args.state_db, captured=True, case_outcomes=cases))
            write_json(Path(args.output_dir)/"capture-wire.json", recorder.calls)
            await service.aclose()
            service.store.close()
            source.close()
    return {**make_trace(turns, config=config, source_db=args.state_db, captured=True),
            "entry_accounting": accounting.entry_manifest}


async def capture_reader_traces(args):
    """Canonical native execution of actual-retrieval READ training inputs.

    Review wrappers and teacher targets are not copied into a C trace. The
    controller builds and executes a new source-first DAG and final ANSWER.
    """
    config = load_config(args.config)
    config = replace(config, engine=replace(config.engine, base_url=args.base_url),
        state_dir=str(Path(args.output_dir)/"state"), prefix_retention={"mode": "native"})
    async with CaptureRunAccounting(config, args.output_dir, args.gpu_device,
            resource_gpu_devices=args.resource_gpu_devices,
            resource_process_roots=args.resource_process_roots) as accounting:
        service = create_controller(config)
        resident_provider = None
        if args.study_config is not None:
            resident_provider, accounting.resident_model_manifest = await asyncio.to_thread(
                load_resident_provider, args.study_config, service.template)
            accounting.record_resident_models(args.study_config, accounting.resident_model_manifest)
        service.observe_prefix = CapturePrefixObserver(service.engine, service.template)
        recorder = RecordingEngine(service.engine)
        service.engine = recorder
        accounting.calls = recorder.calls
        turns, seen, cases = [], set(), []
        accounting.case_outcomes = cases
        try:
            for line in Path(args.reader_traces).read_text().splitlines():
                item = json.loads(line)
                if item["phase"] != "READ":
                    continue
                original = decode(ContextPlan, item["context_plan"])
                identity = item["attempt_id"], original.plan_id
                if identity in seen:
                    continue
                seen.add(identity)
                if len(cases) >= args.max_capture_turns:
                    break
                history = tuple(decode(TurnRecord, row) for row in item["actual_history"])
                session_id = "canonical:"+item["conversation_id"]+":"+original.selected_set.set_id
                snapshot = service.store.read_snapshot(session_id, item["snapshot_id"])
                service.store.sync_history(session_id, history, snapshot.version)
                snapshot = service.store.get_snapshot(session_id)
                plan = build_read_plan(original.task_state, original.selected_set, original.coverage_scope,
                    original.budget, template=service.template, snapshot=snapshot,
                    reading_mode=original.reading_mode, separate_needs=original.separate_needs,
                    numeric_identity=original.numeric_identity, read_layout="source_first", read_packing=args.read_packing)
                accounting.current_source = dict(source_attempt_id=item["attempt_id"], example_id=item["example_id"],
                    split=item["split"], source_partition=item["source_partition"])
                cursor = len(recorder.calls)
                try:
                    response = await service.serve_turn(session_id, item["question"], context_plans=(plan,),
                        corpus_snapshot_id=item["snapshot_id"], request_id=digest(identity))
                except EngineError as error:
                    failure = await completed_capture_failure(service, recorder, cursor, error)
                    cases.append({**accounting.current_source, **failure})
                    write_json(Path(args.output_dir)/"capture-cases.json", cases)
                    continue
                attempt = service.store.get_attempt(response.attempt_id)
                metadata = dict(dataset=original.selected_set.source_blocks[0].key.corpus_id,
                    split=item["split"], source_partition=item["source_partition"],
                    response=to_jsonable(response), scope="study", max_new_tokens=config.context.max_new_tokens,
                    source_example_id=item["example_id"], source_trace_origin=item["trace_origin"],
                    source_mutations=item.get("source_mutations", []),
                    capture_origin="actual_retrieval_recaptured_canonical_native_source_first")
                turn = export_attempt(service.store, attempt, service.template, request_sampling(config.sampling),
                    metadata, recorded_calls=recorder.calls[cursor:])
                turns.append(turn)
                cases.append({**accounting.current_source, "status": "committed", "attempt_id": attempt.attempt_id})
                write_json(Path(args.output_dir)/"capture-cases.json", cases)
                write_json(Path(args.output_dir)/"capture-trace.json", make_trace(turns, config=config,
                    source_db=args.reader_traces, captured=True, case_outcomes=cases))
                print(json.dumps({"captured": len(turns), "events": len(turn["events"]),
                    "origin": "canonical_native_source_first"}), flush=True)
        finally:
            if turns:
                write_json(Path(args.output_dir)/"capture-trace.json", make_trace(turns, config=config,
                    source_db=args.reader_traces, captured=True, case_outcomes=cases))
            write_json(Path(args.output_dir)/"capture-wire.json", recorder.calls)
            await service.aclose()
            service.store.close()
    return {**make_trace(turns, config=config, source_db=args.reader_traces, captured=True),
            "entry_accounting": accounting.entry_manifest}


class PrefixPageSampler:
    """Observe both ready native tiers without introducing independent refs."""
    def __init__(self, engine, *, interval=.25):
        self.runtime, self.interval = PrefixRuntimeClient(engine), interval
        self.watches, self.samples, self.errors, self.controls = {}, [], [], []
        self._stop = asyncio.Event()
        self._task = None
        self._sampling = asyncio.Lock()

    def watch(self, prefix_id, receipt, ids, sources):
        self.watches[prefix_id] = dict(receipt=receipt, prefix_input_ids=tuple(ids), sources=sources)

    async def sample(self):
        async with self._sampling:
            prefixes, started = {}, time.monotonic()
            for prefix_id, kwargs in tuple(self.watches.items()):
                tick = time.monotonic()
                view = await self.runtime.inspect(**kwargs)
                done = time.monotonic()
                self.controls.append(dict(operation="residency_inspection", prefix_id=prefix_id,
                    start=tick, end=done, elapsed_ms=(done-tick)*1000))
                prefixes[prefix_id] = view
            self.samples.append(dict(timestamp=(started+time.monotonic())/2, prefixes=prefixes))

    async def start(self):
        await self.sample()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
                return
            except TimeoutError:
                await self.sample()

    async def stop(self):
        self._stop.set()
        await self._task
        await self.sample()

    def observation(self):
        return dict(clock="client_monotonic", interval_seconds=self.interval,
            page_basis="native inspected ready causal pages; inflight markers retained",
            samples=self.samples, controls=self.controls, errors=self.errors)


def integrate_prefix_residency(observation, prefix_id, start, end, page_size):
    """Integrate one observed prefix, including allocated transfer destinations."""
    samples = observation["samples"]
    before = [row for row in samples if row["timestamp"] <= start and prefix_id in row["prefixes"]]
    after = [row for row in samples if row["timestamp"] >= end and prefix_id in row["prefixes"]]
    if not before or not after:
        return dict(complete_window_coverage=False, reason="missing_native_residency_boundary")
    rows = [{**before[-1], "timestamp": start},
        *(row for row in samples if start < row["timestamp"] < end), {**after[0], "timestamp": end}]
    pending = any(row["prefixes"][prefix_id]["residency"][key]
        for row in rows for key in ("inflight_h2d_tokens", "inflight_d2h_tokens"))
    costs = {}
    for tier in ("gpu", "host"):
        ready_integral = pending_integral = 0.0
        transfer = "inflight_h2d_tokens" if tier == "gpu" else "inflight_d2h_tokens"
        for row, following in zip(rows, rows[1:]):
            prefix = row["prefixes"][prefix_id]
            elapsed = following["timestamp"]-row["timestamp"]
            ready_integral += len(set(prefix[tier+"_page_ids"]))*page_size*elapsed
            pending_integral += prefix["residency"][transfer]*elapsed
        costs[tier+"_ready_token_seconds"] = ready_integral
        costs[tier+"_pending_token_seconds"] = pending_integral
        costs[tier+"_token_seconds"] = ready_integral+pending_integral
    return dict(complete_window_coverage=True, transfer_pending_observed=pending,
        cost_basis="sampled_single_prefix_ready_pages_plus_allocated_transfer_destinations", **costs)


def trace_generations(trace):
    return [(turn, event) for turn in trace["turns"] for event in turn["events"] if event["kind"] != "CPU_REDUCE"]


def read_decisions(trace):
    return [(turn, event) for turn, event in trace_generations(trace) if event["kind"] == "READ"]


def event_id(turn, event):
    return turn["attempt_id"]+":"+event["step_id"]


def actual_page_size(reads, declared=None):
    sizes = {event["prefix_observation"]["inspection"]["page_size"] for _, event in reads
        if event.get("prefix_observation") and "inspection" in event["prefix_observation"]}
    if declared is not None:
        sizes.add(declared)
    if len(sizes) != 1:
        raise ValueError("coverage needs one actual observed native page_size or its explicit declaration")
    return sizes.pop()


def verify_trace(trace):
    if trace["format"] != "contextkv_source_first_trace_v2" or trace["layout"] != "source_first":
        raise ValueError("C replay requires an actually executed canonical source_first trace")
    events = trace_generations(trace)
    if len({event_id(turn, event) for turn, event in events}) != len(events):
        raise ValueError("trace has duplicated model-step identities")
    if [event["input_hash"] for _, event in events] != trace["input_hashes"]:
        raise ValueError("trace generation order/hash manifest changed")
    for _, event in events:
        if token_hash(event["input_ids"]) != event["input_hash"]:
            raise ValueError("trace token input differs from its captured native hash")
        if event["original_receipt"]["input_hash"] != event["input_hash"]:
            raise ValueError("trace input lacks its real original execution proof")
        if tuple(event["request"]["input_ids"]) != tuple(event["input_ids"]):
            raise ValueError("wire generation input differs from the fixed trace")
        if event["kind"] == "READ" and event["input_ids"][:len(event["prefix_input_ids"])] != event["prefix_input_ids"]:
            raise ValueError("retained source prefix is not the actual contiguous input path")


async def replay_arm(trace, config, output, *, gpu_device, intervention=None, horizon_seconds=180,
                     ttl_seconds=None, residency_interval=.25, target=None):
    """Replay the complete supplied window; intervention names one READ event."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = ConfigFingerprint(**trace["fingerprint"])
    config.model.scope.verify_fingerprint(asdict(fingerprint))
    recipes = {digest(event["sampling"]): event["sampling"] for _, event in trace_generations(trace)}
    engines = {key: SGLangEngineClient(config.engine.base_url, fingerprint,
        timeout=config.engine.timeout_seconds, stream=True,
        sampling=bound_recipe(recipe),
        **({"model_scope": config.model.scope.manifest()} if config.model.scope else {}))
        for key, recipe in recipes.items()}
    engine = next(iter(engines.values()))
    await engine.profile_prepare("cold")
    info = await engine.info()
    for client in engines.values():
        await client.info()
    runtime = PrefixRuntimeClient(engine)
    pages = PrefixPageSampler(engine, interval=residency_interval)
    pool = (await engine.metrics())["pool"]
    page_size = pool["page_size"]
    cache_scope_hash = digest([asdict(fingerprint), info.cache_config_hash])
    native = NativePoolSampler(config.engine.base_url, gpu_capacity=config.engine.gpu_capacity_tokens,
        host_enabled=config.engine.host_capacity_tokens > 0, model=config.model.name, interval=.1)
    physical = StudyResourceSampler((gpu_device,), listening_processes((urlsplit(config.engine.base_url).port,)), interval=.1)
    await native.start()
    await physical.start()
    await pages.start()
    epochs, previous_answers, accesses = {}, {}, defaultdict(list)
    locks = defaultdict(asyncio.Lock)
    receipts, controls, turn_windows = {}, [], {}
    start = time.monotonic()
    source_origin = min(event["source_started_s"] for _, event in trace_generations(trace))
    active_owners = set()
    report = dict(source="hardware", scope="fixed_reading_dag_replay", intervention=intervention,
        decision_target=target,
        horizon_seconds=horizon_seconds, ttl_seconds=ttl_seconds,
        engine=to_jsonable(info), config=to_jsonable(config),
        cache_scope_hash=cache_scope_hash, page_size=page_size,
        trace_digest=digest(trace), input_dag_digest=trace["input_dag_digest"], events=receipts,
        output_policy="fresh responses measure actual native cost; captured outputs preserve the fixed DAG",
        parse_policy="captured_original_outputs",
        quality_scope="fixed-input cost attribution only; fresh outputs do not establish new QA quality",
        timing_policy="captured per-turn input-ready arrivals; internal DAG executes causally without fixed model-submit stalls",
        cost_scope="reading DAG/ANSWER, C controls and declared physical process groups; upstream A is frozen",
        status="running", controls=controls)
    report["turn_windows"] = turn_windows

    async def retire(original_id):
        epoch = epochs[original_id]
        tick = time.monotonic()
        ack = await engine.retire(epoch.engine_session_id, epoch.session_generation,
            uuid.uuid4().hex, info.engine_restart_id, "fixed_trace_epoch_lifecycle")
        controls.append(dict(operation="business_retire", start=tick, end=time.monotonic(),
            elapsed_ms=(time.monotonic()-tick)*1000, acknowledged=ack.acknowledged))

    async def execute_turn(turn):
        arrival = min(event["source_started_s"] for event in turn["events"] if event["kind"] != "CPU_REDUCE")
        deadline = start+arrival-source_origin
        await asyncio.sleep(max(0, deadline-time.monotonic()))
        turn_windows[turn["attempt_id"]] = dict(start=deadline, dispatched=time.monotonic())
        async with locks[turn["session_id"]]:
            plan = decode(ContextPlan, turn["context_plan"])
            for event in turn["events"]:
                identity = event_id(turn, event)
                if event["kind"] == "CPU_REDUCE":
                    tick = time.monotonic()
                    result = reduce_claims(event["operation"], tuple(decode(EvidenceClaim, value)
                        for value in event["input_claims"]), claim_id=event["claim_id"], need_id=event["need_id"])
                    receipts[identity] = dict(kind="CPU_REDUCE", result=to_jsonable(result),
                        window={"start": tick, "end": time.monotonic()}, elapsed_ms=(time.monotonic()-tick)*1000)
                    continue
                begun = time.monotonic()
                original_epoch = event["original_epoch"]["engine_session_id"]
                if original_epoch not in epochs:
                    epochs[original_epoch] = await engine.prepare(uuid.uuid4().hex, uuid.uuid4().hex,
                        info.engine_restart_id,
                        **({"business_session_id": turn["session_id"]}
                           if info.business_session_identity_supported else {}))
                epoch = epochs[original_epoch]
                client = engines[digest(event["sampling"])]
                submitted = time.monotonic()
                native_attempt = uuid.uuid4().hex
                receipts[identity] = dict(state="SUBMITTED", kind=event["kind"], input_hash=event["input_hash"],
                    native_attempt_id=native_attempt, engine_session_id=epoch.engine_session_id,
                    session_generation=epoch.session_generation, engine_restart_id=info.engine_restart_id,
                    window={"start": begun})
                write_json(output/"progress.json", receipts[identity])
                result = await client.generate(**event["request"], session_id=epoch.engine_session_id,
                    expected_generation=epoch.session_generation, expected_restart_id=info.engine_restart_id,
                    attempt_id=native_attempt, fingerprint=fingerprint)
                if result.receipt.input_hash != event["input_hash"]:
                    raise ValueError("paired replay changed the actual generation input")
                row = dict(state="COMPLETE", kind=event["kind"], input_hash=result.receipt.input_hash,
                    native_attempt_id=native_attempt,
                    input_tokens=len(event["input_ids"]), output_budget=event["request"]["max_new_tokens"],
                    original_output_tokens=len(event["original_output_ids"]), actual_output_tokens=len(result.output_ids),
                    result=to_jsonable(result), sampling=event["sampling"],
                    generation_window={"start": submitted, "end": time.monotonic()},
                    window={"start": begun}, source_started_s=event["source_started_s"])
                receipts[identity] = row
                if event["kind"] != "ANSWER":
                    tick = time.monotonic()
                    task = decode(ReadTask, event["read_task"])
                    by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
                    parsed = parse_read_result(task, event["original_output_text"],
                        tuple(by_id[key] for key in task.input_unit_ids),
                        tuple(decode(EvidenceClaim, value) for value in event["input_claims"]),
                        elapsed_ms=result.elapsed_ms, input_tokens=len(event["input_ids"]),
                        output_tokens=len(event["original_output_ids"]))
                    row.update(parse_ms=(time.monotonic()-tick)*1000, parsed_output=to_jsonable(parsed),
                               parse_policy="captured_original_outputs")
                if event["kind"] == "READ":
                    prefix_ids = event["prefix_input_ids"][:len(event["prefix_input_ids"])//page_size*page_size]
                    inspected = dict(cache_scope_hash=cache_scope_hash, prefix_hash=token_hash(prefix_ids),
                        boundary_tokens=len(prefix_ids), page_size=page_size,
                        observation="actual executed causal identity; no residency asserted")
                    prefix_id = "trace-prefix:"+digest([turn["session_id"], cache_scope_hash, inspected["prefix_hash"]])
                    if target is not None and identity == target["source_event_id"]:
                        pages.watch(prefix_id, result.receipt, event["prefix_input_ids"], event["sources"])
                        await pages.sample()
                    features = PrefixReadObserver._features(plan, event["sources"], accesses[prefix_id], time.monotonic_ns())
                    accesses[prefix_id].append(dict(observed_ns=time.monotonic_ns(), payload={"turn_id": plan.task_state.current_turn_id}))
                    row.update(prefix_id=prefix_id, inspection=inspected, raw_features=features,
                        decision_time=time.monotonic(), retention_control_ms=0)
                    if target is not None and identity == target["event_id"]:
                        source_turn, source_event = next((other_turn, other_event) for other_turn, other_event in read_decisions(trace)
                            if event_id(other_turn, other_event) == target["source_event_id"])
                        source_row = receipts[target["source_event_id"]]
                        source_receipt = decode(type(result.receipt), source_row["result"]["receipt"])
                        tick = time.monotonic()
                        target_view = await runtime.inspect(receipt=source_receipt,
                            prefix_input_ids=source_event["prefix_input_ids"], sources=source_event["sources"])
                        target_inspect_ms = (time.monotonic()-tick)*1000
                        await pages.sample()
                        target_prefix_id = source_row["prefix_id"]
                        target_features = features if identity == target["source_event_id"] else PrefixReadObserver._features(
                            plan, source_event["sources"], accesses[target_prefix_id], time.monotonic_ns())
                        target_features.update(PrefixReadObserver._native_features(target_view))
                        decision = row["decision_target"] = dict(prefix_id=target_prefix_id, inspection=target_view,
                            raw_features=target_features, decision_time=time.monotonic(), source_event_id=target["source_event_id"],
                            source_session_id=source_turn["session_id"], retention_control_ms=0,
                            action_feasible=target_view["residency"]["recoverable_prefix_tokens"] >= target_view["boundary_tokens"])
                    if identity == intervention and row["decision_target"]["action_feasible"]:
                        decision = row["decision_target"]
                        tick = time.monotonic()
                        decision["retain_receipt"] = await runtime.retain(owner_id=decision["prefix_id"], receipt=source_receipt,
                            prefix_input_ids=source_event["prefix_input_ids"], sources=source_event["sources"], ttl_seconds=ttl_seconds)
                        decision["retention_control_ms"] = (time.monotonic()-tick)*1000+target_inspect_ms
                        active_owners.add(decision["prefix_id"])
                if event["kind"] == "ANSWER":
                    previous = previous_answers.get(turn["session_id"])
                    if previous is not None and previous != original_epoch:
                        await retire(previous)
                    previous_answers[turn["session_id"]] = original_epoch
                else:
                    await retire(original_epoch)
                row["window"]["end"] = time.monotonic()
                row["step_wall_ms"] = (row["window"]["end"]-begun)*1000
                write_json(output/"progress.json", dict(status="running", completed_events=len(receipts),
                    last_event=identity, intervention=intervention))
            turn_windows[turn["attempt_id"]]["end"] = time.monotonic()

    ordered_turns = sorted(trace["turns"], key=lambda turn: min(event["source_started_s"]
        for event in turn["events"] if event["kind"] != "CPU_REDUCE"))
    tasks = [asyncio.create_task(execute_turn(turn)) for turn in ordered_turns]
    try:
        await asyncio.gather(*tasks)
        decision_time = receipts[target["event_id"]]["decision_target"]["decision_time"]
        observation_end = decision_time+horizon_seconds
        await asyncio.sleep(max(0, observation_end-time.monotonic()))
        await pages.sample()
        report["observation_window"] = dict(start=decision_time, end=observation_end)
        report["status"] = "completed"
    finally:
        failure = sys.exc_info()[1]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        workload_end = time.monotonic()
        if failure is not None:
            report.update(status="failed", error=str(failure))
        for prefix_id in active_owners:
            tick = time.monotonic()
            await runtime.release(prefix_id, info.engine_restart_id)
            controls.append(dict(operation="experiment_prefix_cleanup", elapsed_ms=(time.monotonic()-tick)*1000))
        for original_id in epochs:
            await retire(original_id)
        end = time.monotonic()
        await pages.stop()
        await native.stop()
        await physical.stop()
        for client in engines.values():
            await client.aclose()
        report.update(resource_window={"start": start, "end": end}, workload_end=workload_end,
            physical_cost=integrate_physical(physical.samples, start, end),
            native_observation=native.observation(), physical_observation=physical.observation(),
            prefix_observation=pages.observation())
        series, coverage = pool_series(native.observation(), start, end)
        report["native_cost"] = {"integrals": occupancy_integral(series), "coverage": coverage}
        if "observation_window" in report:
            window = report["observation_window"]
            report["observation_physical_cost"] = integrate_physical(physical.samples, window["start"], window["end"])
            series, coverage = pool_series(native.observation(), window["start"], window["end"])
            report["observation_native_cost"] = dict(integrals=occupancy_integral(series), coverage=coverage)
        write_json(output/"report.json", report)
    return report


def choose_windows(trace, *, horizon_seconds, max_candidates, history_steps, max_replay_steps, seed,
                   max_prefixes_per_decision=1, page_size=None):
    """Select bounded contiguous windows; never remove intervening pressure."""
    turns = sorted(trace["turns"], key=lambda turn: min(event["source_started_s"]
        for event in turn["events"] if event["kind"] != "CPU_REDUCE"))
    read_events = sorted(((index, turn, event) for index, turn in enumerate(turns)
                          for event in turn["events"] if event["kind"] == "READ"),
                         key=lambda item: item[2]["source_completed_s"])
    page_size = actual_page_size([(turn, event) for _, turn, event in read_events], page_size)
    candidates, omitted, known = [], [], defaultdict(dict)
    for index, turn, event in read_events:
        signature = digest(event["prefix_input_ids"])
        _, body_pages, _ = causal_pages(event, page_size, "")
        if body_pages:
            known[turn["session_id"]][signature] = (index, turn, event)
        else:
            omitted.append(dict(source_event_id=event_id(turn, event),
                reason="no_complete_source_body_page", reuse_label=None))
        targets = sorted(known[turn["session_id"]].values(),
                         key=lambda item: item[2]["source_completed_s"], reverse=True)
        targets = targets[:max_prefixes_per_decision]
        first = index
        count = sum(len(item["events"]) for item in turns[first:index])
        while first > 0 and count < history_steps:
            first -= 1
            count += len(turns[first]["events"])
        deadline = event["source_completed_s"]+horizon_seconds
        last = index
        while last < len(turns) and min(item["source_started_s"] for item in turns[last]["events"]
                if item["kind"] != "CPU_REDUCE") <= deadline:
            last += 1
        if last == len(turns):
            omitted.append(dict(event_id=event_id(turn, event), reason="no_observed_post_horizon_turn"))
            continue
        # Keep the first whole turn after the horizon, providing a real
        # boundary observation and its pressure, not an invented idle tail.
        for source_index, source_turn, source_event in targets:
            begin = min(first, source_index)
            window = turns[begin:last+1]
            window_start = min(item["source_started_s"] for value in window for item in value["events"]
                               if item["kind"] != "CPU_REDUCE")
            window_end = max(item["source_completed_s"] for value in window for item in value["events"]
                             if item["kind"] != "CPU_REDUCE")
            if any(gap["start"] < window_end and gap["end"] > window_start
                   for gap in trace.get("unexported_pressure_intervals", ())):
                omitted.append(dict(event_id=event_id(turn, event), source_event_id=event_id(source_turn, source_event),
                    reason="unexported_request_pressure_overlaps_replay_window"))
                continue
            steps = sum(len(item["events"]) for item in window)
            if steps > max_replay_steps:
                omitted.append(dict(event_id=event_id(turn, event), source_event_id=event_id(source_turn, source_event),
                    reason="window_exceeds_replay_step_budget", steps=steps))
                continue
            identity, source_identity = event_id(turn, event), event_id(source_turn, source_event)
            candidates.append(dict(event_id=identity, source_event_id=source_identity,
                first_turn=begin, last_turn=last, steps=steps,
                observation_seconds=horizon_seconds,
                source_observation_window={"start": event["source_completed_s"], "end": deadline},
                observed_boundary_turn=turns[last]["attempt_id"],
                source_attempts=[item["attempt_id"] for item in window],
                priority=digest([seed, identity, source_identity]), turns=window))
    candidates.sort(key=lambda item: item["priority"])
    selected = candidates[:max_candidates]
    for candidate in candidates[max_candidates:]:
        omitted.append(dict(event_id=candidate["event_id"], reason="candidate_budget"))
    probability = min(1, max_candidates/len(candidates)) if candidates else 0
    for candidate in selected:
        candidate["sampling_probability"] = probability
        candidate["sampling_weight"] = 1/probability
    return selected, omitted


def window_trace(trace, selected):
    result = {**trace, "turns": selected["turns"], "window_selection": {
        key: value for key, value in selected.items() if key != "turns"}}
    result["input_hashes"] = [event["input_hash"] for _, event in trace_generations(result)]
    result["input_dag_digest"] = digest([[turn["attempt_id"],
        [(event["step_id"], event.get("input_hash"), event["predecessor_step_ids"]) for event in turn["events"]]]
        for turn in result["turns"]])
    return result



def exact_target_demand(target, future, *, page_size, scope):
    key = source_demand_event_key(target["prefix_input_ids"], target["evidence_token_ranges"], page_size, scope)
    return key is not None and key == source_demand_event_key(
        future["prefix_input_ids"], future["evidence_token_ranges"], page_size, scope)

def paired_row(trace, native, retained, selection, *, horizon_seconds):
    identity = selection["event_id"]
    left, right = native["events"][identity]["decision_target"], retained["events"][identity]["decision_target"]
    if native["input_dag_digest"] != retained["input_dag_digest"] or native["input_dag_digest"] != trace["input_dag_digest"]:
        raise ValueError("paired C reading DAG differs")
    left_hashes, right_hashes = [], []
    for turn, event in trace_generations(trace):
        key = event_id(turn, event)
        left_hashes.append(native["events"][key]["input_hash"])
        right_hashes.append(retained["events"][key]["input_hash"])
    if left_hashes != right_hashes or left_hashes != trace["input_hashes"]:
        raise ValueError("paired C actual model input hashes differ")
    if left["inspection"]["cache_scope_hash"] != right["inspection"]["cache_scope_hash"]:
        raise ValueError("paired C reader/cache resource scope differs")
    source_events = {event_id(turn, event): event for turn, event in read_decisions(trace)}
    target_event = source_events[selection["source_event_id"]]
    uses, future = [], []
    for key, actual in native["events"].items():
        if key not in source_events or actual["generation_window"]["start"] <= left["decision_time"]:
            continue
        scope, size = left["inspection"]["cache_scope_hash"], left["inspection"]["page_size"]
        left_use = reuse_pages(target_event, source_events[key], actual["result"]["receipt"], page_size=size, scope=scope)
        right_use = reuse_pages(target_event, source_events[key], retained["events"][key]["result"]["receipt"], page_size=size, scope=scope)
        uses.append(dict(event_id=key, native=left_use, retained=right_use))
        if exact_target_demand(target_event, source_events[key], page_size=size, scope=scope):
            future.append((key, actual))
    future.sort(key=lambda item: item[1]["generation_window"]["start"])
    next_key, next_left = future[0] if future else (None, None)
    next_right = retained["events"][next_key] if next_key is not None else None
    reuse_seconds = next_left["generation_window"]["start"]-left["decision_time"] if next_left else None
    reused = reuse_seconds is not None and reuse_seconds <= horizon_seconds
    ends = [left["decision_time"]+horizon_seconds, right["decision_time"]+horizon_seconds]
    left_residency = integrate_prefix_residency(native["prefix_observation"], left["prefix_id"],
        left["decision_time"], ends[0], left["inspection"]["page_size"])
    right_residency = integrate_prefix_residency(retained["prefix_observation"], right["prefix_id"],
        right["decision_time"], ends[1], right["inspection"]["page_size"])
    eligible = all(value["complete_window_coverage"] for value in (left_residency, right_residency))
    eligible = eligible and native["status"] == retained["status"] == "completed"
    eligible = eligible and left["action_feasible"] and right["action_feasible"] and "retain_receipt" in right
    body_pages = len(causal_pages(target_event, left["inspection"]["page_size"], left["inspection"]["cache_scope_hash"])[1])
    eligible = eligible and body_pages > 0
    eligible = eligible and all(arm["physical_cost"]["complete_window_coverage"] and
        arm["native_cost"]["coverage"]["complete_window_coverage"] for arm in (native, retained))
    turn = next(turn for turn in trace["turns"] if any(event_id(turn, event) == identity for event in turn["events"]))
    source_event = next(event for event in turn["events"] if event_id(turn, event) == identity)
    source_start = source_event["source_started_s"]
    source_end = source_event["source_completed_s"]+horizon_seconds
    affected = [value for value in trace["turns"] if min(event["source_started_s"] for event in value["events"]
        if event["kind"] != "CPU_REDUCE") <= source_end and max(event["source_completed_s"] for event in value["events"]
        if event["kind"] != "CPU_REDUCE") >= source_start]
    affected_ids = [value["attempt_id"] for value in affected]
    window_costs = {arm: sum((report["turn_windows"][key]["end"]-report["turn_windows"][key]["start"])*1000
                            for key in affected_ids) for arm, report in (("native", native), ("retained", retained))}
    target_delta = next_left["step_wall_ms"]-next_right["step_wall_ms"] if reused else 0
    window_delta = window_costs["native"]-window_costs["retained"]
    row = dict(source="hardware", scope="actual_source_first_single_prefix_intervention",
        target_demand_rule=TARGET_DEMAND_RULE,
        dataset=turn["dataset"], split=turn["split"], training_eligible=trace["training_eligible"] and eligible,
        event_id=identity, source_event_id=selection["source_event_id"],
        independent_session_ids=sorted({value["session_id"] for value in trace["turns"]}),
        independent_sample_id=digest(sorted({value["session_id"] for value in trace["turns"]})),
        sample_unit="independent session group; overlapping windows and TTL arms are repeated measures",
        measurement_window_id=digest(selection["source_attempts"]),
        source_body_page_count=body_pages,
        predicted_reuse_event_key=source_demand_event_key(target_event["prefix_input_ids"],
            target_event["evidence_token_ranges"], left["inspection"]["page_size"],
            left["inspection"]["cache_scope_hash"]),
        partition_eligible=trace["training_eligible"], measurement_eligible=eligible,
        features=left["raw_features"], retained_features=right["raw_features"],
        cache_scope_hash=left["inspection"]["cache_scope_hash"], fingerprint=trace["fingerprint"],
        horizon_seconds=horizon_seconds, reused=reused, reuse_interval_seconds=reuse_seconds,
        causal_reuse=uses,
        native_input_hashes=left_hashes, retained_input_hashes=right_hashes,
        input_dag_digest=trace["input_dag_digest"], sampling_weight=selection["sampling_weight"],
        native_reuse_cost_ms=next_left["step_wall_ms"] if reused else None,
        retained_reuse_cost_ms=next_right["step_wall_ms"] if reused else None,
        native_window_cost_ms=window_costs["native"], retained_window_cost_ms=window_costs["retained"],
        full_window_cost_difference_ms=window_delta, target_revisit_cost_difference_ms=target_delta,
        collateral_and_initial_control_cost_difference_ms=window_delta-target_delta,
        affected_turn_ids=affected_ids,
        window_cost_basis="sum of complete component response intervals for all fixed affected turns, including collateral reads/ANSWER and policy controls",
        reuse_event_basis="target reward requires exact complete source-demand key; partial causal body-page demand remains in causal_reuse and whole-window costs",
        retention_control_ms=right["retention_control_ms"],
        native_reuse_result=next_left["result"] if reused else None,
        retained_reuse_result=next_right["result"] if reused else None,
        native_prefix_residency=left_residency, retained_prefix_residency=right_residency,
        page_size=left["inspection"]["page_size"],
        gpu_page_count=left["inspection"]["boundary_tokens"]//left["inspection"]["page_size"],
        host_page_count=(left["inspection"]["boundary_tokens"]//left["inspection"]["page_size"]
            if left["inspection"]["pool"]["host_capacity_tokens"] else 0),
        shared_page_rule="one treated owner per replay; marginal against other independently protected owners (empty); union diagnostics counted once",
        physical_cost={"native": native["physical_cost"], "retained": retained["physical_cost"]},
        complete_controls={"native": native["controls"], "retained": retained["controls"]})
    if eligible:
        row.update(native_gpu_token_seconds=left_residency["gpu_token_seconds"],
            retained_gpu_token_seconds=right_residency["gpu_token_seconds"],
            native_host_token_seconds=left_residency["host_token_seconds"],
            retained_host_token_seconds=right_residency["host_token_seconds"])
    return row


def revisit_outcomes(trace, report, selection):
    """Classify actual source-page demand inside the common observation window."""
    events = {event_id(turn, event): event for turn, event in read_decisions(trace)}
    target = events[selection["source_event_id"]]
    body_pages = len(causal_pages(target, report["page_size"], report["cache_scope_hash"])[1])
    decision = report["events"][selection["event_id"]]["decision_target"]
    window = report["observation_window"]
    counts = dict(gpu_hit=0, host_load=0, recomputed=0)
    visits = []
    for identity, future in events.items():
        actual = report["events"][identity]
        if not window["start"] < actual["generation_window"]["start"] <= window["end"]:
            continue
        use = reuse_pages(target, future, actual["result"]["receipt"],
            page_size=report["page_size"], scope=report["cache_scope_hash"])
        demanded = set(use["shared_evidence_page_ids"])
        if not demanded:
            continue
        gpu = set(use["actual_device_evidence_page_ids"])
        host = set(use["actual_host_evidence_page_ids"])
        recomputed = demanded-gpu-host
        pages = dict(gpu_hit=len(gpu), host_load=len(host), recomputed=len(recomputed))
        for category, count in pages.items():
            counts[category] += count
        visits.append(dict(event_id=identity,
            seconds_after_decision=actual["generation_window"]["start"]-decision["decision_time"],
            categories=[category for category, count in pages.items() if count], page_counts=pages,
            demanded_page_ids=sorted(demanded), recomputed_page_ids=sorted(recomputed),
            step_wall_ms=actual["step_wall_ms"], receipt=actual["result"]["receipt"]))
    return dict(source_body_page_count=body_pages, no_revisit=not visits if body_pages else None,
        revisit_events=len(visits), page_counts=counts,
        event_counts={category: sum(category in visit["categories"] for visit in visits) for category in counts},
        visits=visits, observation_seconds=window["end"]-window["start"],
        classification_basis="actual demanded original-source causal pages and native device/host receipt boundaries; mixed visits retain every category")


async def replay(args):
    trace = json.loads(Path(args.trace).read_text())
    verify_trace(trace)
    config = load_config(args.config)
    page_size = args.page_size if args.page_size is not None else config.engine.page_size
    ttls = tuple(args.ttl_seconds)
    if args.horizon_seconds < max(180, *ttls):
        raise ValueError("all replay arms require one observation window covering at least 180 seconds and every TTL")
    selected, omitted = choose_windows(trace, horizon_seconds=args.horizon_seconds,
        max_candidates=args.max_candidates, history_steps=args.history_steps,
        max_replay_steps=args.max_replay_steps, seed=args.seed,
        max_prefixes_per_decision=args.max_prefixes_per_decision, page_size=page_size)
    arms = {"native": None, **{f"ttl{ttl:g}": ttl for ttl in ttls}}
    planned = sum(len(arms)*len(trace_generations(window_trace(trace, item))) for item in selected)
    if planned > args.max_total_generations:
        raise ValueError(f"selected replay needs {planned} model calls; declared budget is {args.max_total_generations}")
    config = replace(config, engine=replace(config.engine, base_url=args.base_url))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"selection.json", dict(source_trace_digest=digest(trace),
        horizon_seconds=args.horizon_seconds, ttl_seconds=ttls, arms=arms, max_candidates=args.max_candidates,
        max_prefixes_per_decision=args.max_prefixes_per_decision,
        history_steps=args.history_steps, max_replay_steps=args.max_replay_steps,
        max_total_generations=args.max_total_generations, planned_generations=planned,
        selected=[{key: value for key, value in item.items() if key != "turns"} for item in selected],
        omitted=omitted, initial_cache="explicit cold start of each contiguous recorded workload window",
        pressure="same complete request/DAG window for all TTL arms, including intervening pressure and observed post-horizon boundary"))
    rows = []
    for index, item in enumerate(selected):
        local_trace = window_trace(trace, item)
        directory = output/f"pair-{index:04d}"
        write_json(directory/"trace.json", local_trace)
        reports = {}
        order = sorted(arms, key=lambda arm: digest([args.seed, item["event_id"], item["source_event_id"], arm]))
        for arm in order:
            reports[arm] = await replay_arm(local_trace, config, directory/arm,
                gpu_device=args.gpu_device, intervention=item["event_id"] if arms[arm] is not None else None,
                horizon_seconds=args.horizon_seconds, ttl_seconds=arms[arm], residency_interval=args.residency_interval,
                target={"event_id": item["event_id"], "source_event_id": item["source_event_id"]})
        outcomes = {arm: revisit_outcomes(local_trace, report, item) for arm, report in reports.items()}
        write_json(directory/"revisit-outcomes.json", outcomes)
        for arm, ttl in arms.items():
            if ttl is None:
                continue
            row = paired_row(local_trace, reports["native"], reports[arm], item, horizon_seconds=args.horizon_seconds)
            row.update(observation_seconds=args.horizon_seconds, ttl_seconds=ttl, horizon_seconds=ttl,
                retention_arm=arm, reuse_outcomes={"native": outcomes["native"], "retained": outcomes[arm]},
                observation_physical_cost={key: reports[name]["observation_physical_cost"]
                    for key, name in (("native", "native"), ("retained", arm))},
                reports={key: str((directory/name/"report.json").resolve())
                    for key, name in (("native", "native"), ("retained", arm))},
                report_digests={"native": digest(reports["native"]), "retained": digest(reports[arm])},
                arm_order=order)
            rows.append(row)
            write_json(directory/f"paired-row-{arm}.json", row)
        (output/"paired-rows.jsonl").write_text("".join(json.dumps(value, ensure_ascii=False)+"\n" for value in rows))
        print(json.dumps({"window": index, "arms": order, "outcomes": {
            arm: {key: value for key, value in outcome.items() if key not in {"visits", "classification_basis"}}
            for arm, outcome in outcomes.items()}}), flush=True)
    write_json(output/"manifest.json", dict(source="hardware", scope="same_window_multi_TTL_single_prefix_interventions",
        trace_digest=digest(trace), input_dag_digest=trace["input_dag_digest"], windows=len(selected), pairs=len(rows),
        observation_seconds=args.horizon_seconds, ttl_seconds=ttls,
        eligible_pairs=sum(row["training_eligible"] for row in rows),
        planned_generations=planned, training_data=str((output/"paired-rows.jsonl").resolve()),
        cost_rule="native report is a shared comparator, charged once per replay window; TTL rows are paired contrasts, not additive execution costs",
        value_claim="no quality claim; measured signed costs may favor native caching"))
    return rows


def reassess(args):
    """Recompute paired labels from saved native reports without model calls."""
    root = Path(args.replay_root)
    by_split, by_pair = defaultdict(list), defaultdict(dict)
    for path in sorted(root.rglob("paired-row-*.json")):
        previous = json.loads(path.read_text())
        trace = json.loads((path.parent/"trace.json").read_text())
        reports = {arm: json.loads(Path(source).read_text()) for arm, source in previous["reports"].items()}
        for arm, report in reports.items():
            if digest(report) != previous["report_digests"][arm]:
                raise ValueError("paired source report changed after collection")
        selection = trace["window_selection"]
        derived = paired_row(trace, reports["native"], reports["retained"], selection,
                             horizon_seconds=previous["observation_seconds"])
        row = {**previous, **derived, "horizon_seconds": previous["ttl_seconds"],
            "reuse_outcomes": {arm: revisit_outcomes(trace, report, selection) for arm, report in reports.items()}}
        write_json(path, row)
        by_split[path.parent.parent].append(row)
        by_pair[path.parent]["native"] = row["reuse_outcomes"]["native"]
        by_pair[path.parent][row["retention_arm"]] = row["reuse_outcomes"]["retained"]
    for directory, outcomes in by_pair.items():
        write_json(directory/"revisit-outcomes.json", outcomes)
    summary = []
    for directory, rows in by_split.items():
        (directory/"paired-rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
        manifest = json.loads((directory/"manifest.json").read_text())
        manifest.update(eligible_pairs=sum(row["training_eligible"] for row in rows),
            postprocessing=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model_calls=0, basis="original fixed-DAG reports; single-prefix allocated residency includes transfer destinations"))
        write_json(directory/"manifest.json", manifest)
        summary.append(dict(directory=str(directory), pairs=len(rows), eligible_pairs=manifest["eligible_pairs"]))
    print(json.dumps(summary))
    return summary


def fit(args):
    rows = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    accepted, excluded, frequency = [], [], []
    for row in rows:
        if row["source"] != "hardware" or row["scope"] != "actual_source_first_single_prefix_intervention":
            raise ValueError("retention assets consume actual conversation interventions, not qualification fixtures")
        for arm, path in row["reports"].items():
            report = json.loads(Path(path).read_text())
            if digest(report) != row["report_digests"][arm]:
                raise ValueError("paired source report changed after collection")
        if row["split"] == args.split and row["partition_eligible"] and row["source_body_page_count"]:
            frequency.append(row)
        if row["split"] != args.split or not row["training_eligible"]:
            excluded.append(dict(event_id=row["event_id"], split=row["split"],
                source_body_page_count=row["source_body_page_count"], training_eligible=row["training_eligible"]))
            continue
        accepted.append(row)
    model = EmpiricalRetentionModel.fit(accepted, feature_names=args.features, reuse_rows=frequency)
    asset = model.to_dict()
    asset["training_manifest"] = dict(split=args.split, source_path=str(Path(args.records).resolve()),
        source_sha256=hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
        accepted=len(accepted), independent_samples=model.independent_samples,
        sample_unit="session group; TTL arms are paired interventions", excluded=excluded, source="real_native_fixed_DAG_single_prefix_pairs",
        protocol="all real captured inputs and original budgets; fresh outputs are not matched to gold")
    write_json(args.output, asset)
    print(json.dumps({"trained_profiles": len(model.forecasts), "accepted_pairs": len(accepted),
                      "excluded_pairs": len(excluded), "output": args.output}))
    return asset


def unexported_requests(records_path, failures_path, selected, segments):
    def outcome(row, attempt_id, path, reason):
        start, end = row["request_window"]
        return dict(reason=reason, request_window=dict(start=start, end=end,
            scope=row.get("failure_window_scope", "request_wall"),
            attempt_id=attempt_id, task_id=row["task_id"], source_path=str(Path(path).resolve()),
            measurement_segment_id=row["measurement_segment_id"]))

    with Path(records_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status", "completed") != "completed":
                continue
            attempt_id = row["response"]["attempt_id"]
            if attempt_id not in selected and row["measurement_segment_id"] in segments:
                yield outcome(row, attempt_id, records_path, "completed_request_not_selected")
    for row in json.loads(Path(failures_path).read_text()):
        if row["measurement_segment_id"] in segments:
            yield outcome(row, row["attempt_id"], failures_path, "failed_request_not_exported")


def audit_trace_inputs(trace, template, *, trace_path, config_path):
    phases = {kind: {"total": 0, "matched": 0, "mismatches": []} for kind in ("READ", "REDUCE")}
    for turn in trace["turns"]:
        plan = decode(ContextPlan, turn["context_plan"])
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        snapshot = SessionSnapshot(turn["session_id"], raw_history=decode(tuple[TurnRecord, ...], turn["actual_history"]))
        for event in turn["events"]:
            if event["kind"] not in phases:
                continue
            task = decode(ReadTask, event["read_task"])
            claims = decode(tuple[EvidenceClaim, ...], event["input_claims"])
            messages = step_messages(plan, task, tuple(units[key] for key in task.input_unit_ids), claims, snapshot)
            current_hash = template.render(messages).input_hash
            phase = phases[event["kind"]]
            phase["total"] += 1
            if current_hash == event["input_hash"]:
                phase["matched"] += 1
            else:
                phase["mismatches"].append({"event_id": event_id(turn, event),
                    "recorded_input_hash": event["input_hash"], "current_input_hash": current_hash})
    provenance = {}
    for name, path in (("source_trace", trace_path), ("config", config_path),
                       ("reading_compiler", Path(__file__).resolve().parents[1]/"controller"/"reading.py")):
        with Path(path).open("rb") as stream:
            provenance[name] = {"path": str(Path(path).resolve()),
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    return {"scope": "recorded READ and model REDUCE inputs recompiled with current messages and tokenizer",
        "split": sorted({turn["split"] for turn in trace["turns"]}), **provenance,
        "fingerprint": asdict(template.fingerprint), "phases": phases,
        "current_read_inputs_match": not phases["READ"]["mismatches"] if phases["READ"]["total"] else None,
        "current_reduce_inputs_match": not phases["REDUCE"]["mismatches"] if phases["REDUCE"]["total"] else None,
        "excluded_scope": "scheduler, CPU reductions and ANSWER protocol", "model_calls": 0}


def audit_inputs(args):
    config = load_config(args.config)
    result = audit_trace_inputs(json.loads(Path(args.trace).read_text()), create_model_template(config.model),
        trace_path=args.trace, config_path=args.config)
    write_json(args.output, result)
    return result


def export(args):
    config = load_config(args.config)
    template = create_model_template(config.model)
    store = SourceLedger(args.state_db)
    turns = []
    try:
        selected = store.committed_attempts(args.attempt_id, limit=args.max_capture_turns)
        attempt_ids = {attempt.attempt_id for attempt in selected}
        records = source_records(args.records, store=store, attempt_ids=attempt_ids)
        segments = {row["measurement_segment_id"] for row in records.values()}
        if len(segments) > 1:
            raise ValueError("export requires one measurement segment; select its requests with --attempt-id")
        for attempt in selected:
            metadata = {**records.get(attempt.attempt_id, {}), "max_new_tokens": config.context.max_new_tokens}
            if "response" not in metadata:
                raise ValueError("export needs the original study record with actual client timing")
            turns.append(export_attempt(store, attempt, template, request_sampling(config.sampling), metadata))
    finally:
        store.close()
    result = make_trace(turns, config=config, source_db=args.state_db, captured=False,
        case_outcomes=unexported_requests(args.records, args.failed_conversations, attempt_ids, segments))
    records_path = Path(args.records).resolve()
    with records_path.open("rb") as stream:
        result["source_records"] = {"path": str(records_path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "capture", "capture-reader-traces"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--max-capture-turns", type=int, default=32)
        if name != "capture-reader-traces":
            sub.add_argument("--state-db", required=True)
            sub.add_argument("--records", required=True)
            sub.add_argument("--attempt-id", action="append", default=[])
        else:
            sub.add_argument("--reader-traces", required=True)
        if name == "export":
            sub.add_argument("--failed-conversations", required=True)
            sub.add_argument("--output", required=True)
        else:
            sub.add_argument("--base-url", required=True, help="Dedicated canonical native reader service")
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--gpu-device", type=int, required=True)
            sub.add_argument("--resource-gpu-devices", type=lambda value: tuple(map(int, value.split(","))))
            sub.add_argument("--resource-process-roots", type=lambda value: tuple(map(int, value.split(","))), default=())
            sub.add_argument("--study-config", help="Actually load and keep the same resident retrieval/task/support assets")
            sub.add_argument("--read-packing", choices=("mixed", "source_windows"), default="source_windows")
            sub.add_argument("--read-input-budget", type=int, default=12288)
            sub.add_argument("--read-output-budget", type=int, default=2048)
    sub = commands.add_parser("replay")
    sub.add_argument("--trace", required=True)
    sub.add_argument("--config", required=True)
    sub.add_argument("--base-url", required=True, help="Dedicated profiling service; each arm clears cache")
    sub.add_argument("--gpu-device", type=int, required=True)
    sub.add_argument("--output-dir", required=True)
    sub.add_argument("--horizon-seconds", type=float, default=180, help="Common observation window for every arm; at least 180 seconds")
    sub.add_argument("--ttl-seconds", type=lambda value: tuple(map(float, value.split(","))), default=(15, 60, 180))
    sub.add_argument("--max-candidates", type=int, default=8)
    sub.add_argument("--page-size", type=int)
    sub.add_argument("--max-prefixes-per-decision", type=int, default=4)
    sub.add_argument("--history-steps", type=int, default=32)
    sub.add_argument("--max-replay-steps", type=int, default=128)
    sub.add_argument("--max-total-generations", type=int, default=512)
    sub.add_argument("--residency-interval", type=float, default=.25)
    sub.add_argument("--seed", type=int, default=42)
    sub = commands.add_parser("fit")
    sub.add_argument("--records", required=True)
    sub.add_argument("--split", choices=("train", "dev"), required=True)
    sub.add_argument("--features", type=lambda value: value.split(","), default=list(RETENTION_FEATURE_NAMES))
    sub.add_argument("--output", required=True)
    sub = commands.add_parser("reassess")
    sub.add_argument("--replay-root", required=True)
    sub = commands.add_parser("audit-inputs")
    for name in ("trace", "config", "output"):
        sub.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args)
    elif args.command == "capture":
        asyncio.run(capture(args))
    elif args.command == "capture-reader-traces":
        asyncio.run(capture_reader_traces(args))
    elif args.command == "replay":
        asyncio.run(replay(args))
    elif args.command == "reassess":
        reassess(args)
    elif args.command == "audit-inputs":
        audit_inputs(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
