"""Generate actual reader-bound direct/counterfactual gain supervision.

All plans and frozen state keys are written before execution. Each experimental
branch uses Controller.serve_turn, fixed semantic choices, real READ/REDUCE and
one terminal QA. The shared four-component judge scores the actual answers.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time
import sys
from urllib.parse import urlsplit

from controller.context_schema import EvidenceSet, EvidenceUnit, ReadingOutcome, SourceSpan, TaskState
from controller.schema import QueryContext, SessionSnapshot, TurnRecord
from controller.serialization import decode, to_jsonable
from training.collect_plan_gains import collect_pair
from training.plan_gain_protocol import (AuditedTaskInterpreter, EmptyPlannedPremise,
    FixedCounterfactualRuntime, GapRetrievalRequest, ObservedActionFailure, PlannedReductionConflict,
    build_counterfactual_plans, build_observed_action_plans, observed_state_binding)
from training.train_plan_gain import actual_history, working_state_key


COMPONENTS = ("necessary_points", "grounding", "conditions_versions", "task_score")
DEFAULT_WEIGHTS = {"necessary_points": .3, "grounding": .2, "conditions_versions": .2, "task_score": .3}


def digest(value):
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(to_jsonable(value), ensure_ascii=False, allow_nan=False)+"\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def gap_retriever_identity(manifest):
    from controller.resident_models import resident_model_digest
    return {"implementation": "StudyEvidenceProvider.extend_for_plan",
            "resident_model_manifest_digest": resident_model_digest(manifest)}


def observed_rows(rows, packets, *, selection=None):
    """Keep each task's first legal decision for every non-ANSWER action."""
    from controller.plan_policy import CompleteCost
    kinds = ("read_once", "reduce_once", "gap_retrieval")
    by_task = {row["task_id"]: row for row in rows}
    chosen, skipped, seen = {}, Counter(), set()
    observed_tasks = set()
    for packet in packets:
        identity, task_id = packet["observation_id"], packet["task_id"]
        if identity in seen:
            skipped["duplicate_observation"] += 1
            continue
        seen.add(identity)
        if task_id not in by_task:
            skipped["outside_requested_members"] += 1
            continue
        row = by_task[task_id]
        if any(packet[name] != row[name] for name in ("cohort_split", "cohort_manifest_sha256")):
            raise ValueError("observed state belongs to a different registered cohort identity")
        state = decode(ReadingOutcome, packet["working_state"])
        if packet["decision_mode"] == "mandatory_coverage":
            skipped["mandatory_source_coverage_incomplete"] += 1
            continue
        if not state.steps:
            skipped["empty_observed_state"] += 1
            continue
        snapshot = decode(SessionSnapshot, packet["snapshot"])
        expected = [{name: turn[name] for name in ("turn_id", "user_text", "assistant_text")}
                    for turn in row["actual_history"]]
        if (actual_history(snapshot) != expected or state.plan.task_state.current_turn_id != row["turn"]
                or state.plan.task_state.raw_question != row["question"]):
            raise ValueError("observed state differs from the audited actual task/history")
        from controller.context_schema import RetrievalState
        from controller.schema import OnlineTurn
        interpreter = AuditedTaskInterpreter(None)
        interpreter.bind(row, state.plan.task_state)
        interpreter.interpret(OnlineTurn(row["conversation_id"], task_id, row["turn"], "training",
            row["question"], (), (), "P3"), RetrievalState(0, snapshot.raw_history))
        if not packet["native_condition"]:
            raise ValueError("observed state requires its recorded native condition")
        if "direct_answer" not in packet["legal_actions"]:
            skipped["no_legal_direct_pair"] += 1
            continue
        observed_tasks.add(task_id)
        paid = decode(CompleteCost, packet["paid_cost"])
        binding = observed_state_binding(state, snapshot, paid_cost=paid,
            native_condition=packet["native_condition"], reader_checkpoint=packet["reader_checkpoint"],
            reader_profile_id=packet["reader_profile_id"], paid_observation=packet.get("paid_observation"))
        binding.update(observation_id=identity, cohort_split=packet["cohort_split"],
            cohort_manifest_sha256=packet["cohort_manifest_sha256"],
            resident_model_manifest_digest=packet["resident_model_manifest_digest"])
        key = (packet["decision_seq"], packet["native_condition"]["observed_time_ns"], identity)
        candidate = {**row, "task_state": to_jsonable(state.plan.task_state),
                     "observed_packet": packet, "observed_binding": binding}
        for kind in kinds:
            if kind not in packet["legal_actions"]:
                continue
            action = task_id, kind
            if action in chosen:
                skipped["additional_legal_observation_for_action"] += 1
            if action not in chosen or key < chosen[action][0]:
                chosen[action] = key, candidate
    result, coverage, missing = [], {}, []
    for row in rows:
        task_id = row["task_id"]
        decisions = {}
        for kind in kinds:
            if (task_id, kind) not in chosen:
                missing.append({"task_id": task_id, "family": kind, "reason": "no_legal_observation"})
                continue
            _, candidate = chosen[task_id, kind]
            identity = candidate["observed_packet"]["observation_id"]
            if identity not in decisions:
                decisions[identity] = {**candidate, "observed_action_kinds": []}
            decisions[identity]["observed_action_kinds"].append(kind)
            coverage.setdefault(kind, []).append(task_id)
        result.extend(sorted(decisions.values(), key=lambda value: (
            value["observed_packet"]["decision_seq"], value["observed_packet"]["observation_id"])))
    if selection is not None:
        selection.update(rule="first legal decision_seq/native observed time/observation_id per unique task and action; no future outcomes",
            selected=[{"task_id": row["task_id"], "observation_id": row["observed_packet"]["observation_id"],
                       "actions": row["observed_action_kinds"]} for row in result],
            observed_task_ids=[row["task_id"] for row in rows if row["task_id"] in observed_tasks],
            missing_observation_task_ids=[row["task_id"] for row in rows if row["task_id"] not in observed_tasks],
            action_coverage={kind: coverage.get(kind, []) for kind in kinds},
            unavailable_actions=missing, skipped_observations=dict(skipped))
    return result


class GenerationLimitReached(ValueError):
    pass


class CompletedJudgeTruncated(ValueError):
    pass


class CompletedSupportTruncated(ValueError):
    pass


class JudgeWindowUnavailable(ValueError):
    pass


class GenerationBudget:
    """One shared count for support, actual reader and offline judge calls."""
    def __init__(self, maximum, journal):
        self.maximum, self.journal = maximum, Path(journal)
        self.tickets = []
        self.lock = threading.Lock()

    @property
    def remaining(self):
        return self.maximum-len(self.tickets)

    def claim(self, role, identity, inputs):
        with self.lock:
            if len(self.tickets) == self.maximum:
                raise GenerationLimitReached("declared total model-generation allocation exhausted")
            ticket = {"index": len(self.tickets), "role": role, "identity": identity,
                "started": time.monotonic(), "input_tokens": len(inputs), "status": "submitted"}
            self.tickets.append(ticket)
            append(self.journal, ticket)
            return ticket

    def finish(self, ticket, *, output_tokens=None, error=None):
        with self.lock:
            ticket.update(completed=time.monotonic(), output_tokens=output_tokens,
                          status="completed" if error is None else "unresolved", error=error)
            append(self.journal, ticket)


class BudgetedSyncClient:
    def __init__(self, client, budget, path):
        self.client, self.budget, self.path = client, budget, Path(path)
    def post(self, path, **kwargs):
        if path != "/generate":
            return self.client.post(path, **kwargs)
        ticket = self.budget.claim("support", "support:"+str(len(self.budget.tickets)), kwargs["json"]["input_ids"])
        append(self.path, {"ticket": ticket["index"], "status": "submitted", "request": kwargs["json"]})
        try:
            result = self.client.post(path, **kwargs)
            result.raise_for_status()
        except BaseException as error:
            self.budget.finish(ticket, error=f"{type(error).__name__}: {error}")
            append(self.path, {"ticket": ticket["index"], "status": "unresolved", "error": str(error)})
            raise
        self.budget.finish(ticket, output_tokens=result.json()["meta_info"].get("completion_tokens"))
        append(self.path, {"ticket": ticket["index"], "status": "completed", "response": result.json()})
        reason = result.json()["meta_info"].get("finish_reason")
        reason = reason.get("type") if isinstance(reason, dict) else reason
        if reason in {"length", "max_tokens"}:
            raise CompletedSupportTruncated("actual candidate-support teacher reached its output limit")
        return result
    def close(self):
        self.client.close()


class JudgeJournal:
    def __init__(self, client, budget, path):
        self.client, self.budget, self.path = client, budget, Path(path)
    async def post(self, path, **kwargs):
        if path != "/generate":
            return await self.client.post(path, **kwargs)
        payload = kwargs["json"]
        ticket = self.budget.claim("judge", "judge:"+str(len(self.budget.tickets)), payload["input_ids"])
        append(self.path, {"ticket": ticket["index"], "status": "submitted", "request": payload})
        try:
            result = await self.client.post(path, **kwargs)
            result.raise_for_status()
        except BaseException as error:
            self.budget.finish(ticket, error=f"{type(error).__name__}: {error}")
            append(self.path, {"ticket": ticket["index"], "status": "unresolved", "error": str(error)})
            raise
        raw = result.json()
        self.budget.finish(ticket, output_tokens=raw["meta_info"].get("completion_tokens"))
        append(self.path, {"ticket": ticket["index"], "status": "completed", "response": raw})
        finish = raw["meta_info"].get("finish_reason")
        finish = finish.get("type") if isinstance(finish, dict) else finish
        if finish in {"length", "max_tokens"}:
            raise CompletedJudgeTruncated("actual fixed judge reached its output limit")
        return result


def bounded_engine(engine, budget, wire_path):
    from profiling.retention_trace import RecordingEngine
    class BoundedReader(RecordingEngine):
        async def generate(self, **kwargs):
            ticket = budget.claim("reader", kwargs["attempt_id"], kwargs["input_ids"])
            try:
                result = await super().generate(**kwargs)
            except BaseException as error:
                budget.finish(ticket, error=f"{type(error).__name__}: {error}")
                raise
            else:
                budget.finish(ticket, output_tokens=len(result.output_ids))
                return result
            finally:
                write(wire_path, self.calls)
    return BoundedReader(engine)


def audited_cohort(labels, sources, *, split, limit, cohort_manifest):
    from training.train_evidence_policy import qualify_collection_rows
    from training.task_data import select_cohort_rows
    from training.reader_data import file_digest, history_digest
    manifest = json.loads(Path(cohort_manifest).read_text())
    manifest_hash = file_digest(cohort_manifest)
    members = {member["task_id"]: member for member in manifest["members"] if member["split"] == split}
    for row in labels:
        if row["task_id"] not in members:
            continue
        member = members[row["task_id"]]
        if (row["conversation_id"] != member["conversation_id"] or row["turn"] != member["turn"]
                or history_digest(row["actual_history"]) != member["actual_history_sha256"]):
            raise ValueError("gain label differs from the registered actual task/history member")
        if any(name in row and row[name] != value for name, value in (
                ("cohort_split", split), ("cohort_manifest_sha256", manifest_hash))):
            raise ValueError("gain labels retain a different source cohort identity")
    labels = select_cohort_rows(labels, cohort_manifest, split)
    eligible, excluded = qualify_collection_rows(labels, sources)
    unique = {row["task_id"] for row in labels}
    if len(unique) != len(labels):
        raise ValueError("gain source labels must contain unique actual task IDs")
    selected = []
    for row in eligible:
        if row["script_history_mismatch"] is not False or "simulation" in row["rollout_policy"]:
            excluded.append({"task_id": row["task_id"], "reason": "history_not_audited_actual_model_rollout"})
            continue
        if row["turn"] != len(row["actual_history"])+1:
            excluded.append({"task_id": row["task_id"], "reason": "current_turn_not_next_actual_qa"})
            continue
        selected.append(row)
    by_task = {row["task_id"]: row for row in selected}
    selected = [by_task[identity] for identity in members if identity in by_task][:limit]
    return selected, {"split": split, "source_partition": "train",
        "cohort_manifest": str(Path(cohort_manifest).resolve()), "cohort_manifest_sha256": manifest_hash,
        "allocation_unit": "registered conversation_id",
        "declared_task_ids": list(members),
        "missing_label_task_ids": [identity for identity in members if identity not in unique],
        "planned_task_ids": [row["task_id"] for row in selected], "exclusions": excluded,
        "source_label_sha256": digest(labels), "source_originals_sha256": digest(sources)}


def create_audited_provider(config, template):
    from adapters.study_retrieval import create_study_retriever
    provider = create_study_retriever(config, template)
    original = provider.directory.task_interpreter
    interpreter = AuditedTaskInterpreter(original.history_index)
    interpreter.model = original.model
    interpreter.original_interpreter = original
    provider.directory.task_interpreter = interpreter
    return provider, interpreter


class ResourceWindows:
    """Separate online and offline-judge physical windows with raw boundaries."""
    def __init__(self, online, judge):
        self.online, self.judge = online, judge
    async def start(self):
        await self.online.start()
        await self.judge.start()
    async def close(self):
        await self.online.stop()
        await self.judge.stop()
    async def finish(self, role, start, end):
        from bench.study_resources import integrate_physical
        sampler = self.online if role == "online" else self.judge
        sampler.samples.append(await asyncio.to_thread(sampler.sample, sampler.gpu_devices, sampler.process_ids))
        cost = integrate_physical(sampler.samples, start, end)
        if not cost["complete_window_coverage"]:
            raise ValueError("counterfactual physical costs require actual window boundaries")
        return {**cost, "window": {"start": start, "end": end, "clock": "client_monotonic"},
            "resource_scope": {"gpu_devices": sampler.gpu_devices, "process_roots": sampler.process_ids,
                               "host_basis": sampler.observation()["host_basis"]}}


def judge_sources(base, scope_material, row, sources, *, corpus_id):
    from controller.evidence import make_evidence
    by_source = {source["source_id"]: source for source in sources}
    units = {unit.unit_id: unit for unit in base.selected_set.selected_units}
    if scope_material is not None:
        units.update((unit.unit_id, unit) for unit in scope_material.selected_units)
    for number, witness in enumerate(row["verified_grounding"]):
        source = by_source[witness["passage_id"]]
        if source["text"][witness["start"]:witness["end"]] != witness["quote"]:
            raise ValueError("offline judge witness differs from the audited training original")
        block = make_evidence(source["text"], source["source_id"], corpus_id=corpus_id,
            version=source["version"], source_uri=source["source_uri"])
        span = SourceSpan("judge-witness:"+str(number), block.key, witness["start"], witness["end"])
        units[span.span_id] = EvidenceUnit(span.span_id, (span,), witness["quote"])
    return EvidenceSet("offline-judge-material", tuple(units.values()))


def planned_model_calls(plan):
    return sum(step.task.kind == "ANSWER" or not (step.task.kind == "REDUCE"
        and plan.context_plan.reading_mode == "typed_reduce" and step.task.operation in {"count", "sum", "group", "union"})
        for step in plan.steps)


def preparation_usage(provider, before, budget, ticket_start):
    observed = provider._auxiliary_usage(before, ())
    calls = [ticket for ticket in budget.tickets[ticket_start:] if ticket["role"] == "support"]
    models = ([row for row in observed["models"] if "support_utility" not in row["roles"]]
              if calls else observed["models"])
    if calls:
        complete = all(ticket["status"] == "completed" and ticket["output_tokens"] is not None for ticket in calls)
        models.append({"roles": ["support_utility"],
            "input_tokens": sum(ticket["input_tokens"] for ticket in calls) if complete else None,
            "output_tokens": sum(ticket["output_tokens"] for ticket in calls) if complete else None,
            "compute_ms": sum((ticket["completed"]-ticket["started"])*1000 for ticket in calls) if complete else None,
            "compute_scope": "actual client support-call elapsed including transport"})
    return {**{name: sum(row[name] for row in models) if all(row[name] is not None for row in models) else None
               for name in ("input_tokens", "output_tokens", "compute_ms")}, "models": models}


async def retire_completed_failed_steps(service, attempt_id):
    acknowledgements = []
    for stage, record in service.store.execution_steps_with_stage(attempt_id):
        if stage == "COMPLETE":
            continue
        _, typed, _ = service.store.execution_step(attempt_id, record.step_id)
        if typed.receipt is None or not typed.receipt.generation_complete:
            continue
        epoch = typed.engine_epoch
        ack = await service.engine.retire(epoch.engine_session_id, epoch.session_generation,
            "gain-completed-failure:"+attempt_id+":"+typed.step_id, epoch.engine_restart_id,
            "actual_counterfactual_completed_failed_step")
        if not ack.acknowledged:
            raise ValueError("failed counterfactual completed native step has no retirement acknowledgement")
        acknowledgements.append(to_jsonable(ack))
    return acknowledgements


async def execute_branch(service, plan, frozen, snapshot, row, *, case_dir, binding, preparation,
                         resources, cache_start, generation_budget, observed=None,
                         gap_request=None, gap_provider=None):
    from controller.execution import ReadingBudgetExceeded
    from controller.engine_client import EngineError
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True)
    state_key = working_state_key(frozen, snapshot)
    if observed is not None and (observed["state_key"] != state_key
            or observed["actual_history"] != actual_history(snapshot)
            or observed["reader_checkpoint"] != binding["reader_checkpoint"]
            or observed["reader_profile_id"] != binding["reader_profile_id"]):
        raise ValueError("observed branch differs from its frozen reader/history binding")
    registration = {"registered_at": time.monotonic(), "frozen_state_key": state_key,
        "plan_candidate": to_jsonable(plan), "working_state": to_jsonable(frozen), "snapshot": to_jsonable(snapshot),
        "source_task_id": row["task_id"], "source_native_partition": row["split"],
        "observed_state_binding": observed, "gap_request": gap_request if plan.semantic_kind == "gap_retrieval" else None,
        **binding}
    write(case_dir/"before.json", registration)
    session = "gain:"+digest([row["task_id"], plan.plan_id, binding["reader_checkpoint"], observed])[:24]
    before = service.store.read_snapshot(session, row["snapshot_id"])
    before = service.store.sync_history(session, snapshot.raw_history, before.version)
    from controller.plan_policy import CompleteCost
    paid = None if observed is None else decode(CompleteCost, observed["paid_cost"])
    runtime = FixedCounterfactualRuntime(plan, initial_state=frozen if observed is not None else None,
        paid_cost=paid, gap_request=gap_request, gap_provider=gap_provider,
        initial_snapshot=snapshot if observed is not None else None)
    service.rolling_policy = runtime
    service.bind_resident_models(binding["resident_model_manifest"])
    completed = {result.step_id for result in frozen.results} if observed is not None else set()
    required_calls = sum(step.task.step_id not in completed and (step.task.kind == "ANSWER" or not (
        step.task.kind == "REDUCE" and plan.context_plan.reading_mode == "typed_reduce"
        and step.task.operation in {"count", "sum", "group", "union"})) for step in plan.steps)
    if generation_budget.remaining < required_calls+1:
        write(case_dir/"not-run.json", {"status": "generation_budget_unavailable", "registered": registration})
        return None
    if cache_start == "cold":
        await service.engine.profile_prepare("cold")
    response = None
    failure = None
    cleanup = []
    started = time.monotonic()
    response_finished = None
    call_cursor = len(service.engine.calls)
    try:
        material = plan.context_plan.selected_set.source_blocks
        if observed is not None:
            material = tuple({block.key: block for block in (*snapshot.archive, *material)}.values())
        response = await service.serve_turn(session, row["question"], material,
            request_id="counterfactual", corpus_snapshot_id=row["snapshot_id"],
            context_plans=(plan.context_plan,), query_context=QueryContext(
                plan.context_plan.task_state.query or row["question"], "frozen_audited_task",
                task_state=plan.context_plan.task_state))
        response_finished = time.monotonic()
    except (EngineError, ReadingBudgetExceeded, EmptyPlannedPremise, PlannedReductionConflict,
            ObservedActionFailure, GenerationLimitReached) as error:
        response_finished = time.monotonic()
        failure = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, EngineError) and error.code != "READ_STEP_OUTPUT_TRUNCATED":
            raise
        attempt = service.store.find_request(session, "counterfactual")
        cleanup = await retire_completed_failed_steps(service, attempt.attempt_id)
    finally:
        active = sys.exc_info()[1]
        if active is not None:
            failure = {"type": type(active).__name__, "message": str(active)}
        if response_finished is None:
            response_finished = time.monotonic()
        await service.close_session(session)
        ended = time.monotonic()
        physical = await resources.finish("online", started, response_finished)
        cleanup_cost = {"complete_ms": (ended-response_finished)*1000,
            **await resources.finish("online", response_finished, ended),
            "scope": "branch session retirement after response completion; collection cost only"}
        attempt = service.store.find_request(session, "counterfactual")
        steps = (() if attempt is None else tuple(service.store.execution_step(attempt.attempt_id, step.step_id)[1]
                 for step in service.store.execution_steps(attempt.attempt_id)))
        after = service.store.get_snapshot(session)
        actual_budgets = {step.step_id: step.task.output_budget for step in steps}
        if response is not None:
            actual_budgets["ANSWER"] = attempt.plan.candidate.output_budget
        calls = service.engine.calls[call_cursor:]
        native_finished = all(call.get("receipt", {}).get("generation_complete") for call in calls)
        terminal_failure = (failure is not None and active is None and attempt is not None
                            and attempt.stage == "FAILED" and native_finished)
        action_counts = {"READ": sum(step.task.kind == "READ" for step in steps),
            "REDUCE": sum(step.task.kind == "REDUCE" for step in steps),
            "GAP_RETRIEVAL": runtime.new_action_counts["GAP_RETRIEVAL"],
            "ANSWER": sum(call["request"].get("response_schema") is None for call in calls)}
        input_tokens = sum(len(call["request"]["input_ids"]) for call in calls)
        output_tokens = (sum(len(call["result"]["output_ids"]) for call in calls)
                         if all("result" in call for call in calls) else None)
        auxiliary = runtime.new_auxiliary_usage
        auxiliary_input = None if auxiliary is None else auxiliary["input_tokens"]
        auxiliary_output = None if auxiliary is None else auxiliary["output_tokens"]
        execution = {"complete_ms": (response_finished-started)*1000, **physical,
            "total_input_tokens": input_tokens+auxiliary_input if auxiliary_input is not None else None,
            "total_output_tokens": output_tokens+auxiliary_output
                if output_tokens is not None and auxiliary_output is not None else None,
            "scope": "new action and complete ANSWER through response commit"}
        prior = ({"complete_ms": 0, "gpu_gib_seconds": 0, "host_gib_seconds": 0,
                  "total_input_tokens": 0, "total_output_tokens": 0} if paid is None else {
            "complete_ms": paid.latency_ms, "gpu_gib_seconds": paid.gpu_gib_ms/1000,
            "host_gib_seconds": paid.host_gib_ms/1000, "total_input_tokens": paid.input_tokens,
            "total_output_tokens": paid.output_tokens})
        full_cost = {**execution, "scope": "shared_preparation_plus_actual_branch_execution",
            "complete_ms": prior["complete_ms"]+execution["complete_ms"]+preparation["complete_ms"],
            **{name: prior[name]+execution[name]+preparation[name]
               for name in ("gpu_gib_seconds", "host_gib_seconds")},
            "total_input_tokens": prior["total_input_tokens"]+execution["total_input_tokens"]+preparation["input_tokens"]
                if preparation["input_tokens"] is not None and execution["total_input_tokens"] is not None else None,
            "total_output_tokens": prior["total_output_tokens"]+execution["total_output_tokens"]+preparation["output_tokens"]
                if execution["total_output_tokens"] is not None and preparation["output_tokens"] is not None else None,
            "shared_preparation_cost_id": preparation["shared_cost_id"]}
        output = {"task_id": row["task_id"], "conversation_id": row["conversation_id"],
            "question": row["question"], "actual_history": actual_history(snapshot),
            "history_origin": "frozen_actual_rollout", "source_rollout_policy": row["rollout_policy"],
            "history_reader_checkpoint": row.get("reader_checkpoint"),
            "frozen_state_key": state_key, "plan_id": plan.plan_id, "semantic_kind": plan.semantic_kind,
            "registered_plan_path": str((case_dir/"before.json").resolve()),
            "status": "completed" if response else "failed", "failure": failure,
            "termination": {"terminal": terminal_failure or response is not None,
                "user_paused": isinstance(active, (asyncio.CancelledError, KeyboardInterrupt)),
                "native_calls_finished": native_finished,
                "reason": failure["type"] if terminal_failure else "committed_QA" if response else None},
            "metrics": {}, "scoring": {"status": "missing", "identity": binding["scorer_identity"]},
            "answer": response.assistant_text if response else None, "response": to_jsonable(response),
            "steps": to_jsonable(steps), "attempt": to_jsonable(attempt), "cost": full_cost,
            "actual_branch_execution_cost": execution, "executed_output_budgets": actual_budgets,
            "paid_cost": prior, "cleanup_cost": cleanup_cost, "observed_state_binding": observed,
            "pre_action_quote": to_jsonable(runtime.pre_action_quote),
            "new_step_ids": tuple(step.step_id for step in steps), "new_action_counts": action_counts,
            "gap_result": to_jsonable(runtime.gap_result),
            "qa_increment": len(after.raw_history)-len(before.raw_history), "cleanup": cleanup,
            "actual_native_calls": calls,
            "comparison_budget": binding["comparison_budget"], **binding}
        write(case_dir/"run.json", output)
    if response is not None and output["qa_increment"] != 1:
        raise ValueError("counterfactual branch did not commit exactly one current QA")
    if response is None and output["qa_increment"] != 0:
        raise ValueError("failed counterfactual branch committed a QA")
    if response is not None and (attempt.plan.candidate.context_plan_id != plan.plan_id
            or set(actual_budgets)-{"ANSWER"} != {step.task.step_id for step in plan.steps
                if step.task.kind != "ANSWER" and step.task.step_id not in completed}):
        raise ValueError("actual branch execution differs from its pre-registered complete DAG")
    return output


def verify_actual_history(rows, rollouts):
    actual = {(row["conversation_id"], row["turn"]): row for row in rollouts}
    for row in rows:
        for turn in row["actual_history"]:
            source = actual[row["conversation_id"], turn["turn_id"]]
            receipt = source["response"]["receipt"]
            if (source["question"] != turn["user_text"] or source["answer"] != turn["assistant_text"]
                    or "simulation" in source["rollout_policy"] or not receipt["generation_complete"]
                    or not receipt["input_hash"]):
                raise ValueError("frozen assistant history differs from its actual completed source rollout")


async def score_branch(judge, template, row, material, branch, *, path, identity, weights,
                       resources, judge_window, expected_fingerprint):
    from training.preference_rollout import _judge
    base_client = judge.client
    before = await base_client.post("/context/engine/info", json={})
    before.raise_for_status()
    if before.json()["fingerprint"] != expected_fingerprint:
        raise ValueError("fixed gain scorer changed checkpoint/tokenizer before scoring")
    # Determine the actual prompt size at the recorded HTTP call, including the
    # actual generated answer. An oversize prompt is unscored, never truncated.
    class WindowedJudge:
        async def post(self, endpoint, **kwargs):
            if len(kwargs["json"]["input_ids"])+1024 > judge_window:
                raise JudgeWindowUnavailable("fixed counterfactual judge prompt exceeds its declared model window")
            return await judge.post(endpoint, **kwargs)
    start = time.monotonic()
    judged = record = None
    try:
        judged, record = await _judge(WindowedJudge(), template, row, material, branch["answer"])
        after = await base_client.post("/context/engine/info", json={})
        after.raise_for_status()
        if after.json()["fingerprint"] != expected_fingerprint:
            raise ValueError("fixed gain scorer changed checkpoint/tokenizer during scoring")
    finally:
        end = time.monotonic()
        physical = await resources.finish("judge", start, end)
        write(Path(path).with_name("judge.json"), {"quality_components": judged, "call": record,
            "physical_cost": physical, "scorer_identity": identity,
            "scope": "offline_judge_separate_from_online_answer_cost"})
    if set(weights) != set(COMPONENTS) or abs(sum(weights.values())-1) > 1e-9:
        raise ValueError("counterfactual quality uses four declared normalized component weights")
    for name in COMPONENTS:
        if not isinstance(judged[name], (int, float)) or not 0 <= judged[name] <= 1:
            raise ValueError("actual judge component is outside its declared normalized scale")
    quality = sum(weights[name]*judged[name] for name in COMPONENTS)
    branch.update(metrics={"plan_gain_quality": quality}, quality_components=judged,
        scoring={"status": "scored", "identity": identity, "judge_cost": {**record, **physical},
                 "raw_output_path": str(Path(path).with_name("judge.json").resolve())})
    write(path, branch)
    return branch


async def collect(config, settings, *, controller=None, provider=None, interpreter=None,
                  resources=None, judge_client=None, judge_template=None, accounting=None):
    """Bounded actual collection; injected dependencies are protocol-test seams."""
    import httpx
    from bench.study_resources import StudyResourceSampler, listening_processes
    from controller.config import (GenerationProtocol, apply_study_config, create_controller,
                                   create_model_template, load_config)
    from controller.context_schema import RetrievalState
    from controller.schema import OnlineTurn
    from profiling.retention_capture_cost import CaptureRunAccounting
    from training.preference_rollout import _judge

    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    labels, source_rows = read_rows(settings["task_labels"]), read_rows(settings["sources"])
    selected, cohort = audited_cohort(labels, source_rows, split=settings["split"], limit=len(labels),
        cohort_manifest=settings["cohort_manifest"])
    verify_actual_history(selected, read_rows(settings["actual_rollouts"]))
    validator = AuditedTaskInterpreter(None)
    valid, invalid = [], []
    for row in selected:
        history = tuple(decode(TurnRecord, turn) for turn in row["actual_history"])
        state = decode(TaskState, row["task_state"])
        validator.bind(row, state)
        try:
            validator.interpret(OnlineTurn(row["conversation_id"], row["task_id"], row["turn"],
                "training", row["question"], (), (), "P3"), RetrievalState(0, history))
        except ValueError as error:
            invalid.append({"task_id": row["task_id"], "reason": "invalid_audited_task_binding", "error": str(error)})
        else:
            valid.append(row)
    requested = cohort["declared_task_ids"][:settings["limit_tasks"]]
    eligible = {row["task_id"]: row for row in valid}
    valid = [eligible[identity] for identity in requested if identity in eligible]
    mode = settings.get("mode", "whole_plan")
    observation_selection = {}
    if mode == "observed_actions":
        valid = observed_rows(valid, read_rows(settings["observed_states"]), selection=observation_selection)
    elif mode != "whole_plan":
        raise ValueError("gain collection mode must be whole_plan or observed_actions")
    cohort["task_ref_exclusions"] = invalid
    cohort["requested_task_ids"] = requested
    cohort["missing_eligible_label_task_ids"] = [identity for identity in requested if identity not in eligible]
    cohort["planned_task_ids"] = list(dict.fromkeys(row["task_id"] for row in valid))
    cohort["missing_task_ids"] = [identity for identity in requested if identity not in cohort["planned_task_ids"]]
    if mode == "observed_actions":
        observation_selection["unavailable_actions"].extend(
            {"task_id": identity, "family": kind, "reason": "no_eligible_registered_label"}
            for identity in requested if identity not in eligible
            for kind in ("read_once", "reduce_once", "gap_retrieval"))
        cohort["observation_selection"] = observation_selection
        cohort["planned_observations"] = len(valid)
    write(output/"cohort.json", cohort)
    if not valid:
        raise ValueError("no unchanged audited task binding remains for actual gain collection")
    run = load_config(settings["run_config"])
    run = replace(apply_study_config(run, config), state_dir=str(output/"state"), plan_gain=None,
        prefix_retention={"mode": "native"}, planner=replace(run.planner, strategy="always_rebuild"))
    if settings.get("reader_base_url"):
        run = replace(run, engine=replace(run.engine, base_url=settings["reader_base_url"]))
    if run.engine.kind != "sglang" and controller is None:
        raise ValueError("gain supervision collection requires a scoped real SGLang reader")
    judge_run = load_config(settings["judge_config"])
    judge_url = settings["judge_base_url"]
    weights = settings.get("quality_weights", DEFAULT_WEIGHTS)
    budget = GenerationBudget(settings["max_generations"], output/"all-model-calls.jsonl")
    ports = {urlsplit(run.engine.base_url).port, urlsplit(judge_url).port,
             urlsplit(config["generation"]["teacher_url"]).port}
    if accounting is None:
        class GainEntryAccounting(CaptureRunAccounting):
            async def __aexit__(self, error_type, error, traceback):
                result = await super().__aexit__(error_type, error, traceback)
                path = self.output/"capture-entry.json"
                entry = json.loads(path.read_text())
                entry.update(format="contextkv_actual_plan_gain_collection_cost_v1",
                    scope="actual candidate preparation, counterfactual READ/REDUCE/ANSWER, judging and cleanup",
                    actual_collection_manifest="manifest.json", all_models_call_journal="all-model-calls.jsonl",
                    quality_training_scope="actual answer counterfactuals; not C retention value training")
                write(path, entry)
                return result
        accounting = GainEntryAccounting(run, output, settings["online_gpu_devices"][0],
            resource_gpu_devices=sorted(set((*settings["online_gpu_devices"], *settings["judge_gpu_devices"]))),
            resource_process_roots=tuple(sorted(set((*listening_processes(ports), *settings["extra_process_roots"])))))
    if resources is None:
        online_roots = listening_processes((urlsplit(run.engine.base_url).port,
                                           urlsplit(config["generation"]["teacher_url"]).port))
        judge_roots = listening_processes((urlsplit(judge_url).port,))
        resources = ResourceWindows(StudyResourceSampler(settings["online_gpu_devices"],
            tuple(sorted(set((*online_roots, *settings["extra_process_roots"])))), interval=.5),
            StudyResourceSampler(settings["judge_gpu_devices"], judge_roots, interval=.5))
    own_controller, own_provider, own_judge = controller is None, provider is None, judge_client is None
    all_runs, all_pairs, preparations, unavailable = [], [], [], []
    status, stop_reason = "completed", None
    active_error = None
    async with accounting:
        await resources.start()
        try:
            if own_controller:
                controller = create_controller(run)
            info = await controller.engine.info()
            if info.fingerprint != controller.template.fingerprint:
                raise ValueError("actual reader identity differs from the scoped collection template")
            reader_checkpoint = info.fingerprint.model_revision
            recorder = bounded_engine(controller.engine, budget, output/"reader-wire.json")
            controller.engine = recorder
            accounting.calls = recorder.calls
            if own_provider:
                provider, interpreter = await asyncio.to_thread(create_audited_provider, config, controller.template)
            if not isinstance(interpreter, AuditedTaskInterpreter):
                raise ValueError("gain preparation requires the unchanged audited-state interpreter")
            accounting.resident_model_manifest = to_jsonable(provider.manifest)
            if own_provider:
                accounting.record_resident_models(settings["study_config"], provider.manifest)
            from adapters.teacher_support import TeacherSupportBackend
            support_engine_info = None
            if isinstance(provider.support_backend, TeacherSupportBackend):
                provider.support_backend.client = BudgetedSyncClient(provider.support_backend.client, budget,
                    output/"support-wire.jsonl")
                provider.support_backend.record_path = output/"support-calls.jsonl"
                support_info = await asyncio.to_thread(provider.support_backend.client.post,
                    "/context/engine/info", json={})
                support_info.raise_for_status()
                support_engine_info = support_info.json()
            if own_judge:
                judge_client = httpx.AsyncClient(base_url=judge_url, timeout=600)
            if judge_template is None:
                judge_template = create_model_template(judge_run.model)
            judge_info_result = await judge_client.post("/context/engine/info", json={})
            judge_info_result.raise_for_status()
            judge_info = judge_info_result.json()
            if judge_info["fingerprint"] != to_jsonable(judge_template.fingerprint):
                raise ValueError("fixed scorer actual checkpoint differs from its scoped tokenizer/model")
            if own_judge and judge_info["fingerprint"]["model_revision"] != judge_run.model.revision:
                raise ValueError("fixed scorer does not match its explicit reader checkpoint")
            scorer = {"kind": "frozen_four_component_actual_answer_judge",
                "reader_checkpoint": judge_info["fingerprint"]["model_revision"],
                "fingerprint": judge_info["fingerprint"], "weights": weights,
                "implementation_sha256": hashlib.sha256(inspect.getsource(_judge).encode()).hexdigest(),
                "request_sampling": {"temperature": 0, "max_new_tokens": 1024},
                "metric": "plan_gain_quality", "score_range": [0, 1]}
            journal_judge = JudgeJournal(judge_client, budget, output/"judge-wire.jsonl")
            common_binding = {"reader_checkpoint": reader_checkpoint,
                "reader_profile_id": settings["reader_profile_id"], "scorer_identity": scorer,
                "split": settings["split"], "cohort_split": row["cohort_split"],
                "cohort_manifest_sha256": row["cohort_manifest_sha256"],
                "measurement_kind": "hardware" if info.engine_kind == "sglang" else "simulation",
                "resident_model_manifest": to_jsonable(provider.manifest),
                "task_state_source": "frozen_audited_actual", "collection_mode": mode,
                "counterfactual_cache_start": settings["cache_start"]}
            write(output/"binding.json", {**common_binding, "reader_engine_info": info,
                "support_engine_info": support_engine_info,
                "scorer_engine_info": judge_info, "generation_protocol": GenerationProtocol.from_config(run).record(),
                "source_files": {name: {"path": str(Path(settings[name]).resolve()),
                    "sha256": hashlib.sha256(Path(settings[name]).read_bytes()).hexdigest()}
                    for name in ("task_labels", "actual_rollouts", "sources", "run_config", "judge_config", "cohort_manifest")
                        + (("observed_states",) if mode == "observed_actions" else ())},
                "configuration": config, "collection_settings": settings,
                "task_inference": "frozen audited training labels; no new task-teacher inference"})
            for number, row in enumerate(valid):
                if budget.remaining < 2:
                    status, stop_reason = "generation_budget_exhausted", "no_preparation_and_answer_allocation"
                    break
                task_dir = output/"tasks"/f"{number:05d}"
                task_dir.mkdir(parents=True)
                history = tuple(decode(TurnRecord, turn) for turn in row["actual_history"])
                state = decode(TaskState, row["task_state"])
                snap = SessionSnapshot("gain-preparation:"+row["task_id"], raw_history=history,
                                       corpus_snapshot_id=row["snapshot_id"])
                interpreter.bind(row, state)
                tick = time.monotonic()
                preparation = None
                ticket_start = len(budget.tickets)
                plans = ()
                usage_before = {id(model): dict(model.usage) for _, model in provider._usage_sources()
                                if hasattr(model, "usage")}
                observed, gap_request = None, None
                try:
                    if mode == "observed_actions":
                        packet = row["observed_packet"]
                        observed = row["observed_binding"]
                        if (observed["reader_checkpoint"] != reader_checkpoint
                                or observed["reader_profile_id"] != settings["reader_profile_id"]):
                            raise ValueError("observed state belongs to a different actual reader checkpoint/profile")
                        frozen = decode(ReadingOutcome, packet["working_state"])
                        snap = decode(SessionSnapshot, packet["snapshot"])
                        base = frozen.plan
                        actual_resident = gap_retriever_identity(provider.manifest)["resident_model_manifest_digest"]
                        if packet["resident_model_manifest_digest"] != actual_resident:
                            raise ValueError("observed state belongs to different frozen A resident assets")
                        gap_request = decode(GapRetrievalRequest, packet["gap_request"]) if packet.get("gap_request") else None
                        if gap_request is not None and gap_request.retriever_identity != gap_retriever_identity(provider.manifest):
                            raise ValueError("registered GAP retriever differs from the actual collection configuration")
                        plans, omitted = build_observed_action_plans(frozen, snap, controller.template,
                            reader_checkpoint=reader_checkpoint, gap_request=gap_request,
                            answer_system_prompt=controller.policy.system_prompt)
                        available = {plan.semantic_kind for plan in plans}
                        reasons = {"read_once": "no_ready_unread_source_step",
                            "reduce_once": "no_unreduced_extract_basis_for_requested_operation",
                            "gap_retrieval": "no_registered_gap_request"}
                        omitted = (*omitted, *({"family": kind, "reason": reason}
                            for kind, reason in reasons.items() if kind not in available))
                        plans = tuple(plan for plan in plans if plan.semantic_kind == "direct_answer"
                                      or plan.semantic_kind in row["observed_action_kinds"])
                        trace, scope_material = {"source": "recorded_observed_state", "binding": observed}, None
                    else:
                        selection = await provider.retrieve(snap, row["question"], row["task_id"])
                        base = selection.context_plans[0]
                        if base.task_state != state:
                            raise ValueError("A candidate preparation changed the audited current task state")
                        trace = provider.traces[snap.session_id, row["task_id"]]
                        scope_material = None
                        if settings["scope_scan"] and state.source_scope:
                            scope_pool = provider.directory.full_scope(state.source_scope)
                            scope_material = EvidenceSet("gain-explicit-scope", scope_pool.units,
                                                         source_blocks=scope_pool.source_blocks)
                        plans, omitted = build_counterfactual_plans(base, snap, controller.template,
                            reader_checkpoint=reader_checkpoint, read_outputs=settings["read_outputs"],
                            scope_material=scope_material, answer_system_prompt=controller.policy.system_prompt)
                        frozen = ReadingOutcome(base, auxiliary_usage={"input_tokens": 0,
                            "output_tokens": 0, "compute_ms": 0.0, "models": []})
                    unavailable.extend({"task_id": row["task_id"], **reason} for reason in omitted)
                    material = judge_sources(base, scope_material, row, source_rows,
                                             corpus_id=config["retrieval"]["corpus_id"])
                    write(task_dir/"registered-family.json", {"registered_at": time.monotonic(),
                        "model_calls_before_branch_execution": len(budget.tickets),
                        "working_state": frozen, "snapshot": snap, "frozen_state_key": working_state_key(frozen, snap),
                        "observed_state_binding": observed, "gap_request": gap_request,
                        "candidate_plans": plans, "unavailable": omitted,
                        "offline_judge_material": material, "offline_reference": row["reference_answer"],
                        "preparation_trace": trace, "source_task": row})
                except (CompletedSupportTruncated, json.JSONDecodeError) as error:
                    unavailable.append({"task_id": row["task_id"], "reason": "completed_preparation_model_failed",
                                        "error": str(error)})
                    write(task_dir/"preparation-failure.json", {"type": type(error).__name__, "error": str(error)})
                finally:
                    ended = time.monotonic()
                    physical = await resources.finish("online", tick, ended)
                    usage = preparation_usage(provider, usage_before, budget, ticket_start)
                    preparation = {"shared_cost_id": "gain-preparation:"+digest([row["task_id"], tick]),
                        "task_id": row["task_id"], "complete_ms": (ended-tick)*1000, **physical, **usage,
                        "scope": "one actual candidate-family preparation; counted once in collection total",
                        "source_label_costs": {"task_teacher": row.get("teacher_metrics"),
                                               "grounding_review": row.get("verification_metrics")}}
                    preparations.append(preparation)
                    write(task_dir/"preparation-cost.json", preparation)
                if not plans:
                    continue
                comparison = {"body_tokens": base.budget.body_tokens,
                    "answer_output_tokens": base.budget.output_tokens, "window_tokens": base.budget.window_tokens,
                    "read_output_budget_grid": settings["read_outputs"] if mode == "whole_plan" else None,
                    "action_protocol": "whole_registered_plan" if mode == "whole_plan" else "one_registered_action_then_ANSWER",
                    "reference_state": "same frozen original material/actual history; direct has zero new READ"}
                binding = {**common_binding, "comparison_budget": comparison}
                async def retrieve_gap(plan, request, claims):
                    if request.retriever_identity != gap_retriever_identity(provider.manifest):
                        raise ValueError("GAP request is not bound to this actual retriever")
                    from controller.evidence_planner import _passages
                    cap = len(_passages(plan.selected_set.selected_units))+request.limits["max_passages"]
                    if plan.budget.max_passages is not None:
                        cap = min(cap, plan.budget.max_passages)
                    return await provider.extend_for_plan(snap, replace(plan, budget=replace(plan.budget,
                        max_passages=cap)), request.open_needs, claims)
                baseline = None
                for branch_number, plan in enumerate(plans):
                    branch_dir = task_dir/f"branch-{branch_number:02d}"
                    packet = {"plan_candidate": to_jsonable(plan), "working_state": to_jsonable(frozen),
                        "snapshot": to_jsonable(snap), "metric": "plan_gain_quality",
                        "normalization": {"scale": 1, "offset": 0,
                            "protocol_source": "four normalized components weighted by fixed scorer_identity.weights"},
                        "scorer_identity": scorer, "comparison_budget": comparison,
                        "direct_run": str((task_dir/"branch-00"/"run.json").resolve()),
                        "candidate_run": str((branch_dir/"run.json").resolve()),
                        "frozen_state_key": working_state_key(frozen, snap), "registered_at": time.monotonic()}
                    write(task_dir/f"pair-{branch_number:02d}.before.json", packet)
                    result = await execute_branch(controller, plan, frozen, snap, row, case_dir=branch_dir,
                        binding=binding, preparation=preparation, resources=resources,
                        cache_start=settings["cache_start"], generation_budget=budget, observed=observed,
                        gap_request=gap_request, gap_provider=retrieve_gap)
                    if result is None:
                        unavailable.append({"task_id": row["task_id"], "plan_id": plan.plan_id,
                            "reason": "next_complete_branch_unaffordable", "remaining_calls": budget.remaining})
                        status, stop_reason = "generation_budget_limited", "some_predeclared_branches_unaffordable"
                        continue
                    all_runs.append(result)
                    if result["status"] != "completed":
                        continue
                    try:
                        result = await score_branch(journal_judge, judge_template, row, material, result,
                            path=branch_dir/"run.json", identity=scorer, weights=weights, resources=resources,
                            judge_window=judge_run.context.max_total_model_tokens,
                            expected_fingerprint=judge_info["fingerprint"])
                    except (CompletedJudgeTruncated, JudgeWindowUnavailable, json.JSONDecodeError) as error:
                        result["scoring"].update(status="failed", error=str(error),
                                                 raw_output_path=str(branch_dir/"judge.json"))
                        write(branch_dir/"run.json", result)
                        continue
                    if plan.semantic_kind == "direct_answer":
                        if result["new_action_counts"]["READ"] != 0:
                            raise ValueError("gain direct baseline executed a READ")
                        baseline = result
                    elif baseline is not None and common_binding["measurement_kind"] == "hardware":
                        pair = collect_pair(packet, baseline, result)
                        pair["preparation_cost_id"] = preparation["shared_cost_id"]
                        append(output/"gains.jsonl", pair)
                        append(output/"pairs.jsonl", packet)
                        all_pairs.append(pair)
                write(output/"progress.json", {"tasks_prepared": len(preparations), "runs": len(all_runs),
                    "pairs": len(all_pairs), "generations": len(budget.tickets), "status": status})
                if budget.remaining < 2:
                    status, stop_reason = "generation_budget_exhausted", "no_complete_reader_and_judge_allocation"
                    break
        except GenerationLimitReached as error:
            status, stop_reason = "generation_budget_exhausted", str(error)
        finally:
            active_error = sys.exc_info()[1]
            if active_error is not None:
                status = "failed"
            if own_provider and provider is not None:
                await asyncio.to_thread(provider.close)
            if own_controller and controller is not None:
                await controller.aclose()
                controller.store.close()
            if own_judge and judge_client is not None:
                await judge_client.aclose()
            await resources.close()
            write(output/"phase-resources.json", {"online": resources.online.observation(),
                                                   "judge": resources.judge.observation()})
            write(output/"manifest.json", {"schema_version": 1, "status": status, "collection_mode": mode,
                "error": str(active_error) if active_error is not None else stop_reason,
                "scope": "actual reader-bound counterfactual supervision collection",
                "elapsed_ms": (time.monotonic()-started)*1000, "planned_tasks": cohort["planned_task_ids"],
                "cohort": cohort,
                "tasks_prepared": len(preparations), "actual_runs": len(all_runs), "actual_gain_pairs": len(all_pairs),
                "actual_observations": len({row["observed_state_binding"]["observation_id"] for row in all_runs
                                            if row["observed_state_binding"] is not None}),
                "actual_action_task_ids": {kind: sorted({row["task_id"] for row in all_runs
                    if row["semantic_kind"] == kind and row["new_action_counts"][action] == 1})
                    for kind, action in (("read_once", "READ"), ("reduce_once", "REDUCE"),
                                         ("gap_retrieval", "GAP_RETRIEVAL"))},
                "run_statuses": dict(Counter(row["status"] for row in all_runs)),
                "semantic_families": dict(Counter(row["semantic_kind"] for row in all_runs)),
                "unavailable": unavailable, "generations": {"limit": budget.maximum, "actual_calls": len(budget.tickets),
                    "roles": dict(Counter(ticket["role"] for ticket in budget.tickets)),
                    "unresolved": [ticket for ticket in budget.tickets if ticket["status"] != "completed"]},
                "shared_preparations": preparations, "outputs": {"gains": str(output/"gains.jsonl"),
                    "pairs": str(output/"pairs.jsonl"), "raw_reader": str(output/"reader-wire.json"),
                    "raw_support": str(output/"support-wire.jsonl"),
                    "raw_judge": str(output/"judge-wire.jsonl"), "all_calls": str(output/"all-model-calls.jsonl")},
                "cost_accounting": "entry physical/native totals count shared preparation once; each branch also records full hypothetical response cost with shared preparation ID; offline judge remains separate"})
    entry_path = output/"capture-entry.json"
    if entry_path.exists():
        entry = json.loads(entry_path.read_text())
        entry.update(format="contextkv_actual_plan_gain_collection_cost_v1",
            scope="actual candidate preparation, counterfactual READ/REDUCE/ANSWER, offline judging and cleanup",
            actual_collection_manifest="manifest.json", all_models_call_journal="all-model-calls.jsonl",
            quality_training_scope="reader-bound actual answer counterfactuals; not C retention value training")
        write(entry_path, entry)
    return json.loads((output/"manifest.json").read_text())


def main(argv=None):
    import yaml
    def integers(value):
        return tuple(map(int, value.split(",")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--reader-profile-id", required=True)
    parser.add_argument("--reader-base-url")
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--task-labels", required=True)
    parser.add_argument("--cohort-manifest", required=True)
    parser.add_argument("--actual-rollouts", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--limit-tasks", type=int, default=8)
    parser.add_argument("--read-outputs", type=integers, default=(512, 1024, 2048))
    parser.add_argument("--max-generations", type=int, default=256)
    parser.add_argument("--online-gpu-devices", type=integers, required=True)
    parser.add_argument("--judge-gpu-devices", type=integers, required=True)
    parser.add_argument("--extra-process-roots", type=integers, default=())
    parser.add_argument("--cache-start", choices=("cold", "observed"), default="observed")
    parser.add_argument("--scope-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", choices=("whole_plan", "observed_actions"), default="whole_plan")
    parser.add_argument("--observed-states", help="Registered observed-state JSONL packets for observed_actions mode")
    args = parser.parse_args(argv)
    if args.mode == "observed_actions" and not args.observed_states:
        parser.error("observed_actions requires --observed-states")
    if args.limit_tasks < 1 or args.max_generations < 1 or any(
            output < 1 for output in args.read_outputs):
        raise ValueError("collection bounds, output allocations and conversation holdout must be positive")
    config = yaml.safe_load(Path(args.study_config).read_text())
    print(json.dumps(asyncio.run(collect(config, vars(args))), ensure_ascii=False))


if __name__ == "__main__":
    main()
