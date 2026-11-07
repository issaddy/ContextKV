"""Measure standalone model REDUCE phases over real captured extract claims.

Claim text, values, conditions, versions and source coordinates stay frozen.
Only claim/Need reference names are namespaced, with an explicit origin map.
This is a phase-only cost experiment; it makes no QA or reduction-quality claim.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import time

from controller.config import create_controller, load_config
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceSet, Need,
                                      ReadResult, ReadTask, StepRecord, TaskState)
from controller.engine_client import EngineError
from controller.execution import ReadingExecutor
from controller.reading import output_schema, step_generation_kind, step_messages
from controller.reduction import UNIT_FACTORS, reduce_claims
from controller.serialization import to_jsonable
from controller.templates import token_hash
from profiling.retention_capture_cost import CaptureRunAccounting, load_resident_provider
from profiling.retention_trace import RecordingEngine, SourceLedger, digest, write_json


class ProbeGeometryUnavailable(ValueError):
    pass


def actual_extract_claims(entries, *, split, maximum):
    claims, origins, seen = [], [], set()
    for path in entries:
        entry = json.loads(Path(path).read_text())
        trace = json.loads((Path(path).parent/"capture-trace.json").read_text())
        if entry["source"] != "hardware" or trace["config"]["engine"]["kind"] != "sglang":
            raise ValueError("REDUCE cost inputs must originate in real native captures")
        store = SourceLedger(entry["state_database"])
        try:
            for turn in trace["turns"]:
                if turn["split"] != split:
                    continue
                for raw in store.execution_steps(turn["attempt_id"]):
                    stage, step, _ = store.execution_step(turn["attempt_id"], raw.step_id)
                    if stage != "COMPLETE" or step.task.kind != "READ":
                        continue
                    if (step.receipt is None or not step.receipt.generation_complete
                            or token_hash(step.input_ids) != step.receipt.input_hash):
                        raise ValueError("extract input source lacks an actual completed READ receipt")
                    for claim in step.result.claims:
                        if claim.kind != "extract":
                            continue
                        fact = to_jsonable(claim)
                        signature = digest({key: value for key, value in fact.items()
                            if key not in {"claim_id", "need_id", "parent_claim_ids"}})
                        if signature in seen:
                            continue
                        seen.add(signature)
                        alias = "policy:"+digest([str(Path(path).resolve()), turn["attempt_id"], claim.claim_id])[:24]+":claim:0"
                        claims.append(replace(claim, claim_id=alias, need_id="cost:need"))
                        origins.append(dict(input_alias=alias, original_claim=fact,
                            source_capture_entry=str(Path(path).resolve()), source_attempt_id=turn["attempt_id"],
                            source_step_id=step.step_id, source_read_receipt=to_jsonable(step.receipt),
                            source_reader_fingerprint=step.metrics["cache_quote"]["fingerprint"],
                            source_partition=turn["split"]))
                        if len(claims) >= maximum:
                            return tuple(claims), origins
        finally:
            store.close()
    if not claims:
        raise ValueError("no actual extracted leaf claims in the declared source partition")
    return tuple(claims), origins


def reduce_plan(claims, *, operation, input_ceiling, output_budget, window):
    question = "Combine the supplied extracted facts using the declared operation and preserve their parent links."
    state = TaskState("reduce-cost-probe", question,
        needs=(Need("cost:need", operation=operation),), current_turn_id=1)
    task = ReadTask("cost:REDUCE", "REDUCE", ("cost:need",), input_claim_ids=tuple(
        claim.claim_id for claim in claims), input_budget=input_ceiling, output_budget=output_budget,
        operation=operation)
    plan = ContextPlan("reduce-cost-probe", state, EvidenceSet("frozen-extract-claims", ()),
        budget=ContextBudget(window_tokens=window, read_input_tokens=input_ceiling,
            read_output_tokens=output_budget, max_total_tokens=window*4),
        read_layout="source_first", layouts=("existing",))
    if operation not in {"count", "sum", "union", "group"}:
        task = replace(task, output_schema=output_schema(task, (), claims, reading_mode=plan.reading_mode))
    return replace(plan, steps=(task,)), task


def replicated_claims(claims, count):
    """Repeat measured factual records under explicit, unique input aliases."""
    return tuple(replace(claims[index % len(claims)],
        claim_id="policy:"+digest([claims[index % len(claims)].claim_id, "replica", index])[:24]+":claim:0")
        for index in range(count))


def numeric_claim_bank(claims):
    """Use a real compatible numeric group for SUM stress, without new values."""
    groups = {}
    for claim in claims:
        try:
            Decimal(str(claim.value).replace(",", ""))
        except InvalidOperation:
            continue
        canonical = UNIT_FACTORS.get(claim.unit, (claim.unit,))[0]
        groups.setdefault(canonical, []).append(claim)
    return (max(groups.values(), key=len)[0],) if groups else ()


def model_stress_input(claims, *, template, snapshot, operation, target, output_budget,
                       window, max_claims, max_alias_characters):
    """A measured capacity payload; factual fields remain frozen verbatim."""
    trials = []

    def evaluate(count, suffix=0):
        selected = replicated_claims(claims, count)
        if suffix:
            selected = (*selected[:-1], replace(selected[-1], claim_id=selected[-1].claim_id+":r"+"x"*suffix))
        plan, task = reduce_plan(selected, operation=operation, input_ceiling=target,
                                output_budget=output_budget, window=window)
        rendered = template.render(step_messages(plan, task, (), selected, snapshot))
        trials.append(dict(claims=count, reference_name_stress_characters=suffix, input_tokens=len(rendered.input_ids)))
        return len(rendered.input_ids), selected, plan, task

    low, high, best = 1, max_claims, None
    while low <= high:
        middle = (low+high)//2
        result = evaluate(middle)
        if result[0] <= target and (best is None or result[0] > best[0]):
            best = result
        if result[0] == target:
            break
        if result[0] < target:
            low = middle+1
        else:
            high = middle-1
    if best is None:
        raise ProbeGeometryUnavailable("no frozen-claim payload fits the declared model input capacity")
    if best[0] < target:
        count, low, high = len(best[1]), 1, max_alias_characters
        while low <= high:
            middle = (low+high)//2
            result = evaluate(count, middle)
            if result[0] <= target and result[0] > best[0]:
                best = result
            if result[0] == target:
                break
            if result[0] < target:
                low = middle+1
            else:
                high = middle-1
    actual, selected, plan, task = best
    last_alias = selected[-1].claim_id
    return selected, plan, task, dict(requested_input_tokens=target, actual_input_tokens=actual,
        replicated_input=len(selected) > len(claims), new_facts=False, reference_name_stress=":r" in last_alias,
        actual_alias_to_origin=[dict(alias=claim.claim_id, original_alias=claims[index % len(claims)].claim_id)
                               for index, claim in enumerate(selected)],
        trials=trials, quality_scope="capacity-only stress; excluded from QA/gain/normal-throughput/C-value samples")


async def cpu_reduce_phase(service, attempt, task, claims, arrived):
    tick = time.monotonic()
    workload = await service.engine.metrics()
    claim = reduce_claims(task.operation, claims, claim_id=task.step_id+":claim:0", need_id="cost:need")
    record = StepRecord(attempt.attempt_id, task.step_id, task,
        result=ReadResult(task.step_id, (claim,)), metrics={"native_workload": workload,
            "generation_kind": "CPU_REDUCE", "request_arrived_monotonic": arrived})
    service.store.save_execution_step(record, "COMPLETE")
    ended = time.monotonic()
    record = replace(record, result=replace(record.result, elapsed_ms=(ended-tick)*1000),
        metrics={**record.metrics, "cpu_reduce_ms": (ended-tick)*1000,
            "phase_window": {"start": tick, "end": ended, "clock": "client_monotonic", "scope": "step_wall"}})
    service.store.save_execution_step(record, "COMPLETE")
    return record


async def close_completed_failure(service, calls):
    if not calls or any(not call.get("receipt", {}).get("generation_complete") for call in calls):
        raise ValueError("failed native phase outcome remains unresolved")
    acknowledgements = []
    for call in calls:
        request = call["request"]
        operation = "retire:"+request["session_id"]
        ack = await service.engine.retire(request["session_id"], request["expected_generation"],
            operation, request["expected_restart_id"], "completed_truncated_REDUCE_cost_probe")
        if not ack.acknowledged:
            raise ValueError("REDUCE cost probe native retirement is pending")
        service.store.ack_retire(operation)
        acknowledgements.append(to_jsonable(ack))
    return acknowledgements


async def collect(args):
    config = load_config(args.config)
    config = replace(config, engine=replace(config.engine, base_url=args.base_url),
        state_dir=str(Path(args.output_dir)/"state"), prefix_retention={"mode": "native"}, plan_gain=None)
    cpu = args.operation in {"count", "sum", "union", "group"}
    counts = (None,) if args.target_input_tokens is not None and not cpu else args.claim_counts
    cases = ([(count, 0, "native_observed_cpu") for count in counts] if cpu else
        [(count, output, state) for count in counts for output in args.output_budgets for state in args.cache_states])
    if not cpu and len(cases) > args.max_total_generations:
        raise ValueError("declared REDUCE phase matrix exceeds its model-generation budget")
    rows, outcomes = [], []
    async with CaptureRunAccounting(config, args.output_dir, args.gpu_device,
            resource_gpu_devices=args.resource_gpu_devices,
            resource_process_roots=args.resource_process_roots) as accounting:
        service = create_controller(config)
        resident_provider = None
        if args.study_config is not None:
            resident_provider, manifest = await asyncio.to_thread(load_resident_provider,
                                                                  args.study_config, service.template)
            accounting.record_resident_models(args.study_config, manifest)
        recorder = RecordingEngine(service.engine)
        service.engine = recorder
        accounting.calls, accounting.case_outcomes = recorder.calls, outcomes
        claims, origins = actual_extract_claims(args.source_capture_entry, split=args.split, maximum=args.max_claims)
        if args.operation == "sum":
            claims = numeric_claim_bank(claims)
        write_json(Path(args.output_dir)/"claim-input-origins.json", dict(source="actual_native_READ_extracts",
            origins=origins, transformations="claim/Need reference names only; frozen factual fields and source coordinates",
            quality_claim=False))
        try:
            for index, (count, output, cache_state) in enumerate(cases):
                accounting.current_source = dict(scope="standalone_REDUCE_cost_probe", split=args.split,
                    phase=step_generation_kind("REDUCE", args.operation),
                    input_claim_count=count, output_budget=output, cache_state=cache_state,
                    operation=args.operation, declared_input_ceiling=args.max_input_tokens)
                if not claims or count is not None and count > len(claims) and not args.replicate_frozen:
                    outcomes.append({**accounting.current_source, "status": "frozen_claim_bank_unavailable",
                                     "actual_claim_bank_size": len(claims)})
                    write_json(Path(args.output_dir)/"reduce-probe-cases.json", outcomes)
                    continue
                session = "reduce-cost-probe:"+str(index)
                snapshot = service.store.read_snapshot(session, "actual-frozen-claim-cost-scope")
                stress = None
                if args.target_input_tokens is not None and not cpu:
                    try:
                        selected, plan, task, stress = model_stress_input(claims, template=service.template,
                            snapshot=snapshot, operation=args.operation, target=args.target_input_tokens,
                            output_budget=output, window=args.window_tokens,
                            max_claims=args.max_stress_claims if args.replicate_frozen else len(claims),
                            max_alias_characters=args.max_alias_characters if args.reference_name_stress else 0)
                    except ProbeGeometryUnavailable as error:
                        outcomes.append({**accounting.current_source, "status": "input_geometry_unavailable", "reason": str(error)})
                        write_json(Path(args.output_dir)/"reduce-probe-cases.json", outcomes)
                        continue
                    count = len(selected)
                else:
                    selected = replicated_claims(claims, count) if count > len(claims) else claims[:count]
                    plan, task = reduce_plan(selected, operation=args.operation, input_ceiling=args.max_input_tokens,
                                            output_budget=output, window=args.window_tokens)
                rendered = None if cpu else service.template.render(step_messages(plan, task, (), selected, snapshot))
                if not cpu and (len(rendered.input_ids) > args.max_input_tokens or len(rendered.input_ids)+output > args.window_tokens):
                    outcomes.append({**accounting.current_source, "status": "actual_input_outside_declared_window",
                                     "actual_input_tokens": len(rendered.input_ids)})
                    write_json(Path(args.output_dir)/"reduce-probe-cases.json", outcomes)
                    continue
                if cache_state == "cold":
                    await service.engine.profile_prepare("cold")
                arrived = time.monotonic()
                attempt = service.store.begin_attempt(session, "phase:"+str(index),
                    digest(to_jsonable((task, selected))) if cpu else rendered.input_hash,
                    plan.task_state.raw_question, snapshot.version)
                service.store.save_context_plan(attempt.attempt_id, plan)
                executor = ReadingExecutor(service.store, service.engine, service.template,
                                            events=service.events, request_arrived=arrived)
                candidate = None if cpu else executor.request_compiler.compile_read(
                    plan, task, (), selected, snapshot, outputs=(output,))[0]
                cursor = len(recorder.calls)
                phase_call_started = time.monotonic()
                try:
                    record = (await cpu_reduce_phase(service, attempt, task, selected, arrived) if cpu else
                              await executor._generate_step(attempt.attempt_id, snapshot, plan, task, (), selected,
                                  candidate=candidate, rendered=rendered))
                except (EngineError, ValueError) as error:
                    if isinstance(error, EngineError) and error.code != "READ_STEP_OUTPUT_TRUNCATED":
                        raise
                    cleanup = [] if cpu else await close_completed_failure(service, recorder.calls[cursor:])
                    if not cpu:
                        stage, failed_step, _ = service.store.execution_step(attempt.attempt_id, task.step_id)
                        failed_step = replace(failed_step, metrics={**failed_step.metrics,
                            "phase_window": {"start": phase_call_started, "end": time.monotonic(),
                                "clock": "client_monotonic", "scope": "step_wall"},
                            "semantic_outcome": str(error)})
                        service.store.save_execution_step(failed_step, stage)
                        call = recorder.calls[-1]
                        rows.append(dict(observation_id=attempt.attempt_id+"/"+task.step_id,
                            generation_kind=step_generation_kind(task.kind, task.operation, plan.reading_mode),
                            context_action="REBUILD", input_ids=list(failed_step.input_ids),
                            input_tokens=len(failed_step.input_ids), output_budget=output,
                            output_tokens=len(failed_step.output_ids), phase_window=failed_step.metrics["phase_window"],
                            receipt=to_jsonable(failed_step.receipt), cache_quote=call["cache_quote"],
                            system_prompt=failed_step.input_messages[0].content, sampling_recipe=call["sampling"],
                            generation_window={"start": call["started_s"], "end": call["completed_s"]},
                            generation_result=call["result"], split=args.split, dataset="real_frozen_extract_claims",
                            operation=args.operation, input_claim_count=len(selected), actual_step=to_jsonable(failed_step),
                            semantic_outcome=str(error), original_stage=stage, capacity_stress=stress,
                            replicated_input=count > len(claims), new_facts=False,
                            replica_origin_aliases=[claims[index % len(claims)].claim_id for index in range(count)],
                            quality_scope="cost-only completed native attempt; semantic parse/truncation failure; no QA/gain labels"))
                        write_json(Path(args.output_dir)/"phase-capture.json", dict(format="contextkv_phase_capture_v1",
                            source="hardware", engine_kind=config.engine.kind, config=to_jsonable(config),
                            events=rows, source_claim_origins="claim-input-origins.json", quality_claim=False,
                            semantic_turn_policy="phase-only probe; native completion is distinct from semantic parse success"))
                    service.store.fail_attempt(attempt.attempt_id, str(error))
                    outcomes.append({**accounting.current_source, "status": "completed_native_failed_phase",
                                     "reason": str(error), "cleanup": cleanup})
                    write_json(Path(args.output_dir)/"reduce-probe-cases.json", outcomes)
                    await service.close_session(session)
                    continue
                call = None if cpu else recorder.calls[-1]
                if not cpu and call["receipt"]["attempt_id"] != record.receipt.attempt_id:
                    raise ValueError("REDUCE phase wire differs from its persisted native step")
                rows.append(dict(observation_id=attempt.attempt_id+"/"+task.step_id,
                    generation_kind="CPU_REDUCE" if cpu else step_generation_kind(task.kind, task.operation, plan.reading_mode),
                    context_action="CPU" if cpu else "REBUILD", input_ids=list(record.input_ids), input_tokens=len(record.input_ids),
                    output_budget=output, output_tokens=len(record.output_ids), phase_window=record.metrics["phase_window"],
                    receipt=to_jsonable(record.receipt), cache_quote=None if cpu else call["cache_quote"],
                    system_prompt="" if cpu else record.input_messages[0].content,
                    sampling_recipe=dict(service.engine._sampling_params) if cpu else call["sampling"],
                    generation_window=None if cpu else {"start": call["started_s"], "end": call["completed_s"]},
                    generation_result=None if cpu else call["result"], split=args.split, dataset="real_frozen_extract_claims",
                    sampling_binding="actual configured reader channel; CPU does not sample" if cpu else "actual generation wire",
                    fingerprint=asdict(service.template.fingerprint), native_workload=record.metrics.get("native_workload"),
                    operation=args.operation, input_claim_count=len(selected), actual_step=to_jsonable(record),
                    source_claim_aliases=[claim.claim_id for claim in selected],
                    replicated_input=count > len(claims), new_facts=False,
                    capacity_stress=stress,
                    replica_origin_aliases=[claims[index % len(claims)].claim_id for index in range(count)]
                        if count > len(claims) else None,
                    native_input_ceiling=args.max_input_tokens,
                    quality_scope="none; actual model REDUCE output measured and parsed, no gold matching or QA"))
                write_json(Path(args.output_dir)/"phase-capture.json", dict(format="contextkv_phase_capture_v1",
                    source="hardware", engine_kind=config.engine.kind, config=to_jsonable(config),
                    events=rows, source_claim_origins="claim-input-origins.json", quality_claim=False,
                    semantic_turn_policy="phase-only probe; no QA requested or committed"))
                # Close this experiment's business-attempt handle without
                # manufacturing a semantic QA commit. The actual step remains COMPLETE.
                service.store.fail_attempt(attempt.attempt_id, "PHASE_ONLY_PROBE_CLOSED_WITHOUT_QA")
                await service.close_session(session)
                outcomes.append({**accounting.current_source, "status": "phase_complete",
                    "actual_input_tokens": len(record.input_ids), "actual_output_tokens": len(record.output_ids),
                    "actual_native_receipt": to_jsonable(record.receipt), "qa_commits": 0})
                write_json(Path(args.output_dir)/"reduce-probe-cases.json", outcomes)
                print(json.dumps({"phase": rows[-1]["generation_kind"], "claims": count,
                    "input_tokens": len(record.input_ids), "output_budget": output, "cache_state": cache_state}), flush=True)
        finally:
            await service.aclose()
            service.store.close()
    return rows


def main(argv=None):
    def integers(value):
        return tuple(map(int, value.split(",")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--resource-gpu-devices", type=integers)
    parser.add_argument("--resource-process-roots", type=integers, default=())
    parser.add_argument("--study-config")
    parser.add_argument("--source-capture-entry", action="append", required=True)
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--claim-counts", type=integers, default=(1, 4, 16, 64))
    parser.add_argument("--output-budgets", type=integers, default=(2048,))
    parser.add_argument("--operation", choices=("infer", "compare", "summarize", "count", "sum", "union", "group"), default="infer")
    parser.add_argument("--replicate-frozen", action="store_true", help="Explicit cost-only capacity stress with repeated real facts; no new factual/quality labels")
    parser.add_argument("--target-input-tokens", type=int, help="Model REDUCE capacity point; actual input length is always recorded")
    parser.add_argument("--reference-name-stress", action="store_true", help="Only for capacity stress: permit a longer opaque final parent reference to approach the token point")
    parser.add_argument("--max-stress-claims", type=int, default=1024)
    parser.add_argument("--max-alias-characters", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=30720)
    parser.add_argument("--window-tokens", type=int, default=32768)
    parser.add_argument("--max-claims", type=int, default=512)
    parser.add_argument("--cache-states", type=lambda value: tuple(value.split(",")), default=("cold", "warm"))
    parser.add_argument("--max-total-generations", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return asyncio.run(collect(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
