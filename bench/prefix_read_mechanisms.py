"""Real source-first READ/ANSWER semantics and independent-prefix qualification.

This controlled semantic probe uses actual generated history. Its native and
fixed-TTL arms can have different later input tokens. Fixed-token C attribution
is provided separately by profiling.prefix_retention_probe.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from bench.native_metrics import NativePoolSampler
from bench.reading_mechanisms import fixture_set, save_measurements
from bench.study_resources import StudyResourceSampler, listening_processes
from controller.config import create_controller, load_config
from controller.context_schema import ContextBudget, EvidenceUnit, Need, SourceSpan, TaskState, TextRef, VersionIntent
from controller.evidence import make_evidence
from controller.prefix_retention import PrefixRuntimeClient
from controller.reading import build_read_plan, stable_prefix_input_ids, stable_source_payload
from controller.serialization import to_jsonable
from controller.templates import longest_common_prefix, token_hash


@dataclass(frozen=True)
class Case:
    name: str
    version: str
    question: str
    need_id: str
    operation: str
    expected_literals: tuple[str, ...]


CASES = (
    Case("eligibility-2024", "2024", "Under the 2024 Lumen policy, who is eligible for assistance?",
         "eligibility", "lookup", ("18", "permanent")),
    Case("procedure-2024", "2024", "Keep the 2024 version. What application steps and form does it require?",
         "procedure", "procedure", ("W-7", "Assistance")),
    Case("replace-2025", "2025", "Use only the 2025 version. State its minimum age and required application form.",
         "age-and-form", "lookup", ("21", "W-9")),
    Case("revisit-2024", "2024", "Return to the 2024 version. What application form does that older version require?",
         "old-form", "lookup", ("W-7",)),
)


def material(case):
    age, form, desk = (18, "W-7", "Blue") if case.version == "2024" else (21, "W-9", "Green")
    text = (f"Lumen assistance policy, version {case.version}. Permanent residents aged {age} or older "
        f"are eligible. To apply, open the Lumen desktop app, choose Assistance, and submit Form {form}. "
        f"The {desk} Desk reviews applications.")
    block = make_evidence(text, "lumen-policy", corpus_id="prefix-semantic-probe", version=case.version)
    span = SourceSpan(case.name+":span", block.key, 0, len(text))
    unit = EvidenceUnit(case.name+":window", (span,), text, need_ids=(case.need_id,),
                        document_family="lumen-policy")
    return block, unit


def build_case(case, snapshot, template):
    turn = len(snapshot.raw_history)+1
    state = TaskState("lumen:"+case.name, case.question,
        working_history=tuple(TextRef(item.turn_id, "user", 0, len(item.user_text)) for item in snapshot.raw_history),
        needs=(Need(case.need_id, (TextRef(turn, "user", 0, len(case.question)),), case.operation),),
        current_turn_id=turn, version_intent=VersionIntent("explicit", ("lumen-policy",), (case.version,)))
    evidence = fixture_set(state, (material(case),), name=case.name+":set")
    budget = ContextBudget(window_tokens=32768, body_tokens=4096, history_tokens=4096,
        output_tokens=512, read_input_tokens=12288, read_output_tokens=2048, max_steps=16)
    return build_read_plan(state, evidence, "selected_sources", budget, template=template,
                           snapshot=snapshot, read_layout="source_first", read_packing="source_windows")


async def read_proofs(service, plan, steps):
    client = PrefixRuntimeClient(service.engine)
    proofs = []
    for step in steps:
        if step.task.kind != "READ":
            continue
        _, step, _ = service.store.execution_step(step.attempt_id, step.step_id)
        by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        units = tuple(by_id[identity] for identity in step.task.input_unit_ids)
        rendered = service.template.render(step.input_messages)
        prefix = stable_prefix_input_ids(service.template, plan, step.task, units, rendered)
        source_payload, _ = stable_source_payload(units)
        sources = [{"source_unit": item["source_unit"], "source_spans": item["source_spans"]}
                   for item in source_payload]
        inspected = await client.inspect(receipt=step.receipt, prefix_input_ids=prefix, sources=sources)
        registered = step.metrics.get("evidence_prefix")
        owner = await client.status(step.receipt.engine_restart_id, registered["owner_id"]) if registered else None
        epoch = await service.engine.status(engine_session_id=step.engine_epoch.engine_session_id,
                                            expected_restart_id=step.receipt.engine_restart_id)
        proofs.append(dict(step_id=step.step_id, full_input_hash=step.receipt.input_hash,
            stable_prefix_input_ids=prefix, stable_prefix_hash=token_hash(prefix),
            page_prefix_hash=inspected["prefix_hash"], boundary_tokens=inspected["boundary_tokens"],
            cache_scope_hash=inspected["cache_scope_hash"], sources=sources,
            receipt=to_jsonable(step.receipt), inspection=inspected,
            read_epoch_state=epoch.state, independent_owner_after_read_retire=owner))
    return proofs


def qualify(rows, arm):
    by_name = {row["case"]: row for row in rows if row["status"] == "completed"}
    required = [case.name for case in CASES]
    checks = dict(all_cases_completed=set(by_name) == set(required),
        one_qa_per_answer=all(row["qa_increment"] == row["version_increment"] == 1 for row in by_name.values()),
        literal_answer_checks=all(all(row["quality_literal_checks"].values()) for row in by_name.values()),
        read_epoch_retired=all(proof["read_epoch_state"] == "RETIRED" for row in by_name.values() for proof in row["read_proofs"]))
    if set(by_name) == set(required):
        first, second, changed, revisit = (by_name[name]["read_proofs"][0] for name in required)
        checks.update(same_source_different_need_prefix=first["page_prefix_hash"] == second["page_prefix_hash"],
            changing_question_has_different_full_input=first["full_input_hash"] != second["full_input_hash"],
            changed_version_has_distinct_prefix=first["page_prefix_hash"] != changed["page_prefix_hash"],
            old_version_revisit_same_prefix=first["page_prefix_hash"] == revisit["page_prefix_hash"],
            version_diverges_before_evidence_boundary=longest_common_prefix(first["stable_prefix_input_ids"],
                changed["stable_prefix_input_ids"]) < min(first["boundary_tokens"], changed["boundary_tokens"]))
    if arm == "fixed_ttl":
        checks["independent_owner_survives_read_epoch"] = all(
            proof["independent_owner_after_read_retire"]["reference_present"]
            for row in by_name.values() for proof in row["read_proofs"])
    return checks


async def run_arm(config, output, *, arm, gpu_device, ttl_seconds, background_offline_training=False):
    output.mkdir(parents=True, exist_ok=True)
    settings = {"mode": "native"} if arm == "native" else {"mode": "fixed_ttl", "ttl_seconds": ttl_seconds}
    config = replace(config, state_dir=str(output/"state"), prefix_retention=settings)
    service = create_controller(config)
    await service.engine.profile_prepare("cold")
    await service.engine.info()
    native = NativePoolSampler(config.engine.base_url, gpu_capacity=config.engine.gpu_capacity_tokens,
        host_enabled=config.engine.host_capacity_tokens > 0, model=config.model.name, interval=.1)
    port = urlsplit(config.engine.base_url).port
    physical = StudyResourceSampler((gpu_device,), listening_processes((port,)), interval=.1)
    await native.start()
    await physical.start()
    rows, active = [], None
    start = time.monotonic()
    try:
        for case in CASES:
            before = service.store.read_snapshot("policy-session", "prefix-semantic-probe-v1")
            active = dict(case=case.name, status="started", start=time.monotonic(),
                initial_qa_count=len(before.raw_history), initial_version=before.version)
            plan = build_case(case, before, service.template)
            response = await service.serve_turn("policy-session", case.question, context_plans=(plan,),
                corpus_snapshot_id="prefix-semantic-probe-v1", request_id=case.name)
            response_completed = time.monotonic()
            attempt = service.store.get_attempt(response.attempt_id)
            after = service.store.get_snapshot("policy-session")
            steps = service.store.execution_steps(response.attempt_id)
            proofs = await read_proofs(service, attempt.context_plan, steps)
            row = dict(case=case.name, status="completed", scope="semantic_probe", arm=arm,
                input_origin="controlled_task_and_support_annotations", training_eligible=False,
                question=case.question, actual_history=to_jsonable(before.raw_history),
                plan=to_jsonable(attempt.context_plan), outcome=to_jsonable(attempt.reading_outcome),
                response=to_jsonable(response), answer=response.assistant_text, read_proofs=proofs,
                quality_literal_checks={literal: literal.casefold() in response.assistant_text.casefold()
                    for literal in case.expected_literals},
                total_input_tokens=response.total_input_tokens, total_output_tokens=response.total_output_tokens,
                service_complete_ms=response.complete_ms,
                service_resource_window={"start": active["start"], "end": response_completed},
                post_answer_verification_ms=(time.monotonic()-response_completed)*1000,
                resource_window_scope="planning, service and post-answer native verification",
                elapsed_ms=(time.monotonic()-active["start"])*1000,
                qa_increment=len(after.raw_history)-len(before.raw_history), version_increment=after.version-before.version,
                resource_window={"start": active["start"], "end": time.monotonic()})
            rows.append(row)
            active = None
            (output/f"{case.name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
            print(json.dumps({"arm": arm, "case": case.name, "status": "completed",
                "read_count": response.read_count, "qa_increment": row["qa_increment"]}), flush=True)
    finally:
        failure = sys.exc_info()[1]
        end = time.monotonic()
        if active is not None:
            attempt = service.store.find_request("policy-session", active["case"])
            after = service.store.get_snapshot("policy-session")
            steps = service.store.execution_steps(attempt.attempt_id) if attempt is not None else ()
            staged = service.store.execution_steps_with_stage(attempt.attempt_id) if attempt is not None else ()
            unknown = any(stage == "SUBMITTED" for stage, _ in staged) or (
                attempt is not None and attempt.stage == "SUBMITTED" and attempt.result is None)
            generated = tuple(step for step in steps if step.receipt is not None)
            final_input = len(attempt.plan.candidate.input_ids) if attempt is not None and attempt.result is not None else 0
            final_output = len(attempt.result.output_ids) if attempt is not None and attempt.result is not None else 0
            rows.append(dict(case=active["case"], status="failed", error=str(failure),
                scope="semantic_probe", arm=arm, training_eligible=False,
                attempt=to_jsonable(attempt), steps=to_jsonable(steps),
                qa_increment=len(after.raw_history)-active["initial_qa_count"],
                version_increment=after.version-active["initial_version"],
                elapsed_ms=(end-active["start"])*1000,
                total_input_tokens=None if unknown else sum(len(step.input_ids) for step in generated)+final_input,
                total_output_tokens=None if unknown else sum(len(step.output_ids) for step in generated)+final_output,
                token_accounting_status="unresolved_generation" if unknown else "complete_receipts",
                resource_window={"start": active["start"], "end": end}))
        # Prefix lifetime is independent; experiment cleanup explicitly releases it.
        if service.observe_prefix is not None:
            for record in service.store.prefix_owners(active_only=True):
                await service.observe_prefix.owners.release(record.owner_id, reason="semantic_probe_cleanup")
        await service.aclose()
        service.store.close()
        await native.stop()
        await physical.stop()
        manifest = save_measurements(output, rows, native, physical, start, end, config,
            requested=[case.name for case in CASES], background_offline_training=background_offline_training)
        manifest.update(scope="semantic_probe", arm=arm, training_eligible=False,
            fixed_ttl_role="mechanism_qualification_only", qualification=qualify(rows, arm),
            error=str(failure) if failure else None)
        (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        (output/"native-pool-observation.json").write_text(json.dumps(native.observation(), indent=2)+"\n")
        (output/"physical-resource-observation.json").write_text(json.dumps(physical.observation(), indent=2)+"\n")
    return manifest


async def run(args):
    output = Path(args.output_dir)
    config = load_config(args.config)
    config = replace(config, engine=replace(config.engine, base_url=args.base_url))
    arms = {}
    for arm in ("native", "fixed_ttl"):
        arms[arm] = await run_arm(config, output/arm, arm=arm,
            gpu_device=args.gpu_device, ttl_seconds=args.ttl_seconds,
            background_offline_training=args.background_offline_training)
    manifest = dict(scope="semantic_probe", training_eligible=False,
        attribution="actual generated histories; fixed-token C attribution is a separate probe",
        arms=arms, mechanism_qualified=all(all(arm["qualification"].values()) for arm in arms.values()))
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"scope": manifest["scope"], "mechanism_qualified": manifest["mechanism_qualified"]}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", required=True, help="Dedicated native profiling service; the probe clears its cache")
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ttl-seconds", type=float, default=60)
    parser.add_argument("--background-offline-training", action="store_true")
    report = asyncio.run(run(parser.parse_args()))
    return 0 if report["mechanism_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
