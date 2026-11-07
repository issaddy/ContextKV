"""Run controlled task/support fixtures through the real READ/ANSWER service."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from adapters.history_state import apply_condition_updates
from controller.config import create_controller, load_config
from controller.context_schema import (Condition, ContextBudget, EvidencePool, EvidenceSet,
    EvidenceUnit, Need, SourceSpan, SupportBundle, TaskState, TextRef)
from controller.evidence import make_evidence
from controller.reading import build_read_plan
from controller.serialization import to_jsonable
from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.study_resources import (StudyResourceSampler, attribute_physical,
                                  integrate_physical, listening_processes)


def source(text, name):
    block = make_evidence(text, name, corpus_id="mechanism-fixtures", version="v1")
    span = SourceSpan(f"span:{name}", block.key, 0, len(text))
    return block, EvidenceUnit(name, (span,), text, document_family=name)


def fixture_set(state, material, *, name, disposition="complete"):
    blocks, units = zip(*material)
    support = tuple(SupportBundle(f"fixture:{name}:{need.need_id}", need.need_id,
        tuple(unit.unit_id for unit in units), 1, 1,
        tuple(span for unit in units for span in unit.source_spans), disposition)
        for need in state.needs)
    return EvidenceSet(name, units, need_support=support, quality_prediction=1,
                       source_blocks=blocks)


def save_measurements(output_dir, rows, sampler, physical, start, end, config,
                      *, requested, background_offline_training):
    windows = {row["case"]: (row["resource_window"]["start"], row["resource_window"]["end"])
               for row in rows}
    attributed = attribute_physical(physical.samples, windows)
    for row in rows:
        interval = row["resource_window"]
        series, coverage = pool_series(sampler.observation(), interval["start"], interval["end"])
        row["native_pool_cost"] = {"integrals": occupancy_integral(series), "coverage": coverage}
        row["physical_cost"] = integrate_physical(physical.samples, interval["start"], interval["end"])
        row["attributed_physical_cost"] = attributed[row["case"]]
        (output_dir/f"{row['case']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
    batch_series, batch_coverage = pool_series(sampler.observation(), start, end)
    complete = [row for row in rows if row["status"] == "completed"]
    manifest = {"cases": len(rows), "completed_cases": len(complete), "requested_cases": requested,
        "status": "completed" if len(complete) == len(requested) else "partial",
        "config": to_jsonable(config), "input_origin": "controlled_task_and_support_annotations",
        "background_offline_training": background_offline_training,
        "physical_scope": "declared reader GPU and reader/controller process groups",
        "reader": "qwen3_32b_nonthinking" if config.engine.kind == "sglang" else "mock",
        "total_input_tokens": sum(row["total_input_tokens"] for row in rows)
            if all(row["total_input_tokens"] is not None for row in rows) else None,
        "total_output_tokens": sum(row["total_output_tokens"] for row in rows)
            if all(row["total_output_tokens"] is not None for row in rows) else None,
        "sum_complete_ms": sum(row["elapsed_ms"] for row in rows),
        "qa_increments": [row["qa_increment"] for row in rows],
        "version_increments": [row["version_increment"] for row in rows],
        "physical_batch_cost": integrate_physical(physical.samples, start, end),
        "native_pool_batch_cost": {"integrals": occupancy_integral(batch_series), "coverage": batch_coverage},
        "resource_window": {"start": start, "end": end},
        "records": [str(output_dir/f"{row['case']}.json") for row in rows]}
    (output_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


async def run(config_path, output_dir, *, base_url=None, gpu_device=0,
              background_offline_training=False,
              numeric_need_isolation=False, read_layout="task_first",
              cases=("pair", "correction", "long", "multihop")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if base_url:
        config = replace(config, engine=replace(config.engine, base_url=base_url))
    config = replace(config, state_dir=str(output_dir/"state"))
    service = create_controller(config)
    sampler = NativePoolSampler(config.engine.base_url,
        gpu_capacity=config.engine.gpu_capacity_tokens,
        host_enabled=config.engine.host_capacity_tokens > 0, model=config.model.name, interval=.2)
    await sampler.start()
    endpoint = urlsplit(config.engine.base_url)
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    physical = StudyResourceSampler((gpu_device,), listening_processes((port,)))
    await physical.start()
    batch_started = time.monotonic()
    budget = ContextBudget(window_tokens=32768, body_tokens=4096, history_tokens=4096,
                           output_tokens=1024, read_input_tokens=12288,
                           read_output_tokens=4096, max_steps=64)
    rows = []
    active_case = None
    corpus_snapshot_id = "mechanism-fixtures-v1"

    async def answer(case, session, state, material, *, scope="selected_sources", extend=None, expected=None,
                     support_disposition="complete"):
        nonlocal active_case
        started = time.monotonic()
        snapshot = service.store.read_snapshot(session, corpus_snapshot_id)
        active_case = {"case": case, "session_id": session, "question": state.raw_question,
            "task_state": to_jsonable(state), "expected": expected,
            "start": started, "initial_qa_count": len(snapshot.raw_history), "initial_version": snapshot.version}
        evidence_set = fixture_set(state, material, name=f"{case}:set", disposition=support_disposition)
        plan = build_read_plan(state, evidence_set, scope, budget,
            template=service.template, snapshot=snapshot,
            separate_needs=numeric_need_isolation and any(need.operation in {"count", "sum"} for need in state.needs),
            read_layout=read_layout,
            read_packing="source_windows" if read_layout == "source_first" else "mixed",
            answer_system_prompt=config.context.system_prompt,
            numeric_identity="source_ids" if case == "long" else "names")
        source_tokens = sum(len(service.template.tokenizer.encode(block.text,
            add_special_tokens=False)) for block in evidence_set.source_blocks)
        planning_ms = (time.monotonic()-started)*1000
        response = await service.serve_turn(session, state.raw_question, context_plans=(plan,),
            corpus_snapshot_id=corpus_snapshot_id, request_id=case, open_need_provider=extend)
        attempt = service.store.get_attempt(response.attempt_id)
        committed = service.store.get_snapshot(session)
        row = {"case": case, "status": "completed", "input_origin": "controlled_task_and_support_annotations",
            "background_offline_training": background_offline_training,
            "experiment_role": "controlled_mechanism_execution",
            "measurement_kind": "hardware" if config.engine.kind == "sglang" else "simulation",
            "numeric_need_isolation": numeric_need_isolation, "read_layout": read_layout,
            "model": "qwen3_32b_nonthinking" if config.engine.kind == "sglang" else "mock",
            "question": state.raw_question, "actual_history": to_jsonable(snapshot.raw_history),
            "task_state": to_jsonable(state), "sources": to_jsonable(evidence_set.source_blocks),
            "source_body_tokens": source_tokens, "plan": to_jsonable(attempt.context_plan),
            "outcome": to_jsonable(attempt.reading_outcome), "response": to_jsonable(response),
            "answer": response.assistant_text, "expected": expected,
            "elapsed_ms": (time.monotonic()-started)*1000,
            "total_input_tokens": response.total_input_tokens,
            "total_output_tokens": response.total_output_tokens,
            "pre_service_planning_ms": planning_ms,
            "resource_window": {"start": started, "end": time.monotonic()},
            "qa_increment": len(committed.raw_history)-len(snapshot.raw_history),
            "version_increment": committed.version-snapshot.version}
        rows.append(row)
        active_case = None
        (output_dir/f"{case}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({"case": case, "read_count": response.read_count,
            "reduce_count": response.reduce_count, "answer": response.assistant_text,
            "input_tokens": response.total_input_tokens, "output_tokens": response.total_output_tokens,
            "complete_ms": response.complete_ms}, ensure_ascii=False), flush=True)
        return state, committed

    try:
        question = "Who is eligible for Lumen assistance and how do they apply on Windows?"
        platform = TextRef(1, "user", question.index("Windows"), question.index("Windows")+7)
        old_condition = Condition("platform:windows", value_spans=(platform,), relation_type="PLATFORM")
        state = TaskState("lumen", question, conditions=(old_condition,),
            needs=(Need("apply", (TextRef(1, "user", 0, len(question)),), "procedure",
                        condition_ids=(old_condition.condition_id,)),), current_turn_id=1)
        if "pair" in cases or "correction" in cases:
            material = (source("Windows eligibility: residents aged 18 or older may request Lumen assistance.", "windows-eligibility"),
                        source("Windows application: open the Lumen desktop app, choose Assistance, and submit Form W-7.", "windows-steps"))
            await answer("pair", "condition-session", state, material,
                         expected={"eligibility": "residents aged 18 or older", "form": "W-7"})
        if "correction" in cases:
            question = "Change the platform to Linux. Keep the eligibility requirement and explain the application steps."
            start = question.index("Linux")
            update = Condition("platform:linux", value_spans=(TextRef(2, "user", start, start+5),),
                relation_type="PLATFORM", target_refs=(old_condition.condition_id,),
                supersedes=(old_condition.condition_id,), update="REPLACE")
            corrected = replace(state, raw_question=question, current_turn_id=2,
                working_history=(TextRef(1, "user", 0, len(state.raw_question)),),
                conditions=apply_condition_updates(state.conditions, (update,)),
                needs=tuple(replace(need, condition_ids=(update.condition_id,)) for need in state.needs))
            material = (source("Linux eligibility: residents aged 18 or older may request Lumen assistance.", "linux-eligibility"),
                        source("Linux application: run lumen-assist --apply, enter the residence identifier, and submit Form L-9.", "linux-steps"))
            await answer("correction", "condition-session", corrected, material,
                         expected={"platform": "Linux", "form": "L-9", "command": "lumen-assist --apply"})
        if "long" in cases:
            question = "Across all supplied shipment registers, count distinct approved shipments and sum their masses in kilograms."
            state = TaskState("shipments", question, needs=(Need("count", operation="count"),
                Need("mass", operation="sum")), current_turn_id=1)
            material = []
            background = "The register's filing notes describe administrative storage only; these notes add no shipments. "*400
            for document in range(12):
                records = []
                for item in range(document*3, document*3+3):
                    value = f"{item+1} kg" if item % 2 == 0 else f"{(item+1)*1000} g"
                    records.append(f"Approved shipment S{item:02d}: mass {value}.")
                if document == 11:
                    records.append("Repeated registration of approved shipment S00: mass 1 kg; same shipment as register 00.")
                material.append(source(" ".join(records)+" "+background, f"register-{document:02d}"))
            await answer("long", "long-session", state, tuple(material), scope="all_supplied_documents",
                         expected={"distinct_shipments": 36, "mass_kg": 666})
        if "multihop" in cases:
            question = "In what year was the fund supporting the parent body of Meridian Institute created?"
            state = TaskState("meridian", question, needs=(Need("foundation", operation="infer"),), current_turn_id=1)
            initial = source("Meridian Institute is governed by the Arbor Council.", "institute-parent")
            available = [source("Arbor Council is supported by the Silver Fund.", "parent-fund"),
                         source("Silver Fund was established in 1997.", "fund-year")]
            gap_requests = []

            async def extend(plan, open_needs, claims):
                gap_requests.append(to_jsonable(open_needs))
                # The controlled fixture reveals one source per relation request.
                # Public-corpus runs use SourceDirectory.extend_open_needs instead.
                material = available.pop(0) if available else None
                usage = {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []}
                return (EvidencePool((material[1],), (material[0],), auxiliary_usage=usage)
                        if material else EvidencePool((), auxiliary_usage=usage))

            await answer("multihop", "multihop-session", state, (initial,), extend=extend,
                         support_disposition="partial",
                         expected={"year": 1997, "source_chain": ["institute-parent", "parent-fund", "fund-year"]})
            (output_dir/"multihop-gap-requests.json").write_text(json.dumps(gap_requests, ensure_ascii=False, indent=2)+"\n")
    finally:
        failure = sys.exc_info()[1]
        batch_ended = time.monotonic()
        if active_case is not None:
            attempt = service.store.find_request(active_case["session_id"], active_case["case"])
            steps = service.store.execution_steps(attempt.attempt_id) if attempt is not None else ()
            snapshot = service.store.get_snapshot(active_case["session_id"])
            rows.append({**active_case, "status": "failed", "error": str(failure),
                "input_origin": "controlled_task_and_support_annotations",
                "background_offline_training": background_offline_training,
                "attempt": to_jsonable(attempt), "steps": to_jsonable(steps),
                "context_plan": to_jsonable(service.store.context_plan(attempt.attempt_id))
                    if attempt is not None else None,
                "total_input_tokens": sum(len(step.input_ids) for step in steps),
                "total_output_tokens": sum(len(step.output_ids) for step in steps),
                "elapsed_ms": (batch_ended-active_case["start"])*1000,
                "resource_window": {"start": active_case["start"], "end": batch_ended},
                "qa_increment": len(snapshot.raw_history)-active_case["initial_qa_count"],
                "version_increment": snapshot.version-active_case["initial_version"]})
        await service.aclose()
        service.store.close()
        await sampler.stop()
        await physical.stop()
        (output_dir/"native-pool-observation.json").write_text(
            json.dumps(sampler.observation(), ensure_ascii=False, indent=2)+"\n")
        (output_dir/"physical-resource-observation.json").write_text(
            json.dumps(physical.observation(), ensure_ascii=False, indent=2)+"\n")
        requested = list(dict.fromkeys((*(("pair",) if "correction" in cases else ()), *cases)))
        manifest = save_measurements(output_dir, rows, sampler, physical, batch_started, batch_ended,
            config, requested=requested, background_offline_training=background_offline_training)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_32b.yaml")
    parser.add_argument("--output-dir", default="runs/study/mechanisms")
    parser.add_argument("--base-url")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--background-offline-training", action="store_true")
    parser.add_argument("--numeric-need-isolation", action="store_true")
    parser.add_argument("--read-layout", choices=("task_first", "source_first"), default="task_first")
    parser.add_argument("--cases", nargs="+", choices=("pair", "correction", "long", "multihop"),
                        default=["pair", "correction", "long", "multihop"])
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(run(args.config, args.output_dir,
        base_url=args.base_url, gpu_device=args.gpu_device,
        background_offline_training=args.background_offline_training,
        numeric_need_isolation=args.numeric_need_isolation, read_layout=args.read_layout,
        cases=args.cases)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
