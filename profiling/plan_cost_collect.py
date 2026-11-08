"""Collect declared input-ceiling cost probes with real original-source text.

The fixed choice is a profiling intervention, not a trained plan policy.  Every
ANSWER uses the actual finite rolling-proposal compiler, a native quote and the
normal service's generation/QA commit/response tail.  No quality conclusion or
C retention training example is produced by these geometry probes.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

import yaml

from controller.candidates import build_plan_candidates
from controller.config import create_controller, load_config
from controller.engine_client import EngineError
from controller.context_schema import (AnswerPlan, AnswerPoint, ContextBudget, ContextPlan, EvidenceBlock,
    EvidenceSet, EvidenceUnit, Need, ReadTask, ReadingOutcome, SourceSpan, TaskState)
from controller.reading import output_schema, step_messages
from controller.rolling_policy import RollingPlanRuntime, RollingResult
from controller.schema import Plan
from controller.serialization import decode, to_jsonable
from profiling.retention_capture_cost import CaptureRunAccounting, load_resident_provider
from profiling.retention_trace import RecordingEngine, digest, export_attempt, make_trace, write_json


class PackingTargetUnavailable(ValueError):
    def __init__(self, message, *, nearest_plan=None, packing=None):
        super().__init__(message)
        self.nearest_plan, self.packing = nearest_plan, packing


def original_source_bank(paths, *, split, maximum, source_banks=()):
    bank, seen, question, origins = [], set(), None, []
    for path in paths:
        trace = json.loads(Path(path).read_text())
        for turn in trace["turns"]:
            if turn["split"] != split:
                continue
            plan = decode(ContextPlan, turn["context_plan"])
            question = question or turn["query"]
            families = {span.source_key: unit.document_family for unit in plan.selected_set.selected_units
                        for span in unit.source_spans}
            for block in plan.selected_set.source_blocks:
                base = block.span[0] if block.span is not None else 0
                identity = block.key, base, block.text
                if identity in seen or not block.text:
                    continue
                seen.add(identity)
                key = digest(to_jsonable((block.key, base, len(block.text))))[:24]
                unit = EvidenceUnit("cost-source:"+key,
                    (SourceSpan("cost-span:"+key, block.key, base, base+len(block.text)),), block.text,
                    heading=block.title, document_family=families.get(block.key) or block.key.document_id,
                    need_ids=("cost:need",), original_passage_ids=(block.key.passage_id,))
                bank.append((unit, block))
                origins.append(dict(source_trace=str(Path(path).resolve()), source_attempt=turn["attempt_id"],
                    source_key=to_jsonable(block.key), original_start=base, original_end=base+len(block.text)))
                if len(bank) >= maximum:
                    return tuple(bank), question, origins
    for path in source_banks:
        payload = json.loads(Path(path).read_text())
        if payload["format"] != "contextkv_original_source_bank_v1" or payload["split"] != split:
            raise ValueError("original-source bank must declare its cost-calibration partition and format")
        question = question or payload["question"]
        for item in payload["blocks"]:
            block = decode(EvidenceBlock, item["block"])
            base = block.span[0] if block.span is not None else 0
            identity = block.key, base, block.text
            if identity in seen or not block.text:
                continue
            seen.add(identity)
            key = digest(to_jsonable((block.key, base, len(block.text))))[:24]
            bank.append((EvidenceUnit("cost-source:"+key,
                (SourceSpan("cost-span:"+key, block.key, base, base+len(block.text)),), block.text,
                heading=block.title, document_family=item.get("document_family") or block.key.document_id,
                need_ids=("cost:need",), original_passage_ids=(block.key.passage_id,)), block))
            origins.append(dict(source_bank=str(Path(path).resolve()), origin=item["origin"],
                source_key=to_jsonable(block.key), original_start=base, original_end=base+len(block.text)))
            if len(bank) >= maximum:
                return tuple(bank), question, origins
    if not bank:
        raise ValueError("cost probes need original source blocks from the declared partition")
    return tuple(bank), question, origins


def probe_plan(units, bank, question, *, phase, target, read_output, answer_output, window):
    keys = {span.source_key for unit in units for span in unit.source_spans}
    blocks = {block.key: block for _, block in bank if block.key in keys}
    state = TaskState("cost-probe", question, needs=(Need("cost:need", operation="lookup"),),
        source_scope=tuple(sorted(keys)), current_turn_id=1)
    selected = EvidenceSet("cost-probe-sources", tuple(units), source_blocks=tuple(blocks.values()))
    budget = ContextBudget(window_tokens=window, body_tokens=window*4 if phase == "ANSWER" else 8192,
        history_tokens=0, output_tokens=answer_output, read_input_tokens=target,
        read_output_tokens=read_output, max_total_tokens=window*8)
    answer = AnswerPlan((AnswerPoint("cost:need", source_spans=tuple(
        span for unit in units for span in unit.source_spans)),))
    plan = ContextPlan("cost-probe-plan", state, selected, answer_plan=answer, budget=budget,
        layouts=("existing", "need_order", "stable_evidence"), read_layout="source_first", read_packing="mixed")
    if phase == "READ":
        task = ReadTask("cost:READ", "READ", ("cost:need",), tuple(unit.unit_id for unit in units),
            input_budget=target, output_budget=read_output, operation="extract")
        task = replace(task, output_schema=output_schema(task, units, reading_mode=plan.reading_mode))
        plan = replace(plan, steps=(task,))
    return plan


def pack_original_sources(bank, question, *, template, snapshot, policy, phase, target,
                          read_output, answer_output, window, max_renders):
    """Hit the declared actual tokenizer length using original Unicode endpoints.

    A bounded search may return no exact endpoint.  It never inserts random
    tokens, repeated padding, altered source text or an expanded measured bin.
    """
    renders, observations = 0, []
    best, below, above = None, None, None

    def evaluate(units, endpoint):
        nonlocal renders, best, below, above
        if renders >= max_renders:
            raise PackingTargetUnavailable(json.dumps(dict(reason="packing_render_budget",
                target=target, renders=renders, nearest=best[0] if best else None)),
                nearest_plan=below[1] if below is not None and above is not None else None,
                packing=dict(renders=renders, trials=observations, requested_input_tokens=target,
                    actual_input_tokens=below[0] if below is not None else None,
                    observed_bracket_tokens=[below[0], above[0]] if below is not None and above is not None else None,
                    point="actual_observed_below_target; no range expansion"))
        plan = probe_plan(units, bank, question, phase=phase, target=target,
            read_output=read_output, answer_output=answer_output, window=window)
        if phase == "READ":
            messages = step_messages(plan, plan.steps[0], tuple(units), snapshot=snapshot)
            tokens = len(template.render(messages).input_ids)
        else:
            candidate = build_plan_candidates(snapshot, question, plan, template, policy=policy,
                outcome=ReadingOutcome(plan), layouts=("existing",))[0]
            tokens = len(candidate.input_ids)
        renders += 1
        observations.append(dict(source_windows=len(units), last_original_end=endpoint,
                                 actual_input_tokens=tokens))
        if best is None or abs(tokens-target) < abs(best[0]-target):
            best = tokens, plan
        if tokens <= target and (below is None or tokens > below[0]):
            below = tokens, plan
        if tokens > target and (above is None or tokens < above[0]):
            above = tokens, plan
        return tokens, plan

    count, previous = 1, 0
    while True:
        units = tuple(unit for unit, _ in bank[:count])
        tokens, plan = evaluate(units, units[-1].source_spans[0].end)
        if tokens == target:
            return plan, dict(renders=renders, trials=observations, exact_input_tokens=tokens)
        if tokens > target:
            break
        if count == len(bank):
            raise PackingTargetUnavailable(json.dumps(dict(reason="original_source_bank_below_target",
                target=target, actual_input_tokens=tokens, source_windows=count, renders=renders)))
        previous, count = count, min(len(bank), count*2)
    low, high = previous+1, count
    while low < high:
        middle = (low+high)//2
        units = tuple(unit for unit, _ in bank[:middle])
        tokens, plan = evaluate(units, units[-1].source_spans[0].end)
        if tokens == target:
            return plan, dict(renders=renders, trials=observations, exact_input_tokens=tokens)
        if tokens >= target:
            high = middle
        else:
            low = middle+1
    count = low
    prefix = tuple(unit for unit, _ in bank[:count-1])
    tail = bank[count-1][0]

    def cut(length):
        return replace(tail, text=tail.text[:length], source_spans=(replace(tail.source_spans[0],
            end=tail.source_spans[0].start+length),))

    low, high, nearest = 1, len(tail.text), None
    while low <= high:
        middle = (low+high)//2
        unit = cut(middle)
        tokens, plan = evaluate((*prefix, unit), unit.source_spans[0].end)
        if nearest is None or abs(tokens-target) < nearest[0]:
            nearest = abs(tokens-target), middle
        if tokens == target:
            return plan, dict(renders=renders, trials=observations, exact_input_tokens=tokens)
        if tokens < target:
            low = middle+1
        else:
            high = middle-1
    # BPE counts and serialized coordinate hashes need not be monotone at every
    # character. Inspect nearby genuine endpoints within the same fixed budget.
    for distance in range(1, max_renders):
        for length in (nearest[1]-distance, nearest[1]+distance):
            if not 1 <= length <= len(tail.text):
                continue
            unit = cut(length)
            tokens, plan = evaluate((*prefix, unit), unit.source_spans[0].end)
            if tokens == target:
                return plan, dict(renders=renders, trials=observations, exact_input_tokens=tokens)
    raise PackingTargetUnavailable("no actual original-source endpoint reaches the declared input ceiling")


class FixedChoiceCostProbe:
    """Execute the real rolling terminal compiler/quote with an explicit choice."""
    def __init__(self, reader_checkpoint, target, proposal_count, *, output_tokens_per_fact=64):
        self.reader_checkpoint, self.target, self.proposal_count = reader_checkpoint, target, proposal_count
        self.output_budgets = (2048,)
        self.output_tokens_per_fact = output_tokens_per_fact
        self.native = self

    def bind_resident_models(self, manifest):
        self.resident_model_manifest = manifest

    async def start(self):
        return

    async def aclose(self):
        return

    async def run(self, attempt_id, snapshot, context_plans, executor, engine, template,
                  cost_model, compile_answers, *, arrived, auxiliary_usage, events,
                  open_need_provider=None):
        state = ReadingOutcome(context_plans[0], auxiliary_usage={
            "input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0, "models": []})
        variants = tuple(replace(context_plans[0], plan_id="cost-proposal:"+str(index))
                         for index in range(self.proposal_count))
        selection_started = time.monotonic()
        plans = RollingPlanRuntime.proposals(self, variants, state, snapshot, template, compile_answers)
        candidates = tuple(candidate for plan in plans for step in plan.steps
                           for candidate in step.variants if candidate.feasible)
        quotes = await engine.quote_batch(candidates, template.fingerprint)
        selected = next(candidate for candidate in candidates if candidate.layout_id == "existing"
                        and candidate.action == "REBUILD")
        if len(selected.input_ids) != self.target:
            raise ValueError("actual rolling terminal compilation differs from declared profiling input ceiling")
        chosen = next(plan for plan in plans if plan.plan_id == selected.context_plan_id)
        audit = dict(scope="fixed_choice_cost_probe", quality_claim=False, selected_input_tokens=len(selected.input_ids),
            finite_proposals=len(plans), compiled_candidates=len(candidates), actual_quote=to_jsonable(quotes),
            selection_started=selection_started, selection_completed=time.monotonic())
        events.emit("cost_probe_terminal_choice", attempt_id=attempt_id, **audit)
        return RollingResult(replace(state, plan=chosen.context_plan),
            Plan(selected, quote_seq=quotes.observed_seq, reason="fixed_choice_cost_probe"), quotes,
            selection_started)


async def collect(args):
    config = load_config(args.config)
    config = replace(config, engine=replace(config.engine, base_url=args.base_url),
        state_dir=str(Path(args.output_dir)/"state"), prefix_retention={"mode": "native"}, plan_gain=None,
        context=replace(config.context, max_total_model_tokens=args.window_tokens,
                        max_new_tokens=args.answer_output),
        planner=replace(config.planner, strategy="always_rebuild"))
    cases = [(phase, target, output, state, repetition)
        for phase, targets in (("READ", args.read_inputs), ("ANSWER", args.answer_inputs)) if phase in args.phases
        for target in targets for output in (args.read_outputs if phase == "READ" else (args.answer_output,))
        for state in args.cache_states for repetition in range(args.repetitions)]
    if sum(2 if case[0] == "READ" else 1 for case in cases) > args.max_total_generations:
        raise ValueError("declared cost probe matrix exceeds its model-generation budget")
    turns, case_ledger = [], []
    async with CaptureRunAccounting(config, args.output_dir, args.gpu_device,
            resource_gpu_devices=args.resource_gpu_devices,
            resource_process_roots=args.resource_process_roots) as accounting:
        service = create_controller(config)
        resident_provider = None
        output_tokens_per_fact = 64
        if args.study_config is not None:
            study_config = yaml.safe_load(Path(args.study_config).read_text())
            output_tokens_per_fact = study_config.get("reading", {}).get("output_tokens_per_fact", 64)
            resident_provider, accounting.resident_model_manifest = await asyncio.to_thread(
                load_resident_provider, args.study_config, service.template)
            accounting.record_resident_models(args.study_config, accounting.resident_model_manifest)
        accounting.case_outcomes = case_ledger
        recorder = RecordingEngine(service.engine)
        service.engine = recorder
        accounting.calls = recorder.calls
        bank, question, origins = original_source_bank(args.source_trace, split=args.split,
            maximum=args.max_source_windows, source_banks=args.source_bank)
        write_json(Path(args.output_dir)/"source-bank.json", dict(scope="real_original_source_windows",
            origins=origins, question=question, quality_claim=False))
        try:
            for index, (phase, target, output, cache_state, repetition) in enumerate(cases):
                session_id = "physical-cost-probe:"+str(index)
                snapshot = service.store.read_snapshot(session_id, "cost-probe-dev-originals")
                accounting.current_source = dict(scope="cost_protocol_probe", phase=phase, split=args.split,
                    target_input_tokens=target, output_budget=output, cache_state=cache_state,
                    repetition=repetition)
                tick = time.monotonic()
                try:
                    plan, packing = pack_original_sources(bank, question, template=service.template,
                        snapshot=snapshot, policy=service.policy, phase=phase, target=target,
                        read_output=output if phase == "READ" else max(args.read_outputs),
                        answer_output=args.answer_output, window=args.window_tokens,
                        max_renders=args.max_packing_renders)
                except PackingTargetUnavailable as error:
                    if args.require_exact_input or error.nearest_plan is None:
                        case_ledger.append({**accounting.current_source, "status": "input_geometry_unavailable",
                                            "reason": str(error), "preparation_ms": (time.monotonic()-tick)*1000})
                        write_json(Path(args.output_dir)/"cost-probe-cases.json", case_ledger)
                        continue
                    plan, packing = error.nearest_plan, error.packing
                actual_target = packing.get("actual_input_tokens", packing.get("exact_input_tokens"))
                if cache_state == "cold":
                    await service.engine.profile_prepare("cold")
                service.rolling_policy = (FixedChoiceCostProbe(service.template.fingerprint.model_revision,
                    actual_target, args.proposal_count, output_tokens_per_fact=output_tokens_per_fact)
                    if phase == "ANSWER" else None)
                if service.rolling_policy is not None:
                    service.rolling_policy.bind_resident_models(accounting.resident_model_manifest)
                cursor = len(recorder.calls)
                try:
                    response = await service.serve_turn(session_id, question, context_plans=(plan,),
                        corpus_snapshot_id=snapshot.corpus_snapshot_id, request_id="cost-probe:"+str(index))
                except EngineError as error:
                    if error.code != "READ_STEP_OUTPUT_TRUNCATED":
                        raise
                    completed = recorder.calls[cursor:]
                    if not completed or any(not call.get("receipt", {}).get("generation_complete") for call in completed):
                        raise
                    cleanup = []
                    for call in completed:
                        request = call["request"]
                        ack = await service.engine.retire(request["session_id"], request["expected_generation"],
                            "cost-probe-completed-failure:"+request["attempt_id"], request["expected_restart_id"],
                            "completed_truncated_cost_probe")
                        if not ack.acknowledged:
                            raise ValueError("completed cost-probe cleanup is still pending")
                        cleanup.append(to_jsonable(ack))
                    case_ledger.append({**accounting.current_source, "status": "completed_native_failed_phase",
                        "reason": str(error), "packing": packing, "cleanup": cleanup,
                        "actual_native_receipts": [call["receipt"] for call in completed]})
                    write_json(Path(args.output_dir)/"cost-probe-cases.json", case_ledger)
                    await service.close_session(session_id)
                    continue
                attempt = service.store.get_attempt(response.attempt_id)
                metadata = dict(scope="cost_protocol_probe", quality_claim=False, split=args.split,
                    dataset="original_source_geometry", response=to_jsonable(response),
                    max_new_tokens=args.answer_output, source_trace_files=args.source_trace,
                    source_bank_files=args.source_bank,
                    probe=accounting.current_source, packing=packing)
                turns.append(export_attempt(service.store, attempt, service.template, {}, metadata,
                                            recorded_calls=recorder.calls[cursor:]))
                trace = make_trace(turns, config=config, source_db=(args.source_trace or args.source_bank)[0], captured=True)
                trace["quality_scope"] = "none; explicit fixed-choice full-phase cost probes"
                write_json(Path(args.output_dir)/"capture-trace.json", trace)
                case_ledger.append({**accounting.current_source, "status": "completed", "packing": packing,
                    "attempt_id": response.attempt_id, "answer_phase_window": response.answer_phase_window})
                write_json(Path(args.output_dir)/"cost-probe-cases.json", case_ledger)
                await service.close_session(session_id)
                print(json.dumps({"case": index, "phase": phase, "input_tokens": actual_target,
                                  "output_budget": output, "cache_state": cache_state}), flush=True)
        finally:
            await service.aclose()
            service.store.close()
    return case_ledger


def main(argv=None):
    def integers(value):
        return tuple(map(int, value.split(",")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", required=True, help="Dedicated idle profiling-enabled native service")
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--resource-gpu-devices", type=integers)
    parser.add_argument("--resource-process-roots", type=integers, default=())
    parser.add_argument("--study-config", help="Actually load/keep the online query/task/support model assets")
    parser.add_argument("--source-trace", action="append", default=[])
    parser.add_argument("--source-bank", action="append", default=[], help="Original blocks and origins; no prior DAG assertion")
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phases", type=lambda value: tuple(value.split(",")), default=("READ", "ANSWER"))
    parser.add_argument("--read-inputs", type=integers, default=(12288,))
    parser.add_argument("--answer-inputs", type=integers, default=(31744,))
    parser.add_argument("--read-outputs", type=integers, default=(512, 1024, 2048))
    parser.add_argument("--answer-output", type=int, default=1024)
    parser.add_argument("--window-tokens", type=int, default=32768)
    parser.add_argument("--cache-states", type=lambda value: tuple(value.split(",")), default=("cold", "warm"))
    parser.add_argument("--proposal-count", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-source-windows", type=int, default=512)
    parser.add_argument("--max-packing-renders", type=int, default=96)
    parser.add_argument("--require-exact-input", action="store_true",
        help="Otherwise bounded search may execute the actual observed point below a bracketing target")
    parser.add_argument("--max-total-generations", type=int, default=16)
    return asyncio.run(collect(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
