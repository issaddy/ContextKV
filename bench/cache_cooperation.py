"""Paired native requests with identical evidence, consumers and competing work."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import uuid

from controller.config import create_model_template, load_config
from controller.context_schema import (ContextPlan, EvidenceSet, Need, ReadResult,
    ReadTask, ReadingOutcome, StepRecord, TaskState)
from controller.engine_client import SGLangEngineClient
from controller.plan_policy import PlanCandidate, PlanStep
from controller.prefix_retention import PrefixReadObserver
from controller.schema import Message, PromptCandidate
from controller.session_store import SessionStore
from profiling.common import prepare_cache_fixture


def prompt(template, name, text, question, output=32):
    messages = (Message("system", "Answer from the supplied source. Return only the requested form code."),
                Message("user", "Source:\n"+text+"\n\nQuestion: "+question))
    rendered = template.render(messages, with_offsets=True)
    start = rendered.text.index(text)
    end = start+len(text)
    covered = [i for i, (a, b) in enumerate(rendered.offsets) if b > start and a < end]
    return PromptCandidate(name, "REBUILD", 0, messages, rendered.input_ids,
        rendered.input_hash, frozenset(), frozenset(), fingerprint=template.fingerprint,
        output_budget=output, source_body_token_ranges=((covered[0], covered[-1]+1),),
        source_prefix_tokens=covered[-1]+1)


def document(template, name, tokens, form):
    line = f"{name} administrative archive. Applications are reviewed by the regional office. "
    ids = template.tokenizer.encode(line*max(1, tokens//8), add_special_tokens=False)[:tokens]
    return template.tokenizer.decode(ids)+f"\n{name} policy: the required application form is {form}."


def result_row(result, elapsed):
    receipt = result.receipt
    return dict(answer=result.text, complete_ms=elapsed,
        prefill_tokens=receipt.actual_prefill_tokens,
        gpu_hit_tokens=receipt.actual_device_hit_tokens,
        host_load_tokens=receipt.actual_host_loaded_tokens,
        output_tokens=len(result.output_ids), server_ttft_ms=result.server_ttft_ms)


async def paired_case(engine, template, *, background_tokens, concurrency, directory,
                      competing_requests=None, competing_tokens=12000):
    source = document(template, "Lumen", 7000, "W-7")
    warm = prompt(template, "read-source", source, "What form does Lumen require?", output=1)
    answer = prompt(template, "answer", source, "Give the Lumen application form code.")
    background = prompt(template, "background", document(template, "Cedar", background_tokens, "C-3"),
                        "What is the form code?", output=1)
    count = competing_requests or concurrency
    competing = tuple(prompt(template, f"competing-{i}",
        document(template, f"Birch-{i}", competing_tokens//count, "B-2"),
        "What is the form code?", output=1) for i in range(count))
    report = dict(background_tokens=background_tokens, concurrency=concurrency,
        source_tokens=warm.source_prefix_tokens, competing_requests=count,
        competing_tokens=competing_tokens, same_evidence_and_requests=True, arms={})
    for arm in ("native", "next_use"):
        await prepare_cache_fixture(engine, "cold")
        info = await engine.info()
        session = uuid.uuid4().hex
        store = SessionStore(directory/(session+".sqlite3"))
        observer = PrefixReadObserver(store, engine, dict(mode="demand", allocation="next_use",
            ttl_seconds=15, horizon_seconds_choices=(15, 60, 180), protection_budget_tokens=8192))

        async def generate(candidate):
            started = time.monotonic()
            epoch = await engine.prepare(uuid.uuid4().hex, uuid.uuid4().hex, info.engine_restart_id)
            value = await engine.generate(input_ids=candidate.input_ids, session_id=epoch.engine_session_id,
                expected_generation=epoch.session_generation, expected_restart_id=info.engine_restart_id,
                attempt_id=uuid.uuid4().hex, fingerprint=template.fingerprint,
                max_new_tokens=candidate.output_budget)
            await engine.retire(epoch.engine_session_id, epoch.session_generation,
                                uuid.uuid4().hex, info.engine_restart_id)
            return value, (time.monotonic()-started)*1000

        started = time.monotonic()
        try:
            source_result, source_ms = await generate(warm)
            background_result, background_ms = await generate(background)
            await prepare_cache_fixture(engine, "settle")
            quoted = await engine.quote_batch((*competing, answer), template.fingerprint)
            source_task = ReadTask("read-source", "READ", ("form",))
            tasks = tuple(ReadTask(candidate.candidate_id, "READ", ("other",)) for candidate in competing)
            next_task = tasks[0]
            answer_task = ReadTask("answer", "ANSWER", ("form",))
            plan = ContextPlan("context", TaskState("policy", "Find the Lumen form.",
                needs=(Need("form"), Need("other"))), EvidenceSet("sources", ()),
                steps=(source_task, *tasks, answer_task))
            source_record = StepRecord(session, source_task.step_id, source_task,
                input_ids=warm.input_ids, receipt=source_result.receipt, metrics={
                    "cache_config_hash": quoted.cache_config_hash,
                    "source_view": {"body_token_ranges": warm.source_body_token_ranges,
                                    "source_prefix_tokens": warm.source_prefix_tokens}})
            state = ReadingOutcome(plan, results=(ReadResult("read-source", elapsed_ms=source_ms),),
                                   steps=(source_record,))
            route = PlanCandidate("continue", "read_once", plan,
                (PlanStep(next_task, (competing[0],)),), "fixed-qwen",
                route_steps=(*tuple(PlanStep(task, (candidate,)) for task, candidate in zip(tasks, competing)),
                             PlanStep(answer_task, (answer,))))
            control_ms, lease = 0.0, None
            if arm == "next_use":
                control_started = time.monotonic()
                options = await observer.demand_options((route,), state, session, quoted,
                    page_size=(await engine.metrics())["pool"]["page_size"],
                    candidate_arrival_ms={answer.candidate_id: background_ms})
                lease = options[competing[0].candidate_id]
                await observer.apply_demand(lease, state, session, batch=quoted)
                control_ms = (time.monotonic()-control_started)*1000
            slots = asyncio.Semaphore(concurrency)
            async def compete(candidate):
                async with slots:
                    return await generate(candidate)
            competitors = await asyncio.gather(*(compete(candidate) for candidate in competing))
            await prepare_cache_fixture(engine, "settle")
            before_answer = await engine.quote_batch((answer,), template.fingerprint)
            final, answer_ms = await generate(answer)
            await observer.finish_demand(state, session, session)
            report["arms"][arm] = dict(complete_ms=(time.monotonic()-started)*1000,
                correct="W-7" in final.text, answer=result_row(final, answer_ms),
                source=result_row(source_result, source_ms), background=result_row(background_result, background_ms),
                competing=[result_row(value, ms) for value, ms in competitors],
                control_ms=control_ms, protected_tokens=lease.protected_tokens if lease else 0,
                lease_count=len(lease.retain) if lease else 0,
                free_gpu_before_competition=quoted.free_gpu_tokens,
                quote_before_answer=dict(gpu=before_answer.quotes[0].ready_gpu_prefix_tokens,
                                        host=before_answer.quotes[0].host_only_load_tokens))
        finally:
            for owner in store.prefix_owners(session, active_only=True):
                await observer.owners.release(owner.owner_id, reason="paired_case_complete")
            store.close()
    native, retained = (report["arms"][name] for name in ("native", "next_use"))
    report["difference"] = dict(
        complete_ms_saved=native["complete_ms"]-retained["complete_ms"],
        answer_ms_saved=native["answer"]["complete_ms"]-retained["answer"]["complete_ms"],
        prefill_tokens_saved=native["answer"]["prefill_tokens"]-retained["answer"]["prefill_tokens"],
        host_load_tokens_saved=native["answer"]["host_load_tokens"]-retained["answer"]["host_load_tokens"])
    return report


async def run(args):
    config = load_config(args.config)
    template = create_model_template(config.model)
    engine = SGLangEngineClient(args.base_url or config.engine.base_url, template.fingerprint,
        timeout=config.engine.timeout_seconds, stream=False,
        sampling=replace(config.sampling, temperature=0), model_scope=config.model.scope.manifest())
    report = dict(scope="same source, warmup, competing requests and final question; native versus demand allocation",
                  cases=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="contextkv-pair-") as work:
            for background, concurrency in args.cases:
                case = await paired_case(engine, template, background_tokens=background,
                    concurrency=concurrency, directory=Path(work),
                    competing_requests=args.competing_requests, competing_tokens=args.competing_tokens)
                report["cases"].append(case)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
                print(json.dumps({"background": background, "concurrency": concurrency,
                                  "difference": case["difference"]}), flush=True)
    finally:
        await engine.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--competing-requests", type=int)
    parser.add_argument("--competing-tokens", type=int, default=12000)
    parser.add_argument("--cases", type=lambda value: tuple(tuple(map(int, case.split(":")))
                        for case in value.split(",")), default=((1024, 1), (20000, 1), (20000, 4)))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
