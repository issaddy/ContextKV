"""Short, isolated native-workload observation acceptance; no performance claim.

Reuses the existing profiler's cold/warm fixture and guarded generate/retire.
Never starts a server. Run only against an explicitly enabled profiling service.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import uuid

from controller.config import create_model_template, load_config
from controller.engine_client import EngineError, SGLangEngineClient
from profiling.prefill import PrefillCase
from profiling.sglang_probe import SGLangProbe


def verify_snapshot(batch):
    """Check the wire against the pinned native formula, not a fitted predictor."""
    ratio,clip=batch.native_new_token_ratio,batch.native_max_new_tokens_clip
    members=batch.native_running_budget_requests
    if ratio is None or clip is None or members is None or batch.native_running_budget_tokens is None:
        raise ValueError("server lacks the new native budget observation fields")
    expected=sum(min(r.max_new_tokens-r.generated_tokens,clip)*(1.0 if r.ignore_eos else ratio)
                 for r in members)
    if not math.isclose(expected,batch.native_running_budget_tokens,rel_tol=1e-12,abs_tol=1e-9):
        raise ValueError("native running offset does not match its exact member list/ratio/clip")
    if batch.native_running_requests is None or batch.native_waiting_requests is None:
        raise ValueError("server lacks native workload members")
    known=[]
    for req in (*members,*batch.native_running_requests,*batch.native_waiting_requests):
        match=req.last_native_match
        if match is None:
            continue
        if not 0 < match.observed_time_ns <= batch.observed_time_ns:
            raise ValueError("native match and quote timestamps violate server monotonic order")
        if not 0 <= match.matched_prefix_tokens <= min(match.input_tokens,
                match.device_prefix_tokens+match.host_hit_tokens):
            raise ValueError("invalid captured native match lengths")
        known.append(match)
    return dict(running=len(batch.native_running_requests),waiting=len(batch.native_waiting_requests),
        formula_matches=True,unknown_waiting=sum(r.last_native_match is None for r in batch.native_waiting_requests),
        known_waiting=sum(r.last_native_match is not None for r in batch.native_waiting_requests),
        maximum_captured_prefix=max((m.matched_prefix_tokens for m in known),default=0),
        maximum_match_age_ms=max(((batch.observed_time_ns-m.observed_time_ns)/1e6 for m in known),default=None))


class ObservationProbe:
    def __init__(self,engine,token_ids,*,seed=42,poll_seconds=.02,timeout_seconds=60):
        self.engine=engine
        self.profiler=SGLangProbe(engine,token_ids,seed=seed)
        self.poll_seconds,self.timeout_seconds=poll_seconds,timeout_seconds
        self.report={"status":"started","measurement_kind":"functional_protocol_probe",
            "performance_validated":False,"quality_validated":False,"quotes":[],"results":[],
            "scope":"controlled three-request fixture; polling perturbs scheduling; no latency win inferred"}

    async def run(self,*,prompt_tokens=9216,prefix_tokens=1024,output_tokens=256,cache_state="gpu_ready"):
        info=await self.engine.info()
        if info.engine_kind!="sglang" or not info.profiling_enabled:
            raise ValueError("requires a dedicated patched SGLang profiling service")
        initial=await self.engine.metrics()
        capacity,page=initial["pool"]["gpu_capacity_tokens"],initial["pool"]["page_size"]
        # Distinct first tokens imply no shared prefix between the three jobs.
        # One fits with headroom, two full prompts cannot fit; warm prefixes fit.
        if (2*prompt_tokens<=capacity or prompt_tokens+output_tokens+page>=capacity
                or 3*prefix_tokens>=capacity or not 0<prefix_tokens<prompt_tokens
                or prefix_tokens%page or output_tokens<2):
            raise ValueError("fixture needs one fitting request, two non-fitting full prompts and three fitting page-aligned prefixes")
        case=PrefillCase(prompt_tokens,prefix_tokens,output_tokens,3,cache_state)
        await self.profiler.prepare(case)
        info=await self.engine.info()
        self.report.update(engine=asdict(info),pool=initial["pool"],cache_config=initial.get("cache_config"),
                           case=asdict(case),poll_seconds=self.poll_seconds,timeout_seconds=self.timeout_seconds)
        tasks=[]
        deadline=time.monotonic()+self.timeout_seconds

        async def observe():
            # An empty candidate list reads only workload scalars: no candidate
            # prefix peek, token payload or new content hashing during polling.
            batch=await self.engine.quote_batch([],self.engine.fingerprint)
            check=verify_snapshot(batch)
            if self.report["quotes"] and batch.observed_time_ns<self.report["quotes"][-1]["quote"]["observed_time_ns"]:
                raise ValueError("quote server clock moved backwards")
            self.report["quotes"].append({"quote":asdict(batch),"check":check})
            return batch,check

        def check_deadline():
            if time.monotonic()>deadline:
                raise TimeoutError("observation fixture exceeded the bounded polling window")

        try:
            tasks.append(asyncio.create_task(self.profiler._generate(self.profiler._inputs[0],output_tokens)))
            while True:
                check_deadline()
                batch,_=await observe()
                if any(r.generated_tokens>0 for r in batch.native_running_requests):
                    break
                if tasks[0].done():
                    await tasks[0]  # surface any real generation error first
                    self.report["first_request_finished_before_running_observation"]=True
                    break
                await asyncio.sleep(self.poll_seconds)
            tasks.extend(asyncio.create_task(self.profiler._generate(ids,output_tokens)) for ids in self.profiler._inputs[1:])
            while not all(t.done() for t in tasks):
                check_deadline()
                await observe()
                await asyncio.sleep(self.poll_seconds)
            results=await asyncio.gather(*tasks)
            await self.profiler._fixture("settle")
            await observe()
            for result in results:
                receipt=result.receipt
                if not result.generation_complete or not receipt.session_ref_registered or len(result.output_ids)!=output_tokens:
                    raise ValueError("guarded generation did not complete its known ignore-EOS budget")
                status=await self.engine.status(engine_session_id=receipt.engine_session_id,
                                                expected_restart_id=info.engine_restart_id)
                if status.state!="RETIRED" or status.inflight:
                    raise ValueError("profiler generation did not finish retirement")
                self.report["results"].append(dict(receipt=asdict(receipt),output_tokens=len(result.output_ids),
                    retired=True,server_ttft_ms=result.server_ttft_ms,server_elapsed_ms=result.server_elapsed_ms))
            # A retired guard must reject another submission before model work.
            receipt=results[0].receipt
            rejected_attempt=uuid.uuid4().hex
            try:
                await self.engine.generate(input_ids=self.profiler._inputs[0],session_id=receipt.engine_session_id,
                    expected_generation=receipt.session_generation,expected_restart_id=info.engine_restart_id,
                    attempt_id=rejected_attempt,fingerprint=self.engine.fingerprint,max_new_tokens=1)
            except EngineError as error:
                # Native /generate wraps aborts in HTTP errors; unlike the
                # context control endpoint it need not preserve a string code.
                # The authoritative terminal rejection is checked below.
                self.report["retired_submission_error"]={"code":error.code,"message":str(error)}
            else:
                raise ValueError("retired epoch admitted a new request")
            rejected=await self.engine.status(attempt_id=rejected_attempt,expected_restart_id=info.engine_restart_id)
            if rejected.state!="REJECTED" or not rejected.terminal:
                raise ValueError("rejected guard was not recorded as terminal")
            checks=[row["check"] for row in self.report["quotes"]]
            coverage={"one_running_two_waiting":any(c["running"]==1 and c["waiting"]==2 for c in checks),
                "unknown_waiting_match":any(c["unknown_waiting"]>0 for c in checks),
                "known_waiting_match":any(c["known_waiting"]>0 for c in checks),
                "positive_prefix_match":any(c["maximum_captured_prefix"]>=prefix_tokens for c in checks)}
            self.report.update(coverage=coverage,stale_epoch_rejected=True,
                               metrics_after=await self.engine.metrics())
            # Scheduling can advance before an HTTP observation. Lack of a
            # particular state is sampling coverage, not a violated patch contract.
            self.report.update(status="passed",contract_checks_passed=True,
                coverage_status="complete" if all(coverage.values()) else "partial")
            return self.report
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)


async def _run(args):
    config=load_config(args.config)
    template=create_model_template(config.model,model_path=args.model_path,local_files_only=True)
    special=set(template.tokenizer.all_special_ids)
    tokens=sorted(set(template.tokenizer.get_vocab().values())-special)
    engine=SGLangEngineClient(args.base_url,template.fingerprint,timeout=config.engine.timeout_seconds, sampling=config.sampling,stream=False,
                               **({"model_scope":config.model.scope.manifest()} if config.model.scope else {}))
    probe=ObservationProbe(engine,tokens,seed=args.seed,timeout_seconds=args.timeout_seconds)
    target=Path(args.report)
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        with target.open("x") as stream:
            try:
                await probe.run(prompt_tokens=args.prompt_tokens,prefix_tokens=args.prefix_tokens,
                    output_tokens=args.output_tokens,cache_state=args.cache_state)
            except Exception as error:
                probe.report.update(status="failed",error=f"{type(error).__name__}: {error}")
                raise
            finally:
                json.dump(probe.report,stream,indent=2,allow_nan=False)
                stream.write("\n")
        print(json.dumps({"status":probe.report["status"],"report":str(target),
                          "coverage_status":probe.report["coverage_status"],"coverage":probe.report["coverage"]}))
    finally:
        await engine.aclose()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/qwen3_32b_scope.example.yaml")
    parser.add_argument("--base-url",required=True,help="explicit dedicated profiling service")
    parser.add_argument("--model-path",default=os.environ.get("CONTEXTKV_MODEL_PATH"))
    parser.add_argument("--report",required=True,help="new JSON file, including partial state on failure")
    parser.add_argument("--cache-state",choices=("gpu_ready","host_only"),default="gpu_ready")
    parser.add_argument("--prompt-tokens",type=int,default=9216)
    parser.add_argument("--prefix-tokens",type=int,default=1024)
    parser.add_argument("--output-tokens",type=int,default=256)
    parser.add_argument("--timeout-seconds",type=float,default=60)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(argv)
    if args.timeout_seconds<=0:
        parser.error("timeout must be positive")
    asyncio.run(_run(args))


if __name__=="__main__":
    main()
