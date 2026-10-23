"""Paired, controlled attribution test for epoch retirement and native eviction.

Requires an isolated server with CONTEXTKV_ENABLE_PROFILING=1. It deliberately
refreshes the old branch after the new branch, so retaining the old reference
competes against the current working set. This is a mechanism test, not a
representative-workload performance or quality claim.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import uuid

from controller.config import load_config
from controller.engine_client import EngineError, SGLangEngineClient
from controller.schema import PromptCandidate
from controller.templates import HuggingFaceTemplate, token_hash


class LifecycleProbe:
    def __init__(self, engine, token_ids, *, shared_tokens=128, branch_tokens=128, seed=42):
        self.engine = engine
        a,b,c,sentinel = random.Random(seed).sample(tuple(dict.fromkeys(token_ids)),4)
        self.shared = (a,)*shared_tokens
        self.old = self.shared+(b,)*branch_tokens
        self.new = self.shared+(c,)*branch_tokens
        self.sentinel = sentinel
        self.branch_tokens = branch_tokens
        self.seed = seed
        self.candidates = []
        for name,ids in (("old",self.old),("new",self.new),("shared",self.shared)):
            query = ids+(self.sentinel,)
            self.candidates.append(PromptCandidate(name,"APPEND",0,(),query,token_hash(query),frozenset(),frozenset()))

    async def fixture(self, mode, **kwargs):
        for retry in range(20):
            try:
                return await self.engine.profile_prepare(mode,**kwargs)
            except EngineError as error:
                if error.code != "ENGINE_BUSY" or retry == 19:
                    raise
                await asyncio.sleep(.05)

    async def snapshot(self):
        batch = await self.engine.quote_batch(self.candidates,self.engine.fingerprint)
        return dict(paths={q.candidate_id:asdict(q) for q in batch.quotes},
                    metrics=await self.engine.metrics(), observed_seq=batch.observed_seq)

    def validate_paths(self, snapshot, tier):
        field = "ready_gpu_prefix_tokens" if tier == "gpu" else "host_only_load_tokens"
        for name,expected in (("old",len(self.old)),("new",len(self.new)),("shared",len(self.shared))):
            observed = snapshot["paths"][name][field]
            if observed != expected:
                raise ValueError(f"{tier} fixture {name} expected {expected} tokens, observed {observed}; "
                                 "check page alignment, host write policy, pool capacity and transfer completion")

    async def trial(self, *, tier, retire_old):
        await self.fixture("cold")
        info = await self.engine.info()
        metrics = await self.engine.metrics()
        if not info.profiling_enabled or info.engine_kind != "sglang":
            raise ValueError("a dedicated patched SGLang profiling server is required")
        page = metrics["pool"]["page_size"]
        if min(len(self.shared),self.branch_tokens) < page or any(
                size % page for size in (len(self.shared),self.branch_tokens)):
            raise ValueError("shared and branch sizes must be positive native-page multiples")
        if tier == "host" and metrics["pool"]["host_capacity_tokens"] == 0:
            raise ValueError("host trial requires HiCache")
        epochs = {}
        results = {}
        try:
            for name in ("old","new"):
                sid = uuid.uuid4().hex
                epochs[name] = await self.engine.prepare(sid,uuid.uuid4().hex,info.engine_restart_id)
            async def generate(name,ids):
                epoch = epochs[name]
                result = await self.engine.generate(input_ids=ids,session_id=epoch.engine_session_id,
                    expected_generation=epoch.session_generation,expected_restart_id=info.engine_restart_id,
                    attempt_id=uuid.uuid4().hex,fingerprint=self.engine.fingerprint,
                    max_new_tokens=1,ignore_eos=True)
                if not result.generation_complete or not result.receipt.session_ref_registered:
                    raise ValueError("fixture generation did not finish with its native session reference")
                return result
            results["old"] = asdict(await generate("old",self.old))
            results["new"] = asdict(await generate("new",self.new))
            # Make old MRU while new remains the logical current epoch. Without
            # this step plain LRU could evict old in both arms and hide the effect.
            results["old_refresh"] = asdict(await generate("old",self.old))
            await self.fixture("settle")
            ready = await self.snapshot()
            self.validate_paths(ready,"gpu")
            host_preparation = None
            if tier == "host":
                # Evict device copies without releasing ANY epoch references.
                # Write-back is allowed to create the corresponding host copies.
                host_preparation = await self.fixture("pressure",tier="gpu",
                    tokens=ready["metrics"]["pool"]["unique_gpu_tokens"])
                await self.fixture("settle")
            before = await self.snapshot()
            self.validate_paths(before,tier)
            retire_ack = None
            if retire_old:
                epoch = epochs["old"]
                retire_ack = await self.engine.retire(epoch.engine_session_id,epoch.session_generation,
                    uuid.uuid4().hex,info.engine_restart_id,"controlled_after_new_completion")
                if not retire_ack.acknowledged:
                    raise ValueError("old epoch remained inflight in an idle controlled fixture")
            after_retire = await self.snapshot()
            pressure = await self.fixture("pressure",tier=tier,tokens=self.branch_tokens)
            after = await self.snapshot()
            field = "ready_gpu_prefix_tokens" if tier == "gpu" else "host_only_load_tokens"
            retained = {name:after["paths"][name][field] for name in ("old","new","shared")}
            old_state = await self.engine.status(engine_session_id=epochs["old"].engine_session_id,
                                                 expected_restart_id=info.engine_restart_id)
            new_state = await self.engine.status(engine_session_id=epochs["new"].engine_session_id,
                                                 expected_restart_id=info.engine_restart_id)
            return dict(arm="retire_old" if retire_old else "keep_old",tier=tier,
                commit_boundary="simulated_after_verified_new_complete_result",
                old_refreshed_after_new=True,engine=asdict(info),generations=results,
                epochs={name:asdict(epoch) for name,epoch in epochs.items()},
                epoch_states={"old":old_state.state,"new":new_state.state},
                host_preparation=host_preparation,before=before,after_retire=after_retire,
                retire_ack=asdict(retire_ack) if retire_ack else None,
                pressure=pressure,after_pressure=after,retained_prefix_tokens=retained,
                retirement_pool_delta={name:before["metrics"]["pool"][name]-after_retire["metrics"]["pool"][name]
                    for name in ("unique_gpu_tokens","unique_host_tokens")})
        finally:
            for epoch in epochs.values():
                await self.engine.retire(epoch.engine_session_id,epoch.session_generation,
                    uuid.uuid4().hex,info.engine_restart_id,"controlled_fixture_cleanup")

    async def run(self, tiers=("gpu","host")):
        info = await self.engine.info()
        if info.engine_kind != "sglang" or not info.profiling_enabled:
            raise ValueError("a dedicated patched SGLang profiling server is required")
        report = dict(source="hardware",scope="controlled_lifecycle_mechanism",seed=self.seed,
            shared_tokens=len(self.shared),branch_tokens=self.branch_tokens,trials={},
            limitation="Explicit native eviction in isolated paired fixtures. This does not establish general latency, throughput or quality gains.")
        for tier in tiers:
            if tier not in {"gpu","host"}:
                raise ValueError("tier must be gpu or host")
            control = await self.trial(tier=tier,retire_old=False)
            treatment = await self.trial(tier=tier,retire_old=True)
            baseline,current = control["retained_prefix_tokens"],treatment["retained_prefix_tokens"]
            effect = (current["new"] > baseline["new"] and current["old"] < baseline["old"]
                      and current["shared"] == len(self.shared)
                      and not any(treatment["retirement_pool_delta"].values())
                      and treatment["pressure"]["native_evicted_tokens"] > 0)
            report["trials"][tier] = dict(control=control,treatment=treatment,
                effect_observed=effect,
                new_prefix_preserved_gain_tokens=current["new"]-baseline["new"],
                old_prefix_retention_change_tokens=current["old"]-baseline["old"])
        report["status"] = "effect_observed" if all(v["effect_observed"] for v in report["trials"].values()) else "no_effect_observed"
        return report


async def _run(args):
    config = load_config(args.config)
    if config.engine.kind != "sglang":
        raise ValueError("engine.kind must be sglang")
    template = HuggingFaceTemplate(args.model_path or config.model.name,config.model.revision,
        namespace_hash=config.model.namespace_hash,local_files_only=True)
    tokens = sorted(set(template.tokenizer.get_vocab().values())-set(template.tokenizer.all_special_ids))
    engine = SGLangEngineClient(args.base_url or config.engine.base_url,template.fingerprint,
                               timeout=config.engine.timeout_seconds,stream=False)
    try:
        report = await LifecycleProbe(engine,tokens,shared_tokens=args.shared_tokens,
            branch_tokens=args.branch_tokens,seed=args.seed).run(args.tiers)
        target = Path(args.report)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"status":report["status"],"report":str(target)}))
    finally:
        await engine.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/sglang.yaml")
    parser.add_argument("--model-path",default=os.environ.get("CONTEXTKV_MODEL_PATH"))
    parser.add_argument("--base-url")
    parser.add_argument("--report",default="runs/sglang-lifecycle.json")
    parser.add_argument("--tiers",type=lambda value:tuple(value.split(",")),default=("gpu","host"))
    parser.add_argument("--shared-tokens",type=int,default=128)
    parser.add_argument("--branch-tokens",type=int,default=128)
    parser.add_argument("--seed",type=int,default=42)
    asyncio.run(_run(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
