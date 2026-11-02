"""Fixed-token paired probe of independent executed-evidence prefix owners."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import uuid

from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from controller.config import create_model_template, load_config
from controller.engine_client import SGLangEngineClient
from controller.prefix_retention import PrefixRuntimeClient
from profiling.lifecycle_probe import LifecycleProbe


class EvidencePrefixProbe(LifecycleProbe):
    async def trial(self, *, tier, retain):
        await self.fixture("cold")
        info = await self.engine.info()
        prefix_client = PrefixRuntimeClient(self.engine)
        metrics = await self.engine.metrics()
        page = metrics["pool"]["page_size"]
        if min(len(self.shared), self.branch_tokens) < page or any(
                size % page for size in (len(self.shared), self.branch_tokens)):
            raise ValueError("probe sizes must be positive native-page multiples")
        if tier == "host" and not metrics["pool"]["host_capacity_tokens"]:
            raise ValueError("host trial requires HiCache")
        epochs, generations, tokens = {}, {}, {}
        owner_id = uuid.uuid4().hex
        retain_ack = None
        trial_start = time.monotonic()
        try:
            for name in ("old", "new"):
                epoch = await self.engine.prepare(uuid.uuid4().hex, uuid.uuid4().hex, info.engine_restart_id)
                epochs[name] = epoch
            async def generate(name, ids):
                epoch = epochs[name]
                tokens[name] = tuple(ids)
                result = await self.engine.generate(input_ids=ids, session_id=epoch.engine_session_id,
                    expected_generation=epoch.session_generation, expected_restart_id=info.engine_restart_id,
                    attempt_id=uuid.uuid4().hex, fingerprint=self.engine.fingerprint,
                    max_new_tokens=1, ignore_eos=True)
                if not result.generation_complete or not result.receipt.session_ref_registered:
                    raise ValueError("native generation did not commit its actual session reference")
                return result
            generations["old"] = asdict(await generate("old", self.old))
            generated_new = await generate("new", self.new)
            generations["new"] = asdict(generated_new)
            await self.fixture("settle")
            if retain:
                retain_ack = await prefix_client.retain(owner_id=owner_id, receipt=generated_new.receipt,
                    prefix_input_ids=self.new, sources=[{"source_id": "probe-new", "version": "v1"}],
                    ttl_seconds=60)
                if not retain_ack["reference_present"] or retain_ack["boundary_tokens"] != len(self.new):
                    raise ValueError("independent prefix endpoint was not registered")
            # Keep the competing old branch MRU in both arms. Only treatment
            # changes the new source's independent reference, never the tokens.
            generations["old_refresh"] = asdict(await generate("old", self.old))
            await self.fixture("settle")
            before_retire = await self.snapshot()
            retire_acks = []
            for epoch in epochs.values():
                retire_acks.append(asdict(await self.engine.retire(epoch.engine_session_id,
                    epoch.session_generation, uuid.uuid4().hex, info.engine_restart_id)))
            after_retire = await self.snapshot()
            independent_after_retire = await prefix_client.status(info.engine_restart_id, owner_id)
            if retain and not independent_after_retire["reference_present"]:
                raise ValueError("business epoch retirement released independent prefix ownership")
            if tier == "host":
                await self.fixture("pressure", tier="gpu",
                    tokens=after_retire["metrics"]["pool"]["unique_gpu_tokens"])
                await self.fixture("settle")
            ready = await self.snapshot()
            self.validate_paths(ready, tier)
            pressure = await self.fixture("pressure", tier=tier, tokens=self.branch_tokens)
            after_pressure = await self.snapshot()
            status = await prefix_client.status(info.engine_restart_id, owner_id)
            field = "ready_gpu_prefix_tokens" if tier == "gpu" else "host_only_load_tokens"
            revisit_start = time.monotonic()
            epochs["revisit"] = await self.engine.prepare(uuid.uuid4().hex, uuid.uuid4().hex,
                                                         info.engine_restart_id)
            revisit = await generate("revisit", self.new+(self.sentinel,))
            revisit_end = time.monotonic()
            generations["revisit"] = asdict(revisit)
            return dict(arm="retain_prefix" if retain else "native", tier=tier,
                generations=generations, input_token_paths={name: list(ids) for name, ids in tokens.items()},
                input_hashes={name: value["receipt"]["input_hash"] for name, value in generations.items()},
                retain_ack=retain_ack, retire_acks=retire_acks,
                before_retire=before_retire, after_retire=after_retire,
                independent_after_retire=independent_after_retire,
                before_pressure=ready, pressure=pressure, after_pressure=after_pressure,
                prefix_status=status, new_prefix_hit_tokens=after_pressure["paths"]["new"][field],
                trial_window={"start": trial_start, "end": time.monotonic()},
                revisit_window={"start": revisit_start, "end": revisit_end},
                revisit_response_ms=(revisit_end-revisit_start)*1000)
        finally:
            await prefix_client.release(owner_id, info.engine_restart_id)
            for epoch in epochs.values():
                await self.engine.retire(epoch.engine_session_id, epoch.session_generation,
                    uuid.uuid4().hex, info.engine_restart_id)

    async def run(self, tiers=("gpu", "host")):
        info = await self.engine.info()
        if info.engine_kind != "sglang" or not info.profiling_enabled:
            raise ValueError("an isolated native profiling service is required")
        trials = {}
        report = self.report = dict(source="hardware",
            scope="fixed_token_independent_evidence_prefix_lifecycle", engine=asdict(info),
            seed=self.seed, trials=trials, status="running")
        for tier in tiers:
            paired = trials[tier] = {}
            native = await self.trial(tier=tier, retain=False)
            paired["native"] = native
            retained = await self.trial(tier=tier, retain=True)
            paired["retained"] = retained
            if native["input_hashes"] != retained["input_hashes"]:
                raise ValueError("paired C probe input token paths changed")
            gain = retained["new_prefix_hit_tokens"]-native["new_prefix_hit_tokens"]
            paired.update(extra_hit_tokens=gain, effect_observed=gain > 0,
                prefill_tokens_saved=native["generations"]["revisit"]["receipt"]["actual_prefill_tokens"]-
                    retained["generations"]["revisit"]["receipt"]["actual_prefill_tokens"],
                host_loaded_tokens_change=retained["generations"]["revisit"]["receipt"]["actual_host_loaded_tokens"]-
                    native["generations"]["revisit"]["receipt"]["actual_host_loaded_tokens"],
                response_ms_saved=native["revisit_response_ms"]-retained["revisit_response_ms"])
        report["status"] = "effect_observed" if all(row["effect_observed"] for row in trials.values()) else "no_effect_observed"
        return report


async def run(args):
    config = load_config(args.config)
    template = create_model_template(config.model, model_path=args.model_path, local_files_only=True)
    token_ids = sorted(set(template.tokenizer.get_vocab().values())-set(template.tokenizer.all_special_ids))
    engine = SGLangEngineClient(args.base_url or config.engine.base_url, template.fingerprint,
        timeout=config.engine.timeout_seconds, sampling=config.sampling, stream=False,
        **({"model_scope": config.model.scope.manifest()} if config.model.scope else {}))
    base_url = args.base_url or config.engine.base_url
    native = NativePoolSampler(base_url, gpu_capacity=config.engine.gpu_capacity_tokens,
        host_enabled=config.engine.host_capacity_tokens > 0, model=config.model.name, interval=.1)
    physical = StudyResourceSampler((args.gpu_device,), listening_processes((int(base_url.rsplit(":", 1)[1]),)),
                                    interval=.1)
    await native.start()
    await physical.start()
    probe = EvidencePrefixProbe(engine, token_ids, shared_tokens=args.shared_tokens,
        branch_tokens=args.branch_tokens, seed=args.seed)
    probe.report = dict(status="started", trials={})
    experiment_start = time.monotonic()
    try:
        await probe.run(args.tiers)
    finally:
        failure = sys.exc_info()[1]
        experiment_end = time.monotonic()
        await native.stop()
        await physical.stop()
        await engine.aclose()
        report = probe.report
        if failure is not None:
            report.update(status="failed", error=str(failure))
        for paired in report["trials"].values():
            for arm in ("native", "retained"):
                if arm not in paired:
                    continue
                row = paired[arm]
                row["cost"] = {}
                for name in ("trial", "revisit"):
                    window = row[name+"_window"]
                    series, coverage = pool_series(native.observation(), window["start"], window["end"])
                    row["cost"][name] = dict(native={"integrals": occupancy_integral(series), "coverage": coverage},
                        physical=integrate_physical(physical.samples, window["start"], window["end"]))
        series, coverage = pool_series(native.observation(), experiment_start, experiment_end)
        report["experiment_cost"] = dict(native={"integrals": occupancy_integral(series), "coverage": coverage},
            physical=integrate_physical(physical.samples, experiment_start, experiment_end))
        report["native_observation"] = native.observation()
        report["physical_observation"] = physical.observation()
        report["cost_scope"] = "fixed-token controlled fixture and revisit generation; fixture pressure is separately marked"
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"status": report["status"], "report": str(target)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--base-url")
    parser.add_argument("--report", required=True)
    parser.add_argument("--tiers", type=lambda value: tuple(value.split(",")), default=("gpu", "host"))
    parser.add_argument("--shared-tokens", type=int, default=128)
    parser.add_argument("--branch-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-device", type=int, default=0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
