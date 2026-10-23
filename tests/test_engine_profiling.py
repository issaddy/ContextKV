"""CPU protocol contracts for the real-server collector, never hardware evidence."""
import asyncio
from dataclasses import replace

import pytest

from controller.engine_client import EngineError
from controller.mock_engine import MockEngine
from profiling.prefill import PrefillCase
from profiling.sglang_probe import SGLangProbe


class ProtocolDouble(MockEngine):
    async def info(self):
        return replace(await super().info(), engine_kind="sglang", profiling_enabled=True,
                       metrics_supported=True)

    async def profile_prepare(self, mode):
        if mode == "cold":
            await self.restart()
        elif mode == "host_only":
            for slot in self._slots.values():
                slot.gpu, slot.host = False, True
        return {"mode": mode}

    async def metrics(self, after_seq=0):
        return {"next_seq": 0, "events": [], "events_truncated": False,
                "pool": {"unique_gpu_tokens": self.unique_gpu_tokens,
                         "unique_host_tokens": self.unique_host_tokens}}

    async def generate(self, **kwargs):
        result = await super().generate(**kwargs)
        return replace(result, server_ttft_ms=10, server_elapsed_ms=15, decode_ms=5,
                       timing_source="sglang_server_monotonic")


@pytest.mark.parametrize("cache_state,prefix", [("cold",0),("gpu_ready",4),("host_only",4)])
def test_collector_prepares_verified_cache_shapes_and_records_native_fields(cache_state, prefix, tmp_path):
    async def scenario():
        engine = ProtocolDouble(response_text="abcd")
        probe = SGLangProbe(engine, range(10), telemetry_path=tmp_path/"events.jsonl")
        case = PrefillCase(8, prefix, 4, 2, cache_state)
        await probe.prepare(case)
        rows = await probe.probe(case)
        assert [row.phase for row in rows] == ["first_token","decode"]*2
        assert all(row.timing_scope == "server_request" for row in rows)
        assert all(row.prefix_tokens == prefix and row.concurrency == 2 for row in rows)
        assert all(row.elapsed_ms in (10,5) for row in rows)
        assert all(row.host_load_tokens == (prefix if cache_state == "host_only" else 0) for row in rows)
    asyncio.run(scenario())


def test_collector_refuses_non_profiling_or_bad_cache_fixtures():
    async def scenario():
        with pytest.raises(ValueError, match="dedicated patched SGLang"):
            await SGLangProbe(MockEngine(), range(10)).prepare(PrefillCase(8,0,4,1,"cold"))
        probe = SGLangProbe(ProtocolDouble(), range(10))
        with pytest.raises(ValueError, match="cold cases"):
            await probe.prepare(PrefillCase(8,4,4,1,"cold"))
        with pytest.raises(ValueError, match="nonempty"):
            await probe.prepare(PrefillCase(8,0,4,1,"host_only"))
    asyncio.run(scenario())


def test_collector_does_not_retry_configuration_errors():
    class Disabled(ProtocolDouble):
        async def profile_prepare(self, mode):
            raise EngineError("PROFILING_DISABLED")
    async def scenario():
        with pytest.raises(EngineError, match="PROFILING_DISABLED"):
            await SGLangProbe(Disabled(), range(10)).prepare(PrefillCase(8,0,4,1,"cold"))
    asyncio.run(scenario())


def test_smoke_contract_checks_result_replay_and_rejection():
    from engine_patches.smoke import verify
    class AcceptanceDouble(ProtocolDouble):
        async def generate(self, **kwargs):
            try:
                return await super().generate(**kwargs)
            except EngineError as error:
                if error.code == "EPOCH_RETIRED":
                    self._attempts[kwargs["attempt_id"]] = dict(state="REJECTED", terminal=True,
                        result=None, identity=(kwargs["session_id"], kwargs["expected_generation"], ""))
                raise
        async def status(self, **kwargs):
            from controller.engine_client import EpochStatus
            item = self._attempts.get(kwargs.get("attempt_id"))
            if item and item["state"] == "REJECTED":
                return EpochStatus("REJECTED", self.engine_restart_id, terminal=True)
            result = await super().status(**kwargs)
            # The protocol double stores a result before this test wrapper adds
            # native timing. Only actual retained text/IDs/receipt are compared.
            return result
    async def scenario():
        report = await verify(AcceptanceDouble(response_text="abcd"), list(range(20)))
        assert report["replay_exact"] and report["stale_epoch_rejected"]
    asyncio.run(scenario())


def test_calibration_warmup_is_observed_but_never_written_to_training_rows(tmp_path):
    from profiling.common import Measurement, load_measurements
    from profiling.sglang_probe import collect_case
    class ProbeDouble:
        def __init__(self):
            self.flags = []
        async def prepare(self,case):
            self.flags.append(self.warmup)
        async def probe(self,case):
            return [Measurement("first_token",8,0,4,1,0,
                999 if self.warmup else 12,"hardware",timing_scope="server_request")]
    async def scenario():
        probe = ProbeDouble()
        output = tmp_path/"training.jsonl"
        count = await collect_case(probe,PrefillCase(8,0,4,1,"cold"),
                                  warmup=1,repetitions=3,output_path=output)
        assert probe.flags == [True,False,False,False]
        assert count == 3
        assert [row.elapsed_ms for row in load_measurements(output)] == [12,12,12]
    asyncio.run(scenario())
