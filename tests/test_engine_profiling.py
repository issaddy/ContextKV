"""CPU protocol contracts for the real-server collector, never hardware evidence."""
import asyncio
from dataclasses import replace

import pytest
from test_model_scope import local_scope as local_scope

from controller.engine_client import EngineError
from bench.mock_engine import MockEngine
from profiling.prefill import PrefillCase, load_cases
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
        return {"mode": mode, "pool": {"page_size": self.page_size}}

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
    from profiling.smoke import verify
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


def test_mixed_fixture_uses_native_restore_then_checks_both_tiers(tmp_path):
    import json
    async def scenario():
        engine = ProtocolDouble(response_text="abcd", page_size=4)
        probe = SGLangProbe(engine, range(20), telemetry_path=tmp_path/"mixed.jsonl")
        case = PrefillCase(12, 8, 4, 2, "mixed", 4)
        await probe.prepare(case)
        rows = await probe.probe(case)
        assert all(row.prefix_tokens == 8 and row.host_load_tokens == 4 for row in rows)
        events = [json.loads(line) for line in (tmp_path/"mixed.jsonl").read_text().splitlines()]
        prepared = next(row for row in events if row["event"] == "prepared")
        assert all(q["ready_gpu_prefix_tokens"] == q["host_only_load_tokens"] == 4 for q in prepared["quote"]["quotes"])
        requests = [row for row in events if row["event"] == "request"]
        assert len(requests) == 2 and all(row["exact_prefix_split"] for row in requests)
        assert all(row["case"]["device_prefix_tokens"] == 4 for row in requests)
    asyncio.run(scenario())


def test_mixed_execution_tier_drift_is_retained_and_refused(tmp_path):
    import json
    class Drift(ProtocolDouble):
        async def generate(self, **kwargs):
            result = await super().generate(**kwargs)
            if len(kwargs["input_ids"]) == 12:
                result = replace(result,receipt=replace(result.receipt,actual_device_hit_tokens=8,actual_host_loaded_tokens=0))
            return result
    async def scenario():
        probe = SGLangProbe(Drift(page_size=4),range(20),telemetry_path=tmp_path/"drift.jsonl")
        case = PrefillCase(12,8,4,2,"mixed",4)
        await probe.prepare(case)
        with pytest.raises(ValueError,match="execution device/host split"):
            await probe.probe(case)
        rows = [json.loads(line) for line in (tmp_path/"drift.jsonl").read_text().splitlines()]
        requests = [row for row in rows if row["event"] == "request"]
        assert len(requests) == 2 and all(row["exact_prefix_split"] is False for row in requests)
        assert all(row["result"]["receipt"]["actual_host_loaded_tokens"] == 0 for row in requests)
        failure = next(row for row in rows if row["event"] == "fixture_failed")
        assert failure["stage"] == "execution" and failure["request_indices"] == [0,1]
        assert failure["metrics"]["events"] == [] and failure["metrics"]["pool"]["unique_host_tokens"] > 0
    asyncio.run(scenario())


@pytest.mark.parametrize("device", [None,0,8,9,True])
def test_mixed_requires_an_explicit_partial_device_prefix(device):
    with pytest.raises(ValueError,match="device_prefix_tokens"):
        PrefillCase(12,8,4,1,"mixed",device).prefix_split()


def test_default_case_record_stays_compatible_and_mixed_pages_are_checked():
    assert PrefillCase(8,4,4,1,"gpu_ready").record() == dict(
        prompt_tokens=8,prefix_tokens=4,output_tokens=4,concurrency=1,cache_state="gpu_ready")
    async def scenario():
        with pytest.raises(ValueError,match="page size"):
            await SGLangProbe(ProtocolDouble(page_size=4),range(20)).prepare(PrefillCase(12,8,4,1,"mixed",2))
    asyncio.run(scenario())


def test_failed_case_manifest_counts_rows_already_written(tmp_path,monkeypatch,local_scope):
    from argparse import Namespace
    from types import SimpleNamespace
    import json
    import profiling.sglang_probe as module
    from controller.schema import ConfigFingerprint
    from profiling.common import Measurement, write_measurements
    engine = ProtocolDouble()
    async def metrics(after_seq=0):
        return {"pool":{"page_size":1},"cache_config":{}}
    async def aclose():
        pass
    engine.metrics = metrics
    engine.aclose = aclose
    monkeypatch.setattr(module,"SGLangEngineClient",lambda *a,**k:engine)
    tokenizer = SimpleNamespace(all_special_ids=(),get_vocab=lambda:{"a":1,"b":2})
    monkeypatch.setattr("controller.config.create_model_template",lambda *a,**k:
                        SimpleNamespace(tokenizer=tokenizer,fingerprint=ConfigFingerprint()))
    async def interrupted(probe,case,*,warmup,repetitions,output_path):
        write_measurements(output_path,[Measurement("first_token",8,0,4,1,0,10.,"hardware")]*2)
        raise ValueError("third repetition fixture failed")
    monkeypatch.setattr(module,"collect_case",interrupted)
    config=tmp_path/"config.yaml"
    config.write_text(f"engine:\n  kind: sglang\nmodel:\n  scope_path: {local_scope.path}\n")
    output=tmp_path/"probe.jsonl"
    args=Namespace(config=str(config),runtime_manifest=None,model_path=None,base_url=None,output=str(output),
        seed=42,lengths=(8,),prefix_fractions=(0,),outputs=(4,),concurrency=(1,),
        cache_states=("cold",),device_prefix_tokens=None,repetitions=3,warmup=1)
    with pytest.raises(ValueError,match="third repetition"):
        asyncio.run(module._run(args))
    manifest=json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"]=="failed" and manifest["measurement_count"]==2
    assert manifest["completed_case_measurement_count"]==0


@pytest.mark.parametrize("existing", ["measurements", "telemetry", "manifest"])
def test_existing_profile_artifact_is_rejected_before_engine_access(tmp_path, monkeypatch, existing):
    from argparse import Namespace
    import profiling.sglang_probe as module
    output = tmp_path / "probe.jsonl"
    paths = {"measurements": output, "telemetry": output.with_suffix(".telemetry.jsonl"),
             "manifest": output.with_suffix(".manifest.json")}
    paths[existing].write_bytes(b"previous run must remain unchanged\n")

    def no_engine(*args, **kwargs):
        raise AssertionError("existing run must be rejected before contacting the engine")

    monkeypatch.setattr(module, "SGLangEngineClient", no_engine)
    with pytest.raises(FileExistsError, match="new output prefix"):
        # No valid config is needed: artifact ownership is the first boundary.
        asyncio.run(module._run(Namespace(output=str(output), config="missing.yaml")))
    assert paths[existing].read_bytes() == b"previous run must remain unchanged\n"
    assert all(not path.exists() for name, path in paths.items() if name != existing)


def test_profile_manifest_claim_precedes_cache_preparation(tmp_path, monkeypatch, local_scope):
    from argparse import Namespace
    from types import SimpleNamespace
    import json
    import profiling.sglang_probe as module
    from controller.schema import ConfigFingerprint
    from profiling.common import Measurement, write_measurements

    class CollectorEngine(ProtocolDouble):
        async def metrics(self, after_seq=0):
            return {"pool": {"page_size": 1}, "cache_config": {}}

        async def aclose(self):
            pass

    monkeypatch.setattr(module, "SGLangEngineClient", lambda *args, **kwargs: CollectorEngine())
    tokenizer = SimpleNamespace(all_special_ids=(), get_vocab=lambda: {"a": 1, "b": 2})
    monkeypatch.setattr("controller.config.create_model_template", lambda *args, **kwargs:
                        SimpleNamespace(tokenizer=tokenizer, fingerprint=ConfigFingerprint()))
    calls = []

    async def collected(probe, case, *, warmup, repetitions, output_path):
        manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
        assert manifest["status"] == "running" and manifest["run_id"] == probe.run_id
        calls.append(probe.run_id)
        write_measurements(output_path, [Measurement("first_token", 8, 0, 4, 1, 0,
            10.0, "hardware", run_id=probe.run_id)])
        return 1

    monkeypatch.setattr(module, "collect_case", collected)
    config = tmp_path / "config.yaml"
    config.write_text(f"engine:\n  kind: sglang\nmodel:\n  scope_path: {local_scope.path}\n")
    output = tmp_path / "probe.jsonl"
    args = Namespace(config=str(config), runtime_manifest=None, model_path=None, base_url=None,
        output=str(output), seed=42, lengths=(8,), prefix_fractions=(0,), outputs=(4,),
        concurrency=(1,), cache_states=("cold",), device_prefix_tokens=None, repetitions=1, warmup=0)
    asyncio.run(module._run(args))
    before = {path: path.read_bytes() for path in (output, output.with_suffix(".manifest.json"))}
    assert json.loads(before[output.with_suffix(".manifest.json")])["measurement_count"] == 1
    with pytest.raises(FileExistsError, match="new output prefix"):
        asyncio.run(module._run(args))
    assert len(calls) == 1
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("change,error", [
    ({"prefix_tokens": 12}, "prefix < prompt"),
    ({"prefix_tokens": 7}, "page size"),
    ({"device_prefix_tokens": 2}, "page size"),
    ({"output_tokens": 100}, "context limit"),
    ({"concurrency": 0}, "positive"),
    ({"prompt_tokens": True}, "integers"),
    ({"cache_state": "unknown"}, "device_prefix_tokens"),
])
def test_explicit_case_boundary_refuses_invalid_shapes(tmp_path, change, error):
    import json
    path = tmp_path/"cases.json"
    row = PrefillCase(12,8,4,1,"mixed",4).record()
    path.write_text(json.dumps([{**row, **change}]))
    with pytest.raises(ValueError, match=error):
        load_cases(path, page_size=4, context_limit=32)


def test_explicit_cases_execute_only_listed_shapes_through_existing_collector(tmp_path, monkeypatch, local_scope):
    from argparse import Namespace
    from types import SimpleNamespace
    import json
    import profiling.sglang_probe as module
    from controller.schema import ConfigFingerprint
    from profiling.common import load_measurements

    class SelectedEngine(ProtocolDouble):
        async def metrics(self, after_seq=0):
            return {**await super().metrics(after_seq), "pool": {"page_size": 4}}

        async def aclose(self):
            pass

    engine = SelectedEngine(page_size=4, response_text="abcd")
    monkeypatch.setattr(module, "SGLangEngineClient", lambda *a, **k: engine)
    tokenizer = SimpleNamespace(all_special_ids=(), get_vocab=lambda: {str(n): n for n in range(20)})
    monkeypatch.setattr("controller.config.create_model_template", lambda *a, **k:
                        SimpleNamespace(tokenizer=tokenizer, fingerprint=ConfigFingerprint()))
    cases = [PrefillCase(8,0,4,1,"cold"), PrefillCase(12,8,4,2,"mixed",4)]
    path = tmp_path/"cases.json"
    path.write_text(json.dumps([case.record() for case in cases]))
    config = tmp_path/"config.yaml"
    config.write_text(f"engine:\n  kind: sglang\nmodel:\n  scope_path: {local_scope.path}\n")
    output = tmp_path/"profile.jsonl"
    # No grid arguments: the list is the full, ordered request specification.
    args = Namespace(config=str(config), runtime_manifest=None, model_path=None, base_url=None,
                     output=str(output), seed=42, cases_file=str(path), repetitions=2, warmup=1)
    asyncio.run(module._run(args))
    rows = load_measurements(output)
    assert [(r.prompt_tokens, r.concurrency) for r in rows] == [(8,1)]*4 + [(12,2)]*8
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["cases"] == [case.record() for case in cases]
    assert manifest["case_selection"] == "explicit_list"
    assert manifest["planned_generation_requests"] == {
        "measured": 6, "warmup": 3, "prefix_preparation": 12, "total": 21}
    events = [json.loads(line) for line in output.with_suffix(".telemetry.jsonl").read_text().splitlines()]
    assert [e["case"] for e in events if e["event"] == "prepared"] == [
        case.record() for case in cases for _ in range(3)]
    assert sum(e["event"] == "request" and e["warmup"] for e in events) == 3


def test_explicit_case_cli_does_not_silently_ignore_grid_options(tmp_path, monkeypatch):
    import profiling.sglang_probe as module

    async def no_run(args):
        raise AssertionError("ambiguous sampling specification must fail before engine access")

    monkeypatch.setattr(module, "_run", no_run)
    with pytest.raises(SystemExit) as error:
        module.main(["--cases-file", "selected.json", "--lengths", "4096", "--output", str(tmp_path/"out")])
    assert error.value.code == 2
