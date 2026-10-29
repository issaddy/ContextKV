"""Collector-to-fit CPU protocol contracts; synthetic timings are not calibration."""
import asyncio
from dataclasses import asdict, replace
import json

import pytest

from controller.cost_model import CostModel
from profiling.prefill import PrefillCase
from profiling.sglang_probe import SGLangProbe, collect_case
from profiling.stage_fit import load_bound_telemetry, main
from test_engine_profiling import ProtocolDouble


@pytest.fixture
def native_inputs(tmp_path):
    class StageDouble(ProtocolDouble):
        async def generate(self, **kwargs):
            result = await super().generate(**kwargs)
            # Simulate a verified GPU fixture moving to host during admission.
            # It remains a label/diagnostic, never the pre-submit feature.
            receipt = result.receipt
            if len(kwargs["input_ids"]) == 16:
                receipt = replace(receipt, actual_host_loaded_tokens=receipt.actual_device_hit_tokens,
                                  actual_device_hit_tokens=0)
            return replace(result, receipt=receipt, forward_to_first_token_ms=3.0)

    async def collect():
        engine = StageDouble(response_text="abcd", page_size=4)
        telemetry = tmp_path / "probe.telemetry.jsonl"
        measurements = tmp_path / "probe.jsonl"
        probe = SGLangProbe(engine, range(20), telemetry_path=telemetry)
        cases = (PrefillCase(8, 0, 4, 1, "cold"), PrefillCase(16, 4, 4, 2, "gpu_ready"))
        for case in cases:
            await collect_case(probe, case, warmup=1, repetitions=2, output_path=measurements)
        manifest = tmp_path / "probe.manifest.json"
        manifest.write_text(json.dumps({"run_id": probe.run_id, "status": "complete",
            "source": "hardware", "cases": [case.record() for case in cases],
            "engine": asdict(await engine.info()), "pool": {"gpu_capacity_tokens": 16384}}))
        return {"group": telemetry}, measurements, {"group": manifest}

    return asyncio.run(collect())


def test_native_collector_records_original_batch_identity_and_preserves_prepared_features(native_inputs):
    paths, measurements, manifests = native_inputs
    events = [json.loads(line) for line in paths["group"].read_text().splitlines()]
    assert all(type(row["repetition"]) is int and row["repetition"] > 0 for row in events)
    rows, provenance = load_bound_telemetry(paths, measurements, manifests)
    assert len(rows) == 6 and len({row["batch_id"] for row in rows}) == 4
    assert {row["batch_id"].rsplit(":", 1)[1] for row in rows} == {"2", "3", "5", "6"}
    assert all(row["forward_ms"] == 3 and row["pre_forward_ms"] == 7 for row in rows)
    assert all(row["host_load_tokens"] == 0 for row in rows)
    assert sum(row["actual_host_loaded_tokens"] == 4 for row in rows) == 4
    source = provenance["stage_telemetry"]
    assert source["warmup_requests_excluded"] == 3
    assert source["prepared_vs_actual_host_load_changed_requests"] == 4
    assert source["files"][0]["group"] == "group"


@pytest.mark.parametrize("mutation, message", [
    ("forward", "stage intervals"),
    ("run_id", "run_id differs"),
    ("repetition", "repetition differs"),
    ("missing_repetition", "explicit repetition"),
    ("incomplete", "incomplete generation"),
    ("client_timing", "native server-monotonic"),
    ("missing_request", "does not cover every formal"),
    ("duplicate_index", "every member"),
])
def test_native_stage_input_rejects_unbound_or_incomplete_observations(native_inputs, mutation, message):
    paths, measurements, manifests = native_inputs
    path = paths["group"]
    events = [json.loads(line) for line in path.read_text().splitlines()]
    request = next(row for row in events if row["event"] == "request" and not row["warmup"])
    if mutation == "forward":
        request["result"].pop("forward_to_first_token_ms")
    elif mutation == "run_id":
        request["run_id"] = "foreign"
    elif mutation == "repetition":
        request["repetition"] += 1
    elif mutation == "missing_repetition":
        request.pop("repetition")
    elif mutation == "incomplete":
        request["result"]["generation_complete"] = False
    elif mutation == "client_timing":
        request["result"]["timing_source"] = "client_wall"
    elif mutation == "missing_request":
        # Remove the final formal member so even indistinguishable TTFT labels
        # cannot hide a lost denominator.
        last = next(row for row in reversed(events) if row["event"] == "request")
        events.remove(last)
    else:
        duplicate = next(row for row in events if row["event"] == "request"
                         and not row["warmup"] and row["index"] == 1)
        duplicate["index"] = 0
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    with pytest.raises(ValueError, match=message):
        load_bound_telemetry(paths, measurements, manifests)


def test_native_telemetry_cli_fits_current_stage_model(native_inputs, tmp_path):
    paths, measurements, manifests = native_inputs
    base = tmp_path / "base.json"
    manifest = json.loads(manifests["group"].read_text())
    CostModel().save(base, metadata={"profile_manifest": manifest})
    artifact, validation = tmp_path / "stage.json", tmp_path / "validation.json"
    main(["--telemetry", f"group={paths['group']}", "--measurements", str(measurements),
          "--profile-manifest", f"group={manifests['group']}", "--base-artifact", str(base),
          "--output", str(artifact), "--validation", str(validation), "--folds", "2"])
    model = json.loads(artifact.read_text())
    assert model["metadata"]["stage_fit"]["formal_requests"] == 6
    assert model["metadata"]["stage_telemetry"]["files"][0]["path"] == str(paths["group"].resolve())
    assert len(model["prefill_points"]) == 2
    assert all(point["scope"] == "forward_first" for point in model["prefill_points"])
    assert validation.is_file()


def test_native_groups_must_pair_with_supplied_profile_manifests(native_inputs):
    paths, measurements, manifests = native_inputs
    with pytest.raises(ValueError, match="telemetry groups"):
        load_bound_telemetry({"other": paths["group"]}, measurements, manifests)
