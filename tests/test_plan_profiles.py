"""Explicit fixtures test binding and cost accounting, not GPU performance."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import time

import pytest

from controller.context_schema import ReadingOutcome
from controller.plan_profiles import ExecutionProfiles, PhysicalPaidCost
from controller.reading import READ_SYSTEM
from controller.templates import ByteTemplate
from test_context_planning import context_plan


def profile_data():
    return {"schema_version": 1, "measurement_kind": "hardware", "reader_checkpoint": "fixture",
        "reader_profile_id": "fixture-profile", "fingerprint": asdict(ByteTemplate().fingerprint),
        "cache_config_hash": "fixture-cache", "sampling_recipe": {"temperature": 0},
        "resident_model_manifest": None,
        "resident_model_manifest_digest": hashlib.sha256(b"null").hexdigest(),
        "resource_scope": {"gpu_devices": [0], "devices": [{"index": 0, "uuid": "GPU-fixture"}],
            "process_roots": [123], "host_basis": "specified fixture PSS", "native_endpoint": "http://127.0.0.1:30000"},
        "output_budgets": [128], "rows": [{"profile_id": "explicit-unit-fixture",
            "generation_kind": "READ", "context_action": "REBUILD", "input_range": [100, 500],
            "output_budget": 128, "concurrency_range": [1, 1], "ready_gpu_range": [0, 100],
            "host_load_range": [0, 0], "system_prompt_sha256": hashlib.sha256(READ_SYSTEM.encode()).hexdigest(),
            "observation_ids": ["unit-fixture-only"], "source_files": ["unit-fixture-only.json"],
            "cost_scope": "step_wall", "latency_ms": 100, "gpu_gib_ms": 1000,
            "host_gib_ms": 200, "gpu_peak_gib": 10, "host_peak_gib": 2}]}


def test_execution_profile_matches_actual_prompt_output_and_native_geometry():
    profiles = ExecutionProfiles(profile_data())
    assert profiles.row("READ", "REBUILD", 500, 128, 1, 20, 0, READ_SYSTEM)["profile_id"] == "explicit-unit-fixture"
    assert profiles.row("READ", "REBUILD", 501, 128, 1, 20, 0, READ_SYSTEM) is None
    assert profiles.row("READ", "REBUILD", 500, 64, 1, 20, 0, READ_SYSTEM) is None
    assert profiles.row("READ", "REBUILD", 500, 128, 2, 20, 0, READ_SYSTEM) is None
    assert profiles.row("READ", "REBUILD", 500, 128, 1, 20, 1, READ_SYSTEM) is None
    assert profiles.row("READ", "REBUILD", 500, 128, 1, 20, 0, "a different system prompt") is None


def test_profile_cannot_bind_different_reader_cache_or_unobserved_rows():
    from controller.schema import CacheQuoteBatch
    profiles = ExecutionProfiles(profile_data())
    batch = CacheQuoteBatch("q", 1, 1, "engine", "fixture-cache", 1000, 1000, (),
                            fingerprint=ByteTemplate().fingerprint)
    profiles.bind(batch, "fixture", "fixture-profile")
    with pytest.raises(ValueError, match="reader/cache"):
        profiles.bind(replace(batch, cache_config_hash="new-cache"), "fixture", "fixture-profile")
    with pytest.raises(ValueError, match="reader/cache"):
        profiles.bind(batch, "another-reader", "fixture-profile")
    data = profile_data()
    data["rows"][0]["observation_ids"] = []
    with pytest.raises(ValueError, match="observation/file"):
        ExecutionProfiles(data)
    data = profile_data()
    data["rows"][0]["generation_kind"] = "ANSWER"
    with pytest.raises(ValueError, match="complete step/response tail"):
        ExecutionProfiles(data)


class PhysicalSamplerFixture:
    """Known constant fixture values permit exact integral arithmetic."""
    gpu_devices, process_ids = (0,), (123,)
    def __init__(self):
        self.samples = []
    @staticmethod
    def sample(*args):
        return {"timestamp": time.monotonic(), "gpu_gib": 2, "host_gib": 3,
                "devices": [{"index": 0, "uuid": "GPU-fixture"}],
                "process_pss": [{"pid": 123, "pss_kib": 3*1024**2}]}
    def observation(self):
        return {"host_basis": "specified fixture PSS"}
    async def start(self):
        self.samples.append(self.sample())
    async def stop(self):
        self.samples.append(self.sample())


@pytest.mark.asyncio
async def test_paid_window_includes_auxiliary_and_real_steps_and_closes_physical_boundaries():
    sampler = PhysicalSamplerFixture()
    observer = PhysicalPaidCost(sampler)
    await observer.start()
    arrived = time.monotonic()
    await asyncio.sleep(.002)
    plan, _ = context_plan()
    state = ReadingOutcome(plan, auxiliary_usage={"input_tokens": 7, "output_tokens": 3})
    paid = await observer.observe(arrived, state, {"input_tokens": 11, "output_tokens": 5})
    assert paid.latency_ms > 0
    assert paid.gpu_gib_ms == pytest.approx(paid.latency_ms*2)
    assert paid.host_gib_ms == pytest.approx(paid.latency_ms*3)
    assert (paid.input_tokens, paid.output_tokens) == (18, 8)
    assert observer.last_observation["physical"]["complete_window_coverage"]
    with pytest.raises(ValueError, match="both measured physical boundaries"):
        observer.window(arrived-100, time.monotonic())
    with pytest.raises(ValueError, match="measured auxiliary"):
        await observer.observe(arrived, state, {"input_tokens": None, "output_tokens": 5})
    await observer.aclose()


def test_config_does_not_enable_gain_without_explicit_assets():
    from controller.config import RunConfig
    assert RunConfig().plan_gain is None
    with pytest.raises(ValueError, match="gain_asset"):
        RunConfig(plan_gain={"mode": "rolling_gain", "gain_asset": None})


@pytest.mark.asyncio
async def test_resource_profile_binds_actual_devices_process_roots_endpoint_and_uuid():
    from controller.plan_profiles import ProfiledNativeObserver
    def observer(data, *, endpoint="http://127.0.0.1:30000"):
        return ProfiledNativeObserver(ExecutionProfiles(data), PhysicalPaidCost(PhysicalSamplerFixture()),
            reader_checkpoint="fixture", reader_profile_id="fixture-profile", sampling={"temperature": 0},
            answer_system_prompt="answer fixture", engine_base_url=endpoint)
    matched = observer(profile_data())
    with pytest.raises(ValueError, match="loaded resident model manifest"):
        await matched.start()
    matched.bind_resident_models(None)
    await matched.start()
    await matched.aclose()
    with pytest.raises(ValueError, match="device/process scope"):
        observer(profile_data(), endpoint="http://127.0.0.1:30001")
    data = profile_data()
    data["resource_scope"]["process_roots"] = [999]
    with pytest.raises(ValueError, match="device/process scope"):
        observer(data)
    data = profile_data()
    data["resource_scope"]["devices"][0]["uuid"] = "GPU-not-the-measured-device"
    wrong = observer(data)
    wrong.bind_resident_models(None)
    with pytest.raises(ValueError, match="GPU UUID"):
        await wrong.start()
    await wrong.aclose()
