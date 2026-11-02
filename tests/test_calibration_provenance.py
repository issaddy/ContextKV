from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import subprocess

import httpx
import pytest

from controller.cost_model import CostModel, LookupPoint
from controller.schema import CacheQuote, ConfigFingerprint, PromptCandidate
from profiling.common import Measurement
from profiling.fit import main as fit_main, measurement_provenance
from profiling.runtime_manifest import collect_local_runtime, collect_server


def manifest():
    return {"run_id": "profile-1", "status": "complete", "source": "hardware",
            "fingerprint": asdict(ConfigFingerprint()), "engine": {"cache_config_hash": "cache-A"},
            "pool": {"gpu_capacity_tokens": 1000, "host_capacity_tokens": 2000, "page_size": 64},
            "cache_config": {"host_layout": "layer_first", "write_policy": "write_back"}}


def test_fit_preserves_profile_manifest_sampling_ranges_and_input_identity(tmp_path):
    rows = [asdict(Measurement("first_token", n, 0, 1, 1, 0, n/10, "hardware", run_id="profile-1"))
            for n in (100, 200, 300)]
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text("".join(json.dumps(row) + "\n" for row in rows))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(manifest()))
    output = tmp_path / "cost.json"
    fit_main([str(measurements), "--profile-manifest", str(profile), "--output", str(output)])
    loaded = CostModel.load(output)
    metadata = loaded.calibration_metadata
    assert metadata["profile_manifest"] == manifest()
    assert metadata["profile_manifests"] == [manifest()]
    binding = metadata["measurement_provenance"]
    assert binding["binding_status"] == "run_id_matched"
    assert binding["files"]["measurements"]["path"] == str(measurements.resolve())
    assert binding["sampling"]["first_token"]["ranges"]["prompt_tokens"] == {"min": 100, "max": 300}
    assert binding["cross_configuration_validation"] is False
    loaded.save(tmp_path / "roundtrip.json")
    assert CostModel.load(tmp_path / "roundtrip.json").calibration_metadata == metadata
    with pytest.raises(ValueError, match="run_ids"):
        measurement_provenance([{**row, "run_id": "another-profile"} for row in rows], measurements,
                               profile_manifest_path=profile)


def test_single_manifest_legacy_unlinked_measurements_remain_supported(tmp_path):
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text("legacy measurements\n")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(manifest()))
    result = measurement_provenance([{"phase": "first_token", "prompt_tokens": 100}], measurements,
                                    profile_manifest_path=profile)
    assert result["measurement_provenance"]["binding_status"] == "operator_declared_unlinked"
    assert result["measurement_provenance"]["run_ids"] == []


def test_development_prior_binds_model_and_literal_prompt(tmp_path):
    fingerprint = asdict(ConfigFingerprint(model_revision="qwen3-32b"))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({**manifest(), "fingerprint": fingerprint}))
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text("[]")
    report = tmp_path / "development.json"
    report.write_text(json.dumps({"fingerprint": fingerprint,
                                  "system_prompt": "Use the current allowed sources.",
                                  "sampling_recipe": {"temperature": 0}}))
    metadata = measurement_provenance([], measurements, profile_manifest_path=profile,
                                      development_report_path=report)
    model = CostModel()
    model.calibration_metadata = metadata
    scope = model.configure_sampling(None, fallback_mean=64, fallback_p90=128,
                                     system_prompt="Use the current allowed sources.")
    assert metadata["output_prior_fingerprint"] == fingerprint
    assert scope["output_prior_prompt_status"] == "matching_prompt"
    assert scope["output_prior_status"] == "matching_recipe"
    changed = model.configure_sampling(None, fallback_mean=64, fallback_p90=128,
                                       system_prompt="Different generation policy.")
    assert changed["output_prior_status"] == "configured_prior_fallback"


def test_development_lengths_cannot_be_relabelled_as_another_model(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(manifest()))
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text("[]")
    report = tmp_path / "development.json"
    fingerprint = {**manifest()["fingerprint"], "model_revision": "different-model"}
    report.write_text(json.dumps({"fingerprint": fingerprint}))
    with pytest.raises(ValueError, match="different model fingerprints"):
        measurement_provenance([], measurements, profile_manifest_path=profile,
                               development_report_path=report)


def test_multiple_same_configuration_manifests_are_preserved_and_scoped(tmp_path):
    first = manifest()
    second = {**manifest(), "run_id": "profile-2", "runtime_manifest": {"driver": "separately-recorded"}}
    # The runtime fingerprint may be stored in the engine object by older collectors.
    second["engine"] = {**second["engine"], "fingerprint": second.pop("fingerprint")}
    paths = [tmp_path / "profile-1.json", tmp_path / "profile-2.json"]
    for path, entry in zip(paths, (first, second)):
        path.write_text(json.dumps(entry))
    rows = [asdict(Measurement("first_token", n, 0, 1, 1, 0, n/10, "hardware", run_id=run_id))
            for run_id, n in (("profile-1", 100), ("profile-1", 200), ("profile-2", 300))]
    measurements = tmp_path / "merged.jsonl"
    measurements.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "cost.json"
    fit_main([str(measurements), "--profile-manifest", str(paths[0]),
              "--profile-manifest", str(paths[1]), "--output", str(output)])
    model = CostModel.load(output)
    metadata = model.calibration_metadata
    assert metadata["profile_manifest"] == first
    assert metadata["profile_manifests"] == [first, second]
    provenance = metadata["measurement_provenance"]
    assert provenance["binding_status"] == "run_id_matched"
    assert provenance["run_ids"] == ["profile-1", "profile-2"]
    assert provenance["files"]["profile_manifests"] == [
        {"run_id": entry["run_id"], "path": str(path.resolve())}
        for entry, path in zip((first, second), paths)]
    report = model.check_calibration_scope({"fingerprint": first["fingerprint"], "cache_config_hash": "cache-A"})
    assert report["status"] == "matching_scope"
    assert report["profile_run_ids"] == ["profile-1", "profile-2"]
    assert report["gpu_hardware_verified"] is False
    model.save(tmp_path / "roundtrip.json")
    assert CostModel.load(tmp_path / "roundtrip.json").calibration_metadata == metadata


@pytest.mark.parametrize("mismatch", ["missing_manifest", "extra_manifest", "duplicate_manifest", "unlinked_row"])
def test_multiple_runs_require_exact_manifest_coverage(tmp_path, mismatch):
    rows = [{"phase": "first_token", "run_id": run_id} for run_id in ("profile-1", "profile-2")]
    entries = [manifest(), {**manifest(), "run_id": "profile-2"}]
    if mismatch == "missing_manifest":
        entries.pop()
    elif mismatch == "extra_manifest":
        entries.append({**manifest(), "run_id": "profile-3"})
    elif mismatch == "duplicate_manifest":
        entries[1] = manifest()
    else:
        rows.append({"phase": "first_token"})
    paths = []
    for index, entry in enumerate(entries):
        path = tmp_path / f"profile-{index}.json"
        path.write_text(json.dumps(entry))
        paths.append(path)
    measurements = tmp_path / "merged.jsonl"
    measurements.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="run_ids"):
        measurement_provenance(rows, measurements, profile_manifest_path=paths)


@pytest.mark.parametrize("mismatch", ["fingerprint", "cache_config_hash", "missing_identity"])
def test_multiple_manifests_reject_configuration_conflicts(tmp_path, mismatch):
    first, second = manifest(), {**manifest(), "run_id": "profile-2"}
    if mismatch == "fingerprint":
        second["fingerprint"] = {**second["fingerprint"], "model_revision": "different-model"}
    elif mismatch == "cache_config_hash":
        second["engine"] = {"cache_config_hash": "cache-B"}
    else:
        first["engine"] = second["engine"] = {}
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path, entry in zip(paths, (first, second)):
        path.write_text(json.dumps(entry))
    rows = [{"phase": "first_token", "run_id": run_id} for run_id in ("profile-1", "profile-2")]
    measurements = tmp_path / "merged.jsonl"
    measurements.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="same fingerprint and cache_config_hash"):
        measurement_provenance(rows, measurements, profile_manifest_path=paths)


def test_known_engine_config_mismatch_disables_only_calibrated_selection():
    model = CostModel([LookupPoint(0, 100, 1, 2)], default_output_tokens=1, p90_output_tokens=1)
    model.calibration_metadata = {"profile_manifest": manifest()}
    candidate = PromptCandidate("test", "REBUILD", 0, (), (1,)*100, "unused",
                                frozenset(), frozenset())
    quote = CacheQuote("test", "unused", 100, 99)
    info = {"cache_config_hash": "cache-A", "fingerprint": asdict(ConfigFingerprint())}
    report = model.check_calibration_scope(info)
    before = model.predict_native(candidate, quote)
    assert report["status"] == "matching_scope" and report["gpu_hardware_verified"] is False
    assert report["profile_run_ids"] == ["profile-1"]
    assert before.calibrated
    mismatch = model.check_calibration_scope({**info, "cache_config_hash": "different-capacity-layout"})
    after = model.predict_native(candidate, quote)
    assert mismatch["status"] == "mismatch"
    assert after.cost == before.cost and not after.calibrated
    assert model.check_calibration_scope(info)["status"] == "matching_scope"
    assert model.predict_native(candidate, quote).calibrated


def test_legacy_artifact_stays_usable_but_reports_unbound():
    model = CostModel([LookupPoint(0, 100, 1, 2)], default_output_tokens=1, p90_output_tokens=1)
    report = model.check_calibration_scope({"cache_config_hash": "cache-A", "fingerprint": asdict(ConfigFingerprint())})
    assert report["status"] == "legacy_unbound"
    assert report["matching_scope"] is None and model.calibration_scope_compatible is None


def test_runtime_gpu_facts_are_local_and_server_collection_is_read_only():
    def runner(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return subprocess.CompletedProcess(command, 0, "0, Test GPU, GPU-test, 4096, 555.1\n", "")
    runtime = collect_local_runtime(include_torch=False, runner=runner)
    assert runtime["collection_host"] == "local_process_host"
    assert runtime["gpus"][0]["memory_bytes"] == 4096 * 1024**2
    assert runtime["driver_cuda_torch_versions"] == {"driver": "555.1", "cuda": None, "torch": None}
    paths = []
    def respond(request):
        paths.append(request.url.path)
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"page_size": 64, "api_key": "secret", "launch_command": "server --api-key secret"})
        if request.url.path == "/context/engine/info":
            return httpx.Response(200, json=manifest()["engine"])
        if request.url.path == "/context/metrics":
            return httpx.Response(200, json={"pool": manifest()["pool"], "cache_config": manifest()["cache_config"]})
        return httpx.Response(404)
    data = asyncio.run(collect_server("http://fake.test", research=True, transport=httpx.MockTransport(respond)))
    assert paths == ["/server_info", "/context/engine/info", "/context/metrics"]
    assert "secret" not in json.dumps(data)
    assert data["remote_gpu_identity_inferred_from_local_host"] is False
