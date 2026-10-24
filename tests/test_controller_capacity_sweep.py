"""Finite pool sweeps retain provenance and never silently borrow large-pool fits."""
from dataclasses import asdict
import json

import pytest

from bench import controller_capacity_sweep as sweep
from controller.cost_model import CostModel
from controller.schema import ConfigFingerprint


def cost_artifact(tmp_path, capacity=16384, cache_hash="cache-16"):
    fp = asdict(ConfigFingerprint())
    manifest = {"run_id": "real-source-id", "engine": {"fingerprint": fp, "cache_config_hash": cache_hash},
                "pool": {"gpu_capacity_tokens": capacity, "host_capacity_tokens": 279040},
                "cache_config": {"gpu_capacity_tokens": capacity}}
    path = tmp_path/f"cost-{capacity}.json"
    CostModel().save(path, metadata={"profile_manifest": manifest})
    return path


def runtime(capacity=16384, cache_hash="cache-16"):
    return {"engine": {"engine_kind": "sglang", "profiling_enabled": True,
                       "sglang_commit": sweep.server.COMMIT, "fingerprint": asdict(ConfigFingerprint()),
                       "cache_config_hash": cache_hash},
            "pool": {"gpu_capacity_tokens": capacity, "host_capacity_tokens": 279040},
            "cache_config": {"gpu_capacity_tokens": capacity, "page_size": 64, "hicache": True,
                "host_memory_mode": "cache", "host_layout": "layer_first", "hicache_io_backend": "kernel",
                "write_policy": "write_through", "tma_transfer_enabled": False, "sample_pool": True},
            "server_info": {"hicache_size": 16}}


def test_old_121920_pool_is_not_relabelled_for_small_pool(tmp_path):
    path = cost_artifact(tmp_path, 121920)
    original = path.read_bytes()
    for capacity in (16384, 32768):
        planned = sweep.artifact_plan(path, capacity)
        assert planned["status"] == "requires_calibration"
        assert planned["declared_gpu_capacities"] == [121920]
    assert path.read_bytes() == original


def test_matching_declared_capacity_still_needs_actual_scope_check(tmp_path):
    path = cost_artifact(tmp_path)
    planned = sweep.artifact_plan(path, 16384)
    assert planned["status"] == "requires_runtime_scope_check"
    group = {"requested_gpu_tokens": 16384, "cost_methods_requested": True, "calibration": planned}
    observed = sweep.validate_runtime(group, runtime())
    assert observed["calibration_scope"]["status"] == "matching_scope"
    assert observed["gpu_hardware_equivalence_verified"] is False
    with pytest.raises(ValueError, match="newly bound"):
        sweep.validate_runtime(group, runtime(cache_hash="different-physical-config"))
    with pytest.raises(ValueError, match="pool/host"):
        sweep.validate_runtime(group, runtime(32768))


def test_stage_artifact_runtime_check_requires_configured_admission_opt_in(tmp_path):
    path = cost_artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload["prefill_points"] = [dict(prefix_tokens=0,suffix_tokens=1024,concurrency=1,
                                      elapsed_ms=10,scope="forward_first")]
    payload["stage_model"] = {"version":1,"pre_forward":{"cells":[
        {"features":[0,1024,0,1,128,16384],"requests":3,"mean_ms":1.0}]}}
    path.write_text(json.dumps(payload))
    group = {"requested_gpu_tokens":16384,"cost_methods_requested":True,
             "calibration":sweep.artifact_plan(path,16384)}
    with pytest.raises(ValueError,match="explicit admission_estimator"):
        sweep.validate_runtime(group,runtime())
    group["config"] = {"cost_model":{"admission_estimator":"native_workload"}}
    observed = sweep.validate_runtime(group,runtime())
    assert observed["calibration_scope"]["status"] == "matching_scope"
    with pytest.raises(ValueError,match="newly bound"):
        sweep.validate_runtime(group,runtime(cache_hash="different-physical-config"))


def test_actual_write_policy_must_match_owned_server_environment():
    group = {'requested_gpu_tokens': 16384, 'cost_methods_requested': False,
             'server_environment': {'CONTEXTKV_WRITE_POLICY': 'write_back'}}
    observed = runtime()
    observed['cache_config']['write_policy'] = 'write_back'
    assert sweep.validate_runtime(group, observed)['calibration_scope'] is None
    with pytest.raises(ValueError, match='pool/host'):
        sweep.validate_runtime(group, runtime())
