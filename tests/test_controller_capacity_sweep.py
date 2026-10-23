"""Finite pool sweeps retain provenance and never silently borrow large-pool fits."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

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


def test_plan_only_has_no_launch_connection_or_output_side_effects(tmp_path, monkeypatch, capsys):
    config = tmp_path/"config.yaml"
    config.write_text("{}\n")
    output = tmp_path/"not-created"
    def forbidden(*args, **kwargs):
        pytest.fail("plan-only must not start servers, read GPU state or create output")
    monkeypatch.setattr(sweep, "execute", forbidden)
    monkeypatch.setattr(sweep.server, "command", forbidden)
    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    sweep.main(["--config", str(config), "--model-path", "missing-model", "--output", str(output), "--repetitions", "2"])
    plan = json.loads(capsys.readouterr().out)
    assert not plan["execute"] and not output.exists()
    assert plan["controller_thread_environment"]["RAYON_NUM_THREADS"] == "2"
    assert sweep.os.environ["RAYON_NUM_THREADS"] == "2"
    assert [g["requested_gpu_tokens"] for g in plan["groups"]] == [16384, 32768, 32768, 16384]
    for group in plan["groups"]:
        assert group["calibration"]["status"] == "requires_calibration"
        assert group["server_argv"][group["server_argv"].index("--max-total-tokens")+1] == str(group["requested_gpu_tokens"])
        assert group["config"]["context"]["max_total_model_tokens"] == 32768
        assert group["config"]["context"]["max_new_tokens"] == 1024
        assert group["server_environment"]["CONTEXTKV_SAMPLE_POOL"] == "1"
        assert group["server_environment"]["CONTEXTKV_HICACHE_GB"] == "16"
        assert group["server_environment"]["RAYON_NUM_THREADS"] == "4"
        assert not group["native_initial_state_matched"] and not group["b6_comparison_established"]
        methods = {m["id"]: m["overrides"] for m in group["matrix"]["methods"]}
        for main, ablated in (("Ours", "Ours_no_retire"), ("Rebuild", "Rebuild_no_retire")):
            expected = deepcopy(methods[main])
            expected["ablation"]["epoch_retirement"] = False
            assert methods[ablated] == expected
        assert group["matrix"]["protocols"] == ["P2"]
        assert group["matrix"]["concurrency"] == [4] and group["matrix"]["seeds"] == [42]
    assert plan["groups"][2]["matrix"]["methods"][0]["id"] == "Ours_no_retire"


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


def test_fixed_methods_can_run_without_cost_fit_but_must_observe_real_pool(tmp_path, monkeypatch, capsys):
    config = tmp_path/"config.yaml"
    config.write_text("cost_model:\n  artifact: missing-large-pool.json\n")
    sweep.main(["--config", str(config), "--model-path", "model", "--methods", "Rebuild", "Rebuild_no_retire"])
    group = json.loads(capsys.readouterr().out)["groups"][0]
    assert not group["cost_methods_requested"] and group["config"]["cost_model"]["artifact"] is None
    def forbidden(*args, **kwargs):
        pytest.fail("fixed comparison does not need a cost artifact")
    monkeypatch.setattr(sweep.CostModel, "load", forbidden)
    assert sweep.validate_runtime(group, runtime())["calibration_scope"] is None
    changed = runtime()
    changed["cache_config"]["sample_pool"] = False
    with pytest.raises(ValueError, match="pool/host"):
        sweep.validate_runtime(group, changed)
    with pytest.raises(ValueError, match="pool/host"):
        sweep.validate_runtime(group, runtime(), expected_host_tokens=100)


def test_unbound_cost_plan_fails_before_model_or_process_start(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("calibration preflight must precede any source/model/process work")
    monkeypatch.setattr(sweep, "check_research_checkout", forbidden)
    plan = {"groups": [{"requested_gpu_tokens": 16384, "cost_methods_requested": True,
                        "calibration": {"status": "requires_calibration"}}]}
    with pytest.raises(ValueError, match="own calibration"):
        sweep.execute(Namespace(), plan)


def test_each_capacity_can_bind_its_own_manifest(tmp_path, capsys):
    config = tmp_path/"config.yaml"
    config.write_text("{}\n")
    first, second = cost_artifact(tmp_path), cost_artifact(tmp_path, 32768, "cache-32")
    sweep.main(["--config", str(config), "--model-path", "model", "--cost-artifact", f"16384={first}",
                "--cost-artifact", f"32768={second}"])
    groups = json.loads(capsys.readouterr().out)["groups"]
    assert [g["config"]["cost_model"]["artifact"] for g in groups] == [str(first), str(second)]
    assert all(g["calibration"]["status"] == "requires_runtime_scope_check" for g in groups)


@pytest.mark.parametrize("tokens", ["0", "-64", "1000"])
def test_invalid_pool_capacity_rejected_before_plan(tokens):
    with pytest.raises(SystemExit):
        sweep.main(["--model-path", "model", "--gpu-tokens", tokens])


def test_failed_generation_sets_nonzero_cli_status_after_cleanup_and_results(tmp_path, monkeypatch):
    from controller import templates
    from profiling import runtime_manifest
    config = tmp_path/"config.yaml"
    config.write_text("{}\n")
    checkout, output = tmp_path/"checkout", tmp_path/"output"
    monkeypatch.setattr(sweep, "check_research_checkout", lambda path: {"path": str(path)})
    monkeypatch.setattr(templates, "HuggingFaceTemplate", lambda *a, **kw: SimpleNamespace(fingerprint=ConfigFingerprint()))
    monkeypatch.setattr(sweep.server, "command", lambda *a, **kw: str(checkout/"python/sglang/__init__.py"))
    monkeypatch.setattr(sweep.server, "wait_ready", lambda *a: {})
    process = SimpleNamespace(pid=123, returncode=0)
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: process)
    stopped = []
    monkeypatch.setattr(sweep.server, "stop_owned_server", lambda p: stopped.append(p.pid))
    async def fake_runtime(**kwargs):
        return runtime()
    async def failed_matrix(*args):
        return [{"job_id": "Rebuild", "status": "completed",
                 "summary": {"timing": {"failure_fraction": 1.}, "cohort": {"unfinished_turns": 0}}}]
    monkeypatch.setattr(runtime_manifest, "build_manifest", fake_runtime)
    monkeypatch.setattr(sweep, "execute_matrix", failed_matrix)
    code = sweep.main(["--config", str(config), "--model-path", "model", "--checkout", str(checkout),
        "--gpu-tokens", "16384", "--methods", "Rebuild", "--port", "0", "--output", str(output), "--execute"])
    assert code == 1 and stopped == [123]
    assert json.loads((output/"results.json").read_text())[0]["status"] == "generation_failures"
    group = json.loads((output/"gpu-16384/r1/group.json").read_text())
    assert group["server_returncode_after_cleanup"] == 0 and group["jobs"][0]["status"] == "completed"
