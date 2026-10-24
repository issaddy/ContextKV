"""Native sweeps must change real launch capacity and preserve native provenance."""
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import native_policy_sweep as sweep


MODEL_CONFIG = {"hidden_size": 3584, "num_attention_heads": 28,
                "num_hidden_layers": 28, "num_key_value_heads": 4}
HOST_LOG = "Allocating kv hierarchical KV host pool: 279040 tokens, 16.00 GB host memory."


def test_plan_16k_never_starts_services_and_keeps_32k_model_context(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("plan mode must not launch processes, connect or create output")
    monkeypatch.setattr(sweep, "execute", forbidden)
    monkeypatch.setattr(sweep, "get_json", forbidden)
    monkeypatch.setattr(sweep, "check_stock", forbidden)
    output = tmp_path / "not-created"
    sweep.main(["--model-path", str(tmp_path / "model"), "--gpu-tokens", "16384", "--output", str(output)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["execute"] is False and not output.exists()
    assert [j["policy"] for j in plan["jobs"]] == list(sweep.POLICIES)
    for job in plan["jobs"]:
        server, bench = job["server_argv"], job["benchmark_argv"]
        assert server[server.index("--max-total-tokens") + 1] == "16384"
        assert bench[bench.index("--context-length") + 1] == "32768"
        assert bench[bench.index("--max-new-tokens") + 1] == "1024"
        assert bench[bench.index("--split") + 1] == "dev"
        assert bench[bench.index("--split-manifest") + 1] == str(sweep.ROOT / "data/mtrag/split_manifest.json")
        assert "--skip-server-warmup" in server
        assert job["server_environment"]["SGLANG_HICACHE_TMA_TRANSFER"] == "0"
        assert Path(job["server_environment"]["SGLANG_CHECKOUT"]).name == "sglang-stock"


def test_default_capacity_and_repetition_order_are_explicit(capsys):
    sweep.main(["--model-path", "models/Qwen2.5-7B-Instruct", "--repetitions", "3"])
    jobs = json.loads(capsys.readouterr().out)["jobs"]
    assert len(jobs) == 9
    assert all(j["requested_gpu_tokens"] == 32768 for j in jobs)
    assert [j["policy"] for j in jobs[3:6]] == ["write_through_selective", "write_back", "write_through"]


@pytest.mark.parametrize("tokens", ["0", "-64", "1000"])
def test_page_misaligned_capacity_rejected(tokens):
    with pytest.raises(SystemExit) as error:
        sweep.main(["--model-path", "model", "--gpu-tokens", tokens])
    assert error.value.code == 2


def test_pool_size_comes_from_actual_server_and_startup_log():
    actual = sweep.pool_measurements({"max_total_num_tokens": 16384}, HOST_LOG, MODEL_CONFIG, 16384)
    assert actual["actual_kv_pool_tokens"] == 16384
    assert actual["actual_host_pool_tokens"] == 279040
    assert actual["actual_host_pool_bytes"] == 16001269760
    assert actual["actual_host_pool_bytes"] != 16 * 1024**3
    with pytest.raises(ValueError, match="observed 32768"):
        sweep.pool_measurements({"max_total_num_tokens": 32768}, HOST_LOG, MODEL_CONFIG, 16384)
    with pytest.raises(ValueError, match="allocation log"):
        sweep.pool_measurements({"max_total_num_tokens": 16384}, "server ready", MODEL_CONFIG, 16384)


def test_native_manifest_binds_requested_pool_and_observed_policy():
    info = {"enable_session_radix_cache": True, "enable_hierarchical_cache": True,
            "disable_radix_cache": False, "tp_size": 1, "dp_size": 1, "page_size": 64,
            "context_length": 32768, "incremental_streaming_output": False,
            "hicache_host_memory_mode": "cache", "hicache_size": 16,
            "hicache_write_policy": "write_back", "hicache_io_backend": "kernel", "hicache_mem_layout": "layer_first",
            "max_total_tokens": 16384, "skip_server_warmup": True, "random_seed": 42}
    runtime = {"gpus": [{"index": 0, "name": "test-only", "uuid": "test", "memory_bytes": 1}],
               "driver_cuda_torch_versions": {"driver": "test", "cuda": "test", "torch": "test"}}
    fingerprint = {"tokenizer_hash": "a" * 64, "template_hash": "b" * 64}
    result = sweep.native_manifest(info, runtime, {"actual_kv_pool_tokens": 16384}, fingerprint, "write_back", {}, 16384)
    assert result["server"]["max_total_tokens"] == result["runtime"]["actual_kv_pool_tokens"] == 16384
    assert result["engine"]["research_patch_applied"] is False
    with pytest.raises(ValueError, match="configuration differs"):
        sweep.native_manifest(info, runtime, {}, fingerprint, "write_through", {}, 16384)


def test_stock_rejects_research_marker_and_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "command", lambda argv: sweep.COMMIT if "rev-parse" in argv else "")
    assert sweep.check_stock(tmp_path)["python_tree_clean"]
    (tmp_path / ".contextkv-source-only").touch()
    with pytest.raises(ValueError, match="ContextKV marker"):
        sweep.check_stock(tmp_path)
    (tmp_path / ".contextkv-source-only").unlink()
    monkeypatch.setattr(sweep, "command", lambda argv: sweep.COMMIT if "rev-parse" in argv else " M python/sglang/server.py")
    with pytest.raises(ValueError, match="not clean"):
        sweep.check_stock(tmp_path)


def test_stop_signals_only_owned_process_group_even_if_parent_exited(monkeypatch):
    signals = []
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    process = Namespace(pid=12345, poll=lambda: 0)
    sweep.stop_owned_server(process)
    assert signals == [(12345, sweep.signal.SIGTERM)]


def test_native_cache_observations_keep_missing_breakdown_unknown():
    turns = [{"conversation_id": "s", "task_id": f"s::{i}", "turn": i, "status": "success",
              "t_arrive": i, "t_done": i + .5,
              "result": {"output_ids": [1, 2], "truncated": False,
                         "meta_info": {"prompt_tokens": 100, "completion_tokens": 2,
                                       "cached_tokens": cached, "cached_tokens_details": tiers}}}
             for i, (cached, tiers) in enumerate([(0, None), (40, {"device": 10, "host": 30})])]
    payload = {"report": {"turns": turns, "started_at": 0, "arrival_window_end": 2,
                "completed_at": 2, "time_origin": "fixed_retrieval_input_ready", "scheduled_sessions": 1,
                "arrived_sessions": 1, "unfinished_turns": 0, "timed_out_sessions": [], "session_makespans": [["s", 2]]}}
    summary = sweep.summarize_native(payload)
    assert summary["native_cache_observations"]["device_hit_tokens"] == 10
    assert summary["native_cache_observations"]["host_loaded_tokens"] == 30
    turns[1]["result"]["meta_info"]["cached_tokens_details"] = None
    missing = sweep.summarize_native(payload)["native_cache_observations"]
    assert missing["device_hit_tokens"] is missing["host_loaded_tokens"] is None
    assert missing["breakdown_unavailable_turns"] == 1


def test_native_generation_failures_return_nonzero_after_owned_cleanup(tmp_path, monkeypatch):
    from adapters import mtrag
    from controller import templates
    from controller.schema import ConfigFingerprint
    from profiling import runtime_manifest
    model, checkout = tmp_path/"model", tmp_path/"checkout"
    model.mkdir()
    (model/"config.json").write_text("{}")
    split = tmp_path/"split.json"
    split.write_text(json.dumps({"dev": [f"s{i}" for i in range(5)]}))
    args = Namespace(checkout=checkout, model_path=model, input=tmp_path/"input", split_manifest=split,
                     output=tmp_path/"output", port=0, gpu_tokens=16384, startup_timeout=1, job_timeout=1)
    directory = args.output/"job"
    jobs = [{"directory": str(directory), "policy": "write_through", "repetition": 1,
             "server_argv": ["fake-server"], "benchmark_argv": ["fake-benchmark"]}]
    monkeypatch.setattr(sweep, "check_stock", lambda path: {})
    monkeypatch.setattr(mtrag, "load_rows", lambda path: [{"conversation_id": f"s{i%5}", "Collection": mtrag.CLOUD_COLLECTION} for i in range(41)])
    monkeypatch.setattr(templates, "HuggingFaceTemplate", lambda *a, **kw: SimpleNamespace(fingerprint=ConfigFingerprint()))
    monkeypatch.setattr(sweep, "command", lambda *a, **kw: str(checkout/"python/sglang/__init__.py"))
    monkeypatch.setattr(runtime_manifest, "collect_local_runtime", lambda: {})
    monkeypatch.setattr(sweep, "wait_ready", lambda *a: {})
    monkeypatch.setattr(sweep, "pool_measurements", lambda *a: {})
    monkeypatch.setattr(sweep, "native_manifest", lambda *a: {})
    monkeypatch.setattr(sweep, "get_json", lambda *a: {})
    monkeypatch.setattr(sweep, "get_text", lambda *a: "")
    process, stopped = SimpleNamespace(pid=456, returncode=0), []
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(sweep, "stop_owned_server", lambda p: stopped.append(p.pid))
    def benchmark(*args, **kwargs):
        (directory/"native-B1-failed.json").write_text(json.dumps({"report": {"turns": [{"status": "failed"}]}}))
    monkeypatch.setattr(sweep.subprocess, "run", benchmark)
    monkeypatch.setattr(sweep, "summarize_native", lambda payload: {"success_count": 0})
    assert sweep.execute(args, jobs) == 1
    assert stopped == [456]
    assert json.loads((args.output/"results.json").read_text())[0]["status"] == "generation_failures"


def test_native_cli_propagates_execution_failure_status(monkeypatch):
    monkeypatch.setattr(sweep, "execute", lambda args, jobs: 1)
    assert sweep.main(["--model-path", "model", "--execute"]) == 1


def test_b0_requires_observed_disabled_host_without_allocation_log():
    info = {'max_total_num_tokens': 16384, 'enable_hierarchical_cache': False}
    measured = sweep.pool_measurements(info, 'ready', MODEL_CONFIG, 16384, host_enabled=False)
    assert measured['actual_host_pool_tokens'] == measured['actual_host_pool_bytes'] == 0
    with pytest.raises(ValueError, match='disabled HiCache'):
        sweep.pool_measurements({**info, 'enable_hierarchical_cache': True}, '', MODEL_CONFIG, 16384, host_enabled=False)
    with pytest.raises(ValueError, match='no native host allocation'):
        sweep.pool_measurements(info, HOST_LOG, MODEL_CONFIG, 16384, host_enabled=False)


def test_launch_b0_omits_hicache_and_session_flags_with_native_defaults_preserved(tmp_path):
    import os
    import subprocess
    import sys
    fake = tmp_path/'python'
    fake.write_text(f'#!{sys.executable}\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH'],
        'CONTEXTKV_MODEL_PATH': '/fixture/model', 'CONTEXTKV_FINGERPRINT_JSON': '{}',
        'SGLANG_CHECKOUT': str(tmp_path/'stock')}
    for baseline in ('default', 'B0'):
        actual_env = env if baseline == 'default' else {**env, 'CONTEXTKV_ENABLE_HICACHE': '0',
            'CONTEXTKV_ENABLE_SESSION_RADIX': '0'}
        completed = subprocess.run(['bash', str(sweep.ROOT/'scripts/launch_sglang.sh')], env=actual_env,
                                   capture_output=True, text=True, check=True)
        argv = json.loads(completed.stdout)
        assert ('--enable-hierarchical-cache' in argv) == (baseline == 'default')
        assert ('--enable-session-radix-cache' in argv) == (baseline == 'default')
        assert '--disable-radix-cache' not in argv
