"""Native sweeps must change real launch capacity and preserve native provenance."""
from argparse import Namespace
from dataclasses import asdict
import json

import pytest

from bench import native_policy_sweep as sweep
from test_model_scope import local_scope as local_scope, checkpoint_fixture


@pytest.mark.parametrize('module',['bench.native_policy_sweep','bench.controller_capacity_sweep'])
def test_removed_standalone_cli_exits_without_launching(module):
    import subprocess
    import sys
    result=subprocess.run([sys.executable,'-m',module,'--execute','--model-path','historical-model'],
                          capture_output=True,text=True)
    assert result.returncode!=0 and 'Standalone sweep removed' in result.stderr


MODEL_CONFIG = {"hidden_size": 5120, "num_attention_heads": 64, "head_dim": 128,
                "num_hidden_layers": 64, "num_key_value_heads": 8}
HOST_LOG = "Allocating kv hierarchical KV host pool: 61056 tokens, 16.00 GB host memory."


def test_pool_size_comes_from_actual_server_and_startup_log():
    actual = sweep.pool_measurements({"max_total_num_tokens": 16384}, HOST_LOG, MODEL_CONFIG, 16384)
    assert actual["actual_kv_pool_tokens"] == 16384
    assert actual["actual_host_pool_tokens"] == 61056
    assert actual["actual_host_pool_bytes"] == 16005464064
    assert actual["actual_host_pool_bytes"] != 16 * 1024**3
    with pytest.raises(ValueError, match="observed 32768"):
        sweep.pool_measurements({"max_total_num_tokens": 32768}, HOST_LOG, MODEL_CONFIG, 16384)
    with pytest.raises(ValueError, match="allocation log"):
        sweep.pool_measurements({"max_total_num_tokens": 16384}, "server ready", MODEL_CONFIG, 16384)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_manifest_binds_requested_pool_and_observed_policy(local_scope, tmp_path, checkpoint):
    if checkpoint:
        local_scope = checkpoint_fixture(local_scope, tmp_path)
    info = {"enable_session_radix_cache": True, "enable_hierarchical_cache": True,
            "disable_radix_cache": False, "tp_size": 1, "dp_size": 1, "page_size": 64,
            "context_length": 32768, "incremental_streaming_output": False,
            "hicache_host_memory_mode": "cache", "hicache_size": 16,
            "hicache_write_policy": "write_back", "hicache_io_backend": "kernel", "hicache_mem_layout": "layer_first",
            "max_total_tokens": 16384, "skip_server_warmup": True, "random_seed": 42,
            "model_path":local_scope.scope.model_path,"tokenizer_path":local_scope.scope.model_path,
            "revision":None,"dtype":"bfloat16","kv_cache_dtype":"bfloat16","pp_size":1,
            "quantization":None,"load_format":"safetensors","json_model_override_args":"{}"}
    runtime = {"gpus": [{"index": 0, "name": "test-only", "uuid": "test", "memory_bytes": 1}],
               "driver_cuda_torch_versions": {"driver": "test", "cuda": "test", "torch": "test"}}
    fingerprint = asdict(local_scope.fingerprint)
    result = sweep.native_manifest(info, runtime, {"actual_kv_pool_tokens": 16384}, fingerprint, "write_back", {}, 16384,model_scope=local_scope.scope)
    assert result["server"]["max_total_tokens"] == result["runtime"]["actual_kv_pool_tokens"] == 16384
    assert result["engine"]["research_patch_applied"] is False
    assert result["model"]["identity_kind"] == local_scope.scope.manifest()["record"]["identity_kind"]
    with pytest.raises(ValueError, match="configuration differs"):
        sweep.native_manifest(info, runtime, {}, fingerprint, "write_through", {}, 16384,model_scope=local_scope.scope)


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
        'CONTEXTKV_MODEL_SCOPE_PATH': '/fixture/scope.json', 'CONTEXTKV_FINGERPRINT_JSON': '{}',
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
