"""Shared owned-server, capacity and historical-result helpers.

The former fixed-model sweep CLI has been removed. Execute explicitly declared
Qwen3 matrices through bench.b6_joint_sweep; this module does not launch a sweep.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3f36fc08331f64cd63c4fbbdf39cc402f38bb38b"
POLICIES = ("write_through", "write_through_selective", "write_back")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(argv, *, env=None):
    return subprocess.check_output(argv, cwd=ROOT, env=env, text=True).strip()


def get_text(base_url, endpoint):
    with urlopen(base_url + endpoint, timeout=10) as response:
        return response.read().decode()


def get_json(base_url, endpoint):
    return json.loads(get_text(base_url, endpoint))


def check_stock(checkout):
    head = command(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if head != COMMIT:
        raise ValueError(f"stock checkout commit {head} differs from {COMMIT}")
    markers = list(checkout.glob(".contextkv*"))
    runtime_patch = checkout / "python/sglang/srt/managers/contextkv_runtime.py"
    if markers or runtime_patch.exists():
        raise ValueError("checkout contains a ContextKV marker/runtime; an unpatched stock tree is required")
    # A one-time checkout check, not hashes of every request or model weight.
    changed = command(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal", "--", "python"])
    if changed:
        raise ValueError("stock Python source is not clean: " + changed)
    return {"path": str(checkout), "commit": head, "python_tree_clean": True,
            "contextkv_markers_absent": True, "research_patch_applied": False}


def server_command(args, *, mem_fraction_static):
    return ["bash", str(ROOT / "scripts/launch_sglang.sh"), "--host", "127.0.0.1", "--port", str(args.port),
            "--max-total-tokens", str(args.gpu_tokens), "--mem-fraction-static", str(mem_fraction_static),
            "--max-running-requests", "8", "--random-seed", "42", "--skip-server-warmup",
            "--enable-cache-report"]


def pool_measurements(info, startup_log, model_config, expected_gpu_tokens, *, host_enabled=True):
    # This fixed native /get_server_info publishes the actual scheduler capacity.
    gpu = info["max_total_num_tokens"]
    if type(gpu) is not int or gpu != expected_gpu_tokens:
        raise ValueError(f"actual native GPU capacity must be {expected_gpu_tokens}, observed {gpu!r}")
    # Pinned HostKVCache._allocate_host_buffer logs the allocated token slots.
    matches = re.findall(r"Allocating (\S+) hierarchical KV host pool: (\d+) tokens, ([\d.]+) GB host memory", startup_log)
    if not host_enabled:
        if info.get("enable_hierarchical_cache") is not False or matches:
            raise ValueError("B0 requires observed disabled HiCache and no native host allocation")
        return {"actual_kv_pool_tokens": gpu, "actual_host_pool_tokens": 0, "actual_host_pool_bytes": 0,
                "gpu_pool_provenance": "native get_server_info.max_total_num_tokens",
                "host_bytes_provenance": "observed disabled hierarchical cache and no host allocation log"}
    if len(matches) != 1:
        raise ValueError("expected one Qwen TP=1 MHA host-pool allocation log; cannot invent measured host capacity")
    label, tokens, rounded_gb = matches[0]
    head_dim = model_config.get("head_dim", model_config["hidden_size"] // model_config["num_attention_heads"])
    bytes_per_token = 2 * 2 * model_config["num_hidden_layers"] * model_config["num_key_value_heads"] * head_dim
    return {"actual_kv_pool_tokens": gpu, "actual_host_pool_tokens": int(tokens),
            "actual_host_pool_bytes": int(tokens) * bytes_per_token, "host_pool_label": label,
            "host_log_rounded_gb": float(rounded_gb), "kv_bytes_per_token": bytes_per_token,
            "gpu_pool_provenance": "native get_server_info.max_total_num_tokens",
            "host_bytes_provenance": "native startup token slots times TP1 BF16 Qwen KV geometry; excludes allocator overhead"}


def native_manifest(info, runtime, pools, fingerprint, policy, source, expected_gpu_tokens, *, baseline="B1",
                    model_scope, sampling=None, hicache_gb=16):
    if model_scope is None:
        raise ValueError("new native manifests require an explicit Qwen3 model scope")
    if baseline not in {"B0", "B1"}:
        raise ValueError("native manifest baseline must be B0 or B1")
    expected = {"enable_session_radix_cache": baseline == "B1", "enable_hierarchical_cache": baseline == "B1",
                "disable_radix_cache": False, "tp_size": 1, "dp_size": 1, "page_size": 64,
                "context_length": 32768, "incremental_streaming_output": False,
                "max_total_tokens": expected_gpu_tokens, "skip_server_warmup": True, "random_seed": 42}
    if baseline == "B1":
        expected.update(hicache_host_memory_mode="cache", hicache_size=hicache_gb, hicache_write_policy=policy,
                        hicache_io_backend="kernel", hicache_mem_layout="layer_first")
    from types import SimpleNamespace
    model_scope.verify_native_settings(SimpleNamespace(**info))
    model_scope.verify_fingerprint(fingerprint)
    expected["pp_size"] = 1
    if any(info.get(key) != value for key, value in expected.items()):
        mismatch = {key: {"expected": value, "observed": info.get(key)} for key, value in expected.items() if info.get(key) != value}
        raise ValueError(f"native server configuration differs: {mismatch}")
    gpu = next(g for g in runtime["gpus"] if g["index"] == 0)
    result = {"engine": {"base_commit": COMMIT, "research_patch_applied": False, "tree_backend": "python"},
            "model": {"name": model_scope.model_path, "revision": None,
                      "identity": model_scope.identity, "identity_kind": "local_snapshot_metadata",
                      "tokenizer_hash": fingerprint["tokenizer_hash"], "chat_template_hash": fingerprint["template_hash"],
                      "dtype": "bfloat16", "kv_dtype": "bfloat16"},
            "server": expected,
            "runtime": {"gpu_model": gpu["name"], "gpu_uuid": gpu["uuid"], "gpu_memory_bytes": gpu["memory_bytes"],
                        "driver_cuda_torch_versions": runtime["driver_cuda_torch_versions"], **pools},
            "source_checkout": source, "example_only": False,
            "cache_state_protocol": "new_stock_server_skip_generation_warmup_empty_initial_KV",
            "quality_evaluated": False, "performance_claims_validated": False}
    result["model_scope"] = model_scope.manifest()
    from controller.sampling import request_sampling
    result["sampling_recipe"] = request_sampling(sampling)
    return result


def summarize_native(payload):
    from bench.report import summarize_run
    summary = summarize_run(payload)
    turns = payload["report"]["turns"]
    details, missing = [], 0
    for turn in turns:
        result = turn.get("result") or {}
        meta = result.get("meta_info") or {}
        cached = meta.get("cached_tokens")
        tiers = meta.get("cached_tokens_details")
        # Fixed upstream returns None when zero tokens were cached. Missing
        # breakdown with a positive/unknown total stays unknown, never zero.
        if tiers is None and cached == 0:
            tiers = {"device": 0, "host": 0}
        if tiers is None:
            missing += 1
        details.append({"session_id": turn["conversation_id"], "turn": turn["turn"], "status": turn["status"],
            "input_tokens": meta.get("prompt_tokens"), "output_tokens": meta.get("completion_tokens"),
            "cached_tokens": cached, "device_hit_tokens": tiers.get("device") if tiers else None,
            "host_loaded_tokens": tiers.get("host") if tiers else None,
            "num_retractions": meta.get("num_retractions"), "truncated": result.get("truncated"),
            "native_cache_details": meta.get("cached_tokens_details")})
    per_session = summary["latency_by_session"]
    summary.update(native_cache_observations={"source": "native generate meta_info cached_tokens_details",
        "device_hit_tokens": sum(t["device_hit_tokens"] for t in details) if missing == 0 else None,
        "host_loaded_tokens": sum(t["host_loaded_tokens"] for t in details) if missing == 0 else None,
        "breakdown_unavailable_turns": missing, "per_turn": details,
        "physical_transfer_bytes_or_peak_slots_measured": False},
        equal_session_mean_turn_seconds=statistics.mean(statistics.mean(t["value"] for t in rows) for rows in per_session.values()) if per_session else None,
        success_count=sum(t["status"] == "success" for t in turns),
        output_tokens_total=sum(t["output_tokens"] for t in details if t["output_tokens"] is not None),
        statuses=dict(Counter(t["status"] for t in turns)))
    return summary


def wait_ready(process, base_url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"native server exited during startup: {process.returncode}")
        try:
            info = get_json(base_url, "/get_server_info")
            if info.get("status") == "ready":
                return info
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("native server startup timed out; see startup.log")


def stop_owned_server(process):
    # start_new_session makes this PID its process-group leader; no PID search or pkill.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


if __name__ == "__main__":
    raise SystemExit("Standalone sweep removed; use bench.b6_joint_sweep --matrix with an explicit Qwen3 scope.")
