#!/usr/bin/env python3
"""Plan by default; --execute runs three native B1 policies on this local host.

Uses only a clean pinned stock checkout. Owns only the server process group it
starts; never stops an existing server or makes any SSH/remote-host connection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3f36fc08331f64cd63c4fbbdf39cc402f38bb38b"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
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


def server_command(args, *, mem_fraction_static=.15):
    return ["bash", str(ROOT / "scripts/launch_sglang.sh"), "--host", "127.0.0.1", "--port", str(args.port),
            "--max-total-tokens", str(args.gpu_tokens), "--mem-fraction-static", str(mem_fraction_static),
            "--max-running-requests", "8", "--random-seed", "42", "--skip-server-warmup",
            "--enable-cache-report"]


def benchmark_command(args, directory):
    return [sys.executable, "-m", "bench.native_baseline", str(args.input), "--baseline", "B1",
            "--base-url", f"http://127.0.0.1:{args.port}", "--engine-manifest", str(directory / "native-manifest.json"),
            "--model-path", str(args.model_path), "--local-files-only", "--protocol", "P2", "--domain", "cloud",
            "--split-manifest", str(args.split_manifest), "--split", "dev", "--concurrency", "4",
            "--max-new-tokens", "1024", "--context-length", "32768", "--seed", "42", "--output", str(directory)]


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
                    model_scope=None, sampling=None, hicache_gb=16):
    if baseline not in {"B0", "B1"}:
        raise ValueError("native manifest baseline must be B0 or B1")
    expected = {"enable_session_radix_cache": baseline == "B1", "enable_hierarchical_cache": baseline == "B1",
                "disable_radix_cache": False, "tp_size": 1, "dp_size": 1, "page_size": 64,
                "context_length": 32768, "incremental_streaming_output": False,
                "max_total_tokens": expected_gpu_tokens, "skip_server_warmup": True, "random_seed": 42}
    if baseline == "B1":
        expected.update(hicache_host_memory_mode="cache", hicache_size=hicache_gb, hicache_write_policy=policy,
                        hicache_io_backend="kernel", hicache_mem_layout="layer_first")
    if model_scope is not None:
        from types import SimpleNamespace
        model_scope.verify_native_settings(SimpleNamespace(**info))
        model_scope.verify_fingerprint(fingerprint)
        expected["pp_size"] = 1
    if any(info.get(key) != value for key, value in expected.items()):
        mismatch = {key: {"expected": value, "observed": info.get(key)} for key, value in expected.items() if info.get(key) != value}
        raise ValueError(f"native server configuration differs: {mismatch}")
    gpu = next(g for g in runtime["gpus"] if g["index"] == 0)
    result = {"engine": {"base_commit": COMMIT, "research_patch_applied": False, "tree_backend": "python"},
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct", "revision": MODEL_REVISION,
                      "tokenizer_hash": fingerprint["tokenizer_hash"], "chat_template_hash": fingerprint["template_hash"],
                      "dtype": "bfloat16", "kv_dtype": "bfloat16"},
            "server": expected,
            "runtime": {"gpu_model": gpu["name"], "gpu_uuid": gpu["uuid"], "gpu_memory_bytes": gpu["memory_bytes"],
                        "driver_cuda_torch_versions": runtime["driver_cuda_torch_versions"], **pools},
            "source_checkout": source, "example_only": False,
            "cache_state_protocol": "new_stock_server_skip_generation_warmup_empty_initial_KV",
            "quality_evaluated": False, "performance_claims_validated": False}
    if model_scope is not None:
        result["model"].update(name=model_scope.model_path, revision=None, identity=model_scope.identity,
                               identity_kind="local_snapshot_metadata")
        result["model_scope"] = model_scope.manifest()
    if model_scope is not None or sampling is not None:
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


def execute(args, jobs):
    sys.path.insert(0, str(ROOT))
    from adapters.mtrag import CLOUD_COLLECTION, load_rows
    from controller.templates import HuggingFaceTemplate
    from profiling.runtime_manifest import collect_local_runtime

    source = check_stock(args.checkout)
    selected = set(json.loads(args.split_manifest.read_text())["dev"])
    rows = [r for r in load_rows(args.input) if r["conversation_id"] in selected and r["Collection"] == CLOUD_COLLECTION]
    if len(rows) != 41 or len({r["conversation_id"] for r in rows}) != 5:
        raise ValueError("this comparison requires the frozen 5-session / 41-turn Cloud dev cohort")
    template = HuggingFaceTemplate(str(args.model_path), MODEL_REVISION, local_files_only=True)
    fingerprint = asdict(template.fingerprint)
    model_config = json.loads((args.model_path / "config.json").read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", SGLANG_CHECKOUT=str(args.checkout), SGLANG_HICACHE_TMA_TRANSFER="0",
               CONTEXTKV_MODEL_PATH=str(args.model_path), CONTEXTKV_FINGERPRINT_JSON=json.dumps(fingerprint),
               CONTEXTKV_ENABLE_HICACHE="1", CONTEXTKV_HICACHE_GB="16", CONTEXTKV_HICACHE_LAYOUT="layer_first",
               CONTEXTKV_HICACHE_IO_BACKEND="kernel", CONTEXTKV_PAGE_SIZE="64",
               PYTHONPATH=str(args.checkout / "python") + os.pathsep + str(ROOT))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    origin = command([sys.executable, "-c", "import importlib.util; print(importlib.util.find_spec('sglang').origin)"], env=env)
    if not Path(origin).resolve().is_relative_to(args.checkout / "python"):
        raise ValueError(f"Python resolves a different sglang checkout: {origin}")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", jobs)
    runtime = collect_local_runtime()
    write_json(args.output / "runtime.json", runtime)
    completed = []
    for job in jobs:
        directory = Path(job["directory"])
        directory.mkdir()
        base_url = f"http://127.0.0.1:{args.port}"
        # Refuse to share or stop an existing listener. Root/operator stops it separately.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        env["CONTEXTKV_WRITE_POLICY"] = job["policy"]
        process = None
        result = {**job, "status": "started"}
        write_json(directory / "job.json", result)
        try:
            with (directory / "startup.log").open("w") as log:
                process = subprocess.Popen(job["server_argv"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                write_json(directory / "owned-server.json", {"pid": process.pid, "process_group": process.pid})
                info = wait_ready(process, base_url, args.startup_timeout)
                write_json(directory / "server-info-before.json", info)
                pools = pool_measurements(info, (directory / "startup.log").read_text(), model_config, args.gpu_tokens)
                manifest = native_manifest(info, runtime, pools, fingerprint, job["policy"], source, args.gpu_tokens)
                manifest.update(startup_log=str(directory / "startup.log"), source_runtime_manifest=str(args.output / "runtime.json"))
                write_json(directory / "native-manifest.json", manifest)
                (directory / "metrics-before.txt").write_text(get_text(base_url, "/metrics"))
                with (directory / "benchmark.log").open("w") as benchmark_log:
                    subprocess.run(job["benchmark_argv"], cwd=ROOT, env=env, stdout=benchmark_log,
                                   stderr=subprocess.STDOUT, check=True, timeout=args.job_timeout)
                write_json(directory / "server-info-after.json", get_json(base_url, "/get_server_info"))
                (directory / "metrics-after.txt").write_text(get_text(base_url, "/metrics"))
                payload = json.loads(next(directory.glob("native-B1-*.json")).read_text())
                summary = summarize_native(payload)
                write_json(directory / "summary.json", summary)
                result.update(status="completed" if summary["success_count"] == 41 else "generation_failures", summary=summary)
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if process is not None:
                stop_owned_server(process)
                result["server_returncode_after_cleanup"] = process.returncode
            write_json(directory / "job.json", result)
            completed.append(result)
            write_json(args.output / "results.json", completed)
        print(json.dumps({"policy": job["policy"], "repetition": job["repetition"], "status": result["status"]}), flush=True)
    return 0 if all(result["status"] == "completed" for result in completed) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=ROOT / "vendor/sglang-stock")
    parser.add_argument("--input", type=Path, default=ROOT / "data/mtrag/RAG.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data/mtrag/split_manifest.json")
    parser.add_argument("--output", type=Path, help="new directory; default runs/native-policy-sweep/gpu-<tokens>")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--gpu-tokens", type=int, default=32768,
                        help="real native GPU KV pool cap, a positive multiple of page size 64; model context stays 32768")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--execute", action="store_true", help="launch local owned servers; otherwise only print the plan")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = ROOT / "runs/native-policy-sweep" / f"gpu-{args.gpu_tokens}"
    for key in ("model_path", "checkout", "input", "split_manifest", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.repetitions < 1 or len(set(args.policies)) != len(args.policies):
        parser.error("positive repetitions and unique policy entries required")
    if args.gpu_tokens < 64 or args.gpu_tokens % 64:
        parser.error("--gpu-tokens must be positive and divisible by page size 64")
    jobs = []
    for repetition in range(args.repetitions):
        shift = repetition % len(args.policies)
        for policy in args.policies[shift:] + args.policies[:shift]:
            directory = args.output / f"{policy}-r{repetition + 1}"
            jobs.append({"policy": policy, "repetition": repetition + 1, "directory": str(directory),
                "requested_gpu_tokens": args.gpu_tokens, "model_context_tokens": 32768,
                "server_argv": server_command(args), "benchmark_argv": benchmark_command(args, directory),
                "server_environment": {"CUDA_VISIBLE_DEVICES": "0", "SGLANG_CHECKOUT": str(args.checkout),
                    "CONTEXTKV_WRITE_POLICY": policy, "CONTEXTKV_HICACHE_LAYOUT": "layer_first", "CONTEXTKV_HICACHE_GB": "16",
                    "SGLANG_HICACHE_TMA_TRANSFER": "0"},
                "scope": "B1 stock native, P2 Cloud dev 5 sessions/41 turns, c4, cap1024, Q0; quality unvalidated"})
    if args.execute:
        return execute(args, jobs)
    else:
        print(json.dumps({"execute": False, "jobs": jobs}, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
