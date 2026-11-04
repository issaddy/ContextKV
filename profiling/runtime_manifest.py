"""Read-only runtime evidence; local GPU facts never describe a remote server.

Run this with the server's Python environment on its host. Optional HTTP reads
preserve observed server configuration and allocator sizes, without reset or load.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import httpx


def collect_local_runtime(*, include_torch=True, runner=subprocess.run):
    result = {"collection_host": "local_process_host", "host_name": platform.node(),
              "platform": platform.platform(), "python": sys.version,
              "gpus": [], "driver_cuda_torch_versions": {"driver": None, "cuda": None, "torch": None},
              "cuda_version_meaning": "torch_compile_target_not_driver_maximum", "collection_errors": []}
    command = ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"]
    try:
        completed = runner(command, capture_output=True, text=True, timeout=15, check=True)
        for row in csv.reader(io.StringIO(completed.stdout)):
            index, name, identity, memory_mib, driver = (part.strip() for part in row)
            result["gpus"].append({"index": int(index), "name": name, "uuid": identity,
                                    "memory_bytes": int(float(memory_mib) * 1024**2), "driver_version": driver})
        versions = sorted({gpu["driver_version"] for gpu in result["gpus"]})
        result["driver_cuda_torch_versions"]["driver"] = versions[0] if len(versions) == 1 else versions or None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["collection_errors"].append({"component": "nvidia-smi", "error": str(exc)})
    if include_torch:
        try:
            import torch
            result["driver_cuda_torch_versions"].update(torch=torch.__version__, cuda=torch.version.cuda)
        except (ImportError, OSError) as exc:
            result["collection_errors"].append({"component": "torch", "error": str(exc)})
    return result


def _public_server_config(value):
    # Native server_info is observed configuration, not a credential export.
    private_names = {"api_key", "admin_api_key", "hf_token", "access_token", "ssl_keyfile_password", "launch_command"}
    if isinstance(value, dict):
        return {key: "[redacted]" if key in private_names else _public_server_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_server_config(item) for item in value]
    return value


def hardware_identity(runtime, gpu_devices):
    """Compare measured device types/capacities; retain UUIDs in the raw manifest."""
    devices = {gpu["index"]: gpu for gpu in runtime["gpus"]}
    if not set(gpu_devices) <= devices.keys():
        return None
    return {"host_name": runtime["host_name"], "platform": runtime["platform"],
        "gpus": sorted(({key: devices[index][key] for key in
                         ("name", "memory_bytes", "driver_version")}
                        for index in set(gpu_devices)), key=lambda gpu: tuple(gpu.values()))}


async def collect_server(base_url, *, research=False, api_key_env="CONTEXTKV_ENGINE_API_KEY", transport=None):
    key = os.environ.get(api_key_env)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, transport=transport, timeout=30) as client:
        response = await client.get("/server_info")
        if response.status_code == 404:
            response = await client.get("/get_server_info")
        response.raise_for_status()
        result = {"server_info": _public_server_config(response.json()), "server_url": base_url,
                  "remote_gpu_identity_inferred_from_local_host": False}
        if research:
            info = await client.post("/context/engine/info", json={})
            info.raise_for_status()
            metrics = await client.post("/context/metrics", json={"after_seq": 0})
            metrics.raise_for_status()
            snapshot = metrics.json()
            result.update(engine=info.json(), pool=snapshot["pool"], cache_config=snapshot.get("cache_config"),
                          fingerprint=info.json().get("fingerprint"), pool_observed_time_ns=snapshot.get("observed_time_ns"))
        return result


async def build_manifest(*, base_url=None, research=False, include_torch=True, source_checkout=None):
    manifest = {"schema_version": 1, "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": collect_local_runtime(include_torch=include_torch),
                "hardware_performance_validated": False}
    if source_checkout:
        commit = subprocess.check_output(["git", "-C", str(source_checkout), "rev-parse", "HEAD"], text=True).strip()
        manifest["source_checkout"] = {"path": str(Path(source_checkout).resolve()), "commit": commit,
                                        "provenance": "local_checkout_not_remote_attestation"}
    if base_url:
        manifest.update(await collect_server(base_url, research=research))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", help="optional native server_info URL; no generation requests are made")
    parser.add_argument("--research", action="store_true", help="also read patched engine identity, cache config and allocator token capacity")
    parser.add_argument("--source-checkout", help="local server checkout whose Git revision is collected")
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args(argv)
    manifest = asyncio.run(build_manifest(base_url=args.base_url, research=args.research,
                                         include_torch=not args.skip_torch, source_checkout=args.source_checkout))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(target.resolve()), "local_gpu_count": len(manifest["runtime"]["gpus"]),
                      "collection_host": "local_process_host"}))


if __name__ == "__main__":
    main()
