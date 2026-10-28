"""Calibration and actual research-runtime checks for declared matrices.

The fixed-model sweep CLI has been removed. Helpers also permit read-only
validation of previously recorded calibration/runtime evidence.
"""
from __future__ import annotations
import json
from pathlib import Path
from bench import native_policy_sweep as server

ROOT = Path(__file__).resolve().parents[1]


def artifact_plan(path, capacity):
    """Read existing calibration provenance once; never relabel a larger pool."""
    if path is None or not Path(path).is_file():
        return {"status": "requires_calibration", "reason": "artifact_missing", "path": str(path) if path else None}
    metadata = json.loads(Path(path).read_text()).get("metadata", {})
    manifest = metadata.get("profile_manifest") or {}
    manifests = metadata.get("profile_manifests") or ([manifest] if manifest else [])
    capacities = {m.get("cache_config", {}).get("gpu_capacity_tokens", m.get("pool", {}).get("gpu_capacity_tokens"))
                  for m in manifests}
    bound = manifest.get("engine", {}).get("cache_config_hash") and (
        manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint"))
    compatible = bound and capacities == {capacity}
    return {"status": "requires_runtime_scope_check" if compatible else "requires_calibration",
            "reason": "declared_capacity_matches" if compatible else "capacity_mismatch_or_unbound",
            "path": str(Path(path).resolve()), "declared_gpu_capacities": sorted(capacities, key=str),
            "profile_run_ids": [m.get("run_id") for m in manifests],
            "engine_configuration_only": True, "hardware_equivalence_verified": False}


def check_research_checkout(checkout):
    if (checkout/".contextkv-source-only").exists():
        raise ValueError("source-only checkouts cannot launch a server")
    head = server.command(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if head != server.COMMIT:
        raise ValueError("research checkout must use the pinned SGLang commit")
    server.command(["git", "-C", str(checkout), "apply", "--reverse", "--check",
                    str(ROOT/"engine_patches/0001-contextkv-protocol.patch")])
    return {"path": str(checkout), "commit": head, "research_patch_applied": True,
            "verification": "read-only reverse-apply check of the current patch"}


def validate_runtime(capacity, runtime, *, write_policy="write_through",
                     expected_host_tokens=None, expected_hicache_gb=16):
    """Actual scheduler capacities and cache settings, before generation."""
    info, pool, cache = runtime["engine"], runtime["pool"], runtime["cache_config"]
    expected = {"gpu_capacity_tokens": capacity, "page_size": 64, "hicache": True,
        "host_memory_mode": "cache", "host_layout": "layer_first", "hicache_io_backend": "kernel",
        "write_policy": write_policy,
        "tma_transfer_enabled": False, "sample_pool": True}
    mismatch = {key: {"expected": value, "observed": cache.get(key)} for key, value in expected.items()
                if cache.get(key) != value}
    if (info.get("engine_kind") != "sglang" or not info.get("profiling_enabled")
            or info.get("sglang_commit") != server.COMMIT or mismatch
            or pool["gpu_capacity_tokens"] != capacity or pool["host_capacity_tokens"] <= 0
            or runtime["server_info"].get("hicache_size") != expected_hicache_gb
            or (expected_host_tokens is not None and pool["host_capacity_tokens"] != expected_host_tokens)):
        raise ValueError(f"actual research server pool/host settings differ from the sweep: {mismatch}")
    return {"pool": pool, "cache_config": cache,
            "gpu_hardware_equivalence_verified": False}


if __name__ == "__main__":
    raise SystemExit("Standalone sweep removed; use bench.b6_joint_sweep --matrix with an explicit Qwen3 scope.")
