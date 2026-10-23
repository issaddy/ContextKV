"""Calibration records and run manifests, usable without NumPy or CUDA."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping

from adapters.mtrag import DATASET_COMMIT, json_hash
from bench.metrics import distribution

ENGINE_COMMIT = "3f36fc08331f64cd63c4fbbdf39cc402f38bb38b"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


@dataclass(frozen=True)
class Measurement:
    phase: str
    prompt_tokens: int
    prefix_tokens: int
    output_tokens: int
    concurrency: int
    transfer_bytes: int
    elapsed_ms: float
    source: str
    run_id: str = ""
    repetition: int = 0
    cache_state: str = "unspecified"
    host_load_tokens: int = 0
    timing_scope: str = "server_phase"

    def __post_init__(self):
        if self.phase not in {"prefill", "restore_prefill", "first_token", "decode", "H2D", "D2H", "control"}:
            raise ValueError("unknown measured phase")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and nonnegative")
        if self.concurrency < 1 or min(self.prompt_tokens, self.prefix_tokens, self.output_tokens, self.transfer_bytes) < 0:
            raise ValueError("invalid calibration sizes")
        if self.prefix_tokens > self.prompt_tokens:
            raise ValueError("prefix cannot exceed prompt")
        if not 0 <= self.host_load_tokens <= self.prefix_tokens:
            raise ValueError("host-loaded tokens must be part of the reusable prefix")
        if self.phase == "decode" and self.output_tokens <= 1:
            raise ValueError("decode requires at least two output tokens")
        if self.source not in {"hardware", "mock"}:
            raise ValueError("source must explicitly identify hardware or mock")


def load_measurements(path: str | Path) -> list[Measurement]:
    return [Measurement(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_measurements(rows: Iterable[Measurement], *, allow_mock: bool = False) -> list[dict]:
    buckets: dict[tuple, list[float]] = {}
    for row in rows:
        if row.source != "hardware" and not allow_mock:
            raise ValueError("mock samples cannot calibrate a hardware cost model")
        key = (row.phase, row.prompt_tokens, row.prefix_tokens, row.output_tokens, row.concurrency,
               row.transfer_bytes, row.source, row.host_load_tokens, row.timing_scope)
        buckets.setdefault(key, []).append(row.elapsed_ms)
    names = ("phase", "prompt_tokens", "prefix_tokens", "output_tokens", "concurrency", "transfer_bytes",
             "source", "host_load_tokens", "timing_scope")
    return [{**dict(zip(names, key)), "elapsed_ms": distribution(values)} for key, values in sorted(buckets.items())]


def write_measurements(path: str | Path, rows: Iterable[Measurement]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _file_hash(path: str | Path | None):
    return sha256(Path(path).read_bytes()).hexdigest() if path is not None else None


def run_manifest(*, protocol: str, config: Mapping[str, Any], seed: int = 42,
                 tokenizer_path: str | Path | None = None, template_path: str | Path | None = None,
                 split_manifest_path: str | Path | None = None,
                 runtime: Mapping[str, Any] | None = None, source: str = "mock") -> dict:
    """Unavailable values are null; validate_manifest blocks formal hardware runs."""
    if protocol not in {"P1", "P2", "P3"} or source not in {"mock", "hardware"}:
        raise ValueError("invalid protocol or source")
    try:
        patch_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        patch_commit = None
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
            "config": dict(config), "config_sha256": json_hash(config),
            "engine": {"repository": "sgl-project/sglang", "base_commit": ENGINE_COMMIT,
                       "controller_commit": patch_commit, "patch_commit": (runtime or {}).get("patch_commit"), "tree_backend": "python"},
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct", "revision": MODEL_REVISION,
                      "tokenizer_hash": _file_hash(tokenizer_path), "chat_template_hash": _file_hash(template_path),
                      "dtype": "bfloat16", "kv_dtype": "bfloat16"},
            "dataset": {"repository": "IBM/mt-rag-benchmark", "commit": DATASET_COMMIT,
                        "split_manifest_sha256": _file_hash(split_manifest_path)},
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "gpu_model": None, "gpu_memory_bytes": None, "driver_cuda_torch_versions": None,
                        "actual_kv_pool_tokens": None, "actual_host_pool_bytes": None, "page_size": None,
                        **dict(runtime or {})},
            "protocol": {"name": protocol, "history_mode": "reference" if protocol == "P1" else "generated",
                         "evidence_mode": "real_bm25s" if protocol == "P3" else "fixed_top5", "seed": seed}}


def validate_manifest(manifest: Mapping[str, Any], *, formal_hardware: bool = True) -> list[str]:
    missing = []
    required = {"model": ("tokenizer_hash", "chat_template_hash"), "dataset": ("split_manifest_sha256",),
                "engine": ("patch_commit",), "runtime": ("gpu_model", "gpu_memory_bytes", "driver_cuda_torch_versions",
                                                        "actual_kv_pool_tokens", "actual_host_pool_bytes", "page_size")}
    if formal_hardware and manifest.get("source") != "hardware":
        missing.append("source must be hardware")
    for section, names in required.items():
        for name in names:
            value = manifest.get(section, {}).get(name)
            if value is None or value == "" or (isinstance(value, str) and value.startswith("TO_")):
                missing.append(f"{section}.{name}")
    return missing
