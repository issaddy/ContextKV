"""Explicit local model identity, shared by controller, launcher and native guard.

This standard-library-only file is copied unchanged into the pinned engine.
It validates metadata and shard inventory, not the contents of model weights.
Version 2 also binds weight digests recorded at checkpoint publication; startup
does not recompute them or attest that current weight contents match those digests.
No model discovery, download or request-time state is involved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys


def canonical_config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _checkpoint_files(files):
    if not isinstance(files, (list, tuple)) or not files:
        raise ValueError("checkpoint scope requires weight_files")
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"file", "bytes", "sha256"}:
            raise ValueError("invalid checkpoint weight file record")
        name = entry["file"]
        if (not isinstance(name, str) or not name.endswith(".safetensors")
                or "/" in name or "\\" in name or name == ".safetensors"
                or type(entry["bytes"]) is not int or entry["bytes"] <= 0
                or not _valid_sha256(entry["sha256"])):
            raise ValueError("invalid checkpoint weight file name, size or SHA")
    if len({entry["file"] for entry in files}) != len(files):
        raise ValueError("duplicate checkpoint weight file")
    return sorted(deepcopy(files), key=lambda entry: entry["file"])


def _single_tensor_bytes(path):
    """Read safetensors metadata only; the tensor payload is never read here."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError("invalid single safetensors header length")
        header_size = int.from_bytes(length, "little")
        if not 0 < header_size <= size - 8:
            raise ValueError("invalid single safetensors header size")
        header = json.loads(stream.read(header_size))
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1]):
            raise ValueError("invalid single safetensors data offsets")
        intervals.append(offsets)
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("single safetensors payload has gaps or overlaps")
        cursor = end
    if cursor != size - 8 - header_size:
        raise ValueError("single safetensors payload size differs from header")
    return cursor


class LocalModelScope:
    def __init__(self, record, *, source_path, source_sha256):
        record = deepcopy(record)
        expected = {"schema_version", "identity_kind", "model_path", "config", "canonical_config_sha256",
                    "index_tensor_bytes", "shards", "shard_bytes", "tokenizer_hash", "template_hash",
                    "hf_revision", "weight_content_verified", "source"}
        artifact = record.get("schema_version") == 2 and record.get("identity_kind") == "local_checkpoint_artifact"
        if artifact:
            expected.add("weight_files")
        if set(record) != expected or not (artifact or (
                record["schema_version"] == 1 and record["identity_kind"] == "local_snapshot_metadata")):
            raise ValueError("unsupported local model scope schema")
        if record["hf_revision"] is not None or record["weight_content_verified"] is not False:
            raise ValueError("local scope cannot attest an HF revision or currently verified weight contents")
        path, config = Path(record["model_path"]), record["config"]
        if not path.is_absolute():
            raise ValueError("local model scope requires an absolute model path")
        if (config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]
                or config.get("torch_dtype", config.get("dtype")) != "bfloat16"
                or config.get("use_sliding_window") is not False or config.get("sliding_window") is not None
                or config.get("rope_scaling") is not None):
            raise ValueError("local scope currently requires BF16 dense Qwen3 without SWA or RoPE extension")
        for field in ("head_dim", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
            if type(config.get(field)) is not int or config[field] <= 0:
                raise ValueError(f"local scope needs an explicit positive {field}")
        if canonical_config_hash(config) != record["canonical_config_sha256"]:
            raise ValueError("scope configuration disagrees with its canonical SHA")
        for field in ("canonical_config_sha256", "tokenizer_hash", "template_hash"):
            value = record[field]
            if not _valid_sha256(value):
                raise ValueError(f"invalid {field}")
        for field in ("index_tensor_bytes", "shards", "shard_bytes"):
            if type(record[field]) is not int or record[field] <= 0:
                raise ValueError(f"invalid local checkpoint inventory: {field}")
        identity = {key: record[key] for key in ("canonical_config_sha256", "index_tensor_bytes", "shards", "shard_bytes",
                                                "tokenizer_hash", "template_hash")}
        if artifact:
            files = _checkpoint_files(record["weight_files"])
            if len(files) != record["shards"] or sum(entry["bytes"] for entry in files) != record["shard_bytes"]:
                raise ValueError("checkpoint weight files disagree with shard inventory")
            record["weight_files"] = files
            identity["weight_files"] = files
        prefix = "local-checkpoint:" if artifact else "local-snapshot:"
        self.identity = prefix + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.model_path = str(path.resolve())
        record["model_path"] = self.model_path
        self.context_limit = config["max_position_embeddings"]
        self.source_path, self.source_sha256 = str(source_path), source_sha256
        self._record = record

    def manifest(self):
        return {"source_path": self.source_path, "source_sha256": self.source_sha256,
                "identity": self.identity, "record": deepcopy(self._record)}

    def verify_fingerprint(self, fingerprint):
        expected = {"model_revision": self.identity, "tokenizer_hash": self._record["tokenizer_hash"],
                    "template_hash": self._record["template_hash"]}
        if any(fingerprint.get(key) != value for key, value in expected.items()):
            raise ValueError("model scope and client fingerprint differ")

    def verify_files(self, model_path=None):
        path = Path(model_path or self.model_path).resolve(strict=True)
        if str(path) != self.model_path:
            raise ValueError("running model path differs from local scope")
        config = json.loads((path/"config.json").read_text())
        if canonical_config_hash(config) != self._record["canonical_config_sha256"]:
            raise ValueError("local model config differs from scope")
        index_path = path/"model.safetensors.index.json"
        if self._record["schema_version"] == 2:
            files = self._record["weight_files"]
            declared = {entry["file"] for entry in files}
            if ({item.name for item in path.glob("*.safetensors")} != declared
                    or any((path/entry["file"]).stat().st_size != entry["bytes"] for entry in files)):
                raise ValueError("local checkpoint shard inventory differs from scope")
            if not index_path.exists():
                if declared != {"model.safetensors"}:
                    raise ValueError("sharded checkpoint requires a safetensors index")
                if _single_tensor_bytes(path/"model.safetensors") != self._record["index_tensor_bytes"]:
                    raise ValueError("local checkpoint tensor inventory differs from scope")
                return
        index = json.loads(index_path.read_text())
        shards = set(index["weight_map"].values())
        if self._record["schema_version"] == 2 and shards != declared:
            raise ValueError("local checkpoint index names differ from declared weight files")
        if (index["metadata"]["total_size"] != self._record["index_tensor_bytes"]
                or len(shards) != self._record["shards"]
                or sum((path/name).stat().st_size for name in shards) != self._record["shard_bytes"]):
            raise ValueError("local checkpoint shard inventory differs from scope")

    def verify_tokenizer(self, tokenizer):
        actual = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        template = json.dumps({"template": tokenizer.chat_template, "enable_thinking": False}, sort_keys=True)
        if (actual != self._record["tokenizer_hash"]
                or hashlib.sha256(template.encode()).hexdigest() != self._record["template_hash"]):
            raise ValueError("running tokenizer/nonthinking template differs from scope")

    def verify_native_settings(self, settings):
        if (settings.revision is not None or settings.dtype != "bfloat16" or settings.kv_cache_dtype != "bfloat16"
                or settings.quantization is not None or settings.load_format == "dummy"
                or json.loads(settings.json_model_override_args or "{}")):
            raise ValueError("local scope requires actual checkpoint loading, BF16 weights/KV, no revision, quantization or model override")
        if (Path(settings.model_path).resolve() != Path(self.model_path)
                or Path(settings.tokenizer_path).resolve() != Path(self.model_path)):
            raise ValueError("native model/tokenizer paths differ from local scope")
        if settings.context_length is None or not 0 < settings.context_length <= self.context_limit:
            raise ValueError("native context length exceeds the scoped model configuration")


def load_model_scope(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return LocalModelScope(json.loads(raw), source_path=path, source_sha256=hashlib.sha256(raw).hexdigest())


def scope_record_from_inspection(model, *, source):
    """Prepare an explicit record from an already observed local inspection."""
    return {"schema_version": 1, "identity_kind": "local_snapshot_metadata", "model_path": model["resolved_path"],
            "config": deepcopy(model["config"]), "canonical_config_sha256": model["canonical_config_sha256"],
            "index_tensor_bytes": model["index_tensor_bytes"], "shards": model["shards"], "shard_bytes": model["shard_bytes"],
            "tokenizer_hash": model["fingerprint"]["tokenizer_hash"], "template_hash": model["fingerprint"]["template_hash"],
            "hf_revision": None, "weight_content_verified": False, "source": deepcopy(source)}


def scope_record_from_checkpoint(*, model_path, config, tokenizer_hash, template_hash,
                                 index_tensor_bytes, weight_files, source):
    """Bind digests computed once by the checkpoint publisher, without reading weights.

    The caller owns publication into a new checkpoint directory. These digests
    identify that artifact; verify_files later checks only metadata/inventory.
    """
    files = _checkpoint_files(weight_files)
    return {"schema_version": 2, "identity_kind": "local_checkpoint_artifact", "model_path": str(model_path),
            "config": deepcopy(config), "canonical_config_sha256": canonical_config_hash(config),
            "index_tensor_bytes": index_tensor_bytes, "shards": len(files),
            "shard_bytes": sum(entry["bytes"] for entry in files), "weight_files": files,
            "tokenizer_hash": tokenizer_hash, "template_hash": template_hash,
            "hf_revision": None, "weight_content_verified": False, "source": deepcopy(source)}


def launch_argv(scope, arguments):
    # These values are owned by the explicit record, not trailing overrides.
    reserved = {"--model-path", "--tokenizer-path", "--revision"}
    if any(item.split("=", 1)[0] in reserved for item in arguments):
        raise ValueError("model/tokenizer/revision launch values are supplied only by model scope")
    declared_path = os.environ.get("CONTEXTKV_MODEL_PATH")
    if declared_path and Path(declared_path).resolve() != Path(scope.model_path):
        raise ValueError("CONTEXTKV_MODEL_PATH conflicts with explicit model scope")
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", scope.model_path,
            "--tokenizer-path", scope.model_path, *arguments]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--launch", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    scope = load_model_scope(args.scope)
    if not args.launch:
        if remaining:
            parser.error("unexpected arguments without --launch")
        print(json.dumps(scope.manifest(), indent=2))
        return 0
    scope.verify_files()
    scope.verify_fingerprint(json.loads(os.environ["CONTEXTKV_FINGERPRINT_JSON"]))
    arguments = launch_argv(scope, remaining)
    os.environ["CONTEXTKV_MODEL_SCOPE_PATH"] = scope.source_path
    os.execv(sys.executable, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
