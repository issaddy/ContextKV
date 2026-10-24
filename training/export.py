"""Export a completed ContextKV LoRA run as a separate BF16 Qwen3 checkpoint.

This is an offline CPU operation. It neither starts a server nor establishes
answer quality. The output scope is published last, after the dense checkpoint
and original tokenizer files have been written successfully.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import file_digest, sha256
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import struct

from engine_patches.contextkv_model_scope import load_model_scope


def _sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_training_artifact(training_run, base_identity):
    """Check the completed adapter artifact once, at the export boundary."""
    run = Path(training_run).resolve(strict=True)
    manifest = json.loads((run / "manifest.json").read_text())
    if (manifest.get("status") != "completed"
            or manifest.get("artifact_kind") != "peft_adapter_not_native_checkpoint"):
        raise ValueError("export requires a completed ContextKV adapter training run")
    binding = manifest["artifact_binding"]
    if (binding["base_model_identity"] != base_identity
            or manifest["model_scope"]["identity"] != base_identity):
        raise ValueError("training base identity differs from supplied model scope")
    adapter = run / "adapter"
    actual = {path.name: _sha256(path) for path in sorted(adapter.iterdir()) if path.is_file()}
    if actual != binding["files_sha256"]:
        raise ValueError("adapter files differ from the completed training inventory")
    identity = "contextkv-lora:" + sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if identity != manifest["artifact_identity"]:
        raise ValueError("adapter artifact identity differs from its binding")
    if not {"adapter_config.json", "adapter_model.safetensors"} <= actual.keys():
        raise ValueError("export requires the saved safetensors LoRA adapter")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("export currently supports causal language model LoRA adapters")
    return adapter, manifest


def merge_adapter(model, adapter_path):
    """Reuse PEFT's safe merge on an independently loaded CPU BF16 base.

    The supplied model is mutated; callers must not pass a live serving model.
    BF16 merging rounds the updated weights, so unmerged and merged logits are
    not promised to be bitwise equal. PEFT checks finite merged values, not QA.
    """
    import torch
    from peft import PeftModel

    model = model.to(device="cpu", dtype=torch.bfloat16)
    adapter = PeftModel.from_pretrained(model, str(adapter_path),
        local_files_only=True, is_trainable=False)
    merged = adapter.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.generation_config.use_cache = True
    return merged.eval()


def copy_tokenizer_files(base_path, output):
    """Copy Qwen3's original serialization; never resave a mutated tokenizer."""
    base_path, output = Path(base_path), Path(output)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")
    copied = []
    for name in names:
        source = base_path / name
        if source.is_file():
            shutil.copyfile(source, output / name)
            copied.append(name)
    # Recent Transformers can store named chat templates in this directory.
    templates = base_path / "chat_templates"
    if templates.is_dir():
        shutil.copytree(templates, output / "chat_templates")
        copied.extend(str(path.relative_to(output)) for path in
                      sorted((output / "chat_templates").rglob("*")) if path.is_file())
    return copied


def checkpoint_inventory(output):
    """Hash each newly saved weight file once and count actual tensor bytes.

    Header offsets avoid loading a second state dict and count tied tensors only
    once. Both Transformers' single-file and sharded safetensors layouts work.
    """
    files, tensor_bytes = [], 0
    for path in sorted(Path(output).glob("model*.safetensors")):
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        tensor_bytes += sum(value["data_offsets"][1] - value["data_offsets"][0]
                            for name, value in header.items() if name != "__metadata__")
        files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    if not files or tensor_bytes <= 0:
        raise ValueError("merge did not produce dense safetensors weights")
    return files, tensor_bytes


def export_checkpoint(training_run, model_scope, output, *, max_shard_size="5GB"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("export output already exists; choose a new directory")
    scope = load_model_scope(model_scope)
    if output.is_relative_to(Path(scope.model_path)):
        raise ValueError("export output must be separate from the base checkpoint")
    adapter_path, training = load_training_artifact(training_run, scope.identity)
    if output.is_relative_to(Path(training_run).resolve()):
        raise ValueError("export output must be separate from the training run")
    scope.verify_files()

    # CLI --help and artifact checks above need neither torch nor GPU access.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModelForCausalLM
    from controller.templates import HuggingFaceTemplate
    from engine_patches.contextkv_model_scope import scope_record_from_checkpoint

    template = HuggingFaceTemplate(scope.model_path, None, local_files_only=True,
                                  model_identity=scope.identity)
    scope.verify_fingerprint(asdict(template.fingerprint))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "artifact_kind": "merged_dense_checkpoint",
        "base_model_scope": scope.manifest(), "adapter_identity": training["artifact_identity"],
        "training_run": str(Path(training_run).resolve()),
        "settings": {"device": "cpu", "dtype": "bfloat16", "safe_merge": True,
                     "safe_serialization": True, "max_shard_size": max_shard_size},
        "runtime_versions": {name: metadata.version(name) for name in
                             ("torch", "transformers", "peft", "safetensors")},
        "quality_validated": False, "native_serving_validated": False}
    manifest_path, scope_path = output / "export-manifest.json", output / "model-scope.json"
    _write(manifest_path, manifest)
    try:
        model = AutoModelForCausalLM.from_pretrained(scope.model_path,
            local_files_only=True, dtype=torch.bfloat16, device_map={"": "cpu"},
            attn_implementation="sdpa")
        merged = merge_adapter(model, adapter_path)
        merged.save_pretrained(output, safe_serialization=True, max_shard_size=max_shard_size)
        copied = copy_tokenizer_files(scope.model_path, output)
        weights, tensor_bytes = checkpoint_inventory(output)
        record = scope_record_from_checkpoint(model_path=str(output),
            config=json.loads((output / "config.json").read_text()),
            tokenizer_hash=template.fingerprint.tokenizer_hash,
            template_hash=template.fingerprint.template_hash,
            index_tensor_bytes=tensor_bytes, weight_files=weights,
            source={"kind": "contextkv_peft_merge", "base_model_identity": scope.identity,
                    "adapter_identity": training["artifact_identity"],
                    "recipe": manifest["settings"]})
        # Publish the scope only after all files and the completion record exist.
        # A failed/interrupted export is never advertised as a load-ready scope.
        from engine_patches.contextkv_model_scope import LocalModelScope
        exported = LocalModelScope(record, source_path=scope_path, source_sha256="")
        exported.verify_files()
        manifest.update(status="completed", model_identity=exported.identity,
                        tokenizer_files=copied, weight_files=weights,
                        index_tensor_bytes=tensor_bytes, model_scope=str(scope_path))
        _write(manifest_path, manifest)
        pending_scope = output / "model-scope.json.tmp"
        _write(pending_scope, record)
        pending_scope.replace(scope_path)
    except BaseException as error:
        scope_path.unlink(missing_ok=True)
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write(manifest_path, manifest)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--model-scope", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args(argv)
    manifest = export_checkpoint(args.training_run, args.model_scope, args.output,
                                 max_shard_size=args.max_shard_size)
    print(json.dumps({key: manifest[key] for key in
                      ("status", "model_identity", "model_scope", "quality_validated")}))
    return manifest


if __name__ == "__main__":
    main()
