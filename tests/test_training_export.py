"""Export boundary tests plus an opt-in real PEFT merge on tiny random Qwen3."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from training.export import (
    checkpoint_inventory, copy_tokenizer_files, export_checkpoint, load_training_artifact,
)


def write_training_manifest(run, base_identity):
    binding = {"base_model_identity": base_identity, "files_sha256": {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted((run / "adapter").iterdir()) if path.is_file()}}
    manifest = {"status": "completed", "artifact_kind": "peft_adapter_not_native_checkpoint",
        "model_scope": {"identity": base_identity}, "artifact_binding": binding,
        "artifact_identity": "contextkv-lora:" + sha256(
            json.dumps(binding, sort_keys=True).encode()).hexdigest()}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def adapter_run(tmp_path):
    run = tmp_path / "training"
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM"}))
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic inventory fixture")
    write_training_manifest(run, "base-fixture")
    return run


def test_completed_adapter_binding_and_relocation(adapter_run, tmp_path):
    relocated = tmp_path / "relocated"
    adapter_run.rename(relocated)
    adapter, manifest = load_training_artifact(relocated, "base-fixture")
    assert adapter == relocated / "adapter"
    assert manifest["status"] == "completed"


@pytest.mark.parametrize("status", ["running", "failed"])
def test_incomplete_training_rejected(adapter_run, status):
    path = adapter_run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = status
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        load_training_artifact(adapter_run, "base-fixture")


def test_wrong_base_rejected_before_adapter_read(adapter_run):
    (adapter_run / "adapter" / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="base identity"):
        load_training_artifact(adapter_run, "another-base")


@pytest.mark.parametrize("change", ["alter", "remove", "add", "identity"])
def test_adapter_inventory_or_identity_tampering_rejected(adapter_run, change):
    adapter = adapter_run / "adapter"
    if change == "alter":
        (adapter / "adapter_model.safetensors").write_bytes(b"changed bytes")
    elif change == "remove":
        (adapter / "adapter_config.json").unlink()
    elif change == "add":
        (adapter / "unexpected.json").write_text("{}")
    else:
        path = adapter_run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["artifact_identity"] = "contextkv-lora:" + "0" * 64
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory|identity"):
        load_training_artifact(adapter_run, "base-fixture")


def test_tokenizer_copied_byte_exact_without_reserialization(tmp_path):
    source, output = tmp_path / "base", tmp_path / "export"
    source.mkdir()
    output.mkdir()
    originals = {"tokenizer.json": b'{"preserve":  1}\n',
                 "tokenizer_config.json": b'{"padding_side":"left"}',
                 "chat_template.jinja": b"{{ messages }}\n",
                 "chat_templates/tool_use.jinja": b"{{ tools }}\n"}
    for name, value in originals.items():
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(value)
    (source / "adapter_config.json").write_text("{}")
    assert set(copy_tokenizer_files(source, output)) == set(originals)
    assert {name: (output / name).read_bytes() for name in originals} == originals
    assert not (output / "adapter_config.json").exists()


def test_existing_export_never_overwritten(tmp_path):
    sentinel = tmp_path / "config.json"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError, match="already exists"):
        export_checkpoint("missing-training", "missing-scope", tmp_path)
    assert sentinel.read_text() == "original"


def test_export_help_does_not_import_training_runtime():
    code = ("import sys; from training.export import main; "
            "\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code == 0\n"
            "assert not {'torch', 'peft', 'transformers'} & set(sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            text=True, capture_output=True)
    assert "--training-run" in result.stdout


@pytest.mark.skipif(os.environ.get("CONTEXTKV_TEST_TRAINING") != "1",
                   reason="opt-in CPU tiny-Qwen3 PEFT export contract")
def test_tiny_qwen3_merge_export_reload_and_failure(
        qwen3_tokenizer_template, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, set_seed
    from engine_patches.contextkv_model_scope import (
        load_model_scope, scope_record_from_checkpoint,
    )

    template = qwen3_tokenizer_template
    base, run, output = tmp_path / "base", tmp_path / "training", tmp_path / "export"
    base.mkdir()
    run.mkdir()
    set_seed(42)
    config = Qwen3Config(vocab_size=len(template.tokenizer), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
        attention_dropout=0.0, use_cache=False, use_sliding_window=False,
        sliding_window=None, rope_scaling=None, bos_token_id=template.tokenizer.bos_token_id,
        eos_token_id=template.tokenizer.eos_token_id, pad_token_id=template.tokenizer.pad_token_id)
    model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16, device="cpu")
    original_parameters = dict(model.named_parameters())
    original_values = {name: parameter.detach().clone()
                       for name, parameter in original_parameters.items()}
    model.save_pretrained(base, safe_serialization=True)
    tokenizer_source = Path(os.environ["CONTEXTKV_TEST_TOKENIZER_PATH"])
    copy_tokenizer_files(tokenizer_source, base)
    original_weight = (base / "model.safetensors").read_bytes()
    weights, tensor_bytes = checkpoint_inventory(base)
    record = scope_record_from_checkpoint(model_path=str(base),
        config=json.loads((base / "config.json").read_text()),
        tokenizer_hash=template.fingerprint.tokenizer_hash,
        template_hash=template.fingerprint.template_hash,
        index_tensor_bytes=tensor_bytes, weight_files=weights,
        source={"kind": "synthetic_tiny_random_test_only"})
    base_scope_path = tmp_path / "base-scope.json"
    base_scope_path.write_text(json.dumps(record))
    base_scope = load_model_scope(base_scope_path)
    adapter = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
        target_modules="all-linear", lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    # Nonzero deterministic adapter; this is a format/merge test, not training.
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.025)
    adapter.save_pretrained(run / "adapter", safe_serialization=True)
    write_training_manifest(run, base_scope.identity)
    tokens = torch.tensor([[12, 34, 56, 78]])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        # Independent unmerged reference shares the original base file, not a
        # model object that the merge helper may mutate.
        reference = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, dtype=torch.bfloat16), run / "adapter").eval()
        with torch.no_grad():
            unmerged_logits = reference(tokens).logits
        manifest = export_checkpoint(run, base_scope_path, output)
        exported = load_model_scope(output / "model-scope.json")
        exported.verify_files()
        assert manifest["status"] == "completed"
        assert manifest["quality_validated"] is False
        assert exported.identity != base_scope.identity
        assert len(manifest["weight_files"]) == 1
        assert not (output / "model.safetensors.index.json").exists()
        assert not (output / "adapter_config.json").exists()
        assert (base / "model.safetensors").read_bytes() == original_weight
        for name in manifest["tokenizer_files"]:
            assert (base / name).read_bytes() == (output / name).read_bytes()
        loaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True,
                                                     dtype=torch.bfloat16).eval()
        assert loaded.config.use_cache is True
        assert loaded.generation_config.use_cache is True
        assert all(parameter.dtype == torch.bfloat16 for parameter in loaded.parameters())
        assert all("lora_" not in name for name in loaded.state_dict())
        assert not torch.equal(loaded.model.layers[0].self_attn.q_proj.weight,
                               original_values["model.layers.0.self_attn.q_proj.weight"])
        with torch.no_grad():
            merged_logits = loaded(tokens).logits
        # BF16 weight accumulation changes rounding; numerical closeness only,
        # not exact logits, identical sampled outputs or improved QA quality.
        torch.testing.assert_close(merged_logits, unmerged_logits, rtol=0.03, atol=0.003)
        assert not torch.cuda.is_initialized()

        # Identical architecture and file sizes cannot identify a trained model:
        # a second adapter changes only tensor values, and must get a new scope.
        second_run, second_output = tmp_path / "second-training", tmp_path / "second-export"
        second_run.mkdir()
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.05)
        adapter.save_pretrained(second_run / "adapter", safe_serialization=True)
        write_training_manifest(second_run, base_scope.identity)
        second = export_checkpoint(second_run, base_scope_path, second_output)
        assert second["model_identity"] != manifest["model_identity"]
        assert second["index_tensor_bytes"] == manifest["index_tensor_bytes"]
        assert [entry["bytes"] for entry in second["weight_files"]] == [
            entry["bytes"] for entry in manifest["weight_files"]]
        assert (base / "model.safetensors").read_bytes() == original_weight
        for name, parameter in original_parameters.items():
            assert torch.equal(parameter, original_values[name]), name

        bad_run = tmp_path / "nonfinite-training"
        bad_run.mkdir()
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(float("nan"))
        adapter.save_pretrained(bad_run / "adapter", safe_serialization=True)
        write_training_manifest(bad_run, base_scope.identity)
        failed = tmp_path / "failed"
        with pytest.raises(ValueError, match="NaNs detected"):
            export_checkpoint(bad_run, base_scope_path, failed)
        assert json.loads((failed / "export-manifest.json").read_text())["status"] == "failed"
        assert not (failed / "model-scope.json").exists()
    finally:
        torch.set_num_threads(previous_threads)
