"""Opt-in CPU training contract using random tiny Qwen3 weights, never an 8B load.

Enable CONTEXTKV_TEST_TRAINING=1 and the existing local tokenizer fixture flags
CONTEXTKV_TEST_HF_TOKENIZER=1 / CONTEXTKV_TEST_TOKENIZER_PATH. Requires the training
extra; no dependency installation or model download is performed by this test.
"""
import json
import math
import os
from pathlib import Path

import pytest

from controller.schema import Message
from training.sft import train_adapter
from training.tokenization import encode_example


@pytest.mark.skipif(os.environ.get("CONTEXTKV_TEST_TRAINING") != "1",
                    reason="opt-in CPU tiny-Qwen3 training contract")
def test_tiny_qwen3_target_loss_two_steps_and_adapter_roundtrip(
        qwen3_tokenizer_template, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import DataCollatorForSeq2Seq, Qwen3Config, Qwen3ForCausalLM, set_seed

    template = qwen3_tokenizer_template
    tokenizer = template.tokenizer
    tokenizer.padding_side = "right"
    samples = [
        ((Message("system", "Source: applicants must be at least 18."),
          Message("user", "Can I apply?"), Message("assistant", "How old are you?"),
          Message("user", "I am 20.")), "You meet the stated age condition."),
        ((Message("system", "Source: a timeout alone does not authorize retrying."),
          Message("user", "What does a timeout allow?"),
          Message("assistant", "A timeout alone does not authorize a retry."),
          Message("user", "Does that mean I should retry now?")), "\nNo, not on that fact alone."),
    ]
    encoded = [encode_example(template, messages, target, max_tokens=512)
               for messages, target in samples]
    features = [example.as_features() for example in encoded]
    assert len({len(example.input_ids) for example in encoded}) == 2
    batch = DataCollatorForSeq2Seq(tokenizer, model=None, padding=True,
                                  label_pad_token_id=-100)(features)
    for index, example in enumerate(encoded):
        length = len(example.input_ids)
        assert batch["labels"][index, :length].tolist() == list(example.labels)
        assert batch["labels"][index, :example.prompt_tokens].eq(-100).all()
        assert batch["labels"][index, length:].eq(-100).all()
        assert batch["attention_mask"][index, :length].eq(1).all()
        assert batch["attention_mask"][index, length:].eq(0).all()
        assert batch["input_ids"][index, length:].eq(tokenizer.pad_token_id).all()
        assert batch["input_ids"][index, length - 1].item() == tokenizer.eos_token_id
        assert batch["labels"][index, length - 1].item() == tokenizer.eos_token_id
        assert batch["labels"][index].ne(-100).sum().item() == example.target_tokens

    # The vocabulary is the real tokenizer's; all model parameters are random
    # and tiny. This does not load or impersonate the formal Qwen3-8B scope.
    set_seed(42)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=512, attention_dropout=0.0, use_cache=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    initial_config = config.to_dict()
    model = Qwen3ForCausalLM(config).cpu()
    original_parameters = dict(model.named_parameters())
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model.eval()
        with torch.no_grad():
            before = model(**batch)
            expected_loss = torch.nn.functional.cross_entropy(
                before.logits[:, :-1].reshape(-1, len(tokenizer)),
                batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(before.loss, expected_loss, rtol=1e-6, atol=1e-6)
        del before

        result = train_adapter(model, tokenizer, features, output_dir=tmp_path / "run",
            max_steps=2, rank=2, seed=42, use_cpu=True)
        assert result["global_step"] == 2
        assert result["trainable_parameters"] == 1024
        assert math.isfinite(result["training_loss"])
        assert result["arguments"]["use_cpu"] is True
        assert result["arguments"]["bf16"] is False
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert not torch.cuda.is_initialized()
        for name, parameter in original_parameters.items():
            assert not parameter.requires_grad
            assert torch.equal(parameter.detach(), initial_state[name]), name
        assert all("lora_" in name for name, parameter in model.named_parameters()
                   if parameter.requires_grad)

        adapter_path = Path(result["adapter_directory"])
        assert (adapter_path / "tokenizer_config.json").is_file()
        assert (adapter_path / "tokenizer.json").is_file()
        assert not (adapter_path / "model.safetensors").exists()
        adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
        assert adapter_config["peft_type"] == "LORA"
        assert {name.rsplit(".", 1)[-1] for name in adapter_config["target_modules"]} == {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        weights = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
        assert len(weights) == 14
        assert all("lora_A" in name or "lora_B" in name for name in weights)
        assert sum(weight.numel() for weight in weights.values()) == 1024
        assert all(torch.isfinite(weight).all() for weight in weights.values())
        # Standard LoRA B starts at zero. The first warmup step may have LR=0;
        # two completed optimizer steps must actually change an adapter weight.
        assert any(torch.count_nonzero(weight).item() > 0
                   for name, weight in weights.items() if "lora_B" in name)

        fresh_base = Qwen3ForCausalLM(Qwen3Config.from_dict(initial_config)).cpu()
        fresh_base.load_state_dict(initial_state)
        reloaded = PeftModel.from_pretrained(fresh_base, adapter_path, local_files_only=True)
        model.eval()
        reloaded.eval()
        with torch.no_grad():
            trained = model(**batch)
            restored = reloaded(**batch)
            expected_loss = torch.nn.functional.cross_entropy(
                trained.logits[:, :-1].reshape(-1, len(tokenizer)),
                batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(trained.loss, expected_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(restored.logits, trained.logits, rtol=0, atol=0)
        torch.testing.assert_close(restored.loss, trained.loss, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
