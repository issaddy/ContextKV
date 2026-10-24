"""Prepare exact-prefix SFT data or train a bounded offline Qwen3 LoRA adapter.

Uses Transformers Trainer, its label-preserving collator and PEFT. The resulting
adapter is an offline research artifact, not a native SGLang model scope.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import os

from controller.templates import HuggingFaceTemplate
from engine_patches.contextkv_model_scope import load_model_scope
from training.data import load_prepared, prepare_examples


def train_adapter(model, tokenizer, features, *, output_dir, max_steps,
                  learning_rate=2e-4, rank=16, gradient_accumulation_steps=1,
                  seed=42, use_cpu=False):
    """Use the mature training loop; no packing, truncation or rebuilt labels."""
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model.config.use_cache = False
    tokenizer.padding_side = "right"
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2*rank,
        lora_dropout=0.0, target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    arguments = TrainingArguments(output_dir=str(output_dir), max_steps=max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate, lr_scheduler_type="linear", warmup_steps=0.03,
        seed=seed, data_seed=seed, use_cpu=use_cpu, bf16=not use_cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", max_grad_norm=1.0, save_strategy="no", eval_strategy="no",
        logging_steps=1, logging_nan_inf_filter=False, report_to="none",
        disable_tqdm=True, dataloader_num_workers=0,
        dataloader_pin_memory=not use_cpu, remove_unused_columns=False)
    trainer = Trainer(model=model, args=arguments, train_dataset=features,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=None, padding=True,
                                            label_pad_token_id=-100))
    result = trainer.train()
    adapter_path = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_path))
    trainer.save_state()
    return {"metrics": result.metrics, "global_step": result.global_step,
            "training_loss": result.training_loss, "trainable_parameters": trainable,
            "total_parameters": total, "arguments": arguments.to_dict(),
            "adapter_directory": str(adapter_path.resolve())}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def train(args):
    # Device visibility is local to this command, before importing torch.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from transformers import AutoModelForCausalLM

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("training output already exists; choose a new run")
    scope = load_model_scope(args.model_scope)
    scope.verify_files()
    template = HuggingFaceTemplate(scope.model_path, None, local_files_only=True,
                                  model_identity=scope.identity)
    scope.verify_fingerprint(asdict(template.fingerprint))
    features, prepared = load_prepared(args.prepared, template.fingerprint)
    if prepared["max_tokens"] > scope.context_limit:
        raise ValueError("prepared sequence policy exceeds the base model context limit")
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA training requires an available BF16-capable GPU")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "artifact_kind": "peft_adapter_not_native_checkpoint",
        "model_scope": scope.manifest(), "prepared_directory": str(Path(args.prepared).resolve()),
        "prepared_manifest": prepared, "settings": {
            key: getattr(args, key) for key in ("max_steps", "learning_rate", "rank",
                "gradient_accumulation_steps", "seed", "device")},
        "runtime_versions": {name: metadata.version(name) for name in
                             ("torch", "transformers", "peft", "accelerate", "safetensors")},
        "quality_validated": False, "native_serving_compatible": False}
    _write(output / "manifest.json", manifest)
    try:
        model = AutoModelForCausalLM.from_pretrained(scope.model_path, local_files_only=True,
            dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa")
        result = train_adapter(model, template.tokenizer, features, output_dir=output,
            max_steps=args.max_steps, learning_rate=args.learning_rate, rank=args.rank,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed, use_cpu=args.device == "cpu")
        if not math.isfinite(result["training_loss"]):
            raise RuntimeError("training produced a nonfinite loss")
        inventory = {p.name: sha256(p.read_bytes()).hexdigest()
                     for p in sorted((output / "adapter").iterdir()) if p.is_file()}
        binding = {"base_model_identity": scope.identity, "files_sha256": inventory}
        identity = "contextkv-lora:" + sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        manifest.update(status="completed", result=result, artifact_identity=identity,
                        artifact_binding=binding)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "steps": result["global_step"],
                      "adapter_directory": result["adapter_directory"],
                      "native_serving_compatible": False}))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU-only exact inference-prefix/target encoding")
    prepare.add_argument("--examples", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--model-scope", required=True)
    prepare.add_argument("--tokenizer-path", help="Local tokenizer metadata override; no model weights")
    prepare.add_argument("--max-tokens", required=True, type=int)
    fit = commands.add_parser("train", help="Bounded offline PEFT training; does not launch serving")
    fit.add_argument("--prepared", required=True)
    fit.add_argument("--model-scope", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--max-steps", required=True, type=int)
    fit.add_argument("--rank", type=int, default=16)
    fit.add_argument("--learning-rate", type=float, default=2e-4)
    fit.add_argument("--gradient-accumulation-steps", type=int, default=1)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        scope = load_model_scope(args.model_scope)
        if not 0 < args.max_tokens <= scope.context_limit:
            parser.error("max-tokens must fit the scoped context limit")
        template = HuggingFaceTemplate(args.tokenizer_path or scope.model_path, None,
                                       local_files_only=True, model_identity=scope.identity)
        scope.verify_fingerprint(asdict(template.fingerprint))
        manifest = prepare_examples(args.examples, args.output, template, max_tokens=args.max_tokens)
        print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}))
        return manifest
    if (min(args.max_steps, args.rank, args.gradient_accumulation_steps) < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("steps/rank/accumulation and a finite learning rate must be positive")
    return train(args)


if __name__ == "__main__":
    main()
