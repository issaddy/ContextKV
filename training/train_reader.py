"""Train and materialize a BF16 Qwen3-32B reader LoRA; keep R0 immutable."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from controller.templates import token_hash
from training.reader_data import (augment_retrieved_reads, export_reader_traces, prepare_reader_data,
                                  review_reader_traces, rows, write_rows)


def encode_example(row, tokenizer, *, max_length):
    """Train only the reviewed continuation of the exact recorded online prompt."""
    text = tokenizer.apply_chat_template(row["messages"], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prompt = list(tokenizer.encode(text, add_special_tokens=False))
    if token_hash(prompt) != row["prompt_token_hash"]:
        raise ValueError(f"recorded tokenizer/template identity differs for {row['example_id']}")
    target = list(tokenizer.encode(row["target_text"], add_special_tokens=False))
    if not target or target[-1] != tokenizer.eos_token_id:
        target.append(tokenizer.eos_token_id)
    if len(prompt)+len(target) > max_length:
        return None, {"example_id": row["example_id"], "task_id": row["task_id"],
            "prompt_tokens": len(prompt), "target_tokens": len(target), "max_length": max_length}
    return {"input_ids": prompt+target, "attention_mask": [1]*(len(prompt)+len(target)),
            "labels": [-100]*len(prompt)+target}, None


class ReaderCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(feature["input_ids"]) for feature in features)
        return {name: torch.tensor([feature[name]+[fill]*(length-len(feature[name]))
                    for feature in features], dtype=torch.long)
                for name, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def _asset_digest(directory):
    directory = Path(directory)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
             if path.is_file() and path.suffix in {".safetensors", ".json"}}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return identity, files


def train_reader(config, *, max_steps=None, resume_from_checkpoint=None):
    preparation_started = time.monotonic()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from engine_patches.contextkv_model_scope import load_model_scope

    settings = config["reader_training"]
    scope = load_model_scope(settings["base_scope"])
    scope.verify_files()
    output = Path(settings["output_dir"])
    if output.resolve() == Path(scope.model_path).resolve():
        raise ValueError("R1 output must be separate from the immutable R0 checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    set_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(scope.model_path, local_files_only=True, trust_remote_code=False)
    scope.verify_tokenizer(tokenizer)
    data_root = Path(settings["data"]["output_dir"])
    source_manifest = json.loads((data_root/"data_manifest.json").read_text())
    encoded, exclusions, counts = {}, [], {}
    for split in ("train", "validation"):
        encoded[split] = []
        selected = []
        for row in rows(data_root/f"{split}.jsonl"):
            example, oversized = encode_example(row, tokenizer, max_length=settings["max_length"])
            if example is not None:
                encoded[split].append(example)
                selected.append(row)
            else:
                exclusions.append({**oversized, "split": split})
        counts[split] = {"examples": len(selected), "turns": len({row["task_id"] for row in selected}),
            "phases": dict(Counter(row["phase"] for row in selected)),
            "model_input_tokens_per_epoch": sum(len(row["input_ids"]) for row in encoded[split]),
            "supervised_tokens_per_epoch": sum(sum(token != -100 for token in row["labels"]) for row in encoded[split])}
    if counts["train"]["turns"] < settings["data"]["minimum_train_turns"]:
        raise ValueError(f"training needs {settings['data']['minimum_train_turns']} actual turns; selected {counts['train']['turns']}")
    model = AutoModelForCausalLM.from_pretrained(scope.model_path, dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, attn_implementation=settings["attention_implementation"])
    model.config.use_cache = False
    lora = settings["lora"]
    if settings.get("initial_adapter"):
        model = PeftModel.from_pretrained(model, settings["initial_adapter"], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
            bias="none", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    optimizer = settings["optimizer"]
    arguments = TrainingArguments(output_dir=str(output/"checkpoints"),
        num_train_epochs=settings["epochs"], max_steps=settings["max_steps"] if max_steps is None else max_steps,
        per_device_train_batch_size=settings["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=optimizer["learning_rate"], weight_decay=optimizer["weight_decay"],
        warmup_steps=optimizer["warmup_ratio"], lr_scheduler_type="cosine", max_grad_norm=optimizer["max_grad_norm"],
        bf16=True, tf32=True, optim="adamw_torch_fused", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=settings["logging_steps"], save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        dataloader_num_workers=0, dataloader_pin_memory=True,
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        report_to="none", seed=settings["seed"], data_seed=settings["seed"], remove_unused_columns=False)

    class CountingTrainer(Trainer):
        input_tokens_observed = 0
        target_tokens_observed = 0
        microbatches_observed = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if model.training:
                self.input_tokens_observed += int(inputs["attention_mask"].sum().item())
                self.target_tokens_observed += int((inputs["labels"] != -100).sum().item())
                self.microbatches_observed += 1
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    class StepLedger(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            value = {"step": state.global_step, "elapsed_ms": (time.monotonic()-started)*1000,
                "invocation_id": settings.get("invocation_id"),
                "rank": rank, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "model_input_tokens": trainer.input_tokens_observed,
                "supervised_tokens": trainer.target_tokens_observed}
            with (output/f"training_steps.rank{rank}.jsonl").open("a") as sink:
                sink.write(json.dumps(value)+"\n")

    trainer = CountingTrainer(model=model, args=arguments, train_dataset=encoded["train"],
        eval_dataset=encoded["validation"], data_collator=ReaderCollator(tokenizer), callbacks=[StepLedger()])
    if trainer.is_world_process_zero():
        write_rows(output/"oversized_examples.jsonl", exclusions)
        (output/"effective_training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic()-started)*1000
    training_ended = time.monotonic()
    local_cost = {"rank": rank, "device": torch.cuda.get_device_name(rank), "elapsed_ms": elapsed_ms,
        "model_input_tokens": trainer.input_tokens_observed, "supervised_tokens": trainer.target_tokens_observed,
        "microbatches": trainer.microbatches_observed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "generated_tokens": 0}
    gathered = [local_cost]
    if world_size > 1:
        gathered = [None]*world_size
        torch.distributed.all_gather_object(gathered, local_cost)
    trainer.save_model(str(output/"adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output/"adapter")
        identity, files = _asset_digest(output/"adapter")
        provenance = {"schema_version": 1, "reader_id": f"contextkv-reader-r1:{identity}",
            "status": "trained", "base_reader": scope.manifest(), "adapter_files": files,
            "lora": lora, "dtype": "bfloat16", "world_size": world_size,
            "trainable_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
            "global_batch_size": world_size*settings["batch_size"]*settings["gradient_accumulation_steps"],
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "training_data": source_manifest, "encoded_data": counts,
            "actual_max_steps": trainer.state.global_step, "train_result": result.metrics,
            "training_evidence": "training_cost.json and training_steps.rank*.jsonl",
            "answer_quality_evaluation": "not performed by training; report separately against R0"}
        provenance["invocation_id"] = settings.get("invocation_id")
        manifest_text = json.dumps(provenance, ensure_ascii=False, indent=2)+"\n"
        (output/"reader_manifest.json").write_text(manifest_text)
        (output/f"reader_manifest.{identity}.json").write_text(manifest_text)
        cost_text = json.dumps({"ranks": gathered,
            "total_model_input_tokens": sum(row["model_input_tokens"] for row in gathered),
            "total_supervised_tokens": sum(row["supervised_tokens"] for row in gathered),
            "elapsed_ms": max(row["elapsed_ms"] for row in gathered),
            "invocation_id": settings.get("invocation_id"),
            "time_basis": "CUDA-synchronized training wall time; not kernel-only time"}, indent=2)+"\n"
        (output/"training_cost.json").write_text(cost_text)
        (output/f"training_cost.{settings.get('invocation_id', time.time_ns())}.json").write_text(cost_text)
        for name, target in (("README.md", "BASE_MODEL_CARD.md"), ("LICENSE", "LICENSE"), ("NOTICE", "BASE_NOTICE")):
            if (Path(scope.model_path)/name).is_file():
                shutil.copyfile(Path(scope.model_path)/name, output/target)
    checkpoint_ended = time.monotonic()
    windows = {"preparation": [preparation_started, started], "training": [started, training_ended],
               "checkpoint_save": [training_ended, checkpoint_ended]}
    if encoded["validation"] and settings.get("evaluate_train_holdout", True):
        validation_started = time.monotonic()
        metrics = trainer.evaluate()
        torch.cuda.synchronize()
        windows["validation"] = [validation_started, time.monotonic()]
        if trainer.is_world_process_zero():
            evaluation = output/"evaluation"
            evaluation.mkdir(exist_ok=True)
            (evaluation/"train_partition_holdout_loss.json").write_text(json.dumps({
                "evidence_kind": "conversation-heldout loss from native training partition",
                "metrics": metrics, "dataset": counts["validation"],
                "public_dev_test": False, "answer_quality_evaluation": False}, indent=2)+"\n")
    return {"output_dir": str(output), "optimizer_steps": trainer.state.global_step, "encoded_data": counts,
            "measurement_windows": windows}


def merge_reader(config):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine_patches.contextkv_model_scope import load_model_scope
    settings = config["reader_training"]
    scope = load_model_scope(settings["base_scope"])
    output = Path(settings["output_dir"])
    destination = Path(settings["merged_model_dir"])
    if destination.resolve() == Path(scope.model_path).resolve():
        raise ValueError("merged R1 must be separate from R0")
    model = AutoModelForCausalLM.from_pretrained(scope.model_path, dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=False, device_map={"": 0})
    model = PeftModel.from_pretrained(model, output/"adapter").merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(scope.model_path, local_files_only=True)
    tokenizer.save_pretrained(destination)
    manifest = json.loads((output/"reader_manifest.json").read_text())
    record = scope.manifest()["record"]
    record.update(model_path=str(destination.resolve()), revision=manifest["reader_id"],
        identity_kind="contextkv_bf16_reader_lora_merge", config=json.loads((destination/"config.json").read_text()))
    path = output/"model_scope.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    profile = yaml.safe_load(Path(settings["r0_run_config"]).read_text())
    profile["model"] = {"scope_path": str(path.resolve())}
    profile["engine"]["base_url"] = settings["r1_base_url"]
    profile["state_dir"] = str(output/"serving-state")
    (output/"r1_run_config.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    return {"reader_id": manifest["reader_id"], "merged_model_dir": str(destination.resolve()),
            "model_scope": str(path.resolve()), "run_config": str((output/"r1_run_config.yaml").resolve())}


def physical_device_ids(visible=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    if visible is None:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        return list(map(int, output.split()))
    if visible == "":
        return []
    identifiers = visible.split(",")
    if all(value.isdecimal() for value in identifiers):
        return list(map(int, identifiers))
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    indices = {uuid.strip(): int(index) for line in output.splitlines() for index, uuid in [line.split(",")]}
    return [indices[identifier] for identifier in identifiers]


async def measured_command(config, command, function, *, asynchronous=False):
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    settings = config["reader_training"]
    invocation_id = str(time.time_ns())
    settings["invocation_id"] = invocation_id
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if rank != 0:
        return await asyncio.to_thread(function)
    processes = [os.getppid() if world > 1 else os.getpid()]
    if command == "review":
        from urllib.parse import urlparse
        devices = settings["review"]["gpu_devices"]
        processes.extend(listening_processes([urlparse(url).port for url in settings["review"]["base_urls"]]))
    elif command in {"export", "prepare", "augment-reads"}:
        devices = []
    else:
        devices = physical_device_ids()
    sampler = StudyResourceSampler(devices, processes,
        interval=settings.get("resource_sample_seconds", .5))
    await sampler.start()
    started = time.monotonic()
    completed = False
    try:
        result = await function() if asynchronous else await asyncio.to_thread(function)
        completed = True
    finally:
        ended = time.monotonic()
        await sampler.stop()
        output = Path(settings["output_dir"] if command in {"train", "merge"} else settings["data"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        observation = json.dumps(sampler.observation(), indent=2)+"\n"
        (output/f"{command}_physical_resources.json").write_text(observation)
        (output/f"{command}_physical_resources.{invocation_id}.json").write_text(observation)
        cost = {"command": command, "start": started, "end": ended,
            "invocation_id": invocation_id, "status": "completed" if completed else "failed",
            "complete_ms": (ended-started)*1000, "physical": integrate_physical(sampler.samples, started, ended),
            "gpu_devices": devices, "world_size": world,
            "physical_counts": "each physical GPU counted once; PSS summed across declared process roots and descendants"}
        (output/f"{command}_cost.json").write_text(json.dumps(cost, indent=2)+"\n")
        (output/f"{command}_cost.{invocation_id}.json").write_text(json.dumps(cost, indent=2)+"\n")
    if "measurement_windows" in result:
        cost["phases"] = {name: {"start": start, "end": end,
            **integrate_physical(sampler.samples, start, end)} for name, (start, end) in result["measurement_windows"].items()}
        (output/f"{command}_cost.json").write_text(json.dumps(cost, indent=2)+"\n")
        (output/f"{command}_cost.{invocation_id}.json").write_text(json.dumps(cost, indent=2)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "augment-reads", "review", "prepare", "train", "merge"))
    parser.add_argument("--config", default="configs/reader_r1.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.command == "export":
        operation = partial(export_reader_traces, config)
    elif args.command == "augment-reads":
        operation = partial(augment_retrieved_reads, config)
    elif args.command == "review":
        operation = partial(review_reader_traces, config, limit_turns=args.limit_turns, concurrency=args.concurrency)
    elif args.command == "prepare":
        operation = partial(prepare_reader_data, config)
    elif args.command == "train":
        operation = partial(train_reader, config, max_steps=args.max_steps, resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        operation = partial(merge_reader, config)
    value = asyncio.run(measured_command(config, args.command, operation, asynchronous=args.command == "review"))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
