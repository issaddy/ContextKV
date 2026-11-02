"""Compare FP32 masters/Adam with BF16 encoder autocast and inference."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path


def run(examples_path, base_path, same_weights_asset, output_dir, *, device="cuda:2",
        train_examples=32, probe_examples=64):
    import torch
    from adapters.support_model import SupportInput
    from controller.context_schema import EvidenceUnit
    from controller.serialization import decode
    from training.support_model import TrainedSupportBackend, train_support
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(examples_path).read_text().splitlines() if line]
    train_rows = rows[:train_examples]
    probes = rows[train_examples:train_examples+probe_examples]
    training_path = output/"train.examples.jsonl"
    training_path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in train_rows))
    inputs = tuple(SupportInput(row["task_id"], row["need_id"], row["query"],
        tuple(decode(EvidenceUnit, unit) for unit in row["units"]),
        tuple(decode(EvidenceUnit, unit) for unit in row.get("dependencies", []))) for row in probes)

    def infer(asset, dtype):
        backend = TrainedSupportBackend.load(asset, device=device, batch_size=8, encoder_dtype=dtype)
        warmup = backend.predict(inputs[:8])
        warmup_usage = dict(backend.usage)
        torch.cuda.reset_peak_memory_stats(device)
        predictions = backend.predict(inputs)
        usage = {key: value-warmup_usage[key] for key, value in backend.usage.items()}
        labeled = [(prediction, row["labels"]) for prediction, row in zip(predictions, probes, strict=True)
                   if prediction is not None]
        measured = {"deployment": backend.deployment, "warmup_examples": len(warmup),
            "warmup_usage": warmup_usage, "usage": usage,
            "support_accuracy": sum(prediction.disposition == labels["disposition"]
                                    for prediction, labels in labeled)/len(labeled),
            "scalar_mse": {key: sum((getattr(prediction, key)-labels[key])**2
                                     for prediction, labels in labeled)/len(labeled)
                           for key in ("applicability", "direct_support", "joint_support", "bridge_value")},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "predictions": [asdict(prediction) if prediction is not None else None for prediction in predictions]}
        backend.close()
        return measured

    def compare(left, right):
        pairs = [(a, b) for a, b in zip(left["predictions"], right["predictions"], strict=True)
                 if a is not None and b is not None]
        return {"scored_pairs": len(pairs),
            "disposition_agreement": sum(a["disposition"] == b["disposition"] for a, b in pairs)/len(pairs),
            "mean_absolute_difference": {key: sum(abs(a[key]-b[key]) for a, b in pairs)/len(pairs)
                for key in ("applicability", "direct_support", "joint_support", "bridge_value")},
            "speed_ratio_fp32_over_bf16": left["usage"]["support_ms"]/right["usage"]["support_ms"]}

    fixed_fp32 = infer(same_weights_asset, "float32")
    fixed_bf16 = infer(same_weights_asset, "bfloat16")
    trained, trained_predictions = {}, {}
    for profile, dtype in (("fp32", "float32"), ("autocast_bf16", "bfloat16")):
        asset = output/(profile+"-asset")
        trained[profile] = train_support(training_path, base_path, asset, device=device, epochs=1,
            batch_size=4, encoder_autocast_dtype=dtype, inference_encoder_dtype=dtype)
        trained_predictions[profile] = infer(asset, dtype)
    report = {"scope": "precision_and_runtime_probe", "device": device,
        "hardware": torch.cuda.get_device_name(device), "train_examples": len(train_rows),
        "probe_examples": len(probes), "same_weights_asset": str(same_weights_asset),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "same_weights_inference": {"fp32": fixed_fp32, "bf16": fixed_bf16,
                                   "comparison": compare(fixed_fp32, fixed_bf16)},
        "training": trained, "trained_predictions": trained_predictions,
        "trained_prediction_comparison": compare(trained_predictions["fp32"], trained_predictions["autocast_bf16"])}
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return {"output": str(output/"results.json"),
        "same_weights_inference": report["same_weights_inference"]["comparison"],
        "training_ms": {key: value["support_training"]["elapsed_ms"] for key, value in trained.items()},
        "training_parameter_dtypes": {key: value["training_parameter_dtypes"] for key, value in trained.items()},
        "optimizer_state_dtypes": {key: value["optimizer_state_dtypes"] for key, value in trained.items()},
        "trained_prediction_comparison": report["trained_prediction_comparison"]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--same-weights-asset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--probe-examples", type=int, default=64)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.examples, args.base, args.same_weights_asset, args.output,
        device=args.device, train_examples=args.train_examples, probe_examples=args.probe_examples), ensure_ascii=False))


if __name__ == "__main__":
    main()
