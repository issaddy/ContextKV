"""Prepare explicit training examples without selecting or repairing semantics.

File digests belong to the preparation/load boundary. Individual examples carry
ordinary provenance, not another hash or an instruction to a downstream model.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from controller.schema import ConfigFingerprint, Message
from controller.templates import HuggingFaceTemplate
from training.tokenization import encode_example


_FORMAT = "contextkv_sft_v1"
_FEATURES = ("input_ids", "labels", "attention_mask")


def _parse_example(raw: bytes, seen: set[str]):
    row = json.loads(raw)
    if not isinstance(row, dict):
        raise ValueError("each training example must be an object")
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("each training example needs a nonempty string id")
    if identifier in seen:
        raise ValueError("duplicate training example id")
    if row.get("split") != "train":
        raise ValueError("only an explicit train split can be prepared")
    if not isinstance(row.get("target"), str):
        raise ValueError("target must be an explicit original string")
    messages = row.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("messages must be a list of explicit role/content strings")
    provenance = row.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object when supplied")
    seen.add(identifier)
    return row, tuple(Message(message["role"], message["content"]) for message in messages)


def prepare_examples(input_path: str | Path, output_dir: str | Path,
                     template: HuggingFaceTemplate, *, max_tokens: int) -> dict:
    """Stream explicit JSONL into intact target-only training features.

    The output directory must not exist. On a bad example, preserve all written
    rows and consume the remaining input only for its complete file digest and
    denominator; never tokenize, skip or repair the remaining examples. The
    failed manifest cannot be loaded for training. Counts refer to JSONL lines,
    including an invalid or unprocessed line, rather than a filtered subset.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_digest, output_digest = sha256(), sha256()
    counts = {"read": 0, "written": 0, "failed": 0, "unprocessed": 0}
    tokens = {"prompt": 0, "target": 0, "total": 0, "max_sequence": 0}
    manifest = {
        "format": _FORMAT, "status": "failed", "split": "train",
        "fingerprint": asdict(template.fingerprint), "max_tokens": max_tokens,
        "source": {"path": str(input_path.resolve()), "sha256": None},
        "output": {"file": "examples.jsonl", "sha256": None},
        "counts": counts, "tokens": tokens,
        "target_tokens_include_eos": True,
    }
    seen: set[str] = set()
    failure = None
    try:
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        with input_path.open("rb") as source, (output_dir / "examples.jsonl").open("xb") as output:
            for raw in source:
                source_digest.update(raw)
                counts["read"] += 1
                if failure is not None:
                    counts["unprocessed"] += 1
                    continue
                try:
                    row, messages = _parse_example(raw, seen)
                    encoded = encode_example(template, messages, row["target"], max_tokens=max_tokens)
                    record = {"id": row["id"], **encoded.as_features(),
                              "prompt_tokens": encoded.prompt_tokens,
                              "provenance": row.get("provenance", {})}
                    data = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    output.write(data)
                    output_digest.update(data)
                    counts["written"] += 1
                    tokens["prompt"] += encoded.prompt_tokens
                    tokens["target"] += encoded.target_tokens
                    tokens["total"] += len(encoded.input_ids)
                    tokens["max_sequence"] = max(tokens["max_sequence"], len(encoded.input_ids))
                except Exception as exc:
                    failure = exc
                    counts["failed"] += 1
                    manifest["failed_line"] = counts["read"]
        manifest["source"]["sha256"] = source_digest.hexdigest()
        if failure is not None:
            raise failure
        if not counts["written"]:
            raise ValueError("training input contains no examples")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["output"]["sha256"] = output_digest.hexdigest()
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_prepared(path: str | Path, expected_fingerprint: ConfigFingerprint) -> tuple[list[dict], dict]:
    """Verify a completed preparation once and return only Trainer features.

    ``path`` is the preparation directory. The input source need not remain at
    its original location: its digest/provenance are recorded offline. Prepared
    rows are not retokenized or rehashed individually.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != _FORMAT or manifest.get("status") != "completed":
        raise ValueError("training data must have a completed preparation manifest")
    if manifest["fingerprint"] != asdict(expected_fingerprint):
        raise ValueError("training data fingerprint differs from the requested template")
    digest, features = sha256(), []
    with (path / "examples.jsonl").open("rb") as source:
        for raw in source:
            digest.update(raw)
            row = json.loads(raw)
            features.append({key: row[key] for key in _FEATURES})
    if digest.hexdigest() != manifest["output"]["sha256"]:
        raise ValueError("prepared examples file SHA256 does not match its manifest")
    if len(features) != manifest["counts"]["written"]:
        raise ValueError("prepared examples count does not match its manifest")
    return features, manifest
