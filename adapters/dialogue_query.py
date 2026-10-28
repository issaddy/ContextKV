"""Literal dialogue view for retrieval; never becomes a source or generated task."""

QUERY_VIEW_VERSION = "literal-dialogue-query-v1"
QUERY_POOLING = "CLS then FP32 L2 normalization"


def query_view(query, history, tokenizer, *, limit=8192, mode="dialogue"):
    if not isinstance(query, str) or not query:
        raise ValueError("current question must be nonempty text")
    if mode not in ("current", "dialogue"):
        raise ValueError("unknown query view")
    if len(history) % 2 or any(
        m["speaker"] not in (("user",) if i % 2 == 0 else ("agent", "assistant"))
        or not isinstance(m["text"], str)
        for i, m in enumerate(history)
    ):
        raise ValueError("history must contain completed user/assistant exchanges")

    def count(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    size = count(query)
    if size > limit:
        raise ValueError("current question exceeds encoder budget; not truncated")
    count_history = len(history) // 2
    if mode == "current" or not history:
        return query, {
            "history_turns": [],
            "omitted_history_turns": list(range(1, count_history + 1)),
            "input_tokens": size,
        }
    prefix = (
        "Current user question:\n"
        + query
        + "\n\nPrevious conversation (most recent exchange first):\n"
    )
    numbers = list(range(count_history, 0, -1))
    blocks = [
        f"Turn {n} user: {history[2 * n - 2]['text']}\nTurn {n} assistant: {history[2 * n - 1]['text']}"
        for n in numbers
    ]
    while blocks:
        value = prefix + "\n\n".join(blocks)
        size = count(value)
        if size <= limit:
            return value, {
                "history_turns": numbers,
                "omitted_history_turns": list(range(1, count_history - len(numbers) + 1)),
                "input_tokens": size,
            }
        blocks.pop()
        numbers.pop()
    return query, {
        "history_turns": [],
        "omitted_history_turns": list(range(1, count_history + 1)),
        "input_tokens": count(query),
    }


def load_query_adapter(path, document_encoder, *, base_config_sha256):
    """Bind an optional query model to its immutable base document space."""
    import json
    from pathlib import Path
    from hashlib import sha256

    root = Path(path)
    metadata = json.loads((root / "query-adapter.json").read_text())
    if (
        metadata.get("status") != "completed"
        or metadata.get("schema_version") != 1
        or metadata.get("query_mode") not in {"current", "dialogue"}
    ):
        raise ValueError("query adapter requires a completed supported manifest")
    if metadata["document_encoder"] != document_encoder:
        raise ValueError("query adapter was trained for a different document encoder")
    if metadata.get("base_model_config_sha256") != base_config_sha256:
        raise ValueError("query adapter base model configuration differs")
    if metadata.get("query_pooling") != QUERY_POOLING:
        raise ValueError("query adapter requires its declared FP32 query pooling")
    # Schema 1 uses the literal current/dialogue recipe. Preserve the original
    # training-code hash as provenance; it is not an inference-module file hash.
    if metadata.get("query_view_version", QUERY_VIEW_VERSION) != QUERY_VIEW_VERSION:
        raise ValueError("unsupported query adapter input recipe")
    for name, digest in metadata["files"].items():
        if name not in {"adapter_config.json", "adapter_model.safetensors"}:
            raise ValueError("unexpected query adapter file")
        if sha256((root / "adapter" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("query adapter file differs from completed artifact")
    if set(metadata["files"]) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("query adapter inventory is incomplete")
    if metadata.get("adapter_sha256") != metadata["files"]["adapter_model.safetensors"]:
        raise ValueError("query adapter identity differs from its weights")
    config = json.loads((root / "adapter" / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or config.get("task_type") != "FEATURE_EXTRACTION":
        raise ValueError("query adapter requires a feature-extraction LoRA")
    return metadata
