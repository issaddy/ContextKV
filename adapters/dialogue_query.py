"""Literal dialogue view for retrieval; never becomes a source or generated task."""

from controller.schema import Message, from_history_messages, history_messages

QUERY_VIEW_VERSION = "literal-dialogue-query-v1"
QUERY_POOLING = "CLS then FP32 L2 normalization"


def query_view(query, history, tokenizer, *, limit=8192, mode="dialogue"):
    if not isinstance(query, str) or not query:
        raise ValueError("current question must be nonempty text")
    if mode not in ("current", "dialogue"):
        raise ValueError("unknown query view")
    records = from_history_messages(tuple(Message(
        "assistant" if message["speaker"] == "agent" else message["speaker"], message["text"])
        for message in history))

    def count(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    size = count(query)
    if size > limit:
        raise ValueError("current question exceeds encoder budget; not truncated")
    count_history = len(records)
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
    blocks = ["\n".join(f"Turn {row.turn_id} {message.role}: {message.content}"
                        for message in history_messages((row,))) for row in reversed(records)]
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


def load_query_adapter(path, document_encoder):
    """Bind an optional query model to its immutable base document space."""
    import json
    from pathlib import Path

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
    if metadata.get("query_pooling") != QUERY_POOLING:
        raise ValueError("query adapter requires its declared FP32 query pooling")
    if metadata.get("query_view_version", QUERY_VIEW_VERSION) != QUERY_VIEW_VERSION:
        raise ValueError("unsupported query adapter input recipe")
    config = json.loads((root / "adapter" / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA" or config.get("task_type") != "FEATURE_EXTRACTION":
        raise ValueError("query adapter requires a feature-extraction LoRA")
    return metadata
