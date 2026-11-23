"""One source-only window granularity shared by ingestion and READ planning."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import re

SOURCE_WINDOW_TOKENS = 2048


def numbered_text(text):
    words = tuple(re.finditer(r"[\u3400-\u9fff]|\w+|[^\w\s]", text))
    return " ".join(f"{i}:{word.group()}" for i, word in enumerate(words)), tuple(word.span() for word in words)


def source_window_spans(text, tokenizer, *, tokens=SOURCE_WINDOW_TOKENS):
    """Partition literal text, preserving rows and preferring sentence boundaries."""
    if not text:
        return ()
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False,
                        return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(encoded["input_ids"]) <= tokens:
        return ((0, len(text)),)
    starts = [left for left, _ in offsets]
    ends = [right for _, right in offsets]

    def size(start, end):
        return len(tokenizer.encode(text[start:end], add_special_tokens=False))

    def exceeds(start, end):
        approx = bisect_left(starts, end) - bisect_right(ends, start)
        if approx <= tokens - 2:
            return False
        if approx >= tokens + 3:
            return True
        return size(start, end) > tokens

    rows = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        while exceeds(start, end):
            low, high = start, end
            target_tok = bisect_right(ends, start) + tokens
            if target_tok < len(starts):
                guess = starts[target_tok]
                low = max(start, min(end, guess - 16))
                high = min(end, max(low, ends[min(len(ends) - 1, target_tok + 2)] + 16))
                if low > start and size(start, low) > tokens:
                    low = start
                if high < end and size(start, high) <= tokens:
                    high = end
            while low < high:
                middle = (low + high + 1) // 2
                if not exceeds(start, middle):
                    low = middle
                else:
                    high = middle - 1
            if low == start:
                raise ValueError("source window cannot contain one original character")
            boundaries = [start + match.end() for match in re.finditer(
                r"[.!?。！？;；](?:\s+|$)", text[start:low])
                if match.end() >= (low - start) // 2]
            cut = boundaries[-1] if boundaries else low
            rows.append((start, cut))
            start = cut
        rows.append((start, end))
        start = end
    windows, first, last = [], 0, 0
    for start, end in rows:
        if last > first and exceeds(first, end):
            windows.append((first, last))
            first = start
        last = end
    if last > first:
        windows.append((first, last))
    return tuple(windows)
