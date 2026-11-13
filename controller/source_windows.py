"""One source-only window granularity shared by ingestion and READ planning."""
from __future__ import annotations

import re

SOURCE_WINDOW_TOKENS = 2048


def numbered_text(text):
    words = tuple(re.finditer(r"[\u3400-\u9fff]|\w+|[^\w\s]", text))
    return " ".join(f"{i}:{word.group()}" for i, word in enumerate(words)), tuple(word.span() for word in words)


def numbered_lines(text, tokenizer):
    shown, offsets, cursor = [], [], 0
    for index, line in enumerate(text.splitlines(keepends=True)):
        offsets.append((cursor, cursor+len(line)))
        cursor += len(line)
        shown.append(f"[{index}; {len(tokenizer.encode(line, add_special_tokens=False))} tokens] {line}" +
                     ("" if line.endswith("\n") else "\n"))
    return "".join(shown), tuple(offsets)


def source_window_spans(text, tokenizer, *, tokens=SOURCE_WINDOW_TOKENS):
    """Partition literal text, preserving rows and preferring sentence boundaries.

    Only the original text and reader tokenizer determine boundaries. A source
    row longer than the window is split; adjacent ordinary rows stay together.
    """
    if len(tokenizer.encode(text, add_special_tokens=False)) <= tokens:
        return ((0, len(text)),) if text else ()

    def size(start, end):
        return len(tokenizer.encode(text[start:end], add_special_tokens=False))

    rows = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        while size(start, end) > tokens:
            low, high = start, end
            while low < high:
                middle = (low + high + 1) // 2
                if size(start, middle) <= tokens:
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
        if last > first and size(first, end) > tokens:
            windows.append((first, last))
            first = start
        last = end
    if last > first:
        windows.append((first, last))
    return tuple(windows)
