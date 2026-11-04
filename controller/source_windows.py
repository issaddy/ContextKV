"""One source-only window granularity shared by ingestion and READ planning."""
from __future__ import annotations

import re

SOURCE_WINDOW_TOKENS = 512
SOURCE_WINDOW_LINES = 8


def source_window_spans(text, tokenizer, *, tokens=SOURCE_WINDOW_TOKENS):
    """Partition literal text, preserving rows and preferring sentence boundaries.

    Only the original text and reader tokenizer determine boundaries. A source
    row longer than the window is split; adjacent ordinary rows stay together.
    """
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
    windows, first, last, lines = [], 0, 0, 0
    for start, end in rows:
        nonempty = bool(text[start:end].strip())
        if last > first and (size(first, end) > tokens or lines + nonempty > SOURCE_WINDOW_LINES):
            windows.append((first, last))
            first, lines = start, 0
        last = end
        lines += nonempty
    if last > first:
        windows.append((first, last))
    return tuple(windows)
