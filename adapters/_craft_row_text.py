"""CRAFT row-text pure functions, minimally adapted to plain dict inputs.

Source: coral-lab-asu/CRAFT@7c4be45bb292f4e050c4237b4141ccc9e91ebe30, src/craft_tabqa/core/text.py.
Only the Table type annotation/import was replaced; whitespace normalization is
for lexical ranking, never written back into original evidence or source spans.
License: MIT; full copyright and permission in docs/third_party/CRAFT_NOTICE.md.
"""
from typing import Dict, List, Sequence, Tuple

def _clean(text) -> str:
    """Collapse runs of whitespace and coerce to ``str`` (``None`` -> "")."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def build_row_texts(tables: Sequence[Dict]) -> Tuple[List[str], List[Dict]]:
    """Build one text string per table row, for dense row encoding.

    Each row is rendered as ``"{title}  {header}: {cell}  ..."`` so the encoder
    sees the table's subject plus the row's own values.

    Returns ``(row_texts, row_meta)`` where ``row_meta[i]`` is
    ``{"table_id", "row_idx", "text"}`` mapping row ``i`` back to its table.
    """
    row_texts: List[str] = []
    row_meta: List[Dict] = []

    for table in tables:
        title = _clean(table.get("title"))
        headers = [_clean(h) for h in table.get("headers", [])]
        table_id = table["table_id"]

        for row_idx, cells in enumerate(table.get("rows", [])):
            values = [_clean(c) for c in cells]
            if headers:
                pairs = "  ".join(
                    f"{h}: {v}" for h, v in zip(headers, values) if v
                )
            else:
                pairs = "  ".join(v for v in values if v)

            text = _clean(f"{title}  {pairs}") if pairs else title
            row_texts.append(text)
            row_meta.append({"table_id": table_id, "row_idx": row_idx, "text": text})

    return row_texts, row_meta




