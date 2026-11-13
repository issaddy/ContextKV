"""MT-RAG source locations and benchmark row loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_COMMIT = "53fef6f5917c7e10c436f697e69a955dbc795d81"
DATASET_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/mtrag-human/generation_tasks/RAG.jsonl"
MTRAG_UN_URL = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/mtragun-human/generation_tasks/reference.jsonl"


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load original benchmark rows."""
    rows = []
    seen = set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("conversation_id", "task_id", "turn", "Collection", "input", "contexts"):
            if key not in row:
                raise ValueError(f"line {line_no}: missing {key}")
        identity = (str(row["conversation_id"]), int(row["turn"]))
        if identity in seen:
            raise ValueError(f"duplicate conversation/turn: {identity}")
        seen.add(identity)
        rows.append(row)
    return rows
