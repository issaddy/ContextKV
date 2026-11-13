"""Retain answers, source addresses and final benchmark measurements."""
from pathlib import Path
import json
import shutil


METRIC_FIELDS = (
    "method", "dataset", "split", "protocol", "task_id", "conversation_id", "turn", "status",
    "question", "answer", "sources", "computation_sources", "metrics", "cost", "response",
    "measurement_kind", "history_origin", "generation_dir", "request_id",
    "error_type", "error",
)


def metric_record(row):
    result = {name: row[name] for name in METRIC_FIELDS if name in row}
    if row.get("scoring"):
        result["scoring"] = {name: row["scoring"][name] for name in ("status", "error", "judge_cost")
                             if name in row["scoring"]}
    return result


def write_task_metrics(directory, rows):
    records = [metric_record(row) for row in rows]
    path = Path(directory)/"results.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n" for row in records))
    return records


def retain_metrics(directory):
    """Remove temporary scorer material after final metrics have been saved."""
    directory = Path(directory)
    for name in ("official.jsonl", "audit.jsonl", "references.jsonl"):
        (directory/name).unlink(missing_ok=True)
    scoring = directory/"scoring"
    if scoring.is_dir():
        for path in scoring.iterdir():
            if path.name != "manifest.json":
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    state = directory/"state"
    if state.is_dir():
        shutil.rmtree(state)
