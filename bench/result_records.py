"""The small records retained after benchmark scoring."""
from pathlib import Path
import json
import shutil


METRIC_FIELDS = (
    "method", "dataset", "split", "protocol", "task_id", "conversation_id", "turn", "status",
    "metrics", "cost", "response", "measurement_kind", "history_origin", "request_window",
    "record_number", "generation_dir", "measurement_segment_id", "workload_arrival",
)


def metric_record(row):
    result = {name: row[name] for name in METRIC_FIELDS if name in row}
    response = row.get("response")
    if response is not None:
        result["response"] = {name: response[name] for name in (
            "attempt_id", "action", "output_tokens", "truncated", "read_count", "reduce_count") if name in response}
    if row.get("scoring"):
        result["scoring"] = {name: row["scoring"][name] for name in ("status", "judge_cost")
                             if name in row["scoring"]}
    return result


def retain_metrics(directory, rows):
    """Replace completed scoring inputs with the final per-task metrics."""
    directory = Path(directory)
    with (directory/"results.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(metric_record(row), ensure_ascii=False, separators=(",", ":"))+"\n")
    for name in ("official.jsonl", "audit.jsonl", "references.jsonl"):
        (directory/name).unlink(missing_ok=True)
    scoring = directory/"scoring"
    manifest = scoring/"manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({name: value[name] for name in (
            "identity", "primary_metric", "split", "normalization", "requested",
            "generated_answers", "scored", "judge_cost", "groups") if name in value}, ensure_ascii=False)+"\n")
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
