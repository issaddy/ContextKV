"""The small records retained after benchmark scoring."""
from pathlib import Path
import json
import shutil
import sqlite3

from bench.study_costs import unsubmitted_cost


METRIC_FIELDS = (
    "method", "dataset", "split", "protocol", "task_id", "conversation_id", "turn", "status",
    "metrics", "cost", "response", "measurement_kind", "history_origin", "request_window",
    "record_number", "generation_dir", "measurement_segment_id", "workload_arrival",
    "request_id", "attempt_id", "error_type", "error", "blocked_task_ids", "blocked_tasks",
    "blocked_by_task_id", "blocked_reason", "cleanup_error", "cleanup_recovery", "failure_window_scope",
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


def execution_metric_rows(directory, rows, *, failures=None, dispatch=None):
    """One compact record per delivered, failed or unsubmitted blocked task.

    Keep failed outputs out of scoring inputs while retaining their identities
    and paid work for queue totals and paired evaluation. A blocked task has no
    response latency; zero denotes only work that was never submitted.
    """
    directory = Path(directory)
    manifest_path = directory/"generation_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    defaults = {name: manifest[name] for name in ("dataset", "split", "protocol", "measurement_kind")
                if name in manifest}
    if "method" in manifest:
        defaults["method"] = manifest["method"]["id"]
    failed_path, dispatch_path = directory/"failed_conversations.json", directory/"workload_dispatch.json"
    if failures is None:
        failures = json.loads(failed_path.read_text()) if failed_path.exists() else []
    if dispatch is None:
        dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else {}
    records = {str(row["task_id"]): metric_record(row) for row in rows}
    for failure in failures:
        identity = str(failure["task_id"])
        if identity in records:
            continue
        records[identity] = metric_record({**defaults, **failure, "status": "failed", "metrics": {}})
        blocked = {str(row["task_id"]): row for row in failure.get("blocked_tasks", ())}
        for task_id in failure.get("blocked_task_ids", ()):
            records.setdefault(str(task_id), metric_record({**defaults,
                **{name: failure[name] for name in ("method", "dataset", "split", "protocol",
                    "conversation_id", "history_origin", "measurement_segment_id") if name in failure},
                **blocked.get(str(task_id), {}), "task_id": task_id, "status": "blocked", "metrics": {},
                "blocked_by_task_id": failure["task_id"],
                "blocked_reason": failure.get("blocked_reason", "prior_actual_history_unavailable"),
                "cost": unsubmitted_cost()}))
    for task_id, cancelled in dispatch.get("cancelled", {}).items():
        records.setdefault(str(task_id), metric_record({**defaults, **cancelled, "task_id": task_id,
            "status": "blocked", "metrics": {}, "blocked_reason": cancelled["reason"],
            "cost": unsubmitted_cost()}))
    order = {str(task_id): index for index, task_id in enumerate(manifest.get("scheduled_task_ids", ()))}
    return sorted(records.values(), key=lambda row: (order.get(str(row["task_id"]), len(order)), str(row["task_id"])))


def write_task_metrics(directory, rows, *, failures=None, dispatch=None):
    records = execution_metric_rows(directory, rows, failures=failures, dispatch=dispatch)
    path = Path(directory)/"task_metrics.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n" for row in records))
    return records


def pending_state_cleanup(state_directory):
    """Inspect the closed database without reopening or changing its ledgers."""
    database = Path(state_directory)/"sessions.sqlite3"
    if not database.exists():
        return False
    connection = sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "retire_outbox" in tables and connection.execute(
                "SELECT 1 FROM retire_outbox WHERE acked=0 LIMIT 1").fetchone():
            return True
        return bool("prefix_owners" in tables and connection.execute(
            "SELECT 1 FROM prefix_owners WHERE stage IN ('RETAINING','RETAINED','RELEASING') "
            "AND json_extract(payload, '$.policy')='joint_demand' LIMIT 1").fetchone())
    finally:
        connection.close()


def retain_metrics(directory, rows):
    """Replace completed scoring inputs with the final per-task metrics."""
    directory = Path(directory)
    execution = write_task_metrics(directory, rows)
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
    if (state.is_dir() and not any(row.get("cleanup_error") for row in execution)
            and not pending_state_cleanup(state)):
        shutil.rmtree(state)
