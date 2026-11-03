"""Read completed native receipt usage from SQLite attempts and step rows."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import glob
import hashlib
import json
from pathlib import Path
import sqlite3

from controller.context_schema import StepRecord
from controller.session_store import Attempt, load_record
from controller.templates import token_hash


def file_identity(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            sha.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def totals(entries):
    values = list(entries)
    return {"unique_completed_native_calls": len(values),
        "input_tokens": sum(row["input_tokens"] for row in values) if values else None,
        "output_tokens": sum(row["output_tokens"] for row in values) if values else None,
        "total_tokens": sum(row["input_tokens"]+row["output_tokens"] for row in values) if values else None}


def index(patterns):
    paths, unmatched = set(), []
    for pattern in patterns:
        found = [Path(path).resolve() for path in glob.glob(pattern, recursive=True) if Path(path).is_file()]
        paths.update(found)
        if not found:
            unmatched.append(pattern)
    sources, inventory, identities = [], [], {}

    def receipt_record(receipt, inputs, outputs, kind, location):
        value = asdict(receipt) if is_dataclass(receipt) else receipt
        if not isinstance(value, dict):
            inventory.append({**location, "kind": kind, "reason": "native_receipt_missing_or_unsupported"})
            return
        if value.get("generation_complete") is not True:
            inventory.append({**location, "kind": kind, "reason": "native_receipt_not_generation_complete",
                              "observed_receipt": value})
            return
        if not value.get("engine_restart_id") or not value.get("attempt_id"):
            inventory.append({**location, "kind": kind, "reason": "actual_native_identity_missing", "observed_receipt": value})
            return
        actual_hash = token_hash(inputs)
        if actual_hash != value.get("input_hash"):
            inventory.append({**location, "kind": kind, "reason": "actual_input_hash_mismatch",
                "receipt_input_hash": value.get("input_hash"), "actual_input_hash": actual_hash,
                "native_identity": [value["engine_restart_id"], value["attempt_id"]]})
            key = value["engine_restart_id"], value["attempt_id"]
            entry = identities.setdefault(key, {"engine_restart_id": key[0], "native_attempt_id": key[1],
                                                "variants": {}, "locations": []})
            entry["locations"].append(location)
            entry["input_hash_validation_failed"] = True
            return
        key = value["engine_restart_id"], value["attempt_id"]
        signature = kind, actual_hash, token_hash(outputs), len(inputs), len(outputs)
        entry = identities.setdefault(key, {"engine_restart_id": key[0], "native_attempt_id": key[1],
                                            "variants": {}, "locations": []})
        entry["locations"].append(location)
        entry["variants"][signature] = {"kind": kind, "input_hash": actual_hash,
            "output_hash": signature[2], "input_tokens": len(inputs), "output_tokens": len(outputs),
            "receipt_rid": value.get("rid"), "finish_reason": value.get("finish_reason")}

    for path in sorted(paths):
        source_index = len(sources)
        wal = Path(str(path)+"-wal")
        provenance = {"database": file_identity(path), "wal": file_identity(wal) if wal.exists() else None,
            "rows_scanned": {}, "read_protocol": "SQLite read-only URI and one read transaction; base/WAL file hashes identify observed source files"}
        sources.append(provenance)
        connection = None
        try:
            connection = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, contract in (("attempts", Attempt), ("execution_steps", StepRecord)):
                provenance["rows_scanned"][table] = 0
                if table not in tables:
                    inventory.append({"source_index": source_index, "table": table, "reason": "source_table_missing"})
                    continue
                for raw in connection.execute(f"SELECT rowid,stage,payload FROM {table} ORDER BY rowid"):
                    provenance["rows_scanned"][table] += 1
                    location = {"source_index": source_index, "table": table, "rowid": raw["rowid"], "state": raw["stage"]}
                    try:
                        record = load_record(connection, contract, raw["payload"])
                    except Exception as error:
                        inventory.append({**location, "reason": "record_decode_or_schema_failure",
                                          "error_type": type(error).__name__, "error": str(error)})
                        continue
                    location["logical_attempt_id"] = record.attempt_id
                    if table == "attempts":
                        location.update(session_id=record.session_id, request_id=record.request_id)
                        if record.result is None or record.plan is None:
                            inventory.append({**location, "kind": "ANSWER", "reason": "actual_result_or_input_plan_missing",
                                              "recorded_error": record.error})
                            continue
                        receipt_record(record.result.receipt, record.plan.candidate.input_ids,
                                       record.result.output_ids, "ANSWER", location)
                    else:
                        location["step_id"] = record.step_id
                        if record.receipt is None and "cpu_reduce_ms" in record.metrics:
                            inventory.append({**location, "kind": "CPU", "reason": "cpu_step_has_no_native_receipt_excluded"})
                            continue
                        receipt_record(record.receipt, record.input_ids, record.output_ids, record.task.kind, location)
        except sqlite3.DatabaseError as error:
            inventory.append({"source_index": source_index, "reason": "sqlite_open_or_schema_read_failure", "error": str(error)})
        finally:
            if connection is not None:
                connection.close()
    accepted, conflicts = [], []
    for entry in identities.values():
        variants = list(entry["variants"].values())
        entry["alias_occurrences"] = len(entry["locations"])
        if len(variants) != 1 or entry.get("input_hash_validation_failed", False):
            entry.update(status="excluded_conflicting_native_identity", variants=variants)
            conflicts.append(entry)
        else:
            entry.pop("variants")
            entry.update(**variants[0], status="unique_completed_hash_matched_native_receipt")
            accepted.append(entry)
    kinds = sorted({"ANSWER", "READ", "REDUCE"} | {row["kind"] for row in accepted})
    return {"format": "contextkv_sqlite_native_usage_index_v1", "patterns": patterns, "unmatched_patterns": unmatched,
        "sources": sources, "unique_calls": accepted, "conflicts_excluded": conflicts, "unmeasured_inventory": inventory,
        "totals": totals(accepted), "by_kind": {kind: totals(row for row in accepted if row["kind"] == kind) for kind in kinds},
        "CPU": {"logical_rows_excluded": sum(row.get("kind") == "CPU" for row in inventory),
                "native_token_usage": "not applicable; CPU steps have no native receipt"},
        "inventory_reasons": dict(Counter(row["reason"] for row in inventory)),
        "identity_rule": "engine_restart_id + actual receipt.attempt_id; consistent aliases count once; kind/count/input/output hash disagreement excludes whole identity",
        "qualification": "actual receipt.generation_complete=true and token_hash(stored actual input_ids)==receipt.input_hash; logical stage alone never establishes completion",
        "scope": "stream actual attempts and execution_steps only; ReadingOutcome.steps are never traversed as additional calls",
        "category_rule": "native receipt tokens, saved API usage and training-forward tokens remain separate potentially overlapping categories; no physical integral is added",
        "actual_unobserved_call_count": None, "missing_values": "missing receipts/decode failures preserve inventory, not inferred calls or fabricated token zeros"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", action="append", required=True, help="Explicit SQLite file/glob; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = index(args.db)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output_written": True, "databases": len(report["sources"]),
        "unique_completed": len(report["unique_calls"]), "conflicts_excluded": len(report["conflicts_excluded"]),
        "unmeasured_items": len(report["unmeasured_inventory"]), "unmatched_patterns": len(report["unmatched_patterns"])}))


if __name__ == "__main__":
    main()
