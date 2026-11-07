"""Index paid native and auxiliary calls from the SQLite execution ledger."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import glob
import hashlib
import json
from pathlib import Path
import sqlite3

from controller.admission import quote_realization
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


def measured_totals(rows, fields):
    totals = {}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        totals[field] = {"observed": sum(values) if values else None,
                         "observed_calls": len(values), "unknown_calls": len(rows)-len(values)}
    return totals


def index(patterns):
    paths, unmatched = set(), []
    for pattern in patterns:
        found = [Path(path).resolve() for path in glob.glob(pattern, recursive=True) if Path(path).is_file()]
        paths.update(found)
        if not found:
            unmatched.append(pattern)
    sources, inventory, identities, pre_submission = [], [], {}, []
    incomplete, auxiliary = {}, {}

    def receipt_record(receipt, inputs, outputs, kind, location, quote=None):
        value = asdict(receipt) if is_dataclass(receipt) else receipt
        if not isinstance(value, dict):
            inventory.append({**location, "kind": kind, "reason": "native_receipt_missing_or_unsupported"})
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
        work = {name: value.get(name) for name in ("actual_prefill_tokens", "actual_host_loaded_tokens",
                                                   "actual_device_hit_tokens")}
        if quote is not None:
            work["quote_work"] = quote_realization(quote, receipt)
        if outputs is None or value.get("generation_complete") is not True:
            incomplete[key] = {**location, **work, "kind": kind, "native_identity": key,
                "input_tokens": len(inputs), "output_tokens": None if outputs is None else len(outputs),
                "receipt": value}
            return
        signature = kind, actual_hash, token_hash(outputs), len(inputs), len(outputs)
        entry = identities.setdefault(key, {"engine_restart_id": key[0], "native_attempt_id": key[1],
                                            "variants": {}, "locations": []})
        entry["locations"].append(location)
        entry["variants"][signature] = {"kind": kind, "input_hash": actual_hash,
            "output_hash": signature[2], "input_tokens": len(inputs), "output_tokens": len(outputs),
            "receipt_rid": value.get("rid"), "finish_reason": value.get("finish_reason"), **work}

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
            user_version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "auxiliary_calls" in tables:
                for (payload,) in connection.execute("SELECT payload FROM auxiliary_calls ORDER BY rowid"):
                    receipt = json.loads(payload)
                    key = receipt["call_id"]
                    if key not in auxiliary or auxiliary[key]["status"] == "SUBMITTED":
                        auxiliary[key] = receipt
                provenance["rows_scanned"]["auxiliary_calls"] = connection.execute(
                    "SELECT COUNT(*) FROM auxiliary_calls").fetchone()[0]
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
                        if record.final_step_id is not None:
                            continue  # The upstream final call is already a durable execution step.
                        if record.terminal_receipt is not None and record.plan is not None:
                            receipt_record(record.terminal_receipt, record.plan.candidate.input_ids,
                                           None, "ANSWER", location, record.answer_quote)
                            continue
                        if ((record.failed_from_stage or raw["stage"]) in {"STARTED", "PLANNED", "PREPARING", "PREPARED"}
                                and record.result is None):
                            pre_submission.append({**location, "kind": "ANSWER",
                                "native_model_input_tokens": 0, "native_model_output_tokens": 0})
                            continue
                        if record.result is None or record.plan is None:
                            inventory.append({**location, "kind": "ANSWER", "reason": "actual_result_or_input_plan_missing",
                                              "recorded_error": record.error})
                            continue
                        receipt_record(record.result.receipt, record.plan.candidate.input_ids,
                                       record.result.output_ids, "ANSWER", location, record.answer_quote)
                    else:
                        location["step_id"] = record.step_id
                        if (user_version == 2 and raw["stage"] in {"PLANNED", "PREPARED"}
                                and record.receipt is None and record.result is None and not record.output_ids):
                            pre_submission.append({**location, "kind": record.task.kind,
                                "native_model_input_tokens": 0, "native_model_output_tokens": 0})
                            continue
                        if record.receipt is None and "cpu_reduce_ms" in record.metrics:
                            inventory.append({**location, "kind": "CPU", "elapsed_ms": record.metrics["cpu_reduce_ms"],
                                              "reason": "cpu_step_has_no_native_receipt_excluded"})
                            continue
                        if record.task.kind == "GAP":
                            continue  # Provider costs are recorded independently of model generation.
                        outputs = None if raw["stage"] == "FAILED" and record.result is None else record.output_ids
                        receipt_record(record.receipt, record.input_ids, outputs, record.task.kind,
                                       location, record.metrics.get("native_quote_work"))
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
    partial = [row for key, row in incomplete.items() if key not in identities]
    provider = {key: row for key, row in auxiliary.items() if row["kind"] == "provider_usage"}
    aux = [row for key, row in auxiliary.items() if key not in provider]
    standalone = [row for row in aux if row.get("group_id") not in provider]
    observed_calls = [*accepted, *partial]
    native_tokens = measured_totals(observed_calls, ("input_tokens", "output_tokens"))
    provider_tokens = measured_totals([*provider.values(), *standalone], ("input_tokens", "output_tokens"))
    known_tokens = sum(value["observed"] or 0 for values in (native_tokens, provider_tokens) for value in values.values())
    known_tokens += sum(row[name] for row in aux if row.get("group_id") in provider
                        for name in ("input_tokens", "output_tokens")
                        if provider[row["group_id"]][name] is None and row[name] is not None)
    coverage_reasons = {row["reason"] for row in inventory if row.get("kind") != "CPU"}
    if conflicts:
        coverage_reasons.add("conflicting_native_receipts")
    if not sources or unmatched:
        coverage_reasons.add("execution_ledger_missing")
    if standalone:
        coverage_reasons.add("provider_phase_total_unrecorded")
    if any(value["unknown_calls"] for values in (native_tokens, provider_tokens) for value in values.values()):
        coverage_reasons.add("model_token_usage_unknown")
    quote_checks = [row["quote_work"] for row in observed_calls if row.get("quote_work") is not None]
    return {"format": "contextkv_sqlite_native_usage_index_v1", "patterns": patterns, "unmatched_patterns": unmatched,
        "sources": sources, "unique_calls": accepted, "conflicts_excluded": conflicts, "unmeasured_inventory": inventory,
        "partial_native_calls": partial,
        "online_token_ledger": {"total_tokens": known_tokens if not coverage_reasons else None,
            "known_tokens": known_tokens, "reasons": sorted(coverage_reasons),
            "native": native_tokens, "provider": provider_tokens,
            "provider_compute_ms": measured_totals(list(provider.values()), ("compute_ms",))["compute_ms"]
                if all("compute_ms" in row for row in provider.values()) else None,
            "provider_phases": len(provider)},
        "native_work": measured_totals(observed_calls, ("actual_prefill_tokens",
            "actual_host_loaded_tokens", "actual_device_hit_tokens", "output_tokens")),
        "native_quote_checks": measured_totals(quote_checks, ("quote_age_ms", "prefill_error_tokens",
                                                              "host_load_error_tokens")),
        "auxiliary": {"calls": aux, "unique_calls": len(aux), "status_counts": dict(Counter(
            row["status"] for row in aux)), "totals": measured_totals(aux,
            ("input_tokens", "output_tokens", "elapsed_ms")), "provider_phases": list(provider.values())},
        "pre_submission_steps": pre_submission,
        "pre_submission_rule": "SessionStore v2 durably saves SUBMITTED before native generation. PLANNED/PREPARED with no receipt, result or output establish zero native model tokens only; control, prepare and physical costs are separate.",
        "totals": totals(accepted), "by_kind": {kind: totals(row for row in accepted if row["kind"] == kind) for kind in kinds},
        "CPU": {"logical_rows_excluded": sum(row.get("kind") == "CPU" for row in inventory),
                "reduce_ms": sum(row["elapsed_ms"] for row in inventory if row.get("kind") == "CPU"),
                "native_token_usage": "not applicable; CPU steps have no native receipt"},
        "inventory_reasons": dict(Counter(row["reason"] for row in inventory)),
        "identity_rule": "engine_restart_id + actual receipt.attempt_id; consistent aliases count once; kind/count/input/output hash disagreement excludes whole identity",
        "qualification": "actual receipt.generation_complete=true and token_hash(stored actual input_ids)==receipt.input_hash; logical stage alone never establishes completion",
        "scope": "native receipts from attempts and execution_steps; auxiliary receipts from auxiliary_calls",
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
