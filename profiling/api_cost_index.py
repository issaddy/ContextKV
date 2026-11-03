"""Index actual saved OpenAI-style response usage without making API calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import json
from pathlib import Path


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
RESPONSE_SLOTS = {"raw_response", "judge_response", "api_response", "response"}


def pointer_part(value):
    return str(value).replace("~", "~0").replace("/", "~1")


def token_count(value):
    return type(value) is int and value >= 0


def numeric_usage(value, prefix=""):
    found = {}
    for name, item in value.items():
        key = prefix+"/"+pointer_part(name)
        if isinstance(item, dict):
            found.update(numeric_usage(item, key))
        elif token_count(item):
            found[key] = item
    return found


def totals(responses):
    values = list(responses)
    return {"unique_measured_responses": len(values),
        **{name: {"measured_response_count": sum(row["usage"].get(name) is not None for row in values),
            "observed_sum": (sum(row["usage"][name] for row in values if row["usage"].get(name) is not None)
                             if any(row["usage"].get(name) is not None for row in values) else None),
            "complete_unique_response_sum": sum(row["usage"][name] for row in values)
                if values and all(row["usage"].get(name) is not None for row in values) else None}
            for name in TOKEN_FIELDS}}


def index(inputs):
    paths, unmatched = defaultdict(set), []
    for specification in inputs:
        label, pattern = specification.split("=", 1)
        matches = [Path(path).resolve() for path in sorted(glob.glob(pattern, recursive=True)) if Path(path).is_file()]
        if not matches:
            unmatched.append({"label": label, "pattern": pattern})
        for path in matches:
            paths[path].add(label)
    sources, entries, inventory = [], {}, []

    def visit(value, location):
        if isinstance(value, dict):
            usage = value.get("usage")
            marker = ("choices" in value or isinstance(value.get("object"), str)
                      and ("completion" in value["object"] or "response" in value["object"])
                      or "id" in value and "model" in value)
            native = "contextkv_receipt" in value or isinstance(value.get("meta_info"), dict) and "contextkv_receipt" in value["meta_info"]
            api_shape = marker and isinstance(usage, dict) and all(token_count(usage.get(name)) for name in TOKEN_FIELDS[:2])
            if native and not api_shape:
                inventory.append({**location, "kind": "native_receipt_response_not_api_usage"})
            elif marker or isinstance(usage, dict):
                reasons = []
                if not isinstance(value.get("id"), str) or not value["id"]:
                    reasons.append("actual_response_id_missing")
                if not marker:
                    reasons.append("usage_wrapper_without_actual_response_shape")
                if not isinstance(usage, dict):
                    reasons.append("actual_usage_missing")
                elif not all(token_count(usage.get(name)) for name in TOKEN_FIELDS[:2]):
                    reasons.append("actual_prompt_or_completion_token_counter_missing_or_invalid")
                if value.get("error"):
                    reasons.append("response_contains_error")
                if reasons:
                    inventory.append({**location, "kind": "unmeasured_response_candidate", "reasons": reasons,
                        "response_id": value.get("id"), "served_model": value.get("model"),
                        "created": value.get("created"), "observed_usage": usage})
                else:
                    identity = (value["id"], value.get("model"), value.get("created"))
                    key = json.dumps(identity, sort_keys=True, ensure_ascii=False)
                    entry = entries.setdefault(key, {"response_id": identity[0], "served_model": identity[1],
                        "created": identity[2], "locations": [], "usage_variants": {}, "counter_values": defaultdict(set),
                        "labels": set()})
                    entry["locations"].append(location)
                    entry["labels"].update(sources[location["source_index"]]["labels"])
                    signature = json.dumps(usage, sort_keys=True, ensure_ascii=False)
                    entry["usage_variants"][signature] = usage
                    for name, number in numeric_usage(usage).items():
                        entry["counter_values"][name].add(number)
            elif value.get("error"):
                inventory.append({**location, "kind": "saved_error_without_measured_api_response",
                    "error_type": type(value["error"]).__name__, "error": str(value["error"])[:500]})
            for name, item in value.items():
                child = {**location, "json_pointer": location["json_pointer"]+"/"+pointer_part(name)}
                if name in RESPONSE_SLOTS and not isinstance(item, (dict, list)):
                    inventory.append({**child, "kind": "response_slot_without_json_response",
                                      "value_type": type(item).__name__})
                if isinstance(item, (dict, list)):
                    visit(item, child)
        elif isinstance(value, list):
            for ordinal, item in enumerate(value):
                visit(item, {**location, "json_pointer": location["json_pointer"]+"/"+str(ordinal)})

    for path, labels in sorted(paths.items()):
        source_index = len(sources)
        record = {"path": str(path), "labels": sorted(labels), "json_documents": 0,
                  "invalid_json_documents": 0, "bytes_indexed": 0}
        sources.append(record)
        sha = hashlib.sha256()
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            with path.open("rb") as stream:
                for line, raw in enumerate(stream, 1):
                    sha.update(raw)
                    record["bytes_indexed"] += len(raw)
                    if not raw.strip():
                        continue
                    location = {"source_index": source_index, "line": line, "json_pointer": ""}
                    try:
                        value = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        record["invalid_json_documents"] += 1
                        inventory.append({**location, "kind": "unparsed_source_document", "error": str(error)})
                        continue
                    record["json_documents"] += 1
                    visit(value, location)
        else:
            raw = path.read_bytes()
            sha.update(raw)
            record["bytes_indexed"] = len(raw)
            location = {"source_index": source_index, "line": None, "json_pointer": ""}
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                record["invalid_json_documents"] += 1
                inventory.append({**location, "kind": "unparsed_source_document", "error": str(error)})
            else:
                record["json_documents"] += 1
                visit(value, location)
        record["sha256"] = sha.hexdigest()
        record["hash_scope"] = "exact bytes streamed/read to the indexed EOF snapshot; bytes_indexed fixes append-only snapshot coverage"

    responses, conflicts = [], []
    for entry in entries.values():
        counters = entry.pop("counter_values")
        collision = {name: sorted(values) for name, values in counters.items() if len(values) > 1}
        entry["phase_labels"] = sorted(entry.pop("labels"))
        entry["usage_variants"] = list(entry["usage_variants"].values())
        entry["usage"] = {name: next(iter(counters["/"+name])) if "/"+name in counters else None for name in TOKEN_FIELDS}
        entry["alias_occurrences"] = len(entry["locations"])
        entry["status"] = "excluded_conflicting_actual_counters" if collision else "unique_measured_api_response"
        if collision:
            entry["conflicting_counters"] = collision
            conflicts.append(entry)
        else:
            responses.append(entry)
    by_model, by_label = defaultdict(list), defaultdict(list)
    for entry in responses:
        by_model[json.dumps(entry["served_model"], sort_keys=True)].append(entry)
        for label in entry["phase_labels"]:
            by_label[label].append(entry)
    coverage = Counter(item["source_index"] for entry in (*responses, *conflicts) for item in entry["locations"])
    unmeasured = Counter(item["source_index"] for item in inventory)
    for ordinal, record in enumerate(sources):
        record.update(api_response_occurrences=coverage[ordinal], unmeasured_inventory_items=unmeasured[ordinal])
    return {"format": "contextkv_saved_api_usage_index_v1", "inputs": inputs, "sources": sources,
        "unmatched_patterns": unmatched, "unique_responses": responses, "conflicts_excluded": conflicts,
        "unmeasured_inventory": inventory, "totals": totals(responses),
        "by_served_model": [{"served_model": json.loads(model), "totals": totals(values)} for model, values in sorted(by_model.items())],
        "by_phase_label_nonexclusive": {label: totals(values) for label, values in sorted(by_label.items())},
        "identity_rule": "actual response id + served model + created; identical/consistent aliases count once; contradictory observed usage counters exclude the whole identity",
        "location_rule": "source_index resolves file path and SHA; line is 1-based JSONL line; json_pointer is RFC6901 within that JSON document",
        "usage_rule": "prompt/completion are actual nonnegative integer counters; missing total remains null; total is never inferred from prompt+completion",
        "scope": "existing saved OpenAI-style response usage only; unmeasured inventory does not imply a paid call or zero cost",
        "category_rule": "API usage, native completion-receipt tokens and training-forward tokens are separate potentially overlapping ledgers; do not add them",
        "phase_rule": "labels express nonexclusive source membership; do not sum phase-label totals",
        "actual_paid_call_count": None, "coverage_claim": "only explicitly supplied source snapshots; missing/error/unparsed records remain unmeasured"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="LABEL=explicit file/glob; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = index(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output_written": True, "files": len(report["sources"]),
        "unique_measured": len(report["unique_responses"]), "conflicts_excluded": len(report["conflicts_excluded"]),
        "unmeasured_items": len(report["unmeasured_inventory"]), "unmatched_patterns": len(report["unmatched_patterns"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
