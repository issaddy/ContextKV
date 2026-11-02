"""Index study costs and integrate overlapping physical samples only once.

All inputs belong to one explicitly declared client-monotonic clock domain.
Per-job reported integrals remain an audit index; the study total is rebuilt
from device/PID samples, never by adding overlapping job or phase totals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
from pathlib import Path

from bench.study_resources import integrate_physical
from profiling.retention_trace import write_json


def physical_row(row):
    return isinstance(row, dict) and "timestamp" in row and (
        "devices" in row or "process_pss" in row)


def physical_series(value, pointer="root"):
    if isinstance(value, list):
        unwrapped = [row.get("sample", row) if isinstance(row, dict) else row for row in value]
        if unwrapped and all(physical_row(row) for row in unwrapped):
            yield pointer, unwrapped
        else:
            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    yield from physical_series(item, pointer+"/"+str(index))
    elif isinstance(value, dict):
        if physical_row(value):
            yield pointer, [value]
        else:
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    yield from physical_series(item, pointer+"/"+key)


def objects(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from objects(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                yield from objects(item)


def read_document(path):
    contents = path.read_bytes()
    if path.suffix in {".jsonl", ".ndjson"}:
        value = [json.loads(line) for line in contents.splitlines() if line.strip()]
    else:
        value = json.loads(contents)
    return value, hashlib.sha256(contents).hexdigest()


def merge_ranges(ranges):
    merged = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def uncovered(ranges, start, end):
    missing, cursor = [], start
    for left, right in merge_ranges(ranges):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            missing.append([cursor, min(left, end)])
        cursor = max(cursor, min(right, end))
    if cursor < end:
        missing.append([cursor, end])
    return missing


def integrate_unique(segments, start, end):
    """Use the freshest actual left sample on each overlapping interval."""
    events = defaultdict(list)
    for index, (left, right, value, source) in enumerate(segments):
        measured_at = left
        left, right = max(left, start), min(right, end)
        if right > left:
            events[left].append((True, index, value, source, measured_at))
            events[right].append((False, index, value, source, measured_at))
    active, integral, peak, overlap = {}, 0.0, None, 0.0
    intervals, choices = [], defaultdict(float)
    boundaries = sorted(events)
    for position, left in enumerate(boundaries[:-1]):
        for opening, identity, value, source, measured_at in events[left]:
            if opening:
                active[identity] = (measured_at, value, source)
            else:
                active.pop(identity, None)
        right = boundaries[position+1]
        if not active:
            continue
        measured_at, value, source = max(active.values(), key=lambda item: (item[0], item[2]))
        del measured_at
        integral += value*(right-left)
        peak = value if peak is None else max(peak, value)
        overlap += right-left if len(active) > 1 else 0
        choices[source] += right-left
        intervals.append([left, right])
    covered = merge_ranges(intervals)
    return dict(gib_seconds=integral if covered else None, peak_gib=peak,
        observed_intervals=covered, unknown_intervals=uncovered(covered, start, end),
        overlapped_source_seconds_deduplicated=overlap, selected_source_seconds=dict(choices))


def job_index(path, value):
    if not isinstance(value, dict):
        return dict(path=str(path), scope="raw observation/call file", reported_integral_added_to_total=False)
    generation = value.get("generations", {})
    fields = ("format", "source", "status", "scope", "cost_scope", "elapsed_seconds",
        "controller_cpu_seconds", "own_cpu_seconds", "child_cpu_seconds", "resource_window", "window",
        "started_wall_utc", "physical_cost", "native_cost", "cpu_scope", "prior_unobserved_intervals")
    training_names = ("training_wall_ms", "wall_ms", "wall_seconds", "total_model_input_tokens",
        "total_supervised_tokens", "model_input_tokens", "supervised_tokens", "optimizer_steps",
        "ranks", "world_size", "training_gpu_seconds", "gpu_hours", "monetary_cost", "currency")
    return dict(path=str(path), **{name: value[name] for name in fields if name in value},
        generations={name: item for name, item in generation.items()
            if name not in {"native_receipts", "unresolved_native_attempts"}},
        training={name: value[name] for name in training_names if name in value},
        reported_integral_added_to_total=False,
        attribution="reported job/phase cost overlaps other manifests; only unified physical samples form totals")


def build_ledger(paths, *, gpu_devices, start=None, end=None, max_sample_gap=6, clock_domain):
    pending, loaded, documents, aliases, seen_content = list(paths), set(), [], [], {}
    series, sources, calls = [], [], {}
    jobs, missing_links = [], []
    while pending:
        path = Path(pending.pop(0)).resolve()
        if path in loaded:
            continue
        loaded.add(path)
        value, sha = read_document(path)
        if sha in seen_content:
            aliases.append(dict(path=str(path), same_content_as=seen_content[sha], sha256=sha))
            continue
        seen_content[sha] = str(path)
        documents.append(dict(path=str(path), sha256=sha))
        jobs.append(job_index(path, value))
        for pointer, rows in physical_series(value):
            rows = sorted(rows, key=lambda row: row["timestamp"])
            source = str(path)+"#"+pointer
            sources.append(dict(source=source, samples=len(rows), first=rows[0]["timestamp"],
                last=rows[-1]["timestamp"], clock_domain=clock_domain,
                original_series_integral=(integrate_physical(rows, rows[0]["timestamp"], rows[-1]["timestamp"])
                    if len(rows) > 1 and all("gpu_gib" in row and "host_gib" in row for row in rows) else None),
                original_series_integral_added_to_total=False))
            series.append((source, rows))
        for node in objects(value):
            for key in ("physical_samples_path", "wire_path"):
                if node.get(key):
                    linked = Path(node[key])
                    linked = linked if linked.is_absolute() else path.parent/linked
                    if linked.exists():
                        pending.append(linked)
                    else:
                        missing_links.append(dict(parent=str(path), field=key, path=str(linked)))
            request, result = node.get("request"), node.get("result")
            receipt = node.get("receipt") or (result.get("receipt") if isinstance(result, dict) else None)
            if isinstance(request, dict) and isinstance(receipt, dict) and "input_ids" in request:
                identity = json.dumps([receipt["engine_restart_id"], receipt["attempt_id"]])
                observed = dict(identity=identity, input_tokens=len(request["input_ids"]),
                    output_tokens=len(result["output_ids"]) if isinstance(result, dict) and "output_ids" in result else None,
                    source_file=str(path), input_hash=receipt["input_hash"])
                if identity in calls and any(calls[identity][key] != observed[key]
                        for key in ("input_tokens", "output_tokens", "input_hash")):
                    raise ValueError("same native receipt has conflicting token observations")
                calls[identity] = observed
    times = [row["timestamp"] for _, rows in series for row in rows]
    if start is None:
        start = min(times) if times else None
    if end is None:
        end = max(times) if times else None
    if start is None or end is None or end <= start:
        return dict(format="contextkv_study_cost_ledger_v1", jobs=jobs, input_files=documents,
            duplicate_content=aliases, physical_total=None,
            unavailable_reason="no nonempty declared physical observation window", missing_links=missing_links)
    gpu, host = defaultdict(list), defaultdict(list)
    large_gaps, device_identities = [], defaultdict(set)
    for source, rows in series:
        for row, nxt in zip(rows, rows[1:]):
            left, right = row["timestamp"], nxt["timestamp"]
            if right <= left:
                continue
            if right-left > max_sample_gap:
                large_gaps.append(dict(source=source, start=left, end=right))
                continue
            next_devices = {(item["index"], item["uuid"]) for item in nxt.get("devices", ())}
            for item in row.get("devices", ()):
                index, uuid = item["index"], item["uuid"]
                device_identities[index].add(uuid)
                if index in gpu_devices and (index, uuid) in next_devices:
                    gpu[index].append((left, right, item["memory_mib"]/1024, source))
            for item in row.get("process_pss", ()):
                host[item["pid"]].append((left, right, item["pss_kib"]/(1024**2), source))
    devices = {str(index): integrate_unique(gpu[index], start, end) for index in gpu_devices}
    processes = {str(pid): integrate_unique(values, start, end) for pid, values in host.items()}
    per_device_missing = [interval for row in devices.values() for interval in row["unknown_intervals"]]
    all_gpu_unknown = merge_ranges(per_device_missing)
    total_gpu = sum(row["gib_seconds"] for row in devices.values() if row["gib_seconds"] is not None)
    total_host = sum(row["gib_seconds"] for row in processes.values() if row["gib_seconds"] is not None)
    return dict(format="contextkv_study_cost_ledger_v1", clock_domain=clock_domain,
        resource_window=dict(start=start, end=end), requested_gpu_devices=list(gpu_devices),
        physical_total=dict(observed_gpu_gib_seconds=total_gpu if any(gpu.values()) else None,
            observed_process_pss_gib_seconds=total_host if processes else None,
            gpu_complete_window_coverage=not all_gpu_unknown, gpu_unknown_intervals=all_gpu_unknown,
            host_complete_machine_coverage=False,
            host_basis="union of actually observed PID PSS; known-process lower bound, not whole machine RAM",
            pre_window_cost="unknown unless an earlier explicit --window-start and source observation covers it",
            gpu_rule="each physical index counted once; freshest sampled interval wins overlap; no sum of reported manifest integrals",
            host_rule="each PID counted once on the common clock; sampled PID identity only, no process-birth identity was recorded",
            monetary_total=None, monetary_basis="no price schedule supplied; explicit reported training/monetary fields remain per-job audit only"),
        devices=devices, processes=processes,
        device_identities={str(index): sorted(uuids) for index, uuids in device_identities.items()},
        native_tokens=dict(uniquely_identified_completed_calls=len(calls),
            input_tokens=sum(row["input_tokens"] for row in calls.values()),
            observed_output_tokens=sum(row["output_tokens"] for row in calls.values() if row["output_tokens"] is not None),
            output_tokens_complete=all(row["output_tokens"] is not None for row in calls.values()),
            scope="linked actual native request/receipt wires only; unlinked job aggregates are not added"),
        jobs=jobs, input_files=documents, duplicate_content=aliases, physical_sources=sources,
        gaps_over_sampling_budget=large_gaps, max_sample_gap_seconds=max_sample_gap, missing_links=missing_links,
        training_cost_rule="training is indexed with actual reported tokens/wall/ranks; its physical samples join the same unified timeline and are not added again")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[], help="Explicit file or quoted glob; repeatable")
    parser.add_argument("--physical", action="append", default=[], help="Physical JSON/NDJSON file or quoted glob; repeatable")
    parser.add_argument("--gpu-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--clock-domain", default="sjy-0518_client_monotonic")
    parser.add_argument("--window-start", type=float)
    parser.add_argument("--window-end", type=float)
    parser.add_argument("--max-sample-gap", type=float, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths, unmatched = [], []
    for pattern in (*args.manifest, *args.physical):
        found = sorted(glob.glob(pattern, recursive=True))
        paths.extend(found)
        if not found:
            unmatched.append(pattern)
    report = build_ledger(paths, gpu_devices=tuple(map(int, args.gpu_devices.split(","))),
        start=args.window_start, end=args.window_end, max_sample_gap=args.max_sample_gap,
        clock_domain=args.clock_domain)
    report["unmatched_explicit_patterns"] = unmatched
    write_json(args.output, report)
    total = report["physical_total"]
    print(json.dumps(dict(jobs=len(report["jobs"]), input_files=len(report["input_files"]),
        gpu_gib_seconds=total["observed_gpu_gib_seconds"] if total else None,
        process_pss_gib_seconds=total["observed_process_pss_gib_seconds"] if total else None,
        gpu_complete=total["gpu_complete_window_coverage"] if total else False,
        unknown_gpu_intervals=len(total["gpu_unknown_intervals"]) if total else None,
        output=args.output)))
    return report


if __name__ == "__main__":
    main()
