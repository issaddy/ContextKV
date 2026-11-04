"""Audit observable C reuse horizons without selecting on retention benefit."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from bench.metrics import distribution
from profiling.retention_trace import (causal_pages, choose_windows, digest, event_id,
    read_decisions, revisit_outcomes, trace_generations, verify_trace, write_json)


def actual_page_size(reads, declared):
    sizes = {event["prefix_observation"]["inspection"]["page_size"] for _, event in reads
        if event.get("prefix_observation") and "inspection" in event["prefix_observation"]}
    if declared is not None:
        sizes.add(declared)
    if len(sizes) != 1:
        raise ValueError("coverage needs one actual observed native page_size or its explicit declaration")
    return sizes.pop()


def decision_boundary(event):
    observed = event.get("prefix_observation")
    if observed is not None and observed.get("observed_ns") is not None:
        return observed["observed_ns"]/1e9, "actual_capture_prefix_observation", True
    return event["source_completed_s"], "actual_native_READ_completion_pre_control", False


def source_families(event):
    return sorted({(span["source_key"]["corpus_id"], source["document_family"],
                    span["source_key"]["version"]) for source in event["sources"]
                   for span in source["source_spans"]})


def audit_coverage(trace, *, horizons, page_size=None, max_candidates=8,
                   max_prefixes_per_decision=4, history_steps=32, max_replay_steps=128, seed=42):
    verify_trace(trace)
    reads = sorted(read_decisions(trace), key=lambda item: item[1]["source_started_s"])
    if not reads:
        raise ValueError("coverage needs actual canonical READ observations")
    size = actual_page_size(reads, page_size)
    rows, by_id, page_reads, session_page_reads = [], {}, defaultdict(list), defaultdict(list)
    for index, (turn, event) in enumerate(reads):
        scope = digest([event["fingerprint"], event["original_cache_config_hash"]])
        pages, body, system = causal_pages(event, size, scope)
        boundary, basis, exact = decision_boundary(event)
        row = dict(event_id=event_id(turn, event), session_id=turn["session_id"],
            split=turn["split"], dataset=turn["dataset"], cache_scope_hash=scope,
            input_hash=event["input_hash"], submitted_s=event["source_started_s"],
            decision_s=boundary, decision_boundary_basis=basis, exact_decision_observation=exact,
            causal_prefix_signature=digest(tuple(pages)), evidence_page_signature=digest(sorted(body)),
            source_body_pages=len(body), system_only_pages=len(system),
            source_families_versions=source_families(event),
            body_pages=body, prefix_pages=pages)
        rows.append(row)
        by_id[row["event_id"]] = row
        for page in body:
            page_reads[page].append((row["submitted_s"], index))
            session_page_reads[page, row["session_id"]].append((row["submitted_s"], index))

    def next_demand(source, boundary, *, same_session=None):
        found = []
        for page in source["body_pages"]:
            visits = (page_reads[page] if same_session is None else session_page_reads[page, same_session])
            position = bisect_right(visits, (boundary, len(rows)))
            if position < len(visits):
                found.append(visits[position])
        if not found:
            return None
        timestamp, index = min(found)
        target = rows[index]
        shared = source["body_pages"] & target["body_pages"]
        return dict(event_id=target["event_id"], session_id=target["session_id"],
            gap_seconds=timestamp-boundary, shared_body_pages=len(shared),
            shared_body_page_signature=digest(sorted(shared)),
            category="full_retained_body_prefix" if shared == source["body_pages"] else "partial_body_prefix")

    trace_end = max(event["source_completed_s"] for _, event in trace_generations(trace))
    turn_starts = sorted(min(event["source_started_s"] for event in turn["events"]
        if event["kind"] != "CPU_REDUCE") for turn in trace["turns"])
    for row in rows:
        row["followup_seconds"] = max(0, trace_end-row["decision_s"])
        row["next_actual_body_demand"] = next_demand(row, row["decision_s"])
        row["next_same_session_body_demand"] = next_demand(row, row["decision_s"], same_session=row["session_id"])
    horizon_rows = []
    capacity = len(reads)*max_prefixes_per_decision
    for horizon in horizons:
        all_windows, omissions = choose_windows(trace, horizon_seconds=horizon,
            max_candidates=capacity, max_prefixes_per_decision=max_prefixes_per_decision,
            history_steps=history_steps, max_replay_steps=max_replay_steps, seed=seed)
        selected = all_windows[:max_candidates]
        probability = min(1, max_candidates/len(all_windows)) if all_windows else None
        cases = []
        for window in all_windows:
            decision, source = by_id[window["event_id"]], by_id[window["source_event_id"]]
            demand = next_demand(source, decision["decision_s"])
            cases.append(dict(event_id=window["event_id"], source_event_id=window["source_event_id"],
                source_session_id=source["session_id"], decision_session_id=decision["session_id"],
                source_body_pages=source["source_body_pages"], next_actual_body_demand=demand,
                within_ttl=demand is not None and demand["gap_seconds"] <= horizon,
                exact_decision_observation=decision["exact_decision_observation"], steps=window["steps"],
                first_turn=window["first_turn"], last_turn=window["last_turn"],
                sampled=window in selected, sampling_probability=probability,
                sampling_weight=1/probability if probability is not None else None))
        identifiable = [case for case in cases if case["source_body_pages"] and case["exact_decision_observation"]]
        sampled = [case for case in identifiable if case["sampled"]]
        complete = [row for row in rows if row["exact_decision_observation"] and row["source_body_pages"]
            and bisect_right(turn_starts, row["decision_s"]+horizon) < len(turn_starts)]
        horizon_rows.append(dict(horizon_seconds=horizon,
            complete_source_read_followups=len(complete),
            right_censored_source_reads=sum(row["source_body_pages"] and row not in complete for row in rows),
            replay_eligible_windows=len(cases), identifiable_body_windows=len(identifiable),
            observed_demand_within_ttl=sum(case["within_ttl"] for case in identifiable),
            descriptive_observed_fraction=(sum(case["within_ttl"] for case in identifiable)/len(identifiable)
                                           if identifiable else None),
            sampled_windows=len(selected), sampled_identifiable_body_windows=len(sampled),
            sampling_probability=probability,
            sampled_weight_sum=sum(case["sampling_weight"] for case in sampled),
            omission_counts=dict(Counter(row["reason"] for row in omissions)), windows=cases,
            interpretation="historical observable support, not an estimated probability for censored or unseen TTLs"))
    demands = [row["next_actual_body_demand"] for row in rows if row["next_actual_body_demand"] is not None]
    public = [{key: value for key, value in row.items() if key not in {"body_pages", "prefix_pages"}} for row in rows]
    return dict(format="contextkv_retention_coverage_v1", trace_digest=digest(trace),
        input_dag_digest=trace["input_dag_digest"], page_size=size,
        training_eligible=trace["training_eligible"], cache_scopes=sorted({row["cache_scope_hash"] for row in rows}),
        reads=len(rows), sessions=len({row["session_id"] for row in rows}),
        body_prefixes=len({row["causal_prefix_signature"] for row in rows if row["source_body_pages"]}),
        repeated_actual_prefix_visits=len(rows)-len({row["causal_prefix_signature"] for row in rows}),
        recorded_seconds=trace_end-min(row["submitted_s"] for row in rows),
        observed_next_body_demand_gaps=distribution(row["gap_seconds"] for row in demands),
        observed_same_session_body_demand_gaps=distribution(row["next_same_session_body_demand"]["gap_seconds"]
            for row in rows if row["next_same_session_body_demand"] is not None),
        cross_session_next_demands=sum(row["next_actual_body_demand"] is not None and
            row["next_actual_body_demand"]["session_id"] != row["session_id"] for row in rows),
        demand_categories=dict(Counter(row["category"] for row in demands)),
        source_reads=public, horizons=horizon_rows,
        replay_sampling=dict(max_candidates=max_candidates, max_prefixes_per_decision=max_prefixes_per_decision,
            history_steps=history_steps, max_replay_steps=max_replay_steps, seed=seed,
            rule="uniform deterministic hash sample of each horizon's observable windows; no retention outcome used"),
        claim_scope="source-body causal demand and follow-up support only; no retention benefit, model training or TTL optimization",
        censoring_rule="a full observed post-horizon turn is required; an unseen future demand is not labelled no reuse")


def workload_revisit_coverage(trace, report):
    """Measure repeated body-page demand after earlier completed READs."""
    reads, first_completed = [], {}
    for turn, event in read_decisions(trace):
        identity = event_id(turn, event)
        actual = report["events"][identity]
        pages, body, _ = causal_pages(event, report["page_size"], report["cache_scope_hash"])
        reads.append((identity, actual, pages, body))
        completed = actual["generation_window"]["end"]
        for page in body:
            first_completed[page] = min(first_completed.get(page, completed), completed)
    window = report["observation_window"]
    totals = Counter(gpu_hit=0, host_load=0, recomputed=0)
    visits, observed_reads, first_only_reads = [], 0, 0
    for identity, actual, pages, body in reads:
        started = actual["generation_window"]["start"]
        if not window["start"] < started <= window["end"]:
            continue
        observed_reads += 1
        repeated = {page for page in body if first_completed[page] < started}
        if not repeated:
            first_only_reads += 1
            continue
        receipt = actual["result"]["receipt"]
        gpu_end = receipt["actual_device_hit_tokens"]
        host_end = gpu_end+receipt["actual_host_loaded_tokens"]
        gpu = {page for page in repeated if pages[page] <= gpu_end}
        host = {page for page in repeated if gpu_end < pages[page] <= host_end}
        groups = {"gpu_hit": gpu, "host_load": host, "recomputed": repeated-gpu-host}
        counts = {name: len(values) for name, values in groups.items()}
        totals.update(counts)
        visits.append({"event_id": identity, "generation_window": actual["generation_window"],
            "page_counts": counts, "page_proof": {name: [
                {"page_id": page, "first_completed_s": first_completed[page], "token_end": pages[page]}
                for page in sorted(values)[:3]] for name, values in groups.items()}})
    return {"scope": "all repeated source-body demand in the observed workload window",
        "observed_reads": observed_reads, "reads_without_repeated_body_demand": first_only_reads,
        "revisit_events": len(visits), "no_revisit": not visits, "page_counts": dict(totals),
        "event_counts": {name: sum(visit["page_counts"][name] > 0 for visit in visits) for name in totals},
        "visits": visits,
        "repeat_rule": "the same causal body page completed in a READ strictly before the current READ started",
        "attribution": "workload coverage only; these totals do not measure intervention-target benefit"}


def audit_replay_coverage(root):
    root = Path(root)
    windows, unread = [], []
    for trace_path in sorted(root.rglob("trace.json")):
        pair = trace_path.parent
        if not pair.name.startswith("pair-"):
            continue
        trace = json.loads(trace_path.read_text())
        arms = {}
        for report_path in sorted(pair.glob("*/report.json")):
            report = json.loads(report_path.read_text())
            if report["status"] != "completed" or "observation_window" not in report:
                unread.append({"report": str(report_path), "status": report["status"]})
                continue
            target = revisit_outcomes(trace, report, report["decision_target"])
            arms[report_path.parent.name] = {"report": str(report_path),
                "observation_window": report["observation_window"],
                "target": {key: target[key] for key in (
                    "no_revisit", "revisit_events", "page_counts", "event_counts", "observation_seconds")},
                "target_event_ids": [visit["event_id"] for visit in target["visits"]],
                "workload": workload_revisit_coverage(trace, report)}
        windows.append({"window": str(pair), "trace": str(trace_path), "arms": arms})
    return {"format": "contextkv_retention_replay_coverage_v1", "replay_root": str(root),
        "windows": windows, "unread_reports": unread,
        "reports_read": sum(len(window["arms"]) for window in windows),
        "scope": "existing completed arms only; target demand and whole-window workload coverage reported separately",
        "model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace")
    source.add_argument("--replay-root")
    parser.add_argument("--horizons", type=lambda value: tuple(map(float, value.split(","))),
                        default=(1, 2, 5, 8, 15, 30, 60, 120))
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-prefixes-per-decision", type=int, default=4)
    parser.add_argument("--history-steps", type=int, default=32)
    parser.add_argument("--max-replay-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.replay_root:
        report = audit_replay_coverage(args.replay_root)
        summary = {"windows": len(report["windows"]), "reports_read": report["reports_read"]}
    else:
        report = audit_coverage(json.loads(Path(args.trace).read_text()), horizons=args.horizons,
            page_size=args.page_size, max_candidates=args.max_candidates,
            max_prefixes_per_decision=args.max_prefixes_per_decision, history_steps=args.history_steps,
            max_replay_steps=args.max_replay_steps, seed=args.seed)
        summary = {key: report[key] for key in (
            "reads", "sessions", "body_prefixes", "cross_session_next_demands")}
    write_json(args.output, report)
    print(json.dumps(summary))
    return report


if __name__ == "__main__":
    main()
