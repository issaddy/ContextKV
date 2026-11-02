"""Trace observability and censoring; no fixture-derived retention asset."""
from dataclasses import asdict

from controller.schema import ConfigFingerprint
from controller.templates import token_hash
from profiling.retention_coverage import audit_coverage


def recorded_fixture():
    turns = []
    prefixes = ([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 31, 32, 33, 34, 35, 36],
                [1, 2, 3, 4, 5, 6, 91, 92], [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 2, 51, 52, 53, 54, 55, 56])
    for index, prefix in enumerate(prefixes):
        key = dict(corpus_id="fixture", document_id="source", passage_id="p", version="v1", content_sha256="source")
        events = []
        for kind, offset in (("READ", 0), ("ANSWER", 2)):
            ids = prefix if kind == "READ" else [index+30]
            hashed = token_hash(ids)
            events.append(dict(step_id=kind, kind=kind, input_ids=ids, input_hash=hashed,
                prefix_input_ids=ids if kind == "READ" else [],
                fingerprint=asdict(ConfigFingerprint()), original_cache_config_hash="fixture-cache",
                request={"input_ids": ids, "max_new_tokens": 64}, original_receipt={"input_hash": hashed},
                predecessor_step_ids=[], source_started_s=index*10+offset,
                source_completed_s=index*10+offset+1, evidence_token_ranges=[[4, 8]], fixed_system_prefix_tokens=2,
                sources=[{"document_family": "source", "source_spans": [{"source_key": key}]}],
                prefix_observation={"observed_ns": int((index*10+1.25)*1e9), "inspection": {"page_size": 2}}))
        turns.append(dict(attempt_id="turn"+str(index), session_id="b" if index == 2 else "a",
                          split="dev", dataset="fixture", events=events))
    return dict(format="contextkv_source_first_trace_v2", layout="source_first", training_eligible=False,
        turns=turns, input_hashes=[event["input_hash"] for turn in turns for event in turn["events"]],
        input_dag_digest="fixture-DAG")


def test_partial_actual_body_demand_crosses_sessions_and_excludes_system_only_matches():
    report = audit_coverage(recorded_fixture(), horizons=(5, 20, 100), history_steps=2,
        max_replay_steps=20, max_candidates=2, max_prefixes_per_decision=2)
    first, system_only, partial = report["source_reads"][:3]
    demand = first["next_actual_body_demand"]
    assert demand["event_id"] == "turn2:READ" and demand["gap_seconds"] == 18.75
    assert demand["category"] == "partial_body_prefix" and demand["shared_body_pages"] == 1
    assert first["next_same_session_body_demand"]["event_id"] == "turn3:READ"
    assert system_only["next_actual_body_demand"] is None
    assert partial["next_actual_body_demand"]["event_id"] == "turn3:READ"
    assert report["cross_session_next_demands"] >= 1
    assert not report["training_eligible"]


def test_unobserved_long_horizon_is_censored_and_not_zero_probability_or_zero_cost():
    report = audit_coverage(recorded_fixture(), horizons=(5, 20, 100), history_steps=2,
        max_replay_steps=20, max_candidates=2, max_prefixes_per_decision=2)
    long = report["horizons"][-1]
    assert long["horizon_seconds"] == 100 and long["replay_eligible_windows"] == 0
    assert long["descriptive_observed_fraction"] is None and long["sampling_probability"] is None
    assert long["right_censored_source_reads"] == report["reads"]
    assert long["omission_counts"]["no_observed_post_horizon_turn"] == report["reads"]
    medium = report["horizons"][1]
    assert medium["observed_demand_within_ttl"] > 0
    assert medium["sampled_windows"] == 2
    for window in medium["windows"]:
        assert window["sampling_weight"] == 1/medium["sampling_probability"]
