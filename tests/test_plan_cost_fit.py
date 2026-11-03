"""Measured phase envelopes, provenance and explicit coverage boundaries."""
from copy import deepcopy
from dataclasses import asdict
import hashlib

import pytest

from controller.plan_profiles import ExecutionProfiles
from controller.schema import ConfigFingerprint
from controller.templates import token_hash
from profiling.plan_cost_fit import UnqualifiedPhase, fit_profiles, qualify_phase


def phase(*, inputs=1000, kind="READ", ready=0, output=128, start=1, end=3):
    ids = list(range(inputs))
    fingerprint = asdict(ConfigFingerprint(model_revision="reader-R0"))
    quote = dict(request_id="quote", engine_restart_id="restart", cache_config_hash="cache",
        fingerprint=fingerprint, observed_seq=8, active_requests=0, queued_requests=0,
        quotes=[dict(input_hash=token_hash(ids), input_len=inputs,
            ready_gpu_prefix_tokens=ready, host_only_load_tokens=0,
            inflight_h2d_tokens=0, inflight_d2h_tokens=0)])
    return dict(observation_id="step:"+str(inputs)+":"+str(ready), generation_kind=kind,
        context_action="REBUILD", input_ids=ids, input_tokens=inputs, output_budget=output,
        output_tokens=34, phase_window=dict(start=start, end=end, clock="client_monotonic",
            scope="answer_until_commit" if kind == "ANSWER" else "step_wall",
            start_definition="final_answer_compilation_start" if kind == "ANSWER" else "reading_step_entry"),
        receipt=dict(generation_complete=True, input_hash=token_hash(ids), engine_restart_id="restart"),
        cache_quote=quote, generation_window={"start": start+.1, "end": end-.1},
        system_prompt="actual system", sampling_recipe={"temperature": .7},
        split="dev", dataset="actual_source", source_files=["native-proof.json"],
        source_sha256={"native-proof.json": "immutable-sha256"}, input_claim_count=0,
        resource_scope=dict(gpu_devices=[3], devices=[{"index": 3, "uuid": "fixture-gpu"}],
            process_roots=[123], host_basis="fixture PSS roots plus controller descendants",
            native_endpoint="http://127.0.0.1:30002"), resident_model_manifest=None)


def qualified(event):
    samples = [dict(timestamp=0, gpu_gib=4, host_gib=2), dict(timestamp=10, gpu_gib=4, host_gib=2)]
    return qualify_phase(event, samples=samples, all_windows=[], isolation="dedicated_service_device")


def test_full_phase_uses_complete_observed_device_envelope_without_request_share():
    observed = qualified(phase())
    assert observed["latency_ms"] == 2000
    assert observed["gpu_gib_ms"] == 8000 and observed["host_gib_ms"] == 4000
    assert observed["gpu_peak_gib"] == 4 and observed["host_peak_gib"] == 2
    assert observed["quote_binding"]["observed_seq"] == 8
    assert "not marginal" in observed["physical_basis"]


def test_missing_absolute_windows_quotes_boundaries_and_concurrent_steps_are_unqualified():
    event = phase()
    with pytest.raises(UnqualifiedPhase, match="absolute_complete_phase_window"):
        qualified({**event, "phase_window": None})
    with pytest.raises(UnqualifiedPhase, match="native_quote_unobserved"):
        qualified({**event, "cache_quote": None})
    with pytest.raises(UnqualifiedPhase, match="physical_phase_boundary"):
        qualify_phase(event, samples=[{"timestamp": 2, "gpu_gib": 4, "host_gib": 2}],
            all_windows=[], isolation="dedicated_service_device")
    with pytest.raises(UnqualifiedPhase, match="concurrent_phase"):
        qualify_phase(event, samples=[], all_windows=[("other", 2, 4)], isolation="dedicated_service_device")
    replay_answer = phase(kind="ANSWER")
    replay_answer["phase_window"]["scope"] = "step_wall"
    with pytest.raises(UnqualifiedPhase, match="absolute_complete_phase_window"):
        qualified(replay_answer)


def test_fit_preserves_reader_prompt_sampling_and_observed_support_with_explicit_prediction_bins():
    events = [qualified(phase(inputs=1100, start=1, end=2)), qualified(phase(inputs=1300, start=1, end=3)),
              qualified(phase(inputs=1100, ready=64))]
    asset = fit_profiles(events, reader_checkpoint="reader-R0", reader_profile_id="dev-profile", split="dev")
    profiles = ExecutionProfiles(asset)
    row = profiles.row("READ", "REBUILD", 1200, 128, 1, 0, 0, "actual system")
    assert row["input_range"] == [1100, 1300] and row["latency_ms"] == 2000
    assert row["observed_input_range"] == [1100, 1300] and row["prediction_bin"] == [1024, 2047]
    assert row["gpu_gib_ms"] == 8000 and row["ready_gpu_range"] == [0, 0]
    assert row["cache_conditioning"] == "measured_cache_and_concurrency_ranges"
    warm = profiles.row("READ", "REBUILD", 1200, 128, 1, 64, 0, "actual system")
    assert warm["observed_input_range"] == [1100, 1100] and warm["ready_gpu_range"] == [64, 64]
    assert profiles.row("READ", "REBUILD", 2048, 128, 1, 0, 0, "actual system") is None
    concurrent = profiles.row("READ", "REBUILD", 1200, 128, 2, 0, 0, "actual system")
    unobserved_hit = profiles.row("READ", "REBUILD", 1200, 128, 1, 32, 0, "actual system")
    assert concurrent["cache_conditioning"] == unobserved_hit["cache_conditioning"] == "cache_state_envelope"
    assert concurrent["concurrency_inference"]["duration_multiplier"] == 2
    assert unobserved_hit["observed_input_range"] == [1100, 1300]
    assert {tuple(item["observed_input_range"]) for item in unobserved_hit["observed_cache_ranges"]} == {
        (1100, 1300), (1100, 1100)}
    assert profiles.row("READ", "REBUILD", 1200, 128, 1, 0, 0, "different prompt") is None
    assert asset["coverage"]["missing_full_tail_kinds"] == ["ANSWER"]
    changed = deepcopy(events[0])
    changed["fingerprint"]["model_revision"] = "reader-R1"
    with pytest.raises(ValueError, match="cannot mix reader"):
        fit_profiles([events[0], changed], reader_checkpoint="reader-R0", reader_profile_id="dev-profile", split="dev")


def test_cpu_profiles_require_actual_workload_and_operation_claim_count_support():
    event = phase(inputs=0)
    event.update(generation_kind="CPU_REDUCE", context_action="CPU", output_budget=0,
        output_tokens=0, system_prompt="", operation="count", input_claim_count=4,
        fingerprint=asdict(ConfigFingerprint(model_revision="reader-R0")), native_workload={
            "active_requests": 0, "queued_requests": 0, "cache_config": {"page_size": 64}})
    observed = qualified(event)
    assert observed["cache_config_hash"] == hashlib.sha256(b'{"page_size": 64}').hexdigest()
    asset = fit_profiles([observed], reader_checkpoint="reader-R0", reader_profile_id="dev-profile", split="dev")
    row = asset["rows"][0]
    assert row["input_measure"] == "claim_count" and row["operation"] == "count"
    assert row["input_range"] == [4, 4]
    with pytest.raises(UnqualifiedPhase, match="cpu_native_workload_unobserved"):
        qualified({**event, "native_workload": None})
