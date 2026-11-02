import pytest

from profiling.reading_measurements import reading_measurements


def measured_row():
    return {"task_id": "dev-1", "split": "dev", "reading_outcome": {"steps": [{
        "step_id": "read:0", "task": {"kind": "READ", "operation": "extract", "output_budget": 128},
        "input_ids": [1]*16, "output_ids": [2]*4,
        "receipt": {"actual_device_hit_tokens": 5, "actual_host_loaded_tokens": 3},
        "metrics": {"cache_quote": {"active_requests": 2, "queued_requests": 7,
                "cache_config_hash": "abc", "quotes": [{"ready_gpu_prefix_tokens": 10}]},
            "ttft_ms": 12, "last_token_ms": 18, "decode_ms": 5,
            "step_wall_ms": 25, "elapsed_ms": 20}}]}}


def test_export_uses_actual_prefix_and_active_concurrency_with_independent_phase_scopes():
    groups, ledger = reading_measurements([measured_row()], source="hardware", run_id="r")
    first, decode, control = groups["READ"]
    assert first.prefix_tokens == 8 and first.host_load_tokens == 3
    assert first.concurrency == 3 and ledger[0]["queued_requests"] == 7
    assert (first.elapsed_ms, decode.elapsed_ms, control.elapsed_ms) == (12, 6, 7)
    assert sum(phase.elapsed_ms for phase in groups["READ"]) == 25
    assert ledger[0]["metrics"]["decode_ms"] == 5
    assert len({row.timing_scope for row in groups["READ"]}) == 3


def test_missing_quote_is_unmeasured_and_test_split_is_rejected():
    row = measured_row()
    del row["reading_outcome"]["steps"][0]["metrics"]["cache_quote"]
    groups, ledger = reading_measurements([row], source="hardware", run_id="r")
    assert not groups and ledger[0]["status"] == "unmeasured_native_features"
    row["split"] = "test"
    with pytest.raises(ValueError, match="development"):
        reading_measurements([row], source="hardware", run_id="r")


def test_client_profile_does_not_mix_legacy_native_decode_and_separates_coverage():
    row = measured_row()
    row["reading_outcome"]["steps"][0]["metrics"]["generation_kind"] = "READ_COVERAGE"
    groups, ledger = reading_measurements([row], source="hardware", run_id="r")
    assert set(groups) == {"READ_COVERAGE"}
    del row["reading_outcome"]["steps"][0]["metrics"]["last_token_ms"]
    groups, ledger = reading_measurements([row], source="hardware", run_id="r")
    assert not groups and ledger[0]["status"] == "unmeasured_client_timing"
