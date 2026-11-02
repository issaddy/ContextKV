"""Failed collection work is costed even when no frozen turn can be exported."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from controller.session_store import SessionStore
from profiling.retention_capture_cost import CaptureRunAccounting, capture_generation_summary


def test_native_completed_truncation_and_unresolved_transport_are_not_silent_skips():
    request = dict(input_ids=[1, 2, 3], attempt_id="native-failed-read", session_id="epoch",
                   expected_restart_id="restart")
    call = dict(request=request, receipt={"finish_reason": "length", "generation_complete": True},
                result={"output_ids": [4, 5]})
    costs = capture_generation_summary([call, dict(request={**request, "attempt_id": "unresolved"},
                                                  error={"type": "ReadTimeout"})])
    assert costs["submitted"] == 2 and costs["native_completed"] == 1
    assert costs["observed_output_tokens"] == 2 and costs["observed_input_tokens"] == 6
    assert not costs["output_tokens_complete"]
    assert costs["native_receipts"][0]["finish_reason"] == "length"
    assert costs["unresolved_native_attempts"][0]["attempt_id"] == "unresolved"


class NativeSamples:
    def __init__(self):
        self.samples = []
    async def start(self):
        self.samples.append(dict(timestamp=time.monotonic(), gpu_slots=128, host_slots=64))
    async def stop(self):
        self.samples.append(dict(timestamp=time.monotonic(), gpu_slots=128, host_slots=64))
    def observation(self):
        return dict(samples=self.samples, errors=[], measurement="sampled_unique_pool_slots",
                    interval_seconds=.1, scope="test fixture")


class PhysicalSamples:
    def __init__(self):
        self.samples = []
    async def start(self):
        self.samples.append(dict(timestamp=time.monotonic(), gpu_gib=2, host_gib=1))
    async def stop(self):
        self.samples.append(dict(timestamp=time.monotonic(), gpu_gib=2, host_gib=1))
    def observation(self):
        return dict(samples=self.samples, scope="test fixture")


@pytest.mark.asyncio
async def test_capture_failure_preserves_entry_cost_and_links_successful_turn_subset(tmp_path):
    config = SimpleNamespace(state_dir=str(tmp_path/"state"), engine=SimpleNamespace(kind="mock"))
    (tmp_path/"capture-trace.json").write_text(json.dumps({"turns": [{"attempt_id": "success"}]}))
    accounting = CaptureRunAccounting(config, tmp_path, 3,
        native=NativeSamples(), physical=PhysicalSamples())
    with pytest.raises(ValueError, match="TRUNCATED"):
        async with accounting:
            accounting.current_source = {"example_id": "failed-entry"}
            store = SessionStore(tmp_path/"state"/"sessions.sqlite3")
            store.read_snapshot("business", "corpus")
            attempt = store.begin_attempt("business", "failed-request", "input", "Question", 0)
            store.fail_attempt(attempt.attempt_id, "READ_STEP_OUTPUT_TRUNCATED")
            store.close()
            raise ValueError("READ_STEP_OUTPUT_TRUNCATED")
    report = json.loads((tmp_path/"capture-entry.json").read_text())
    assert report["status"] == "failed" and report["current_source"]["example_id"] == "failed-entry"
    assert report["error"]["message"] == "READ_STEP_OUTPUT_TRUNCATED"
    assert report["physical_cost"]["complete_window_coverage"]
    assert report["native_cost"]["coverage"]["complete_window_coverage"]
    assert report["controller_cpu_seconds"] >= 0
    assert report["attempts"][0]["request_id"] == "failed-request"
    assert report["attempts"][0]["stage"] == "FAILED"
    assert report["attempts"][0]["error"] == "READ_STEP_OUTPUT_TRUNCATED"
    manifest = json.loads((tmp_path/"capture-trace.json").read_text())["entry_accounting"]
    assert manifest["status"] == "failed" and manifest["successful_turns"] == 1
    assert len(manifest["sha256"]) == 64 and Path(tmp_path/manifest["path"]).is_file()
