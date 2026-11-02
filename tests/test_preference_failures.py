from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from controller.context_schema import ReadResult, ReadTask, StepRecord
from controller.schema import CompletionReceipt
from controller.session_store import Attempt
from training.preference_rollout import failed_read_candidate
from training.reader_data import write_rows
from training.train_evidence_policy import attach_collection_costs
from test_evidence_policy_training import binding, reward


def test_terminal_read_failure_preserves_paid_output_and_partial_batch_costs(tmp_path):
    attempt = Attempt("attempt", "session", "answer", "hash", 7, "Question", stage="FAILED",
                      error="READ_STEP_OUTPUT_TRUNCATED: read:0")
    receipt = CompletionReceipt("rid", "attempt", "engine", 1, "restart", "input", "length", True)
    record = StepRecord("attempt", "read:0", ReadTask("read:0", "READ"), input_ids=(1, 2, 3), output_ids=(4, 5),
        receipt=receipt, result=ReadResult("read:0", raw_output='{"unfinished', input_tokens=3, output_tokens=2),
        metrics={"elapsed_ms": 12})
    store = SimpleNamespace(find_request=lambda *args: attempt,
        execution_steps_with_stage=lambda identity: (("RESULT_READY", record),),
        get_snapshot=lambda session: SimpleNamespace(raw_history=("actual old turn",), version=7),
        context_plan=lambda identity: None)
    row = {"task_id": "task", "question": "Question", "actual_history": [{"turn_id": 1}]}
    failure = failed_read_candidate(SimpleNamespace(store=store), row, "complementary", binding(), tmp_path)
    assert failure["qa_increment"] == 0 and failure["quality_components"] is None
    assert failure["request_window"] is None and failure["phase_cost"]["complete_ms"] is None
    assert failure["step_records"][0]["record"]["result"]["raw_output"] == '{"unfinished'
    assert failure["step_records"][0]["record"]["receipt"]["finish_reason"] == "length"
    assert failure["phase_cost"]["reader_input_tokens"] == 3 and failure["phase_cost"]["reader_output_tokens"] == 2
    write_rows(tmp_path/"failed_cases.jsonl", [failure])
    successful = reward("good", .9)
    successful.pop("cost")
    successful.update(request_window=[2, 4], input_total_tokens=100, output_total_tokens=20)
    successful["response"]["complete_ms"] = 2000
    write_rows(tmp_path/"rewards.jsonl", [successful])
    (tmp_path/"collection_physical.old.json").write_text(json.dumps({"samples": [
        {"timestamp": 1, "gpu_gib": 2, "host_gib": 3}, {"timestamp": 5, "gpu_gib": 2, "host_gib": 3}]}))
    (tmp_path/"collection_cost.old.json").write_text("{}")
    enriched = attach_collection_costs(tmp_path, binding())
    assert enriched[0]["cost"]["gpu_gib_seconds"] == 4
    preserved = json.loads((tmp_path/"failed_cases.jsonl").read_text())
    assert preserved["physical_cost"] is None and preserved["batch_cost_paths"]
    attempt = replace(attempt, error="UNKNOWN_ENGINE_ERROR")
    with pytest.raises(ValueError, match="not a terminal READ truncation"):
        failed_read_candidate(SimpleNamespace(store=store), row, "complementary", binding(), tmp_path)
