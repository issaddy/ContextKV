



import pytest






from bench.study_metrics import declared_queue_summary


def saved(task, *, method="R0", status="completed", score=.8, history=()):
    return {"task_id": task, "method": method, "dataset": "qa", "split": "dev",
        "conversation_id": "conversation", "question": "Which version applies?",
        "actual_history": list(history), "answer": "Version 2" if status == "completed" else "",
        "status": status, "metrics": {"RB_llm": score} if score is not None else {},
        "cost": {"total_input_tokens": 100, "total_output_tokens": 10,
                 "complete_ms": 20, "gpu_gib_seconds": 2, "host_gib_seconds": 1}}


def test_declared_queue_keeps_failed_absent_tasks_and_all_paid_costs():
    rows = [saved("answer"), saved("failure", status="failed", score=None)]
    result = declared_queue_summary(rows, {"primary_metric": "RB_llm"},
                                    ["answer", "failure", "not_started"])
    assert result["queue_primary_score"] == pytest.approx(.8/3)
    assert result["scored_subset_primary"] == .8
    assert result["completion_rate"] == pytest.approx(1/3)
    assert result["missing_execution_task_ids"] == ["not_started"]
    assert result["cost"]["online_tokens"] == 220
    assert result["paid_cost"]["total_input_tokens"]["failed_request_total"] == 100
