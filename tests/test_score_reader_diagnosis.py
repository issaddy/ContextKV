import pytest

from bench.score_reader_diagnosis import comparison_report


def test_reader_comparisons_use_same_completed_states_and_preserve_missing_costs():
    rows = []
    for level in range(3):
        for reader in ("R0", "R1"):
            for binding in ("inferred", "verified"):
                for task in ("a", "b", "topic"):
                    missing = task == "topic" and level == 2
                    rows.append({"method": f"E{level}_{reader}_{binding}", "task_id": task,
                        "status": "review_incomplete" if missing else "completed",
                        "metrics": {} if missing else {"quality": .2+level*.1+(.1 if reader == "R1" else 0)},
                        "cost": {"complete_ms": 10, "total_input_tokens": 20, "total_output_tokens": 5,
                                 "gpu_gib_seconds": 3, "host_gib_seconds": None}})
    report = comparison_report(rows, "quality")
    assert report["common_task_ids"] == ["a", "b"] and report["excluded_task_ids"] == ["topic"]
    assert all(row["paired_tasks"] == 2 for row in report["comparisons"])
    assert all(row["mean_delta"] == pytest.approx(.1) for row in report["comparisons"] if row["intervention"] == "reader")
    cost = report["methods"]["E0_R0_inferred"]["paired_cost_totals"]
    assert cost["total_input_tokens"] == 40 and cost["gpu_gib_seconds"] == 6
    assert cost["host_gib_seconds"] is None
