from copy import deepcopy

import pytest

from bench.study_metrics import paired_comparisons, select_development, summarize_methods


def matrix():
    return {"methods": [{"id": "B0"}, {"id": "B3"}], "seed": 42,
        "datasets": {"mtrag": {"primary_metric": "mtrageval_harmonic", "normalization": 1, "weight": .5},
                     "loong": {"primary_metric": "loong_reference_score", "normalization": 100, "weight": .5}},
        "cohorts": {"dev": {"mtrag": ["a", "b"], "loong": ["long"]}},
        "resources": {"p95_complete_ms": 5000, "max_online_tokens": 100000,
                      "gpu_capacity_tokens": 32768, "host_capacity_tokens": 61056},
        "selection": {"near_equal_quality": .002, "bootstrap_samples": 100}}


def cost(complete_ms=1000):
    return {"complete_ms": complete_ms, "total_input_tokens": 100, "total_output_tokens": 20,
        "gpu_gib_seconds": 2, "host_gib_seconds": 1, "online_compute_ms": 50,
        "gpu_capacity_tokens": 32768, "host_capacity_tokens": 61056}


def records(method="B0", *, latency=1000, long_score=80):
    return [{"method": method, "dataset": "mtrag", "task_id": "a", "conversation_id": "c",
             "status": "completed", "metrics": {"RL_F": [1], "RB_llm": [.1], "RB_agg": [.1],
                 "RL_F_idk": [1], "RB_llm_idk": [.1], "RB_agg_idk": [.1]},
             "cost": cost(latency), "split": "dev"},
            {"method": method, "dataset": "mtrag", "task_id": "b", "conversation_id": "c",
             "status": "completed", "metrics": {"RL_F": [.1], "RB_llm": [1], "RB_agg": [1],
                 "RL_F_idk": [.1], "RB_llm_idk": [1], "RB_agg_idk": [1]},
             "cost": cost(latency), "split": "dev"},
            {"method": method, "dataset": "loong", "task_id": "long", "conversation_id": "d",
             "status": "completed", "metrics": {"loong_reference_score": long_score},
             "cost": cost(latency), "split": "dev"}]


def test_mtrag_harmonic_uses_benchmark_aggregates_and_preserves_official_alias():
    summary = summarize_methods(matrix(), records())["B0"]
    assert summary["datasets"]["mtrag"]["primary_score"] == pytest.approx(.55)
    assert summary["quality"] == pytest.approx(.675)
    assert summary["eligible"]
    assert summary["datasets"]["mtrag"]["raw_metrics"]["RB_agg"]["count"] == 2
    assert summary["datasets"]["mtrag"]["metric_source_alias_counts"]["RB_agg_idk"] == 2
    assert "run_algorithmic.py" in summary["datasets"]["mtrag"]["alias_provenance"]["RB_agg_idk"]


def test_mtrag_primary_uses_answerability_conditioned_components_and_retains_raw_scores():
    rows = records()
    for row in rows[:2]:
        row["metrics"].update({name+"_idk": [0] for name in ("RL_F", "RB_llm", "RB_agg")})
    summary = summarize_methods(matrix(), rows)["B0"]
    assert summary["datasets"]["mtrag"]["primary_score"] == 0
    assert summary["datasets"]["mtrag"]["raw_metrics"]["RL_F"]["mean"] == pytest.approx(.55)
    for row in rows[:2]:
        row["metrics"].update({name+"_idk_underspecified": [1]
                               for name in ("RL_F", "RB_llm", "RB_agg")})
    assert summarize_methods(matrix(), rows)["B0"]["datasets"]["mtrag"]["primary_score"] == 1


def test_missing_harmonic_component_or_cohort_never_freezes_partial_scores():
    rows = records()
    del rows[0]["metrics"]["RL_F_idk"]
    summary = summarize_methods(matrix(), rows)["B0"]
    assert summary["quality"] is None
    assert summary["datasets"]["mtrag"]["scored"] == 1
    with pytest.raises(ValueError):
        select_development(matrix(), rows)
    undeclared = matrix()
    del undeclared["cohorts"]
    assert "cohort_not_declared:mtrag" in summarize_methods(undeclared, records())["B0"]["selection_reasons"]


def test_continuous_quality_wins_then_near_equal_uses_complete_cost():
    rows = records("B0", latency=100, long_score=70)+records("B3", latency=4000, long_score=90)
    assert select_development(matrix(), rows)["method"] == "B3"
    near = records("B0", latency=100, long_score=89.8)+records("B3", latency=4000, long_score=90)
    assert select_development(matrix(), near)["method"] == "B0"
    over_budget = records("B0", latency=100, long_score=70)+records("B3", latency=6000, long_score=90)
    assert select_development(matrix(), over_budget)["method"] == "B0"


def test_missing_resource_measurements_and_duplicate_tasks_are_reported():
    rows = records()
    del rows[0]["cost"]["gpu_gib_seconds"]
    rows.append(deepcopy(rows[0]))
    summary = summarize_methods(matrix(), rows)["B0"]
    assert not summary["eligible"]
    assert summary["datasets"]["mtrag"]["duplicate_task_ids"] == ["a"]
    assert "missing_cost:gpu_gib_seconds" in summary["selection_reasons"]


def test_paired_bootstrap_resamples_conversations_and_preserves_raw_score_units():
    spec = {"methods": [{"id": "B0"}, {"id": "B3"}], "seed": 42,
        "datasets": {"loong": {"primary_metric": "loong_reference_score", "normalization": 100, "weight": 1}},
        "cohorts": {"test": {"loong": ["0", "1", "2"]}},
        "selection": {"bootstrap_samples": 50, "equal_cost_relative_tolerance": .05}}
    rows = []
    for method, gain in (("B0", 0), ("B3", 10)):
        for index, conversation in enumerate(("one", "one", "two")):
            rows.append({"method": method, "dataset": "loong", "task_id": str(index),
                "conversation_id": conversation, "status": "completed", "split": "test",
                "metrics": {"loong_reference_score": 50+index+gain}, "cost": cost()})
    comparisons = paired_comparisons(spec, rows)
    assert comparisons[0]["clusters"] == 2
    assert comparisons[0]["paired_tasks"] == 3
    assert comparisons[0]["score_interval"] == pytest.approx([10, 10])
    assert comparisons[0]["equal_cost_match"]
    assert comparisons[1]["dataset"] == "engineering_Q"
    assert comparisons[1]["score_interval"] == pytest.approx([.1, .1])


def test_paired_subset_reports_tasks_both_methods_did_not_score_without_full_cohort_q():
    spec = {"methods": [{"id": "B0"}, {"id": "B3"}], "seed": 42,
        "datasets": {"loong": {"primary_metric": "loong_reference_score", "normalization": 100, "weight": 1}},
        "cohorts": {"test": {"loong": ["a", "b", "both_failed"]}}}
    rows = [{"method": method, "dataset": "loong", "task_id": task,
        "conversation_id": task, "status": "failed" if task == "both_failed" else "completed",
        "split": "test", "metrics": {} if task == "both_failed" else {"loong_reference_score": score},
        "cost": cost()} for method, score in (("B0", 50), ("B3", 60))
        for task in ("a", "b", "both_failed")]
    comparisons = paired_comparisons(spec, rows, samples=20)
    dataset, combined = comparisons
    assert dataset["planned_tasks"] == 3
    assert dataset["paired_task_ids"] == ["a", "b"]
    assert dataset["both_unscored_task_ids"] == ["both_failed"]
    assert dataset["baseline_unscored_task_ids"] == ["both_failed"]
    assert dataset["method_unscored_task_ids"] == ["both_failed"]
    assert dataset["score_difference"] == 10
    assert combined["dataset"] == "engineering_Q_common_scored"
    assert combined["comparison_scope"] == "common_scored_subset"
    assert not combined["complete_cohort"]
    assert combined["score_difference"] == pytest.approx(.1)


def test_empty_paired_subset_still_reports_declared_cohort_coverage():
    spec = {"methods": [{"id": "B0"}, {"id": "B3"}],
        "datasets": {"loong": {"primary_metric": "loong_reference_score", "weight": 1}},
        "cohorts": {"test": {"loong": ["missing"]}}}
    rows = [{"method": method, "dataset": "loong", "task_id": "missing",
             "conversation_id": "c", "status": "failed", "split": "test", "metrics": {}}
            for method in ("B0", "B3")]
    result, = paired_comparisons(spec, rows)
    assert result["paired_tasks"] == 0
    assert result["both_unscored_task_ids"] == ["missing"]
    assert result["score_difference"] is None


def test_development_track_excludes_evaluation_only_dataset_without_renormalizing():
    spec = matrix()
    spec["datasets"]["mtrag_un"] = {"primary_metric": "mtrageval_harmonic", "normalization": 1, "weight": .2}
    spec["selection"]["development_weights"] = {"mtrag": .4, "loong": .6}
    summary = summarize_methods(spec, records())["B0"]
    assert summary["eligible"]
    assert summary["quality"] == pytest.approx(.55*.4+.8*.6)
    assert summary["quality_track"] == "development_selection"
    assert summary["excluded_datasets"] == ["mtrag_un"]
    assert summary["track_weights"] == {"mtrag": .4, "loong": .6}
    spec["cohorts"]["test"] = spec["cohorts"]["dev"]
    test_rows = [{**item, "split": "test"} for item in records()]
    test_summary = summarize_methods(spec, test_rows, "test")["B0"]
    assert not test_summary["eligible"]
    assert "mtrag_un" in test_summary["datasets"]


def test_online_totals_already_include_auxiliary_model_tokens():
    rows = records()
    for row in rows:
        row["cost"].update(total_input_tokens=150, total_output_tokens=25,
                           auxiliary_input_tokens=50, auxiliary_output_tokens=5)
    cost_summary = summarize_methods(matrix(), rows)["B0"]["cost"]
    assert cost_summary["online_tokens"] == 3*175
    assert cost_summary["auxiliary_input_tokens"] == 150
    assert cost_summary["token_accounting"] == "all_online_models"
