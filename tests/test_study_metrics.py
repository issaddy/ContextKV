
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


def test_continuous_quality_wins_then_near_equal_uses_complete_cost():
    rows = records("B0", latency=100, long_score=70)+records("B3", latency=4000, long_score=90)
    assert select_development(matrix(), rows)["method"] == "B3"
    near = records("B0", latency=100, long_score=89.8)+records("B3", latency=4000, long_score=90)
    assert select_development(matrix(), near)["method"] == "B0"
    over_budget = records("B0", latency=100, long_score=70)+records("B3", latency=6000, long_score=90)
    assert select_development(matrix(), over_budget)["method"] == "B0"


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


def test_final_index_compares_fixed_arrivals_and_declared_slo(tmp_path):
    import json
    from profiling.final_study_index import full_queue_cells, full_queue_comparisons, main, markdown

    run = tmp_path / 'reader.json'
    run.write_text(json.dumps({'engine': {'base_url': 'local'}, 'state_dir': 'temporary'}))
    plan = dict(reader_run={'path': str(run)},
        source_config={'sha256': 'A'}, source_matrix={'sha256': 'matrix'},
        cohort={'track': ['task']}, reader_profile_id='fixed', lambda_time=0,
        workload_plans={'4': {'track': {'sha256': 'arrivals-A'}}},
        assets={'gain': {'sha256': 'B'}, 'value': None},
        jobs=[dict(name='job', output='generation', complete_slo_seconds=30)])
    plan_path = tmp_path / 'capacity-plan.json'
    plan_path.write_text(json.dumps(plan))
    row = dict(job='job', method='B6_native', concurrency=4, requested_qa=1, completed_qa=1,
        queue_quality=.5, p95_complete_ms=100, paid_native_usage={},
        datasets={'track': dict(missing_execution_task_ids=[], failed_task_ids=[], unscored_answer_task_ids=[])})
    path = tmp_path / 'workflow.json'
    def cells(value):
        return full_queue_cells({'format': 'contextkv_full_queue_joint_workflow_v1', 'results': [value]}, path)
    reference = cells(row)
    baseline = cells({**row, 'method': 'E0_native'})
    assert full_queue_comparisons(baseline + reference)[0]['baseline'] == 'E0_native'
    candidate = {**row, 'method': 'S3_native', 'queue_quality': .6}
    same = full_queue_comparisons(reference + cells(candidate))[0]
    assert same['comparable'] and same['differences']['queue_quality'] == pytest.approx(.1)
    faster = full_queue_comparisons(reference + cells({**candidate, 'p95_complete_ms': 85}))[0]
    assert faster['relative_p95_change'] == pytest.approx(-.15)
    assert faster['quality_not_lower'] is True
    incomplete = full_queue_comparisons(reference + cells({**candidate, 'completed_qa': 0}))[0]
    assert incomplete['relative_p95_change'] is None
    assert incomplete['quality_not_lower'] is None
    assert incomplete['differences']['queue_quality'] == pytest.approx(.1)
    unscored = {**candidate, 'datasets': {'track': dict(missing_execution_task_ids=[],
        failed_task_ids=[], unscored_answer_task_ids=['task'])}}
    assert full_queue_comparisons(reference + cells(unscored))[0]['relative_p95_change'] is None
    assert full_queue_comparisons(reference + cells({**candidate, 'queue_quality': None}))[0]['quality_not_lower'] is None
    assert full_queue_comparisons(cells({**row, 'p95_complete_ms': 0}) + cells(candidate))[0]['relative_p95_change'] is None
    plan['workload_plans']['4']['track']['sha256'] = 'arrivals-B'
    plan_path.write_text(json.dumps(plan))
    changed = full_queue_comparisons(reference + cells(candidate))[0]
    assert not changed['comparable'] and changed['differences']['queue_quality'] is None
    assert changed['incompatible_identity_fields'] == ['workload_sha256']
    assert changed['relative_p95_change'] is None
    plan['workload_plans']['4']['track']['sha256'] = 'arrivals-A'
    plan['jobs'][0]['complete_slo_seconds'] = 60
    plan_path.write_text(json.dumps(plan))
    assert full_queue_comparisons(reference + cells(candidate))[0]['incompatible_identity_fields'] == ['complete_slo_seconds']

    out = tmp_path / 'index'
    report = main(['--baseline-commit', '85f4ed4', '--physical', 'recorded-physical.json',
                  '--gpu-devices', '2', '--output', str(out)])
    assert report['baseline_commit'] == '85f4ed4'
    report['capacity_comparisons'] = [faster, incomplete]
    rendered = markdown(report)
    assert '-15.00%' in rendered and '未知' in rendered
    assert '质量差（候选−基线）' in rendered and '负值表示降低' in rendered
    command = json.loads((out / 'cost-ledger-command.json').read_text())
    assert not command['executed']
    assert command['argv'][command['argv'].index('--gpu-devices') + 1] == '2'


def test_public_and_constructed_tables_keep_split_metrics_and_full_denominators():
    from bench.study_metrics import evaluation_track_protocol, declared_queue_summary
    from profiling.final_study_index import full_queue_track_tables
    spec = dict(split_kind="independent_training_sources_dev_and_public_test",
        primary_metric="loong_reference_score", normalization=100, weight=.25,
        metric_by_split={"dev": {"primary_metric": "project_condition_version_score", "normalization": 1}})
    dev = evaluation_track_protocol(spec, "dev")
    public = evaluation_track_protocol(spec, "test")
    assert dev["origin"] == "constructed_task" and dev["normalization"] == 1
    assert public["origin"] == "public_benchmark" and public["normalization"] == 100
    rows = [dict(task_id="scored", status="completed", metrics={"project_condition_version_score": 1}, cost={}),
            dict(task_id="unscored", status="completed", metrics={}, cost={})]
    track = declared_queue_summary(rows, {**spec, **spec["metric_by_split"]["dev"]},
                                    ["scored", "unscored", "missing"])
    cell = dict(evaluation_kind="full_queue_own_P3", method="E1", concurrency=4,
                dataset_protocols={"loong": dev}, datasets={"loong": track})
    rendered = '\n'.join(full_queue_track_tables([cell]))
    assert "构造任务机制轨" in rendered and "公开基准数据轨" not in rendered
    assert "2/3 | 1/2" in rendered
    assert track["queue_primary_score"] is None
    assert track["queue_score_bounds"] == pytest.approx([1/3, 2/3])
    assert track["scored_subset_primary"] == 1
