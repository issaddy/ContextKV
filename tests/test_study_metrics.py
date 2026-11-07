
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


def test_registered_joint_reference_survives_freeze_and_report(tmp_path):
    import json
    from bench.study import freeze, report

    spec = matrix()
    spec["methods"] = [{"id": "E0_native"}, {"id": "S3_value"}]
    spec["selection"].update(baseline="E0_native", test_comparators=["E0_native"])
    spec["cohorts"]["test"] = spec["cohorts"]["dev"]
    config = {"study": {"learning": True}, "task_model": {"mode": "teacher"},
              "support_model": {"mode": "teacher"}}
    rows = records("E0_native", long_score=70) + records("S3_value", long_score=90)
    for row in rows:
        row["response"] = {"action": "REBUILD", "read_count": 1,
                           "reduce_count": 0, "truncated": False}
    frozen = freeze(config, spec, rows, tmp_path/"frozen.json")
    assert [method["id"] for method in frozen["test_methods"]] == ["E0_native", "S3_value"]
    report(frozen, [{**row, "split": "test"} for row in rows], tmp_path/"report")
    paired = json.loads((tmp_path/"report"/"paired_intervals.json").read_text())
    assert paired and all(row["baseline"] == "E0_native" for row in paired)
    assert {row["dataset"]: row["paired_task_ids"] for row in paired
            if row["dataset"] in spec["datasets"]} == spec["cohorts"]["test"]


def test_final_index_compares_fixed_arrivals_and_declared_slo(tmp_path):
    import json
    from profiling.final_study_index import (full_queue_cells, full_queue_comparisons,
                                            full_queue_interactions, main, markdown)

    run = tmp_path / 'reader.json'
    run.write_text(json.dumps({'engine': {'base_url': 'local'}, 'state_dir': 'temporary'}))
    plan = dict(reader_run={'path': str(run)},
        source_config={'sha256': 'A'}, source_matrix={'sha256': 'matrix'},
        cohort={'track': ['task']}, reader_profile_id='fixed', lambda_time=0,
        resource_scope={'gpu_devices': [0], 'gpu_capacity_tokens': 32768, 'host_capacity_tokens': 65536},
        workload_plans={'4': {'track': {'sha256': 'arrivals-A'}}},
        assets={'gain': {'sha256': 'B'}, 'value': None},
        jobs=[dict(name='job', output='generation', complete_slo_seconds=30)])
    plan_path = tmp_path / 'capacity-plan.json'
    plan_path.write_text(json.dumps(plan))
    row = dict(job='job', method='B6_native', concurrency=4, requested_qa=1, completed_qa=1,
        queue_quality=.5, p95_complete_ms=100, paid_native_usage={},
        native_cache_config={'gpu_capacity_tokens': 32768, 'host_capacity_tokens': 65536, 'page_size': 16},
        runtime_hardware={'host_name': 'fixture-host', 'platform': 'fixture-platform',
            'gpus': [{'index': 0, 'uuid': 'fixture-device', 'name': 'H200',
                      'memory_bytes': 141*1024**3, 'driver_version': 'fixture-driver'}]},
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
    other_hardware = {**row['runtime_hardware'], 'gpus': [
        {**row['runtime_hardware']['gpus'][0], 'name': 'A100'}]}
    changed_device = full_queue_comparisons(reference + cells({**candidate, 'runtime_hardware': other_hardware}))[0]
    assert not changed_device['comparable'] and changed_device['relative_p95_change'] is None
    assert changed_device['incompatible_identity_fields'] == ['resource_scope']
    unknown = full_queue_comparisons(reference + cells({**candidate, 'runtime_hardware': None}))[0]
    assert not unknown['comparable'] and unknown['missing_identity_fields'] == ['resource_scope.hardware']
    plan['resource_scope']['host_capacity_tokens'] = 131072
    plan_path.write_text(json.dumps(plan))
    assert full_queue_comparisons(reference + cells(candidate))[0]['incompatible_identity_fields'] == ['resource_scope']
    plan['resource_scope']['host_capacity_tokens'] = 65536
    plan_path.write_text(json.dumps(plan))
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
    assert '质量差（候选−基线）' in rendered and '预登记的完整队列验收' in rendered
    command = json.loads((out / 'cost-ledger-command.json').read_text())
    assert not command['executed']
    assert command['argv'][command['argv'].index('--gpu-devices') + 1] == '2'

    plan['compilation_asset'] = {'sha256': 'C'}
    plan_path.write_text(json.dumps(plan))
    quartet = [cells({**row, 'method': method, 'queue_quality': quality, 'p95_complete_ms': latency})[0]
        for method, quality, latency in (('B6_native', .5, 100), ('S3_native', .6, 90),
                                        ('B6_contextpilot', .55, 95), ('S3_contextpilot', .7, 80))]
    effects = full_queue_interactions(quartet)[0]['contrasts']
    assert effects['queue_quality']['interaction'] == pytest.approx(.05)
    assert effects['queue_quality']['B_contextpilot'] == pytest.approx(.15)
    assert effects['p95_complete_ms']['interaction'] == -5
    assert {item['baseline'] for item in full_queue_comparisons(quartet)
            if item['candidate'] == 'S3_contextpilot'} == {'B6_native', 'S3_native', 'B6_contextpilot'}
    assert full_queue_interactions(quartet[:3]) == []
    quartet[-1]['full_queue_delivered'] = False
    assert full_queue_interactions(quartet)[0]['contrasts']['p95_complete_ms'] is None
    quartet[-1]['active_assets']['gain'] = {'sha256': 'different-B'}
    assert full_queue_interactions(quartet) == []
    mismatched = next(item for item in full_queue_comparisons(quartet)
                     if item['candidate'] == 'S3_contextpilot' and item['baseline'] == 'S3_native')
    assert not mismatched['comparable']
    assert mismatched['incompatible_identity_fields'] == ['assets.gain']


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
    assert track["queue_score_bounds"] == pytest.approx([1/3, 1])
    assert track["scored_subset_primary"] == 1
