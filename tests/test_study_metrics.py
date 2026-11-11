
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
    from bench.capacity_workflow import FULL_QUEUE_METHODS, OPTIONAL_FULL_QUEUE_METHODS, full_queue_method_specs
    from profiling.final_study_index import (full_queue_cells, full_queue_comparisons,
                                            full_queue_interactions, main, markdown)

    specs = full_queue_method_specs((*FULL_QUEUE_METHODS, *OPTIONAL_FULL_QUEUE_METHODS),
        {}, {'checkout': 'C'}, {'mode': 'value'}, rule_settings={},
        demand_policy={'allocation': 'paired_value', 'pricing_asset': 'pricing'})
    plan = dict(source_config={'path': 'A'}, source_matrix={'path': 'matrix'},
        source_arrival_plan={'path': 'arrivals-A'}, reader_profile_id='fixed', lambda_time=0,
        resource_scope={'gpu_devices': [0], 'gpu_capacity_tokens': 32768, 'host_capacity_tokens': 65536},
        assets={'gain': {'path': 'B'}, 'value': None},
        method_factors={spec['id']: spec['evaluation_factors'] for spec in specs},
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
    assert incomplete['differences']['p95_complete_ms'] is None
    assert incomplete['quality_not_lower'] is None
    assert incomplete['differences']['queue_quality'] == pytest.approx(.1)
    unscored = {**candidate, 'datasets': {'track': dict(missing_execution_task_ids=[],
        failed_task_ids=[], unscored_answer_task_ids=['task'])}}
    assert full_queue_comparisons(reference + cells(unscored))[0]['relative_p95_change'] is None
    assert full_queue_comparisons(reference + cells({**candidate, 'queue_quality': None}))[0]['quality_not_lower'] is None
    assert full_queue_comparisons(cells({**row, 'p95_complete_ms': 0}) + cells(candidate))[0]['relative_p95_change'] is None
    plan['source_arrival_plan']['path'] = 'arrivals-B'
    plan_path.write_text(json.dumps(plan))
    changed = full_queue_comparisons(reference + cells(candidate))[0]
    assert not changed['comparable'] and changed['differences']['queue_quality'] is None
    assert changed['incompatible_identity_fields'] == ['arrival_plan']
    assert changed['relative_p95_change'] is None
    plan['source_arrival_plan']['path'] = 'arrivals-A'
    plan['jobs'][0]['complete_slo_seconds'] = 60
    plan_path.write_text(json.dumps(plan))
    assert full_queue_comparisons(reference + cells(candidate))[0]['incompatible_identity_fields'] == ['complete_slo_seconds']
    plan['jobs'][0]['complete_slo_seconds'] = 30
    plan['execution_commit'] = 'new-release'
    plan_path.write_text(json.dumps(plan))
    assert full_queue_comparisons(reference + cells(candidate))[0]['incompatible_identity_fields'] == ['execution_commit']
    plan.pop('execution_commit')

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

    plan['compilation_policy'] = {'checkout': 'C'}
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
    repeated = [{**cell, 'replicate_scope': 'paired-queues', 'repetition': index}
                for index in range(2) for cell in quartet]
    assert [row['repetition'] for row in full_queue_interactions(repeated)] == [0, 1]
    assert len(full_queue_comparisons(repeated)) == 2 * len(full_queue_comparisons(quartet))
    quartet[-1]['full_queue_delivered'] = False
    assert full_queue_interactions(quartet)[0]['contrasts']['p95_complete_ms'] is None
    quartet[-1]['active_assets']['gain'] = 'different-B'
    assert full_queue_interactions(quartet) == []
    mismatched = next(item for item in full_queue_comparisons(quartet)
                     if item['candidate'] == 'S3_contextpilot' and item['baseline'] == 'S3_native')
    assert not mismatched['comparable']
    assert mismatched['incompatible_identity_fields'] == ['assets.gain']

    e5 = cells({**row, 'method': 'context_cache_order'})
    e6 = {**row, 'method': 'context_cache_demand'}
    rule = cells({**row, 'method': 'Rule_contextpilot_joint_demand'})
    plan['assets']['demand_pricing'] = {'path': 'pricing'}
    plan_path.write_text(json.dumps(plan))
    assert full_queue_comparisons(e5 + cells(e6))[0]['comparable']
    assert full_queue_comparisons(cells(e6) + rule)[0]['incompatible_identity_fields'] == ['assets.demand_pricing']
    plan['assets']['completion_timing'] = {'path': 'timing'}
    plan_path.write_text(json.dumps(plan))
    assert full_queue_comparisons(e5 + cells(e6))[0]['incompatible_identity_fields'] == ['assets.completion_timing']

    current = [cells({**row, 'method': method})[0] for method in (
        'RAG_native', 'LlamaIndex_native', 'context_native', 'context_cache_order',
        'Rule_contextpilot_joint_demand', 'context_no_fact_reuse', 'context_cache_demand')]
    comparisons = full_queue_comparisons(current)
    pairs = {(item['candidate'], item['baseline']) for item in comparisons}
    assert {('context_cache_demand', 'RAG_native'), ('context_cache_demand', 'LlamaIndex_native'),
            ('context_cache_order', 'context_native'), ('context_cache_demand', 'context_cache_order'),
            ('context_cache_demand', 'context_native'),
            ('context_cache_demand', 'Rule_contextpilot_joint_demand'),
            ('context_cache_demand', 'context_no_fact_reuse')} <= pairs
    assert ('Rule_contextpilot_joint_demand', 'context_cache_demand') not in pairs
    effects = {item['baseline']: (item['comparison_kind'], item['changed_factors'])
               for item in comparisons if item['candidate'] == 'context_cache_demand'}
    assert effects['Rule_contextpilot_joint_demand'] == ('single_factor', ['B'])
    assert effects['context_no_fact_reuse'] == ('single_factor', ['fact_reuse'])
    assert effects['context_cache_order'] == ('single_factor', ['retention'])
    assert effects['context_native'] == ('joint_factors', ['compilation', 'retention'])
    assert current[-1]['active_assets'] == {'gain': 'B', 'completion_timing': 'timing',
                                           'demand_pricing': 'pricing', 'compilation': 'C'}
    # The recorded strategy, rather than its display name, defines the comparison.
    renamed = [{**cell, 'method': 'renamed-'+cell['method']} for cell in current]
    assert {(item['candidate'].removeprefix('renamed-'), item['baseline'].removeprefix('renamed-'))
            for item in full_queue_comparisons(renamed)} == pairs
    assert full_queue_comparisons([current[-1], {**current[-1], 'method': 'same-factors'}]) == []

    plan['method_factors']['context_cache_demand'] = full_queue_method_specs(
        ('context_cache_demand',), {}, {'checkout': 'C'})[0]['evaluation_factors']
    plan_path.write_text(json.dumps(plan))
    next_use = cells({**row, 'method': 'context_cache_demand'})[0]
    assert 'demand_pricing' not in next_use['active_assets']
    next_use['method'] = 'next_use'
    allocation = full_queue_comparisons([next_use, current[-1]])[0]
    assert allocation['comparison_kind'] == 'single_factor'
    assert allocation['changed_factors'] == ['retention_allocation']


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


def test_full_queue_acceptance_allows_distinct_helpers_and_reports_slo_separately(tmp_path, monkeypatch):
    from copy import deepcopy
    from profiling import full_queue_acceptance as acceptance

    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text("{}")
    monkeypatch.setattr("bench.capacity_workflow.declared_execution_rows", lambda _: ([], ()))
    monkeypatch.setattr(acceptance, "declared_queue_quality_interval", lambda *_: {"interval": [0, 0]})
    limits = dict(gpu_capacity_tokens=32768, host_capacity_tokens=65536, max_online_tokens=10000)
    policy = dict(baseline="RAG_native", candidate="context_cache_demand", declared_qa=1,
        resource_limits=limits, p95_slo_ms=120000, quality_improvement=.05,
        quality_noninferiority=.002, p95_ratio=.85, seed=42, bootstrap_samples=20, confidence=.95)
    common = dict(concurrency=1, comparison_identity={"resource_scope": {
        "hardware": "same-device", "native_cache_config": "same-cache",
        "gpu_capacity_tokens": 32768, "host_capacity_tokens": 65536}},
        acceptance_protocol=policy, replicate_scope="queue", repetition=0, planned_repetitions=[0],
        requested_qa=1, completed_qa=1, unexpected_execution_task_ids=[],
        datasets={"loong": dict(declared=1, missing_execution_task_ids=[], unresolved_task_ids=[])},
        queue_quality=.6, queue_quality_bounds=[.6, .6], scorer_contracts={"loong": "official"},
        execution_status="completed", full_queue_delivered=True,
        reader_runtime=dict(fingerprint="fixed-Qwen", sglang_commit="fixed-engine"),
        online_occupancy=dict(hardware_only=True, physical_coverage=True, native_coverage=True,
            gpu_peak_gib=10, host_peak_gib=10, gpu_gib_seconds=100, host_gib_seconds=100),
        paid_native_usage={"online_token_ledger": dict(reasons=[], provider_phases=["answer"], total_tokens=1000),
            "native_work": {name: {"unknown_calls": 0} for name in
                            ("actual_prefill_tokens", "actual_host_loaded_tokens", "output_tokens")}},
        queue_job={"matrix": str(matrix_path)},
        required_experiment_cells={"baseline": "RAG_native", "candidate": "context_cache_demand"})

    def arm(method, latency, resident, assets):
        return {**deepcopy(common), "method": method, "generation_dir": method,
            "p95_complete_ms": latency, "active_assets": assets,
            "online_windows": {"loong": [{"resident_model_digest": resident,
                "engine_pool": dict(gpu_capacity_tokens=32768, host_capacity_tokens=65536)}]}}

    left = arm("RAG_native", 180000, "retriever-only", {})
    right = arm("context_cache_demand", 120000, "retriever-and-task-helper", {"gain": "B"})
    report = acceptance.evaluate_full_queue([left, right])[0]
    assert report["paired_queues"] == 1 and not report["pair_issues"]
    assert report["resource_budget_pass"] and report["measurement_complete"]
    assert report["cells"]["RAG_native"]["p95_slo"]["pass"] is False
    assert report["cells"]["context_cache_demand"]["p95_slo"]["pass"] is True
    assert report["p95_ratio"] == pytest.approx(2/3)
    assert report["latency_gate"] == "point_met_single_queue_descriptive"
    ablation = {**right, "required_experiment_cells": {"candidate": "context_cache_demand"}}
    assert acceptance.evaluate_full_queue([ablation]) == []
    missing = acceptance.evaluate_full_queue([right])[0]
    assert missing["pair_issues"][0]["reason"] == "missing_or_ambiguous_queue"
    assert missing["required_cells_measured"] == {"baseline": False, "candidate": True}
    split = acceptance.evaluate_full_queue([
        {**left, "required_experiment_cells": {"baseline": "RAG_native"}}, ablation])[0]
    assert split["paired_queues"] == 1
    assert split["required_cells_measured"] == {"baseline": True, "candidate": True}

    # Method-specific helpers may differ between arms, but remain fixed within an arm.
    repeats = [dict(deepcopy(cell), repetition=1, generation_dir=cell["generation_dir"]+"-r2")
               for cell in (left, right)]
    repeats[1]["online_windows"]["loong"][0]["resident_model_digest"] = "changed-task-helper"
    changed = acceptance.evaluate_full_queue([left, right, *repeats])[0]
    assert changed["paired_queues"] == 0
    assert changed["pair_issues"][-1]["reason"] == "reader_scorer_or_assets_differ_across_replicates"


def test_retention_validation_uses_online_length_scaling_without_scaling_full_window(tmp_path):
    import json
    from controller.prefix_retention import EmpiricalRetentionModel, PrefixForecast, TARGET_COST_RULE
    from profiling.final_study_index import retention_validation

    features = {"native_residency": "device"}
    forecast = PrefixForecast(15, 1, 100, 0, 0, 0, "scope", "measured",
        retention_control_ms=60, native_window_cost_ms=200, retained_window_cost_ms=175,
        source_token_range=(1000, 1000))
    key = EmpiricalRetentionModel._key(features, "scope", 15)
    model = EmpiricalRetentionModel({key: forecast}, {"native_residency": {"kind": "categorical"}},
                                   independent_samples=2)
    path = tmp_path / "retention.json"
    path.write_text(json.dumps({**model.to_dict(), "training_manifest": {"split": "train"}}))
    pairs = [dict(split="dev", training_eligible=True, fixed_decode_work=True,
        target_cost_rule=TARGET_COST_RULE, independent_session_ids=[str(tokens)],
        features=features, cache_scope_hash="scope", horizon_seconds=15,
        page_size=100, gpu_page_count=tokens//100, event_id=str(tokens),
        full_window_cost_difference_ms=actual) for tokens, actual in ((500, -5), (2000, 30))]
    report = retention_validation(["C="+str(path)], pairs)["models"][0]
    assert report["matched_pairs"] == 2 and report["prediction_coverage"] == 1
    assert report["within_training_length_pairs"] == 0 and report["outside_training_length_pairs"] == 2
    # 500 tokens: 50 - 60 < 0; 2000 tokens: 200 - 60 > 0. Control is fixed.
    assert report["selected_positive_predictions"] == 1
    assert report["paired_arm_acceptance_delta_ms"] == 15
    # Complete-window prediction remains the measured 25 ms for both lengths.
    assert report["weighted_mae_ms"] == 17.5


def test_retention_workflow_uses_selected_ttl_observation_window(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from profiling import retention_pair_workflow as workflow

    paths = {}
    for split in ("train", "dev"):
        paths[split] = tmp_path/(split+".json")
        paths[split].write_text(json.dumps({"training_eligible": True, "turns": [
            {"split": split, "session_id": split, "dataset": "fixture", "source_metadata": {}}]}))
    monkeypatch.setattr(workflow, "load_config", lambda path: SimpleNamespace(model=None))
    monkeypatch.setattr(workflow, "create_model_template", lambda model: None)
    monkeypatch.setattr(workflow, "verify_trace", lambda trace: None)
    monkeypatch.setattr(workflow, "audit_coverage", lambda trace, **kwargs: {})
    monkeypatch.setattr(workflow, "audit_trace_inputs", lambda trace, template, **kwargs: {})
    windows = []

    def pairs(trace, **kwargs):
        windows.append(kwargs["horizon_seconds"])
        return {}, {}

    monkeypatch.setattr(workflow, "fit_window_pairs", pairs)
    plan = workflow.main(["--train-trace", str(paths["train"]), "--dev-trace", str(paths["dev"]),
        "--config", "fixture", "--base-url", "http://fixture", "--gpu-device", "0",
        "--output-dir", str(tmp_path/"workflow"), "--ttl-seconds", "15", "--horizon-seconds", "15"])
    assert windows == [15, 15]
    assert plan["observation_seconds"] == 15 and plan["ttl_seconds"] == [15]
