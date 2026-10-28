from __future__ import annotations

import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest
from test_model_scope import local_scope as local_scope

from bench.experiments import generation_provenance, matrix_jobs, run_matrix, select_development

FIXTURE = Path(__file__).parent / "fixtures/mtrag_small.jsonl"


def completed_results(value):
    identity = generation_provenance(value)
    return [{**job, "status": "completed", "experiment_provenance": identity,
             "summary": {"metric_schema_version": 2, "measurement_kind": "simulation", "timing": {
                 "failure_fraction": 0, "turn_seconds": {"mean": 1}}, "cohort": {"unfinished_turns": 0}}}
            for job in matrix_jobs(value)]


def spec(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["session-a"], "test": ["session-b"]}))
    return {"schema_version": 1, "input": str(FIXTURE), "domain": "all", "split_manifest": str(split),
            "protocols": ["P2"], "concurrency": [1], "seeds": [42],
            "tuning": {"equal_budget": 2}, "methods": [
                {"id": "B3", "tune": True, "b6_eligible": True, "overrides": {"planner": {"strategy": "fixed_threshold"}},
                 "candidates": [{"planner": {"fixed_token_threshold": 1000}}, {"planner": {"fixed_token_threshold": 4000}}]},
                {"id": "full", "overrides": {"planner": {"strategy": "cost"}}}]}


def test_matrix_rejects_unequal_budget_and_leaking_split(tmp_path):
    value = spec(tmp_path)
    assert len(matrix_jobs(value)) == 3
    invalid = deepcopy(value)
    invalid["tuning"]["equal_budget"] = 3
    with pytest.raises(ValueError, match="equal_budget"):
        matrix_jobs(invalid)
    Path(value["split_manifest"]).write_text(json.dumps({"dev": ["session-a"], "test": ["session-a"]}))
    with pytest.raises(ValueError, match="disjoint"):
        matrix_jobs(value)


def test_development_only_check_needs_no_unused_test_cohort(tmp_path):
    value = spec(tmp_path)
    Path(value["split_manifest"]).write_text(json.dumps({"dev":["session-a"],"test":[]}))
    results = asyncio.run(run_matrix(value,tmp_path/"dev-only"))
    assert all(r["status"] == "completed" for r in results)
    assert all(set(r["summary"]["latency_by_session"]) == {"session-a"} for r in results)
    with pytest.raises(ValueError,match="requested test split contains no conversations"):
        matrix_jobs(value,split="test")
    # Overlapping or unknown IDs still fail before any workload runs.
    Path(value["split_manifest"]).write_text(json.dumps({"dev":["absent"],"test":[]}))
    with pytest.raises(ValueError,match="absent from the input"):
        matrix_jobs(value)


def test_local_tune_freeze_test_end_to_end(tmp_path):
    value = spec(tmp_path)
    results = asyncio.run(run_matrix(value, tmp_path / "dev"))
    assert [r["status"] for r in results] == ["completed"] * 3
    frozen = select_development(value, results)
    assert frozen["simulation_only"] and frozen["b6"] == "B3"
    assert frozen["quality_noninferiority_established"] is False
    tests = asyncio.run(run_matrix(value, tmp_path / "test", stage="test", frozen=frozen))
    assert {r["method"] for r in tests} == {"B3", "full"}
    assert next(r for r in tests if r["method"] == "B3")["method_aliases"] == ["B6"]
    assert all(r["status"] == "completed" for r in tests)
    assert all(set(r["summary"]["latency_by_session"]) == {"session-b"} for r in tests)
    with pytest.raises(ValueError, match="test results"):
        select_development(value, [{**r, "split": "test"} for r in results])
    changed = deepcopy(value)
    changed["seeds"] = [43]
    with pytest.raises(ValueError, match="frozen development"):
        asyncio.run(run_matrix(changed, tmp_path / "changed", stage="test", frozen=frozen))


def test_failure_cannot_win_by_dropping_hard_turns(tmp_path):
    value = spec(tmp_path)
    jobs = matrix_jobs(value)
    result = [{**job, "status": "completed", "summary": {"metric_schema_version": 2, "measurement_kind": "simulation",
              "timing": {"failure_fraction": .1 if job["candidate_index"] == 0 and job["method"] == "B3" else 0,
                         "turn_seconds": {"mean": 1 if job["candidate_index"] == 0 else 10}},
              "cohort": {"unfinished_turns": 0}}, "experiment_provenance": generation_provenance(value)} for job in jobs]
    selected = select_development(value, result)
    assert selected["winners"]["B3"]["candidate_index"] == 1


def test_development_generation_identity_and_later_offline_quality_gate(tmp_path):
    value = spec(tmp_path)
    results = asyncio.run(run_matrix(value, tmp_path / "dev-identity"))
    changed = deepcopy(value)
    changed["methods"][0]["candidates"][0]["planner"]["fixed_token_threshold"] = 9999
    with pytest.raises(ValueError, match="different or unrecorded"):
        select_development(changed, results)
    gated = deepcopy(value)
    gated["tuning"]["quality_eligible"] = {"B3": {"0": False, "1": True}, "full": {"0": True}}
    frozen = select_development(gated, results)
    assert frozen["quality_gate_applied"]
    assert frozen["winners"]["B3"]["candidate_index"] == 1




def test_documentation_changes_do_not_invalidate_generation_identity(tmp_path, monkeypatch):
    import bench.experiments as experiments
    value = spec(tmp_path)
    results = completed_results(value)
    original = Path.read_bytes
    root = Path(experiments.__file__).resolve().parents[1]
    def docs_changed(path):
        return original(path)+(b"\nnew research notes\n" if path.resolve().is_relative_to(root/"docs") else b"")
    monkeypatch.setattr(Path, "read_bytes", docs_changed)
    assert select_development(value, results)["winners"]
    paths = results[0]["experiment_provenance"]["implementation"]["files"]
    assert not any(p.startswith(("docs/", "tests/")) or p == "README.md" for p in paths)




def test_legacy_unbound_source_results_cannot_be_frozen_as_current_implementation(tmp_path):
    value = spec(tmp_path)
    results = completed_results(value)
    for result in results:
        result["experiment_provenance"] = {k: v for k, v in result["experiment_provenance"].items() if k != "implementation"}
    with pytest.raises(ValueError, match="legacy unbound"):
        select_development(value, results)


def test_matrix_reads_one_source_snapshot_and_reuses_existing_controller_run_manifests(tmp_path, monkeypatch):
    import bench.experiments as experiments
    value = spec(tmp_path)
    calls = []
    original = experiments.implementation_manifest
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(experiments, "implementation_manifest", counted)
    results = asyncio.run(run_matrix(value, tmp_path/"one-source-read"))
    assert len(calls) == 1  # three jobs, no extra implementation hash per job
    recorded = json.loads((tmp_path/"one-source-read/implementation-manifest.json").read_text())
    assert all(r["experiment_provenance"]["implementation"] == recorded for r in results)
    select_development(value, results)
    assert len(calls) == 2  # one new snapshot for the whole selection boundary


def test_changed_method_revision_requires_new_generation_before_test(tmp_path, monkeypatch):
    value = spec(tmp_path)
    frozen = select_development(value, completed_results(value))
    value["methods"][0]["revision"] = "changed-execution"
    async def forbidden(*args):
        pytest.fail("mismatched implementation must not start heldout execution")
    with pytest.raises(ValueError, match="frozen development.*configuration"):
        asyncio.run(run_matrix(value, tmp_path/"mismatched-test", stage="test", frozen=frozen, executor=forbidden))
    assert not (tmp_path/"mismatched-test").exists()


def test_explicit_setup_failure_is_reported_without_running_generation(tmp_path):
    value = spec(tmp_path)
    value["lifecycle"] = {"cache_state_protocol": "dedicated_server_reset",
                          "before_job": [sys.executable, "-c", "print('reset failed'); raise SystemExit(3)"]}
    results = asyncio.run(run_matrix(value, tmp_path / "setup-failure"))
    assert all(r["status"] == "failed" for r in results)
    assert all(r["before_job"]["returncode"] == 3 for r in results)
    assert all("reset failed" in r["before_job"]["stdout"] for r in results)
    assert not list((tmp_path / "setup-failure").rglob("benchmark-*.json"))


def test_output_length_artifact_contents_bind_generation_identity(tmp_path):
    value = spec(tmp_path)
    artifact = tmp_path / "length.json"
    artifact.write_text('{"model":"first"}')
    value["methods"][1]["overrides"]["cost_model"] = {"output_length_artifact": str(artifact)}
    before = generation_provenance(value)
    artifact.write_text('{"model":"second"}')
    after = generation_provenance(value)
    assert before["external_inputs"][str(artifact)] != after["external_inputs"][str(artifact)]


def test_ablations_inherit_frozen_parent_and_do_not_consume_tuning_budget(tmp_path):
    value = spec(tmp_path)
    value["methods"][0]["overrides"]["planner"]["redundancy_threshold"] = .25
    value["methods"].append({"id": "B3_no_retire", "ablation_of": "B3",
                             "overrides": {"ablation": {"epoch_retirement": False}}})
    assert all(job["method"] != "B3_no_retire" for job in matrix_jobs(value))
    results = asyncio.run(run_matrix(value, tmp_path / "parent-dev"))
    value["tuning"]["quality_eligible"] = {"B3": {"0": False, "1": True}, "full": {"0": True}}
    frozen = select_development(value, results)
    assert "B3_no_retire" not in frozen["winners"]
    test_jobs = matrix_jobs(value, split="test", frozen=frozen)
    parent = next(j for j in test_jobs if j["method"] == "B3")
    child = next(j for j in test_jobs if j["method"] == "B3_no_retire")
    assert child["candidate_index"] == parent["candidate_index"] == 1
    assert child["overrides"]["planner"] == parent["overrides"]["planner"]
    assert child["overrides"]["planner"]["fixed_token_threshold"] == 4000
    assert child["overrides"]["planner"]["redundancy_threshold"] == .25
    assert child["overrides"]["ablation"]["epoch_retirement"] is False
    ablation_results = asyncio.run(run_matrix(value, tmp_path / "matched-dev", stage="dev_ablation", frozen=frozen))
    assert len(ablation_results) == 1 and ablation_results[0]["status"] == "completed"
    assert ablation_results[0]["method"] == "B3_no_retire" and ablation_results[0]["split"] == "dev"
    with pytest.raises(ValueError, match="complete matrix"):
        select_development(value, results + ablation_results)


@pytest.mark.parametrize("prompt_source",["base_config","inline","file"])
def test_matrix_shares_prompt_and_default_budget_between_native_and_controller(tmp_path,monkeypatch,prompt_source,local_scope):
    import bench.experiments as experiments
    import bench.native_baseline as native_module
    import controller.cli as cli
    import yaml
    value = spec(tmp_path)
    base = tmp_path/"base.yaml"
    base.write_text(yaml.safe_dump({"model":{"scope_path":str(local_scope.path)},"context":{"system_prompt":"base prompt", "max_new_tokens":1024,
        "max_total_model_tokens":8192,
        "allow_version_comparison":True, "reference_prefix_length":6}}))
    value["base_config"] = str(base)
    expected = "base prompt"
    if prompt_source == "inline":
        expected = "  Shared inline prompt. \n"
        value["system_prompt"] = expected
    elif prompt_source == "file":
        raw = b"  Shared file prompt. \r\n"
        prompt_file = tmp_path/"prompt.txt"
        prompt_file.write_bytes(raw)
        expected = raw.decode()
        value["system_prompt_file"] = str(prompt_file)
    manifest = tmp_path/"engine.json"
    manifest.write_text("{}")
    value["methods"] = [
        {"id":"B0","kind":"native","native":{"baseline":"B0","base_url":"http://fake.test",
            "engine_manifest":str(manifest),"model_path":str(local_scope.model_path)}},
        {"id":"full","overrides":{"planner":{"strategy":"cost"}}}]
    seen = {}
    async def native_run(args):
        seen["native"] = args
        (Path(args.output)/"native-fake.json").write_text(json.dumps({"baseline":"B0",
            "engine_manifest_sha256": sha256(Path(args.engine_manifest).read_bytes()).hexdigest()}))
    async def controller_run(args,config):
        seen["controller"] = config
        (Path(config.state_dir)/"benchmark-fake.json").write_text(json.dumps({"engine_kind":"mock"}))
    monkeypatch.setattr(native_module,"_run",native_run)
    monkeypatch.setattr(cli,"benchmark",controller_run)
    monkeypatch.setattr(experiments,"summarize_run",lambda payload:{"measurement_kind":"simulation"})
    monkeypatch.setattr(experiments,"markdown_report",lambda summary:"fake contract report")
    async def scenario():
        for job in matrix_jobs(value):
            await experiments.execute_job(value,job,tmp_path/job["job_id"])
    asyncio.run(scenario())
    assert seen["native"].system_prompt == seen["controller"].context.system_prompt == expected
    assert seen["native"].system_prompt_file is None
    assert seen["native"].max_new_tokens == seen["controller"].context.max_new_tokens == 1024
    assert seen["native"].context_length == seen["controller"].context.max_total_model_tokens == 8192
    assert seen["native"].model_path == str(local_scope.model_path)
    assert seen["native"].context_policy == seen["controller"].context
    assert native_module.resolve_context_policy(seen["native"]) == seen["controller"].context


@pytest.mark.parametrize("field,value", [("allow_version_comparison", True), ("reference_prefix_length", 6),
    ("max_total_model_tokens", 16384), ("max_new_tokens", 1024)])
def test_matrix_rejects_method_specific_required_policy_or_budget(tmp_path, field, value):
    matrix = spec(tmp_path)
    matrix["methods"][0]["overrides"]["context"] = {field:value}
    with pytest.raises(ValueError, match="shared matrix policy"):
        matrix_jobs(matrix)


def test_matrix_normalizes_shared_explicit_keys_but_allows_layout_and_annotation_studies(tmp_path):
    from dataclasses import asdict
    from controller.evidence import make_evidence
    import yaml
    key = asdict(make_evidence("fixed source", "one").key)
    base = tmp_path/"shared-context.yaml"
    base.write_text(yaml.safe_dump({"context":{"explicit_keys":[key]}}))
    matrix = spec(tmp_path)
    matrix["base_config"] = str(base)
    matrix["methods"][0]["overrides"]["context"] = {
        "explicit_keys":[key], "layout":"contextpilot", "table_relations":True}
    assert len(matrix_jobs(matrix)) == 3
    matrix["methods"][0]["overrides"]["context"]["explicit_keys"] = []
    with pytest.raises(ValueError, match="context.explicit_keys"):
        matrix_jobs(matrix)


@pytest.mark.parametrize("key,value", [("context_length", 8192), ("max_new_tokens", 128)])
def test_matrix_native_budget_must_match_common_policy(tmp_path, key, value, local_scope):
    matrix = spec(tmp_path)
    path=tmp_path/'shared.json'
    path.write_text(json.dumps({'model':{'scope_path':str(local_scope.path)}}))
    matrix['base_config']=str(path)
    matrix["methods"] = [{"id":"B0", "kind":"native", "native":{key:value}}]
    with pytest.raises(ValueError, match="shared matrix budget"):
        matrix_jobs(matrix)


@pytest.mark.parametrize("response_mode,slo,key", [
    ("streaming", {"ttft_seconds":.5, "tpot_seconds":.02}, "ttft_seconds"),
    ("buffered", {"complete_seconds":3.0}, "complete_seconds"),
])
def test_matrix_persists_explicit_slo_and_freezes_it_with_selection(tmp_path, response_mode, slo, key):
    matrix = spec(tmp_path)
    matrix["response_mode"] = response_mode
    matrix["slo"] = slo
    rows = asyncio.run(run_matrix(matrix, tmp_path/"slo-dev"))
    assert all(r["summary"]["timing"]["slo"] == matrix["slo"] for r in rows)
    from bench.report import summarize_run
    for row in rows:
        recorded = json.loads(Path(row["raw_output"]).read_text())
        assert recorded["experiment_protocol"]["response_mode"] == response_mode
        assert summarize_run(recorded)["timing"]["slo"] == slo
    frozen = select_development(matrix, rows)
    matrix["slo"][key] += .3
    with pytest.raises(ValueError, match="frozen development"):
        asyncio.run(run_matrix(matrix, tmp_path/"slo-test", stage="test", frozen=frozen))


@pytest.mark.parametrize("slo", [{"ttft_seconds":float("nan")}, {"tpot_seconds":-1}, {"complete_seconds":-1}, {"ttft_ms":500}])
def test_matrix_rejects_invalid_slo_before_generation(tmp_path, slo):
    matrix = spec(tmp_path)
    matrix["slo"] = slo
    with pytest.raises(ValueError, match="matrix slo"):
        matrix_jobs(matrix)


def test_matrix_prompt_provenance_changes_with_file_content_and_rejects_method_mismatch(tmp_path):
    value = spec(tmp_path)
    prompt = tmp_path/"prompt.txt"
    prompt.write_text("first prompt\n")
    value["system_prompt_file"] = str(prompt)
    before = generation_provenance(value)
    prompt.write_text("second prompt\n")
    after = generation_provenance(value)
    assert before["base_config_sha256"] != after["base_config_sha256"]
    assert before["external_inputs"][str(prompt)] != after["external_inputs"][str(prompt)]
    invalid = deepcopy(value)
    invalid["methods"][0]["overrides"]["context"] = {"system_prompt":"different method prompt"}
    with pytest.raises(ValueError,match="method-specific"):
        matrix_jobs(invalid)
    invalid = deepcopy(value)
    invalid["methods"][0]["native"] = {"system_prompt":"native-only prompt"}
    with pytest.raises(ValueError,match="must be shared"):
        matrix_jobs(invalid)
    invalid = deepcopy(value)
    invalid["system_prompt"] = "conflicting inline prompt"
    with pytest.raises(ValueError,match="mutually exclusive"):
        matrix_jobs(invalid)


@pytest.mark.parametrize("explicit_root",[False,True])
def test_matrix_default_and_explicit_default_prompt_are_equivalent(tmp_path,explicit_root):
    from controller.templates import SYSTEM_PROMPT
    value = spec(tmp_path)
    if explicit_root:
        value["system_prompt"] = SYSTEM_PROMPT
    value["methods"][0]["overrides"]["context"] = {
        "system_prompt":None if explicit_root else SYSTEM_PROMPT}
    assert len(matrix_jobs(value)) == 3


@pytest.mark.parametrize("system_comparison", [False, True])
def test_matrix_shared_retriever_config_binds_assets_and_both_execution_paths(tmp_path, monkeypatch, local_scope, system_comparison):
    import bench.experiments as experiments
    import bench.native_baseline as native_module
    import controller.cli as cli
    value = spec(tmp_path)
    base=tmp_path/'shared.json'
    base.write_text(json.dumps({'model':{'scope_path':str(local_scope.path)}}))
    value['base_config']=str(base)
    config = tmp_path/"retriever.json"
    for name in ("sparse", "dense"):
        (tmp_path/name).mkdir()
        (tmp_path/name/"manifest.json").write_text('{"fixture":1}')
    config_body = {"sparse_index":str(tmp_path/"sparse"), "dense_index":str(tmp_path/"dense"),
                   "encoder_path":"encoder", "reranker_path":"reranker", "query_mode":"current"}
    config.write_text(json.dumps(config_body))
    engine = tmp_path/"engine.json"
    engine.write_text("{}")
    value.update(protocols=["P3"], retriever_config=str(config), methods=[
        {"id":"B0","kind":"native","native":{"baseline":"B0","base_url":"http://fake.test",
            "engine_manifest":str(engine)}}, {"id":"full"}])
    alternate = tmp_path/"alternate.json"
    if system_comparison:
        alternate.write_text(json.dumps({**config_body,"query_mode":"last3"}))
        value['comparison_scope']='system'
        value['methods'][1]['retriever_config']=str(alternate)
    before = generation_provenance(value)
    if system_comparison:
        assert str(alternate) in before['external_inputs']
    assert {str(config), str(tmp_path/"sparse/manifest.json"), str(tmp_path/"dense/manifest.json")} <= before["external_inputs"].keys()
    config_body["query_mode"] = "last3"
    config.write_text(json.dumps(config_body))
    assert generation_provenance(value)["external_inputs"][str(config)] != before["external_inputs"][str(config)]
    seen = []
    async def native_run(args):
        seen.append(args.retriever_config)
        (Path(args.output)/"native-fake.json").write_text(json.dumps({"baseline":"B0",
            "engine_manifest_sha256": sha256(Path(args.engine_manifest).read_bytes()).hexdigest()}))
    async def controller_run(args, config):
        seen.append(args.retriever_config)
        (Path(config.state_dir)/"benchmark-fake.json").write_text(json.dumps({"engine_kind":"mock"}))
    monkeypatch.setattr(native_module, "_run", native_run)
    monkeypatch.setattr(cli, "benchmark", controller_run)
    monkeypatch.setattr(experiments, "summarize_run", lambda payload:{"measurement_kind":"simulation"})
    monkeypatch.setattr(experiments, "markdown_report", lambda summary:"fake contract report")
    async def scenario():
        for job in matrix_jobs(value):
            await experiments.execute_job(value, job, tmp_path/job["job_id"])
    asyncio.run(scenario())
    assert seen == [str(config), str(alternate if system_comparison else config)]
    value["methods"][0]["native"]["retriever_config"] = str(config)
    with pytest.raises(ValueError, match="matrix root"):
        matrix_jobs(value)


def test_runtime_native_manifest_is_supplied_only_after_owned_server_validation(tmp_path, monkeypatch, local_scope):
    from bench import experiments, native_baseline
    from bench import native_metrics
    value = spec(tmp_path)
    base=tmp_path/'shared.json'
    base.write_text(json.dumps({'model':{'scope_path':str(local_scope.path)}}))
    value['base_config']=str(base)
    value['server_profiles'] = {'stock': {'kind': 'stock', 'write_policy': 'write_back',
                                        'gpu_tokens': 16384, 'hicache': True}}
    value['methods'] = [{'id': 'B1', 'kind': 'native', 'server_profile': 'stock',
        'native': {'baseline': 'B1', 'base_url': 'http://fake.test', 'audit_prompt_ids': True}}]
    path = tmp_path/'observed-native.json'
    path.write_text('{"actual_runtime_fixture":true}')
    seen = []
    sampling = []
    class FakeSampler:
        def __init__(self, *args, **kwargs):
            sampling.append(kwargs)
        async def start(self):
            sampling.append('start')
        async def stop(self):
            sampling.append('stop')
        def observation(self):
            return {'samples': [], 'errors': [], 'source': 'fake_transport'}
    monkeypatch.setattr(native_metrics, 'NativePoolSampler', FakeSampler)
    async def generate(args):
        seen.append(args.engine_manifest)
        (Path(args.output)/'native-fixture.json').write_text(json.dumps({'baseline': 'B1',
            'engine_manifest_sha256': sha256(Path(args.engine_manifest).read_bytes()).hexdigest()}))
    monkeypatch.setattr(native_baseline, '_run', generate)
    monkeypatch.setattr(experiments, 'summarize_run', lambda value: {'measurement_kind': 'fake_contract'})
    monkeypatch.setattr(experiments, 'markdown_report', lambda value: '')
    job = matrix_jobs(value)[0]
    record = {'status': 'running', 'server_profile': 'stock', 'profile': value['server_profiles']['stock']}
    result = asyncio.run(experiments.execute_job(value, job, tmp_path/'actual',
        native_engine_manifest=path, verified_server=record))
    assert sampling[0]['gpu_capacity'] == 16384 and sampling[0]['initial_empty'] is False
    assert sampling[1:] == ['start', 'stop']
    assert seen == [str(path)]
    assert result['native_engine_manifest'] == {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}
    job['native']['engine_manifest'] = str(tmp_path/'another-declared.json')
    with pytest.raises(ValueError, match='cannot replace a declared'):
        asyncio.run(experiments.execute_job(value, job, tmp_path/'wrong',
            native_engine_manifest=path, verified_server=record))
    assert len(seen) == 1


def test_predeclared_fixed_test_needs_no_artificial_development_run(tmp_path):
    value=spec(tmp_path)
    value['methods']=[{'id':'fixed','overrides':{'planner':{'strategy':'always_rebuild'}}}]
    Path(value['split_manifest']).write_text(json.dumps({'dev':[],'test':['session-b']}))
    results=asyncio.run(run_matrix(value,tmp_path/'fixed-test',stage='test'))
    assert len(results)==1 and results[0]['status']=='completed'
    assert set(results[0]['summary']['latency_by_session'])=={'session-b'}
    from bench.b6_joint_sweep import frozen_jobs
    assert frozen_jobs(value,'test',None)==matrix_jobs(value,split='test')


def test_system_comparison_binds_method_retrieval_and_execution_inherits_shared(tmp_path, monkeypatch):
    from bench import experiments
    value=spec(tmp_path)
    value.update(comparison_scope='system',protocols=['P3'],methods=[{'id':'baseline'},
        {'id':'complete','retriever_config':'different.yaml'}],retriever_config='shared.yaml')
    monkeypatch.setattr(experiments,'retriever_config_sources',lambda path:())
    jobs=matrix_jobs(value)
    assert [j['retrieval']['retriever_config'] for j in jobs]==['shared.yaml','different.yaml']
    assert jobs[1]['overrides']=={}
    value['comparison_scope']='execution'
    with pytest.raises(ValueError,match='comparison_scope=system'):
        matrix_jobs(value)


@pytest.mark.parametrize("relative", ["bench/report.py", "adapters/hybrid_retrieval.py"])
def test_display_and_disabled_branches_do_not_block_freeze_or_test(tmp_path, monkeypatch, relative):
    import bench.experiments as experiments
    value = spec(tmp_path)
    results = completed_results(value)
    frozen = select_development(value, results)
    original = Path.read_bytes
    target = Path(experiments.__file__).resolve().parents[1] / relative
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p) + (b" changed" if p == target else b""))
    assert select_development(value, results)["winners"]
    observed = []
    async def execute(spec, job, directory):
        observed.append(job["job_id"])
        return {**job, "status": "failed"}  # No model or unrelated summary fixture.
    asyncio.run(run_matrix(value, tmp_path/"test-display", stage="test", frozen=frozen, executor=execute))
    assert observed


def test_metric_change_reuses_bound_raw_answers_and_requires_refreezing(tmp_path, monkeypatch):
    import bench.experiments as experiments
    value = spec(tmp_path)
    results = asyncio.run(run_matrix(value, tmp_path/"metric-dev"))
    frozen = select_development(value, results)
    originals = {r["raw_output"]: Path(r["raw_output"]).read_bytes() for r in results}
    import bench.report as report
    monkeypatch.setattr(report, "METRIC_SCHEMA_VERSION", report.METRIC_SCHEMA_VERSION + 1)
    monkeypatch.setattr(experiments, "METRIC_SCHEMA_VERSION", report.METRIC_SCHEMA_VERSION)
    with pytest.raises(ValueError, match="recompute metrics and refreeze"):
        asyncio.run(run_matrix(value, tmp_path/"stale-metrics", stage="test", frozen=frozen))
    updated = select_development(value, results)
    assert updated["provenance"]["metric_schema_version"] == report.METRIC_SCHEMA_VERSION
    assert all(Path(path).read_bytes() == raw for path, raw in originals.items())
    assert updated["winners"] == frozen["winners"]
