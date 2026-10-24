from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from bench.experiments import generation_provenance, matrix_jobs, run_matrix, select_development

FIXTURE = Path(__file__).parent / "fixtures/mtrag_small.jsonl"


def completed_results(value):
    identity = generation_provenance(value)
    return [{**job, "status": "completed", "experiment_provenance": identity,
             "summary": {"measurement_kind": "simulation", "timing": {
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


def test_local_tune_freeze_test_end_to_end(tmp_path):
    value = spec(tmp_path)
    results = asyncio.run(run_matrix(value, tmp_path / "dev"))
    assert [r["status"] for r in results] == ["completed"] * 3
    frozen = select_development(value, results)
    assert frozen["simulation_only"] and frozen["b6"] == "B3"
    assert frozen["quality_noninferiority_established"] is False
    tests = asyncio.run(run_matrix(value, tmp_path / "test", stage="test", frozen=frozen))
    assert {r["method"] for r in tests} == {"B3", "B6", "full"}
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
    result = [{**job, "status": "completed", "summary": {"measurement_kind": "simulation",
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


@pytest.mark.parametrize("relative", ["adapters/hybrid_retrieval.py", "adapters/query_views.py",
    "controller/planner.py", "bench/run_sessions.py", "bench/report.py", "adapters/assets/convsearch_r1/prompt.txt"])
def test_freeze_rejects_old_results_after_execution_source_or_prompt_change(tmp_path, monkeypatch, relative):
    import bench.experiments as experiments
    value = spec(tmp_path)
    results = completed_results(value)
    target = Path(experiments.__file__).resolve().parents[1]/relative
    original = Path.read_bytes
    # Observe changed file bytes without mutating shared repository files.
    def changed(path):
        return original(path)+(b"\nchanged implementation\n" if path.resolve() == target else b"")
    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="implementation"):
        select_development(value, results)


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
    assert all(r["controller_source_sha256"] == recorded["controller_source_sha256"] for r in results)
    assert all(r["experiment_provenance"]["implementation"] == recorded for r in results)
    select_development(value, results)
    assert len(calls) == 2  # one new snapshot for the whole selection boundary
    results[0]["controller_source_sha256"] = "different-controller-run"
    with pytest.raises(ValueError, match="run source manifest"):
        select_development(value, results)


def test_frozen_test_is_rejected_before_executor_after_adapter_change(tmp_path, monkeypatch):
    import bench.experiments as experiments
    value = spec(tmp_path)
    frozen = select_development(value, completed_results(value))
    original = experiments.implementation_manifest
    def changed():
        result = original()
        result["files"]["adapters/hybrid_retrieval.py"] = "new-adapter-version"
        return result
    monkeypatch.setattr(experiments, "implementation_manifest", changed)
    async def forbidden(*args):
        pytest.fail("mismatched implementation must not start heldout execution")
    with pytest.raises(ValueError, match="frozen development.*implementation"):
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
    value["methods"][0]["overrides"]["planner"]["delta_rel"] = .25
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
    assert child["overrides"]["planner"]["delta_rel"] == .25
    assert child["overrides"]["ablation"]["epoch_retirement"] is False
    ablation_results = asyncio.run(run_matrix(value, tmp_path / "matched-dev", stage="dev_ablation", frozen=frozen))
    assert len(ablation_results) == 1 and ablation_results[0]["status"] == "completed"
    assert ablation_results[0]["method"] == "B3_no_retire" and ablation_results[0]["split"] == "dev"
    with pytest.raises(ValueError, match="complete matrix"):
        select_development(value, results + ablation_results)


@pytest.mark.parametrize("prompt_source",["base_config","inline","file"])
def test_matrix_shares_prompt_and_default_budget_between_native_and_controller(tmp_path,monkeypatch,prompt_source):
    import bench.experiments as experiments
    import bench.native_baseline as native_module
    import controller.cli as cli
    import yaml
    value = spec(tmp_path)
    base = tmp_path/"base.yaml"
    base.write_text(yaml.safe_dump({"context":{"system_prompt":"base prompt", "max_new_tokens":1024,
        "max_total_model_tokens":8192, "revisit_mode":"explicit", "revisit_history_turns":6,
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
            "engine_manifest":str(manifest),"model_path":"/pinned/model-snapshot"}},
        {"id":"full","overrides":{"planner":{"strategy":"cost"}}}]
    seen = {}
    async def native_run(args):
        seen["native"] = args
        (Path(args.output)/"native-fake.json").write_text(json.dumps({"baseline":"B0"}))
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
    assert seen["native"].model_path == "/pinned/model-snapshot"
    assert seen["native"].context_policy == seen["controller"].context
    assert native_module.resolve_context_policy(seen["native"]) == seen["controller"].context


@pytest.mark.parametrize("field,value", [("revisit_mode", "explicit"), ("revisit_history_turns", 6),
    ("allow_version_comparison", True), ("reference_prefix_length", 6),
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
def test_matrix_native_budget_must_match_common_policy(tmp_path, key, value):
    matrix = spec(tmp_path)
    matrix["methods"] = [{"id":"B0", "kind":"native", "native":{key:value}}]
    with pytest.raises(ValueError, match="shared matrix budget"):
        matrix_jobs(matrix)


def test_matrix_persists_explicit_slo_and_freezes_it_with_selection(tmp_path):
    matrix = spec(tmp_path)
    matrix["slo"] = {"ttft_seconds":.5, "tpot_seconds":.02}
    rows = asyncio.run(run_matrix(matrix, tmp_path/"slo-dev"))
    assert all(r["summary"]["timing"]["slo"] == matrix["slo"] for r in rows)
    frozen = select_development(matrix, rows)
    matrix["slo"]["ttft_seconds"] = .8
    with pytest.raises(ValueError, match="frozen development"):
        asyncio.run(run_matrix(matrix, tmp_path/"slo-test", stage="test", frozen=frozen))


@pytest.mark.parametrize("slo", [{"ttft_seconds":float("nan")}, {"tpot_seconds":-1}, {"ttft_ms":500}])
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


def test_matrix_shared_retriever_config_binds_assets_and_both_execution_paths(tmp_path, monkeypatch):
    import bench.experiments as experiments
    import bench.native_baseline as native_module
    import controller.cli as cli
    value = spec(tmp_path)
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
    before = generation_provenance(value)
    assert {str(config), str(tmp_path/"sparse/manifest.json"), str(tmp_path/"dense/manifest.json")} <= before["external_inputs"].keys()
    config_body["query_mode"] = "last3"
    config.write_text(json.dumps(config_body))
    assert generation_provenance(value)["external_inputs"][str(config)] != before["external_inputs"][str(config)]
    seen = []
    async def native_run(args):
        seen.append(args.retriever_config)
        (Path(args.output)/"native-fake.json").write_text(json.dumps({"baseline":"B0"}))
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
    assert seen == [str(config), str(config)]
    value["methods"][0]["native"]["retriever_config"] = str(config)
    with pytest.raises(ValueError, match="shared at the matrix root"):
        matrix_jobs(value)
