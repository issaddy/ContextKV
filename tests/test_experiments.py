from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from bench.experiments import generation_provenance, matrix_jobs, run_matrix, select_development

FIXTURE = Path(__file__).parent / "fixtures/mtrag_small.jsonl"


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


def test_explicit_setup_failure_is_reported_without_running_generation(tmp_path):
    value = spec(tmp_path)
    value["lifecycle"] = {"cache_state_protocol": "dedicated_server_reset",
                          "before_job": [sys.executable, "-c", "print('reset failed'); raise SystemExit(3)"]}
    results = asyncio.run(run_matrix(value, tmp_path / "setup-failure"))
    assert all(r["status"] == "failed" for r in results)
    assert all(r["before_job"]["returncode"] == 3 for r in results)
    assert all("reset failed" in r["before_job"]["stdout"] for r in results)
    assert not list((tmp_path / "setup-failure").rglob("benchmark-*.json"))


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
                                              "max_total_model_tokens":8192}}))
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
