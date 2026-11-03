import json
from pathlib import Path

import httpx
import pytest

from bench.study_scoring import score_dataset


def write_scores(path, records):
    path.write_text("".join(json.dumps(record)+"\n" for record in records))


def row(task_id):
    return {"method": "B3", "dataset": "fixture", "task_id": task_id, "conversation_id": task_id,
        "status": "completed", "metrics": {}, "split": "dev",
        "cost": {"complete_ms": 3, "gpu_gib_seconds": 2, "host_gib_seconds": 1}}


@pytest.mark.asyncio
async def test_import_official_components_keeps_raw_algorithmic_alias_and_online_cost(tmp_path):
    source = tmp_path/"source.jsonl"
    write_scores(source, [{"task_id": "a", "metrics": {"RL_F": [.8], "RB_llm": [.7], "RB_agg": [.6],
        "RL_F_idk": [0], "RB_llm_idk": [0], "RB_agg_idk": [0]}}])
    initial = row("a")
    spec = {"primary_metric": "mtrageval_harmonic", "scoring": {
        "scores_path": str(source), "identity": {"protocol": "explicit_test_scorer"}}}
    result = await score_dataset({}, spec, [initial], tmp_path/"generation")
    assert result[0]["metrics"]["RB_agg"] == [.6]
    assert result[0]["metrics"]["RB_alg"] == [.6]
    assert result[0]["metrics"]["RB_alg_idk"] == [0]
    assert result[0]["scoring"]["status"] == "scored"
    assert result[0]["scoring"]["metric_aliases"]["RB_alg"]["from"] == "RB_agg"
    assert result[0]["scoring"]["metric_aliases"]["RB_alg_idk"]["from"] == "RB_agg_idk"
    assert result[0]["cost"] == initial["cost"]
    assert initial["metrics"] == {}
    manifest = json.loads((tmp_path/"generation/scoring/manifest.json").read_text())
    assert manifest["judge_cost"]["unknown_input_token_records"] == 1


@pytest.mark.asyncio
async def test_raw_only_mtrag_scores_remain_diagnostic_until_official_conditioning(tmp_path):
    source = tmp_path/"raw.jsonl"
    write_scores(source, [{"task_id": "a", "metrics": {"RL_F": [.8], "RB_llm": [.7], "RB_agg": [.6]}}])
    spec = {"primary_metric": "mtrageval_harmonic", "scoring": {
        "scores_path": str(source), "identity": {"protocol": "explicit_test_scorer"}}}
    result = await score_dataset({}, spec, [row("a")], tmp_path/"generation")
    assert result[0]["scoring"]["status"] == "missing"
    assert result[0]["metrics"]["RB_alg"] == [.6]


@pytest.mark.asyncio
async def test_official_pipeline_conditions_algorithmic_component_after_merging(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from bench import study_scoring

    generation = tmp_path/"generation"
    directory = generation/"scoring"
    directory.mkdir(parents=True)
    checkout = tmp_path/"official-source"
    checkout.mkdir()
    (checkout/"contextkv-source.json").write_text(json.dumps({"commit": "pinned-fixture"}))
    prediction = {"task_id": "a", "input": [{"speaker": "user", "text": "Question"}],
                  "predictions": [{"text": "Answer"}], "contexts": []}
    write_scores(generation/"official.jsonl", [prediction])
    write_scores(generation/"references.jsonl", [{**prediction, "Answerability": ["UNANSWERABLE"],
                                                "targets": [{"text": "No supported answer"}]}])
    commands = []

    @contextmanager
    def no_http(endpoint, journal):
        yield endpoint, []

    async def execute(arguments, workdir, label):
        commands.append((arguments, label))
        source = study_scoring.load_jsonl(arguments[arguments.index("--input")+1])
        destination = Path(arguments[arguments.index("--output")+1])
        if label == "official-generation":
            destination.mkdir()
            source[0]["metrics"] = {"RL_F": [.8], "RB_llm": [.7], "idk_eval": [0],
                                    "RL_F_idk": [0], "RB_llm_idk": [0]}
            write_scores(destination/"scores.jsonl", source)
        elif label == "official-algorithmic":
            source[0]["metrics"] = {"RB_agg": [.6]}
            write_scores(destination, source)
        else:
            assert label == "official-condition"
            assert arguments[arguments.index("--metrics")+1:] == ["RL_F", "RB_llm", "RB_agg"]
            assert source[0]["metrics"]["idk_eval"] == [0]
            assert source[0]["metrics"]["RB_agg"] == [.6]
            source[0]["metrics"]["RB_agg_idk"] = [0]
            write_scores(destination, source)
        return {"elapsed_ms": 1}

    monkeypatch.setattr(study_scoring, "record_judge_http", no_http)
    monkeypatch.setattr(study_scoring, "_command", execute)
    spec = {"primary_metric": "mtrageval_harmonic"}
    settings = {"official_checkout": str(checkout), "ragas_json_schema": True,
                "judge": {"model": "fixture", "revision": "v1", "base_url": "http://judge.test/v1"}}
    records, identity, outputs, _ = await study_scoring._official_scores({}, spec, directory, settings)
    assert [label for _, label in commands] == ["official-generation", "official-algorithmic", "official-condition"]
    assert "--ragas-json-schema" in commands[0][0]
    assert records[0]["metrics"]["RB_agg"] == [.6]
    assert records[0]["metrics"]["RB_agg_idk"] == [0]
    assert identity["answerability_adjustment"]
    assert str(directory/"official.conditioned.jsonl") in outputs


@pytest.mark.asyncio
async def test_missing_mtrag_component_stays_missing_and_does_not_become_rbllm(tmp_path):
    source = tmp_path/"source.jsonl"
    write_scores(source, [{"task_id": "a", "metrics": {"RB_llm": [.9], "RB_agg": [.8]}}])
    spec = {"primary_metric": "mtrageval_harmonic", "scoring": {
        "scores_path": str(source), "identity": {"protocol": "explicit_test_scorer"}}}
    result = await score_dataset({}, spec, [row("a"), row("missing")], tmp_path/"generation")
    assert result[0]["scoring"]["status"] == "missing"
    assert "RL_F" not in result[0]["metrics"]
    assert result[1]["metrics"] == {}


@pytest.mark.asyncio
async def test_loong_zero_is_observed_while_missing_score_is_absent(tmp_path):
    source = tmp_path/"loong.jsonl"
    write_scores(source, [{"task_id": "a", "score": 0}, {"task_id": "b", "score": None}])
    spec = {"primary_metric": "loong_reference_score", "scoring": {
        "scores_path": str(source), "identity": {"protocol": "loong_fixture"}}}
    result = await score_dataset({}, spec, [row("a"), row("b")], tmp_path/"generation")
    assert result[0]["metrics"]["loong_reference_score"] == 0
    assert result[0]["scoring"]["status"] == "scored"
    assert result[1]["metrics"] == {}


@pytest.mark.asyncio
async def test_version_project_rubric_and_scorer_identity_are_explicit(tmp_path):
    source = tmp_path/"project.jsonl"
    write_scores(source, [{"task_id": "a", "scores": {"correctness": .8, "conditions_versions": .6},
        "judge": {"provider": "test", "model": "fixture", "revision": "v1"},
        "judge_response": {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}}])
    spec = {"primary_metric": "project_condition_version_score", "scoring": {"scores_path": str(source)}}
    result = await score_dataset({}, spec, [row("a")], tmp_path/"generation")
    assert result[0]["metrics"]["project_condition_version_score"] == pytest.approx(.7)
    assert result[0]["metrics"]["project_conditions_versions"] == .6
    assert result[0]["scoring"]["judge_cost"]["input_tokens"] == 100
    write_scores(source, [{"task_id": "a", "metrics": {"RB_llm": .9}}])
    with pytest.raises(ValueError, match="scorer identity"):
        await score_dataset({}, spec, [row("a")], tmp_path/"unattributed")


@pytest.mark.asyncio
@pytest.mark.parametrize("offline_reference,expected_targets", (
    ({"answer": "Linux permits 20", "source": "policy-v2"},
     {"answer": "Linux permits 20", "source": "policy-v2"}),
    ({"targets": [{"text": "Linux permits 20"}]}, [{"text": "Linux permits 20"}]),
))
async def test_dynamic_goals_pass_actual_branch_conditions_and_offline_targets_to_judge_only(
        tmp_path, monkeypatch, offline_reference, expected_targets):
    from bench import judge as judge_module
    captured = []
    def respond(request):
        payload = json.loads(request.content)
        captured.append(json.loads(payload["messages"][1]["content"]))
        text = {"scores": {name: .8 for name in judge_module.DIMENSIONS},
                "script_history_mismatch": False, "reason": "current branch satisfied",
                "unsupported_claims": []}
        return httpx.Response(200, json={"id": "judge-test", "model": "fixed-test",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(text)}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 10}})
    offline_judge = judge_module.OfflineJudge
    monkeypatch.setattr(judge_module, "OfflineJudge", lambda config: offline_judge(
        config, transport=httpx.MockTransport(respond)))
    generation = tmp_path/"dynamic"
    generation.mkdir()
    actual_history = [{"turn_id": 1, "user_text": "Use Windows", "assistant_text": "Windows permits 10"}]
    audit = {"task_id": "a", "conversation_id": "goal-conversation", "query": "Change to Linux",
        "answer": "Linux permits 20", "actual_history": actual_history, "retrieved_evidence": [],
        "visible_evidence_manifest": [], "audit_evidence": []}
    conditions = {"PLATFORM": "Linux", "VERSION": "current"}
    reference = {"task_id": "a", "input": [{"speaker": "user", "text": "authored old branch"}],
        "targets": [{"text": "stale authored target"}], "actual_history": actual_history,
        "goal_conditions": conditions, "offline_reference": offline_reference}
    write_scores(generation/"audit.jsonl", [audit])
    write_scores(generation/"references.jsonl", [reference])
    spec = {"primary_metric": "project_condition_version_score", "scoring": {"judge": {
        "provider": "test", "model": "fixed-test", "revision": "test-v1", "base_url": "http://judge.test/v1"}}}
    initial = {**row("a"), "dataset": "dynamic_goal_dev"}
    result = await score_dataset({}, spec, [initial], generation)
    assert captured == [{**{name: audit[name] for name in (
        "task_id", "query", "answer", "actual_history", "retrieved_evidence",
        "visible_evidence_manifest", "audit_evidence")},
        "reference_targets": expected_targets, "task_requirements": conditions}]
    assert result[0]["cost"] == initial["cost"]
    assert result[0]["scoring"]["identity"]["evaluation_track"] == "answer_driven_dynamic_goal"
    assert result[0]["scoring"]["identity"]["public_benchmark_quality_aggregate"] is False
    assert result[0]["scoring"]["judge_cost"]["input_tokens"] == 40
    assert json.loads((generation/"references.jsonl").read_text()) == reference
