import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

import pytest

from controller.config import RunConfig
from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.engine_client import EngineError
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.serialization import dumps
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from profiling.decision_pairs import (DecisionPairCollector, analyze_pairs, freeze_cases,
                                      load_frozen, main, rescore_pairs, select_cases, write_frozen)


def source_run(tmp_path, *, layout="stable_key", strategy="always_append"):
    async def prepare():
        config = RunConfig()
        config = replace(config, context=replace(config.context, layout=layout),
                         planner=PlannerKnobs(strategy=strategy))
        engine = MockEngine(response_text="ONLY_PREVIOUS_ANSWER")
        store = SessionStore(tmp_path / "source.sqlite3")
        service = Controller(store, engine, policy=config.context,
                             knobs=config.planner)
        a, b = make_evidence("body alpha", "alpha"), make_evidence("body beta", "beta")
        try:
            await service.serve_turn("run:dev-session", "first query", (a,), request_id="first")
            engine.response_text = "CURRENT_ANSWER_MUST_NOT_BE_EXPORTED"
            await service.serve_turn("run:dev-session", "second query", (b, a), request_id="second",
                                     retrieval_priority=(a.key, b.key))
            await service.close_session("run:dev-session")
        finally:
            await service.aclose()
            store.close()
        return config
    config = asyncio.run(prepare())
    split = {"dev": ["dev-session"], "test": ["heldout-session"]}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    manifest = {"run_id": "run", "protocol": "P2", "measurement_kind": "simulation",
                "config": asdict(config), "fingerprint": asdict(ByteTemplate().fingerprint),
                "dataset": {"selected_sessions": ["dev-session"],
                            "split_manifest_sha256": sha256(split_path.read_bytes()).hexdigest()}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return tmp_path / "source.sqlite3", manifest, split, manifest_path, split_path


@pytest.mark.parametrize("layout", ("stable_key", "contextpilot"))
def test_freeze_only_prior_committed_inputs_and_exact_candidate(tmp_path, layout):
    database, manifest, split, manifest_path, split_path = source_run(tmp_path, layout=layout)
    cases, skipped = freeze_cases(database, manifest, split, ByteTemplate())
    assert len(cases) == 1 and skipped[0]["reason"] == "initial_turn_has_one_candidate"
    case = cases[0]
    assert len(case.snapshot.raw_history) == 1
    assert case.snapshot.raw_history[0].assistant_text == "ONLY_PREVIOUS_ANSWER"
    assert "CURRENT_ANSWER_MUST_NOT_BE_EXPORTED" not in dumps(case)
    history = case.snapshot.execution.last_input_ids + case.snapshot.execution.last_output_ids
    assert case.warmup_input_ids[:-1] == history
    assert {candidate.action for candidate in case.candidates} == {"APPEND", "REBUILD"}
    target = tmp_path / "cases.jsonl"
    write_frozen(database, manifest_path, split_path, target, ByteTemplate())
    loaded, metadata = load_frozen(target)
    assert loaded[0].candidates == case.candidates or all(
        a.input_ids == b.input_ids for a, b in zip(loaded[0].candidates, case.candidates))
    assert metadata["split"] == "dev"
    with pytest.raises(FileExistsError):
        write_frozen(database, manifest_path, split_path, target, ByteTemplate())


def test_freeze_rejects_test_or_reference_history_and_reports_prompt_drift(tmp_path):
    database, manifest, split, _, _ = source_run(tmp_path)
    bad = {**manifest, "dataset": {"selected_sessions": ["heldout-session"]}}
    with pytest.raises(ValueError, match="development"):
        freeze_cases(database, bad, split, ByteTemplate())
    with pytest.raises(ValueError, match="reference P1"):
        freeze_cases(database, {**manifest, "protocol": "P1"}, split, ByteTemplate())
    manifest["config"]["context"]["layout"] = "contextpilot"
    cases, skipped = freeze_cases(database, manifest, split, ByteTemplate())
    assert not cases and skipped[-1]["reason"] == "source_candidate_changed"


def test_freeze_and_replay_redundancy_baseline_retain_body_measurements(tmp_path):
    database, manifest, split, _, _ = source_run(tmp_path, strategy="redundancy_threshold")
    case = freeze_cases(database, manifest, split, ByteTemplate())[0][0]
    for candidate in case.candidates:
        assert candidate.evidence_body_tokens == len("body alpha") + len("body beta")
        assert candidate.nonrequired_evidence_body_tokens == 0

    async def scenario():
        engine = PairEngineDouble()
        knobs = PlannerKnobs(strategy="redundancy_threshold")
        row = await DecisionPairCollector(engine, CostModel(), knobs).measure(case, "cold")
        assert row["status"] == "completed" and row["selected_action"] == "APPEND"
        rescored = rescore_pairs([row], [case], CostModel(), knobs, await engine.info())
        assert rescored[0]["selected_action"] == "APPEND"
    asyncio.run(scenario())


def test_stratified_selection_covers_conversations_and_epoch_age_without_current_output(tmp_path):
    database, manifest, split, _, _ = source_run(tmp_path)
    original = freeze_cases(database, manifest, split, ByteTemplate())[0][0]
    cases = [replace(original, case_id=f"{session}-{age}-{repeat}", conversation_id=session,
                     snapshot=replace(original.snapshot, turns_since_rebuild=age))
             for session in ("a", "b") for age in (0, 1, 4) for repeat in range(3)]
    selected = select_cases(cases, limit=6, selection="stratified", seed=42)
    assert {(c.conversation_id, min(c.snapshot.turns_since_rebuild, 2)) for c in selected} == {
        (s, age) for s in ("a", "b") for age in (0, 1, 2)}
    assert selected == select_cases(list(reversed(cases)), limit=6, selection="stratified", seed=42)
    assert len({c.case_id for c in select_cases(cases, selection="stratified")}) == len(cases)
    assert select_cases(cases, limit=3) == tuple(cases[:3])
    assert "CURRENT_ANSWER_MUST_NOT_BE_EXPORTED" not in dumps(selected)


class PairEngineDouble(MockEngine):
    def __init__(self):
        super().__init__(response_text="PAIR_ANSWER")
        self.resets = 0
        self.warm_inputs = []
        self.target_inputs = []

    async def info(self):
        return replace(await super().info(), engine_kind="sglang", profiling_enabled=True)

    async def profile_prepare(self, mode):
        if mode == "cold":
            self.resets += 1
            await self.restart()
        elif mode == "host_only":
            for slot in self._slots.values():
                slot.gpu, slot.host = False, True
        return {"mode": mode}

    async def generate(self, **kwargs):
        (self.warm_inputs if kwargs["ignore_eos"] else self.target_inputs).append(tuple(kwargs["input_ids"]))
        result = await super().generate(**kwargs)
        return replace(result, server_elapsed_ms=float(len(kwargs["input_ids"])),
                       server_ttft_ms=1, timing_source="protocol_double")


@pytest.mark.parametrize("state", ("cold", "gpu_ready", "host_only"))
def test_each_arm_resets_and_warms_identical_history_without_target_answer(tmp_path, state):
    database, manifest, split, _, _ = source_run(tmp_path)
    case = freeze_cases(database, manifest, split, ByteTemplate())[0][0]
    async def scenario():
        engine = PairEngineDouble()
        collector = DecisionPairCollector(engine, CostModel(), PlannerKnobs(strategy="cost"))
        row = await collector.measure(case, state)
        assert row["status"] == "completed", row
        assert engine.resets == 2 and len(engine.target_inputs) == 2
        assert engine.warm_inputs == ([] if state == "cold" else [case.warmup_input_ids]*2)
        assert row["plan_reason"] == "uncalibrated_cost_fallback"
        assert not row["quality_evaluated"]
        assert all(arm["result"]["text"] == "PAIR_ANSWER" for arm in row["observations"].values())
        if state == "cold":
            assert all(quote["raw_match_tokens"] == 0 for arm in row["observations"].values()
                       for quote in arm["quote"]["quotes"])
    asyncio.run(scenario())


def test_replay_refuses_production_engine_and_keeps_partial_observations(tmp_path):
    database, manifest, split, _, _ = source_run(tmp_path)
    case = freeze_cases(database, manifest, split, ByteTemplate())[0][0]
    class Failing(PairEngineDouble):
        async def generate(self, **kwargs):
            if len(self.target_inputs) == 1:
                raise EngineError("SECOND_ARM_FAILURE")
            return await super().generate(**kwargs)
    async def scenario():
        with pytest.raises(ValueError, match="dedicated SGLang"):
            await DecisionPairCollector(MockEngine(), CostModel()).measure(case, "cold")
        row = await DecisionPairCollector(Failing(), CostModel()).measure(case, "cold")
        assert row["status"] == "failed" and "SECOND_ARM_FAILURE" in row["error"]
        assert sum("result" in arm for arm in row["observations"].values()) == 1
    asyncio.run(scenario())


def test_replay_records_scope_and_falls_back_for_a_different_cache_config(tmp_path):
    database, manifest, split, _, _ = source_run(tmp_path)
    case = freeze_cases(database, manifest, split, ByteTemplate())[0][0]

    async def scenario():
        engine = PairEngineDouble()
        model = CostModel(
            [LookupPoint(0, len(c.input_ids), 1, 20) for c in case.candidates],
            [DecodePoint(4096, 1, 1, context_start=1, context_end=8192)],
            allow_mock_calibration=True)
        info = await engine.info()
        model.calibration_metadata = {"profile_manifest": {
            "fingerprint": asdict(info.fingerprint),
            "engine": {"cache_config_hash": info.cache_config_hash}}}
        collector = DecisionPairCollector(engine, model)
        matching = await collector.measure(case, "cold")
        assert matching["calibration_scope"]["status"] == "matching_scope"
        assert all(arm["prediction"]["calibrated"] for arm in matching["observations"].values())
        model.calibration_metadata["profile_manifest"]["engine"]["cache_config_hash"] = "other-cache"
        mismatched = await collector.measure(case, "cold")
        assert mismatched["status"] == "completed"
        assert mismatched["calibration_scope"]["status"] == "mismatch"
        assert mismatched["plan_reason"] == "uncalibrated_cost_fallback"
        assert all(not arm["prediction"]["calibrated"] for arm in mismatched["observations"].values())
        assert all(arm["prediction"]["cost"] > 0 for arm in mismatched["observations"].values())
    asyncio.run(scenario())


def test_analysis_distinguishes_prediction_sign_selection_regret_and_fallback():
    def pair(case_id, session, a, r, predicted_a, predicted_r, selected, calibrated=True):
        return {"case_id": case_id, "conversation_id": session, "cache_state": "cold",
                "status": "completed", "selected_action": selected,
                "plan_reason": "cost_saving" if calibrated else "uncalibrated_cost_fallback",
                "observations": {action: {"server_elapsed_ms": actual, "output_tokens": output,
                    "prediction": {"cost": predicted, "calibrated": calibrated}}
                    for action, actual, predicted, output in (("APPEND", a, predicted_a, 4), ("REBUILD", r, predicted_r, 5))}}
    rows = [pair("1", "s1", 10, 5, 10, 5, "REBUILD"),
            pair("2", "s1", 5, 10, 10, 5, "REBUILD"),
            pair("3", "s2", 20, 5, 10, 5, "APPEND", False),
            {"status": "failed", "error": "fixture_failed"}]
    report = analyze_pairs(rows, tie_ms=0)
    assert report["calibration_coverage"] == pytest.approx(2/3)
    assert report["sign_accuracy_on_calibrated"] == .5
    assert report["fallback_pairs"] == 1 and report["failed_or_unmeasured_pairs"] == 1
    assert report["mean_selected_regret_ms"] == pytest.approx(20/3)
    assert report["equal_session_mean_regret_ms"] == 8.75


def measured_pair(tmp_path):
    database, manifest, split, manifest_path, split_path = source_run(tmp_path)
    case = freeze_cases(database, manifest, split, ByteTemplate())[0][0]
    async def measure():
        engine = PairEngineDouble()
        info = await engine.info()
        row = await DecisionPairCollector(engine, CostModel()).measure(case, "cold")
        row["measurement_source"] = "simulation"
        return row, info
    row, info = asyncio.run(measure())
    return case, row, info, (database, manifest_path, split_path)


def rescore_model(case, *, prefer="REBUILD"):
    return CostModel([LookupPoint(0, len(c.input_ids), 1, 20 if c.action == prefer else 100)
                      for c in case.candidates], include_decode_cost=False, allow_mock_calibration=True)


def test_rescore_uses_only_frozen_history_and_quotes_and_keeps_all_measurements(tmp_path):
    case, row, info, _ = measured_pair(tmp_path)
    untouched = deepcopy(row)
    knobs = PlannerKnobs(min_stable_turns=0, delta_rel=0, z=0)
    changed = deepcopy(row)
    for index, arm in enumerate(changed["observations"].values()):
        arm.update(server_elapsed_ms=99999-index, client_complete_ms=index, output_tokens=400-index)
        arm["result"].update(text="CURRENT FUTURE ANSWER", output_ids=[90]*(400-index))
    first = rescore_pairs([row], [case], rescore_model(case), knobs, info)[0]
    second = rescore_pairs([changed], [case], rescore_model(case), knobs, info)[0]
    assert first["selected_action"] == second["selected_action"] == "REBUILD"
    assert first["plan_reason"] == second["plan_reason"] == "cost_saving"
    for action in ("APPEND", "REBUILD"):
        assert first["observations"][action]["prediction"] == second["observations"][action]["prediction"]
        for key, value in untouched["observations"][action].items():
            if key != "prediction":
                assert first["observations"][action][key] == value
    other_model = rescore_pairs([row], [case], rescore_model(case, prefer="APPEND"), knobs, info)[0]
    assert other_model["selected_action"] == "APPEND"
    assert first["measurement_source"] == "simulation"
    assert first["rescoring"]["source_predictions"]["APPEND"] == row["observations"]["APPEND"]["prediction"]
    assert row == untouched


@pytest.mark.parametrize("corruption", ("case_id", "input_hash", "quote_hash", "shape"))
def test_rescore_rejects_wrong_frozen_input_or_cache_preparation(tmp_path, corruption):
    case, row, info, _ = measured_pair(tmp_path)
    changed = deepcopy(row)
    if corruption == "case_id":
        changed["case_id"] = "different-source-case"
    elif corruption == "input_hash":
        changed["observations"]["APPEND"]["input_hash"] = "different-input"
    elif corruption == "quote_hash":
        changed["observations"]["REBUILD"]["quote"]["quotes"][0]["input_hash"] = "different-input"
    else:
        changed["observations"]["REBUILD"]["quote"]["free_gpu_tokens"] += 1
    with pytest.raises(ValueError, match="case_id|input_hash|quote does not describe|different prepared"):
        rescore_pairs([changed], [case], rescore_model(case), PlannerKnobs(), info)


def test_rescore_preserves_failed_rows_and_calibration_scope_mismatch(tmp_path):
    case, row, info, _ = measured_pair(tmp_path)
    failed = {"case_id": case.case_id, "conversation_id": case.conversation_id,
              "status": "failed", "error": "SOURCE_SECOND_ARM_FAILED", "observations": {}}
    model = rescore_model(case)
    model.calibration_metadata = {"profile_manifest": {"fingerprint": asdict(info.fingerprint),
                                    "engine": {"cache_config_hash": "other-engine-config"}}}
    result = rescore_pairs([row, failed], [case], model, PlannerKnobs(), info)
    assert result[0]["plan_reason"] == "uncalibrated_cost_fallback"
    assert result[0]["calibration_scope"]["status"] == "mismatch"
    assert result[1]["error"] == failed["error"] and result[1]["status"] == "failed"
    assert result[1]["rescoring"]["status"] == "source_failure_preserved"
    assert analyze_pairs(result)["failed_or_unmeasured_pairs"] == 1
    assert "rescoring" not in failed


def test_rescore_cli_records_source_identity_without_constructing_engine(tmp_path, monkeypatch):
    import profiling.decision_pairs as module
    case, row, info, inputs = measured_pair(tmp_path)
    cases_path = tmp_path/"cases.jsonl"
    frozen = write_frozen(*inputs, cases_path, ByteTemplate())
    pairs = tmp_path/"pairs.jsonl"
    pairs.write_text(dumps(row)+"\n")
    original = pairs.read_bytes()
    source_manifest = {"split": "dev", "status": "complete", "frozen": frozen,
                       "engine": asdict(info), "config": asdict(RunConfig())}
    pairs.with_suffix(".manifest.json").write_text(json.dumps(source_manifest))
    config = tmp_path/"config.yaml"
    config.write_text(json.dumps(asdict(RunConfig())))
    artifact = tmp_path/"cost.json"
    rescore_model(case).save(artifact)
    def unexpected_engine(*args, **kwargs):
        raise AssertionError("offline rescore constructed an engine client")
    monkeypatch.setattr(module, "SGLangEngineClient", unexpected_engine)
    output = tmp_path/"rescored.jsonl"
    args = ["rescore", "--cases", str(cases_path), "--input", str(pairs), "--config", str(config),
            "--cost-artifact", str(artifact), "--output", str(output)]
    main(args)
    metadata = json.loads(output.with_suffix(".manifest.json").read_text())
    assert metadata["source_pairs_sha256"] == sha256(original).hexdigest()
    assert metadata["source_manifest_sha256"] == sha256(pairs.with_suffix(".manifest.json").read_bytes()).hexdigest()
    assert metadata["measurement_source_counts"] == {"simulation": 1}
    assert metadata["new_generations"] == metadata["new_engine_requests"] == 0
    assert not metadata["current_outcomes_used_for_selection"]
    assert pairs.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(args)
