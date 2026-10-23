import asyncio
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
                                      load_frozen, select_cases, write_frozen)


def source_run(tmp_path, *, layout="stable_key"):
    async def prepare():
        config = RunConfig()
        config = replace(config, context=replace(config.context, layout=layout))
        engine = MockEngine(response_text="ONLY_PREVIOUS_ANSWER")
        store = SessionStore(tmp_path / "source.sqlite3")
        service = Controller(store, engine, policy=config.context,
                             knobs=PlannerKnobs(strategy="always_append"))
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
