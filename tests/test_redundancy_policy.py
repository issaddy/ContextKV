from dataclasses import replace

import pytest

from controller.candidates import build_candidates
from controller.config import load_config
from controller.cost_model import CostModel
from controller.evidence import EvidencePolicy, make_evidence
from controller.events import EventLog
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs, choose_plan
from controller.schema import ExecutionLedger, Message, SessionSnapshot, TurnRecord
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, ByteTokenizer


def committed_with_bodies(blocks, template, policy):
    snapshot = SessionSnapshot("s")
    query, answer = "Original user question " * 30, "Original complete answer " * 30
    first = build_candidates(snapshot, query, blocks, policy, template)[0]
    output = tuple(template.tokenizer.encode(answer))
    ledger = ExecutionLedger(1, "engine-session", 1,
        first.messages + (Message("assistant", answer),), first.input_ids, output,
        first.visible_blocks, template.fingerprint.template_hash, template.fingerprint.model_revision,
        snapshot.corpus_snapshot_id, first.evidence_spans, first.reference_map)
    return replace(snapshot, version=1, raw_history_version=1, execution=ledger,
        raw_history=(TurnRecord(1, query, answer, output, tuple(first.required_blocks)),),
        archive=blocks, reference_map=first.reference_map)


class RecordingTokenizer(ByteTokenizer):
    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return super().encode(text, **kwargs)


def test_ratio_counts_original_bodies_and_required_dependency_not_qa_or_prompt_difference():
    template = ByteTemplate()
    template.tokenizer = RecordingTokenizer()
    # Evidence token_ids have no tokenizer binding, so the policy must not use them.
    old = replace(make_evidence("中文", "old"), token_ids=(999,) * 90)
    condition = make_evidence("xy", "condition")
    current = make_evidence("ABCD", "current", dependencies=(condition.key,))
    policy = EvidencePolicy(revisit_mode="explicit")
    snapshot = committed_with_bodies((old, condition), template, policy)
    template.tokenizer.calls.clear()
    plain = build_candidates(snapshot, "Current question", (current,), policy, template)
    assert all(c.evidence_body_tokens is None for c in plain)
    assert not any(text in {old.text, condition.text, current.text} for text, _ in template.tokenizer.calls)
    template.tokenizer.calls.clear()
    append, rebuild = build_candidates(snapshot, "Current question", (current,), policy, template,
                                      measure_evidence_redundancy=True)
    assert append.required_blocks == rebuild.required_blocks == {condition.key, current.key}
    assert (append.evidence_body_tokens, append.nonrequired_evidence_body_tokens) == (12, 6)
    assert (rebuild.evidence_body_tokens, rebuild.nonrequired_evidence_body_tokens) == (6, 0)
    # The current QA, source headers and old evidence reference reminders are excluded.
    assert len(append.input_ids) - len(rebuild.input_ids) != 6
    assert [(c.input_ids, c.input_hash, c.messages) for c in (append, rebuild)] == [
        (c.input_ids, c.input_hash, c.messages) for c in plain]
    for body in (old.text, condition.text, current.text):
        assert [(text, kw) for text, kw in template.tokenizer.calls if text == body] == [
            (body, {"add_special_tokens": False})]
    for threshold, action in ((.49, "REBUILD"), (.5, "APPEND"), (.6, "APPEND")):
        plan = choose_plan((append, rebuild), None, CostModel(), snapshot,
                           PlannerKnobs(strategy="redundancy_threshold", redundancy_threshold=threshold))
        assert plan.candidate.action == action
    # Safety constraints still take precedence, even when all old bodies look redundant.
    plan = choose_plan((append, replace(rebuild, feasible=False)), None, CostModel(), snapshot,
                       PlannerKnobs(strategy="redundancy_threshold", redundancy_threshold=0))
    assert plan.candidate == append and plan.reason == "only_feasible_candidate"
    with pytest.raises(ValueError, match="measured evidence body"):
        choose_plan(plain, None, CostModel(), snapshot, PlannerKnobs(strategy="redundancy_threshold"))


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan"), float("inf")])
def test_redundancy_threshold_is_a_fraction(threshold):
    with pytest.raises(ValueError, match="redundancy_threshold"):
        PlannerKnobs(redundancy_threshold=threshold)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["redundancy_threshold", "cost"])
async def test_controller_measures_explicit_policy_or_fallback_and_logs_body_counts(tmp_path, strategy):
    engine = MockEngine()
    policy = EvidencePolicy(revisit_mode="explicit")
    events = EventLog(tmp_path / "events.jsonl")
    controller = Controller(SessionStore(tmp_path / "session.sqlite3"), engine, policy=policy,
        knobs=PlannerKnobs(strategy=strategy, fallback_strategy="redundancy_threshold",
                           redundancy_threshold=.4), events=events)
    try:
        first = await controller.serve_turn("s", "Initial question", (make_evidence("a" * 80, "a"),))
        second = await controller.serve_turn("s", "Current question", (make_evidence("b" * 20, "b"),))
        assert first.action == second.action == "REBUILD"
        assert second.epoch_no > first.epoch_no
        import json
        rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        measured = [r for r in rows if r["event"] == "candidate_built" and r["action"] == "APPEND"]
        assert len(measured) == 1
        assert measured[0]["evidence_body_tokens"] == 100
        assert measured[0]["nonrequired_evidence_body_tokens"] == 80
    finally:
        await controller.aclose()


def test_example_config_loads_without_calibration_or_gpu():
    config = load_config("configs/redundancy.yaml")
    assert config.planner.strategy == "redundancy_threshold"
    assert config.planner.redundancy_threshold == .4


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy,expected_quotes", [("redundancy_threshold", 0), ("cost", 2)])
async def test_control_profile_matches_policy_measurement_and_quote_use(tmp_path, monkeypatch, strategy, expected_quotes):
    from controller.config import RunConfig
    from profiling import control

    engine = MockEngine()
    config = RunConfig(planner=PlannerKnobs(strategy=strategy, fallback_strategy="redundancy_threshold"))
    controller = Controller(SessionStore(tmp_path / "profile.sqlite3"), engine,
                            knobs=config.planner, policy=config.context)
    await controller.serve_turn("s", "Previous question", (make_evidence("Old unrelated body", "old"),))
    observed = []
    original = engine.quote_batch

    async def quote(*args):
        observed.append(args)
        return await original(*args)

    monkeypatch.setattr(engine, "quote_batch", quote)
    monkeypatch.setattr(control, "create_controller", lambda _: controller)
    report = await control.measure(config, samples=2, warmup=0, session_id="s")
    assert len(observed) == expected_quotes
    assert report["samples"] == 2 and report["hardware_generation_measured"] is False
