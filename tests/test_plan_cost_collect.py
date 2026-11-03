"""Protocol-only tests; CPU fixtures are not hardware calibration data."""
import pytest

from bench.mock_engine import MockEngine
from controller.context_schema import EvidenceUnit, SourceSpan
from controller.evidence import make_evidence
from controller.events import EventLog
from controller.schema import SessionSnapshot
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from profiling.plan_cost_collect import FixedChoiceCostProbe, PackingTargetUnavailable, pack_original_sources
from profiling.retention_trace import RecordingEngine


def source_bank():
    block = make_evidence("Actual original sentence. "*700, "original-source")
    unit = EvidenceUnit("source-unit", (SourceSpan("source-span", block.key, 0, len(block.text)),),
        block.text, need_ids=("cost:need",), document_family="original-source")
    return ((unit, block),)


@pytest.mark.parametrize("phase", ["READ", "ANSWER"])
def test_target_is_actual_rendered_tokens_and_every_character_keeps_original_coordinates(phase):
    template = ByteTemplate()
    controller = Controller(SessionStore(":memory:"), MockEngine(), template=template)
    bank = source_bank()
    target = 6000 if phase == "READ" else 5000
    plan, proof = pack_original_sources(bank, "What does the original say?", template=template,
        snapshot=SessionSnapshot("session"), policy=controller.policy, phase=phase, target=target,
        read_output=512, answer_output=512, window=12000, max_renders=96)
    assert proof["exact_input_tokens"] == target and proof["renders"] <= 96
    for unit in plan.selected_set.selected_units:
        span = unit.source_spans[0]
        assert unit.text == bank[0][1].text[span.start:span.end]
        assert span.source_key == bank[0][1].key
    controller.store.close()


def test_insufficient_original_source_has_no_synthetic_padding_or_expanded_range():
    template = ByteTemplate()
    store = SessionStore(":memory:")
    controller = Controller(store, MockEngine(), template=template)
    with pytest.raises(PackingTargetUnavailable, match="bank_below_target"):
        pack_original_sources(source_bank(), "Question", template=template, snapshot=SessionSnapshot("s"),
            policy=controller.policy, phase="ANSWER", target=100000,
            read_output=512, answer_output=512, window=120000, max_renders=96)
    store.close()


@pytest.mark.asyncio
async def test_fixed_probe_executes_actual_finite_compiler_quote_answer_and_one_qa_commit(tmp_path):
    template = ByteTemplate()
    engine = MockEngine(response_text="Protocol fixture answer.")
    engine._sampling_params = {"temperature": 0}
    recorder = RecordingEngine(engine)
    store = SessionStore(tmp_path/"state.sqlite3")
    service = Controller(store, recorder, template=template,
        events=EventLog(tmp_path/"events.jsonl", engine_kind="mock"))
    snapshot = store.read_snapshot("session", "source-corpus")
    plan, _ = pack_original_sources(source_bank(), "Question", template=template,
        snapshot=snapshot, policy=service.policy, phase="ANSWER", target=5000,
        read_output=512, answer_output=512, window=12000, max_renders=96)
    service.rolling_policy = FixedChoiceCostProbe(template.fingerprint.model_revision, 5000, 4)
    response = await service.serve_turn("session", "Question", context_plans=(plan,), request_id="one",
                                        corpus_snapshot_id=snapshot.corpus_snapshot_id)
    assert response.input_tokens == 5000 and len(recorder.calls) == 1
    assert recorder.calls[0]["quote_origin"] == "actual_controller_quote"
    assert response.answer_phase_window["start_definition"] == "rolling_proposals_compilation_start"
    assert response.answer_phase_window["start"] < response.answer_phase_window["commit_at"] <= response.answer_phase_window["end"]
    assert len(store.get_snapshot("session").raw_history) == 1
    assert store.get_attempt(response.attempt_id).plan.reason == "fixed_choice_cost_probe"
    await service.close_session("session")
    await service.aclose()
    store.close()
