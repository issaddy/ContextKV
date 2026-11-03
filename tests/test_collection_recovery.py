"""CPU pause/restart/resume keeps paid native records and candidate uniqueness."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import yaml

from bench.mock_engine import MockEngine
from controller.context_schema import EvidencePool, EvidenceSet, Need, SupportScores, TaskState
from controller.planner import PlannerKnobs
from controller.serialization import to_jsonable
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from training.collection_recovery import candidate_session, prepare_collection_resume
from training.preference_rollout import collect_preferences
from training.reader_data import history_digest, rows, write_rows
from test_reading_runtime import material, fact, read_output


@pytest.mark.asyncio
async def test_pause_restart_retry_keeps_completed_rows_and_original_submitted_step(tmp_path, monkeypatch):
    first, a = material("Item A is eligible.", "a")
    second, b = material("Item B is eligible.", "b")
    pool = EvidencePool((a, b), (first, second))
    sets = {"current": EvidenceSet("both", (a, b), source_blocks=(first, second)),
        "a_only": EvidenceSet("a", (a,), source_blocks=(first,)),
        "b_only": EvidenceSet("b", (b,), source_blocks=(second,)), "deleted_premise": EvidenceSet("empty", ())}
    usage = {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0}
    model = SimpleNamespace(backend=SimpleNamespace(usage=usage), score_support=lambda *args: SupportScores(),
        release_task=lambda *args: None, close=lambda: None, resolve=None)
    monkeypatch.setattr("training.preference_rollout.SupportModel.from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr("training.preference_rollout.candidate_sets", lambda *args, **kwargs: sets)
    # This fixture retrieves supplied objects without a retrieval model call.
    monkeypatch.setattr("training.preference_rollout._encoder_usage", lambda encoder:
        {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0})
    directory = SimpleNamespace(retrieve_pool=lambda *args: pool, units_by_id={"a": a, "b": b})

    class Engine(MockEngine):
        calls = 0
        pause = True

        async def generate(self, *, response_schema=None, **kwargs):
            self.calls += 1
            if self.pause and self.calls == 3:
                raise asyncio.CancelledError("CPU fixture: worker stopped while READ was submitted")
            if response_schema:
                spans = response_schema["properties"]["claims"]["items"]["properties"]["span_ids"]["items"]["enum"]
                self.response_text = json.dumps(read_output(*(fact(span, entity=span) for span in spans)))
            else:
                self.response_text = "Supported final answer."
            return await super().generate(**kwargs)

    output = tmp_path/"collection"
    store = SessionStore(output/"runtime"/"sessions.sqlite3")
    template = ByteTemplate()
    engine = Engine(fingerprint=template.fingerprint)
    controller = Controller(store, engine, template=template, knobs=PlannerKnobs(strategy="always_rebuild"))
    state = TaskState("task", "Which items are eligible?", needs=(Need("n"),), current_turn_id=1)
    label = {"task_id": "task", "conversation_id": "session", "turn": 1, "split": "train",
        "actual_history": [], "question": state.raw_question, "snapshot_id": "native", "task_state": to_jsonable(state)}
    labels_path = tmp_path/"labels.jsonl"
    write_rows(labels_path, [label])
    cohort = tmp_path/"cohort.json"
    cohort.write_text(json.dumps({"members": [{"task_id": "task", "conversation_id": "session", "turn": 1,
        "split": "train", "actual_history_sha256": history_digest([])}], "minimum_candidates_per_state": 4, "maximum_candidates_per_state": 6}))
    config = {"preferences": {"turns_path": str(labels_path), "output_dir": str(output),
        "cohort_manifest": str(cohort), "cohort_split": "train", "reading": True, "reader_data_only": True,
        "read_packing": "mixed", "max_total_tokens": 4194304}}
    with pytest.raises(asyncio.CancelledError):
        await collect_preferences(config, controller=controller, directory=directory)
    completed_before = rows(output/"turns.jsonl")
    assert len(completed_before) == 1 and completed_before[0]["category"] == "current"
    interrupted = store.find_request(candidate_session("task", "a_only"), "answer")
    assert interrupted.stage == "STARTED"
    stage, record, _ = store.execution_step(interrupted.attempt_id, "read:0")
    assert stage == "SUBMITTED" and record.receipt is None
    old_step_bytes = store.db.execute("SELECT payload FROM execution_steps WHERE attempt_id=?", (interrupted.attempt_id,)).fetchone()[0]
    old_restart = engine.engine_restart_id
    receipt = tmp_path/"gpu_pause.json"
    receipt.write_text(json.dumps({"worker_ps": [], "server_ps": [], "gpu_compute_processes": []}))
    prepared = prepare_collection_resume(config, pause_receipt=receipt, confirmed_stopped=True)
    assert prepared["model_requests"] == 0 and engine.calls == 3
    resume_manifest = json.loads(open(prepared["resume_manifest"]).read())
    assert resume_manifest["completed_candidates"] == 1
    entry, = resume_manifest["interrupted_candidates"]
    assert entry["old_engine_restart_ids"] == [old_restart]
    assert entry["old_steps"][0]["stage"] == "SUBMITTED"
    unknown, = entry["cost"]["unresolved_submissions"]
    assert unknown["submitted_input_tokens"] > 0 and unknown["executed_input_tokens"] is None
    assert unknown["output_tokens"] is None and entry["cost"]["reader_input_tokens_total"] is None
    resumed = yaml.safe_load(open(prepared["resume_config"]))
    # Re-preparing before any retry keeps the same registered retry route and
    # retains the first recovery ledger as immutable cost provenance.
    prepared_again = prepare_collection_resume(resumed, pause_receipt=receipt, confirmed_stopped=True)
    again = json.loads(open(prepared_again["resume_manifest"]).read())
    assert again["interrupted_candidates"][0]["retry_session_id"] == entry["retry_session_id"]
    assert again["inherited_interrupted_costs"][0]["old_attempt_id"] == interrupted.attempt_id
    resumed = yaml.safe_load(open(prepared_again["resume_config"]))
    with pytest.raises(ValueError, match="different engine restart"):
        await collect_preferences(resumed, controller=controller, directory=directory)
    assert engine.calls == 3
    await engine.restart()
    engine.pause = False
    manifest = await collect_preferences(resumed, controller=controller, directory=directory)
    completed = rows(output/"turns.jsonl")
    assert len(completed) == 4
    assert len({(row["task_id"], row["category"]) for row in completed}) == 4
    assert completed[0] == completed_before[0]
    retried = next(row for row in completed if row["category"] == "a_only")
    assert retried["collection_recovery"]["old_attempt_id"] == interrupted.attempt_id
    assert retried["collection_recovery"]["old_cost"]["reader_input_tokens_total"] is None
    assert retried["response"]["online_token_budget_verified"]
    assert manifest["complete_reader_input_tokens"] is None
    assert store.get_attempt(interrupted.attempt_id) == interrupted
    assert store.db.execute("SELECT payload FROM execution_steps WHERE attempt_id=?", (interrupted.attempt_id,)).fetchone()[0] == old_step_bytes
    retry_attempt = store.get_attempt(retried["response"]["attempt_id"])
    assert retry_attempt.stage == "COMMITTED" and retry_attempt.session_id == entry["retry_session_id"]
    assert retry_attempt.request_id == entry["retry_request_id"] != "answer"
    calls_after_retry = engine.calls
    await collect_preferences(resumed, controller=controller, directory=directory)
    assert engine.calls == calls_after_retry
    assert rows(output/"turns.jsonl") == completed
    # The original config also keeps recovery costs from the persisted result rows.
    repeated = await collect_preferences(config, controller=controller, directory=directory)
    assert engine.calls == calls_after_retry and repeated["complete_reader_input_tokens"] is None
    assert repeated["interrupted_attempt_ids"] == [interrupted.attempt_id]
    await controller.aclose()
    store.close()


def test_recovery_preparation_requires_stopped_worker_attestation_before_any_io(tmp_path):
    with pytest.raises(ValueError, match="old collector and engine stopped"):
        prepare_collection_resume({}, pause_receipt=tmp_path/"not-read.json", confirmed_stopped=False)
    assert list(tmp_path.iterdir()) == []
