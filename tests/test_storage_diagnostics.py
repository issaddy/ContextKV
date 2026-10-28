import json
from pathlib import Path

import pytest

from controller.candidates import build_candidates
from controller.events import EventLog
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.schema import SessionSnapshot
from controller.service import Controller
from controller.session_store import SessionStore, load_record


async def test_text_is_stored_once_and_all_historical_inputs_reconstruct(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    engine = MockEngine()
    controller = Controller(store, engine)
    originals = []
    for index in range(10):
        block = make_evidence(f"{index:05d}:" + "x" * 1000, str(index))
        await controller.serve_turn("s", f"Read source {index}", (block,), request_id=str(index))
        originals.append(store.get_snapshot("s"))
    rows = store.db.execute("SELECT snapshot FROM ledger_history ORDER BY version").fetchall()
    repeated = sum(len(b.text) for snapshot in originals for b in snapshot.archive)
    unique = sum(len(row[0]) for row in store.db.execute(
        "SELECT text FROM text_blobs WHERE sha256 IN (" + ",".join("?" for _ in originals) + ")",
        [snapshot.archive[-1].key.content_sha256 for snapshot in originals]))
    assert repeated == 55330 and unique == 10060
    for row, original in zip(rows, originals):
        stored = json.loads(row[0])
        assert all(set(block["text"]) == {"$text"} for block in stored["archive"])
        assert load_record(store.db, SessionSnapshot, row[0]) == original
    expected = build_candidates(originals[-1], "Continue", ())[0].input_ids
    await controller.aclose()
    store.close()
    reopened = SessionStore(tmp_path / "sessions.sqlite3")
    assert build_candidates(reopened.get_snapshot("s"), "Continue", ())[0].input_ids == expected
    assert reopened.find_request("s", "9").stage == "COMMITTED"
    assert not (tmp_path / "evidence").exists()
    reopened.close()


@pytest.mark.parametrize("stage", ["write", "post_commit_callback"])
async def test_diagnostic_failure_cannot_fail_a_committed_answer_or_retirement(tmp_path, monkeypatch, caplog, stage):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    events = EventLog(tmp_path / "events.jsonl")
    controller = Controller(store, MockEngine(), events=events)
    if stage == "post_commit_callback":
        emit = events.emit
        def broken_callback(event, **fields):
            if event == "ledger_committed":
                raise RuntimeError("injected diagnostic callback failure")
            return emit(event, **fields)
        monkeypatch.setattr(events, "emit", broken_callback)
    else:
        original = Path.open
        def fail(path, *args, **kwargs):
            if path == events.path:
                raise OSError("diagnostic disk unavailable")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", fail)
    result = await controller.serve_turn("s", "question", request_id="one")
    assert result.assistant_text and store.find_request("s", "one").stage == "COMMITTED"
    replay = await controller.serve_turn("s", "question", request_id="one")
    assert replay.idempotent_replay and replay.assistant_text == result.assistant_text
    await controller.close_session("s")
    assert not store.outbox()
    assert "Diagnostic output failed" in caplog.text
    if events.path.exists():
        assert all(json.loads(line)["event"] != "attempt_failed" for line in events.path.read_text().splitlines())
    await controller.aclose()
    store.close()
