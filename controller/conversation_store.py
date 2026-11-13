"""Committed conversation, source catalogue and reusable computations."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import uuid

from .cooperative_request import source_address
from .schema import EvidenceBlock, QueryContext, SessionSnapshot, TurnRecord
from .serialization import decode, dumps


@dataclass(frozen=True)
class RegisteredDocuments:
    document_set_id: str
    corpus_snapshot_id: str
    blocks: tuple
    source_metadata: tuple


class ConversationStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS qa_sessions (
                session_id TEXT PRIMARY KEY, corpus_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS qa_sources (
                address TEXT PRIMARY KEY, block TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS qa_documents (
                document_set_id TEXT PRIMARY KEY, sources TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS qa_turns (
                session_id TEXT, turn_id INTEGER, request_id TEXT,
                turn TEXT NOT NULL, facts TEXT NOT NULL,
                PRIMARY KEY(session_id, turn_id));
            CREATE TABLE IF NOT EXISTS qa_results (
                session_id TEXT, request_id TEXT, result TEXT NOT NULL,
                PRIMARY KEY(session_id, request_id));
        """)

    def close(self):
        self.db.close()

    def read_snapshot(self, session_id, corpus_snapshot_id="default"):
        with self.db:
            self.db.execute("INSERT INTO qa_sessions VALUES(?,?) ON CONFLICT(session_id) "
                            "DO UPDATE SET corpus_id=excluded.corpus_id", (session_id, corpus_snapshot_id))
        return self.get_snapshot(session_id)

    def get_snapshot(self, session_id):
        corpus = self.db.execute("SELECT corpus_id FROM qa_sessions WHERE session_id=?", (session_id,)).fetchone()[0]
        turns = tuple(decode(TurnRecord, json.loads(row[0])) for row in self.db.execute(
            "SELECT turn FROM qa_turns WHERE session_id=? ORDER BY turn_id", (session_id,)))
        return SessionSnapshot(session_id, version=len(turns), raw_history=turns,
                               corpus_snapshot_id=corpus)

    def sync_history(self, session_id, history):
        with self.db:
            self.db.execute("DELETE FROM qa_turns WHERE session_id=?", (session_id,))
            self.db.executemany("INSERT INTO qa_turns VALUES(?,?,?,?,?)", (
                (session_id, turn.turn_id, None, dumps(turn), "[]") for turn in history))
        return self.get_snapshot(session_id)

    def register_sources(self, blocks):
        self.db.executemany("INSERT OR IGNORE INTO qa_sources VALUES(?,?)", (
            (source_address(block), dumps(block)) for block in blocks))

    def register_document_set(self, blocks, *, source_metadata=None):
        blocks = tuple(blocks)
        document_id = uuid.uuid4().hex
        metadata = tuple((key, value) for key, value in (source_metadata or {}).items())
        with self.db:
            self.register_sources(blocks)
            self.db.execute("INSERT INTO qa_documents VALUES(?,?,?)", (
                document_id, dumps([source_address(block) for block in blocks]), dumps(metadata)))
        return RegisteredDocuments(document_id, document_id, blocks, metadata)

    def load_document_set(self, document_set_id):
        sources, metadata = self.db.execute("SELECT sources,metadata FROM qa_documents WHERE document_set_id=?",
                                            (document_set_id,)).fetchone()
        blocks = tuple(decode(EvidenceBlock, json.loads(self.db.execute(
            "SELECT block FROM qa_sources WHERE address=?", (address,)).fetchone()[0]))
            for address in json.loads(sources))
        from .schema import EvidenceKey
        source_metadata = tuple((decode(EvidenceKey, key), value) for key, value in json.loads(metadata))
        return RegisteredDocuments(document_set_id, document_set_id, blocks, source_metadata)

    def committed_facts(self, snapshot):
        return tuple(json.loads(row[0]) for row in self.db.execute(
            "SELECT facts FROM qa_turns WHERE session_id=? ORDER BY turn_id", (snapshot.session_id,))
            if row[0] != "[]")

    def result(self, session_id, request_id):
        row = self.db.execute("SELECT result FROM qa_results WHERE session_id=? AND request_id=?",
                              (session_id, request_id)).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self, session_id, request_id, query, answer, *, evidence, task, facts, metrics):
        snapshot = self.get_snapshot(session_id)
        turn = TurnRecord(snapshot.version + 1, query, answer, evidence_keys=tuple(b.key for b in evidence),
                          query_context=QueryContext(query, "shared", task_state=task))
        with self.db:
            self.db.execute("INSERT INTO qa_turns VALUES(?,?,?,?,?)", (
                session_id, turn.turn_id, request_id, dumps(turn), dumps(facts)))
            self.db.execute("INSERT OR REPLACE INTO qa_results VALUES(?,?,?)", (
                session_id, request_id, dumps({"status": "completed", **metrics})))
        return turn

    def fail(self, session_id, request_id, error, metrics):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO qa_results VALUES(?,?,?)", (
                session_id, request_id, dumps({"status": "failed", "error": str(error), **metrics})))
