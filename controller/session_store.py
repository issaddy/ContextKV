"""Durable dual ledgers, one pending turn per session, CAS and retire outbox."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import uuid

from .schema import (ContextEpoch, EvidenceBlock, EvidenceKey, ExecutionLedger, GenerationResult, Message,
                     Plan, RetireOperation, SessionSnapshot, TurnRecord)
from .serialization import dumps, loads


class StoreError(RuntimeError):
    pass


class VersionConflict(StoreError):
    pass


class PendingAttempt(StoreError):
    pass


class IdempotencyConflict(StoreError):
    pass


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    session_id: str
    request_id: str
    request_hash: str
    expected_version: int
    query: str
    stage: str = "STARTED"
    evidence: tuple[EvidenceBlock, ...] = ()
    plan: Plan | None = None
    epoch: ContextEpoch | None = None
    result: GenerationResult | None = None
    new_epoch: bool = False
    error: str | None = None
    created_ns: int = 0
    branch_reason: str = ""
    retrieval_priority: tuple[EvidenceKey, ...] | None = None
    retrieval_ms: float | None = None


ACTIVE_STAGES = ("STARTED", "PLANNED", "PREPARING", "PREPARED", "SUBMITTED", "RESULT_READY")


class SessionStore:
    def __init__(self, path: str | Path, evidence_dir: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = Path(evidence_dir or self.path.parent / "evidence")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self.db = sqlite3.connect(str(self.path), timeout=10, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
            snapshot TEXT NOT NULL, closed INTEGER NOT NULL DEFAULT 0,
            epoch_counter INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
            request_id TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL,
            UNIQUE(session_id, request_id));
          CREATE UNIQUE INDEX IF NOT EXISTS one_pending_turn ON attempts(session_id)
            WHERE stage NOT IN ('COMMITTED', 'FAILED');
          CREATE TABLE IF NOT EXISTS retire_outbox (
            operation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            payload TEXT NOT NULL, acked INTEGER NOT NULL DEFAULT 0,
            retries INTEGER NOT NULL DEFAULT 0, last_error TEXT,
            created_ns INTEGER NOT NULL, acked_ns INTEGER);
          CREATE TABLE IF NOT EXISTS ledger_history (
            session_id TEXT NOT NULL, version INTEGER NOT NULL, snapshot TEXT NOT NULL,
            PRIMARY KEY(session_id, version));
          PRAGMA user_version=1;
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        with self._mutex:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def _read(self, session_id: str) -> SessionSnapshot:
        row = self.db.execute("SELECT snapshot FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return loads(SessionSnapshot, row[0])

    def read_snapshot(self, session_id: str, corpus_snapshot_id: str = "demo-v1") -> SessionSnapshot:
        with self._transaction():
            self.db.execute("INSERT OR IGNORE INTO sessions(session_id,version,snapshot) VALUES(?,?,?)",
                            (session_id, 0, dumps(SessionSnapshot(session_id, corpus_snapshot_id=corpus_snapshot_id))))
            return self._read(session_id)

    def get_snapshot(self, session_id: str) -> SessionSnapshot:
        """Read-only inspection; unlike serving, it never creates a session."""
        with self._mutex:
            return self._read(session_id)

    def _write_snapshot(self, snapshot: SessionSnapshot, expected_version: int):
        payload = dumps(snapshot)
        changed = self.db.execute("UPDATE sessions SET version=?,snapshot=? WHERE session_id=? AND version=?",
                                  (snapshot.version, payload, snapshot.session_id, expected_version)).rowcount
        if changed != 1:
            raise VersionConflict("Session ledger changed; refusing to overwrite")
        self.db.execute("INSERT INTO ledger_history VALUES(?,?,?)", (snapshot.session_id, snapshot.version, payload))

    def get_attempt(self, attempt_id: str) -> Attempt:
        with self._mutex:
            row = self.db.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            return loads(Attempt, row[0])

    def find_request(self, session_id: str, request_id: str) -> Attempt | None:
        with self._mutex:
            row = self.db.execute("SELECT payload FROM attempts WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            return loads(Attempt, row[0]) if row else None

    def pending_attempt(self, session_id: str) -> Attempt | None:
        with self._mutex:
            row = self.db.execute("SELECT payload FROM attempts WHERE session_id=? AND stage NOT IN ('COMMITTED','FAILED')", (session_id,)).fetchone()
            return loads(Attempt, row[0]) if row else None

    def begin_attempt(self, session_id: str, request_id: str, request_hash: str,
                      query: str, expected_version: int) -> Attempt:
        with self._transaction():
            previous = self.find_request(session_id, request_id)
            if previous:
                if previous.request_hash != request_hash:
                    raise IdempotencyConflict("request_id reused with a different request")
                if previous.stage not in ("COMMITTED", "FAILED"):
                    raise PendingAttempt("This request already has an executing or unresolved owner")
                return previous
            snap = self._read(session_id)
            if snap.version != expected_version:
                raise VersionConflict("Snapshot changed before begin_attempt")
            if self.db.execute("SELECT closed FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]:
                raise StoreError("Session is closed")
            if self.pending_attempt(session_id):
                raise PendingAttempt("Session has an unresolved attempt; recover it before another turn")
            attempt = Attempt(uuid.uuid4().hex, session_id, request_id, request_hash, expected_version, query, created_ns=time.monotonic_ns())
            self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,?)", (attempt.attempt_id, session_id, request_id, attempt.stage, dumps(attempt)))
            return attempt

    def save_attempt(self, attempt: Attempt) -> Attempt:
        with self._transaction():
            old = self.get_attempt(attempt.attempt_id)
            if old.stage in ("COMMITTED", "FAILED"):
                raise StoreError("Terminal attempt is immutable")
            if (old.session_id, old.request_id, old.request_hash, old.expected_version) != (attempt.session_id, attempt.request_id, attempt.request_hash, attempt.expected_version):
                raise StoreError("Attempt identity is immutable")
            allowed = {"STARTED": {"PLANNED"}, "PLANNED": {"PREPARED"},
                       "PREPARING": {"PREPARING", "PREPARED"}, "PREPARED": {"SUBMITTED"},
                       "SUBMITTED": {"RESULT_READY"}, "RESULT_READY": set()}
            if attempt.stage not in allowed.get(old.stage, set()):
                if attempt == old:
                    return old
                raise StoreError(f"Invalid or stale attempt transition {old.stage} -> {attempt.stage}")
            if old.plan is not None and (attempt.plan != old.plan or attempt.evidence != old.evidence):
                raise StoreError("A selected plan and its evidence are immutable")
            if old.result is not None and attempt.result != old.result:
                raise StoreError("A durable result is immutable")
            if old.epoch is not None and attempt.epoch != old.epoch:
                expected = replace(old.epoch, session_generation=attempt.epoch.session_generation) if attempt.epoch else None
                if old.epoch.session_generation != 0 or expected != attempt.epoch:
                    raise StoreError("Prepared epoch identity is immutable")
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (attempt.stage, dumps(attempt), attempt.attempt_id))
            return attempt

    def allocate_epoch(self, attempt: Attempt, restart_id: str, reason: str) -> Attempt:
        with self._transaction():
            current = self.get_attempt(attempt.attempt_id)
            if current.epoch is not None:
                return current
            if current.stage != "PLANNED":
                raise StoreError("Only a selected plan may prepare an epoch")
            snap = self._read(current.session_id)
            self.db.execute("UPDATE sessions SET epoch_counter=epoch_counter+1 WHERE session_id=?", (current.session_id,))
            number = self.db.execute("SELECT epoch_counter FROM sessions WHERE session_id=?", (current.session_id,)).fetchone()[0]
            epoch = ContextEpoch(current.session_id, number, f"ctx-{uuid.uuid4().hex}", 0,
                                 snap.current_epoch.epoch_no if snap.current_epoch else None,
                                 reason=reason, engine_restart_id=restart_id)
            updated = replace(current, stage="PREPARING", epoch=epoch, new_epoch=True)
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (updated.stage, dumps(updated), updated.attempt_id))
            return updated

    def _archive_evidence(self, evidence: tuple[EvidenceBlock, ...]):
        """Persist evidence already validated when it entered candidate construction."""
        for block in evidence:
            target = self.evidence_dir / f"{block.key.content_sha256}.txt"
            if target.exists():
                continue
            fd, temp = tempfile.mkstemp(dir=self.evidence_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(block.text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)

    def _enqueue(self, session_id: str, epoch: ContextEpoch, reason: str, version: int):
        operation = RetireOperation(f"retire:{epoch.engine_session_id}", epoch.engine_session_id,
                                   epoch.session_generation, epoch.engine_restart_id, reason, version)
        self.db.execute("INSERT OR IGNORE INTO retire_outbox(operation_id,session_id,payload,created_ns) VALUES(?,?,?,?)",
                        (operation.operation_id, session_id, dumps(operation), time.monotonic_ns()))

    def commit_turn_and_outbox(self, attempt_id: str) -> SessionSnapshot:
        attempt = self.get_attempt(attempt_id)
        if attempt.stage == "COMMITTED":
            return self.read_snapshot(attempt.session_id)
        if attempt.plan is None or attempt.epoch is None or attempt.result is None or attempt.stage != "RESULT_READY":
            raise StoreError("Only a complete durable result can be committed")
        self._archive_evidence(attempt.evidence)
        with self._transaction():
            attempt = self.get_attempt(attempt_id)
            if attempt.stage == "COMMITTED":
                return self._read(attempt.session_id)
            if attempt.stage != "RESULT_READY" or attempt.plan is None or attempt.result is None or attempt.epoch is None:
                raise StoreError("Attempt changed before the commit transaction")
            snap = self._read(attempt.session_id)
            if snap.version != attempt.expected_version:
                raise VersionConflict("CAS failed: completed result remains durable for recovery")
            candidate, result, epoch = attempt.plan.candidate, attempt.result, attempt.epoch
            if not result.generation_complete or not result.receipt.generation_complete:
                raise StoreError("Partial output cannot become semantic history")
            version = snap.version + 1
            history_version = snap.raw_history_version + 1
            turn = TurnRecord(len(snap.raw_history) + 1, attempt.query, result.text, result.output_ids,
                              result.finish_reason, tuple(sorted(candidate.required_blocks)))
            ledger = ExecutionLedger(epoch.epoch_no, epoch.engine_session_id, history_version,
                                     candidate.messages + (Message("assistant", result.text),),
                                     candidate.input_ids, result.output_ids, candidate.visible_blocks,
                                     candidate.fingerprint.template_hash if candidate.fingerprint else "",
                                     candidate.fingerprint.model_revision if candidate.fingerprint else "",
                                     candidate.corpus_snapshot_id, candidate.evidence_spans,
                                     candidate.reference_map, result.receipt.inserted_cache_tokens)
            archive = {b.key: b for b in snap.archive}
            archive.update({b.key: b for b in attempt.evidence})
            updated = replace(snap, version=version, raw_history=snap.raw_history + (turn,),
                              raw_history_version=history_version, execution=ledger,
                              archive=tuple(archive[k] for k in sorted(archive)),
                              current_epoch=replace(epoch, state="COMMITTED"), history_sync_pending=False,
                              force_new_epoch_for_recovery=False, must_rebuild=False,
                              reference_map=candidate.reference_map,
                              turns_since_rebuild=0 if candidate.action == "REBUILD" else snap.turns_since_rebuild + 1)
            self._write_snapshot(updated, snap.version)
            completed = replace(attempt, stage="COMMITTED")
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (completed.stage, dumps(completed), attempt_id))
            if attempt.new_epoch and snap.current_epoch:
                self._enqueue(snap.session_id, snap.current_epoch, epoch.reason, version)
            return updated

    def fail_attempt(self, attempt_id: str, error: str, force_recovery: bool = False,
                     retire_candidate: bool = True):
        """Call only before submit, or after engine completion/cancellation is confirmed."""
        with self._transaction():
            attempt = self.get_attempt(attempt_id)
            if attempt.stage in ("COMMITTED", "FAILED"):
                return
            updated = replace(attempt, stage="FAILED", error=error)
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (updated.stage, dumps(updated), attempt_id))
            snap = self._read(attempt.session_id)
            if force_recovery:
                self._write_snapshot(replace(snap, version=snap.version + 1, force_new_epoch_for_recovery=True), snap.version)
            if retire_candidate and attempt.new_epoch and attempt.epoch and attempt.epoch.session_generation:
                self._enqueue(attempt.session_id, attempt.epoch, "failed_candidate", snap.version)

    def sync_history(self, session_id: str, history: tuple[TurnRecord, ...], expected_version: int) -> SessionSnapshot:
        with self._transaction():
            snap = self._read(session_id)
            if self.db.execute("SELECT closed FROM sessions WHERE session_id=?", (session_id,)).fetchone()[0]:
                raise StoreError("Session is closed")
            if snap.version != expected_version:
                raise VersionConflict("History synchronization lost CAS")
            if self.pending_attempt(session_id):
                raise PendingAttempt("Cannot synchronize history during an unresolved attempt")
            if tuple((t.user_text, t.assistant_text) for t in snap.raw_history) == tuple((t.user_text, t.assistant_text) for t in history):
                return snap
            # P1 can replace an answer with reference history, but evidence used
            # for that same earlier user question remains an observed association.
            previous = {(turn.turn_id, turn.user_text): turn for turn in snap.raw_history}
            history = tuple(replace(turn, evidence_keys=previous[(turn.turn_id, turn.user_text)].evidence_keys)
                            if (turn.turn_id, turn.user_text) in previous else turn for turn in history)
            updated = replace(snap, version=snap.version + 1, raw_history=history,
                              raw_history_version=snap.raw_history_version + 1, history_sync_pending=True)
            self._write_snapshot(updated, snap.version)
            return updated

    def outbox(self, session_id: str | None = None) -> tuple[RetireOperation, ...]:
        with self._mutex:
            sql = "SELECT payload FROM retire_outbox WHERE acked=0"
            params = ()
            if session_id is not None:
                sql += " AND session_id=?"
                params = (session_id,)
            return tuple(loads(RetireOperation, row[0]) for row in self.db.execute(sql, params))

    def ack_retire(self, operation_id: str):
        with self._transaction():
            self.db.execute("UPDATE retire_outbox SET acked=1,acked_ns=?,last_error=NULL WHERE operation_id=?", (time.monotonic_ns(), operation_id))

    def retire_failed(self, operation_id: str, error: str):
        with self._transaction():
            self.db.execute("UPDATE retire_outbox SET retries=retries+1,last_error=? WHERE operation_id=?", (error, operation_id))

    def close_session(self, session_id: str):
        with self._transaction():
            if self.pending_attempt(session_id):
                raise PendingAttempt("Resolve the pending attempt before closing")
            snap = self._read(session_id)
            if snap.current_epoch:
                self._enqueue(session_id, snap.current_epoch, "session_closed", snap.version)
            self.db.execute("UPDATE sessions SET closed=1 WHERE session_id=?", (session_id,))
