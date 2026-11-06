"""Durable dual ledgers, one pending turn per session, CAS and retire outbox."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .evidence import merge_evidence_block
from .schema import (CompletionReceipt, ContextEpoch, EvidenceBlock, EvidenceKey, ExecutionLedger, GenerationResult, Message, QueryContext,
                     Plan, RetireOperation, SessionSnapshot, TurnRecord)
from .serialization import decode, dumps, loads
from .context_schema import ContextPlan, OpenNeed, ReadingOutcome, StepRecord
from .prefix_retention import PrefixOwnerRecord


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
    query_context: QueryContext | None = None
    context_plan: ContextPlan | None = None
    reading_outcome: ReadingOutcome | None = None
    reading_elapsed_ms: float = 0.0
    auxiliary_usage: dict | None = None


ACTIVE_STAGES = ("STARTED", "PLANNED", "PREPARING", "PREPARED", "SUBMITTED", "RESULT_READY")


@dataclass(frozen=True)
class ReadingProgress:
    expanded: tuple[tuple[str, str, tuple[str, ...], str], ...] = ()
    blocked_open_needs: tuple[OpenNeed, ...] = ()
    auxiliary_usage: dict | None = None


@dataclass(frozen=True)
class DocumentSet:
    document_set_id: str
    corpus_snapshot_id: str
    blocks: tuple[EvidenceBlock, ...]
    source_metadata: tuple[tuple[EvidenceKey, dict], ...] = ()

    @classmethod
    def create(cls, blocks, source_metadata=None):
        blocks = tuple(blocks)
        metadata = tuple(sorted((source_metadata or {}).items()))
        digest = sha256(dumps((blocks, metadata)).encode("utf-8")).hexdigest()
        identity = "documents:"+digest
        return cls(identity, identity, blocks, metadata)


def load_record(connection, cls, payload):
    """Reconstruct exact persisted text for serving, recovery and frozen replay."""
    texts = {}

    def resolve(value):
        if set(value) == {"$text"}:
            digest = value["$text"]
            if digest not in texts:
                row = connection.execute("SELECT text FROM text_blobs WHERE sha256=?", (digest,)).fetchone()
                if row is None:
                    raise StoreError(f"missing stored text: {digest}")
                texts[digest] = row[0]
            return texts[digest]
        return value

    return decode(cls, json.loads(payload, object_hook=resolve))


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self.db = sqlite3.connect(str(self.path), timeout=10, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("PRAGMA user_version").fetchone()[0] not in (0, 2):
            self.db.close()
            raise StoreError("obsolete session database format; use a new state directory")
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
          CREATE TABLE IF NOT EXISTS text_blobs (
            sha256 TEXT PRIMARY KEY, text TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS document_sets (
            document_set_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS turn_plans (
            attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id), plan TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS execution_steps (
            attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id), step_id TEXT NOT NULL,
            stage TEXT NOT NULL, payload TEXT NOT NULL, generation TEXT,
            PRIMARY KEY(attempt_id,step_id));
          CREATE TABLE IF NOT EXISTS reading_progress (
            attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id), payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS prefix_owners (
            owner_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, stage TEXT NOT NULL,
            engine_restart_id TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS prefix_owners_by_session ON prefix_owners(session_id,stage);
          CREATE TABLE IF NOT EXISTS prefix_owner_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id TEXT NOT NULL,
            event TEXT NOT NULL, payload TEXT NOT NULL, created_ns INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS prefix_accesses (
            access_id INTEGER PRIMARY KEY AUTOINCREMENT, owner_id TEXT NOT NULL,
            session_id TEXT NOT NULL, native_attempt_id TEXT NOT NULL,
            observed_ns INTEGER NOT NULL, payload TEXT NOT NULL,
            UNIQUE(owner_id,native_attempt_id));
          CREATE INDEX IF NOT EXISTS prefix_accesses_by_owner ON prefix_accesses(owner_id,access_id);
          PRAGMA user_version=2;
        """)

    def close(self):
        self.db.close()

    def _dump(self, value):
        """Text and its references are persisted in the same SQLite transaction.

        Dataclass text fields share bodies across archives, attempts, prompt
        messages and actual QA. Primitive token arrays keep native JSON encoding.
        """
        saved = set()

        def encode(item):
            if isinstance(item, (set, frozenset)):
                return sorted(item)
            if not is_dataclass(item):
                raise TypeError(f"unsupported stored value: {type(item).__name__}")
            result = {field.name: getattr(item, field.name) for field in fields(item)}
            for name in ("text", "content", "user_text", "assistant_text", "query"):
                text = result.get(name)
                if isinstance(text, str) and text:
                    digest = sha256(text.encode("utf-8")).hexdigest()
                    if digest not in saved:
                        self.db.execute("INSERT OR IGNORE INTO text_blobs VALUES(?,?)", (digest, text))
                        saved.add(digest)
                    result[name] = {"$text": digest}
            return result

        return json.dumps(value, default=encode, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

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
        return load_record(self.db, SessionSnapshot, row[0])

    def read_snapshot(self, session_id: str, corpus_snapshot_id: str = "demo-v1") -> SessionSnapshot:
        with self._mutex:
            row = self.db.execute("SELECT snapshot FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is not None:
                return load_record(self.db, SessionSnapshot, row[0])
        with self._transaction():
            self.db.execute("INSERT OR IGNORE INTO sessions(session_id,version,snapshot) VALUES(?,?,?)",
                            (session_id, 0, self._dump(SessionSnapshot(session_id, corpus_snapshot_id=corpus_snapshot_id))))
            return self._read(session_id)

    def get_snapshot(self, session_id: str) -> SessionSnapshot:
        """Read-only inspection; unlike serving, it never creates a session."""
        with self._mutex:
            return self._read(session_id)

    def register_document_set(self, blocks, *, source_metadata=None) -> DocumentSet:
        documents = DocumentSet.create(blocks, source_metadata)
        with self._transaction():
            self.db.execute("INSERT OR IGNORE INTO document_sets VALUES(?,?)",
                            (documents.document_set_id, self._dump(documents)))
        return documents

    def load_document_set(self, document_set_id: str) -> DocumentSet:
        with self._mutex:
            row = self.db.execute("SELECT payload FROM document_sets WHERE document_set_id=?",
                                  (document_set_id,)).fetchone()
            if row is None:
                raise KeyError(document_set_id)
            return load_record(self.db, DocumentSet, row[0])

    def committed_readings(self, snapshot: SessionSnapshot):
        """Project actual committed READ outcomes without another facts ledger."""
        turns = {turn.turn_id: turn for turn in snapshot.raw_history}
        readings = []
        with self._mutex:
            rows = self.db.execute("SELECT payload FROM attempts WHERE session_id=? AND stage='COMMITTED' "
                                   "ORDER BY rowid", (snapshot.session_id,)).fetchall()
            for row in rows:
                attempt = load_record(self.db, Attempt, row[0])
                outcome = attempt.reading_outcome
                if outcome is None or outcome.plan.task_state.current_turn_id not in turns:
                    continue
                turn = turns[outcome.plan.task_state.current_turn_id]
                if ((attempt.query, attempt.result.text) == (turn.user_text, turn.assistant_text)
                        and turn.evidence_set == outcome.plan.selected_set):
                    readings.append((attempt.attempt_id, outcome))
        return tuple(readings)

    def _write_snapshot(self, snapshot: SessionSnapshot, expected_version: int):
        payload = self._dump(snapshot)
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
            return load_record(self.db, Attempt, row[0])

    def find_request(self, session_id: str, request_id: str) -> Attempt | None:
        with self._mutex:
            row = self.db.execute("SELECT payload FROM attempts WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            return load_record(self.db, Attempt, row[0]) if row else None

    def pending_attempt(self, session_id: str) -> Attempt | None:
        with self._mutex:
            row = self.db.execute("SELECT payload FROM attempts WHERE session_id=? AND stage NOT IN ('COMMITTED','FAILED')", (session_id,)).fetchone()
            return load_record(self.db, Attempt, row[0]) if row else None

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
            row = self.db.execute("SELECT version,closed FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row["version"] != expected_version:
                raise VersionConflict("Snapshot changed before begin_attempt")
            if row["closed"]:
                raise StoreError("Session is closed")
            if self.pending_attempt(session_id):
                raise PendingAttempt("Session has an unresolved attempt; recover it before another turn")
            attempt = Attempt(uuid.uuid4().hex, session_id, request_id, request_hash, expected_version, query, created_ns=time.monotonic_ns())
            self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,?)", (attempt.attempt_id, session_id, request_id, attempt.stage, self._dump(attempt)))
            return attempt

    def save_attempt(self, attempt: Attempt) -> Attempt:
        with self._transaction():
            old = self.get_attempt(attempt.attempt_id)
            if not self._check_update(old, attempt):
                return old
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (attempt.stage, self._dump(attempt), attempt.attempt_id))
            return attempt

    def save_context_plan(self, attempt_id: str, context_plan: ContextPlan, *, initial_outcome=None):
        with self._transaction():
            if initial_outcome is not None:
                attempt = self.get_attempt(attempt_id)
                attempt = replace(attempt, context_plan=context_plan, reading_outcome=initial_outcome)
                self.db.execute("UPDATE attempts SET payload=? WHERE attempt_id=?",
                                (self._dump(attempt), attempt_id))
            self.db.execute("INSERT INTO turn_plans VALUES(?,?) ON CONFLICT(attempt_id) "
                            "DO UPDATE SET plan=excluded.plan",
                            (attempt_id, self._dump(context_plan)))

    def save_reading_progress(self, attempt_id, context_plan, progress, *, completed_step=None):
        """Persist the completed gap work and its resulting plan atomically."""
        with self._transaction():
            self.db.execute("INSERT INTO turn_plans VALUES(?,?) ON CONFLICT(attempt_id) "
                            "DO UPDATE SET plan=excluded.plan", (attempt_id, self._dump(context_plan)))
            self.db.execute("INSERT INTO reading_progress VALUES(?,?) ON CONFLICT(attempt_id) "
                            "DO UPDATE SET payload=excluded.payload", (attempt_id, self._dump(progress)))
            if completed_step is not None:
                if completed_step.attempt_id != attempt_id:
                    raise ValueError("completed reading step belongs to a different attempt")
                self.db.execute("INSERT INTO execution_steps VALUES(?,?,?,?,NULL) "
                    "ON CONFLICT(attempt_id,step_id) DO UPDATE SET stage=excluded.stage, payload=excluded.payload",
                    (attempt_id, completed_step.step_id, "COMPLETE", self._dump(completed_step)))

    def reading_progress(self, attempt_id):
        with self._mutex:
            row = self.db.execute("SELECT payload FROM reading_progress WHERE attempt_id=?", (attempt_id,)).fetchone()
            return load_record(self.db, ReadingProgress, row[0]) if row else None

    def save_prefix_owner(self, record: PrefixOwnerRecord, event: str, *, details=None):
        """A prefix record survives request completion and business epoch retire."""
        with self._transaction():
            previous = self.prefix_owner(record.owner_id)
            if previous is not None and (previous.prefix_input_ids, previous.sources,
                    previous.fingerprint) != (record.prefix_input_ids, record.sources, record.fingerprint):
                raise IdempotencyConflict("prefix owner reused for another causal source path")
            payload = self._dump(record)
            self.db.execute("INSERT INTO prefix_owners VALUES(?,?,?,?,?) ON CONFLICT(owner_id) "
                "DO UPDATE SET session_id=excluded.session_id,stage=excluded.stage,"
                "engine_restart_id=excluded.engine_restart_id,payload=excluded.payload",
                (record.owner_id, record.session_id, record.stage, record.receipt.engine_restart_id, payload))
            self.db.execute("INSERT INTO prefix_owner_events(owner_id,event,payload,created_ns) VALUES(?,?,?,?)",
                (record.owner_id, event, self._dump({"record": record, "details": details}), time.monotonic_ns()))

    def prefix_owner(self, owner_id):
        with self._mutex:
            row = self.db.execute("SELECT payload FROM prefix_owners WHERE owner_id=?", (owner_id,)).fetchone()
            return load_record(self.db, PrefixOwnerRecord, row[0]) if row else None

    def prefix_owners(self, session_id=None, *, active_only=False):
        with self._mutex:
            where, params = [], []
            if session_id is not None:
                where.append("session_id=?")
                params.append(session_id)
            if active_only:
                where.append("stage IN ('RETAINING','RETAINED','RELEASING')")
            sql = "SELECT payload FROM prefix_owners" + (" WHERE "+" AND ".join(where) if where else "")
            return tuple(load_record(self.db, PrefixOwnerRecord, row[0]) for row in
                         self.db.execute(sql+" ORDER BY rowid", params))

    def prefix_owner_events(self, owner_id):
        with self._mutex:
            return tuple({"event_id": row["event_id"], "event": row["event"],
                "created_ns": row["created_ns"], **load_record(self.db, dict, row["payload"])} for row in
                self.db.execute("SELECT event_id,event,payload,created_ns FROM prefix_owner_events "
                                "WHERE owner_id=? ORDER BY event_id", (owner_id,)))

    def record_prefix_access(self, *, owner_id, session_id, native_attempt_id, observed_ns, payload):
        """A recovered COMPLETE step is the same access, not another READ."""
        with self._transaction():
            self.db.execute("INSERT INTO prefix_accesses(owner_id,session_id,native_attempt_id,observed_ns,payload) "
                "VALUES(?,?,?,?,?) ON CONFLICT(owner_id,native_attempt_id) DO NOTHING",
                (owner_id, session_id, native_attempt_id, observed_ns, self._dump(payload)))

    def prefix_accesses(self, owner_id=None, *, session_id=None):
        with self._mutex:
            where, params = [], []
            if owner_id is not None:
                where.append("owner_id=?")
                params.append(owner_id)
            if session_id is not None:
                where.append("session_id=?")
                params.append(session_id)
            sql = "SELECT * FROM prefix_accesses" + (" WHERE "+" AND ".join(where) if where else "")
            return tuple({"access_id": row["access_id"], "owner_id": row["owner_id"],
                "session_id": row["session_id"], "native_attempt_id": row["native_attempt_id"],
                "observed_ns": row["observed_ns"], "payload": load_record(self.db, dict, row["payload"])}
                for row in self.db.execute(sql+" ORDER BY access_id", params))

    def complete_prefix_access(self, *, owner_id, native_attempt_id, result):
        """Finish one policy decision without registering another source READ."""
        with self._transaction():
            row = self.db.execute("SELECT payload FROM prefix_accesses "
                "WHERE owner_id=? AND native_attempt_id=?", (owner_id, native_attempt_id)).fetchone()
            if row is None:
                raise KeyError((owner_id, native_attempt_id))
            payload = load_record(self.db, dict, row[0])
            payload["policy_result"] = result
            self.db.execute("UPDATE prefix_accesses SET payload=? WHERE owner_id=? AND native_attempt_id=?",
                (self._dump(payload), owner_id, native_attempt_id))

    def add_reading_elapsed(self, attempt_id, elapsed_ms):
        """Accumulate measured execution segments across transport retries."""
        with self._transaction():
            attempt = self.get_attempt(attempt_id)
            updated = replace(attempt, reading_elapsed_ms=attempt.reading_elapsed_ms+elapsed_ms)
            self.db.execute("UPDATE attempts SET payload=? WHERE attempt_id=?",
                            (self._dump(updated), attempt_id))
            return updated.reading_elapsed_ms

    def save_retrieval(self, attempt_id, *, evidence, retrieval_priority, query_context,
                       retrieval_ms, auxiliary_usage):
        """Persist completed auxiliary work before the first reading generation."""
        with self._transaction():
            attempt = self.get_attempt(attempt_id)
            if attempt.stage != "STARTED":
                raise StoreError("retrieval belongs to the unplanned request stage")
            updated = replace(attempt, evidence=tuple(evidence), retrieval_priority=retrieval_priority,
                query_context=query_context, retrieval_ms=retrieval_ms, auxiliary_usage=auxiliary_usage)
            self.db.execute("UPDATE attempts SET payload=? WHERE attempt_id=?",
                            (self._dump(updated), attempt_id))
            return updated

    def context_plan(self, attempt_id: str) -> ContextPlan | None:
        with self._mutex:
            row = self.db.execute("SELECT plan FROM turn_plans WHERE attempt_id=?",
                                  (attempt_id,)).fetchone()
            return load_record(self.db, ContextPlan, row[0]) if row else None

    def save_execution_step(self, record: StepRecord, stage: str, generation=None):
        with self._transaction():
            self.db.execute("INSERT INTO execution_steps VALUES(?,?,?,?,?) "
                "ON CONFLICT(attempt_id,step_id) DO UPDATE SET stage=excluded.stage, "
                "payload=excluded.payload, generation=COALESCE(excluded.generation,generation)",
                (record.attempt_id, record.step_id, stage, self._dump(record),
                 self._dump(generation) if generation is not None else None))
            if generation is not None:
                self._enqueue(record.engine_epoch.business_session_id, record.engine_epoch,
                              "reading_step_complete", self.get_attempt(record.attempt_id).expected_version)

    def execution_steps(self, attempt_id: str):
        with self._mutex:
            return tuple(load_record(self.db, StepRecord, row[0]) for row in
                         self.db.execute("SELECT payload FROM execution_steps WHERE attempt_id=? "
                                         "ORDER BY rowid", (attempt_id,)))

    def execution_steps_with_stage(self, attempt_id):
        with self._mutex:
            return tuple((row["stage"], load_record(self.db, StepRecord, row["payload"])) for row in
                self.db.execute("SELECT stage,payload FROM execution_steps WHERE attempt_id=? ORDER BY rowid",
                                (attempt_id,)))

    def execution_step(self, attempt_id: str, step_id: str):
        with self._mutex:
            row = self.db.execute("SELECT stage,payload,generation FROM execution_steps "
                                  "WHERE attempt_id=? AND step_id=?", (attempt_id, step_id)).fetchone()
            if row is None:
                return None
            record = load_record(self.db, StepRecord, row["payload"])
            record = replace(record, engine_epoch=decode(ContextEpoch, record.engine_epoch),
                             input_messages=tuple(decode(Message, message) for message in record.input_messages),
                             receipt=decode(CompletionReceipt, record.receipt) if record.receipt is not None else None)
            generation = load_record(self.db, GenerationResult, row["generation"]) if row["generation"] else None
            return row["stage"], record, generation

    @staticmethod
    def _check_update(old: Attempt, attempt: Attempt) -> bool:
        if old.stage in ("COMMITTED", "FAILED"):
            raise StoreError("Terminal attempt is immutable")
        if (old.session_id, old.request_id, old.request_hash, old.expected_version) != (attempt.session_id, attempt.request_id, attempt.request_hash, attempt.expected_version):
            raise StoreError("Attempt identity is immutable")
        allowed = {"STARTED": {"PLANNED"}, "PLANNED": {"PREPARED"},
                   "PREPARING": {"PREPARING", "PREPARED"}, "PREPARED": {"SUBMITTED"},
                   "SUBMITTED": {"RESULT_READY"}, "RESULT_READY": set()}
        if attempt.stage not in allowed.get(old.stage, set()):
            if attempt == old:
                return False
            raise StoreError(f"Invalid or stale attempt transition {old.stage} -> {attempt.stage}")
        if old.plan is not None and (attempt.plan != old.plan or attempt.evidence != old.evidence
                                     or attempt.query_context != old.query_context):
            raise StoreError("A selected plan and its evidence are immutable")
        if old.result is not None and attempt.result != old.result:
            raise StoreError("A durable result is immutable")
        if old.epoch is not None and attempt.epoch != old.epoch:
            expected = replace(old.epoch, session_generation=attempt.epoch.session_generation) if attempt.epoch else None
            if old.epoch.session_generation != 0 or expected != attempt.epoch:
                raise StoreError("Prepared epoch identity is immutable")
        return True

    def _allocate_epoch(self, current: Attempt, restart_id: str, reason: str) -> Attempt:
        snap = self._read(current.session_id)
        self.db.execute("UPDATE sessions SET epoch_counter=epoch_counter+1 WHERE session_id=?", (current.session_id,))
        number = self.db.execute("SELECT epoch_counter FROM sessions WHERE session_id=?", (current.session_id,)).fetchone()[0]
        epoch = ContextEpoch(current.session_id, number, f"ctx-{uuid.uuid4().hex}", 0,
                             snap.current_epoch.epoch_no if snap.current_epoch else None,
                             reason=reason, engine_restart_id=restart_id)
        updated = replace(current, stage="PREPARING", epoch=epoch, new_epoch=True)
        self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (updated.stage, self._dump(updated), updated.attempt_id))
        return updated

    def plan_new_epoch(self, attempt: Attempt, restart_id: str, reason: str) -> Attempt:
        """Persist the chosen plan and epoch identity together, before prepare RPC."""
        with self._transaction():
            old = self.get_attempt(attempt.attempt_id)
            if old.stage != "STARTED" or attempt.stage != "PLANNED":
                raise StoreError("Only an unplanned attempt can prepare a new epoch")
            self._check_update(old, attempt)
            return self._allocate_epoch(attempt, restart_id, reason)

    def submit_attempt(self, attempt: Attempt, epoch: ContextEpoch) -> Attempt:
        """Persist the prepare ACK and submission barrier in one transaction.

        No external action occurs between PREPARED and SUBMITTED. Historical
        PREPARED records remain readable/recoverable through save_attempt.
        """
        with self._transaction():
            prepared = replace(attempt, stage="PREPARED", epoch=epoch)
            self._check_update(self.get_attempt(attempt.attempt_id), prepared)
            submitted = replace(prepared, stage="SUBMITTED")
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?",
                            (submitted.stage, self._dump(submitted), submitted.attempt_id))
            return submitted

    def allocate_epoch(self, attempt: Attempt, restart_id: str, reason: str) -> Attempt:
        with self._transaction():
            current = self.get_attempt(attempt.attempt_id)
            if current.epoch is not None:
                return current
            if current.stage != "PLANNED":
                raise StoreError("Only a selected plan may prepare an epoch")
            return self._allocate_epoch(current, restart_id, reason)

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
        with self._transaction():
            # A RESULT_READY plan/result is immutable. Recheck the durable
            # stage under the write transaction, without decoding it twice.
            row = self.db.execute("SELECT stage FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError(attempt_id)
            if row["stage"] == "COMMITTED":
                return self._read(attempt.session_id)
            if row["stage"] != "RESULT_READY":
                raise StoreError("Attempt changed before the commit transaction")
            snap = self._read(attempt.session_id)
            if snap.version != attempt.expected_version:
                raise VersionConflict("CAS failed: completed result remains durable for recovery")
            candidate, result, epoch = attempt.plan.candidate, attempt.result, attempt.epoch
            if not result.generation_complete or not result.receipt.generation_complete:
                raise StoreError("Partial output cannot become semantic history")
            if (candidate.corpus_snapshot_id != snap.corpus_snapshot_id
                    and (candidate.action != "REBUILD" or not attempt.new_epoch
                         or attempt.branch_reason != "corpus_snapshot_change")):
                raise StoreError("Corpus snapshot transition requires its own rebuilt epoch")
            if attempt.context_plan is not None:
                steps = attempt.reading_outcome.steps if attempt.reading_outcome is not None else ()
                reader_tokens = (len(candidate.input_ids)+len(result.output_ids)
                    +sum(len(step.input_ids)+len(step.output_ids) for step in steps))
                auxiliary = attempt.auxiliary_usage
                auxiliary_tokens = (0 if auxiliary is None else
                    auxiliary["input_tokens"]+auxiliary["output_tokens"]
                    if auxiliary["input_tokens"] is not None and auxiliary["output_tokens"] is not None else None)
                context_plan = replace(attempt.context_plan,
                    online_tokens_spent=reader_tokens+auxiliary_tokens if auxiliary_tokens is not None else None)
                attempt = replace(attempt, context_plan=context_plan,
                    reading_outcome=replace(attempt.reading_outcome, plan=context_plan)
                        if attempt.reading_outcome is not None else None)
                self.db.execute("INSERT INTO turn_plans VALUES(?,?) ON CONFLICT(attempt_id) "
                                "DO UPDATE SET plan=excluded.plan", (attempt_id, self._dump(context_plan)))
            version = snap.version + 1
            history_version = snap.raw_history_version + 1
            turn = TurnRecord(len(snap.raw_history) + 1, attempt.query, result.text, result.output_ids,
                              result.finish_reason, tuple(sorted(candidate.required_blocks)), attempt.query_context,
                              attempt.context_plan.selected_set if attempt.context_plan else None,
                              candidate.need_support,
                              attempt.context_plan.answer_plan if attempt.context_plan else None)
            ledger = ExecutionLedger(epoch.epoch_no, epoch.engine_session_id, history_version,
                                     candidate.messages + (Message("assistant", result.text),),
                                     candidate.input_ids, result.output_ids, candidate.visible_blocks,
                                     candidate.fingerprint.template_hash if candidate.fingerprint else "",
                                     candidate.fingerprint.model_revision if candidate.fingerprint else "",
                                     candidate.corpus_snapshot_id, candidate.evidence_spans,
                                     candidate.reference_map, result.receipt.inserted_cache_tokens)
            archive = {b.key: b for b in snap.archive}
            for block in attempt.evidence:
                prior = archive.get(block.key)
                archive[block.key] = merge_evidence_block(prior, block) if prior is not None else block
            updated = replace(snap, version=version, raw_history=snap.raw_history + (turn,),
                              raw_history_version=history_version, execution=ledger,
                              archive=tuple(archive[k] for k in sorted(archive)),
                              corpus_snapshot_id=candidate.corpus_snapshot_id,
                              current_epoch=replace(epoch, state="COMMITTED"), history_sync_pending=False,
                              force_new_epoch_for_recovery=False, must_rebuild=False,
                              reference_map=candidate.reference_map,
                              turns_since_rebuild=0 if candidate.action == "REBUILD" else snap.turns_since_rebuild + 1)
            self._write_snapshot(updated, snap.version)
            completed = replace(attempt, stage="COMMITTED")
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (completed.stage, self._dump(completed), attempt_id))
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
            self.db.execute("UPDATE attempts SET stage=?,payload=? WHERE attempt_id=?", (updated.stage, self._dump(updated), attempt_id))
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
