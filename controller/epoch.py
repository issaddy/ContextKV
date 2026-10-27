"""Retryable retirement happens strictly after the durable ledger transaction."""
import asyncio

from .engine_client import Engine, EngineError, UnknownOutcomeError
from .events import EventLog
from .session_store import SessionStore


class EpochCoordinator:
    def __init__(self, store: SessionStore, engine: Engine, events: EventLog):
        self.store, self.engine, self.events = store, engine, events
        self._lock = asyncio.Lock()

    async def flush(self, session_id: str | None = None) -> int:
        acknowledged = 0
        async with self._lock:
            for op in self.store.outbox(session_id):
                self.events.emit("epoch_retire_requested", operation=op)
                try:
                    ack = await self.engine.retire(op.engine_session_id, op.expected_generation,
                                                   op.operation_id, op.engine_restart_id, op.reason)
                    if (ack.engine_session_id, ack.session_generation, ack.engine_restart_id, ack.operation_id) != (
                            op.engine_session_id, op.expected_generation, op.engine_restart_id, op.operation_id):
                        raise EngineError("RETIRE_ACK_MISMATCH")
                    if not ack.acknowledged or ack.state != "RETIRED":
                        raise EngineError("RETIRE_PENDING")
                    self.store.ack_retire(op.operation_id)
                    self.events.emit("epoch_retire_acked", operation=op, acknowledgement=ack,
                                     physical_free_tokens=0)
                    acknowledged += 1
                except EngineError as exc:
                    if exc.code == "STALE_RESTART":
                        self.store.ack_retire(op.operation_id)
                        self.events.emit("epoch_retire_acked", operation=op, reason="engine_restart_invalidated_reference", physical_free_tokens=0)
                        acknowledged += 1
                    elif isinstance(exc, UnknownOutcomeError) or exc.code == "RETIRE_PENDING":
                        self.store.retire_failed(op.operation_id, str(exc))
                        self.events.emit("epoch_retire_retry", operation=op, error=str(exc))
                    else:
                        raise
        return acknowledged
