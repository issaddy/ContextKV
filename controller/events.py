"""Append-only events with explicit measurement provenance."""
import json
from pathlib import Path
import threading
import time
import uuid

from .serialization import to_jsonable


class EventLog:
    def __init__(self, path: str | Path | None = None, *, run_id: str | None = None,
                 config_hash: str = "", engine_kind: str = "mock"):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self.config_hash = config_hash
        self.engine_kind = engine_kind
        self._lock = threading.Lock()

    def emit(self, event: str, **fields):
        entry = {"event": event, "run_id": self.run_id, "config_hash": self.config_hash,
                 "engine_kind": self.engine_kind, "monotonic_timestamp_ns": time.monotonic_ns(),
                 **to_jsonable(fields)}
        if self.path:
            line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return entry
