"""Append-only events with explicit measurement provenance."""
import json
import logging
from pathlib import Path
import threading
import time
import uuid

from .serialization import to_jsonable


class EventLog:
    def __init__(self, path: str | Path | None = None, *, run_id: str | None = None,
                 config_hash: str = "", engine_kind: str = "mock"):
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex
        self.config_hash = config_hash
        self.engine_kind = engine_kind
        self._lock = threading.Lock()
        self.last_error = None
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                self.report_error(error)

    def emit(self, event: str, **fields):
        try:
            entry = self._write(event, fields)
        except Exception as error:
            self.report_error(error)
            return None
        self.last_error = None
        return entry

    def report_error(self, error):
        message = f"{type(error).__name__}: {error}"
        if message != self.last_error:
            logging.getLogger(__name__).error("Diagnostic output failed: %s", message)
        self.last_error = message

    def _write(self, event, fields):
        entry = {"event": event, "run_id": self.run_id, "config_hash": self.config_hash,
                 "engine_kind": self.engine_kind, "monotonic_timestamp_ns": time.monotonic_ns(),
                 **to_jsonable(fields)}
        if self.path:
            line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return entry
