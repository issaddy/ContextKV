"""Committed dialogue and task state shared by online retrieval entry points."""
from __future__ import annotations

from dataclasses import dataclass
from controller.schema import QueryContext


@dataclass(frozen=True)
class PriorContext:
    snapshot_version: int | None = None
    interpretations: tuple[QueryContext | None, ...] = ()


def dialogue_turns(turn, turn_ids=None):
    """Literal completed QA for discourse resolution, not factual evidence.

    Default to the existing recent three-turn window. Explicit turn IDs also
    support older selected referents without copying unrelated answer history.
    The caller's OnlineTurn already contains only permitted actual/P1 history.
    """
    count = len(turn.history) // 2
    ids = range(max(1, count - 2), count + 1) if turn_ids is None else sorted(set(turn_ids))
    return [{"turn": n, "question": turn.history[2*n-2].text,
             "answer": turn.history[2*n-1].text} for n in ids if 1 <= n <= count]


def snapshot_context(turn, snapshot):
    if snapshot is None:
        if turn.history:
            raise ValueError("missing committed snapshot for nonempty generated history")
        return PriorContext()
    history = tuple((t.user_text, t.assistant_text) for t in snapshot.raw_history)
    supplied = tuple((turn.history[i].text, turn.history[i+1].text)
                     for i in range(0, len(turn.history), 2))
    if history != supplied:
        raise ValueError("retrieval history differs from committed snapshot")
    return PriorContext(snapshot.version, tuple(t.query_context for t in snapshot.raw_history))


def controller_context_provider(store, run_prefix):
    def provide(turn):
        try:
            snapshot = store.get_snapshot(f"{run_prefix}:{turn.conversation_id}")
        except KeyError:
            snapshot = None
        return snapshot_context(turn, snapshot)
    return provide


def native_context_provider(baseline):
    return lambda turn: snapshot_context(turn, baseline.snapshots.get(turn.conversation_id))
