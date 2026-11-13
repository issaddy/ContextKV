"""Conversation and request types used by document QA and benchmark scoring."""
from __future__ import annotations

from dataclasses import dataclass
from .context_schema import EvidenceKey, EvidenceBlock, EvidenceUnit, TaskState, Message


def _tuples(instance, *names: str) -> None:
    for name in names:
        object.__setattr__(instance, name, tuple(getattr(instance, name)))


@dataclass(frozen=True)
class EvidenceSpan:
    """Body location, original source coordinates and citation ID."""
    key: EvidenceKey
    message_index: int
    start: int
    end: int
    source_uri: str
    provenance: tuple[str, ...]
    reference_id: str
    source_start: int | None = None
    source_end: int | None = None

    def __post_init__(self) -> None:
        _tuples(self, "provenance")


@dataclass(frozen=True)
class QueryContext:
    """Derived conversational intent, never an evidence source or a raw QA edit."""
    query: str
    relation: str
    history_turns: tuple[int, ...] = ()
    task_state: TaskState | None = None

    def __post_init__(self) -> None:
        _tuples(self, "history_turns")


@dataclass(frozen=True)
class TurnRecord:
    turn_id: int
    user_text: str | None
    assistant_text: str | None
    assistant_output_ids: tuple[int, ...] = ()
    finish_reason: str = "stop"
    evidence_keys: tuple[EvidenceKey, ...] = ()
    query_context: QueryContext | None = None

    def __post_init__(self) -> None:
        _tuples(self, "assistant_output_ids", "evidence_keys")


def history_messages(history: tuple[TurnRecord, ...]) -> tuple[Message, ...]:
    """Project only messages that actually occurred, retaining their boundaries."""
    return tuple(Message(role, text) for turn in history for role, text in (
        ("user", turn.user_text), ("assistant", turn.assistant_text)) if text is not None)


def from_history_messages(messages: tuple[Message, ...]) -> tuple[TurnRecord, ...]:
    """Import an original transcript without merging consecutive speakers."""
    history = []
    for message in messages:
        if message.role == "user":
            history.append(TurnRecord(len(history) + 1, message.content, None))
        elif message.role == "assistant":
            if history and history[-1].assistant_text is None:
                previous = history[-1]
                history[-1] = TurnRecord(previous.turn_id, previous.user_text, message.content)
            else:
                history.append(TurnRecord(len(history) + 1, None, message.content))
        else:
            raise ValueError(f"conversation history requires user/assistant roles, got {message.role!r}")
    return tuple(history)


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    version: int = 0
    raw_history: tuple[TurnRecord, ...] = ()
    corpus_snapshot_id: str = "default"


@dataclass(frozen=True)
class OnlineMessage:
    speaker: str
    text: str

    @property
    def role(self) -> str:
        return "user" if self.speaker == "user" else "assistant"

    @property
    def content(self) -> str:
        return self.text


@dataclass(frozen=True)
class OnlineContext:
    document_id: str
    text: str
    source: str = ""
    title: str = ""
    score: float | None = None

    @property
    def content_sha256(self) -> str:
        from .evidence import content_hash
        return content_hash(self.text)


@dataclass(frozen=True)
class OnlineTurn:
    conversation_id: str
    task_id: str
    turn: int
    collection: str
    query: str
    history: tuple[OnlineMessage, ...]
    contexts: tuple[OnlineContext, ...]
    protocol: str
    query_context: QueryContext | None = None
    supporting_contexts: tuple[OnlineContext, ...] = ()
    retrieval_contexts: tuple[OnlineContext, ...] | None = None

    @property
    def question(self) -> str:
        return self.query

    @property
    def user_questions(self) -> tuple[str, ...]:
        return tuple(x.text for x in self.history if x.speaker == "user") + (self.query,)

    @property
    def branch_reason(self) -> str | None:
        return "reference_history_sync" if self.protocol == "P1" and self.history else None
