"""Shared task, source, conversation and benchmark data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class EvidenceKey:
    corpus_id: str
    document_id: str
    passage_id: str
    version: str
    content_sha256: str


@dataclass(frozen=True)
class EvidenceBlock:
    key: EvidenceKey
    source_uri: str
    title: str
    text: str
    token_ids: tuple[int, ...] = ()
    span: tuple[int, int] | None = None
    parent_section: str | None = None
    provenance: tuple[str, ...] = ()
    dependencies: tuple[EvidenceKey, ...] = ()
    anchors: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("token_ids", "provenance", "dependencies", "anchors"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.span is not None:
            object.__setattr__(self, "span", tuple(self.span))


@dataclass(frozen=True)
class TextRef:
    turn_id: int
    speaker: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceSpan:
    span_id: str
    source_key: EvidenceKey
    start: int
    end: int


@dataclass(frozen=True)
class Condition:
    condition_id: str
    key_span: TextRef | None = None
    value_spans: tuple[TextRef, ...] = ()
    relation_type: str = "PREDICATE"
    predicate_span: TextRef | None = None
    target_refs: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    update: str = "ADD"


@dataclass(frozen=True)
class Need:
    need_id: str
    anchor_spans: tuple[TextRef, ...] = ()
    operation: str = "lookup"
    subject_refs: tuple[TextRef, ...] = ()
    condition_ids: tuple[str, ...] = ()
    predecessor_need_ids: tuple[str, ...] = ()
    scope: str = "selected_sources"


@dataclass(frozen=True)
class VersionIntent:
    mode: str = "current"
    document_families: tuple[str, ...] = ()
    version_ids: tuple[str, ...] = ()
    time_refs: tuple[TextRef, ...] = ()
    source_turns: tuple[int, ...] = ()
    temporal_relation: str = "at"
    resolved_version_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskState:
    task_id: str
    raw_question: str
    working_history: tuple[TextRef, ...] = ()
    conditions: tuple[Condition, ...] = ()
    needs: tuple[Need, ...] = ()
    version_intent: VersionIntent = field(default_factory=VersionIntent)
    source_scope: tuple[EvidenceKey, ...] = ()
    current_turn_id: int = 0
    query: str = ""
    effective_source_scope: tuple[EvidenceKey, ...] | None = None


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    source_spans: tuple[SourceSpan, ...]
    text: str
    heading: str = ""
    table_id: str | None = None
    dependencies: tuple[str, ...] = ()
    applicability: tuple[tuple[str, str], ...] = ()
    body_tokens: int = 0
    retrieval_score: float = 0.0
    document_family: str = ""
    valid_from: str | None = None
    valid_to: str | None = None
    published_at: str | None = None
    need_ids: tuple[str, ...] = ()
    original_passage_ids: tuple[str, ...] = ()
    version_order: tuple[int, ...] = ()
    current: bool | None = None


@dataclass(frozen=True)
class EvidencePool:
    units: tuple[EvidenceUnit, ...]
    source_blocks: tuple[EvidenceBlock, ...] = ()
    scope_id: str = "selected_sources"


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    need_id: str
    kind: str
    text: str = ""
    value: Any = None
    unit: str = ""
    conditions: tuple[str, ...] = ()
    source_spans: tuple[SourceSpan, ...] = ()
    operation: str = "extract"
    parent_claim_ids: tuple[str, ...] = ()
    entity: str = ""
    relation: str = ""
    version: str = ""
    shared_need_ids: tuple[str, ...] = ()

    def serves(self, need_id):
        return need_id == self.need_id or need_id in self.shared_need_ids


@dataclass(frozen=True)
class Message:
    role: str
    content: str


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
