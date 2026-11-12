"""Shared query, source-address and document-computation contracts."""
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
