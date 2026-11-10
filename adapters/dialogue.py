"""Shared request/result types for conventional document QA adapters."""
from __future__ import annotations

from dataclasses import dataclass

from controller.schema import EvidenceBlock, EvidenceKey, EvidenceSpan, Message


@dataclass(frozen=True)
class DialogueCall:
    step_id: str
    stage: str
    messages: tuple[Message, ...]
    output_budget: int
    source_blocks: tuple[EvidenceBlock, ...]
    reference_map: tuple[tuple[str, EvidenceKey], ...]
    evidence_spans: tuple[EvidenceSpan, ...] = ()


@dataclass(frozen=True)
class DialogueResult:
    answer: str
    condensed_query: str
    source_blocks: tuple[EvidenceBlock, ...]
    reference_map: tuple[tuple[str, EvidenceKey], ...]
    calls: tuple[DialogueCall, ...]
    framework_version: str
