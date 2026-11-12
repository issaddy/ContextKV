"""Generation callback input for conventional document QA adapters."""
from __future__ import annotations

from dataclasses import dataclass

from controller.schema import EvidenceBlock, EvidenceKey, EvidenceSpan, Message


@dataclass(frozen=True)
class DialogueCall:
    step_id: str
    messages: tuple[Message, ...]
    source_blocks: tuple[EvidenceBlock, ...]
    reference_map: tuple[tuple[str, EvidenceKey], ...]
    evidence_spans: tuple[EvidenceSpan, ...] = ()
