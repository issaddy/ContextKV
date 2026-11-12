"""Original evidence construction and short request-local citations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from .schema import EvidenceBlock, EvidenceKey


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class EvidencePolicy:
    max_total_model_tokens: int = 32768
    max_new_tokens: int = 1024
    system_prompt: str | None = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_evidence(text: str, document_id: str, *, corpus_id: str = "demo",
                  passage_id: str | None = None, version: str = "v1",
                  source_uri: str | None = None, title: str = "",
                  provenance: tuple[str, ...] = (),
                  dependencies: tuple[EvidenceKey, ...] = (),
                  anchors: tuple[str, ...] = ()) -> EvidenceBlock:
    source = source_uri or f"{corpus_id}://{document_id}"
    return EvidenceBlock(EvidenceKey(corpus_id, document_id,
                         passage_id or document_id, version, content_hash(text)),
                         source, title, text, provenance=provenance or (source,),
                         dependencies=dependencies, anchors=anchors)



def assign_references(keys):
    return tuple((f"S{index+1}", key) for index, key in enumerate(dict.fromkeys(keys)))
