"""Exact evidence identity, immutable references and explicit revisit recovery."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Mapping

from .schema import EvidenceBlock, EvidenceKey, SessionSnapshot


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class EvidencePolicy:
    max_total_model_tokens: int = 32768
    max_new_tokens: int = 512
    allow_version_comparison: bool = False
    explicit_keys: tuple[EvidenceKey, ...] = ()
    reference_prefix_length: int = 10
    layout: str = "stable_key"
    system_prompt: str | None = None
    table_relations: bool = False
    table_annotation_style: str = "all_rows"
    table_structure_mode: str = "legacy"

    def __post_init__(self) -> None:
        if self.max_total_model_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("token budgets must be positive")
        if self.layout not in {"stable_key", "contextpilot"}:
            raise ValueError("layout must be stable_key or contextpilot")
        if self.system_prompt is not None and not self.system_prompt.strip():
            raise ValueError("system_prompt must be nonempty when provided")
        if self.table_annotation_style not in {"all_rows", "focus_refs"}:
            raise ValueError("table_annotation_style must be all_rows or focus_refs")
        if self.table_structure_mode not in {"legacy", "explicit"}:
            raise ValueError("table_structure_mode must be legacy or explicit")


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


def validate_block(block: EvidenceBlock) -> None:
    if content_hash(block.text) != block.key.content_sha256:
        raise EvidenceError("evidence content hash mismatch")
    if not block.text or not block.source_uri or not block.provenance:
        raise EvidenceError("evidence requires original text, source URI and provenance")
    if not all((block.key.corpus_id, block.key.document_id,
                block.key.passage_id, block.key.version)):
        raise EvidenceError("evidence identity and version must be nonempty")
    if block.span is not None and not 0 <= block.span[0] <= block.span[1]:
        raise EvidenceError("invalid corpus span")


def evidence_index(blocks: Iterable[EvidenceBlock], *,
                   committed: Iterable[EvidenceBlock] = ()) -> dict[EvidenceKey, EvidenceBlock]:
    """Validate incoming evidence once and extend a trusted committed archive."""
    result = {block.key: block for block in committed}
    for block in blocks:
        validate_block(block)
        prior = result.get(block.key)
        if prior is not None and prior != block:
            raise EvidenceError("same identity has inconsistent source/provenance metadata")
        result[block.key] = block
    return result


def identity_hash(key: EvidenceKey) -> str:
    return hashlib.sha256(json.dumps([key.corpus_id, key.document_id,
        key.passage_id, key.version, key.content_sha256],
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def assign_references(keys: Iterable[EvidenceKey],
                      existing: Iterable[tuple[str, EvidenceKey]] = (), *,
                      prefix_length: int = 10) -> tuple[tuple[str, EvidenceKey], ...]:
    """Extend only new IDs on collision; previously published IDs never change."""
    if not 1 <= prefix_length <= 64:
        raise ValueError("reference prefix length must be in [1, 64]")
    mapping: dict[str, EvidenceKey] = {}
    inverse: dict[EvidenceKey, str] = {}
    for reference, key in existing:
        if reference in mapping and mapping[reference] != key:
            raise EvidenceError("existing reference collision")
        if key in inverse and inverse[key] != reference:
            raise EvidenceError("existing evidence has multiple reference IDs")
        mapping[reference], inverse[key] = key, reference
    for key in sorted(set(keys)):
        if key in inverse:
            continue
        digest = identity_hash(key)
        length = prefix_length
        ref = "E_" + digest[:length]
        while ref in mapping and mapping[ref] != key:
            length += 1
            if length > len(digest):
                raise EvidenceError("full evidence identity hash collision")
            ref = "E_" + digest[:length]
        mapping[ref], inverse[key] = key, ref
    return tuple(sorted(mapping.items()))


_PRIOR_VERSION = re.compile(r"\b(?:old|older|previous|earlier)\s+versions?\b|"
                            r"旧版本|旧版|先前版本|以前的版本", re.I)


def allows_version_comparison(query: str, policy: EvidencePolicy) -> bool:
    return policy.allow_version_comparison or bool(_PRIOR_VERSION.search(query)) or bool(re.search(
        r"\b(compare|comparison|versus|vs\.?|difference)\b|对比|比较|差异|新旧", query, re.I))


def select_shared_required_evidence(query: str, retrieved: Iterable[EvidenceBlock],
                                   snapshot: SessionSnapshot,
                                   policy: EvidencePolicy | None = None
                                   ) -> tuple[EvidenceBlock, ...]:
    retrieved = tuple(retrieved)
    archive = evidence_index(retrieved, committed=snapshot.archive)
    return resolve_required_evidence(query, (block.key for block in retrieved),
                                     archive, snapshot, policy)


def resolve_required_evidence(query: str, current: Iterable[EvidenceKey],
                              archive: Mapping[EvidenceKey, EvidenceBlock],
                              snapshot: SessionSnapshot,
                              policy: EvidencePolicy | None = None
                              ) -> tuple[EvidenceBlock, ...]:
    """Resolve revisits and dependencies using the already validated turn index."""
    return resolve_required_evidence_with_reasons(query, current, archive, snapshot, policy)[0]


def resolve_required_evidence_with_reasons(query: str, current: Iterable[EvidenceKey],
                              archive: Mapping[EvidenceKey, EvidenceBlock],
                              snapshot: SessionSnapshot,
                              policy: EvidencePolicy | None = None, *,
                              revisit_turns: tuple[int, ...] | None = None
                              ) -> tuple[tuple[EvidenceBlock, ...],
                                         tuple[tuple[EvidenceKey, str], ...]]:
    policy = policy or EvidencePolicy()
    current = set(current)
    reasons = {key: "current_retrieval" for key in current}
    reasons.update((key, "explicit_key") for key in policy.explicit_keys)
    references = dict(snapshot.reference_map)
    if snapshot.execution:
        references.update(snapshot.execution.reference_map)
    for ref in re.findall(r"\bE_[a-zA-Z0-9]+\b", query):
        if ref not in references:
            raise EvidenceError(f"unknown evidence reference: {ref}")
        reasons[references[ref]] = "explicit_reference:" + ref
    # Interpretation owns historical scope. This layer resolves exact identities
    # and dependencies; it does not infer an object from anchors or topic words.
    for ordinal, turn in enumerate(snapshot.raw_history, 1):
        if ordinal in (revisit_turns or ()):
            for key in turn.evidence_keys:
                reasons.setdefault(key, f"selected_revisit:turn={ordinal}")
    if not allows_version_comparison(query, policy):
        # An explicitly requested archived revision replaces a conflicting
        # retrieval hit for the SAME passage. Other documents and declared
        # dependencies remain subject to the existing visibility checks.
        explicit = {}
        for key, reason in reasons.items():
            if reason == "explicit_key" or reason.startswith(("explicit_reference:", "selected_revisit:")):
                group = (key.corpus_id, key.document_id, key.passage_id)
                explicit.setdefault(group, set()).add(key)
        for key in tuple(reasons):
            choices = explicit.get((key.corpus_id, key.document_id, key.passage_id), ())
            if reasons[key] == "current_retrieval" and len(choices) == 1 and key not in choices:
                del reasons[key]
    required = set(reasons)
    pending = list(required)
    while pending:
        key = pending.pop()
        if key not in archive:
            raise EvidenceError(f"required evidence unavailable in archive: {key}")
        for dependency in archive[key].dependencies:
            if dependency not in required:
                required.add(dependency)
                reasons[dependency] = "dependency_of:" + key.passage_id
                pending.append(dependency)
    selected = tuple(archive[key] for key in sorted(required))
    if not allows_version_comparison(query, policy) and version_conflicts(
            (b.key for b in selected), (b.key for b in selected)):
        raise EvidenceError("required evidence contains conflicting versions; explicit comparison required")
    return selected, tuple(sorted(reasons.items()))


def version_conflicts(visible: Iterable[EvidenceKey],
                     required: Iterable[EvidenceKey]) -> bool:
    requested: dict[tuple[str, str, str], set[EvidenceKey]] = {}
    for key in required:
        requested.setdefault((key.corpus_id, key.document_id, key.passage_id), set()).add(key)
    for key in visible:
        group = requested.get((key.corpus_id, key.document_id, key.passage_id))
        if group and any(other != key for other in group):
            return True
    return False
