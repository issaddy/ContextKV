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
    revisit_mode: str = "conservative"
    revisit_history_turns: int = 3
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.max_total_model_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("token budgets must be positive")
        if self.layout not in {"stable_key", "contextpilot"}:
            raise ValueError("layout must be stable_key or contextpilot")
        if self.revisit_mode not in {"explicit", "conservative"}:
            raise ValueError("revisit_mode must be explicit or conservative")
        if self.revisit_history_turns < 1:
            raise ValueError("revisit_history_turns must be positive")
        if self.system_prompt is not None and not self.system_prompt.strip():
            raise ValueError("system_prompt must be nonempty when provided")


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


def _mentioned(term: str, query: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", term):
        return term.casefold() in query.casefold()
    return bool(term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", query, re.I))


# These are cues in the current question, not dataset annotations or gold labels.
_REVISIT = re.compile(
    r"\b(previous(?:ly)?|earlier|above|aforementioned|that|those|same)\b|"
    r"前面|前文|之前|刚才|上述|那个|那些|这一|这个|这些|同一", re.I)
_TOPICS = {
    "condition": re.compile(r"\b(condition|restriction|limitation|limit|exception|only|unless)s?\b|"
                            r"条件|限制|例外|仅当|仅限|除非", re.I),
    "timeout": re.compile(r"\b(timeout|time-out|deadline)s?\b|超时", re.I),
    "version": re.compile(r"\b(version|release|supported)s?\b|版本|支持", re.I),
    "configuration": re.compile(r"\b(configuration|config|setting|parameter)s?\b|配置|参数", re.I),
}


def _natural_revisits(query: str, archive: Mapping[EvidenceKey, EvidenceBlock],
                      snapshot: SessionSnapshot, policy: EvidencePolicy,
                      current: set[EvidenceKey]) -> dict[EvidenceKey, str]:
    """Resolve explicit anaphora only when a recent committed turn disambiguates it.

    A semantic topic must occur in the evidence itself. Ties are left unresolved;
    recency or a dataset's follow-up label alone is never enough to invent a match.
    """
    prior_version = bool(_PRIOR_VERSION.search(query))
    if policy.revisit_mode != "conservative" or not (_REVISIT.search(query) or prior_version):
        return {}
    topics = [name for name, pattern in _TOPICS.items() if pattern.search(query)]
    current_groups = {(k.corpus_id, k.document_id, k.passage_id) for k in current}
    for turn in reversed(snapshot.raw_history[-policy.revisit_history_turns:]):
        keys = [key for key in turn.evidence_keys if key in archive and
                (prior_version or key in current or (key.corpus_id, key.document_id, key.passage_id)
                 not in current_groups)]
        if not keys:
            continue
        if prior_version:
            previous = [key for key in keys if key not in current and
                        (key.corpus_id, key.document_id, key.passage_id) in current_groups]
            if len(previous) == 1:
                return {previous[0]: f"natural_revisit:turn={turn.turn_id}:prior_revision"}
            if previous:
                return {}
            continue
        if topics:
            matched = [key for key in keys if all(_TOPICS[topic].search(
                archive[key].title + "\n" + archive[key].text) for topic in topics)]
            # Multiple passages from the same document/section can jointly state
            # a rule; the explicit corpus dependencies provide the rest.
            if len(matched) == 1:
                return {matched[0]: f"natural_revisit:turn={turn.turn_id}:topic={','.join(topics)}"}
            if matched:
                return {}
        elif len(keys) == 1:
            return {keys[0]: f"natural_revisit:turn={turn.turn_id}:unique_antecedent"}
        else:
            return {}
    return {}


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
                              policy: EvidencePolicy | None = None
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
    folded = query.casefold()
    current_groups = {(key.corpus_id,key.document_id,key.passage_id) for key in current}
    anchor_matches: dict[str, list[EvidenceBlock]] = {}
    for block in archive.values():
        group = (block.key.corpus_id,block.key.document_id,block.key.passage_id)
        if block.key in current or group not in current_groups:
            for anchor in block.anchors:
                if _mentioned(anchor, folded):
                    anchor_matches.setdefault(anchor, []).append(block)
        # A version plus its named document is an explicit revisit, not a
        # similarity-based restoration of arbitrary historical evidence.
        names = (block.key.document_id, block.title)
        if _mentioned(block.key.version,folded) and any(
                name and name.casefold() in folded for name in names):
            reasons.setdefault(block.key, "explicit_document_version")
    for matches in anchor_matches.values():
        # A configuration identifier can occur in unrelated products. Scope it
        # using explicit source/version names or the current retrieval, rather
        # than treating lexical agreement as proof of the same referent.
        named = [block for block in matches if any(_mentioned(name, folded)
                 for name in (block.key.document_id, block.title) if name)]
        scoped = named or matches
        versions = [block for block in scoped if _mentioned(block.key.version, folded)]
        if versions:
            scoped = versions
        current_scopes = {(block.key.corpus_id, block.key.document_id, block.key.version)
                          for block in scoped if block.key in current}
        if current_scopes and not versions:
            scoped = [block for block in scoped if
                      (block.key.corpus_id, block.key.document_id, block.key.version) in current_scopes]
        elif not named and len({(block.key.corpus_id, block.key.document_id)
                               for block in scoped}) > 1:
            continue
        for block in scoped:
            reasons.setdefault(block.key, "configuration_anchor")
    for key, reason in _natural_revisits(query, archive, snapshot, policy, current).items():
        reasons.setdefault(key, reason)
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
