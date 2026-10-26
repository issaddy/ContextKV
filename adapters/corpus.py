"""Original-passage corpus metadata and explicit condition dependency adapter.

JSONL accepts BEIR ``_id/text/title`` rows or named ``passage_id/document_id``
rows. Version, source, span, parent section, anchors, and dependency IDs may live
under ``metadata``. Only those public fields are read; labels and answers are
never copied. Text is neither normalized nor split, summarized or rewritten.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from controller.evidence import EvidenceError, make_evidence
from controller.schema import EvidenceBlock, EvidenceKey


def configuration_anchors(text: str) -> tuple[str, ...]:
    """Extract literal code identifiers, not semantic labels or generated facts."""
    patterns = (r"`([A-Za-z][A-Za-z0-9_.-]{2,})`",
                r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b",
                r"(?<!\w)(--[A-Za-z][A-Za-z0-9-]+)\b")
    return tuple(sorted({match for pattern in patterns for match in re.findall(pattern, text)}))


class CorpusIndex:
    """Public evidence metadata indexed once, with version-specific dependencies."""
    def __init__(self, blocks: Iterable[EvidenceBlock], *, corpus_id: str, snapshot_id: str,
                 source_positions=None):
        self.corpus_id, self.snapshot_id = corpus_id, snapshot_id
        self._blocks = {block.key: block for block in blocks}
        self._passages: dict[str, list[EvidenceKey]] = {}
        for key in self._blocks:
            self._passages.setdefault(key.passage_id, []).append(key)
        # Positions order original passages; they are never used to splice bodies.
        groups = {}
        for key, (document, start, source) in (source_positions or {}).items():
            block = self._blocks[key]
            group = (key.corpus_id, document, key.version, source, block.parent_section)
            groups.setdefault(group, []).append((start, key))
        self._neighbors = {}
        for members in groups.values():
            ordered = [key for _, key in sorted(members)]
            for index, key in enumerate(ordered):
                self._neighbors[key] = tuple(ordered[max(0, index-1):index] + ordered[index+1:index+2])

    @classmethod
    def from_jsonl(cls, path: str | Path, *, corpus_id: str, snapshot_id: str) -> CorpusIndex:
        with Path(path).open(encoding="utf-8") as source:
            return cls.from_records((json.loads(line) for line in source if line.strip()),
                                    corpus_id=corpus_id, snapshot_id=snapshot_id)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], *,
                     corpus_id: str, snapshot_id: str) -> CorpusIndex:
        blocks: dict[tuple[str, str, str, str], EvidenceBlock] = {}
        identities: dict[tuple[str, str, str], list[EvidenceKey]] = {}
        pending: dict[EvidenceKey, tuple[Any, ...]] = {}
        positions = {}
        for row in records:
            metadata = {**row.get("metadata", {}), **row}
            passage_id = str(metadata.get("passage_id", metadata.get("_id", "")))
            document_id = str(metadata.get("document_id", passage_id))
            version = str(metadata.get("version", snapshot_id))
            text = row.get("text")
            if not passage_id or not document_id or not version or not isinstance(text, str) or not text:
                raise EvidenceError("corpus rows require passage/document ID, version and original text")
            source = str(metadata.get("source_uri") or metadata.get("source") or metadata.get("url") or
                         f"{corpus_id}://{snapshot_id}/{document_id}#{passage_id}")
            provenance = tuple(metadata.get("provenance") or (source, f"corpus_snapshot:{snapshot_id}"))
            block = make_evidence(text, document_id, corpus_id=corpus_id,
                passage_id=passage_id, version=version, source_uri=source,
                title=str(metadata.get("title") or ""), provenance=provenance,
                anchors=tuple(dict.fromkeys((*metadata.get("anchors", ()),
                                            *configuration_anchors(text)))))
            span = metadata.get("span")
            if span is not None:
                span = tuple(span)
                if len(span) != 2 or not 0 <= span[0] <= span[1]:
                    raise EvidenceError(f"invalid character span for {passage_id}")
            block = replace(block, span=span, parent_section=metadata.get("parent_section"))
            source_identity = str(metadata.get("source_uri") or metadata.get("source") or metadata.get("url") or "")
            if span is not None:
                positions[block.key] = (document_id, span[0], source_identity)
            elif snapshot_id.startswith("mtrag-"):
                # MT-RAG passage-level IDs encode ingestion offsets. Preserve
                # the existing evidence identity; only use the parent/order.
                match = re.fullmatch(r"(.+)-(\d+)-(\d+)", passage_id)
                if match:
                    positions[block.key] = (match[1], int(match[2]), source_identity)
            identity = (corpus_id, document_id, passage_id, version)
            if identity in blocks:
                raise EvidenceError(f"duplicate corpus passage/version: {identity}")
            blocks[identity] = block
            identities.setdefault((corpus_id, passage_id, version), []).append(block.key)
            pending[block.key] = tuple(metadata.get("dependencies", ())) + tuple(
                metadata.get("condition_passage_ids", ())) + tuple(metadata.get("header_passage_ids", ()))

        def dependency_key(value: Any, owner: EvidenceKey) -> EvidenceKey:
            if isinstance(value, Mapping):
                passage = str(value["passage_id"])
                wanted_version = str(value.get("version", owner.version))
                document = value.get("document_id")
                corpus = str(value.get("corpus_id", owner.corpus_id))
            else:
                passage, wanted_version, document, corpus = str(value), owner.version, None, owner.corpus_id
            matches = [key for key in identities.get((corpus, passage, wanted_version), ())
                       if document is None or key.document_id == str(document)]
            if len(matches) != 1:
                raise EvidenceError(f"missing or ambiguous dependency {value!r} for {owner.passage_id}")
            return matches[0]

        resolved = [replace(block, dependencies=tuple(dict.fromkeys(
                    dependency_key(value, block.key) for value in pending[block.key])))
                    for block in blocks.values()]
        return cls(resolved, corpus_id=corpus_id, snapshot_id=snapshot_id, source_positions=positions)

    @property
    def blocks(self) -> tuple[EvidenceBlock, ...]:
        return tuple(self._blocks.values())

    def enrich(self, contexts: Iterable[Any]) -> tuple[EvidenceBlock, ...]:
        """Add metadata/conditions to retrieved OnlineContext rows without changing bodies.

        Retrieval document IDs identify corpus passages. Distinct source document
        IDs and revision names come from this index. If two revisions have the
        same body, callers must select exact EvidenceKeys through ``resolve``.
        """
        return self.resolve(self.retrieved_keys(contexts))

    def retrieved_keys(self, contexts: Iterable[Any]) -> tuple[EvidenceKey, ...]:
        """Map original retrieval order to exact keys, excluding added dependencies.

        Call ``resolve(keys)`` to add supporting conditions without matching or
        hashing the bodies again. Duplicate retrieved passages keep their first
        rank; no supporting passage receives an invented retrieval rank.
        """
        from adapters.mtrag import clean_context

        keys = []
        for value in contexts:
            context = clean_context(value)
            keys.append(self._context_key(context))
        return tuple(dict.fromkeys(keys))

    def _context_matches(self, context):
        return [key for key in self._passages.get(context.document_id, ())
                if self._blocks[key].text == context.text]

    def _context_key(self, context):
        matches = self._context_matches(context)
        if len(matches) != 1:
            raise EvidenceError("metadata must match one exact retrieved body/version: "
                                + context.document_id)
        return matches[0]

    def contextualize(self, contexts):
        """Attach known source/title fields to a ranking pool without adding passages.

        Partial or ambiguous metadata leaves candidates unannotated. Attaching a
        source still requires one exact body/version match.
        Generation resolves selected evidence and its dependencies separately.
        """
        result = []
        for context in contexts:
            matches = self._context_matches(context)
            if len(matches) == 1:
                block = self._blocks[matches[0]]
                context = replace(context, source=block.source_uri, title=block.title)
            result.append(context)
        return tuple(result)

    def selection_group(self, context):
        """Exact content with declared applicability for retrieval slot allocation.

        This does not merge source identities or establish independent corroboration.
        Different revisions, parent scopes or condition dependencies remain separate.
        """
        matches = self._context_matches(context)
        if len(matches) != 1:
            return (context.document_id,)
        block = self._blocks[matches[0]]
        return (block.title, block.text, block.key.corpus_id, block.key.version,
                block.parent_section, frozenset(block.dependencies))

    def source_window(self, contexts):
        """Add each hit's immediate original neighbors once, without recursive growth.

        Windows stay within one source, document, version and declared section.
        Explicit document/span metadata or MT-RAG ingestion IDs supply order.
        Offsets need not describe identical overlapping text: each passage keeps
        its own complete body and citation identity. Neighbors have no rank score.
        """
        from adapters.mtrag import OnlineContext

        primary = self.retrieved_keys(contexts)
        seen, result = set(primary), []
        for key in primary:
            for neighbor in self._neighbors.get(key, ()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                block = self._blocks[neighbor]
                result.append(OnlineContext(block.key.passage_id, block.text, block.source_uri, block.title))
        return tuple(result)

    def resolve(self, passages: Iterable[str | EvidenceKey | EvidenceBlock]) -> tuple[EvidenceBlock, ...]:
        """Return requested passages plus complete explicit condition/header closure.

        A bare passage ID is accepted only when unambiguous. Supply an EvidenceKey
        or EvidenceBlock to select an exact revision; versions are never ordered
        lexically or replaced by an assumed latest version.
        """
        result: dict[EvidenceKey, EvidenceBlock] = {}
        for passage in passages:
            if isinstance(passage, EvidenceBlock):
                key = passage.key
            elif isinstance(passage, EvidenceKey):
                key = passage
            else:
                matches = self._passages.get(passage, ())
                if len(matches) != 1:
                    raise EvidenceError(f"missing or ambiguous corpus passage: {passage}")
                key = matches[0]
            if key not in self._blocks:
                raise EvidenceError(f"corpus does not contain requested revision: {key.passage_id}/{key.version}")
            result[key] = self._blocks[key]
        pending = list(result)
        while pending:
            key = pending.pop()
            for dependency in result[key].dependencies:
                if dependency not in result:
                    result[dependency] = self._blocks[dependency]
                    pending.append(dependency)
        return tuple(result.values())
