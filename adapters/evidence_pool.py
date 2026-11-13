"""Canonical source windows and their original-source dependencies."""
from __future__ import annotations

from dataclasses import replace
from functools import cached_property

from adapters.table_relations import extract_table_relations
from controller.schema import EvidencePool, EvidenceUnit, SourceSpan
from controller.source_windows import SOURCE_WINDOW_TOKENS, source_window_spans


def unit_identity(key, start, end):
    return f"{key.corpus_id}:{key.document_id}:{key.passage_id}:{key.version}:{start}:{end}"


class SourceDirectory:
    def __init__(self, corpus_index, tokenizer, *, unit_tokens=SOURCE_WINDOW_TOKENS):
        self.corpus_index, self.tokenizer = corpus_index, tokenizer
        self.unit_tokens = unit_tokens
        self._units = {}
        self._blocks = {block.key: block for block in corpus_index.blocks}

    def _base_units(self, key):
        if key not in self._units:
            self._units[key] = self._split(self._blocks[key])
        return self._units[key]

    def _source_units(self, key):
        units = self._base_units(key)
        dependencies = self._blocks[key].dependencies
        if not dependencies:
            return units
        identities = tuple(unit.unit_id for dependency in dependencies for unit in self._base_units(dependency))
        return tuple(replace(unit, dependencies=tuple(dict.fromkeys((*unit.dependencies, *identities))))
                     for unit in units)

    @cached_property
    def units_by_id(self):
        return {unit.unit_id: unit for key in self._blocks for unit in self._source_units(key)}

    def _split(self, block):
        metadata = self.corpus_index.source_metadata.get(block.key, {})
        base = block.span[0] if block.span else 0
        tables = extract_table_relations(block.text)
        # Keep headers and qualifiers as shared dependencies of their data rows.
        boundaries = {0, len(block.text)}
        for table in tables:
            anchors = (*table.headers, *table.source_notes,
                *((table.caption,) if table.caption is not None else ()),
                *(anchor for row in table.relations for anchor in (*row.parent_path, *row.notes)))
            for anchor in anchors:
                left = block.text.rfind("\n", 0, anchor.start) + 1
                newline = block.text.find("\n", anchor.end)
                right = len(block.text) if newline < 0 else newline + 1
                boundaries.update((left, right))
        ordered = sorted(boundaries)
        windows = tuple((start+left, start+right)
            for start, end in zip(ordered, ordered[1:])
            for left, right in source_window_spans(block.text[start:end], self.tokenizer,
                                                   tokens=self.unit_tokens))
        units = []
        for left, right in windows:
            text = block.text[left:right]
            unit_id = unit_identity(block.key, base+left, base+right)
            spans = (SourceSpan(f"span:{unit_id}", block.key, base+left, base+right),)
            applicability = tuple(sorted(metadata.get("applicability", {}).items()))
            units.append(EvidenceUnit(unit_id, spans, text, metadata.get("heading", block.title),
                metadata.get("table_id"), applicability=applicability,
                body_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
                document_family=metadata.get("document_family", block.key.document_id),
                valid_from=metadata.get("valid_from"), valid_to=metadata.get("valid_to"),
                published_at=metadata.get("published_at"), original_passage_ids=(block.key.passage_id,),
                version_order=tuple(metadata.get("version_order") or ()), current=metadata.get("current")))
        for table in tables:
            table_id = "table:"+unit_identity(block.key, base+table.span.start, base+table.span.end)
            for row in table.relations:
                anchors = (*table.headers, *row.parent_path, *row.notes, *table.source_notes,
                           *((table.caption,) if table.caption is not None else ()))
                dependency_ids = tuple(unit.unit_id for unit in units if any(
                    unit.source_spans[0].start < base+anchor.end and base+anchor.start < unit.source_spans[0].end
                    for anchor in anchors))
                units = [replace(unit, table_id=metadata.get("table_id") or table_id,
                    heading=" / ".join(filter(None, (block.title,
                        " / ".join(span.text for span in row.parent_path)))) or unit.heading,
                    dependencies=tuple(dict.fromkeys((*unit.dependencies,
                        *(identity for identity in dependency_ids if identity != unit.unit_id)))))
                    if unit.source_spans[0].start < base+row.row_span.end
                    and base+row.row_span.start < unit.source_spans[0].end else unit for unit in units]
        return tuple(units)

    def source_fetch(self, keys):
        result = {}
        def add(key):
            if key in result:
                return
            block = self._blocks[key]
            result[key] = block
            for dependency in block.dependencies:
                add(dependency)
        for key in keys:
            add(key)
        return tuple(result.values())

    def full_scope(self, keys):
        blocks = self.source_fetch(keys)
        return EvidencePool(tuple(unit for block in blocks for unit in self._source_units(block.key)),
                            blocks, "all_supplied_documents")
