"""Version-aware source directory and quota-balanced evidence candidate pools."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import re
import time

import numpy as np

from controller.context_schema import EvidencePool, EvidenceUnit, SourceSpan
from adapters.table_relations import extract_table_relations


@dataclass(frozen=True)
class PoolProfile:
    current_top_k: int = 50
    per_need_top_k: int = 20
    archive_top_k: int = 12
    primary_pool_size: int = 96
    unit_tokens: int = 2048


def unit_identity(key, start, end):
    return f"{key.corpus_id}:{key.document_id}:{key.passage_id}:{key.version}:{start}:{end}"


class SourceDirectory:
    """Original source metadata remains distinct for each version and identity."""
    def __init__(self, corpus_index, search, tokenizer, *, task_interpreter=None,
                 reference_date=None, profile=None, archive_encoder=None):
        self.corpus_index, self.search, self.tokenizer = corpus_index, search, tokenizer
        self.task_interpreter = task_interpreter
        self.archive_encoder = archive_encoder
        self._archive_vectors = {}
        self.reference_date = reference_date or date.today().isoformat()
        self.profile = profile or PoolProfile()
        self._units = {}
        self._blocks = {b.key: b for b in corpus_index.blocks}
        for block in corpus_index.blocks:
            self._units[block.key] = self._split(block)
        self._unit_metadata = {key: dict(corpus_index.source_metadata.get(key, {})) for key in self._units}
        for key, units in tuple(self._units.items()):
            dependency_ids = tuple(unit.unit_id for dependency in self._blocks[key].dependencies
                                   for unit in self._units[dependency])
            self._units[key] = tuple(replace(unit, dependencies=tuple(dict.fromkeys(
                (*unit.dependencies, *dependency_ids)))) for unit in units)
        self.units_by_id = {unit.unit_id: unit for units in self._units.values() for unit in units}

    def _split(self, block):
        metadata = self.corpus_index.source_metadata.get(block.key, {})
        base = block.span[0] if block.span else 0
        # Keep source rows intact; long prose prefers literal sentence boundaries.
        regions = [(match.start(), match.end()) for match in re.finditer(r"[^\n]+(?:\n|$)", block.text)]
        units = []
        for start, end in regions:
            offsets = self.tokenizer(block.text[start:end], add_special_tokens=False,
                                     return_offsets_mapping=True)["offset_mapping"]
            sentence_ends = {match.end() for match in re.finditer(r"[.!?。！？;；](?:\s+|$)", block.text[start:end])}
            first = 0
            while first < len(offsets):
                last = min(first+self.profile.unit_tokens, len(offsets))
                boundaries = [index+1 for index in range(first, last)
                    if offsets[index][1] in sentence_ends and index-first >= self.profile.unit_tokens//2]
                if last < len(offsets) and boundaries:
                    last = boundaries[-1]
                section = offsets[first:last]
                left = start + section[0][0]
                right = start + section[-1][1]
                text = block.text[left:right]
                unit_id = unit_identity(block.key, base+left, base+right)
                spans = (SourceSpan(f"span:{unit_id}", block.key, base+left, base+right),)
                applicability = tuple(sorted(metadata.get("applicability", {}).items()))
                units.append(EvidenceUnit(unit_id, spans, text, metadata.get("heading", block.title),
                    metadata.get("table_id"), applicability=applicability,
                    body_tokens=len(self.tokenizer(text, add_special_tokens=False)["input_ids"]),
                    document_family=metadata.get("document_family", block.key.document_id),
                    valid_from=metadata.get("valid_from"), valid_to=metadata.get("valid_to"),
                    published_at=metadata.get("published_at"), original_passage_ids=(block.key.passage_id,),
                    version_order=tuple(metadata.get("version_order") or ()), current=metadata.get("current")))
                if last == len(offsets):
                    break
                overlap = 0 if offsets[last-1][1] in sentence_ends else min(32, self.profile.unit_tokens//8)
                first = max(first+1, last-overlap)
        tables = extract_table_relations(block.text, mode="explicit")
        for table in tables:
            table_id = f"{block.key.document_id}:{block.key.passage_id}:{block.key.version}:table:{base+table.span.start}"
            for row in table.relations:
                anchors = (*table.headers, *row.parent_path, *row.notes, *table.source_notes,
                           *((table.caption,) if table.caption is not None else ()))
                dependency_ids = tuple(unit.unit_id for unit in units if any(
                    unit.source_spans[0].start < base+anchor.end and base+anchor.start < unit.source_spans[0].end
                    for anchor in anchors))
                units = [replace(unit, table_id=metadata.get("table_id") or table_id,
                    heading=" / ".join(span.text for span in row.parent_path) or unit.heading,
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

    def add_archive(self, blocks):
        for block in blocks:
            metadata = self.corpus_index.source_metadata.get(block.key, {})
            if block.key not in self._blocks or self._unit_metadata[block.key] != metadata:
                self._blocks[block.key] = block
                self._units[block.key] = self._split(block)
                self._unit_metadata[block.key] = dict(metadata)
        for block in blocks:
            dependency_ids = tuple(unit.unit_id for dependency in block.dependencies for unit in self._units[dependency])
            self._units[block.key] = tuple(replace(unit, dependencies=tuple(dict.fromkeys(
                (*unit.dependencies, *dependency_ids)))) for unit in self._units[block.key])
        self.units_by_id = {unit.unit_id: unit for group in self._units.values() for unit in group}

    def archive_lookup(self, keys):
        return tuple(self._blocks[key] for key in keys)

    def archive_search(self, query, keys, *, top_k=12):
        keys = tuple(keys)
        if not keys or self.archive_encoder is None:
            return keys[:top_k]
        for key in keys:
            if key not in self._archive_vectors:
                vectors = np.asarray(self.archive_encoder([u.text for u in self._units[key]]), dtype=np.float32)
                vector = vectors.mean(axis=0)
                self._archive_vectors[key] = vector / np.linalg.norm(vector)
        query_vector = np.asarray(self.archive_encoder([query])[0], dtype=np.float32)
        query_vector /= np.linalg.norm(query_vector)
        return tuple(sorted(keys, key=lambda key: -float(self._archive_vectors[key] @ query_vector)))[:top_k]

    def candidate_search(self, query, *, top_k=50):
        return tuple(self.search(query, top_k=top_k))

    def _temporal_sources(self, state, memory):
        intent = state.version_intent
        if intent.mode not in {"as_of", "compare"} or intent.temporal_relation == "at":
            return None
        metadata = self.corpus_index.source_metadata
        anchors = []
        for row in memory.committed_history:
            if row.turn_id not in intent.source_turns:
                continue
            keys = list(row.evidence_keys)
            prior = next((prior for prior in reversed(memory.prior_task_states)
                          if prior.current_turn_id == row.turn_id), None)
            if prior is not None and prior.version_intent.mode == "current":
                keys = [key for key in keys if (metadata.get(key, {}).get("valid_from") is None
                    or metadata[key]["valid_from"] <= self.reference_date) and (
                    metadata.get(key, {}).get("valid_to") is None or self.reference_date < metadata[key]["valid_to"])
                    and metadata.get(key, {}).get("current") is not False]
            elif prior is not None and prior.version_intent.version_ids:
                keys = [key for key in keys if key.version in prior.version_intent.version_ids]
            anchors.extend(keys)
        anchors.extend(key for key in self._blocks if key.version in intent.version_ids)
        families = set(intent.document_families)
        if anchors:
            families.update(metadata.get(key, {}).get("document_family", key.document_id) for key in anchors)
        if not families:
            families = {metadata.get(key, {}).get("document_family", key.document_id) for key in self._blocks}
        selected = set()
        for family in families:
            members = [key for key in self._blocks if metadata.get(key, {}).get("document_family", key.document_id) == family]
            starts = [metadata[key]["valid_from"] for key in anchors if key in members
                      and metadata.get(key, {}).get("valid_from") is not None]
            orders = [tuple(metadata[key]["version_order"]) for key in anchors if key in members
                      and metadata.get(key, {}).get("version_order")]
            axis = "valid_from" if starts or not orders else "version_order"
            anchor_values = starts or orders
            boundary = ((min(anchor_values) if intent.temporal_relation == "before" else max(anchor_values))
                        if anchor_values else None)
            if boundary is None:
                continue
            ordered = [(tuple(metadata[key][axis]) if axis == "version_order" else metadata[key][axis], key)
                       for key in members if metadata.get(key, {}).get(axis)]
            eligible = [(value, key) for value, key in ordered if (
                value < boundary if intent.temporal_relation == "before" else value > boundary)]
            if eligible:
                nearest = max(start for start, _ in eligible) if intent.temporal_relation == "before" else min(start for start, _ in eligible)
                selected.update(key for start, key in eligible if start == nearest)
        if intent.mode == "compare":
            anchored_versions = {(metadata.get(key, {}).get("document_family", key.document_id), key.version)
                                 for key in anchors}
            selected.update(key for key in self._blocks if (
                metadata.get(key, {}).get("document_family", key.document_id), key.version) in anchored_versions)
        return selected

    def _applicable(self, key, state, memory):
        intent = state.version_intent
        metadata = self.corpus_index.source_metadata.get(key, {})
        family = metadata.get("document_family", key.document_id)
        if intent.document_families and family not in intent.document_families:
            return False
        if intent.resolved_version_ids:
            return key.version in intent.resolved_version_ids
        temporal_sources = self._temporal_sources(state, memory)
        if temporal_sources is not None:
            return key in temporal_sources
        if intent.mode == "explicit":
            return key.version in intent.version_ids
        if intent.mode == "compare":
            return not intent.version_ids or key.version in intent.version_ids
        if intent.mode == "as_of":
            if intent.version_ids:
                return key.version in intent.version_ids
            if intent.source_turns:
                anchored = {k for row in memory.committed_history if row.turn_id in intent.source_turns for k in row.evidence_keys}
                versions = {(self.corpus_index.source_metadata.get(k, {}).get("document_family", k.document_id), k.version)
                            for k in anchored}
                return (family, key.version) in versions
            return False
        moment = self.reference_date
        start, end = metadata.get("valid_from"), metadata.get("valid_to")
        if start is not None or end is not None:
            return (start is None or start <= moment) and (end is None or moment < end)
        if intent.mode == "current" and metadata.get("current") is not None:
            return metadata["current"]
        return True

    def retrieve_pool(self, state, memory):
        started = time.perf_counter()
        profile = self.profile
        streams, trace = [], []

        def recall(query, top_k, origin, need_id=""):
            contexts = self.candidate_search(query, top_k=top_k)
            keys = self.corpus_index.retrieved_keys(contexts)
            hits = [(key, float(context.score or 0.0), need_id) for key, context in zip(keys, contexts, strict=True)
                    if self._applicable(key, state, memory)]
            trace.append({"origin": origin, "need_id": need_id, "query": query,
                          "retrieved": len(contexts), "applicable": len(hits)})
            return hits

        streams.append(recall(state.query or state.raw_question, profile.current_top_k, "current"))
        from adapters.history_state import history_text
        active_ids = {condition.condition_id for condition in state.conditions}
        inactive = tuple(ref for prior in memory.prior_task_states for condition in prior.conditions
                         if prior.task_id == state.task_id and condition.condition_id not in active_ids
                         for ref in condition.value_spans)

        def need_text(ref):
            text = (state.raw_question[ref.start:ref.end] if ref.turn_id == state.current_turn_id
                    else history_text(ref, memory))
            intervals = sorted((max(ref.start, old.start)-ref.start, min(ref.end, old.end)-ref.start)
                for old in inactive if ref.turn_id == old.turn_id and ref.speaker == old.speaker
                and ref.start < old.end and old.start < ref.end)
            cursor, pieces = 0, []
            for start, end in intervals:
                pieces.append(text[cursor:start])
                cursor = max(cursor, end)
            pieces.append(text[cursor:])
            return "".join(pieces)

        for need in state.needs:
            pieces = [need_text(ref) for ref in (*need.anchor_spans, *need.subject_refs)]
            pieces.extend(state.raw_question[ref.start:ref.end] if ref.turn_id == state.current_turn_id
                          else history_text(ref, memory)
                          for condition in state.conditions for ref in condition.value_spans)
            streams.append(recall("\n".join(pieces), profile.per_need_top_k, "need", need.need_id))
        archive = [key for key in memory.archive_source_keys if self._applicable(key, state, memory)]
        archive = self.archive_search(state.query or state.raw_question, archive, top_k=profile.archive_top_k)
        streams.append([(key, 0.0, "") for key in archive])
        explicit = [key for key in self._blocks if key.version in state.version_intent.version_ids
                    and self._applicable(key, state, memory)]
        streams.append([(key, 0.0, "") for key in (*state.source_scope, *explicit)])
        selected, scores, needs = [], {}, {}
        # Round robin reserves each need and archive a place before filling the pool.
        for rank in range(max((len(stream) for stream in streams), default=0)):
            for stream in streams:
                if rank >= len(stream):
                    continue
                key, score, need_id = stream[rank]
                if key not in scores and len(selected) < profile.primary_pool_size:
                    selected.append(key)
                scores[key] = max(scores.get(key, score), score)
                if need_id:
                    needs.setdefault(key, set()).add(need_id)
        blocks = self.source_fetch(selected)
        units = tuple(replace(unit, retrieval_score=scores.get(block.key, 0.0),
                              need_ids=tuple(sorted(needs.get(block.key, ()))))
                      for block in blocks for unit in self._units[block.key])
        return EvidencePool(units, blocks, "selected_sources", (time.perf_counter()-started)*1000, tuple(trace))

    def full_scope(self, keys):
        blocks = self.source_fetch(keys)
        return EvidencePool(tuple(unit for block in blocks for unit in self._units[block.key]),
                            blocks, "all_supplied_documents")

    @staticmethod
    def open_need_query(state, need):
        return "\n".join((state.query or state.raw_question, *need.bound_entities,
                           need.missing_relation, need.temporal_scope))

    def retrieve_gap_pool(self, state, memory, open_needs):
        started = time.perf_counter()
        units, blocks, trace = {}, {}, []
        for need in open_needs:
            query = self.open_need_query(state, need)
            contexts = self.candidate_search(query, top_k=self.profile.per_need_top_k)
            scored = {key: float(context.score) for key, context in zip(
                self.corpus_index.retrieved_keys(contexts), contexts, strict=True) if self._applicable(key, state, memory)}
            keys = tuple(scored)
            for block in self.source_fetch(keys):
                blocks[block.key] = block
                for unit in self._units[block.key]:
                    units[unit.unit_id] = replace(unit, need_ids=(need.need_id,),
                        retrieval_score=scored.get(block.key, unit.retrieval_score))
            trace.append({"origin": "open_need", "query": query, "need_id": need.need_id,
                          "parent_claim_ids": need.parent_claim_ids,
                          "sources": [{"passage_id": key.passage_id, "version": key.version, "score": score}
                                      for key, score in scored.items()]})
        return EvidencePool(tuple(units.values()), tuple(blocks.values()), "selected_sources",
                            (time.perf_counter()-started)*1000, tuple(trace))

    def extend_open_needs(self, state, memory, pool, open_needs):
        extension = self.retrieve_gap_pool(state, memory, open_needs)
        units = {unit.unit_id: unit for unit in (*pool.units, *extension.units)}
        blocks = {block.key: block for block in (*pool.source_blocks, *extension.source_blocks)}
        return EvidencePool(tuple(units.values()), tuple(blocks.values()), pool.scope_id,
                            pool.retrieval_ms+extension.retrieval_ms, (*pool.trace, *extension.trace))


def retrieve_pool(state, memory):
    return memory.source_index.retrieve_pool(state, memory)
