"""Version-aware source directory and quota-balanced evidence candidate pools."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from functools import cached_property
import re
import time

import numpy as np

from adapters.table_relations import extract_table_relations
from controller.context_budget import condition_references, text_ref_sources, text_ref_text
from controller.context_schema import EvidencePool, EvidenceUnit, SourceSpan
from controller.source_windows import SOURCE_WINDOW_TOKENS, source_window_spans


SOURCE_CATEGORIES = frozenset({"PLATFORM", "LOCATION", "REGION", "PRODUCT", "POPULATION"})


def _category_value(text):
    return str(text).strip().rstrip(".,;:!?。！？").casefold()


@dataclass(frozen=True)
class PoolProfile:
    current_top_k: int = 50
    per_need_top_k: int = 20
    archive_top_k: int = 12
    primary_pool_size: int = 96
    unit_tokens: int = SOURCE_WINDOW_TOKENS
    window_protocol: str = "canonical"

    def __post_init__(self):
        if self.window_protocol not in {"canonical", "source_rows_pr3"}:
            raise ValueError("unknown source window protocol")


def unit_identity(key, start, end):
    return f"{key.corpus_id}:{key.document_id}:{key.passage_id}:{key.version}:{key.content_sha256}:{start}:{end}"


def merge_retrieved_units(*groups):
    """Keep every need that recalled a source unit and its strongest retrieval score."""
    merged = {}
    for group in groups:
        for unit in group:
            previous = merged.get(unit.unit_id)
            merged[unit.unit_id] = unit if previous is None else replace(unit,
                need_ids=tuple(dict.fromkeys((*previous.need_ids, *unit.need_ids))),
                retrieval_score=max(previous.retrieval_score, unit.retrieval_score))
    return tuple(merged.values())


class SourceDirectory:
    """Original source metadata remains distinct for each version and identity."""
    def __init__(self, corpus_index, search, tokenizer, *, task_interpreter=None,
                 reference_date=None, profile=None, archive_encoder=None):
        self.corpus_index, self.search, self.tokenizer = corpus_index, search, tokenizer
        self.task_interpreter = task_interpreter
        self.archive_encoder = archive_encoder
        self._unit_vectors = {}
        self.reference_date = reference_date or date.today().isoformat()
        self.profile = profile or PoolProfile()
        self._units = {}
        self._blocks = {b.key: b for b in corpus_index.blocks}
        self._family_members, self._family_by_key, self._version_members = {}, {}, {}
        for key in self._blocks:
            self._index_family(key)
        self._unit_metadata = {key: dict(corpus_index.source_metadata.get(key, {})) for key in self._blocks}

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

    def _index_family(self, key):
        family = self.corpus_index.source_metadata.get(key, {}).get("document_family", key.document_id)
        if key in self._family_by_key and self._family_by_key[key] != family:
            previous = self._family_members[self._family_by_key[key]]
            del previous[key]
            if not previous:
                del self._family_members[self._family_by_key[key]]
        self._family_by_key[key] = family
        self._family_members.setdefault(family, {})[key] = None
        self._version_members.setdefault(key.version, {})[key] = None

    def _split(self, block):
        metadata = self.corpus_index.source_metadata.get(block.key, {})
        base = block.span[0] if block.span else 0
        tables = extract_table_relations(block.text, mode="explicit")
        # Shared table context must not be grouped with data: otherwise a data
        # window becomes another data window's header dependency and is excluded
        # from extraction-output reservation as though it contained only context.
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
        if self.profile.window_protocol == "canonical":
            ordered = sorted(boundaries)
            windows = tuple((start+left, start+right)
                for start, end in zip(ordered, ordered[1:])
                for left, right in source_window_spans(block.text[start:end], self.tokenizer,
                                                       tokens=self.profile.unit_tokens))
        else:
            # Frozen PR3 source-row protocol, including its long-line overlap.
            windows = []
            for match in re.finditer(r"[^\n]+(?:\n|$)", block.text):
                start, end = match.span()
                offsets = self.tokenizer(block.text[start:end], add_special_tokens=False,
                    return_offsets_mapping=True)["offset_mapping"]
                sentence_ends = {match.end() for match in re.finditer(
                    r"[.!?。！？;；](?:\s+|$)", block.text[start:end])}
                first = 0
                while first < len(offsets):
                    last = min(first+self.profile.unit_tokens, len(offsets))
                    cuts = [index+1 for index in range(first, last)
                        if offsets[index][1] in sentence_ends
                        and index-first >= self.profile.unit_tokens//2]
                    if last < len(offsets) and cuts:
                        last = cuts[-1]
                    section = offsets[first:last]
                    windows.append((start+section[0][0], start+section[-1][1]))
                    if last == len(offsets):
                        break
                    overlap = (0 if offsets[last-1][1] in sentence_ends
                               else min(32, self.profile.unit_tokens//8))
                    first = max(first+1, last-overlap)
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
                self._index_family(block.key)
                self._units.pop(block.key, None)
                self._unit_metadata[block.key] = dict(metadata)
                self.__dict__.pop("units_by_id", None)

    def archive_lookup(self, keys):
        return tuple(self._blocks[key] for key in keys)

    def archive_search(self, query, keys, *, top_k=12):
        keys = tuple(keys)
        if not keys or self.archive_encoder is None:
            return keys[:top_k]
        vectors = {}
        for key in keys:
            vector = self.encode_source_units(self._source_units(key), self.archive_encoder).mean(axis=0)
            vectors[key] = vector / np.linalg.norm(vector)
        query_vector = np.asarray(self.archive_encoder([query])[0], dtype=np.float32)
        query_vector /= np.linalg.norm(query_vector)
        return tuple(sorted(keys, key=lambda key: -float(vectors[key] @ query_vector)))[:top_k]

    def encode_source_units(self, units, encode):
        """Reuse the fixed document encoder's vectors by original source identity."""
        units = tuple(units)
        missing = tuple({unit.unit_id: unit for unit in units
                         if unit.unit_id not in self._unit_vectors}.values())
        if missing:
            values = encode([unit.text for unit in missing])
            self._unit_vectors.update((unit.unit_id, np.asarray(value, dtype=np.float32))
                for unit, value in zip(missing, values, strict=True))
        return np.asarray([self._unit_vectors[unit.unit_id] for unit in units], dtype=np.float32)

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
        anchors.extend(key for version in intent.version_ids for key in self._version_members.get(version, ()))
        if not anchors:
            return set()
        anchors_by_family = {}
        for key in anchors:
            family = metadata.get(key, {}).get("document_family", key.document_id)
            anchors_by_family.setdefault(family, []).append(key)
        families = set(intent.document_families)
        if anchors:
            families.update(metadata.get(key, {}).get("document_family", key.document_id) for key in anchors)
        if not families:
            families = set(self._family_members)
        selected = set()
        for family in families:
            members = self._family_members.get(family, {})
            family_anchors = anchors_by_family.get(family, ())
            starts = [metadata[key]["valid_from"] for key in family_anchors if key in members
                      and metadata.get(key, {}).get("valid_from") is not None]
            orders = [tuple(metadata[key]["version_order"]) for key in family_anchors if key in members
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

    def _version_applicable(self, key, state, memory, temporal_sources):
        intent = state.version_intent
        metadata = self.corpus_index.source_metadata.get(key, {})
        family = metadata.get("document_family", key.document_id)
        if intent.document_families and family not in intent.document_families:
            return False
        if intent.resolved_version_ids:
            return key.version in intent.resolved_version_ids
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

    def _source_admission(self, state, memory):
        temporal = None if state.version_intent.resolved_version_ids else self._temporal_sources(state, memory)
        texts = text_ref_sources(memory.committed_history,
            current_turn_id=state.current_turn_id, current_user_text=state.raw_question)
        literals = {condition.condition_id: (category,
            {_category_value(text_ref_text(ref, texts)) for ref in condition.value_spans})
            for condition in state.conditions if (category := condition.relation_type.upper()) in SOURCE_CATEGORIES
            and condition.value_spans}
        declared_values = {}
        if literals:
            for metadata in self.corpus_index.source_metadata.values():
                for name, value in metadata.get("applicability", {}).items():
                    declared_values.setdefault(name.upper(), set()).add(_category_value(value))
        requirements = {}
        for need in state.needs:
            allowed = requirements[need.need_id] = {}
            for identity in need.condition_ids:
                if identity in literals:
                    category, values = literals[identity]
                    grounded = values.intersection(declared_values.get(category, ()))
                    if grounded:
                        allowed.setdefault(category, set()).update(grounded)

        def admit(key):
            if not self._version_applicable(key, state, memory, temporal):
                return None
            declared = self.corpus_index.source_metadata.get(key, {}).get("applicability", {})
            if not declared or not requirements:
                return tuple(requirements)
            scope = {name.upper(): _category_value(value) for name, value in declared.items()}
            matching = tuple(need_id for need_id, allowed in requirements.items()
                if all(name not in scope or scope[name] in values for name, values in allowed.items()))
            return matching or None

        return admit

    def retrieve_pool(self, state, memory):
        started = time.perf_counter()
        profile = self.profile
        streams, trace = [], []
        admit = self._source_admission(state, memory)

        def recall(query, top_k, origin, need_id=""):
            contexts = self.candidate_search(query, top_k=top_k)
            keys = self.corpus_index.retrieved_keys(contexts)
            hits = [(key, float(context.score or 0.0), need_id) for key, context in zip(keys, contexts, strict=True)
                    if (matching := admit(key)) is not None and (not need_id or need_id in matching)]
            trace.append({"origin": origin, "need_id": need_id, "query": query,
                          "retrieved": len(contexts), "applicable": len(hits)})
            return hits

        streams.append(recall(state.query or state.raw_question, profile.current_top_k, "current"))
        from adapters.history_state import active_ref_parts, history_text, retired_condition_refs
        inactive = retired_condition_refs(memory.prior_task_states, state.task_id, state.conditions)

        def need_text(ref):
            return (state.raw_question[ref.start:ref.end] if ref.turn_id == state.current_turn_id
                    else history_text(ref, memory))

        for need in state.needs:
            refs = dict.fromkeys((
                *(piece for ref in (*need.anchor_spans, *need.subject_refs)
                  for piece in active_ref_parts(ref, inactive)),
                *(ref for condition in state.conditions if condition.condition_id in need.condition_ids
                  for ref in condition_references(condition)),
                *state.version_intent.time_refs))
            streams.append(recall("\n".join(map(need_text, refs)), profile.per_need_top_k, "need", need.need_id))
        archive = [key for key in memory.archive_source_keys if admit(key) is not None]
        archive = self.archive_search(state.query or state.raw_question, archive, top_k=profile.archive_top_k)
        streams.append([(key, 0.0, "") for key in archive])
        explicit = [key for key in self._blocks if key.version in state.version_intent.version_ids]
        streams.append([(key, 0.0, "") for key in (*state.source_scope, *explicit) if admit(key) is not None])
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
        restricted = {key: matching for key in selected if len(matching := admit(key)) < len(state.needs)}
        units = tuple(replace(unit, retrieval_score=scores.get(block.key, 0.0),
                              need_ids=tuple(sorted(needs.get(block.key, restricted.get(block.key, ())))))
                      for block in blocks for unit in self._source_units(block.key))
        return EvidencePool(units, blocks, "selected_sources", (time.perf_counter()-started)*1000, tuple(trace))

    def full_scope(self, keys):
        blocks = self.source_fetch(keys)
        return EvidencePool(tuple(unit for block in blocks for unit in self._source_units(block.key)),
                            blocks, "all_supplied_documents")

    def applicable_scope(self, keys, state, memory):
        admit = self._source_admission(state, memory)
        roots = {key: matching for key in keys if (matching := admit(key)) is not None}
        pool = self.full_scope(roots)
        restricted = {key: matching for key, matching in roots.items() if len(matching) < len(state.needs)}
        return replace(pool, units=tuple(replace(unit, need_ids=restricted[unit.source_spans[0].source_key])
            if unit.source_spans[0].source_key in restricted else unit for unit in pool.units))

    @staticmethod
    def open_need_query(state, need):
        return "\n".join((state.query or state.raw_question, *need.bound_entities,
                           need.missing_relation, need.temporal_scope))

    def retrieve_gap_pool(self, state, memory, open_needs):
        started = time.perf_counter()
        units, blocks, trace = [], {}, []
        admit = self._source_admission(state, memory)
        for need in open_needs:
            query = self.open_need_query(state, need)
            contexts = self.candidate_search(query, top_k=self.profile.per_need_top_k)
            scored = {key: float(context.score) for key, context in zip(
                self.corpus_index.retrieved_keys(contexts), contexts, strict=True)
                if (matching := admit(key)) is not None and need.need_id in matching}
            keys = tuple(scored)
            for block in self.source_fetch(keys):
                blocks[block.key] = block
                for unit in self._source_units(block.key):
                    units.append(replace(unit, need_ids=(need.need_id,),
                        retrieval_score=scored.get(block.key, unit.retrieval_score)))
            trace.append({"origin": "open_need", "query": query, "need_id": need.need_id,
                          "parent_claim_ids": need.parent_claim_ids,
                          "sources": [{"passage_id": key.passage_id, "version": key.version, "score": score}
                                      for key, score in scored.items()]})
        return EvidencePool(merge_retrieved_units(units), tuple(blocks.values()), "selected_sources",
                            (time.perf_counter()-started)*1000, tuple(trace))

    def extend_open_needs(self, state, memory, pool, open_needs):
        extension = self.retrieve_gap_pool(state, memory, open_needs)
        units = merge_retrieved_units(pool.units, extension.units)
        blocks = {block.key: block for block in (*pool.source_blocks, *extension.source_blocks)}
        return EvidencePool(units, tuple(blocks.values()), pool.scope_id,
                            pool.retrieval_ms+extension.retrieval_ms, (*pool.trace, *extension.trace))


def retrieve_pool(state, memory):
    return memory.source_index.retrieve_pool(state, memory)
