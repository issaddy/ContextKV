"""Scope-first retrieval and shared evidence encoding for the cooperative executor."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

from adapters.corpus import CorpusIndex
from adapters.document_compute import DocumentIndex
from adapters.evidence_pool import SourceDirectory
from adapters.hybrid_retrieval import DenseIndex, reciprocal_rank_fusion
from adapters.retrieval import BM25SRetriever, run_retrieval_worker
from adapters.shared_encoding import EvidenceEncoding, SharedEvidenceEncoder, SharedQueryEncoder
from adapters.version_structure import VersionIndex
from controller.context_schema import EvidencePool, SourceSpan
from controller.schema import OnlineContext, QueryContext, history_messages
from controller.cooperative_request import EvidenceSelection


def computation_signature(program):
    from controller.serialization import to_jsonable
    return {name: value for name, value in to_jsonable(program).items()
            if name not in {"need_id", "condition_ids"}}


def need_scope_signature(query, need_id, index):
    need = next(item for item in query.task.needs if item.need_id == need_id)
    conditions = [item for item in query.task.conditions if item.condition_id in need.condition_ids]
    return {"operation": need.operation, "request": query.text(need.anchor_spans).strip().casefold(),
        "subjects": [query.text((ref,)).strip().casefold() for ref in need.subject_refs],
        "conditions": [{"type": item.relation_type, "key": query.text((item.key_span,)) if item.key_span else "",
            "values": [query.text((ref,)) for ref in item.value_spans],
            "operator": query.filter_operators[item.condition_id]} for item in conditions],
        "join": need_id in query.join_needs, "distinct": need_id in query.distinct_needs,
        "sources": [[key.corpus_id, key.document_id, key.passage_id, key.version]
                    for key in index.blocks]}


@dataclass
class SharedEvidenceState:
    query: object
    evidence: object
    pool: EvidencePool
    selected_ids: tuple[str, ...]
    document_index: DocumentIndex
    programs: tuple = ()
    reused_claims: tuple = ()
    semantic_work: tuple = ()
    pending_units: tuple = ()
    provider: object = field(default=None, repr=False)
    body_tokens: int = 0

    async def expand(self):
        if not self.pending_units:
            return None
        return await run_retrieval_worker(self.provider.expand_state, self)


class SharedEvidenceProvider:
    def __init__(self, config, template):
        self.config, self.template = config, template
        settings, retrieval = config["shared"], config["retrieval"]
        self.query_encoder = SharedQueryEncoder(**settings["query"])
        self.evidence_encoder = SharedEvidenceEncoder(**settings["evidence"])
        if retrieval.get("corpus_path"):
            self.sparse = BM25SRetriever.load(retrieval["sparse_index"])
            self.dense = DenseIndex.load(retrieval["dense_index"])
            self.corpus_snapshot_id = self.sparse.corpus_snapshot_id
            self.corpus = CorpusIndex.from_jsonl(retrieval["corpus_path"], corpus_id=retrieval["corpus_id"],
                                                snapshot_id=self.corpus_snapshot_id)
        else:
            self.sparse = self.dense = None
            self.corpus_snapshot_id = "supplied"
            self.corpus = CorpusIndex((), corpus_id=retrieval["corpus_id"], snapshot_id=self.corpus_snapshot_id)
        self.directory = SourceDirectory(self.corpus, template.tokenizer)
        self.versions = VersionIndex.from_blocks(self.corpus.blocks, self.corpus.source_metadata)
        self.fact_history = None
        self.candidate_k = retrieval.get("candidate_k", 50)
        self.evidence_batch = retrieval.get("evidence_batch", 24)
        self._supplied = {}

    @property
    def manifest(self):
        return {"query": self.query_encoder.manifest, "evidence": self.evidence_encoder.manifest,
                "retrieval": "scope_first_bm25_dense_rrf", "selection": "generator_set_utility",
                "corpus_snapshot_id": self.corpus_snapshot_id}

    async def retrieve(self, snapshot, query, request_id):
        facts = self.fact_history(snapshot) if self.fact_history else ()
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id, facts=facts)

    async def retrieve_supplied(self, snapshot, query, request_id, blocks, *, source_metadata=None,
                                coverage_scope="selected_sources"):
        facts = self.fact_history(snapshot) if self.fact_history else ()
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id,
            supplied_blocks=tuple(blocks), source_metadata=source_metadata, coverage_scope=coverage_scope, facts=facts)

    def _catalogue(self, corpus):
        catalogue = tuple({(corpus.source_metadata.get(block.key, {}).get("document_family", block.key.document_id),
                       block.key.version): {"family": corpus.source_metadata.get(block.key, {}).get(
                           "document_family", block.key.document_id), "version": block.key.version,
                           **{key: value for key, value in corpus.source_metadata.get(block.key, {}).items()
                              if key in {"valid_from", "valid_to", "version_order", "current"}}}
                       for block in corpus.blocks}.values())
        counts = {}
        for row in catalogue:
            counts[row["family"]] = counts.get(row["family"], 0)+1
        return tuple(row for row in catalogue if counts[row["family"]] > 1)

    def _retrieve(self, snapshot, question, request_id, *, supplied_blocks=None, source_metadata=None,
                  coverage_scope="selected_sources", facts=()):
        import torch
        started = time.perf_counter()
        document_tokens = 0
        corpus, directory, sparse, dense, versions = self.corpus, self.directory, self.sparse, self.dense, self.versions
        cached = None
        if supplied_blocks is not None:
            key = tuple(block.key for block in supplied_blocks)
            if key not in self._supplied:
                supplied_corpus = CorpusIndex(supplied_blocks, corpus_id=self.corpus.corpus_id,
                    snapshot_id=self.corpus_snapshot_id, source_metadata=source_metadata)
                supplied_directory = SourceDirectory(supplied_corpus, self.template.tokenizer)
                self._supplied[key] = {"corpus": supplied_corpus, "directory": supplied_directory,
                    "versions": VersionIndex.from_blocks(supplied_blocks, source_metadata)}
            cached = self._supplied[key]
            corpus, directory, versions = cached["corpus"], cached["directory"], cached["versions"]
        history = tuple((row.turn_id, message.role, message.content) for row in snapshot.raw_history
                        for message in history_messages((row,)))
        current_turn = max((row.turn_id for row in snapshot.raw_history), default=0) + 1
        history_refs = []
        for row in snapshot.raw_history[-1:]:
            if row.query_context is not None and row.query_context.task_state is not None:
                previous = row.query_context.task_state
                history_refs.extend((*previous.working_history, *previous.version_intent.time_refs))
                for need in previous.needs:
                    history_refs.extend((*need.anchor_spans, *need.subject_refs))
                for condition in previous.conditions:
                    history_refs.extend(ref for ref in (condition.key_span, condition.predicate_span,
                                                        *condition.value_spans) if ref is not None)
        query = self.query_encoder.encode(history, question, request_id, current_turn,
            catalogue=self._catalogue(corpus), source_scope=tuple(block.key for block in supplied_blocks or ()),
            history_turns=tuple(dict.fromkeys(ref.turn_id for ref in history_refs)))
        intent = query.task.version_intent
        families = intent.document_families
        wanted_versions = intent.version_ids
        if intent.temporal_relation in {"before", "after"}:
            related = tuple(versions.adjacent(family, version, intent.temporal_relation)
                for family, version in versions.versions if (not families or family in families)
                and version in wanted_versions)
            wanted_versions = tuple(dict.fromkeys((*wanted_versions, *filter(None, related)))) if intent.mode == "compare" else tuple(filter(None, related))
        version_keys = versions.legal_keys(families=families, versions=wanted_versions, current=intent.mode == "current")
        if intent.mode in {"explicit", "as_of"} and not wanted_versions:
            version_keys = ()
        legal = tuple(key for key in version_keys if self._matches_scope(query, corpus.source_metadata.get(key, {})))
        query.task = replace(query.task, effective_source_scope=legal)
        complete_population = coverage_scope == "all_supplied_documents" or any(
            need.scope == "all_supplied_documents" for need in query.task.needs)
        full = complete_population and (supplied_blocks is not None or bool(families))
        if full:
            document_index = DocumentIndex.from_blocks(corpus.resolve(legal))
            pool = directory.full_scope(legal)
            # Column bindings consume schema addresses; CPU operations read the full rows.
            units = document_index.binding_units()
            units += self._outside_tables(pool.units, document_index)
        else:
            if supplied_blocks is not None:
                if "dense" not in cached:
                    contexts = tuple(OnlineContext(unit.unit_id, unit.text,
                        corpus.resolve((unit.source_spans[0].source_key,))[0].source_uri, unit.heading)
                        for unit in directory.units_by_id.values())
                    cached["sparse"] = BM25SRetriever({context.document_id: {"text": context.text,
                        "source": context.source, "title": context.title} for context in contexts},
                        corpus_snapshot_id=self.corpus_snapshot_id)
                    batches = [self.query_encoder.encode_documents_with_usage([context.title+"\n"+context.text
                        for context in contexts[start:start+16]]) for start in range(0, len(contexts), 16)]
                    import numpy as np
                    document_tokens += sum(tokens for _, tokens in batches)
                    cached["dense"] = DenseIndex(contexts, np.concatenate([vectors for vectors, _ in batches]), {})
                sparse, dense = cached["sparse"], cached["dense"]
                legal_ids = {unit.unit_id for unit in directory.units_by_id.values()
                             if unit.source_spans[0].source_key in set(legal)}
            else:
                legal_ids = {key.passage_id for key in legal}
            vectors = torch.cat((query.query_vector[None, :], query.need_vectors)).cpu().numpy()
            dense_rows = dense.search_vectors(vectors, legal_ids=legal_ids, top_k=self.candidate_k)
            sparse_rows = [sparse.retrieve([text], legal_ids=legal_ids, top_k=self.candidate_k)
                           for text in query.search_texts]
            fused = reciprocal_rank_fusion([[context.document_id for context in rows]
                                            for pair in zip(sparse_rows, dense_rows) for rows in pair])
            contexts = tuple(dense.by_id[identity] for identity, _ in fused[:self.candidate_k])
            if supplied_blocks is None:
                keys = corpus.retrieved_keys(contexts)
                pool = directory.full_scope(keys)
                units = pool.units
            else:
                roots = tuple(directory.units_by_id[context.document_id] for context in contexts)
                units = self._closure(roots, directory.units_by_id)
                keys = tuple(dict.fromkeys(span.source_key for unit in units for span in unit.source_spans))
                pool = EvidencePool(units, corpus.resolve(keys))
        if intent.mode == "compare" and len(wanted_versions) >= 2:
            selected_spans = tuple(span for unit in units for span in unit.source_spans)
            pairs = []
            for family in dict.fromkeys(family for family, _ in versions.versions):
                ordered = [version for name, version in versions.versions
                           if name == family and version in wanted_versions]
                ordered.sort(key=lambda version: (versions.versions[family, version]["order"],
                    versions.versions[family, version]["valid_from"] or "", version))
                for before, after in zip(ordered, ordered[1:]):
                    pairs.extend(versions.paired_units(family, before, after, selected_spans))
            if pairs:
                units = tuple({unit.unit_id: unit for unit in (*pairs, *units)}.values())
                keys = tuple(dict.fromkeys(span.source_key for unit in units for span in unit.source_spans))
                pool = replace(pool, units=units, source_blocks=corpus.resolve(keys))
        if not full:
            document_index = DocumentIndex.from_blocks(pool.source_blocks)
            query.task = replace(query.task, effective_source_scope=tuple(block.key for block in pool.source_blocks))
        units = tuple(replace(unit, body_tokens=len(self.template.tokenizer.encode(unit.text))) for unit in units)
        usage = dict(query.usage)
        available = self._body_budget(snapshot, query)
        by_id = {unit.unit_id: unit for unit in units}
        initial_ids = {unit.unit_id for unit in (units if full else units[:self.evidence_batch])}
        pending_dependencies = list(initial_ids)
        while pending_dependencies:
            for identity in by_id[pending_dependencies.pop()].dependencies:
                if identity not in initial_ids:
                    initial_ids.add(identity)
                    pending_dependencies.append(identity)
        initial = tuple(unit for unit in units if unit.unit_id in initial_ids)
        pending = tuple(unit for unit in units if unit.unit_id not in initial_ids)
        evidence = self.evidence_encoder.encode(query, initial)
        usage["input_tokens"] += evidence.usage["input_tokens"]+document_tokens
        selected = evidence.select(initial, body_tokens=available)
        if complete_population and not full:
            # An open search collection is not the population supplied for computation.
            anchors = tuple(span.source_key for identity in selected for span in by_id[identity].source_spans)
            population_keys = corpus.document_keys(anchors, legal)
            pool = replace(directory.full_scope(population_keys), scope_id="located_documents")
            document_index = DocumentIndex.from_blocks(pool.source_blocks)
            population_keys = {block.key for block in pool.source_blocks}
            positions = [index for index, unit in enumerate(initial)
                         if all(span.source_key in population_keys for span in unit.source_spans)]
            retained = tuple(initial[index] for index in positions)
            binding_units = document_index.binding_units()
            outside = self._outside_tables(pool.units, document_index)
            aliases = {}
            for unit in (*binding_units, *outside):
                original = next((prior for prior in retained if all(any(
                    old.source_key == span.source_key and old.start <= span.start and span.end <= old.end
                    for old in prior.source_spans) for span in unit.source_spans)), None)
                aliases[unit.unit_id] = original.unit_id if original is not None else unit.unit_id
            fresh = tuple(replace(unit,
                dependencies=tuple(dict.fromkeys(aliases[dependency] for dependency in unit.dependencies)))
                for unit in (*binding_units, *outside) if aliases[unit.unit_id] == unit.unit_id)
            fresh = tuple(replace(unit, body_tokens=len(self.template.tokenizer.encode(unit.text))) for unit in fresh)
            evidence = replace(evidence, unit_ids=tuple(unit.unit_id for unit in retained),
                embeddings=evidence.embeddings[positions], relevance=evidence.relevance[positions],
                bindings=tuple(binding for binding in evidence.bindings
                               if binding.source_span.source_key in population_keys),
                token_states=(), token_sources=())
            if fresh:
                extra = self.evidence_encoder.encode(query, fresh)
                usage["input_tokens"] += extra.usage["input_tokens"]
                evidence = EvidenceEncoding((*evidence.unit_ids, *extra.unit_ids),
                    torch.cat((evidence.embeddings, extra.embeddings)),
                    torch.cat((evidence.relevance, extra.relevance)),
                    (*evidence.bindings, *extra.bindings), evidence.heads)
            initial, pending = (*retained, *fresh), ()
            selected = evidence.select(initial, body_tokens=available)
            query.task = replace(query.task,
                effective_source_scope=tuple(block.key for block in pool.source_blocks))
        state = SharedEvidenceState(query, evidence, replace(pool, units=tuple(initial)), selected,
            document_index, pending_units=tuple(pending), provider=self, body_tokens=available)
        self._finish_state(state)
        state.reused_claims = self._reuse_computations(state, facts)
        context = QueryContext(question, "shared", task_state=query.task,
            history_turns=tuple(dict.fromkeys(ref.turn_id for ref in query.task.working_history)))
        usage["compute_ms"] = (time.perf_counter()-started)*1000
        return EvidenceSelection(pool.source_blocks, tuple(block.key for block in pool.source_blocks),
            context, auxiliary_usage=usage, shared_state=state)

    def _body_budget(self, snapshot, query):
        from controller.schema import Message
        from controller.cooperative_request import SYSTEM, working_history
        settings = self.config.get("execution", {})
        system = SYSTEM
        messages = (Message("system", system), Message("user", ""),
                    *history_messages(working_history(query, snapshot.raw_history)), Message("user", query.task.raw_question))
        prompt = self.template.render(messages)
        reserved = len(prompt.input_ids) + settings.get("output_tokens", 1024)
        return min(self.config.get("selection", {}).get("body_tokens", 8192), settings.get("window_tokens", 32768)-reserved)

    @staticmethod
    def _closure(roots, by_id):
        found, pending = {}, list(roots)
        while pending:
            unit = pending.pop(0)
            if unit.unit_id not in found:
                found[unit.unit_id] = unit
                pending.extend(by_id[identity] for identity in unit.dependencies)
        return tuple(found.values())

    def _outside_tables(self, units, index):
        output, replacements = [], {}
        table_ranges = {}
        for table in index.tables.values():
            table_ranges.setdefault(table.source.source_key, set()).add((table.source.start, table.source.end))
            for row in table.rows:
                for cell in row:
                    for source in cell.qualifiers:
                        table_ranges.setdefault(source.source_key, set()).add((source.start, source.end))
        for unit in units:
            source = unit.source_spans[0]
            parts = [(source.start, source.end)]
            for left, right in sorted(table_ranges.get(source.source_key, ())):
                parts = [(start, end) for first, last in parts
                         for start, end in ((first, min(last, left)), (max(first, right), last)) if start < end]
            replacements[unit.unit_id] = []
            for start, end in parts:
                text = unit.text[start-source.start:end-source.start]
                if text.strip():
                    identity = f"{unit.unit_id}:text:{start}:{end}"
                    replacements[unit.unit_id].append(identity)
                    span = SourceSpan(identity, source.source_key, start, end)
                    output.append(replace(unit, unit_id=identity, source_spans=(span,), text=text))
        return tuple(replace(unit, dependencies=tuple(identity for dependency in unit.dependencies
            for identity in replacements.get(dependency, ()))) for unit in output)

    @staticmethod
    def _reuse_computations(state, history):
        from controller.serialization import decode
        legal = set(state.query.task.effective_source_scope)
        reused = []
        for past in history:
            for computation in past["computations"]:
                if not computation["complete"]:
                    continue
                sources = tuple(decode(SourceSpan, source) for source in computation["sources"])
                if any(source.source_key not in legal for source in sources):
                    continue
                for program in state.programs:
                    if computation["input"] != computation_signature(program):
                        continue
                    # Identical source-addressed input fields and predicates identify a reusable calculation.
                    reused.append({**computation, "need_id": program.need_id})
                for need in state.query.task.needs:
                    if computation.get("scope") == need_scope_signature(state.query, need.need_id, state.document_index):
                        reused.append({**computation, "need_id": need.need_id})
        return tuple({item["need_id"]: item for item in reused}.values())

    @staticmethod
    def _matches_scope(query, metadata):
        declared = {name.upper(): str(value).casefold() for name, value in metadata.get("applicability", {}).items()}
        conditions = {condition.condition_id: condition for condition in query.task.conditions}
        for need in query.task.needs:
            matching = True
            for identity in need.condition_ids:
                condition = conditions[identity]
                name = condition.relation_type.upper()
                if name in declared and condition.value_spans:
                    allowed = {query.text((ref,)).casefold() for ref in condition.value_spans}
                    matching &= declared[name] in allowed
            if matching:
                return True
        return not query.task.needs

    def _finish_state(self, state):
        from adapters.document_compute import build_programs
        state.programs = build_programs(state.query, state.evidence, state.document_index)
        programmed = {program.need_id for program in state.programs}
        state.semantic_work = tuple(need for need in state.query.task.needs
            if need.need_id not in programmed and need.operation in {"count", "sum", "mean", "min", "max", "group", "infer", "compare"})

    def expand_state(self, state):
        import torch
        from adapters.shared_encoding import EvidenceEncoding
        existing = set(state.evidence.unit_ids)
        all_units = {unit.unit_id: unit for unit in (*state.pool.units, *state.pending_units)}
        fresh = tuple(unit for unit in self._closure(state.pending_units[:self.evidence_batch], all_units)
                      if unit.unit_id not in existing)
        new = self.evidence_encoder.encode(state.query, fresh)
        old = state.evidence
        evidence = EvidenceEncoding((*old.unit_ids, *new.unit_ids), torch.cat((old.embeddings, new.embeddings)),
            torch.cat((old.relevance, new.relevance)), (*old.bindings, *new.bindings), old.heads,
            usage=new.usage)
        units = (*state.pool.units, *fresh)
        selected = evidence.select(units, body_tokens=state.body_tokens)
        fresh_ids = {unit.unit_id for unit in fresh}
        result = replace(state, evidence=evidence, pool=replace(state.pool, units=units), selected_ids=selected,
                         pending_units=tuple(unit for unit in state.pending_units if unit.unit_id not in fresh_ids))
        self._finish_state(result)
        return result

    def close(self):
        self.query_encoder.close()
        self.evidence_encoder.close()
