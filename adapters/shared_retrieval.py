"""Scope-first retrieval and shared evidence encoding for the cooperative executor."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

from adapters.corpus import CorpusIndex
from adapters.document_compute import DocumentIndex
from adapters.evidence_pool import SourceDirectory
from adapters.hybrid_retrieval import DenseIndex, reciprocal_rank_fusion
from adapters.retrieval import BM25SRetriever, run_retrieval_worker
from adapters.shared_encoding import SharedEvidenceEncoder, SharedQueryEncoder
from adapters.version_structure import VersionIndex, catalogue_versions
from controller.schema import EvidencePool, SourceSpan
from controller.schema import OnlineContext, QueryContext, history_messages
from controller.cooperative_request import EvidenceSelection


def computation_signature(program):
    from controller.serialization import to_jsonable
    return {name: value for name, value in to_jsonable(program).items()
            if name not in {"need_id", "condition_ids"}}


def need_scope_signature(query, need_id, index):
    need = next(item for item in query.task.needs if item.need_id == need_id)
    conditions = [item for item in query.task.conditions if item.condition_id in need.condition_ids]
    return {"operation": need.operation, "coverage": query.task.coverage_scope,
        "request": query.task.raw_question.strip().casefold(), "need": query.text(need.anchor_spans),
        "subjects": [query.text((ref,)).strip().casefold() for ref in need.subject_refs],
        "conditions": [{"type": item.relation_type, "key": query.text((item.key_span,)) if item.key_span else "",
            "values": [query.text((ref,)) for ref in item.value_spans],
            "operator": query.filter_operators[item.condition_id]} for item in conditions],
        "join": need_id in query.join_needs, "distinct": need_id in query.distinct_needs,
        "sources": [[block.key.corpus_id, block.key.document_id, block.key.passage_id, block.key.version,
                     *(block.span or (0, len(block.text)))] for block in index.blocks]}


@dataclass
class SharedEvidenceState:
    query: object
    evidence: object
    pool: EvidencePool
    selected_ids: tuple[str, ...]
    document_index: DocumentIndex
    programs: tuple = ()
    reused_claims: tuple = ()
    body_tokens: int = 0
    population_units: tuple = ()
    source_headers: tuple = ()
    version_catalogue: dict = field(default_factory=dict)
    costs: object = None
    compiled_answer: object = None
    computations: tuple = ()
    table_parts: tuple = ()
    table_read: bool = False
    projections: tuple = ()
    read_units: set = field(default_factory=set)
    processed_tables: set = field(default_factory=set)
    resolved_operations: dict = field(default_factory=dict)
    prepared_programs: tuple = ()
    totals: dict | None = None
    budget_exhausted: bool = False
    cache_scope: str = "shared"
    cache_namespace: str | None = None
    session_id: str | None = None

    @property
    def completed_needs(self):
        return {record["need_id"] for record in self.computations
                if record["complete"] and record.get("records_extracted") is False}

    @property
    def needs_complete(self):
        completed = self.completed_needs
        return bool(self.query.task.needs) and all(
            need.need_id in completed for need in self.query.task.needs)

    @property
    def open_needs(self):
        completed = self.completed_needs
        return tuple(need for need in self.query.task.needs if need.need_id not in completed)


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
        self.catalogue = self._catalogue(self.corpus)
        self.candidate_k = retrieval.get("candidate_k", 50)
        self.evidence_batch = retrieval.get("evidence_batch", 24)
        self._supplied = {}

    @property
    def manifest(self):
        return {"query": self.query_encoder.manifest, "evidence": self.evidence_encoder.manifest,
                "retrieval": "scope_first_bm25_dense_rrf", "selection": "semantic_utility_with_source_dependencies",
                "corpus_snapshot_id": self.corpus_snapshot_id}

    async def retrieve(self, snapshot, query, request_id, *, facts=()):
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id, facts=facts)

    async def retrieve_candidates(self, snapshot, query, request_id, *, initial_only=False):
        """Return serving candidates, optionally restricted to the first encoded batch."""
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id,
                                          candidate_view="initial" if initial_only else "all")

    async def retrieve_supplied(self, snapshot, query, request_id, blocks, *, source_metadata=None,
                                coverage_scope="selected_sources", facts=()):
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
                  coverage_scope="selected_sources", facts=(), candidate_view=None):
        import torch
        started = time.perf_counter()
        document_tokens = 0
        corpus, directory, sparse, dense, versions = self.corpus, self.directory, self.sparse, self.dense, self.versions
        catalogue = self.catalogue
        cached = None
        if supplied_blocks is not None:
            supplied_blocks = tuple(replace(block, title=f"D{index+1} | " + (block.title or next(
                (line.strip().lstrip("#").strip() for line in block.text.splitlines() if line.strip()), "")))
                for index, block in enumerate(supplied_blocks))
            key = snapshot.corpus_snapshot_id, tuple(block.key for block in supplied_blocks)
            if key not in self._supplied:
                supplied_corpus = CorpusIndex(supplied_blocks, corpus_id=self.corpus.corpus_id,
                    snapshot_id=self.corpus_snapshot_id, source_metadata=source_metadata)
                supplied_directory = SourceDirectory(supplied_corpus, self.template.tokenizer, unit_tokens=256)
                self._supplied[key] = {"corpus": supplied_corpus, "directory": supplied_directory,
                    "versions": VersionIndex.from_blocks(supplied_blocks, source_metadata),
                    "catalogue": self._catalogue(supplied_corpus)}
            cached = self._supplied[key]
            corpus, directory, versions = cached["corpus"], cached["directory"], cached["versions"]
            catalogue = cached["catalogue"]
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
            catalogue=catalogue, source_scope=tuple(block.key for block in supplied_blocks or ()),
            history_turns=tuple(dict.fromkeys(ref.turn_id for ref in history_refs)))
        intent = query.task.version_intent
        families = intent.document_families
        wanted_versions = intent.resolved_version_ids
        # The catalogue head resolves as_of text to version IDs during the shared
        # query pass; natural-language date spans are not sortable timestamps.
        version_keys = versions.legal_keys(families=families, versions=wanted_versions,
            current=intent.mode == "current")
        if intent.mode in {"explicit", "as_of"} and not wanted_versions:
            version_keys = ()
        legal = tuple(key for key in version_keys if self._matches_scope(query, corpus.source_metadata.get(key, {})))
        query.task = replace(query.task, effective_source_scope=legal)
        legal_set = set(legal)
        previous_spans = tuple(span for turn in snapshot.raw_history[-1:] for span in turn.source_spans
                               if span.source_key in legal_set)
        complete_population = "all_supplied_documents" in (
            coverage_scope, query.task.coverage_scope, *(need.scope for need in query.task.needs))
        if complete_population:
            query.task = replace(query.task, coverage_scope="all_supplied_documents")
        full = complete_population and (supplied_blocks is not None or bool(families))
        usage = dict(query.usage)
        available = self._body_budget(snapshot, query)
        def finish(state):
            state.version_catalogue = catalogue_versions(catalogue, families)
            if supplied_blocks is not None:
                from controller.cooperative_request import document_headers
                state.source_headers = document_headers(corpus.resolve(legal), self.template.tokenizer)
            context = QueryContext(question, "shared", task_state=query.task,
                history_turns=tuple(dict.fromkeys(ref.turn_id for ref in query.task.working_history)))
            usage["compute_ms"] = (time.perf_counter()-started)*1000
            return EvidenceSelection(state.pool.source_blocks, context, auxiliary_usage=usage, shared_state=state)
        fixed_claims = ()
        population_units = ()
        historical_ids = ()
        if full:
            blocks = corpus.resolve(legal)
            if facts and not candidate_view:
                state = SharedEvidenceState(query, None, EvidencePool((), blocks), (), DocumentIndex((), blocks),
                    body_tokens=available)
                fixed_claims = self._reuse_computations(state, facts)
                state.reused_claims = state.computations = fixed_claims
                if state.needs_complete:
                    return finish(state)
            document_index = DocumentIndex.from_blocks(blocks)
            pool = directory.full_scope(legal)
            population_units = pool.units
            query.source_titles = tuple(block.title for block in pool.source_blocks)
            units = pool.units
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
                             if unit.source_spans[0].source_key in legal_set}
            else:
                legal_ids = {key.passage_id for key in legal}
            vectors = torch.cat((query.query_vector[None, :], query.need_vectors)).cpu().numpy()
            dense_rows = dense.search_vectors(vectors, legal_ids=legal_ids, top_k=self.candidate_k)
            sparse_rows = [sparse.retrieve([text], legal_ids=legal_ids, top_k=self.candidate_k)
                           for text in query.search_texts]
            fused = reciprocal_rank_fusion([[context.document_id for context in rows]
                for pair in zip(sparse_rows, dense_rows) for rows in pair],
                weights=[weight for index in range(len(sparse_rows))
                         for weight in ([1.0, 1.0] if index == 0 else [1.0/(len(sparse_rows)-1)]*2)])
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
            if previous_spans:
                prior = directory.full_scope(tuple(dict.fromkeys(span.source_key for span in previous_spans)))
                observed = tuple(unit for unit in prior.units if any(
                    origin.source_key == span.source_key and origin.start < span.end and span.start < origin.end
                    for origin in unit.source_spans for span in previous_spans))
                historical_ids = tuple(unit.unit_id for unit in observed)
                roots = (*units, *observed)
                units = self._closure(roots, {unit.unit_id: unit for unit in (*units, *prior.units)})
                pool = EvidencePool(units, tuple({b.key: b for b in (*pool.source_blocks, *prior.source_blocks)}.values()))
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
        by_id = {unit.unit_id: unit for unit in units}
        if full and not candidate_view:
            state = SharedEvidenceState(query, None, replace(pool, units=units), (), document_index,
                body_tokens=available,
                population_units=tuple(by_id[unit.unit_id] for unit in population_units),
                reused_claims=fixed_claims, computations=fixed_claims)
            return finish(state)
        initial_ids = {unit.unit_id for unit in (units if full else units[:self.evidence_batch])}
        initial_ids.update(historical_ids)
        pending_dependencies = list(initial_ids)
        while pending_dependencies:
            for identity in by_id[pending_dependencies.pop()].dependencies:
                if identity not in initial_ids:
                    initial_ids.add(identity)
                    pending_dependencies.append(identity)
        initial = tuple(unit for unit in units if unit.unit_id in initial_ids)
        if candidate_view:
            pending = tuple(unit for unit in units if unit.unit_id not in initial_ids)
            return query, initial if candidate_view == "initial" else (*initial, *pending)
        evidence = self.evidence_encoder.encode(query, initial)
        usage["input_tokens"] += evidence.usage["input_tokens"]+document_tokens
        selected = ()
        if complete_population and not full:
            # An open search collection is not the population supplied for computation.
            selected = evidence.select(initial, body_tokens=available,
                need_ids=tuple(need.need_id for need in query.task.needs))
            anchors = tuple(span.source_key for identity in selected for span in by_id[identity].source_spans)
            population_keys = corpus.document_keys(anchors, legal)
            pool = replace(directory.full_scope(population_keys), scope_id="located_documents")
            document_index = DocumentIndex.from_blocks(pool.source_blocks)
            population_units = tuple(replace(unit, body_tokens=len(self.template.tokenizer.encode(unit.text)))
                                     for unit in pool.units)
            query.task = replace(query.task,
                effective_source_scope=tuple(block.key for block in pool.source_blocks))
            query.source_titles = tuple(block.title for block in pool.source_blocks)
            state = SharedEvidenceState(query, None, replace(pool, units=population_units), (), document_index,
                body_tokens=available, population_units=population_units)
            state.reused_claims = self._reuse_computations(state, facts)
            state.computations = state.reused_claims
            return finish(state)
        state = SharedEvidenceState(query, evidence, replace(pool, units=tuple(initial)), selected,
            document_index, body_tokens=available, population_units=population_units)
        self._finish_state(state)
        state.reused_claims = fixed_claims if full else self._reuse_computations(state, facts)
        state.computations = state.reused_claims
        return finish(state)

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
        found, seen = [], set()
        def add(unit):
            if unit.unit_id in seen:
                return
            seen.add(unit.unit_id)
            for identity in unit.dependencies:
                add(by_id[identity])
            found.append(unit)
        for unit in roots:
            add(unit)
        return tuple(found)

    @staticmethod
    def _reuse_computations(state, history):
        from controller.serialization import decode
        legal = set(state.query.task.effective_source_scope)
        programs = () if state.query.task.coverage_scope == "all_supplied_documents" else state.programs
        reused = []
        for past in history:
            for computation in past["computations"]:
                if not computation["complete"] or computation.get("records_extracted") is not False:
                    continue
                sources = tuple(decode(SourceSpan, source) for source in computation["sources"])
                if any(source.source_key not in legal for source in sources):
                    continue
                for program in programs:
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

    async def bind_tables(self, state):
        """Bind only headers and sample rows before a required table computation."""
        def bind():
            from adapters.document_compute import build_programs
            units = state.document_index.binding_units()
            if not units:
                return {"input_tokens": 0, "compute_ms": 0.0, "calls": 0}
            evidence = self.evidence_encoder.encode(state.query, units)
            state.programs = build_programs(state.query, evidence, state.document_index)
            return evidence.usage
        started = time.perf_counter()
        usage = await run_retrieval_worker(bind)
        return {**usage, "compute_ms": (time.perf_counter()-started)*1000}

    async def encode_evidence(self, state, units=None):
        """Encode original spans only when the chosen execution needs selection."""
        def encode():
            chosen = state.pool.units if units is None else self._closure(
                units, {unit.unit_id: unit for unit in state.pool.units})
            state.evidence = self.evidence_encoder.encode(state.query, chosen)
            state.pool = replace(state.pool, units=tuple(chosen))
            state.selected_ids, state.costs = (), None
            return state.evidence.usage
        started = time.perf_counter()
        usage = await run_retrieval_worker(encode)
        return {**usage, "compute_ms": (time.perf_counter()-started)*1000}

    @staticmethod
    def _finish_state(state):
        from adapters.document_compute import build_programs
        state.programs = build_programs(state.query, state.evidence, state.document_index)

    def close(self):
        self.query_encoder.close()
        self.evidence_encoder.close()
