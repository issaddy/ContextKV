"""Scope-first retrieval and shared evidence encoding for the cooperative executor."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from math import log1p
import time

from adapters.corpus import CorpusIndex
from adapters.document_compute import DocumentIndex, numeric_field_units
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
    pending_units: tuple = ()
    body_tokens: int = 0
    evidence_batch: int = 24
    population_units: tuple = ()
    source_headers: tuple = ()
    document_titles: dict = field(default_factory=dict)
    title_note_budget: str = ""
    version_catalogue: dict = field(default_factory=dict)
    missing_version_identifiers: tuple[str, ...] = ()
    symbol_version_presence: dict = field(default_factory=dict)
    costs: object = None
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
    retained_sources: dict = field(default_factory=dict)

    @property
    def completed_needs(self):
        return {record["need_id"] for record in self.computations if record["complete"]}

    @property
    def open_needs(self):
        selected = [unit for unit in self.pool.units if unit.unit_id in self.selected_ids]
        supported = {binding.need_id for binding in self.evidence.bindings if binding.role == "source" and any(
            span.source_key == binding.source_span.source_key and span.start <= binding.source_span.start
            and binding.source_span.end <= span.end for unit in selected for span in unit.source_spans)}
        return tuple(need for need in self.query.task.needs if need.need_id not in self.completed_needs
                     and (need.operation in {"count", "sum", "mean", "min", "max", "group"}
                          or need.need_id not in supported))


def ranked_population_windows(units, question, budget, tokenizer, witnesses):
    """Keep witness windows and dependencies, then fill by BM25 rank within each document."""
    groups = {}
    for index, unit in enumerate(units):
        key = unit.source_spans[0].source_key
        groups.setdefault((key.corpus_id, key.document_id, key.version), []).append(index)
    terms = set(tokenizer.encode(question, add_special_tokens=False))
    tokens = tokenizer([u.text for u in units], add_special_tokens=False, return_attention_mask=False)["input_ids"]
    frequencies = [Counter(token for token in row if token in terms) for row in tokens]
    df = Counter(token for row in frequencies for token in row)
    idf = {token: log1p((len(units)-count+0.5)/(count+0.5)) for token, count in df.items()}
    mean_length = sum(map(len, tokens))/len(units)
    k1, b = 1.5, 0.75
    scores = [sum(idf[token]*count*(k1+1)/(count+k1*(1-b+b*len(row)/mean_length))
                  for token, count in counts.items()) for row, counts in zip(tokens, frequencies)]

    selected = {units[group[0]].unit_id for group in groups.values()}
    selected.update(dep for unit in units for dep in unit.dependencies)
    selected.update(unit.unit_id for unit in units for span in unit.source_spans
                    if any(span.source_key == block.key and span.start < block.span[1] and block.span[0] < span.end
                           for block in witnesses))
    remaining = budget - sum(unit.body_tokens for unit in units if unit.unit_id in selected)
    ranked = [sorted(group, key=scores.__getitem__, reverse=True) for group in groups.values()]
    for rank in range(max(map(len, ranked))):
        for group in ranked:
            if rank < len(group):
                unit = units[group[rank]]
                if unit.unit_id not in selected and unit.body_tokens <= remaining:
                    selected.add(unit.unit_id)
                    remaining -= unit.body_tokens
    return selected


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
                "retrieval": "scope_first_bm25_dense_rrf", "selection": "learned_utility_and_source_coverage",
                "corpus_snapshot_id": self.corpus_snapshot_id}

    async def retrieve(self, snapshot, query, request_id, *, facts=()):
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id, facts=facts)

    async def retrieve_candidates(self, snapshot, query, request_id):
        """Return the serving encoder's first batch and its pending candidates."""
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id, candidates_only=True)

    async def retrieve_supplied(self, snapshot, query, request_id, blocks, *, source_metadata=None,
                                coverage_scope="selected_sources", facts=(), available_kv_tokens=None):
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id,
            supplied_blocks=tuple(blocks), source_metadata=source_metadata, coverage_scope=coverage_scope, facts=facts,
            available_kv_tokens=available_kv_tokens)

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
                  coverage_scope="selected_sources", facts=(), candidates_only=False, available_kv_tokens=None):
        import torch
        started = time.perf_counter()
        document_tokens = 0
        corpus, directory, sparse, dense, versions = self.corpus, self.directory, self.sparse, self.dense, self.versions
        catalogue = self.catalogue
        cached = None
        parsed_documents = None
        if supplied_blocks is not None:
            titles = {block.key: block.title or next(
                (line.strip().lstrip("#").strip() for line in block.text.splitlines() if line.strip()), "")
                for block in supplied_blocks}
            supplied_blocks = tuple(replace(block, title=f"D{index+1} | " + titles[block.key])
                for index, block in enumerate(supplied_blocks))
            key = snapshot.corpus_snapshot_id, tuple(block.key for block in supplied_blocks)
            if key not in self._supplied:
                supplied_corpus = CorpusIndex(supplied_blocks, corpus_id=self.corpus.corpus_id,
                    snapshot_id=self.corpus_snapshot_id, source_metadata=source_metadata)
                supplied_directory = SourceDirectory(supplied_corpus, self.template.tokenizer, unit_tokens=256)
                parsed_documents = DocumentIndex.from_blocks(supplied_blocks)
                self._supplied[key] = {"corpus": supplied_corpus, "directory": supplied_directory,
                    "versions": VersionIndex.from_blocks(supplied_blocks, source_metadata, parsed_documents),
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
        wanted_versions = intent.version_ids
        if intent.temporal_relation in {"before", "after"}:
            related = tuple(versions.adjacent(family, version, intent.temporal_relation)
                for family, version in versions.versions if (not families or family in families)
                and version in wanted_versions)
            wanted_versions = tuple(dict.fromkeys((*wanted_versions, *filter(None, related)))) if intent.mode == "compare" else tuple(filter(None, related))
        elif intent.mode == "compare" and len(wanted_versions) == 1:
            predecessors = tuple(
                versions.adjacent(family, version, "before")
                for family, version in versions.versions
                if (not families or family in families) and version in wanted_versions
            )
            wanted_versions = tuple(dict.fromkeys((*filter(None, predecessors), *wanted_versions)))
        # The catalogue head resolves as_of text to version IDs during the shared
        # query pass; natural-language date spans are not sortable timestamps.
        has_catalogue_dates = any(row.get("valid_from") or row.get("valid_to") for row in catalogue)
        if intent.mode == "as_of" and not wanted_versions and intent.time_refs and has_catalogue_dates:
            version_keys = ()
        else:
            restrict_current = intent.mode == "current" and not wanted_versions
            version_keys = versions.legal_keys(families=families, versions=wanted_versions,
                current=restrict_current)
            if restrict_current and version_keys:
                # If a question names specific technical symbols absent from current
                # editions (e.g. a historical issue ID or removed symbol), include
                # matching earlier editions within the legal family scope.
                candidate_families = families or tuple(dict.fromkeys(f for f, _ in versions.versions))
                missing_in_current = versions.missing_identifiers(question, version_keys, candidate_families)
                if missing_in_current:
                    expanded = versions.legal_keys(families=families, versions=(), current=False)
                    version_keys = tuple(dict.fromkeys((*version_keys, *expanded)))
        legal = tuple(key for key in version_keys if self._matches_scope(query, corpus.source_metadata.get(key, {})))
        legal_set = set(legal)
        previous_spans = tuple(span for turn in snapshot.raw_history[-1:] for span in turn.source_spans
                               if span.source_key in legal_set)
        declared = supplied_blocks is not None or bool(families)
        complete_population = (coverage_scope == "all_supplied_documents"
            or any(need.scope == "all_supplied_documents" for need in query.task.needs)
            or (declared and query.task.coverage_scope == "all_supplied_documents"))
        query.task = replace(query.task, effective_source_scope=legal,
            coverage_scope="all_supplied_documents" if complete_population else "selected_sources")
        full = complete_population and declared
        available, screen_tokens = self._body_budgets(snapshot, query, available_kv_tokens)
        population_units = ()
        historical_ids = ()
        if full:
            document_blocks = corpus.resolve(legal)
            pool = directory.full_scope(legal)
            population_units = pool.units
            if supplied_blocks is not None and sum(unit.body_tokens for unit in pool.units) > screen_tokens:
                rows = numeric_field_units(question, pool, self.template.tokenizer)
                if rows and sum(unit.body_tokens for unit in rows) <= screen_tokens:
                    population_units = rows
            query.source_titles = tuple(block.title for block in pool.source_blocks)
            units = population_units
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
            for family, editions in versions.editions.items():
                ordered = [version for version in editions if version in wanted_versions]
                for before, after in zip(ordered, ordered[1:]):
                    pairs.extend(versions.paired_units(family, before, after, selected_spans))
            if pairs:
                units = tuple({unit.unit_id: unit for unit in (*pairs, *units)}.values())
                keys = tuple(dict.fromkeys(span.source_key for unit in units for span in unit.source_spans))
                pool = replace(pool, units=units, source_blocks=corpus.resolve(keys))
        elif not full and intent.mode in {"compare", "history"} and len(wanted_versions) < 2:
            candidate_spans = tuple(span for unit in units for span in unit.source_spans)
            evo_units = versions.evolution_units(question, candidate_spans, families=families)
            if evo_units:
                units = tuple({unit.unit_id: unit for unit in (*evo_units, *units)}.values())
                keys = tuple(dict.fromkeys(span.source_key for unit in units for span in unit.source_spans))
                pool = replace(pool, units=units, source_blocks=corpus.resolve(keys))
        if not full:
            document_blocks = pool.source_blocks
            query.task = replace(query.task, effective_source_scope=tuple(block.key for block in pool.source_blocks))
        document_index = (parsed_documents if parsed_documents is not None and document_blocks == supplied_blocks
                          else DocumentIndex.from_blocks(document_blocks))
        units = self._with_body_tokens(units)
        usage = dict(query.usage)
        by_id = {unit.unit_id: unit for unit in units}
        initial_ids = {unit.unit_id for unit in (units if full else units[:self.evidence_batch])}
        if supplied_blocks is not None:
            from controller.cooperative_request import document_headers, source_title_mentions
            documents = corpus.resolve(legal)
            headers = document_headers(documents, self.template.tokenizer)
            note, witnesses = source_title_mentions(documents, titles)
            if full and witnesses:
                headers += witnesses
                if sum(unit.body_tokens for unit in units) > screen_tokens:
                    initial_ids = ranked_population_windows(units, question, screen_tokens, self.template.tokenizer, witnesses)
        initial_ids.update(historical_ids)
        pending_dependencies = list(initial_ids)
        while pending_dependencies:
            for identity in by_id[pending_dependencies.pop()].dependencies:
                if identity not in initial_ids:
                    initial_ids.add(identity)
                    pending_dependencies.append(identity)
        initial = tuple(unit for unit in units if unit.unit_id in initial_ids)
        pending = tuple(unit for unit in units if unit.unit_id not in initial_ids)
        if candidates_only:
            return query, initial, pending
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
            population_units = pool.units
            document_index = DocumentIndex.from_blocks(pool.source_blocks)
            population_keys = {block.key for block in pool.source_blocks}
            positions = [index for index, unit in enumerate(initial)
                         if all(span.source_key in population_keys for span in unit.source_spans)]
            retained = tuple(initial[index] for index in positions)
            binding_units = document_index.binding_units()
            outside = document_index.outside_tables(pool.units)
            aliases = {}
            for unit in (*binding_units, *outside):
                original = next((prior for prior in retained if all(any(
                    old.source_key == span.source_key and old.start <= span.start and span.end <= old.end
                    for old in prior.source_spans) for span in unit.source_spans)), None)
                aliases[unit.unit_id] = original.unit_id if original is not None else unit.unit_id
            fresh = tuple(replace(unit,
                dependencies=tuple(dict.fromkeys(aliases[dependency] for dependency in unit.dependencies)))
                for unit in (*binding_units, *outside) if aliases[unit.unit_id] == unit.unit_id)
            fresh = self._with_body_tokens(fresh)
            evidence = replace(evidence, unit_ids=tuple(unit.unit_id for unit in retained),
                embeddings=evidence.embeddings[positions], relevance=evidence.relevance[positions],
                bindings=tuple(binding for binding in evidence.bindings
                               if binding.source_span.source_key in population_keys))
            if fresh:
                extra = self.evidence_encoder.encode(query, fresh)
                usage["input_tokens"] += extra.usage["input_tokens"]
                evidence = evidence.merge(extra)
            initial, pending = (*retained, *fresh), ()
            selected = ()
            query.task = replace(query.task,
                effective_source_scope=tuple(block.key for block in pool.source_blocks))
        state = SharedEvidenceState(query, evidence, replace(pool, units=tuple(initial)), selected,
            document_index, pending_units=tuple(pending), body_tokens=available, population_units=population_units,
            evidence_batch=self.evidence_batch)
        state.version_catalogue = catalogue_versions(catalogue, families)
        if wanted_versions and version_keys and state.version_catalogue:
            state.missing_version_identifiers = versions.missing_identifiers(
                question, version_keys, state.version_catalogue)
        if state.version_catalogue and (intent.mode in {"compare", "history"} or not wanted_versions):
            state.symbol_version_presence = versions.symbol_version_presence(question, families)
        if supplied_blocks is not None:
            state.source_headers = headers
            state.document_titles = titles
            state.title_note_budget = note
        self._finish_state(state)
        state.reused_claims = self._reuse_computations(state, facts)
        state.computations = state.reused_claims
        context = QueryContext(question, "shared", task_state=query.task,
            history_turns=tuple(dict.fromkeys(ref.turn_id for ref in query.task.working_history)))
        usage["compute_ms"] = (time.perf_counter()-started)*1000
        return EvidenceSelection(pool.source_blocks, context, auxiliary_usage=usage, shared_state=state)

    def _body_budgets(self, snapshot, query, available_kv_tokens):
        from controller.schema import Message
        from controller.cooperative_request import SYSTEM, working_history
        settings = self.config.get("execution", {})
        system = SYSTEM
        messages = (Message("system", system), Message("user", ""),
                    *history_messages(working_history(query, snapshot.raw_history)), Message("user", query.task.raw_question))
        prompt = self.template.render(messages)
        output = settings.get("output_tokens", 1024)
        available = min(self.config.get("selection", {}).get("body_tokens", 8192),
                        settings.get("window_tokens", 32768) - len(prompt.input_ids) - output)
        screen = available if available_kv_tokens is None else min(available, available_kv_tokens - output)
        return available, screen

    def _with_body_tokens(self, units):
        missing = [i for i, unit in enumerate(units) if not unit.body_tokens and unit.text]
        if not missing:
            return tuple(units)
        texts = [units[i].text for i in missing]
        counts = [len(ids) for ids in self.template.tokenizer(
            texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]]
        updates = dict(zip(missing, counts))
        return tuple(replace(unit, body_tokens=updates[i]) if i in updates else unit
                     for i, unit in enumerate(units))

    @staticmethod
    def _closure(roots, by_id):
        found, pending = {}, list(roots)
        while pending:
            unit = pending.pop(0)
            if unit.unit_id not in found:
                found[unit.unit_id] = unit
                pending.extend(by_id[identity] for identity in unit.dependencies)
        return tuple(found.values())

    @staticmethod
    def _reuse_computations(state, history):
        from controller.serialization import decode
        legal = set(state.query.task.effective_source_scope)
        programs = () if state.query.task.coverage_scope == "all_supplied_documents" else state.programs
        reused = []
        for past in history:
            for computation in past["computations"]:
                if not computation["complete"]:
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

    @staticmethod
    def _finish_state(state):
        from adapters.document_compute import build_programs
        state.programs = build_programs(state.query, state.evidence, state.document_index)

    async def expand(self, state):
        return await run_retrieval_worker(self._expand_state, state)

    def _expand_state(self, state):
        existing = set(state.evidence.unit_ids)
        all_units = {unit.unit_id: unit for unit in (*state.pool.units, *state.pending_units)}
        fresh = tuple(unit for unit in self._closure(state.pending_units[:self.evidence_batch], all_units)
                      if unit.unit_id not in existing)
        new = self.evidence_encoder.encode(state.query, fresh)
        evidence = state.evidence.merge(new)
        units = (*state.pool.units, *fresh)
        fresh_ids = {unit.unit_id for unit in fresh}
        result = replace(state, evidence=evidence, pool=replace(state.pool, units=units), selected_ids=(), costs=None,
                         pending_units=tuple(unit for unit in state.pending_units if unit.unit_id not in fresh_ids))
        self._finish_state(result)
        return result

    def close(self):
        self.query_encoder.close()
        self.evidence_encoder.close()
