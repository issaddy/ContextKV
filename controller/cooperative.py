"""Shared evidence, document computation and one-submission native execution."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import json
import time
from weakref import WeakValueDictionary

from .cooperative_request import (SYSTEM, ANSWER_SCOPE, EvidenceCosts, blocks_for_units, compile_request,
                                  source_address, source_body, with_document_headers, working_history)
from .schema import EvidenceKey, QueryContext
from .serialization import decode, to_jsonable


AGGREGATES = frozenset({"count", "sum", "mean", "min", "max"})


@dataclass
class AnswerResult:
    session_id: str
    request_id: str
    assistant_text: str
    evidence: tuple
    query_context: QueryContext | None
    visible_evidence_manifest: tuple
    reference_map: tuple
    input_tokens: int
    output_tokens: int
    output_ids: tuple
    complete_ms: float
    total_input_tokens: int
    total_output_tokens: int
    auxiliary_model_ms: float
    total_device_hit_tokens: int
    total_host_loaded_tokens: int
    total_prefill_tokens: int
    engine_elapsed_ms: float
    read_count: int = 0
    reduce_count: int = 0
    truncated: bool = False
    computations: tuple = ()
    native: dict = field(default_factory=dict)
    evaluation_evidence: tuple = ()
    new_compute_kv_pages: int = 0


def selected_blocks(state):
    """Selected units retain exact original-text coordinates and dependency text."""
    units = {unit.unit_id: unit for unit in state.pool.units}
    return blocks_for_units((units[identity] for identity in state.selected_ids), state.pool.source_blocks)


def execution_totals():
    return {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0,
            "device_hit_tokens": 0, "host_loaded_tokens": 0, "prefill_tokens": 0, "calls": 0,
            "new_compute_kv_pages": 0}


def computation_record(program, result):
    from adapters.shared_retrieval import computation_signature
    observations = program.operation == "lookup" or not result.complete
    return {"need_id": program.need_id, "operation": program.operation,
            "input": computation_signature(program),
            "fields": list(dict.fromkeys(field.name for projection in program.projections for field in projection.fields)),
            "conditions": to_jsonable(program.predicates),
            "results": [{"entity": claim.entity, "qualifiers": claim.qualifiers,
                         **({"fields": claim.attributes} if observations else
                            {"field": claim.relation, "value": claim.value, "unit": claim.unit})}
                        for claim in result.conclusions],
            "sources": to_jsonable(result.source_spans), "complete": result.complete}


class CooperativeController:
    def __init__(self, store, engine, template, *, settings, policy, dialogue_mode="contextkv"):
        self.store, self.engine, self.template = store, engine, template
        self.settings, self.policy, self.dialogue_mode = settings, policy, dialogue_mode
        self._locks = WeakValueDictionary()
        self.engine_info = None
        self.kv_policy = {}
        if dialogue_mode == "contextkv":
            from .config import load_kv_policy
            self.kv_policy = load_kv_policy(settings)

    async def start(self):
        if self.engine_info is None:
            self.engine_info = await self.engine.info()
        return self.engine_info

    async def aclose(self):
        await self.engine.aclose()

    async def close_session(self, session_id):
        self._locks.pop(session_id, None)
        await self.engine.close_session(session_id)

    def _lock(self, session_id):
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = asyncio.Lock()
        return lock

    def executor(self, session_id, request_id, totals, *, cache_scope="shared", cache_namespace=None):
        async def complete(request, name, *, schema=None, output_tokens=None):
            output_tokens = self.policy.max_new_tokens if output_tokens is None else output_tokens
            remaining = self.settings["max_total_tokens"]-totals["input_tokens"]-totals["output_tokens"]
            answer_output = self.policy.max_new_tokens if schema is not None else 0
            output_tokens = min(output_tokens, self.policy.max_total_model_tokens-len(request.input_ids),
                                remaining-len(request.input_ids)-answer_output)
            if output_tokens <= 0:
                raise ValueError("the request input exhausts the remaining token budget")
            result = await self.engine.compute(request, session_id=session_id,
                request_id=f"{request_id}:{totals['calls']}:{name}", max_new_tokens=output_tokens,
                mode=self.settings.get("kv_mode", "exact") if self.dialogue_mode == "contextkv" else "exact",
                **self.kv_policy, response_schema=schema, cache_scope=cache_scope, cache_namespace=cache_namespace)
            meta = result["meta"]
            native = meta["contextkv_compute"]
            totals["input_tokens"] += meta["prompt_tokens"]
            totals["output_tokens"] += meta["completion_tokens"]
            totals["elapsed_ms"] += result["elapsed_ms"]
            totals["device_hit_tokens"] += native["device_hit_tokens"]
            totals["host_loaded_tokens"] += native["host_loaded_tokens"]
            totals["prefill_tokens"] += native["prefill_tokens"]
            totals["new_compute_kv_pages"] += native["new_compute_kv_pages"]
            totals["calls"] += 1
            return result
        return complete

    async def serve_turn(self, session_id, query, *, request_id, task_id, provider,
                         corpus_snapshot_id="default", supplied_blocks=None, source_metadata=None,
                         coverage_scope="selected_sources", request_arrived=None):
        arrived = request_arrived if request_arrived is not None else time.monotonic()
        async with self._lock(session_id):
            snapshot = self.store.read_snapshot(session_id, corpus_snapshot_id)
            totals = execution_totals()
            auxiliary = {"compute_ms": 0.0}

            facts = ({"facts": self.store.committed_facts(snapshot)}
                     if self.dialogue_mode == "contextkv" else {})

            async def retrieve(question):
                if supplied_blocks is None:
                    return await provider.retrieve(snapshot, question, task_id, **facts)
                return await provider.retrieve_supplied(snapshot, question, task_id, supplied_blocks,
                    source_metadata=source_metadata, coverage_scope=coverage_scope, **facts)

            complete = self.executor(session_id, request_id, totals)

            try:
                if self.dialogue_mode != "contextkv":
                    return await self._ordinary(snapshot, query, request_id, retrieve,
                                                complete, totals, arrived)
                selection = await retrieve(query)
                state = selection.shared_state
                state.totals = totals
                snapshot = replace(snapshot, raw_history=working_history(state.query, snapshot.raw_history))
                auxiliary = selection.auxiliary_usage
                auxiliary["compute_ms"] += await self.prepare_state(state, snapshot, complete, provider=provider)
                request = self.answer_request(state, snapshot)
                computations = state.computations
                final = await complete(request, "answer")
                source_keys = {decode(EvidenceKey, span["source_key"]) for item in computations for span in item["sources"]}
                calculation_sources = tuple(block for block in state.document_index.blocks
                                            if block.key in source_keys)
                return self._finish(snapshot, query, request_id, request, final, totals, arrived,
                    auxiliary, state.query.task, tuple(computations), calculation_sources)
            except Exception as exc:
                self.store.fail(session_id, request_id, exc,
                    {"complete_ms": (time.monotonic()-arrived)*1000, **totals,
                     "auxiliary_ms": auxiliary["compute_ms"]})
                raise

    async def prepare_state(self, state, snapshot, complete, *, provider=None):
        """Choose necessary work before encoding a fixed source population."""
        auxiliary_ms = 0.0
        if state.needs_complete:
            state.selected_ids = ()
            return auxiliary_ms
        if state.evidence is None:
            compute = self.settings.get("document_compute", True) and any(
                need.operation in AGGREGATES | {"group"} for need in state.query.task.needs)
            if compute and state.document_index.tables:
                usage = await provider.bind_tables(state)
                auxiliary_ms += usage["compute_ms"]
                if not state.document_index.outside_tables(state.population_units):
                    await self.read_required(state, snapshot, complete)
                    if state.needs_complete:
                        state.selected_ids = ()
                        return auxiliary_ms
            if await self._source_answer(state, snapshot, state.population_units):
                return auxiliary_ms
            if compute:
                await self.read_required(state, snapshot, complete)
                if state.needs_complete:
                    state.selected_ids = ()
                    return auxiliary_ms
            usage = await provider.encode_evidence(state)
            auxiliary_ms += usage["compute_ms"]
            await self._select_evidence(state, snapshot, state.computations)
            return auxiliary_ms
        await self._select_evidence(state, snapshot, state.computations)
        if self.settings.get("document_compute", True) is False:
            return auxiliary_ms
        previous = state.computations
        paid_calls = state.totals["calls"]
        await self.read_required(state, snapshot, complete)
        if state.needs_complete:
            state.selected_ids = ()
            return auxiliary_ms
        if state.computations != previous or state.totals["calls"] != paid_calls:
            state.costs = None
            await self._select_evidence(state, snapshot, state.computations)
        return auxiliary_ms

    async def _source_answer(self, state, snapshot, units):
        """Compile known source context once when it fits the actual Native capacity."""
        from engine_patches.kv_cost import page_tokens
        budget, _ = self._answer_budget(state, snapshot, state.computations)
        if sum(unit.body_tokens for unit in units) > budget:
            return False
        state.selected_ids = tuple(unit.unit_id for unit in units)
        request = self.answer_request(state, snapshot)
        tokens = len(request.input_ids) + self.policy.max_new_tokens
        remaining = self.settings["max_total_tokens"]-state.totals["input_tokens"]-state.totals["output_tokens"]
        info = await self.start()
        if (len(request.input_ids) <= info.max_input_tokens
                and tokens <= min(self.policy.max_total_model_tokens, remaining)
                and page_tokens(tokens, info.page_size) <= info.gpu_capacity_tokens):
            state.compiled_answer = request
            return True
        state.selected_ids = ()
        return False

    async def read_required(self, state, snapshot, complete):
        from adapters.shared_retrieval import need_scope_signature
        reused = state.completed_needs
        population = state.query.task.coverage_scope == "all_supplied_documents"
        aggregates = {need.need_id for need in state.query.task.needs if need.operation in AGGREGATES} - reused
        required = aggregates | {program.need_id for program in state.programs
                                 if program.operation == "group" and program.need_id not in reused}
        if not required:
            return
        if population:
            outside = state.document_index.outside_tables(state.population_units)
            if outside:
                if not {unit.unit_id for unit in state.population_units}.issubset(state.selected_ids):
                    numeric = aggregates.difference(state.query.join_needs)
                    numeric.update(program.need_id for program in state.programs
                                   if program.need_id in aggregates and program.joins
                                   and self._population_complete(state.document_index, program))
                    if numeric:
                        await self._read_mixed_population(state, snapshot, complete, sorted(numeric))
                return
            selected_keys = {span.source_key for unit in state.population_units for span in unit.source_spans}
        else:
            blocks = selected_blocks(state)
            selected_keys = {block.key for block in blocks}
            outside = state.document_index.outside_tables(
                tuple(unit for unit in state.pool.units if unit.unit_id in state.selected_ids))
        bound = [program for program in state.programs if program.need_id in required
                 and program.operation in AGGREGATES | {"group"}
                 and not outside
                 and self._population_complete(state.document_index, program)
                 and all(state.document_index.tables[projection.table_id].source.source_key in selected_keys
                         for projection in program.projections)]
        records = {record["need_id"]: record for record in state.computations}
        for program in bound:
            record = computation_record(program, state.document_index.execute(program))
            if population:
                record["scope"] = need_scope_signature(state.query, program.need_id, state.document_index)
            records[program.need_id] = record
        state.computations = tuple(records.values())
        if population:
            return
        budget, _ = self._answer_budget(state, snapshot, state.computations)
        raw_size = len(self.template.tokenizer.encode(source_body(blocks)[0], add_special_tokens=False))
        if raw_size <= budget:
            return
        fresh = any(unit.unit_id not in state.read_units for unit in outside)
        if not state.table_read or fresh or state.programs != state.prepared_programs:
            need_ids = ({need.need_id for need in state.query.task.needs} - reused
                        if not state.table_read else required)
            results = await self._compute_documents(state, need_ids, snapshot, complete, outside)
            state.prepared_programs = state.programs
            merged = {record["need_id"]: record for record in (*state.computations, *results)}
            state.computations = tuple(merged.values())
            state.query.task = replace(state.query.task, needs=tuple(replace(need,
                operation=state.resolved_operations.get(need.need_id, need.operation))
                for need in state.query.task.needs))

    def answer_request(self, state, snapshot):
        """Online answers and training branches use this same conclusion path."""
        if state.compiled_answer is not None:
            return state.compiled_answer
        if state.needs_complete:
            selected = ()
        else:
            selected = selected_blocks(state)
        selected = with_document_headers(selected, state.source_headers)
        return compile_request(self.template, state.query.task.raw_question, snapshot.raw_history, selected,
            results=state.computations, system_prompt=self.policy.system_prompt or SYSTEM,
            instruction=self._answer_instruction(state))

    def _reading_allowance(self, state, snapshot, request, output_tokens):
        """Every read funds its own output and the reserved terminal context first."""
        _, base = self._answer_budget(state, snapshot, ())
        _, known = self._answer_budget(state, snapshot, state.computations)
        answer = max(known+self.policy.max_new_tokens,
            min(self.policy.max_total_model_tokens, base+state.body_tokens+self.policy.max_new_tokens))
        remaining = self.settings["max_total_tokens"]-state.totals["input_tokens"]-state.totals["output_tokens"]
        return max(0, min(output_tokens, self.policy.max_total_model_tokens-len(request.input_ids),
                          remaining-len(request.input_ids)-answer))

    def _population_complete(self, index, program):
        used = ((0, *(join.right_index for join in program.joins)) if program.joins
                else range(len(program.projections)))
        covered = {program.projections[i].table_id for i in used} | set(program.excluded_table_ids)
        return set(program.population_table_ids) <= covered and not program.unresolved_conditions

    def _source_packs(self, units, input_tokens, source_text, *, dependencies=None):
        """Pack source windows by their actual input size."""
        from adapters.shared_retrieval import SharedEvidenceProvider
        pack, inputs, seen, sizes = [], 0, set(), {}
        for unit in units:
            sources = (SharedEvidenceProvider._closure((unit,), dependencies)
                       if dependencies is not None else (unit,))
            for source in sources:
                if source.unit_id not in sizes:
                    sizes[source.unit_id] = len(self.template.tokenizer.encode(
                        source_text(source), add_special_tokens=False))
            tokens = sum(sizes[source.unit_id] for source in sources if source.unit_id not in seen)
            if pack and inputs + tokens > input_tokens:
                yield tuple(pack)
                pack, inputs, seen = [], 0, set()
                tokens = sum(sizes[source.unit_id] for source in sources)
            pack.append(unit)
            inputs += tokens
            seen.update(source.unit_id for source in sources)
        if pack:
            yield tuple(pack)

    async def _read_mixed_population(self, state, snapshot, complete, need_ids):
        from itertools import chain, pairwise
        from adapters.document_compute import (DocumentIndex, literal_population_projection,
            merge_bound_population, merge_projections, population_relations)
        from adapters.shared_retrieval import SharedEvidenceProvider, need_scope_signature
        index = state.document_index
        bound = {program.need_id: program for program in state.programs if program.need_id in need_ids
                 and self._population_complete(index, program)}
        groups = {}
        for need_id in need_ids:
            program = bound.get(need_id)
            tables = (tuple(sorted({projection.table_id for projection in program.projections}
                                  | set(program.excluded_table_ids))) if program else ())
            groups.setdefault(tables, []).append(need_id)
        input_limit = self.settings.get("semantic_input_tokens", 8192)
        records = {record["need_id"]: record for record in state.computations}
        for group_index, (table_ids, targets) in enumerate(groups.items()):
            parts = [(index, bound[identity]) for identity in targets if identity in bound]
            relations = population_relations(index, [program for _, program in parts])
            primary = index.outside_tables(state.population_units, table_ids) if table_ids else state.population_units
            used = {projection.table_id for _, program in parts for projection in program.projections}
            headers = DocumentIndex((table for key, table in index.tables.items() if key in used),
                                    index.blocks).binding_units(sample_rows=0)
            primary = tuple(replace(unit, dependencies=tuple(dict.fromkeys((*unit.dependencies,
                *(header.unit_id for header in headers
                  if header.source_spans[0].source_key == unit.source_spans[0].source_key))))) for unit in primary)
            by_id = {unit.unit_id: unit for unit in (*primary, *headers)}
            population = tuple(unit.unit_id for unit in primary)
            packs = self._source_packs(primary, input_limit, lambda unit: unit.text, dependencies=by_id)
            packets = ((roots, SharedEvidenceProvider._closure(roots, by_id)) for roots in packs)
            reads, processed_primary = [], set()
            # Only the next packet is needed to describe future KV reuse.
            for i, ((roots, packet), (_, upcoming)) in enumerate(pairwise(chain(packets, (((), ()),)))):
                projection = literal_population_projection(state.query, packet, state.pool.source_blocks,
                    targets, population, {unit.unit_id for unit in roots}, relations=relations)
                future = blocks_for_units(upcoming, state.pool.source_blocks)
                request = compile_request(self.template, state.query.task.raw_question, snapshot.raw_history,
                    projection.blocks, future_sources=future, instruction=projection.instruction,
                    system_prompt=self.policy.system_prompt or SYSTEM)
                limit = self._reading_allowance(state, snapshot, request, self.settings.get("semantic_output_tokens", 8192))
                if not limit:
                    state.budget_exhausted = True
                    break
                result = await complete(request, f"population:{group_index}:{i}", schema=projection.schema, output_tokens=limit)
                if result["meta"]["finish_reason"]["type"] == "length":
                    state.budget_exhausted = True
                    break
                reads.append(projection.decode(result["text"]))
                processed_primary.update(unit.unit_id for unit in roots)
                state.read_units.update(unit.unit_id for unit in roots)
            if reads:
                if parts:
                    parts = merge_bound_population(reads, index, [program for _, program in parts])
                else:
                    merged = merge_projections(reads)
                    parts = [(merged.index, program) for program in merged.programs]
            for calculated, program in parts:
                try:
                    if any(identity not in processed_primary for identity in population):
                        program = replace(program, unresolved_conditions=(*program.unresolved_conditions, "unread_population"))
                    result = calculated.execute(program)
                except (KeyError, IndexError, ValueError, ArithmeticError):
                    # Conflicting fields, grouping, or units leave the Need open for the final answer.
                    records.pop(program.need_id, None)
                    continue
                record = computation_record(program, result)
                need = next(n for n in state.query.task.needs if n.need_id == program.need_id)
                record["target"] = state.query.text(need.anchor_spans)
                record["scope"] = need_scope_signature(state.query, program.need_id, state.document_index)
                records[program.need_id] = record
        state.computations = tuple(records.values())

    async def _compute_documents(self, state, need_ids, snapshot, complete, outside):
        from adapters.document_compute import (DocumentIndex, combine_populations,
            merge_projections, semantic_binding, semantic_projection)
        from adapters.shared_retrieval import need_scope_signature
        from .schema import Message, history_messages
        from .source_windows import numbered_text
        index = state.document_index
        output_tokens = self.settings.get("semantic_output_tokens", 8192)
        fixed = self.template.render((Message("system", self.policy.system_prompt or SYSTEM),
            *history_messages(snapshot.raw_history), Message("user", state.query.task.raw_question)))
        input_tokens = min(self.settings.get("semantic_input_tokens", 8192),
            self.policy.max_total_model_tokens-output_tokens-len(fixed.input_ids))
        semantic_needs = {program.need_id for _, program in state.table_parts}
        parts = [(index, program) for program in state.programs if program.need_id in need_ids
                 and program.need_id not in semantic_needs and self._population_complete(index, program)]
        resolved_operations = state.resolved_operations
        bound = {program.need_id for _, program in parts}
        unbound = need_ids - bound
        if index.tables and unbound and not state.table_read:
            table_units = index.binding_units()
            pending, size = [], 0
            packets = []
            for table in index.tables.values():
                units = [unit for unit in table_units if unit.table_id == table.table_id]
                tokens = sum(len(self.template.tokenizer.encode(numbered_text(unit.text)[0], add_special_tokens=False)) for unit in units)
                if pending and size + tokens > input_tokens:
                    packets.append(tuple(pending))
                    pending, size = [], 0
                pending.append(table)
                size += tokens
            if pending:
                packets.append(tuple(pending))
            for number, tables in enumerate(packets):
                subset = DocumentIndex(tables, index.blocks)
                binding = semantic_binding(state.query, subset, sorted(unbound), operations=resolved_operations)
                request = compile_request(self.template, state.query.task.raw_question, snapshot.raw_history,
                    binding.blocks, instruction=binding.instruction,
                    system_prompt=self.policy.system_prompt or SYSTEM)
                allowance = self._reading_allowance(state, snapshot, request, output_tokens)
                if not allowance:
                    state.budget_exhausted = True
                    break
                generated = await complete(request, f"binding:{number}", schema=binding.schema,
                                           output_tokens=allowance)
                if generated["meta"]["finish_reason"]["type"] == "length":
                    state.budget_exhausted = True
                    break
                programs = binding.decode(generated["text"])
                resolved_operations.update((program.need_id, program.operation) for program in programs)
                state.table_parts = (*state.table_parts, *((subset, program) for program in programs))
        state.table_read = True
        parts.extend(state.table_parts)
        fresh = tuple(unit for unit in outside if unit.unit_id not in state.read_units)
        operations_by_need = {need.need_id: need.operation for need in state.query.task.needs} | resolved_operations
        aggregate_needs = {identity for identity in need_ids
                       if operations_by_need[identity] in AGGREGATES}
        if fresh and aggregate_needs and not state.budget_exhausted:
            operations = resolved_operations or None
            population = tuple(unit.unit_id for unit in outside)
            for number, units in enumerate(self._source_packs(fresh, input_tokens, lambda unit: numbered_text(unit.text)[0])):
                projection = semantic_projection(state.query, units, sorted(need_ids),
                    population_unit_ids=population, operations=operations)
                request = compile_request(self.template, state.query.task.raw_question, snapshot.raw_history,
                    projection.blocks, instruction=projection.instruction,
                    system_prompt=self.policy.system_prompt or SYSTEM)
                allowance = self._reading_allowance(state, snapshot, request, output_tokens)
                if not allowance:
                    state.budget_exhausted = True
                    break
                generated = await complete(request, f"projection:{len(state.projections)}", schema=projection.schema,
                                           output_tokens=allowance)
                if generated["meta"]["finish_reason"]["type"] == "length":
                    state.budget_exhausted = True
                    break
                parsed = projection.decode(generated["text"])
                state.projections = (*state.projections, parsed)
                state.read_units.update(unit.unit_id for unit in units)
                operations = parsed.operations
                resolved_operations.update(operations)
        if state.projections:
            merged = merge_projections(state.projections)
            parts.extend((merged.index, program) for program in merged.programs)
        parts = [(part, replace(program, operation=resolved_operations.get(program.need_id, program.operation)))
                 for part, program in parts if program.need_id in need_ids]
        calculated_index, programs = combine_populations(parts)
        computations = []
        for program in programs:
            unread = any(unit.unit_id not in state.read_units for unit in outside)
            if state.budget_exhausted or (unread and state.query.task.coverage_scope == "all_supplied_documents"):
                program = replace(program, unresolved_conditions=(*program.unresolved_conditions, "unread_population"))
            result = calculated_index.execute(program)
            record = computation_record(program, result)
            record["scope"] = need_scope_signature(state.query, program.need_id, index)
            computations.append(record)
        return computations

    def _answer_instruction(self, state):
        instruction = ANSWER_SCOPE
        processed = {record["need_id"] for record in state.computations if record["complete"]}
        if state.budget_exhausted or (
                state.query.task.coverage_scope == "all_supplied_documents"
                and any(need.operation in AGGREGATES and need.need_id not in processed
                        for need in state.query.task.needs)
                and not {unit.unit_id for unit in state.population_units}.issubset(state.selected_ids)):
            instruction += ("\nUse the supplied results and source excerpts for each target. "
                            "Identify any unread part of the required scope before giving a total.")
        if state.version_catalogue:
            instruction += "\nAvailable document versions from the source index:\n" + json.dumps(
                state.version_catalogue, ensure_ascii=False, separators=(",", ":"))
        return instruction

    def _answer_budget(self, state, snapshot, computations):
        from .cooperative_request import answer_task
        from .schema import Message, history_messages
        question = answer_task(state.query.task.raw_question, computations)+"\n"+self._answer_instruction(state)
        headers = "".join(f"[D{i+1}] {block.title}\n{block.text}\n\n"
                          for i, block in enumerate(state.source_headers))
        fixed = self.template.render((Message("system", self.policy.system_prompt or SYSTEM), Message("user", headers),
                                     *history_messages(snapshot.raw_history), Message("user", question)))
        remaining = self.settings["max_total_tokens"]-state.totals["input_tokens"]-state.totals["output_tokens"]
        body_tokens = min(state.body_tokens,
            min(self.policy.max_total_model_tokens, remaining)-self.policy.max_new_tokens-len(fixed.input_ids))
        return body_tokens, len(fixed.input_ids)

    async def _select_evidence(self, state, snapshot, computations):
        computed = state.completed_needs
        budget, prompt_tokens = self._answer_budget(state, snapshot, computations)
        units = state.pool.units
        blocks = blocks_for_units(units, state.document_index.blocks)
        available_sources = {source_address(block) for block in (*state.source_headers, *blocks)}
        native = (state.costs.snapshot if state.costs is not None else
            await self.engine.evidence_state(tuple(available_sources),
                cache_scope=state.cache_scope, cache_namespace=state.cache_namespace,
                mode=self.settings.get("kv_mode", "exact")))
        def select():
            costs = EvidenceCosts(units, state.document_index.blocks, self.template.tokenizer,
                native, prompt_tokens=prompt_tokens, output_tokens=self.policy.max_new_tokens,
                policy=self.kv_policy, mode=self.settings.get("kv_mode", "exact"), source_headers=state.source_headers)
            if state.query.task.coverage_scope == "all_supplied_documents":
                ids = state.evidence.project_population(units, body_tokens=budget, cost_model=costs)
            elif self.settings.get("evidence_selection", "utility") == "ranked":
                sizes = {unit.unit_id: unit.body_tokens for unit in units}
                frontier = state.evidence.frontier(units)
                ids = costs.order(next((ids for ids in reversed(frontier)
                    if sum(sizes[identity] for identity in ids) <= budget), ()))
            else:
                ids = state.evidence.select(units, body_tokens=budget,
                    cost_model=costs,
                    need_ids=tuple(need.need_id for need in state.query.task.needs), completed=computed)
            return costs, ids
        state.costs, state.selected_ids = await asyncio.to_thread(select)

    async def _ordinary(self, snapshot, query, request_id, retrieve_evidence, complete, totals, arrived):
        from .cooperative_request import CompiledRequest
        from adapters import rag_dialogue, llamaindex_dialogue
        auxiliary = {"compute_ms": 0.0}
        last_request = last_result = None

        async def retrieve(question):
            selection = await retrieve_evidence(question)
            auxiliary["compute_ms"] += selection.auxiliary_usage["compute_ms"]
            return selection.evidence

        async def generate(call):
            nonlocal last_request, last_result
            rendered = self.template.render(call.messages)
            last_request = CompiledRequest(rendered.input_ids, call.messages, call.source_blocks,
                call.evidence_spans, call.reference_map, (), ())
            last_result = await complete(last_request, call.step_id)
            return last_result["text"]

        runner = rag_dialogue if self.dialogue_mode == "rag" else llamaindex_dialogue
        settings = {"template": self.template} if self.dialogue_mode == "rag" else {"tokenizer": self.template.tokenizer}
        await runner.run_dialogue(query, snapshot.raw_history, retrieve_callback=retrieve,
            complete_callback=generate, context_window=self.policy.max_total_model_tokens,
            output_tokens=self.policy.max_new_tokens, history_tokens=self.settings.get("history_tokens", 8192),
            system_prompt=self.policy.system_prompt or SYSTEM, **settings)
        return self._finish(snapshot, query, request_id, last_request, last_result, totals, arrived,
                            auxiliary, None, ())

    def _finish(self, snapshot, query, request_id, request, final, totals, arrived, auxiliary, task, computations,
                calculation_sources=()):
        meta, native = final["meta"], final["meta"]["contextkv_compute"]
        response = AnswerResult(snapshot.session_id, request_id, final["text"], request.evidence,
            QueryContext(query, "shared", task_state=task), request.spans, request.reference_map,
            meta["prompt_tokens"], meta["completion_tokens"], final["output_ids"],
            (time.monotonic()-arrived)*1000, totals["input_tokens"],
            totals["output_tokens"], auxiliary["compute_ms"], totals["device_hit_tokens"],
            totals["host_loaded_tokens"], totals["prefill_tokens"], totals["elapsed_ms"],
            read_count=totals["calls"]-1, reduce_count=len(computations),
            truncated=meta["finish_reason"]["type"] == "length", computations=computations, native=native,
            new_compute_kv_pages=totals["new_compute_kv_pages"],
            evaluation_evidence=tuple({(block.key, block.span): block
                for block in (*request.evidence, *calculation_sources)}.values()))
        self.store.commit(snapshot.session_id, request_id, query, final["text"], evidence=response.evaluation_evidence,
            task=task, facts={"computations": computations},
            metrics={"complete_ms": response.complete_ms, **totals})
        return response
