"""Shared evidence, document computation and one-submission native execution."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import time
from weakref import WeakValueDictionary

from .cooperative_request import (SYSTEM, EvidenceCosts, blocks_for_units, compile_request,
                                  source_address, working_history)
from .schema import EvidenceBlock, EvidenceKey, QueryContext
from .serialization import decode, to_jsonable


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


def selected_blocks(state):
    """Selected units retain exact original-text coordinates and dependency text."""
    units = {unit.unit_id: unit for unit in state.pool.units}
    return blocks_for_units((units[identity] for identity in state.selected_ids), state.pool.source_blocks)


def computation_record(program, result):
    from adapters.shared_retrieval import computation_signature
    return {"need_id": program.need_id, "operation": program.operation,
            "input": computation_signature(program),
            "fields": list(dict.fromkeys(field.name for projection in program.projections for field in projection.fields)),
            "conditions": to_jsonable(program.predicates),
            "results": [{"entity": claim.entity, "value": claim.value, "unit": claim.unit}
                        for claim in result.conclusions],
            "sources": to_jsonable(result.source_spans), "complete": result.complete}


class CooperativeController:
    def __init__(self, store, engine, template, *, settings, policy, dialogue_mode="contextkv"):
        self.store, self.engine, self.template = store, engine, template
        self.settings, self.policy, self.dialogue_mode = settings, policy, dialogue_mode
        self._locks = WeakValueDictionary()
        self.native_state = {}
        self.engine_info = None
        self.kv_policy = {}
        if dialogue_mode == "contextkv":
            from .value_policy import StopValueHead, ValuePrices
            self.stop = StopValueHead.load(settings["stop_asset"])
            self.prices = ValuePrices(**settings.get("prices", {}))
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
        self.native_state.pop(session_id, None)
        await self.engine.close_session(session_id)

    def _lock(self, session_id):
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = asyncio.Lock()
        return lock

    async def serve_turn(self, session_id, query, *, request_id, task_id, provider,
                         corpus_snapshot_id="default", supplied_blocks=None, source_metadata=None,
                         coverage_scope="selected_sources", request_arrived=None):
        arrived = request_arrived if request_arrived is not None else time.monotonic()
        async with self._lock(session_id):
            snapshot = self.store.read_snapshot(session_id, corpus_snapshot_id)
            totals = {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0,
                      "device_hit_tokens": 0, "host_loaded_tokens": 0, "prefill_tokens": 0,
                      "calls": 0}
            auxiliary = {"compute_ms": 0.0}

            facts = ({"facts": self.store.committed_facts(snapshot)}
                     if self.dialogue_mode == "contextkv" else {})

            async def retrieve(question):
                if supplied_blocks is None:
                    return await provider.retrieve(snapshot, question, task_id, **facts)
                return await provider.retrieve_supplied(snapshot, question, task_id, supplied_blocks,
                    source_metadata=source_metadata, coverage_scope=coverage_scope, **facts)

            async def complete(request, name, *, schema=None, output_tokens=None):
                output_tokens = self.policy.max_new_tokens if output_tokens is None else output_tokens
                if len(request.input_ids) + output_tokens > self.policy.max_total_model_tokens:
                    raise ValueError("selected evidence and results exceed the answer window")
                if (totals["input_tokens"] + totals["output_tokens"] + len(request.input_ids) + output_tokens
                        > self.settings["max_total_tokens"]):
                    raise ValueError("the remaining computation exceeds the request token budget")
                result = await self.engine.compute(request, session_id=session_id,
                    request_id=f"{request_id}:{name}", max_new_tokens=output_tokens,
                    mode=self.settings.get("kv_mode", "auto") if self.dialogue_mode == "contextkv" else "exact",
                    **self.kv_policy, response_schema=schema)
                meta = result["meta"]
                native = meta["contextkv_compute"]
                self.native_state[session_id] = native
                totals["input_tokens"] += meta["prompt_tokens"]
                totals["output_tokens"] += meta["completion_tokens"]
                totals["elapsed_ms"] += result["elapsed_ms"]
                totals["device_hit_tokens"] += native["device_hit_tokens"]
                totals["host_loaded_tokens"] += native["host_loaded_tokens"]
                totals["prefill_tokens"] += native["prefill_tokens"]
                totals["calls"] += 1
                return result

            try:
                if self.dialogue_mode != "contextkv":
                    return await self._ordinary(snapshot, query, request_id, retrieve,
                                                complete, totals, arrived)
                selection = await retrieve(query)
                state = selection.shared_state
                snapshot = replace(snapshot, raw_history=working_history(state.query, snapshot.raw_history))
                auxiliary = selection.auxiliary_usage
                computations = list(state.reused_claims)
                computed_needs = {item["need_id"] for item in computations if item["complete"]}
                required = {need.need_id for need in state.query.task.needs
                            if need.operation in {"count", "sum", "group", "mean", "min", "max"}}
                population = state.query.task.coverage_scope == "all_supplied_documents"
                while True:
                    if not population:
                        await self._select_evidence(state, snapshot, computations)
                    outside = state.document_index.outside_tables(state.pool.units)
                    if not outside:
                        from adapters.shared_retrieval import need_scope_signature
                        for program in state.programs:
                            if program.need_id in computed_needs or not self._population_complete(state.document_index, program):
                                continue
                            result = state.document_index.execute(program)
                            record = computation_record(program, result)
                            record["scope"] = need_scope_signature(state.query, program.need_id, state.document_index)
                            record["conditions"] = record["scope"]["conditions"]
                            computations.append(record)
                            computed_needs.add(program.need_id)
                    complete_scope = required <= computed_needs
                    selected = selected_blocks(state)
                    feature_state = self._value_state(state, totals, arrived, complete_scope, session_id)
                    action = ("answer" if population else self.stop.choose(
                        state.evidence.summary(state.selected_ids), feature_state, self.prices,
                        can_continue=bool(state.pending_units), complete_scope=complete_scope))
                    if action == "continue":
                        state = await provider.expand(state)
                        auxiliary = {**auxiliary,
                            "compute_ms": auxiliary["compute_ms"] + state.evidence.usage["compute_ms"]}
                        continue
                    if not complete_scope:
                        computations.extend(await self._compute_documents(state, required-computed_needs,
                                                                           snapshot, complete,
                            selected if any(need.need_id not in required for need in state.query.task.needs) else ()))
                    break
                if population and (not state.query.task.needs or any(
                        need.need_id not in {item["need_id"] for item in computations} for need in state.query.task.needs)):
                    selected = await self._population_sources(state, snapshot, computations, complete)
                elif computations:
                    await self._select_evidence(state, snapshot, computations)
                    selected = selected_blocks(state)
                else:
                    selected = selected_blocks(state)
                request = compile_request(self.template, query, snapshot.raw_history, selected,
                    results=tuple(computations), system_prompt=self.policy.system_prompt or SYSTEM)
                final = await complete(request, "answer")
                source_keys = {decode(EvidenceKey, span["source_key"]) for item in computations for span in item["sources"]}
                calculation_sources = tuple(block for block in state.document_index.blocks.values()
                                            if block.key in source_keys)
                return self._finish(snapshot, query, request_id, request, final, totals, arrived,
                    auxiliary, state.query.task, tuple(computations), calculation_sources)
            except Exception as exc:
                self.store.fail(session_id, request_id, exc,
                    {"complete_ms": (time.monotonic()-arrived)*1000, **totals,
                     "auxiliary_ms": auxiliary["compute_ms"]})
                raise

    def _population_complete(self, index, program):
        used = ((0, *(join.right_index for join in program.joins)) if program.joins
                else range(len(program.projections)))
        covered = {program.projections[i].table_id for i in used} | set(program.excluded_table_ids)
        return set(program.population_table_ids) <= covered and not program.unresolved_conditions

    def _source_packs(self, units, input_tokens):
        """Pack source windows by their actual input size."""
        pack, inputs = [], 0
        for unit in units:
            tokens = len(self.template.tokenizer.encode(unit.text, add_special_tokens=False))
            if pack and inputs + tokens > input_tokens:
                yield tuple(pack)
                pack, inputs = [], 0
            pack.append(unit)
            inputs += tokens
        if pack:
            yield tuple(pack)

    async def _compute_documents(self, state, need_ids, snapshot, complete, selected):
        from adapters.document_compute import (DocumentIndex, combine_populations,
            merge_projections, semantic_binding, semantic_projection)
        from adapters.shared_retrieval import need_scope_signature
        from .schema import Message, history_messages
        index = state.document_index
        output_tokens = self.settings.get("semantic_output_tokens", 8192)
        fixed = self.template.render((Message("system", self.policy.system_prompt or SYSTEM),
            *history_messages(snapshot.raw_history), Message("user", state.query.task.raw_question)))
        input_tokens = min(self.settings.get("semantic_input_tokens", 8192),
            self.policy.max_total_model_tokens-output_tokens-len(fixed.input_ids))
        parts = [(index, program) for program in state.programs if program.need_id in need_ids
                 and self._population_complete(index, program)]
        bound = {program.need_id for _, program in parts}
        unbound = need_ids - bound
        if index.tables and unbound:
            table_units = index.binding_units()
            pending, size = [], 0
            packets = []
            for table in index.tables.values():
                units = [unit for unit in table_units if unit.table_id == table.table_id]
                tokens = sum(len(self.template.tokenizer.encode(unit.text, add_special_tokens=False)) for unit in units)
                if pending and size + tokens > input_tokens:
                    packets.append(tuple(pending))
                    pending, size = [], 0
                pending.append(table)
                size += tokens
            if pending:
                packets.append(tuple(pending))
            for number, tables in enumerate(packets):
                subset = DocumentIndex(tables, index.blocks.values())
                binding = semantic_binding(state.query, subset, sorted(unbound))
                request = compile_request(self.template, state.query.task.raw_question, snapshot.raw_history,
                    binding.blocks, instruction=binding.instruction, future_sources=selected,
                    system_prompt=self.policy.system_prompt or SYSTEM)
                generated = await complete(request, f"binding:{number}", schema=binding.schema,
                                           output_tokens=output_tokens)
                programs = binding.decode(generated["text"])
                ready = {program.need_id for program in programs if self._population_complete(subset, program)}
                if ready != unbound:
                    raise ValueError("document fields or table applicability remain unresolved")
                parts.extend((subset, program) for program in programs)
        outside = index.outside_tables(state.pool.units)
        if outside:
            projections = []
            population = tuple(unit.unit_id for unit in outside)
            for number, units in enumerate(self._source_packs(outside, input_tokens)):
                projection = semantic_projection(state.query, units, sorted(need_ids), population_unit_ids=population)
                request = compile_request(self.template, state.query.task.raw_question, snapshot.raw_history,
                    projection.blocks, instruction=projection.instruction, future_sources=selected,
                    system_prompt=self.policy.system_prompt or SYSTEM)
                generated = await complete(request, f"projection:{number}", schema=projection.schema,
                                           output_tokens=output_tokens)
                projections.append(projection.decode(generated["text"]))
            merged = merge_projections(projections)
            if merged.unresolved_need_ids or set(merged.covered_units) != set(population):
                raise ValueError("document projection did not resolve the complete requested population")
            parts.extend((merged.index, program) for program in merged.programs)
        calculated_index, programs = combine_populations(parts)
        computations = []
        for program in programs:
            result = calculated_index.execute(program)
            if not result.complete:
                raise ValueError(f"document population incomplete for {program.need_id}")
            record = computation_record(program, result)
            record["scope"] = need_scope_signature(state.query, program.need_id, index)
            record["conditions"] = record["scope"]["conditions"]
            computations.append(record)
        if {item["need_id"] for item in computations} != need_ids:
            raise ValueError("requested numeric computation has no complete source population")
        return computations

    def _answer_budget(self, state, snapshot, computations):
        from .cooperative_request import compact_computations
        from .schema import Message, history_messages
        import json
        question = state.query.task.raw_question
        if computations:
            question += "\nComputed results with original-source addresses:\n" + json.dumps(
                compact_computations(computations), ensure_ascii=False, separators=(",", ":"))
        fixed = self.template.render((Message("system", self.policy.system_prompt or SYSTEM), Message("user", ""),
                                     *history_messages(snapshot.raw_history), Message("user", question)))
        body_tokens = min(state.body_tokens, self.policy.max_total_model_tokens-self.policy.max_new_tokens-len(fixed.input_ids))
        return body_tokens, len(fixed.input_ids)

    async def _select_evidence(self, state, snapshot, computations):
        computed = {item["need_id"] for item in computations}
        if computations and all(need.need_id in computed for need in state.query.task.needs):
            state.selected_ids = ()
            return
        budget, prompt_tokens = self._answer_budget(state, snapshot, computations)
        blocks = blocks_for_units(state.pool.units, state.pool.source_blocks)
        native = await self.engine.evidence_state(tuple(source_address(block) for block in blocks))
        state.costs = EvidenceCosts(state.pool.units, state.pool.source_blocks, self.template.tokenizer,
            native, prompt_tokens=prompt_tokens, output_tokens=self.policy.max_new_tokens,
            policy=self.kv_policy, mode=self.settings.get("kv_mode", "auto"))
        state.selected_ids = state.evidence.select(state.pool.units, body_tokens=budget,
            cost_model=state.costs, seconds_price=self.prices.seconds)

    async def _population_sources(self, state, snapshot, computations, complete):
        from adapters.document_compute import semantic_evidence_projection
        budget, _ = self._answer_budget(state, snapshot, computations)
        units = state.population_units
        blocks = blocks_for_units(units, state.pool.source_blocks)
        encode = self.template.tokenizer.encode
        sizes = [len(encode(f"[S{i+1}] {block.title} | version={block.key.version}\n{block.text}\n\n",
                            add_special_tokens=False)) for i, block in enumerate(blocks)]
        if sum(sizes) <= budget:
            return blocks
        packs = tuple(self._source_packs(units, self.settings.get("semantic_input_tokens", 8192)))
        weights = [sum(len(encode(unit.text, add_special_tokens=False)) for unit in pack) for pack in packs]
        total, spent, projected = sum(weights), 0, {}
        for number, (pack, weight) in enumerate(zip(packs, weights)):
            quota = budget-spent if number == len(packs)-1 else int(budget*weight/total)
            spent += quota
            projection = semantic_evidence_projection(state.query.task.raw_question, pack)
            request = compile_request(self.template, state.query.task.raw_question, snapshot.raw_history,
                projection.blocks, instruction=projection.instruction,
                system_prompt=self.policy.system_prompt or SYSTEM)
            result = await complete(request, f"coverage:{number}", schema=projection.schema, output_tokens=quota)
            if result["meta"]["finish_reason"]["type"] == "length":
                raise ValueError("complete-population evidence exceeds its conclusion budget")
            for block in projection.decode(result["text"]):
                projected[source_address(block)] = block
        return tuple(projected.values())

    def _value_state(self, state, totals, arrived, complete_scope, session_id):
        native = {**self.kv_policy["costs"], **self.native_state.get(session_id, {})}
        row_count = sum(len(table.rows) for table in state.document_index.tables.values())
        return {"remaining_sources": len(state.pending_units),
                "open_needs": len(state.semantic_work), "required_rows": row_count,
                "processed_rows": row_count if complete_scope else 0,
                "input_tokens": sum(unit.body_tokens for unit in state.pool.units
                                    if unit.unit_id in state.selected_ids),
                "remaining_tokens": self.settings.get("max_total_tokens", 131072) - totals["input_tokens"] - totals["output_tokens"],
                "elapsed_seconds": time.monotonic()-arrived,
                "answer_cost_ms": state.costs.estimate(state.selected_ids) if state.costs else 0.0,
                **{name: native.get(name, 0) for name in (
                    "queue_depth", "gpu_ready_tokens", "host_ready_tokens", "prefill_token_ms", "transfer_token_ms")}}

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
            evaluation_evidence=tuple({(block.key, block.span): block
                for block in (*request.evidence, *calculation_sources)}.values()))
        self.store.commit(snapshot.session_id, request_id, query, final["text"], evidence=response.evaluation_evidence,
            task=task, facts={"computations": computations},
            metrics={"complete_ms": response.complete_ms, **totals})
        return response
