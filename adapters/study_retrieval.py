"""Task/support/search/reading assembly shared by serving and study evaluation."""
from __future__ import annotations

from dataclasses import replace
import threading
import time
from pathlib import Path

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import PoolProfile, SourceDirectory, merge_retrieved_units
from adapters.history_state import HistoryIndex, TaskInterpreter, TeacherTaskModel, restore_source_metadata
from adapters.retrieval import run_retrieval_worker
from adapters.retriever_factory import create_retriever
from adapters.serving_retrieval import ServingRetriever
from adapters.support_model import SetPrediction, SupportModel, SupportProfile
from controller.candidates import CandidatePolicy, build_plan_candidates, measure_plan_contexts
from controller.config import load_config
from controller.context_budget import resolve_text, working_history_messages
from controller.context_schema import (AnswerPlan, AnswerPoint, ContextBudget, ContextPlan,
    EvidencePool, EvidenceSet, Need, SupportScores, TaskState, TextRef)
from controller.fact_reuse import build_reuse_seed, complete_reuse_ranges
from controller.evidence_planner import (BudgetedEvidencePlanner, SearchProfile, _passages,
    best_need_supports, close_dependencies, render_unit_union, unresolved_needs)
from controller.reading import build_read_plan
from controller.schema import OnlineMessage, OnlineTurn, QueryContext
from controller.serialization import to_jsonable
from controller.service import EvidenceSelection


STUDY_STRATEGIES = ("B0", "Btoken", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "Lmap")


class SupportCoverageUtility:
    """Mean Need coverage from the model-scored single/pair/triple factors."""
    def predict_set(self, state, units, supports):
        best, _ = best_need_supports(state, units, supports)
        quality = sum(s.applicability*s.joint_support for s in best)/max(1, len(state.needs))
        complete = len(best) == len(state.needs) and all(
            s.disposition == "complete" and s.applicability*s.joint_support > 0 for s in best)
        return SetPrediction(quality, "answer" if quality >= .8 and complete else "partial")


def study_budget(config):
    execution = config.get("execution", {})
    selection = config.get("selection", {})
    reading = config.get("reading", {})
    return ContextBudget(window_tokens=execution.get("window_tokens", 32768),
        body_tokens=selection.get("body_tokens", 8192), history_tokens=execution.get("history_tokens", 8192),
        output_tokens=execution.get("output_tokens", 1024), read_input_tokens=reading.get("input_target_tokens", 12288),
        read_output_tokens=reading.get("output_tokens", 2048), max_steps=execution.get("max_steps", 512),
        max_total_tokens=execution.get("max_total_tokens", 4194304),
        max_passages=selection.get("max_passages"), max_complete_ms=execution.get("max_complete_ms"),
        lambda_time=execution.get("lambda_time", 0.0), lambda_gpu=execution.get("lambda_gpu", 0.0),
        lambda_host=execution.get("lambda_host", 0.0))


class StudyEvidenceProvider:
    def __init__(self, config, template, serving, directory, *, support_backend=None,
                 frozen_history=None, task_model=None, teacher_client=None, helper_engine_info=None,
                 query_encoder_identity=None):
        self.config, self.template, self.serving, self.directory = config, template, serving, directory
        self.support_backend, self.frozen_history = support_backend, frozen_history
        self.task_model, self.teacher_client = task_model, teacher_client
        self.helper_engine_info = helper_engine_info
        self.query_encoder_identity = query_encoder_identity
        self.strategy = config.get("study", {}).get("strategy", "B7")
        self.external_dialogue = config.get("execution", {}).get("dialogue_mode") == "llamaindex"
        self.budget = study_budget(config)
        run_config = config.get("generation", {}).get("run_config")
        runtime = load_config(run_config) if run_config else None
        self.answer_system_prompt = runtime.context.system_prompt if runtime else None
        plan_gain = config.get("execution", {}).get("plan_gain", runtime.plan_gain if runtime else None)
        self.whole_plan_candidates = plan_gain is not None and plan_gain["mode"] == "whole_plan"
        self.corpus_snapshot_id = serving.corpus_snapshot_id
        self.traces = {}
        self._lock = threading.Lock()
        self.committed_readings = None

    def finish_request(self, task_id):
        """Release support predictions after this QA, including its GAP rounds."""
        with self._lock:
            if self.support_backend is not None:
                owners = {task_id, *(trace["support_task_id"] for (_, request_id), trace in self.traces.items()
                    if request_id == task_id and "support_task_id" in trace)}
                for owner in owners:
                    self.support_backend.release_task(owner)

    @property
    def manifest(self):
        return {"strategy": self.strategy, "dialogue_mode": "llamaindex" if self.external_dialogue else "contextkv",
                "retrieval": self.serving.manifest,
                "source_window_profile": to_jsonable(self.directory.profile),
                "budget": to_jsonable(self.budget), "task_model": getattr(self.task_model, "manifest", None),
                "history_encoder": getattr(self.frozen_history, "manifest", None),
                "query_encoder_binding": self.query_encoder_identity,
                "support_model": getattr(self.support_backend, "manifest", None),
                "helper_engine_info": self.helper_engine_info,
                "selection_utility": self._utility_mode(),
                "evidence_selection_order": "utility_complete_needs_bundle_anchor_reranker_tokens_v1",
                "material_handoff": "whole_plan_candidates" if self.whole_plan_candidates else "single_material",
                "rerank_retained_sets": self.config.get("selection", {}).get("rerank_retained_sets", False),
                "coverage_mode": self.config.get("reading", {}).get("coverage_mode", "selected_sources")}

    def _utility_mode(self):
        return "support_coverage" if self.strategy in {"B2", "B3"} else self.config.get(
            "selection", {}).get("utility", "support_coverage")

    def _search_utility(self, support):
        mode = self._utility_mode()
        if mode == "support_coverage":
            return SupportCoverageUtility()
        if mode == "learned" and self.config.get("support_model", {}).get("mode") == "trained":
            return support
        raise ValueError("selection.utility must be support_coverage, or learned with a trained support model")

    def _rerank_sets(self, state, evidence_sets, scores, support):
        """Optional model assessment of at most three final budgeted sets."""
        if self.strategy in {"B2", "B3"} or not self.config.get("selection", {}).get("rerank_retained_sets", False):
            return evidence_sets, 0.0
        started = time.perf_counter()
        reranked, completeness, relevance = [], {}, {}
        for selected in evidence_sets[:3]:
            ids = {unit.unit_id for unit in selected.selected_units}
            available = tuple(score for score in (*scores.singles, *scores.bundles)
                              if set(score.unit_ids) <= ids)
            prediction = support.predict_set(state, selected.selected_units, available)
            best, relevance[selected.set_id] = best_need_supports(state, selected.selected_units, available)
            completeness[selected.set_id] = sum(s.disposition == "complete" and s.applicability*s.joint_support > 0
                                                for s in best)
            reranked.append(replace(selected, quality_prediction=prediction.quality,
                                    disposition=prediction.disposition))
        return tuple(sorted(reranked, key=lambda selected: (
            -selected.quality_prediction, -completeness[selected.set_id], -relevance[selected.set_id], selected.token_cost,
            selected.set_id))), (time.perf_counter()-started)*1000

    async def retrieve(self, snapshot, query, request_id):
        if self.external_dialogue:
            return await run_retrieval_worker(self._external_retrieve, snapshot, query, request_id)
        if self.strategy == "B0":
            return await run_retrieval_worker(self._baseline_retrieve, snapshot, query, request_id)
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id)

    def _baseline_retrieve(self, snapshot, query, request_id):
        with self._lock:
            before = {id(model): dict(model.usage) for _, model in self._usage_sources()
                      if hasattr(model, "usage")}
            selection = self.serving._retrieve(snapshot, query, request_id, time.monotonic())
            usage = self._auxiliary_usage(before, ())
            self.traces[snapshot.session_id, request_id] = {"auxiliary_usage": usage}
            return replace(selection, auxiliary_usage=usage)

    async def retrieve_supplied(self, snapshot, query, request_id, blocks, *, source_metadata=None,
                                coverage_scope="all_supplied_documents"):
        if self.external_dialogue:
            return await run_retrieval_worker(self._external_retrieve, snapshot, query, request_id,
                supplied_blocks=tuple(blocks), source_metadata=source_metadata, coverage_scope=coverage_scope)
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id,
            supplied_blocks=tuple(blocks), supplied_metadata=source_metadata,
            supplied_coverage_scope=coverage_scope)

    def _external_retrieve(self, snapshot, query, request_id, *, supplied_blocks=None,
                           source_metadata=None, coverage_scope="selected_sources"):
        """Rank the upstream condensed query; upstream compact owns synthesis."""
        started = time.perf_counter()
        with self._lock:
            before = {id(model): dict(model.usage) for _, model in self._usage_sources()
                      if hasattr(model, "usage")}
            top_k = self.config.get("selection", {}).get("max_passages") or 5
            if supplied_blocks is None:
                contexts = self.serving.candidate_search(query,
                    top_k=self.config.get("retrieval", {}).get("current_top_k", 50))[:top_k]
                priority = self.directory.corpus_index.retrieved_keys(contexts)
                blocks = self.directory.source_fetch(priority)
            else:
                if source_metadata is not None:
                    self.directory.corpus_index.source_metadata.update(source_metadata)
                self.directory.add_archive(supplied_blocks)
                if coverage_scope == "all_supplied_documents":
                    priority = tuple(block.key for block in supplied_blocks)
                else:
                    pool = self.directory.full_scope(tuple(block.key for block in supplied_blocks))
                    ranked = self._rank_supplied(None, snapshot, pool, query_override=query)
                    priority = tuple(dict.fromkeys(span.source_key
                        for unit in sorted(ranked.units, key=lambda unit: -unit.retrieval_score)
                        for span in unit.source_spans))[:top_k]
                blocks = self.directory.source_fetch(priority)
            usage = self._auxiliary_usage(before, ())
            self.traces[snapshot.session_id, request_id] = {
                "retrieval_query": query, "dialogue_mode": "llamaindex", "auxiliary_usage": usage,
                "retrieval_ms": (time.perf_counter()-started)*1000,
                "retrieval_priority": to_jsonable(priority), "coverage_scope": coverage_scope}
            return EvidenceSelection(blocks, priority, auxiliary_usage=usage)

    async def extend_for_plan(self, snapshot, plan, open_needs, claims):
        """Resume directed gap selection from a persisted plan and committed QA."""
        return await run_retrieval_worker(self._select_gap, snapshot, None, plan,
                                          open_needs, claims, None)

    def _usage_sources(self):
        sources = []
        if self.strategy not in {"B0", "Btoken"} and self.directory.task_interpreter is not None:
            sources.append(("task", self.directory.task_interpreter.model))
        if self.frozen_history is not None:
            sources.append(("history_document_encoder", self.frozen_history))
        encoder = getattr(self.serving.retriever, "encoder", None)
        if encoder is not None:
            sources.append(("retrieval_query_encoder", encoder))
        reranker = getattr(self.serving.retriever, "reranker", None)
        if reranker is not None:
            sources.append(("published_reranker", reranker))
        if self.support_backend is not None:
            sources.append(("support_utility", self.support_backend))
        return sources

    def _auxiliary_usage(self, before, auxiliary):
        grouped = {}
        for role, model in self._usage_sources():
            entry = grouped.setdefault(id(model), {"roles": [], "model": model})
            entry["roles"].append(role)
        models = []
        for identity, entry in grouped.items():
            usage = getattr(entry["model"], "usage", None)
            delta = ({name: value-before.get(identity, {}).get(name, 0) for name, value in usage.items()}
                     if usage is not None else None)
            compute = (delta.get("elapsed_ms") if delta is not None else None)
            if delta is not None and compute is None and "support_ms" in delta and "utility_ms" in delta:
                compute = delta["support_ms"]+delta["utility_ms"]
            models.append({"roles": entry["roles"], "usage": delta,
                "input_tokens": delta.get("input_tokens") if delta is not None else None,
                "output_tokens": delta.get("output_tokens") if delta is not None else None,
                "compute_ms": compute})
        totals = {name: sum(row[name] for row in models) if all(row[name] is not None for row in models) else None
                  for name in ("input_tokens", "output_tokens")}
        return {**totals, "compute_ms": sum(row["compute_ms"] for row in models)
                if all(row["compute_ms"] is not None for row in models) else None, "models": models}

    def _retrieve(self, snapshot, query, request_id, *, supplied_blocks=None, supplied_metadata=None,
                  supplied_coverage_scope="all_supplied_documents"):
        started = time.perf_counter()
        with self._lock:
            usage_baseline = {id(model): dict(model.usage) for _, model in self._usage_sources()
                              if hasattr(model, "usage")}
            usage_before = dict(getattr(self.support_backend, "usage", {}))
            if supplied_blocks is not None:
                if supplied_metadata is not None:
                    self.directory.corpus_index.source_metadata.update(supplied_metadata)
                self.directory.add_archive(supplied_blocks)
            memory = self.serving.retrieval_state(snapshot)
            if self.committed_readings is not None:
                memory = replace(memory, committed_readings=self.committed_readings(snapshot))
            if self.strategy in {"B0", "Btoken"}:
                refs = tuple(ref for row in snapshot.raw_history for ref in (
                    TextRef(row.turn_id, "user", 0, len(row.user_text)),
                    TextRef(row.turn_id, "assistant", 0, len(row.assistant_text))))
                current = len(snapshot.raw_history)+1
                state = TaskState(request_id, query, refs, needs=(Need("question",
                    (TextRef(current, "user", 0, len(query)),)),), current_turn_id=current, query=query)
                auxiliary = ()
            else:
                history = tuple(message for row in snapshot.raw_history for message in (
                    OnlineMessage("user", row.user_text), OnlineMessage("assistant", row.assistant_text)))
                turn = OnlineTurn(snapshot.session_id, request_id, len(snapshot.raw_history)+1,
                    self.serving.corpus_id, query, history, (), "P3")
                state = self.directory.task_interpreter.interpret(turn, memory)
                auxiliary = (("task", self.directory.task_interpreter.last_metrics["task_model_ms"]),)
            if supplied_blocks is not None:
                state = replace(state, source_scope=tuple(block.key for block in supplied_blocks))
            self.traces.setdefault((snapshot.session_id, request_id), {})["support_task_id"] = state.task_id
            interpreter = self.directory.task_interpreter
            reuse_enabled = self.config.get("reading", {}).get("fact_reuse", True)
            bindings = (interpreter.last_prediction.read_reuse
                if interpreter is not None and self.strategy not in {"B0", "Btoken"} and reuse_enabled else ())
            contracts = interpreter.last_read_contracts if bindings else ()
            preliminary = ContextPlan(state.task_id+":reuse", state, EvidenceSet("reuse", ()), budget=self.budget)
            seed = build_reuse_seed(preliminary, bindings, contracts, memory)
            covered = complete_reuse_ranges(seed)
            from controller.context_budget import _covers
            satisfied = set()
            for need in state.needs:
                results = tuple(result for result in seed.results if need.need_id in result.support)
                if not results:
                    continue
                if need.scope == "all_supplied_documents":
                    required = self.directory.full_scope(state.source_scope).units if state.source_scope else ()
                    ranges = {}
                    for span in covered.get(need.need_id, ()):
                        ranges.setdefault(span.source_key, []).append((span.start, span.end))
                    if not required or not all(_covers((span.start, span.end), ranges.get(span.source_key, ()))
                        for unit in required for span in unit.source_spans):
                        continue
                satisfied.add(need.need_id)
            active_state = replace(state, needs=tuple(need for need in state.needs if need.need_id not in satisfied))
            reading = self.config.get("reading", {}).get("enabled", True) and self.strategy in {"B6", "B7", "Lmap"}
            coverage = (supplied_coverage_scope if supplied_blocks is not None else
                        self.config.get("reading", {}).get("coverage_mode", "selected_sources"))
            if not reading:
                coverage = "selected_sources"
            elif (any(need.scope == "all_supplied_documents" for need in state.needs)
                    and (supplied_blocks is not None or state.source_scope)):
                # Exhaustive Needs scan their declared source boundary; the
                # operation alone never expands a task to the whole corpus.
                coverage = "all_supplied_documents"
            full_scope_read = reading and coverage == "all_supplied_documents"
            if supplied_blocks is not None:
                pool = self.directory.full_scope(state.source_scope)
            elif full_scope_read:
                document_ids = self.config.get("reading", {}).get("document_ids", ())
                keys = state.source_scope or tuple(block.key for block in self.directory.corpus_index.blocks
                    if not document_ids or block.key.document_id in document_ids)
                pool = self.directory.full_scope(keys)
            elif active_state.needs:
                pool = self.directory.retrieve_pool(active_state, memory)
            else:
                pool = EvidencePool((), (), "selected_sources")
            if full_scope_read:
                state = replace(state, source_scope=tuple(block.key for block in pool.source_blocks))
            selection = self.config.get("selection", {})
            policy = CandidatePolicy(max_total_model_tokens=self.budget.window_tokens,
                                     max_new_tokens=self.budget.output_tokens)
            if supplied_blocks is not None and not full_scope_read:
                pool = (self._rank_supplied(active_state, snapshot, pool) if active_state.needs else
                        EvidencePool((), (), "selected_sources"))

            def measure_contexts(state, groups):
                plans = []
                for units in groups:
                    source_keys = {span.source_key for unit in units for span in unit.source_spans}
                    blocks = tuple(block for block in pool.source_blocks if block.key in source_keys)
                    evidence_set = EvidenceSet("measurement", units, source_blocks=blocks)
                    plans.append(ContextPlan("measurement", state, evidence_set, budget=self.budget))
                return measure_plan_contexts(snapshot, query, plans, self.template, policy=policy)

            scores = SupportScores()
            support_model = None
            search_trace = ()
            support_ms = search_ms = 0.0
            if not active_state.needs and not full_scope_read:
                evidence_sets = (EvidenceSet(f"{state.task_id}:reuse", (), disposition="answer"),)
            elif full_scope_read:
                token_cost = len(self.template.tokenizer.encode(render_unit_union(pool.units), add_special_tokens=False))
                evidence_sets = (EvidenceSet(f"{state.task_id}:full_scope", pool.units,
                    token_cost=token_cost, disposition="partial", source_blocks=pool.source_blocks),)
            elif self.strategy in {"B0", "Btoken", "B1"}:
                search_started = time.perf_counter()
                evidence_sets = (self._relevance_set(state, pool,
                    lambda state, units: measure_contexts(state, (units,))[0]),)
                search_ms = (time.perf_counter()-search_started)*1000
            else:
                support_config = self.config.get("support_model", {})
                group_mode = support_config.get("group_mode", "single_pair_triple")
                profile = SupportProfile(support_config.get("single_candidates_per_need", 24),
                    0 if self.strategy == "B2" or group_mode == "direct" else support_config.get("bridge_candidates_per_need", 6),
                    0 if self.strategy == "B2" or group_mode in {"direct", "single_pair"}
                    else support_config.get("triple_seed_pairs", 2),
                    support_config.get("triple_successors_per_pair", 3))
                support_model = SupportModel(self.support_backend,
                    history_resolver=lambda ref: resolve_text(ref, state, snapshot), profile=profile)
                utility = self._search_utility(support_model)
                planner = BudgetedEvidencePlanner(self.template.tokenizer, utility,
                    profile=SearchProfile(selection.get("beam_width", 4), selection.get("max_actions", 12),
                        selection.get("retained_sets", 3)),
                    measure_contexts=measure_contexts)
                scores = support_model.score_support(active_state, pool)
                support_ms = scores.scoring_ms
                evidence_sets = planner.search_sets(active_state, pool, scores, self.budget)
                search_ms += evidence_sets[0].search_ms
                for _ in range(selection.get("initial_gap_search_rounds", 1)):
                    gaps = unresolved_needs(active_state, evidence_sets[0])
                    if not gaps:
                        break
                    if supplied_blocks is not None:
                        break
                    pool = self.directory.extend_open_needs(active_state, memory, pool, gaps)
                    scores = support_model.score_support(active_state, pool)
                    support_ms += scores.scoring_ms
                    evidence_sets = planner.search_sets(active_state, pool, scores, self.budget)
                    search_ms += evidence_sets[0].search_ms
                search_trace = planner.trace
                evidence_sets, rerank_ms = self._rerank_sets(active_state, evidence_sets, scores, support_model)
                search_ms += rerank_ms
            material_candidates = evidence_sets
            if not self.whole_plan_candidates:
                evidence_sets = evidence_sets[:1]
            auxiliary += (("retrieval", pool.retrieval_ms), ("support", support_ms), ("search", search_ms))
            quality_sets = {selected.set_id: selected for selected in evidence_sets}
            plans, initial_outcomes = [], []
            candidate_quality_ms = 0.0
            for selected in evidence_sets:
                current = None
                read_state = state if full_scope_read else active_state
                plan = (build_read_plan(read_state, selected, coverage, self.budget, template=self.template, snapshot=snapshot,
                            output_tokens_per_fact=None if self.strategy == "Lmap" else
                                self.config.get("reading", {}).get("output_tokens_per_fact", 64),
                            reading_mode=self.config.get("reading", {}).get("mode", "typed_reduce"),
                            separate_needs=self.config.get("reading", {}).get("need_isolation", False)
                                or (self.config.get("reading", {}).get("numeric_need_isolation", False)
                                    and any(need.operation in {"count", "sum"} for need in state.needs)),
                            numeric_identity=self.config.get("reading", {}).get("numeric_identity", "names"),
                            read_layout=self.config.get("reading", {}).get("layout", "task_first"),
                            answer_system_prompt=self.answer_system_prompt,
                            read_packing=self.config.get("reading", {}).get("packing", "source_windows"
                                if self.config.get("reading", {}).get("layout") == "source_first" else "mixed"))
                        if reading else ContextPlan(f"{request_id}:{selected.set_id}", state, selected,
                            coverage, budget=self.budget))
                plan = replace(plan, task_state=state)
                if self.strategy == "Lmap":
                    plan = replace(plan, steps=tuple(replace(step, operation="summarize", shared_numeric_scopes=(),
                        requires_reduction=True)
                        if step.kind == "READ" else step for step in plan.steps))
                plan = replace(plan, layouts=tuple(selection.get("layouts", ("existing", "need_order", "stable_evidence"))))
                if self.strategy in {"B5", "B6", "B7", "Lmap"}:
                    points = tuple(AnswerPoint(need.need_id, source_spans=tuple(span for support in selected.need_support
                        if support.need_id == need.need_id for span in support.source_spans)) for need in state.needs)
                    plan = replace(plan, answer_plan=AnswerPlan(points, selected.disposition))
                if bindings:
                    from controller.fact_reuse import apply_reuse_to_plan
                    current = build_reuse_seed(plan, bindings, contracts, memory)
                    current = apply_reuse_to_plan(plan, current, self.template, snapshot,
                        output_tokens_per_fact=None if self.strategy == "Lmap" else
                            self.config.get("reading", {}).get("output_tokens_per_fact", 64))
                    plan = current.plan
                    initial_outcomes.append(current)
                if support_model is not None and self._utility_mode() == "learned":
                    quality_started = time.perf_counter()
                    quality_material = quality_sets[selected.set_id]
                    history_tokens = sum(len(self.template.tokenizer.encode(message.content, add_special_tokens=False))
                        for message in working_history_messages(state, snapshot))
                    candidates = build_plan_candidates(snapshot, query, plan, self.template, policy=policy,
                        outcome=current,
                        layouts=tuple(selection.get("layouts", ("existing", "need_order", "stable_evidence"))))
                    qualities = []
                    for candidate in candidates:
                        if not candidate.feasible:
                            continue
                        required = candidate.required_blocks
                        old_spans = (() if snapshot.execution is None or candidate.action == "REBUILD" else tuple({
                            (span.message_index, span.start, span.end): span for span in snapshot.execution.evidence_spans
                            if span.key not in required}.values()))
                        old_tokens = sum(len(self.template.tokenizer.encode(snapshot.execution.messages[span.message_index].content[
                            span.start:span.end], add_special_tokens=False)) for span in old_spans)
                        selected_ids = {unit.unit_id for unit in quality_material.selected_units}
                        available = tuple(s for s in (*scores.singles, *scores.bundles) if set(s.unit_ids) <= selected_ids)
                        quality = support_model.predict_candidate(active_state, quality_material.selected_units,
                            available, action=candidate.action, layout=candidate.layout_id,
                            old_body_tokens=old_tokens, history_tokens=history_tokens,
                            input_tokens=len(candidate.input_ids), output_budget=self.budget.output_tokens)
                        if quality is not None:
                            qualities.append((candidate.action, candidate.layout_id, quality.quality))
                    plan = replace(plan, candidate_qualities=tuple(qualities))
                    candidate_quality_ms += (time.perf_counter()-quality_started)*1000
                plans.append(plan)
            auxiliary += (("candidate_quality", candidate_quality_ms),)
            plans = [replace(plan, auxiliary_ms=auxiliary) for plan in plans]
            trace = {"task_state": to_jsonable(state), "pool": to_jsonable(pool),
                "support_task_id": state.task_id,
                "reused_needs": sorted(satisfied),
                "read_reuse": to_jsonable(bindings),
                "support_scores": to_jsonable(scores), "search_trace": search_trace,
                "selection_policy_heads_used": support_model is not None,
                "support_schedule": to_jsonable(support_model.last_schedule) if support_model else (),
                "selection_utility": self._utility_mode(),
                "rerank_retained_sets": selection.get("rerank_retained_sets", False),
                "material_handoff": {"mode": "whole_plan_candidates" if self.whole_plan_candidates else "single_material",
                    "candidate_set_ids": [selected.set_id for selected in material_candidates],
                    "selected_set_ids": [selected.set_id for selected in evidence_sets]},
                "oversized_support_inputs": to_jsonable(getattr(self.support_backend, "oversized_inputs", ())),
                "support_model_usage": {name: value-usage_before.get(name, 0)
                    for name, value in getattr(self.support_backend, "usage", {}).items()},
                "task_metrics": (self.directory.task_interpreter.last_metrics
                    if self.strategy not in {"B0", "Btoken"} else None),
                "auxiliary_usage": self._auxiliary_usage(usage_baseline, auxiliary),
                "context_plans": to_jsonable(plans), "planning_ms": (time.perf_counter()-started)*1000}
            self.traces[snapshot.session_id, request_id] = trace

            async def extend(plan, open_needs, claims):
                return await self.extend_for_plan(snapshot, plan, open_needs, claims)

            selected_blocks = tuple({block.key: block for plan in plans for block in plan.selected_set.source_blocks}.values())
            context = QueryContext(state.query or query, "task_state", task_state=state)
            return EvidenceSelection(selected_blocks, tuple(block.key for block in selected_blocks), context,
                                     tuple(plans), open_need_provider=extend,
                                     auxiliary_usage=trace["auxiliary_usage"], initial_outcomes=tuple(initial_outcomes))

    def _relevance_set(self, state, pool, measure_context):
        by_id = {unit.unit_id: unit for unit in pool.units}
        chosen = ()
        for unit in sorted(pool.units, key=lambda item: (-item.retrieval_score, item.unit_id)):
            if self.strategy == "B0" and len({item.text for item in chosen}) >= 5:
                break
            proposed = close_dependencies((*[u.unit_id for u in chosen], unit.unit_id), by_id)
            size = len(self.template.tokenizer.encode(render_unit_union(proposed), add_special_tokens=False))
            passages = {pid for member in proposed for pid in member.original_passage_ids}
            if (size <= self.budget.body_tokens and measure_context(state, proposed)+self.budget.output_tokens <= self.budget.window_tokens
                    and (self.budget.max_passages is None or len(passages) <= self.budget.max_passages)):
                chosen = proposed
        keys = {span.source_key for unit in chosen for span in unit.source_spans}
        blocks = tuple(block for block in pool.source_blocks if block.key in keys)
        return EvidenceSet(f"{state.task_id}:relevance", chosen,
            token_cost=len(self.template.tokenizer.encode(render_unit_union(chosen), add_special_tokens=False)),
            source_blocks=blocks)

    def _rank_supplied(self, state, snapshot, pool, *, query_override=None):
        from adapters.hybrid_retrieval import DenseIndex, HybridRetriever, LocalBGE, search_variants
        from adapters.retrieval import BM25SRetriever
        from controller.schema import OnlineContext
        from adapters.dialogue_query import query_view
        if not pool.units:
            return pool
        started = time.perf_counter()
        if self.frozen_history is None:
            task = self.config.get("task_model", {})
            self.frozen_history = LocalBGE(task.get("base_path", "models/bge-m3"),
                device=task.get("device", "cuda:1"), dtype="bfloat16", max_length=8192,
                sdpa_backend=task.get("sdpa_backend", "auto"))
        query_encoder = self.serving.retriever.encoder if isinstance(self.serving.retriever, HybridRetriever) else self.frozen_history
        query = query_override if query_override is not None else state.query or state.raw_question
        if query_override is None and self.strategy in {"B0", "Btoken"}:
            messages = [message for row in snapshot.raw_history for message in (
                {"speaker": "user", "text": row.user_text}, {"speaker": "assistant", "text": row.assistant_text})]
            query, _ = query_view(state.raw_question, messages, query_encoder.tokenizer, limit=8192)
        contexts = tuple(OnlineContext(unit.unit_id, unit.text, title=unit.heading) for unit in pool.units)
        encoded = self.directory.encode_source_units(pool.units, self.frozen_history.encode)
        dense = DenseIndex(contexts, encoded, {"corpus_snapshot_id": snapshot.corpus_snapshot_id,
                                            "encoder": self.frozen_history.manifest})
        sparse = BM25SRetriever({context.document_id: {"text": context.text, "title": context.title}
            for context in contexts}, corpus_snapshot_id=snapshot.corpus_snapshot_id)
        retriever = self.serving.retriever
        reranker = getattr(retriever, "reranker", None)
        stages = search_variants(query, sparse, dense, query_encoder, reranker,
            candidate_k=getattr(retriever, "candidate_k", 50),
            rerank_k=getattr(retriever, "rerank_k", 50), rrf_k=getattr(retriever, "rrf_k", 60))
        ranked = stages["hybrid_rerank" if reranker is not None else "hybrid"]["contexts"]
        scores = {context.document_id: float(len(ranked)-index) for index, context in enumerate(ranked)}
        return replace(pool, units=tuple(replace(unit, retrieval_score=scores.get(unit.unit_id, 0.0)) for unit in pool.units),
                       retrieval_ms=pool.retrieval_ms+(time.perf_counter()-started)*1000)

    def _select_gap(self, snapshot, memory, plan, open_needs, claims, parent_trace):
        started = time.perf_counter()
        with self._lock:
            usage_before = {id(model): dict(model.usage) for _, model in self._usage_sources()
                            if hasattr(model, "usage")}
            source_started = time.perf_counter()
            if memory is None:
                restore_source_metadata(self.directory, plan.selected_set.selected_units)
                self.directory.add_archive(plan.selected_set.source_blocks)
                memory = self.serving.retrieval_state(snapshot)
            if parent_trace is None:
                parent_trace = self.traces.setdefault((snapshot.session_id, plan.task_state.task_id),
                    {"task_state": to_jsonable(plan.task_state), "resumed_from_plan": plan.plan_id})
            state = plan.task_state
            gaps = {need.need_id for need in open_needs}
            state = replace(state, needs=tuple(need for need in state.needs if need.need_id in gaps))
            previous = plan.selected_set.selected_units
            ranges = {}
            for unit in previous:
                for span in unit.source_spans:
                    ranges.setdefault(span.source_key, []).append((span.start, span.end))

            def already_read(unit):
                for span in unit.source_spans:
                    cursor = span.start
                    for left, right in sorted(ranges.get(span.source_key, ())):
                        if left > cursor:
                            break
                        cursor = max(cursor, right)
                        if cursor >= span.end:
                            break
                    if cursor < span.end:
                        return False
                return True

            fresh_pool = (self.directory.full_scope(state.source_scope) if state.source_scope
                          else self.directory.retrieve_gap_pool(state, memory, open_needs))
            fresh_units = tuple(unit for unit in fresh_pool.units if not already_read(unit))
            phases = (("source_directory", (time.perf_counter()-source_started)*1000),) if state.source_scope else (
                ("retrieval", fresh_pool.retrieval_ms),)
            if not fresh_units:
                usage = self._auxiliary_usage(usage_before, phases)
                return replace(fresh_pool, units=(), source_blocks=(), auxiliary_ms=phases,
                               auxiliary_usage=usage)
            if plan.coverage_scope == "all_supplied_documents":
                by_id = {unit.unit_id: unit for unit in (*previous, *fresh_pool.units)}
                material = close_dependencies(tuple(unit.unit_id for unit in fresh_units), by_id)
                source_keys = {span.source_key for unit in material for span in unit.source_spans}
                blocks = tuple(block for block in fresh_pool.source_blocks if block.key in source_keys)
                phases = (("source_directory", (time.perf_counter()-source_started)*1000),)
                return replace(fresh_pool, units=material, source_blocks=blocks, auxiliary_ms=phases,
                               auxiliary_usage=self._auxiliary_usage(usage_before, phases))
            old_ids = {unit.unit_id for unit in previous}
            recalled = tuple(replace(unit, need_ids=()) if already_read(unit) and unit.unit_id not in old_ids
                             else unit for unit in fresh_pool.units)
            units = merge_retrieved_units(recalled, previous)
            blocks = tuple({block.key: block for block in
                (*plan.selected_set.source_blocks, *fresh_pool.source_blocks)}.values())
            context_pool = replace(fresh_pool, units=units, source_blocks=blocks)
            query = "\n\n".join(self.directory.open_need_query(state, need) for need in open_needs)
            context_pool = self._rank_supplied(state, snapshot, context_pool, query_override=query)
            by_id = {unit.unit_id: unit for unit in context_pool.units}

            def read_material(selected):
                fresh = tuple(unit.unit_id for unit in selected if not already_read(unit))
                return close_dependencies(fresh, by_id)

            def new_bodies(groups):
                return tuple(len(ids) for ids in self.template.tokenizer(
                    [render_unit_union(read_material(selected)) for selected in groups], add_special_tokens=False,
                    return_attention_mask=False, return_token_type_ids=False)["input_ids"])

            config = self.config.get("support_model", {})
            direct = self.strategy == "B2" or config.get("group_mode") == "direct"
            profile = SupportProfile(config.get("single_candidates_per_need", 24),
                0 if direct else config.get("bridge_candidates_per_need", 6),
                0 if direct or config.get("group_mode") == "single_pair" else config.get("triple_seed_pairs", 2),
                config.get("triple_successors_per_pair", 3))
            support = SupportModel(self.support_backend, history_resolver=lambda ref: resolve_text(ref, state, snapshot),
                                   profile=profile)
            scores = support.score_support(state, context_pool, open_needs=tuple(open_needs))
            used_passages = _passages(previous)
            budget = replace(plan.budget, max_passages=None if plan.budget.max_passages is None
                             else plan.budget.max_passages-len(used_passages))
            selection = self.config.get("selection", {})
            planner = BudgetedEvidencePlanner(self.template.tokenizer,
                self._search_utility(support),
                profile=SearchProfile(selection.get("beam_width", 4), selection.get("max_actions", 12),
                    selection.get("retained_sets", 3)), measure_bodies=new_bodies, passage_exclusions=used_passages)
            evidence_sets = planner.search_sets(state, context_pool, scores, budget)
            evidence_sets, rerank_ms = self._rerank_sets(state, evidence_sets, scores, support)
            chosen = next((selected for selected in evidence_sets if any(not already_read(unit)
                           for unit in selected.selected_units)), None)
            selected_units = () if chosen is None else read_material(chosen.selected_units)
            source_keys = {span.source_key for unit in selected_units for span in unit.source_spans}
            selected_blocks = tuple(block for block in blocks if block.key in source_keys)
            phases = (("retrieval", context_pool.retrieval_ms), ("support", scores.scoring_ms),
                      ("search", evidence_sets[0].search_ms+rerank_ms))
            usage = self._auxiliary_usage(usage_before, phases)
            gap_trace = {"open_needs": to_jsonable(open_needs), "candidate_pool": to_jsonable(context_pool),
                "support_scores": to_jsonable(scores), "selected_set": to_jsonable(chosen),
                "support_schedule": to_jsonable(support.last_schedule),
                "selection_utility": self._utility_mode(),
                "rerank_retained_sets": selection.get("rerank_retained_sets", False),
                "used_passages": sorted(used_passages), "remaining_passages": budget.max_passages,
                "auxiliary_usage": usage, "elapsed_ms": (time.perf_counter()-started)*1000}
            parent_trace.setdefault("gap_rounds", []).append(gap_trace)
            return replace(fresh_pool, units=selected_units, source_blocks=selected_blocks,
                retrieval_ms=context_pool.retrieval_ms,
                trace=(*fresh_pool.trace, gap_trace), auxiliary_ms=phases, auxiliary_usage=usage,
                selected_set=chosen)

    def close(self):
        query_encoder = self.serving.retriever.encoder
        self.serving.close()
        if self.task_model is not None and self.task_model is not query_encoder and self.teacher_client is None:
            self.task_model.close()
        if self.frozen_history is not None:
            self.frozen_history.close()
        if self.teacher_client is not None:
            self.teacher_client.close()
        if self.support_backend is not None:
            self.support_backend.close()


def create_study_retriever(config, template, *, retriever=None, task_interpreter=None,
                           support_backend=None, frozen_history=None):
    retrieval = config["retrieval"]
    strategy = config.get("study", {}).get("strategy", "B7")
    external = config.get("execution", {}).get("dialogue_mode") == "llamaindex"
    if strategy not in STUDY_STRATEGIES:
        raise ValueError(f"unknown study strategy: {strategy}")
    retriever = retriever or create_retriever("P3", config_path=retrieval["config_path"])
    corpus = CorpusIndex.from_jsonl(retrieval["corpus_path"], corpus_id=retrieval["corpus_id"],
                                   snapshot_id=retriever.corpus_snapshot_id)
    serving = ServingRetriever(retriever, corpus_id=retrieval["corpus_id"], corpus_index=corpus)
    task_config, support_config = config.get("task_model", {}), config.get("support_model", {})
    task_model = teacher_client = query_encoder_identity = None
    helper = helper_info = None
    teacher_task = not external and strategy not in {"B0", "Btoken"} and task_interpreter is None and task_config.get("mode", "trained") == "teacher"
    teacher_support = not external and strategy not in {"B0", "Btoken", "B1"} and support_backend is None and support_config.get("mode", "trained") == "teacher"
    if teacher_task or teacher_support:
        import httpx
        from controller.config import load_config
        generation = config.get("generation", {})
        helper = load_config(generation.get("helper_run_config", generation.get("run_config", "configs/qwen3_32b.yaml")))
        endpoint = generation.get("teacher_url", "http://127.0.0.1:30001").rstrip("/").removesuffix("/v1")
        with httpx.Client(timeout=30) as client:
            response = client.post(endpoint+"/context/engine/info", json={})
            response.raise_for_status()
            helper_info = response.json()
        fingerprint = helper_info["fingerprint"]
        if fingerprint["model_revision"] != helper.model.revision:
            raise ValueError("actual helper checkpoint differs from generation.helper_run_config")
        if any(fingerprint[name] != getattr(template.fingerprint, name) for name in ("tokenizer_hash", "template_hash")):
            raise ValueError("helper tokenizer/template differs from the supplied token-counting template")
    if not external and strategy not in {"B0", "Btoken"} and task_interpreter is None:
        from adapters.hybrid_retrieval import LocalBGE
        frozen_history = frozen_history or LocalBGE(task_config.get("base_path", "models/bge-m3"),
            device=task_config.get("device", "cuda:1"), dtype="bfloat16", max_length=8192,
            sdpa_backend=task_config.get("sdpa_backend", "auto"))
        if task_config.get("mode", "trained") == "teacher":
            from adapters.query_selector import OpenAICompletionClient, SelectorConfig
            teacher_client = OpenAICompletionClient(SelectorConfig(
                config.get("generation", {}).get("teacher_url", "http://127.0.0.1:30001"),
                helper.model.name, helper.model.revision, max_input_tokens=24576,
                max_new_tokens=task_config.get("max_output_tokens", 4096)),
                tokenizer=template.tokenizer)
            task_model = TeacherTaskModel(teacher_client, record_path=task_config.get("teacher_record_path",
                str(Path(config.get("study", {}).get("output_dir", "runs/study"))/"task_teacher_calls.jsonl")))
        else:
            from training.task_model import TaskModel
            task_model = TaskModel(task_config["asset"], base_path=task_config.get("base_path"),
                                   device=task_config.get("device", "cuda:1"))
            from training.preference_rollout import query_encoder_binding
            query_encoder_identity = query_encoder_binding({"task_asset": task_config["asset"]})
            previous_encoder = retriever.encoder
            retriever.encoder = task_model
            previous_encoder.close()
        task_interpreter = TaskInterpreter(task_model, HistoryIndex(frozen_history.tokenizer, frozen_history.encode,
            unit_tokens=task_config.get("history_unit_tokens", 384), candidates=task_config.get("history_candidates", 16),
            view_tokens=task_config.get("history_view_tokens", 8192)))
    if not external and retrieval.get("task_query_asset"):
        from training.task_model import TaskModel
        from training.preference_rollout import query_encoder_binding
        query_encoder_identity = query_encoder_binding({"task_asset": retrieval["task_query_asset"]})
        previous_encoder = retriever.encoder
        retriever.encoder = TaskModel(retrieval["task_query_asset"], base_path=task_config.get("base_path"),
                                      device=task_config.get("device", "cuda:1"))
        if previous_encoder is not task_model:
            previous_encoder.close()
    layout = config.get("reading", {}).get("layout", "task_first")
    protocols = {"task_first": "source_rows_pr3", "source_first": "canonical"}
    if layout not in protocols:
        raise ValueError("unknown reading layout for source window protocol")
    profile = PoolProfile(retrieval.get("current_top_k", 50), retrieval.get("per_need_top_k", 20),
        retrieval.get("archive_top_k", 12), retrieval.get("primary_pool_size", 96),
        window_protocol=protocols[layout])
    if task_interpreter is not None:
        directory = serving.configure_study(task_interpreter, template.tokenizer, profile=profile,
                                           reference_date=retrieval.get("reference_date"))
    else:
        directory = SourceDirectory(corpus, serving.candidate_search, template.tokenizer, profile=profile)
        serving.source_directory = directory
    if not external and strategy not in {"B0", "Btoken", "B1"} and support_backend is None:
        if support_config.get("mode", "trained") == "teacher":
            from adapters.teacher_support import TeacherSupportBackend
            support_backend = TeacherSupportBackend(config.get("generation", {}).get("teacher_url",
                "http://127.0.0.1:30001"), template, batch_size=support_config.get("batch_size", 8),
                record_path=support_config.get("teacher_record_path"))
            support_backend.manifest.update(helper_model_revision=helper.model.revision, helper_engine_info=helper_info)
        else:
            from training.support_model import TrainedSupportBackend
            asset = support_config.get("preference_asset", support_config["asset"]) if strategy in {"B4", "B5", "B6", "B7", "Lmap"} else support_config["asset"]
            support_backend = TrainedSupportBackend.load(asset, device=support_config.get("device", "cuda:2"),
                                                         batch_size=support_config.get("batch_size", 8))
    return StudyEvidenceProvider(config, template, serving, directory, support_backend=support_backend,
        frozen_history=frozen_history, task_model=task_model, teacher_client=teacher_client, helper_engine_info=helper_info,
        query_encoder_identity=query_encoder_identity)
