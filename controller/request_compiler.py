"""Compile immutable native requests before action selection and cache quotation."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

from .admission import quoted_native_work
from .candidates import build_candidates, build_plan_candidates
from .context_schema import SourceSpan
from .engine_client import EngineInfo
from .prefix_view import actual_source_geometry, source_fragment_id
from .reading import output_schema, step_generation_kind, step_messages
from .schema import CacheQuoteBatch, CompletionReceipt, ConfigFingerprint, EvidenceSpan, PromptCandidate
from .serialization import decode, dumps
from .source_prompt import SOURCE_ENCODING, source_envelope, source_evidence_spans, stable_source_payload


_ANSWER_BODY_ENCODING = "answer-evidence-body:v1"


def read_task_hash(task):
    return sha256(dumps(task).encode()).hexdigest()


def unique_answer_inputs(candidates):
    unique, alternatives = {}, []
    for candidate in candidates:
        if not candidate.feasible:
            alternatives.append(candidate)
            continue
        key = (candidate.context_plan_id, candidate.evidence_set_id, candidate.working_history_id,
               candidate.action, candidate.input_hash, candidate.output_budget, candidate.evidence_spans,
               candidate.reference_map, candidate.required_blocks, candidate.quality_prediction)
        unique.setdefault(key, candidate)
    return (*unique.values(), *alternatives)


class RequestCompiler:
    """One compiler for C-on/off; the optional index supplies permutations only."""
    def __init__(self, template, index=None):
        self.template, self.index = template, index

    def _scope(self, cache_scope: EngineInfo | CacheQuoteBatch):
        if not isinstance(cache_scope, (EngineInfo, CacheQuoteBatch)):
            raise TypeError("cache ordering requires current EngineInfo or CacheQuoteBatch")
        if cache_scope.fingerprint is not None and cache_scope.fingerprint != self.template.fingerprint:
            raise ValueError("cache scope and compiler use different fingerprints")
        return dict(fingerprint=self.template.fingerprint,
                    cache_config_hash=cache_scope.cache_config_hash,
                    engine_restart_id=cache_scope.engine_restart_id)

    def _orders(self, source_ids, cache_scope):
        return ((tuple(source_ids),) if self.index is None else
                self.index.propose_orders(source_ids, **self._scope(cache_scope)))

    def bind_geometry(self, candidate, rendered):
        geometry = actual_source_geometry(self.template, candidate, rendered)
        return replace(candidate, evidence_body_tokens=geometry["body_tokens"],
                       source_body_token_ranges=geometry["body_token_ranges"],
                       source_prefix_tokens=geometry["source_prefix_tokens"])

    @staticmethod
    def _read_id(unit, layout):
        if layout == "source_first":
            return source_fragment_id(unit.source_spans, unit.text, SOURCE_ENCODING)
        return source_fragment_id(unit.source_spans, json.dumps(unit.text, ensure_ascii=False)[1:-1],
                                  "read-json:"+layout)

    def compile_read(self, plan, task, units, claims, snapshot, *, outputs, cache_scope=None):
        """Compile READ/REDUCE inputs; selection must retain read_unit_order verbatim."""
        units, claims, outputs = tuple(units), tuple(claims), tuple(outputs)
        identities = tuple(self._read_id(unit, plan.read_layout) for unit in units)
        # A unit is the atomic READ body; identical original windows need no second
        # physical permutation handle. Preserve their complete alias/provenance list.
        original_ids = tuple(dict.fromkeys(identities))
        orders = self._orders(original_ids, cache_scope) if units else ((),)
        blocks = {block.key: block for block in (*snapshot.archive, *plan.selected_set.source_blocks)}
        candidates = []
        for order_index, order in enumerate(orders):
            rank = {identity: index for index, identity in enumerate(order)}
            ordered = (units if order_index == 0 else tuple(sorted(units,
                key=lambda unit: rank[self._read_id(unit, plan.read_layout)])))
            unit_order = tuple(unit.unit_id for unit in ordered)
            current = replace(task, input_unit_ids=unit_order,
                output_schema=output_schema(task, ordered, claims, reading_mode=plan.reading_mode,
                                            numeric_identity=plan.numeric_identity))
            source_prompt, canonical_spans = None, ()
            if task.kind == "READ" and plan.read_layout == "source_first":
                sources, aliases = stable_source_payload(ordered)
                content, bodies = source_envelope(sources)
                source_prompt = content, aliases
                canonical_spans = source_evidence_spans(ordered, bodies, blocks, 1)
            for output in outputs:
                actual = replace(current, output_budget=output)
                messages = step_messages(plan, actual, ordered, claims, snapshot, source_prompt=source_prompt)
                rendered = self.template.render(messages, with_offsets=bool(ordered))
                spans, cursor = list(canonical_spans), 0
                if ordered and source_prompt is None:
                    body_message = messages[1].content
                    for unit in ordered:
                        literal = json.dumps(unit.text, ensure_ascii=False)
                        marker = '"text":'+literal
                        found = body_message.index(marker, cursor)
                        first = found+len('"text":')+1
                        last = first+len(literal)-2
                        for source in unit.source_spans:
                            block = blocks.get(source.source_key)
                            spans.append(EvidenceSpan(source.source_key, 1, first, last,
                                block.source_uri if block is not None else "",
                                block.provenance if block is not None else (), source.span_id,
                                source_start=source.start, source_end=source.end))
                        cursor = found+len(marker)
                candidate_id = ([plan.plan_id, task.step_id, output] if order_index == 0 else
                                [plan.plan_id, task.step_id, output, "contextpilot", unit_order])
                keys = frozenset(source.source_key for unit in ordered for source in unit.source_spans)
                candidate = PromptCandidate(dumps(candidate_id), "REBUILD", snapshot.version,
                    messages, rendered.input_ids, rendered.input_hash, keys, keys,
                    feasible=(len(rendered.input_ids) <= task.input_budget
                              and len(rendered.input_ids)+output <= plan.budget.window_tokens),
                    evidence_spans=tuple(spans), fingerprint=self.template.fingerprint,
                    corpus_snapshot_id=snapshot.corpus_snapshot_id, context_plan_id=plan.plan_id,
                    evidence_set_id=plan.selected_set.set_id, output_budget=output,
                    generation_kind=step_generation_kind(task.kind, task.operation, plan.reading_mode),
                    read_unit_order=unit_order, source_ids=tuple(order), read_task_hash=read_task_hash(actual),
                    layout_id="existing" if order_index == 0 else "existing:contextpilot")
                candidates.append(self.bind_geometry(candidate, rendered))
        return tuple(candidates)

    def restore_read(self, record, snapshot, plan):
        """Reattach the compiled source geometry to the ledger's exact input."""
        bound = record.metrics.get("compiled_input")
        if bound is None:
            return None
        fingerprint = decode(ConfigFingerprint, bound["fingerprint"])
        if fingerprint != self.template.fingerprint:
            raise ValueError("saved reading input belongs to a different model template")
        spans = decode(tuple[EvidenceSpan, ...], bound["evidence_spans"])
        keys = frozenset(span.key for span in spans)
        return PromptCandidate(bound["candidate_id"], "REBUILD", snapshot.version,
            record.input_messages, record.input_ids, bound["input_hash"], keys, keys,
            evidence_spans=spans, fingerprint=fingerprint, corpus_snapshot_id=snapshot.corpus_snapshot_id,
            context_plan_id=plan.plan_id, evidence_set_id=plan.selected_set.set_id,
            layout_id=bound["layout_id"], output_budget=record.task.output_budget,
            generation_kind=record.metrics["generation_kind"], read_unit_order=record.task.input_unit_ids,
            read_task_hash=read_task_hash(record.task), source_ids=bound["source_ids"],
            source_body_token_ranges=decode(tuple[tuple[int, int], ...], bound["source_body_token_ranges"]),
            source_prefix_tokens=bound["source_prefix_tokens"])

    @staticmethod
    def _answer_unit_ids(unit, blocks, source_first):
        if source_first:
            return (source_fragment_id(unit.source_spans, unit.text, SOURCE_ENCODING),)
        ids = []
        for span in unit.source_spans:
            block = blocks[span.source_key]
            base = block.span[0] if block.span else 0
            body = block.text[span.start-base:span.end-base]
            ids.append(source_fragment_id((span,), body, _ANSWER_BODY_ENCODING))
        return tuple(ids)

    def _answer_orders(self, units, blocks, cache_scope, *, source_first):
        identities = tuple(dict.fromkeys(identity for unit in units
            for identity in self._answer_unit_ids(unit, blocks, source_first)))
        orders = self._orders(identities, cache_scope)
        proposed = [tuple(units)]
        if len(orders) > 1:
            rank = {identity: index for index, identity in enumerate(orders[1])}
            ordered = tuple(sorted(units, key=lambda unit: min(
                rank[identity] for identity in self._answer_unit_ids(unit, blocks, source_first))))
            if ordered != tuple(units):
                proposed.append(ordered)
        return tuple(proposed)

    def _answer_block_orders(self, blocks, cache_scope):
        by_id = {}
        for block in blocks:
            start = block.span[0] if block.span else 0
            source = SourceSpan("", block.key, start, start+len(block.text))
            by_id[source_fragment_id((source,), block.text, _ANSWER_BODY_ENCODING)] = block
        return tuple(tuple(by_id[identity] for identity in order)
                     for order in self._orders(tuple(by_id), cache_scope))

    def compile_answers(self, snapshot, query, plan=None, *, policy=None, outcome=None,
                        layouts=None, actions=("REBUILD", "APPEND"), cache_scope=None,
                        representations=("existing",), **direct_options):
        """Delegate semantic/history/conclusion compilation to the existing builder.

        Source permutations occur after the builders choose original material.
        """
        blocks = {block.key: block for block in snapshot.archive}
        if plan is None:
            direct_options["retrieved"] = tuple(direct_options["retrieved"])
            blocks.update((block.key, block) for block in direct_options["retrieved"])
            candidates = build_candidates(snapshot, query, policy=policy, template=self.template,
                source_orderer=(None if self.index is None else
                                lambda blocks: self._answer_block_orders(blocks, cache_scope)),
                bind_geometry=self.bind_geometry, **direct_options)
        else:
            blocks.update((block.key, block) for block in plan.selected_set.source_blocks)
            candidates = tuple(candidate for representation in representations
                for candidate in build_plan_candidates(snapshot, query, plan, self.template, policy=policy,
                    outcome=outcome, layouts=layouts, actions=actions, bind_geometry=self.bind_geometry,
                    representation=representation,
                    source_orderer=lambda units, blocks: self._answer_orders(units, blocks, cache_scope,
                        source_first=plan.read_layout == "source_first")))
        result = []
        for candidate in candidates:
            groups = {}
            for index, span in enumerate(candidate.evidence_spans):
                body = candidate.messages[span.message_index].content[span.start:span.end]
                if span.source_start is None:
                    block = blocks[span.key]
                    start = block.span[0] if block.span else 0
                    end = start+len(block.text)
                else:
                    start, end = span.source_start, span.source_end
                source = SourceSpan("", span.key, start, end)
                key = span.source_declaration if span.source_declaration is not None else index
                encoding = SOURCE_ENCODING if span.source_declaration is not None else _ANSWER_BODY_ENCODING
                groups.setdefault(key, ([], body, encoding))[0].append(source)
            ids = (source_fragment_id(sources, body, encoding) for sources, body, encoding in groups.values())
            result.append(replace(candidate, source_ids=tuple(dict.fromkeys(ids))))
        return tuple(result)

    @staticmethod
    def choose_variant(candidates, quote_batch: CacheQuoteBatch):
        """Choose a compiled permutation by declared native token-work proxy.

        Minimize newly prefilled suffix tokens, then estimated host-load tokens,
        then input length and original candidate order. This is not measured or
        calibrated latency. The caller retains the unchanged quote batch for B.
        """
        if not isinstance(quote_batch, CacheQuoteBatch):
            raise TypeError("variant selection requires the actual CacheQuoteBatch")
        quotes = {quote.candidate_id: quote for quote in quote_batch.quotes}
        choices = []
        for index, candidate in enumerate(candidates):
            if not candidate.feasible:
                continue
            quote = quotes[candidate.candidate_id]
            if (quote.input_hash != candidate.input_hash or quote.input_len != len(candidate.input_ids)
                    or candidate.fingerprint != quote_batch.fingerprint):
                raise ValueError("variant quote differs from its compiled input")
            work = (*quoted_native_work(quote), len(candidate.input_ids), index)
            choices.append((work, candidate))
        if not choices:
            raise ValueError("no feasible compiled source permutation")
        return min(choices, key=lambda row: row[0])[1]

    def observe_executed(self, candidate, receipt: CompletionReceipt, *, cache_config_hash, engine_restart_id):
        """Describe actual causal/body geometry and optionally register its source order.

        No KV retain, lease, expiry, quality gate, or alternative compilation occurs.
        """
        if not isinstance(receipt, CompletionReceipt):
            raise TypeError("source observation requires CompletionReceipt")
        if (candidate.fingerprint != self.template.fingerprint
                or receipt.engine_restart_id != engine_restart_id
                or receipt.input_hash != candidate.input_hash or not receipt.generation_complete):
            raise ValueError("executed source observation differs from the compiled request/scope")
        geometry = actual_source_geometry(self.template, candidate)
        geometry.pop("causal_prefix_input_ids")
        request_id = None
        if self.index is not None and candidate.source_ids:
            request_id = self.index.observe_executed(candidate.source_ids,
                fingerprint=candidate.fingerprint, cache_config_hash=cache_config_hash,
                receipt=receipt, input_ids=candidate.input_ids)
        return {"candidate_id": candidate.candidate_id, "input_hash": candidate.input_hash,
                "source_ids": candidate.source_ids, "evidence_spans": candidate.evidence_spans,
                "index_request_id": request_id, **geometry}
