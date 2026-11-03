"""Pure APPEND/REBUILD construction over one immutable committed snapshot."""
from __future__ import annotations

import time
import re
import json
from dataclasses import replace
from typing import Iterable

from .evidence import (EvidenceError, EvidencePolicy, allows_version_comparison, assign_references,
                       evidence_index, resolve_required_evidence_with_reasons, version_conflicts)
from .schema import EvidenceBlock, EvidenceKey, Message, PromptCandidate, QueryContext, SessionSnapshot
from .templates import (SYSTEM_PROMPT, ByteTemplate, Template, current_user_message,
                        current_source_guide, longest_common_prefix, render_evidence_relations, verify_rendered_evidence)


CandidatePolicy = EvidencePolicy

PLAN_SYSTEM_PROMPT = (
    "Answer the current question using the active task conditions, supplied original sources, "
    "and Reading conclusions produced by this request's READ/REDUCE steps. Follow the Answer plan "
    "and cover its requested information points. Reading conclusions retain a source-linked "
    "derivation in the execution ledger; count and sum values are computed from their leaf facts. "
    "Use those conclusions together with the key original excerpts in this window. State the "
    "operation when presenting a derived result. "
    "For counts and sums, report the Computed results with their units and covered source scope. "
    "Key original excerpts are representative samples of the completed reading scope. "
    "Use working history to resolve the user's referents and preserve the active object, "
    "platform, time and version. Cite allowed source "
    "IDs inline in square brackets exactly as [E_...]. State the supported answer and any "
    "remaining information gap concisely. State the requested results directly; quote or enumerate "
    "example source rows only when the question asks for them."
)


def build_candidates(snapshot: SessionSnapshot, query: str,
                     retrieved: Iterable[EvidenceBlock], policy: EvidencePolicy | None = None,
                     template: Template | None = None, *,
                     retrieval_priority: tuple[EvidenceKey, ...] | None = None,
                     query_context: QueryContext | None = None,
                     measure_evidence_redundancy: bool = False
                     ) -> tuple[PromptCandidate, ...]:
    """No archive/seen/epoch/template state is changed by candidate generation."""
    started = time.perf_counter()
    policy, template = policy or EvidencePolicy(), template or ByteTemplate()
    system_message = Message("system", SYSTEM_PROMPT if policy.system_prompt is None else policy.system_prompt)
    retrieved = tuple(retrieved)
    retrieval_priority = (tuple(block.key for block in retrieved) if retrieval_priority is None
                          else tuple(retrieval_priority))
    if not set(retrieval_priority) <= {block.key for block in retrieved}:
        raise EvidenceError("retrieval priority must reference supplied retrieval evidence")
    archive = evidence_index(retrieved, committed=snapshot.archive)
    intent_query = query if query_context is None else query_context.query
    # Retain explicit source references even if a resolver expands their names.
    evidence_query = intent_query if query_context is None else (
        intent_query + " " + " ".join(re.findall(r"\bE_[a-zA-Z0-9]+\b", query)))
    revisit_turns = (() if query_context is None else
                     query_context.history_turns if query_context.relation == "revisit" else ())
    required, resolution = resolve_required_evidence_with_reasons(
        evidence_query, (block.key for block in retrieved), archive, snapshot, policy,
        revisit_turns=revisit_turns)
    if policy.layout == "contextpilot":
        from adapters.contextpilot import order_evidence
        required = order_evidence(required, snapshot, retrieval_priority)
    required_keys = frozenset(b.key for b in required)
    existing_refs = snapshot.reference_map
    if snapshot.execution:
        existing_refs = (*existing_refs, *snapshot.execution.reference_map)
    references = assign_references(archive, existing_refs, prefix_length=policy.reference_prefix_length)
    evidence_relations = render_evidence_relations(required, references)
    source_guide = current_source_guide(intent_query, required, references,
                                      retrieval_priority=retrieval_priority)
    groups: dict[tuple[str, str, str], list[str]] = {}
    for key, reason in resolution:
        groups.setdefault((key.corpus_id, key.document_id, key.passage_id), []).append(reason)
    has_revisit_conflict = any("current_retrieval" in origins and
        any(origin.startswith("selected_revisit:") for origin in origins)
        for origins in groups.values())
    origin_guide = ""
    if has_revisit_conflict:
        by_key = {key: ref for ref, key in references}
        origins = [{"id": by_key[key], "origin": reason} for key, reason in resolution]
        origin_guide = ("Evidence origin metadata (retrieval does not by itself establish applicability): "
                        + json.dumps(origins, ensure_ascii=False))
    table_annotations = ""
    if policy.table_relations:
        from adapters.table_relations import extract_table_relations, render_table_relations
        by_key = {key: reference for reference, key in references}
        options = {} if policy.table_structure_mode == "legacy" else {"mode": policy.table_structure_mode}
        tables = tuple((by_key[block.key], extract_table_relations(block.text, **options)) for block in required)
        if policy.table_annotation_style == "focus_refs":
            from adapters.table_focus import focused_annotation
            table_annotations, _ = focused_annotation(query, tables)
        else:
            table_annotations = "\n".join(annotation for ref, extracted in tables
                if (annotation := render_table_relations(ref, extracted)))
    previous = (() if snapshot.execution is None else
                snapshot.execution.last_input_ids + snapshot.execution.last_output_ids)
    prompts = []
    body_lengths = {}

    def candidate(action, messages, spans, visible, reason, rendered):
        verify_rendered_evidence(messages, spans, rendered.text)
        total_body_tokens = nonrequired_body_tokens = None
        if measure_evidence_redundancy:
            # Count each complete body independently with the generation tokenizer.
            # This is a text redundancy proxy, not a prompt-token or KV subtraction.
            # Alias identities share one physical span. A body used by any
            # required source is required, even when its original owner is historical.
            total_body_tokens = nonrequired_body_tokens = 0
            physical = {}
            for span in spans:
                physical.setdefault((span.message_index, span.start, span.end), set()).add(span.key)
            for (index, start, end), keys in physical.items():
                body = messages[index].content[start:end]
                if body not in body_lengths:
                    body_lengths[body] = len(template.tokenizer.encode(body, add_special_tokens=False))
                count = body_lengths[body]
                total_body_tokens += count
                if not keys & required_keys:
                    nonrequired_body_tokens += count
        if not required_keys <= visible:
            reason = "required_evidence_not_visible"
        if len(rendered.input_ids) + policy.max_new_tokens > policy.max_total_model_tokens:
            reason = "shared_context_policy_overflow"
        return PromptCandidate(
            candidate_id=f"{snapshot.session_id}:{snapshot.version}:{action}", action=action,
            snapshot_version=snapshot.version, messages=messages, input_ids=rendered.input_ids,
            input_hash=rendered.input_hash, visible_blocks=visible, required_blocks=required_keys,
            build_ms=(time.perf_counter() - started) * 1000, feasible=reason is None,
            reason=reason, evidence_spans=spans, reference_map=references,
            theoretical_lcp=longest_common_prefix(rendered.input_ids, previous),
            corpus_snapshot_id=snapshot.corpus_snapshot_id, fingerprint=template.fingerprint,
            evidence_resolution=resolution, evidence_body_tokens=total_body_tokens,
            nonrequired_evidence_body_tokens=nonrequired_body_tokens)

    execution = snapshot.execution
    if execution is not None:
        # The committed ledger is immutable; only newly supplied bodies need to
        # be added. REBUILD below derives a fresh visible set from its own bodies.
        visible = execution.visible_blocks
        reason = None
        if execution.history_version != snapshot.raw_history_version or snapshot.history_sync_pending:
            reason = "history_sync_requires_rebuild"
        elif execution.template_hash != template.fingerprint.template_hash or (
                execution.model_revision != template.fingerprint.model_revision):
            reason = "configuration_changed"
        elif execution.messages[:1] != (system_message,):
            reason = "system_prompt_changed"
        elif snapshot.force_new_epoch_for_recovery or snapshot.must_rebuild:
            reason = "recovery_requires_rebuild"
        elif not allows_version_comparison(evidence_query, policy) and version_conflicts(
                visible - required_keys, required_keys):
            reason = "invalid_visible_version"
        # An intent label alone does not invalidate cached bodies. Current
        # citation scope excludes superseded sources, versions are checked
        # above, and B prices the extra context against rebuilding it. Even
        # disjoint evidence can reuse a cached history prefix; overlap alone
        # does not establish that rebuilding has lower execution cost.
        missing = tuple(block for block in required if block.key not in visible)
        visible_bodies = {}
        for span in execution.evidence_spans:
            if span.body_reference_id is None and span.key in visible:
                visible_bodies.setdefault(archive[span.key].text, span)
        current, spans = current_user_message(query, missing, references,
            message_index=len(execution.messages), reused=required_keys & visible,
            layout=policy.layout, retrieval_priority=retrieval_priority, table_annotations=table_annotations,
            evidence_relations=evidence_relations, citation_scope=required_keys, source_guide=source_guide,
            query_context=query_context, visible_bodies=visible_bodies, origin_guide=origin_guide)
        prompts.append(("APPEND", execution.messages + (current,), execution.evidence_spans + spans,
                        visible | frozenset(span.key for span in spans), reason))

    messages = [system_message]
    for turn in snapshot.raw_history:
        messages.extend((Message("user", turn.user_text), Message("assistant", turn.assistant_text)))
    current, spans = current_user_message(query, required, references,
                                         message_index=len(messages), layout=policy.layout,
                                         retrieval_priority=retrieval_priority, table_annotations=table_annotations,
                                         evidence_relations=evidence_relations, citation_scope=required_keys,
                                         source_guide=source_guide, query_context=query_context,
                                         origin_guide=origin_guide)
    messages.append(current)
    prompts.append(("REBUILD", tuple(messages), spans, frozenset(span.key for span in spans), None))
    conversations = tuple(prompt[1] for prompt in prompts)
    rendered = template.render_many(conversations)
    return tuple(candidate(*prompt, output) for prompt, output in zip(prompts, rendered, strict=True))


def build_plan_candidates(snapshot, query, plan, template, *, policy=None, outcome=None,
                          layouts=None, actions=("REBUILD", "APPEND")):
    """Compile each legal action/layout for one semantic evidence plan."""
    from .context_budget import (append_legal, measure_window, task_instruction,
                                 working_history_messages, evidence_interval_covered, _covers)
    from .schema import EvidenceSpan
    from .serialization import dumps
    from .templates import _evidence_alias, _evidence_header

    policy = policy or CandidatePolicy()
    layouts = plan.layouts if layouts is None else layouts
    started = time.perf_counter()
    state, selected = plan.task_state, plan.selected_set
    claims = () if outcome is None else outcome.claims
    answer_plan = plan.answer_plan if outcome is None else outcome.plan.answer_plan
    blocks = {b.key: b for b in (*snapshot.archive, *selected.source_blocks)}
    units = selected.selected_units
    required = frozenset(span.source_key for unit in units for span in unit.source_spans)
    source_keys = {*blocks, *(span.source_key for unit in units for span in unit.source_spans),
                   *(span.source_key for point in answer_plan.points for span in point.source_spans)}
    references = assign_references(source_keys, snapshot.reference_map)
    by_key = {key: ref for ref, key in references}
    history = working_history_messages(state, snapshot)
    history_id = dumps(state.working_history)
    system_text = plan.answer_system_prompt if policy.system_prompt is None else policy.system_prompt
    system = Message("system", PLAN_SYSTEM_PROMPT if system_text is None else system_text)
    previous = (() if snapshot.execution is None else
                snapshot.execution.last_input_ids + snapshot.execution.last_output_ids)
    final_claim_ids = {cid for point in answer_plan.points for cid in point.claim_ids}
    from .claim_graph import ClaimGraph
    graph = ClaimGraph(claims)
    conclusion_keys = frozenset(span.source_key for point in answer_plan.points for span in point.source_spans)
    conclusions = graph.conclusion_table(answer_plan, references)
    if outcome is not None:
        conclusions["open_needs"] = outcome.open_needs
    computed = "\n".join(f"{claim.need_id}: {claim.operation} = {claim.value} {claim.unit}; "
        f"original sources = {len({span.source_key for span in graph.source_spans(claim.claim_id)})}."
        for claim in claims if claim.claim_id in final_claim_ids and claim.kind == "derived"
        and claim.operation in {"count", "sum"})
    context = ("\nCurrent task and active conditions:\n" + task_instruction(state, snapshot)
        + "\nReading conclusions (shared scopes and complete source paths):\n" + dumps(conclusions)
        + "\nComputed results for the complete reading scope:\n" + computed)

    def compile_user(ordered, message_index, reused=frozenset()):
        content, spans, physical = "", [], {}
        for unit in ordered:
            for source in unit.source_spans:
                block, ref = blocks[source.source_key], by_key[source.source_key]
                base = block.span[0] if block.span else 0
                text = block.text[source.start-base:source.end-base]
                # Same original body is rendered once; every source identity has a span.
                if text in physical:
                    start, end, owner_ref = physical[text]
                    if owner_ref == ref:
                        continue
                    declaration_start = len(content)
                    content += _evidence_alias(block, ref, owner_ref)
                    spans.append(EvidenceSpan(block.key, message_index, start, end,
                        block.source_uri, block.provenance, ref, owner_ref,
                        (message_index, declaration_start, len(content)),
                        source.start, source.end))
                    continue
                else:
                    content += _evidence_header(block, ref)
                    start = len(content)
                    content += text
                    end = len(content)
                    content += "\n</evidence_body>\n"
                    physical[text] = start, end, ref
                spans.append(EvidenceSpan(block.key, message_index, start, end,
                    block.source_uri, block.provenance, ref,
                    source_start=source.start, source_end=source.end))
        content += "\nSource spans [citation_id,start,end]: " + dumps(tuple(
            (by_key[span.source_key], span.start, span.end) for unit in ordered for span in unit.source_spans))
        if reused:
            content += "\nPreviously supplied sources: " + ", ".join(by_key[k] for k in sorted(reused))
        content += context
        allowed = (conclusion_keys | {span.source_key for unit in ordered for span in unit.source_spans}
                   | (reused & required))
        content += "\nCitations allowed for this answer: " + (
            ", ".join(f"[{by_key[k]}]" for k in sorted(allowed)) or "none") + ".\n"
        content += f"\nQuestion:\n{query}\n"
        return Message("user", content), tuple(spans)

    # Full-scope reading retains every source in the step graph. The final window
    # carries root conclusions and a budgeted set of original supporting units.
    if plan.coverage_scope == "all_supplied_documents" or outcome is not None:
        empty, _ = compile_user((), len(history)+1)
        fixed_input_tokens = len(template.render((system, *history, empty)).input_ids)
        excerpt_budget = min(plan.budget.body_tokens,
            max(0, plan.budget.window_tokens-plan.budget.output_tokens-fixed_input_tokens))

        def excerpt_tokens(proposed):
            message, _ = compile_user(proposed, len(history)+1)
            return len(template.render((system, *history, message)).input_ids)-fixed_input_tokens

        def supports_point(unit, point):
            return any(source.source_key == leaf.source_key and source.start < leaf.end
                       and leaf.start < source.end for source in unit.source_spans for leaf in point.source_spans)

        prioritized = sorted(units, key=lambda u: (
            not any(supports_point(u, point) for point in answer_plan.points), -u.retrieval_score, u.unit_id))
        # A derived answer point can join many originals. Share the excerpt
        # allocation across its source paths before filling background text.
        representatives, represented = [], set()
        for point in answer_plan.points:
            for source_key in dict.fromkeys(span.source_key for span in point.source_spans):
                if source_key in represented:
                    continue
                unit = next((unit for unit in prioritized if supports_point(unit, point)
                    and any(span.source_key == source_key for span in unit.source_spans)), None)
                if unit is not None:
                    representatives.append(unit)
                    represented.update(span.source_key for span in unit.source_spans)
        representative_ids = tuple(unit.unit_id for unit in representatives)
        final_units = []

        def add_excerpt(unit, ceiling):
            def size(proposed):
                return excerpt_tokens(proposed)
            if size((*final_units, unit)) <= ceiling:
                final_units.append(unit)
                return
            anchors = [unit.text.find(text) for claim in claims
                if any(span.source_key == source.source_key for span in claim.source_spans
                       for source in unit.source_spans)
                for text in (claim.text, claim.entity) if text]
            start = next((position for position in anchors if position >= 0), 0)
            low, high = 0, len(unit.text)-start
            while low < high:
                middle = (low+high+1)//2
                excerpt = replace(unit, text=unit.text[start:start+middle], source_spans=tuple(
                    replace(span, start=span.start+start, end=span.start+start+middle)
                    for span in unit.source_spans))
                if size((*final_units, excerpt)) <= ceiling:
                    low = middle
                else:
                    high = middle-1
            if low:
                final_units.append(replace(unit, text=unit.text[start:start+low], source_spans=tuple(
                    replace(span, start=span.start+start, end=span.start+start+low)
                    for span in unit.source_spans)))

        for index, unit in enumerate(representatives):
            add_excerpt(unit, excerpt_budget*(index+1)//len(representatives))
        for unit in prioritized:
            if unit.unit_id in representative_ids:
                continue
            if any(supports_point(unit, point) for point in answer_plan.points) or not final_units:
                add_excerpt(unit, excerpt_budget)
            elif excerpt_tokens((*final_units, unit)) <= excerpt_budget:
                final_units.append(unit)
        units = tuple(final_units)
        required = conclusion_keys | frozenset(s.source_key for u in units for s in u.source_spans)

    def visible_originals(spans):
        ranges = {}
        for span in spans:
            block = blocks[span.key]
            base = block.span[0] if block.span else 0
            interval = ((base, base+len(block.text)) if span.source_start is None else
                        (span.source_start, span.source_end))
            ranges.setdefault(span.key, []).append(interval)
        return frozenset(key for key, intervals in ranges.items() if _covers(
            (blocks[key].span[0], blocks[key].span[0]+len(blocks[key].text))
            if blocks[key].span else (0, len(blocks[key].text)), intervals))

    prompts = []
    for layout in layouts:
        if layout == "existing":
            ordered = units
        elif layout == "need_order":
            rank = {uid: i for i, uid in enumerate(dict.fromkeys(
                uid for support in selected.need_support for uid in support.unit_ids))}
            ordered = tuple(sorted(units, key=lambda u: (rank.get(u.unit_id, len(rank)), u.unit_id)))
        elif layout == "stable_evidence":
            prior_order = ({span.key: index for index, span in enumerate(snapshot.execution.evidence_spans)}
                           if snapshot.execution else {})
            def stable_rank(unit):
                present = (snapshot.execution is not None and all(
                    evidence_interval_covered(snapshot, span) for span in unit.source_spans))
                return (not present, min((prior_order.get(span.source_key, len(prior_order))
                    for span in unit.source_spans), default=len(prior_order)),
                    tuple((span.source_key, span.start, span.end) for span in unit.source_spans))
            ordered = tuple(sorted(units, key=stable_rank))
        else:
            raise ValueError(f"unknown plan layout: {layout}")
        variants = []
        if "REBUILD" in actions:
            current, spans = compile_user(ordered, len(history)+1)
            variants.append(("REBUILD", (system, *history, current), spans, visible_originals(spans), None))
        if snapshot.execution is not None and "APPEND" in actions:
            legal, reason = append_legal(snapshot, state, required,
                                        tuple(span for unit in units for span in unit.source_spans))
            execution = snapshot.execution
            missing = tuple(u for u in ordered if any(not evidence_interval_covered(snapshot, s)
                                                       for s in u.source_spans))
            current, new_spans = compile_user(missing, len(execution.messages), execution.visible_blocks)
            variants.append(("APPEND", (*execution.messages, current),
                (*execution.evidence_spans, *new_spans), visible_originals((*execution.evidence_spans, *new_spans)),
                None if legal else reason))
        prompts.extend((layout, *variant) for variant in variants)
    rendered_prompts = template.render_many(tuple(prompt[2] for prompt in prompts))
    candidates = []
    for (layout, action, messages, spans, visible, reason), rendered in zip(prompts, rendered_prompts, strict=True):
        window = measure_window(template, messages, plan.budget, rendered, history=history)
        if not window.feasible:
            reason = "context_window_budget"
        if window.history_tokens > plan.budget.history_tokens:
            reason = "working_history_budget"
        if (action == "APPEND" and (snapshot.must_rebuild or snapshot.force_new_epoch_for_recovery
                or snapshot.history_sync_pending)):
            reason = "required_rebuild"
        if action == "APPEND":
            if (snapshot.execution.template_hash != template.fingerprint.template_hash
                    or snapshot.execution.model_revision != template.fingerprint.model_revision):
                reason = "configuration_changed"
            elif snapshot.execution.messages[:1] != (system,):
                reason = "system_prompt_changed"
            elif snapshot.execution.history_version != snapshot.raw_history_version:
                reason = "history_sync_requires_rebuild"
        candidates.append(PromptCandidate(
            candidate_id=dumps([plan.plan_id, selected.set_id, history_id, layout, action]),
            action=action, snapshot_version=snapshot.version, messages=messages,
            input_ids=rendered.input_ids, input_hash=rendered.input_hash,
            visible_blocks=visible, required_blocks=required,
            build_ms=(time.perf_counter()-started)*1000, feasible=reason is None, reason=reason,
            evidence_spans=spans, reference_map=references,
            theoretical_lcp=longest_common_prefix(rendered.input_ids, previous),
            corpus_snapshot_id=snapshot.corpus_snapshot_id, fingerprint=template.fingerprint,
            context_plan_id=plan.plan_id, evidence_set_id=selected.set_id,
            working_history_id=history_id, layout_id=layout,
            quality_prediction=next((quality for candidate_action, candidate_layout, quality
                in plan.candidate_qualities if candidate_action == action and candidate_layout == layout),
                selected.quality_prediction),
            need_support=selected.need_support,
            output_budget=plan.budget.output_tokens, predicted_read_ms=plan.predicted_cost_ms,
            auxiliary_ms=sum(value for _, value in plan.auxiliary_ms),
            cost_prediction_eligible=plan.cost_prediction_eligible if outcome is None else True))
    return tuple(candidates)
