"""Pure APPEND/REBUILD construction over one immutable committed snapshot."""
from __future__ import annotations

import time
import re
from typing import Iterable

from .evidence import (EvidenceError, EvidencePolicy, allows_version_comparison, assign_references,
                       evidence_index, resolve_required_evidence_with_reasons, version_conflicts)
from .schema import EvidenceBlock, EvidenceKey, Message, PromptCandidate, QueryContext, SessionSnapshot
from .templates import (SYSTEM_PROMPT, ByteTemplate, Template, current_user_message,
                        current_source_guide, longest_common_prefix, render_evidence_relations, verify_rendered_evidence)


CandidatePolicy = EvidencePolicy


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
    revisit_turns = (None if query_context is None else
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
                                      retrieval_priority=retrieval_priority if query_context is not None else ())
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
            # Reuse lengths only inside this build; still count every physical span.
            total_body_tokens = nonrequired_body_tokens = 0
            for span in spans:
                if span.key not in body_lengths:
                    body = messages[span.message_index].content[span.start:span.end]
                    body_lengths[span.key] = len(template.tokenizer.encode(body, add_special_tokens=False))
                count = body_lengths[span.key]
                total_body_tokens += count
                if span.key not in required_keys:
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
        elif not allows_version_comparison(evidence_query, policy) and version_conflicts(visible, required_keys):
            reason = "invalid_visible_version"
        elif visible and not required_keys & visible:
            # No current source body would be reused. Carrying the old evidence
            # only adds an ineligible source set; rebuild keeps the actual QA.
            reason = "evidence_working_set_replaced"
        # An intent label alone does not invalidate cached bodies. Current
        # citation scope excludes superseded sources, versions are checked
        # above, and B prices the extra context against rebuilding it.
        missing = tuple(block for block in required if block.key not in visible)
        current, spans = current_user_message(query, missing, references,
            message_index=len(execution.messages), reused=required_keys & visible,
            layout=policy.layout, retrieval_priority=retrieval_priority, table_annotations=table_annotations,
            evidence_relations=evidence_relations, citation_scope=required_keys, source_guide=source_guide,
            query_context=query_context)
        prompts.append(("APPEND", execution.messages + (current,), execution.evidence_spans + spans,
                        visible | frozenset(span.key for span in spans), reason))

    messages = [system_message]
    for turn in snapshot.raw_history:
        messages.extend((Message("user", turn.user_text), Message("assistant", turn.assistant_text)))
    current, spans = current_user_message(query, required, references,
                                         message_index=len(messages), layout=policy.layout,
                                         retrieval_priority=retrieval_priority, table_annotations=table_annotations,
                                         evidence_relations=evidence_relations, citation_scope=required_keys,
                                         source_guide=source_guide, query_context=query_context)
    messages.append(current)
    prompts.append(("REBUILD", tuple(messages), spans, frozenset(span.key for span in spans), None))
    conversations = tuple(prompt[1] for prompt in prompts)
    render_many = getattr(template, "render_many", None)
    rendered = (render_many(conversations) if render_many is not None else
                tuple(template.render(messages) for messages in conversations))
    return tuple(candidate(*prompt, output) for prompt, output in zip(prompts, rendered, strict=True))
