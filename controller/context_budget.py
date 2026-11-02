"""Compile task-selected original history and account complete model windows."""
from __future__ import annotations

from dataclasses import dataclass
import json

from .schema import Message
from .serialization import to_jsonable


def text_ref_sources(history, *, current_turn_id=None, current_user_text=None):
    sources = {(row.turn_id, speaker): text for row in history for speaker, text in (
        ("user", row.user_text), ("assistant", row.assistant_text))}
    if current_turn_id is not None:
        sources[current_turn_id, "user"] = current_user_text
    return sources


def text_ref_text(ref, sources, *, field="TextRef"):
    owner = ref.turn_id, ref.speaker
    if owner not in sources:
        raise ValueError(f"{field} names absent actual-QA message {owner}; available={tuple(sources)}")
    text = sources[owner]
    if not 0 <= ref.start < ref.end <= len(text):
        raise ValueError(f"{field} span [{ref.start}:{ref.end}] is outside actual-QA message {owner} length {len(text)}")
    return text[ref.start:ref.end]


def resolve_text(ref, state, snapshot):
    return text_ref_text(ref, text_ref_sources(snapshot.raw_history,
        current_turn_id=state.current_turn_id, current_user_text=state.raw_question))


def working_history_messages(state, snapshot):
    return tuple(Message(ref.speaker,
        f"[turn={ref.turn_id} speaker={ref.speaker} chars={ref.start}:{ref.end}]\n"
        + resolve_text(ref, state, snapshot)) for ref in state.working_history)


def task_instruction(state, snapshot):
    conditions = [{"id": c.condition_id, "relation": c.relation_type,
        "values": [resolve_text(ref, state, snapshot) for ref in c.value_spans],
        "predicate": resolve_text(c.predicate_span, state, snapshot) if c.predicate_span else "",
        "supersedes": c.supersedes} for c in state.conditions]
    return json.dumps({"task_id": state.task_id, "conditions": conditions,
        "needs": to_jsonable(state.needs), "version_intent": to_jsonable(state.version_intent)},
        ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class WindowCost:
    input_tokens: int
    output_tokens: int
    history_tokens: int
    source_tokens: int
    window_tokens: int

    @property
    def feasible(self):
        return self.input_tokens + self.output_tokens <= self.window_tokens


def measure_window(template, messages, budget, rendered, *, history=(), source_text=""):
    encode = template.tokenizer.encode
    return WindowCost(len(rendered.input_ids), budget.output_tokens,
        sum(len(encode(m.content, add_special_tokens=False)) for m in history),
        len(encode(source_text, add_special_tokens=False)), budget.window_tokens)


def _covers(interval, ranges):
    start, end = interval
    for left, right in sorted(ranges):
        if left > start:
            break
        if right > start:
            start = right
        if start >= end:
            return True
    return start >= end


def evidence_interval_covered(snapshot, source):
    archive = {block.key: block for block in snapshot.archive}
    ranges = []
    for span in snapshot.execution.evidence_spans:
        if span.key != source.source_key:
            continue
        block = archive[span.key]
        base = block.span[0] if block.span else 0
        ranges.append((base, base+len(block.text)) if span.source_start is None else
                      (span.source_start, span.source_end))
    return _covers((source.start, source.end), ranges)


def append_legal(snapshot, state, required_keys, source_spans):
    if snapshot.execution is None:
        return False, "initial_rebuild"
    if not snapshot.execution.visible_blocks <= required_keys:
        return False, "source_scope_shrank"
    previous = next((turn.query_context.task_state for turn in reversed(snapshot.raw_history)
                     if turn.query_context and turn.query_context.task_state), None)
    if previous is None:
        return False, "task_state_initialization"
    if previous.task_id != state.task_id:
        return False, "task_changed"
    if previous.conditions != state.conditions:
        return False, "conditions_changed"
    if previous.version_intent != state.version_intent:
        return False, "version_intent_changed"
    last_turn = snapshot.raw_history[-1]
    retained = [(ref.turn_id, ref.speaker, ref.start, ref.end) for ref in previous.working_history]
    retained.extend(((last_turn.turn_id, "user", 0, len(last_turn.user_text)),
                     (last_turn.turn_id, "assistant", 0, len(last_turn.assistant_text))))
    for turn_id, speaker, start, end in retained:
        ranges = [(ref.start, ref.end) for ref in state.working_history
                  if ref.turn_id == turn_id and ref.speaker == speaker]
        if not _covers((start, end), ranges):
            return False, "working_history_shrank"
    archive = {block.key: block for block in snapshot.archive}
    for span in snapshot.execution.evidence_spans:
        block = archive[span.key]
        base = block.span[0] if block.span else 0
        interval = ((base, base+len(block.text)) if span.source_start is None else
                    (span.source_start, span.source_end))
        if not _covers(interval, [(s.start, s.end) for s in source_spans if s.source_key == span.key]):
            return False, "source_span_scope_shrank"
    return True, ""
