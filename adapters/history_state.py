"""Task interpretation from model-selected spans in committed, actual QA."""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from controller.context_schema import Condition, Need, RetrievalState, TaskState, TextRef, VersionIntent
from controller.context_budget import text_ref_sources, text_ref_text
from controller.schema import OnlineTurn
from controller.serialization import decode


@dataclass(frozen=True)
class HistoryWindow:
    ref: TextRef
    text: str
    tokens: int
    score: float = 0.0


@dataclass(frozen=True)
class TaskPrediction:
    relation: str
    history_spans: tuple[TextRef, ...] = ()
    conditions: tuple[Condition, ...] = ()
    needs: tuple[Need, ...] = ()
    version_intent: VersionIntent = VersionIntent()
    resume_task_id: str = ""
    source_scope: tuple = ()


def history_text(ref: TextRef, memory: RetrievalState, turn: OnlineTurn | None = None) -> str:
    return text_ref_text(ref, text_ref_sources(memory.committed_history,
        current_turn_id=turn.turn if turn is not None else None,
        current_user_text=turn.query if turn is not None else None))


def validate_task_refs(prediction, turn, memory):
    sources = text_ref_sources(memory.committed_history, current_turn_id=turn.turn, current_user_text=turn.query)
    refs = [(f"history_spans[{index}]", ref) for index, ref in enumerate(prediction.history_spans)]
    for index, condition in enumerate(prediction.conditions):
        refs.extend((f"conditions[{index}].{name}", getattr(condition, name))
                    for name in ("key_span", "predicate_span") if getattr(condition, name) is not None)
        refs.extend((f"conditions[{index}].value_spans[{offset}]", ref)
                    for offset, ref in enumerate(condition.value_spans))
    for index, need in enumerate(prediction.needs):
        refs.extend((f"needs[{index}].{name}[{offset}]", ref)
                    for name in ("anchor_spans", "subject_refs") for offset, ref in enumerate(getattr(need, name)))
    refs.extend((f"version_intent.time_refs[{index}]", ref)
                for index, ref in enumerate(prediction.version_intent.time_refs))
    for field, ref in refs:
        text_ref_text(ref, sources, field=field)


TASK_WIRE_SCHEMA = "owned_task_graph_v7_visible_token_boundaries"


def indexed_ref_windows(windows):
    """Keep literal source text and attach local indices to words and punctuation."""
    import regex

    visible, mapping = [], {}
    for ref, text in windows:
        boundaries = [0, *(match.start() for match in regex.finditer(r"\b", text, flags=regex.WORD)
                           if 0 < match.start() < len(text)), len(text)]
        parts, spans = [], []
        for start, end in zip(boundaries, boundaries[1:]):
            segment = text[start:end]
            if segment and not segment.isspace():
                parts.append(f"[{len(spans)}]")
                spans.append((ref.start+start, ref.start+end))
            parts.append(segment)
        if not spans:
            continue
        window_id = f"w{len(visible)}"
        visible.append({"window_id": window_id, "turn_id": ref.turn_id, "speaker": ref.speaker,
                        "indexed_text": "".join(parts)})
        mapping[window_id] = {"turn_id": ref.turn_id, "speaker": ref.speaker, "token_spans": spans}
    return visible, mapping


def owned_text_ref_definitions(windows):
    branches = [{"type": "object", "properties": {
        "window_id": {"type": "string", "const": window_id},
        "boundary_tokens": {"type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "integer", "minimum": 0, "maximum": len(window["token_spans"])-1}}},
        "required": ["window_id", "boundary_tokens"], "additionalProperties": False}
        for window_id, window in windows.items()]
    return {"TextRef": {"anyOf": branches} if branches else False}


def indexed_ref_value(ref, windows):
    window = windows[ref["window_id"]]
    first, last = sorted(ref["boundary_tokens"])
    spans = window["token_spans"]
    if not 0 <= first <= last < len(spans):
        raise ValueError("task reference leaves its visible token window")
    return {"turn_id": window["turn_id"], "speaker": window["speaker"],
            "start": spans[first][0], "end": spans[last][1]}


def decode_indexed_task_refs(value, windows):
    """Translate the online wire into the unchanged persisted TextRef contract."""
    def resolve(ref):
        return indexed_ref_value(ref, windows)

    value["history_spans"] = [resolve(ref) for ref in value["history_spans"]]
    for condition in value["conditions"].values():
        for name in ("key_span", "predicate_span"):
            condition[name] = None if condition[name] is None else resolve(condition[name])
        condition["value_spans"] = [resolve(ref) for ref in condition["value_spans"]]
    for event in value["condition_removals"]:
        event["predicate_span"] = resolve(event["predicate_span"])
    for need in value["needs"].values():
        for name in ("anchor_spans", "subject_refs"):
            need[name] = [resolve(ref) for ref in need[name]]
    value["version_intent"]["time_refs"] = [resolve(ref) for ref in value["version_intent"]["time_refs"]]
    return value


def active_ref_text(ref, memory, turn, excluded):
    text = history_text(ref, memory, turn)
    intervals = sorted((max(ref.start, old.start)-ref.start, min(ref.end, old.end)-ref.start)
                       for old in excluded if ref.turn_id == old.turn_id and ref.speaker == old.speaker
                       and ref.start < old.end and old.start < ref.end)
    pieces, cursor = [], 0
    for start, end in intervals:
        pieces.append(text[cursor:start])
        cursor = max(cursor, end)
    pieces.append(text[cursor:])
    return "".join(pieces)


class HistoryIndex:
    """Frozen-encoder windows retain character coordinates and actual speakers."""
    def __init__(self, tokenizer, encode, *, unit_tokens=384, candidates=16, view_tokens=8192):
        self.tokenizer, self.encode = tokenizer, encode
        self.unit_tokens, self.candidates, self.view_tokens = unit_tokens, candidates, view_tokens
        self._windows = {}
        self._vectors = {}

    def index(self, memory: RetrievalState):
        pending = []
        for row in memory.committed_history:
            for speaker, text in (("user", row.user_text), ("assistant", row.assistant_text)):
                identity = (row.turn_id, speaker, text)
                if identity in self._windows:
                    continue
                offsets = self.tokenizer(text, add_special_tokens=False,
                                         return_offsets_mapping=True)["offset_mapping"]
                windows = []
                for first in range(0, len(offsets), self.unit_tokens):
                    section = offsets[first:first+self.unit_tokens]
                    ref = TextRef(row.turn_id, speaker, section[0][0], section[-1][1])
                    windows.append(HistoryWindow(ref, text[ref.start:ref.end], len(section)))
                self._windows[identity] = tuple(windows)
                pending.extend(windows)
        if pending:
            vectors = np.asarray(self.encode([w.text for w in pending]), dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            self._vectors.update(((window.ref, window.text), vector) for window, vector in zip(pending, vectors, strict=True))

    def select(self, query: str, memory: RetrievalState, *, current_turn_id=None) -> tuple[HistoryWindow, ...]:
        self.index(memory)
        identities = [(row.turn_id, speaker, text) for row in memory.committed_history
                      for speaker, text in (("user", row.user_text), ("assistant", row.assistant_text))]
        windows = [w for identity in identities for w in self._windows[identity]]
        if not windows:
            return ()
        query_vector = np.asarray(self.encode([query])[0], dtype=np.float32)
        query_vector /= np.linalg.norm(query_vector)
        ranked = sorted((replace(w, score=float(self._vectors[(w.ref, w.text)] @ query_vector)) for w in windows),
                        key=lambda w: (-w.score, w.ref.turn_id, w.ref.speaker, w.ref.start))
        recent = {row.turn_id for row in memory.committed_history[-2:]}
        active_refs = set()
        if memory.prior_task_states:
            active = memory.prior_task_states[-1]
            active_refs.update(active.working_history)
            active_refs.update(active.version_intent.time_refs)
            for row in memory.committed_history:
                if row.turn_id in active.version_intent.source_turns:
                    active_refs.update((TextRef(row.turn_id, "user", 0, len(row.user_text)),
                                        TextRef(row.turn_id, "assistant", 0, len(row.assistant_text))))
            for need in active.needs:
                active_refs.update((*need.anchor_spans, *need.subject_refs))
            for condition in active.conditions:
                active_refs.update(condition.value_spans)
        active = [w for w in ranked if any(
            w.ref.turn_id == ref.turn_id and w.ref.speaker == ref.speaker
            and w.ref.start < ref.end and ref.start < w.ref.end for ref in active_refs)]
        recent_windows = []
        for turn_id in sorted(recent, reverse=True):
            for speaker in ("user", "assistant"):
                recent_windows.extend(w for w in ranked if w.ref.turn_id == turn_id and w.ref.speaker == speaker)
                recent_windows = list(dict.fromkeys(recent_windows))
        # One highest-scoring window per recent message leaves room for remote antecedents.
        recent_windows = [next(w for w in recent_windows if w.ref.turn_id == turn_id and w.ref.speaker == speaker)
                          for turn_id in sorted(recent, reverse=True) for speaker in ("user", "assistant")
                          if any(w.ref.turn_id == turn_id and w.ref.speaker == speaker for w in recent_windows)]
        candidates = list(dict.fromkeys((*active, *recent_windows, *ranked)))[:self.candidates]
        current_turn_id = current_turn_id or len(memory.committed_history)+1
        used = 2 + len(self.tokenizer(f"Turn {current_turn_id} user: ", add_special_tokens=False)["input_ids"])
        used += len(self.tokenizer(query, add_special_tokens=False)["input_ids"])
        selected = []
        for window in candidates:
            overhead = 1 + len(self.tokenizer(f"Turn {window.ref.turn_id} {window.ref.speaker}: ",
                                             add_special_tokens=False)["input_ids"])
            if used + window.tokens + overhead <= self.view_tokens:
                selected.append(window)
                used += window.tokens + overhead
        return tuple(sorted(selected, key=lambda w: (w.ref.turn_id, w.ref.speaker != "user", w.ref.start)))


def apply_condition_updates(prior: tuple[Condition, ...], updates: tuple[Condition, ...]):
    active = {condition.condition_id: condition for condition in prior}
    retired = dict.fromkeys(target for condition in updates if condition.update in {"REPLACE", "REMOVE"}
                            for target in (*condition.target_refs, *condition.supersedes))
    for target in retired:
        active.pop(target)
    for condition in updates:
        if condition.update == "KEEP":
            continue
        if condition.update != "REMOVE":
            active[condition.condition_id] = condition
    return tuple(active.values())


def apply_need_condition_updates(needs: tuple[Need, ...], updates: tuple[Condition, ...]):
    """Rewrite only the condition edges already assigned to each Need."""
    retired = {target for condition in updates if condition.update in {"REPLACE", "REMOVE"}
               for target in (*condition.target_refs, *condition.supersedes)}
    replacements = {target: tuple(condition.condition_id for condition in updates
        if condition.update == "REPLACE" and target in (*condition.target_refs, *condition.supersedes))
        for target in retired}
    return tuple(replace(need, condition_ids=tuple(dict.fromkeys(
        current for previous in need.condition_ids for current in replacements.get(previous, (previous,)))))
        for need in needs)


class TaskInterpreter:
    def __init__(self, model, history_index: HistoryIndex):
        self.model, self.history_index = model, history_index
        self.last_metrics = {}

    def interpret(self, turn: OnlineTurn, memory: RetrievalState) -> TaskState:
        started = time.perf_counter()
        windows = self.history_index.select(turn.query, memory, current_turn_id=turn.turn)
        prediction = self.model.predict(turn, windows, memory)
        self.last_prediction = prediction
        validate_task_refs(prediction, turn, memory)
        inherited = prediction.relation not in {"standalone", "topic_switch"}
        base = (next(state for state in reversed(memory.prior_task_states) if state.task_id == prediction.resume_task_id)
                if inherited and prediction.resume_task_id else memory.prior_task_states[-1]
                if inherited and memory.prior_task_states else None)
        conditions = apply_condition_updates(base.conditions if base else (), prediction.conditions)
        needs = prediction.needs or (base.needs if base else ())
        needs = apply_need_condition_updates(needs, prediction.conditions)
        spans = tuple(dict.fromkeys((*prediction.history_spans,
            *(ref for need in needs for ref in (*need.anchor_spans, *need.subject_refs)),
            *(ref for condition in conditions for ref in condition.value_spans))))
        working = tuple(ref for ref in spans if ref.turn_id != turn.turn)
        active_ids = {condition.condition_id for condition in conditions}
        earlier_conditions = tuple(condition for earlier in memory.prior_task_states
            if base is not None and earlier.task_id == base.task_id for condition in earlier.conditions)
        excluded = tuple(ref for condition in earlier_conditions if condition.condition_id not in active_ids
                         for ref in condition.value_spans)
        parts = [active_ref_text(ref, memory, turn, excluded)
                 for need in needs for ref in (*need.anchor_spans, *need.subject_refs)]
        parts.extend(history_text(ref, memory, turn) for condition in conditions for ref in condition.value_spans)
        parts.extend(history_text(ref, memory, turn) for ref in prediction.version_intent.time_refs)
        if prediction.version_intent.temporal_relation in {"before", "after"}:
            parts.append(prediction.version_intent.temporal_relation.upper())
        parts.append(turn.query)
        state = TaskState(base.task_id if base else turn.task_id, turn.query, working, conditions, needs,
                          prediction.version_intent, prediction.source_scope, turn.turn,
                          "\n".join(dict.fromkeys(parts)))
        self.last_metrics = {"task_model_ms": (time.perf_counter()-started)*1000,
                             "history_windows": len(windows), "history_tokens": sum(w.tokens for w in windows),
                             "model_metrics": getattr(self.model, "last_metrics", {})}
        return state


def interpret(turn: OnlineTurn, memory: RetrievalState) -> TaskState:
    return memory.source_index.task_interpreter.interpret(turn, memory)


def restore_source_metadata(source_index, units):
    """Rehydrate the public catalogue carried by persisted selected evidence."""
    for unit in units:
        metadata = {name: value for name, value in (
            ("document_family", unit.document_family), ("valid_from", unit.valid_from),
            ("valid_to", unit.valid_to), ("published_at", unit.published_at), ("heading", unit.heading),
            ("table_id", unit.table_id), ("applicability", dict(unit.applicability)),
            ("version_order", unit.version_order), ("current", unit.current)) if value is not None and value != ""}
        for span in unit.source_spans:
            current = source_index.corpus_index.source_metadata.get(span.source_key, {})
            source_index.corpus_index.source_metadata[span.source_key] = {**metadata, **current}


def retrieval_state(snapshot, source_index):
    """One committed-memory projection shared by online and offline retrieval."""
    if snapshot is None:
        return RetrievalState(0, source_index=source_index)
    for row in snapshot.raw_history:
        if row.evidence_set is None:
            continue
        restore_source_metadata(source_index, row.evidence_set.selected_units)
    source_index.add_archive(snapshot.archive)
    return RetrievalState(snapshot.version, snapshot.raw_history,
        tuple(row.query_context.task_state for row in snapshot.raw_history
              if row.query_context is not None and row.query_context.task_state is not None),
        tuple(block.key for block in snapshot.archive), source_index)



TASK_NEED_SEMANTICS_VERSION = "task_need_semantics_v2_condition_links"
TASK_NEED_SEMANTICS = (
    "Need operation semantics:\n"
    "- lookup: retrieve a directly stated fact or relation answering a requested point.\n"
    "- procedure: identify the actions, prerequisites and ordered steps for carrying out a task. "
    "Requests for required actions or how to complete a process request a procedure, "
    "even when one relevant source states the answer directly.\n"
    "- enumerate: list the requested entities, items or alternatives; use procedure when the requested items "
    "are actions for accomplishing a task.\n"
    "- infer: connect multiple factual relations or premises to derive the requested answer; "
    "a multi-step procedure alone is not infer.\n"
    "- compare: establish requested similarities, differences or ordering between subjects or versions.\n"
    "- count: compute the number of distinct matching entities or records in the requested range.\n"
    "- sum: add the requested quantities, respecting units and the requested matching range.\n"
    "- group: partition or aggregate the requested items by the requested grouping keys.\n"
    "Choose each Need's operation from the user's requested result, not from how many documents exist "
    "or whether a reader could retrieve the answer. Keep separately requested results as separate Needs.\n"
    "Need condition links:\n"
    "- condition_ids names only the active restrictions that apply to that Need. An empty list means "
    "the Need has no condition restriction. Conditions elsewhere in the task do not automatically apply.\n"
    "- When a restriction is shared, explicitly assign the same condition ID to every affected Need. "
    "Keep different dates, subjects or eligibility restrictions attached to their respective requested results.\n"
    "- Return the active Needs for a followup with their resulting condition links, including new ADD "
    "conditions. REPLACE and REMOVE update only Needs that reference their targeted prior condition.\n"
    "Need scope semantics (independent of operation):\n"
    "- selected_sources: answer from relevant selected sources. This is the ordinary scope for lookup, "
    "procedure and other requests without a demand to cover the complete supplied source range.\n"
    "- all_supplied_documents: the user requires exhaustive coverage across the complete supplied source "
    "range, for example counting every matching record in a supplied ledger or listing matches across "
    "every provided document. The complete range may be established in unchanged actual conversation history.\n"
    "Multiple available documents do not establish all_supplied_documents. Asking for all required steps "
    "of a procedure does not by itself demand a scan of all supplied documents. Do not infer exhaustive "
    "source scope from operation=enumerate/count/sum alone. Do not expand a previously selected task's "
    "scope without a current or inherited actual user request for the complete source range."
)

class TeacherTaskModel:
    """Structured Qwen teacher for data collection and first integration runs."""
    def __init__(self, client, *, record_path=None):
        self.client = client
        self.record_path = Path(record_path) if record_path is not None else None
        self.usage = {"tasks": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}

    @property
    def manifest(self):
        return {"kind": "qwen3_32b_task_span_teacher", "wire_schema": TASK_WIRE_SCHEMA,
                "need_semantics_version": TASK_NEED_SEMANTICS_VERSION,
                "transport": self.client.manifest}

    def predict(self, turn, windows, memory):
        from controller.serialization import to_jsonable
        latest = {state.task_id: state for state in memory.prior_task_states}
        task_alias = {task_id: f"S{i}" for i, task_id in enumerate(latest)}
        condition_ids = tuple(dict.fromkeys(condition.condition_id for state in memory.prior_task_states
                                             for condition in state.conditions))
        need_ids = tuple(dict.fromkeys(need.need_id for state in memory.prior_task_states for need in state.needs))
        condition_alias = {identity: f"C{i}" for i, identity in enumerate(condition_ids)}
        need_alias = {identity: f"N{i}" for i, identity in enumerate(need_ids)}
        current_conditions = {f"c{i}": f"{turn.task_id}:c{i}" for i in range(8)}
        current_needs = {f"n{i}": f"{turn.task_id}:n{i}" for i in range(8)}
        catalogue = {}
        if memory.source_index is not None:
            for key, block in memory.source_index._blocks.items():
                metadata = memory.source_index.corpus_index.source_metadata.get(key, {})
                family = metadata.get("document_family", key.document_id)
                catalogue.setdefault(family, []).append({"version": key.version, "title": block.title,
                    **{name: metadata[name] for name in ("valid_from", "valid_to", "version_order", "current", "applicability")
                       if name in metadata}})
        versions = tuple(dict.fromkeys([row["version"] for rows in catalogue.values() for row in rows]
            +[version for state in latest.values() for version in
              (*state.version_intent.version_ids, *state.version_intent.resolved_version_ids)]))
        families = tuple(dict.fromkeys([family for family, rows in catalogue.items()
            if len({row["version"] for row in rows}) > 1 or any(row.get("valid_from") or row.get("version_order") for row in rows)]
            +[family for state in latest.values() for family in state.version_intent.document_families]))
        version_alias = {version: f"V{i}" for i, version in enumerate(versions)}
        family_alias = {family: f"F{i}" for i, family in enumerate(families)}
        prior_view = to_jsonable(tuple(latest.values()))
        for item in prior_view:
            item["task_id"] = task_alias[item["task_id"]]
            for condition in item["conditions"]:
                condition["condition_id"] = condition_alias[condition["condition_id"]]
                condition["target_refs"] = [condition_alias[identity] for identity in condition["target_refs"]]
                condition["supersedes"] = [condition_alias[identity] for identity in condition["supersedes"]]
            for need in item["needs"]:
                need["need_id"] = need_alias[need["need_id"]]
                need["condition_ids"] = [condition_alias[identity] for identity in need["condition_ids"]]
                need["predecessor_need_ids"] = [need_alias[identity] for identity in need["predecessor_need_ids"]]
            item["version_intent"]["version_ids"] = [version_alias[identity] for identity in item["version_intent"]["version_ids"]]
            item["version_intent"]["resolved_version_ids"] = [version_alias[identity]
                for identity in item["version_intent"]["resolved_version_ids"]]
            item["version_intent"]["document_families"] = [family_alias[identity] for identity in item["version_intent"]["document_families"]]
        owners = text_ref_sources(memory.committed_history, current_turn_id=turn.turn, current_user_text=turn.query)
        source_windows = [(TextRef(turn.turn, "user", 0, len(turn.query)), turn.query)]
        source_windows.extend((w.ref, w.text) for w in windows)
        condition_by_id = {condition.condition_id: condition for state in memory.prior_task_states
                           for condition in state.conditions}
        active_conditions = [condition for state in latest.values() for condition in state.conditions]
        prior_condition_ids = dict.fromkeys(identity for condition in active_conditions
            for identity in (condition.condition_id, *condition.target_refs, *condition.supersedes))
        condition_text = {}
        for identity in prior_condition_ids:
            condition = condition_by_id[identity]
            parts = {}
            for name, refs_ in (("key", (condition.key_span,)), ("predicate", (condition.predicate_span,)),
                                ("values", condition.value_spans)):
                parts[name] = []
                for ref in refs_:
                    if ref is None:
                        continue
                    text = text_ref_text(ref, owners)
                    parts[name].append(text)
                    if not any(existing.turn_id == ref.turn_id and existing.speaker == ref.speaker
                               and existing.start <= ref.start and ref.end <= existing.end
                               for existing, _ in source_windows):
                        source_windows.append((ref, text))
            condition_text[condition_alias[identity]] = parts
        visible, ref_windows = indexed_ref_windows(source_windows)
        self.last_text_ref_windows = ref_windows
        body = {"turn_id": turn.turn, "question_window_id": visible[0]["window_id"],
                "visible_windows": visible, "prior_condition_text": condition_text,
                "reference_date": memory.source_index.reference_date if memory.source_index is not None else None,
                "prior_states": prior_view,
                "source_versions": [{"version_id": alias, "literal_version": value} for value, alias in version_alias.items()],
                "source_families": [{"family_id": alias, "literal_family": family,
                    "sources": [{**row, "version_id": version_alias[row["version"]]} for row in catalogue.get(family, ())]}
                    for family, alias in family_alias.items()]}
        body["wire_schema"] = self.manifest["wire_schema"]
        body["need_semantics_version"] = TASK_NEED_SEMANTICS_VERSION
        definitions = owned_text_ref_definitions(ref_windows)
        ref_schema = {"$ref": "#/$defs/TextRef"}
        refs = {"type": "array", "maxItems": 16, "items": ref_schema}
        definitions["TextRefs"] = refs
        refs = {"$ref": "#/$defs/TextRefs"}
        optional_ref = {"anyOf": [ref_schema, {"type": "null"}]}
        definitions["Operation"] = {"type": "string", "enum": ["lookup", "enumerate", "procedure", "compare", "count", "sum", "group", "infer"]}
        definitions["NeedScope"] = {"type": "string", "enum": ["selected_sources", "all_supplied_documents"]}
        strings = {"type": "array", "maxItems": 16, "items": {"type": "string", "maxLength": 512}}
        condition_schema = {"type": "object", "properties": {
            "key_span": optional_ref, "value_spans": refs,
            "relation_type": {"type": "string", "enum": ["PREDICATE", "TIME", "VERSION", "PLATFORM", "LOCATION", "SUBJECT"]},
            "predicate_span": optional_ref,
            "target_refs": strings, "supersedes": strings,
            "update": {"type": "string", "enum": ["ADD", "REPLACE", "COMPARE"]}},
            "required": ["key_span", "value_spans", "relation_type", "predicate_span",
                         "target_refs", "supersedes", "update"], "additionalProperties": False}
        need_schema = {"type": "object", "properties": {
            "anchor_spans": refs, "subject_refs": refs,
            "operation": {"$ref": "#/$defs/Operation"},
            "condition_ids": strings, "predecessor_need_ids": strings,
            "scope": {"$ref": "#/$defs/NeedScope"}},
            "required": ["anchor_spans", "subject_refs", "operation", "condition_ids", "predecessor_need_ids", "scope"],
            "additionalProperties": False}
        properties = {"relation": {"type": "string", "enum": ["standalone", "followup", "correction", "revisit", "topic_switch"]},
            "history_spans": refs, "resume_task_id": {"type": "string"},
            "version_intent": {"type": "object", "properties": {
                "mode": {"type": "string", "enum": ["current", "explicit", "as_of", "compare"]},
                "temporal_relation": {"type": "string", "enum": ["at", "before", "after"]},
                "document_families": {"type": "array", "maxItems": min(16, len(family_alias)),
                    "items": {"type": "string", **({"enum": list(family_alias.values())} if family_alias else {})}},
                "version_ids": {"type": "array", "maxItems": min(16, len(version_alias)),
                    "items": {"type": "string", **({"enum": list(version_alias.values())} if version_alias else {})}},
                "resolved_version_ids": {"type": "array", "maxItems": min(16, len(version_alias)),
                    "items": {"type": "string", **({"enum": list(version_alias.values())} if version_alias else {})}},
                "time_refs": refs,
                "source_turns": {"type": "array", "maxItems": min(2, len(memory.committed_history)),
                    "items": {"type": "integer", **({"enum": [row.turn_id for row in memory.committed_history]}
                        if memory.committed_history else {})}}},
                "required": ["mode", "temporal_relation", "document_families", "version_ids", "resolved_version_ids", "time_refs", "source_turns"],
                "additionalProperties": False}}
        definitions["VersionIntent"] = properties.pop("version_intent")
        properties["version_intent"] = {"$ref": "#/$defs/VersionIntent"}

        def pointer(values):
            return {"type": "array", "maxItems": min(16, len(values)),
                "items": {"type": "string", **({"enum": list(values)} if values else {})}}

        for need_index, need_key in enumerate(current_needs):
            definitions[f"Predecessors_{need_key}"] = pointer(
                [*need_alias.values(), *tuple(current_needs)[:need_index]])

        def owned_object(keys, value_schema):
            return {"type": "object", "properties": {key: value_schema(key) for key in keys},
                    "required": list(keys), "additionalProperties": False}

        branches = []
        for base_index, base in enumerate((None, *latest.values())):
            targets = [condition_alias[condition.condition_id] for condition in base.conditions] if base else []
            condition = deepcopy(condition_schema)
            item = condition["properties"]
            item["target_refs"], item["supersedes"] = pointer(targets), pointer(targets)
            if not targets:
                item["update"]["enum"] = ["ADD", "COMPARE"]
            condition_name = f"Condition_{base_index}"
            definitions[condition_name] = condition
            for condition_count in range(9):
                branch = deepcopy(properties)
                branch["relation"]["enum"] = ["standalone", "topic_switch"] if base is None else ["followup", "correction", "revisit"]
                branch["resume_task_id"] = {"type": "string", "enum": [task_alias[base.task_id] if base else ""]}
                condition_keys = tuple(current_conditions)[:condition_count]
                condition_refs_name = f"ConditionRefs_{base_index}_{condition_count}"
                definitions[condition_refs_name] = pointer([*targets, *condition_keys])
                branch["conditions"] = owned_object(condition_keys,
                    lambda key: {"$ref": f"#/$defs/{condition_name}"})
                branch["condition_removals"] = {"type": "array", "maxItems": len(targets),
                    "items": {"type": "object", "properties": {"target_refs": pointer(targets),
                        "predicate_span": ref_schema}, "required": ["target_refs", "predicate_span"],
                        "additionalProperties": False}}
                for need_index, need_key in enumerate(current_needs):
                    need = deepcopy(need_schema)
                    need["properties"]["condition_ids"] = {"$ref": f"#/$defs/{condition_refs_name}"}
                    need["properties"]["predecessor_need_ids"] = {"$ref": f"#/$defs/Predecessors_{need_key}"}
                    definitions[f"Need_{base_index}_{condition_count}_{need_key}"] = need
                branch["needs"] = {"anyOf": [owned_object(tuple(current_needs)[:count],
                    lambda key: {"$ref": f"#/$defs/Need_{base_index}_{condition_count}_{key}"}) for count in range(9)]}
                branches.append({"type": "object", "properties": branch, "required": list(branch),
                                 "additionalProperties": False})
        schema = {"type": "object", "anyOf": branches, "$defs": definitions}
        self.last_response_schema = schema
        system = ("Interpret the current task using the indexed literal text of visible_windows. "
                  "The current question is the window named by question_window_id. "
                  "Each [i] marks the next word or punctuation token, with indices local to that window. "
                  "For a TextRef return window_id and boundary_tokens:[i,j], the two INCLUDED visible indices "
                  "of the first and last tokens to cite. Equal indices cite one token. Order is irrelevant. "
                  "Use only indices shown in that window; do not calculate character offsets or cite hidden text. "
                  "Prior-state ledger refs are stored character coordinates and are not output citation indices. "
                  "Use null for an absent key_span or predicate_span, and empty arrays when no references apply. "
                  "If the actual request adds no condition, output conditions:{}. "
                  "VERSION requires an actual user version or time restriction. Catalogue versions and output "
                  "measurement or currency units do not establish VERSION conditions. "
                  "Keep the active object and needs for followups. For a correction emit REPLACE linked to prior condition IDs. "
                  "Represent every distinct requested answer point as a need, including related points requested together; "
                  "choose the operation for each point and preserve its question anchors. "
                  "Select an old task for revisits. Predict operations and current/explicit/as_of/compare version intent. "
                  "Distinguish at-date from before/after. For relative previous/next policy, point source_turns "
                  "to the actual turn whose source version anchors that relation. "
                  "Do not produce answer facts. Use an empty resume_task_id for a new task. "
                  "A continuation must name its exact prior task ID. Condition targets and supersedes must select "
                  "condition IDs from that task's active state; a new task adds conditions without old targets. "
                  "conditions and needs are objects with contiguous keys c0..c{k-1} and n0..n{m-1}. "
                  "Their values do not contain condition_id or need_id: the controller assigns those IDs. "
                  "A need may name only existing prior needs or earlier current keys, never itself or a later need. "
                  "Counting and summing the same population are independent operations. A shared pronoun or "
                  "the order of the requests does not create a dependency unless the count itself is used in the later formula. "
                  "A need's current condition IDs must name conditions actually emitted in this object. "
                  "Emit ADD/REPLACE/COMPARE only for materialized active conditions. KEEP means no update. "
                  "Use condition_removals with original predicate spans and prior targets for REMOVE events; "
                  "removal events cannot be referenced as current conditions. "
                  "Select document_families and version_ids only through supplied F/V catalogue aliases; "
                  "literal versions may have prefixes different from user wording. Leave unknown IDs empty and "
                  "retain literal version/date wording through original TextRefs. The catalogue is source metadata, "
                  "not an answer oracle. For an as-of date, use the source catalogue's declared valid_from/to "
                  "and reference_date to select the final applicable resolved_version_ids; never invent unknown dates. "
                  "resolved_version_ids are the final selected versions, while version_ids/source_turns may identify "
                  "anchors for before/after. Preserve the user's temporal_relation and original time_refs. "
                  "When dates are unknown, leave date-based resolved_version_ids empty.\n\n"
                  "Example current question window w0: [0]Report [1]the [2]distance [3]in "
                  "[4]kilometers [5]and [6]the [7]duration [8]in [9]minutes[10].\n"
                  'Example output: {"relation":"standalone","history_spans":[],"resume_task_id":"",'
                  '"version_intent":{"mode":"current","temporal_relation":"at","document_families":[],'
                  '"version_ids":[],"resolved_version_ids":[],"time_refs":[],"source_turns":[]},'
                  '"conditions":{},"condition_removals":[],"needs":{'
                  '"n0":{"anchor_spans":[{"window_id":"w0","boundary_tokens":[2,4]}],'
                  '"subject_refs":[],"operation":"lookup","condition_ids":[],"predecessor_need_ids":[],'
                  '"scope":"selected_sources"},'
                  '"n1":{"anchor_spans":[{"window_id":"w0","boundary_tokens":[7,9]}],'
                  '"subject_refs":[],"operation":"lookup","condition_ids":[],"predecessor_need_ids":[],'
                  '"scope":"selected_sources"}}}\n'
                  "Here both units specify the requested results. A user request for a named edition supplies "
                  "a VERSION restriction; a user request for the rule in force on a specified date supplies "
                  "as_of intent with a reference to that date.\n\n"+TASK_NEED_SEMANTICS)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]
        completion = self.client.complete(messages, response_format={"type": "json_schema", "json_schema": {
                "name": "contextkv_task_spans", "strict": True, "schema": schema}})
        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"task_id": turn.task_id, "status": completion.status,
                    "wire_schema": self.manifest["wire_schema"],
                    "need_semantics_version": TASK_NEED_SEMANTICS_VERSION,
                    "finish_reason": completion.finish_reason, "input_tokens": completion.input_tokens,
                    "usage": completion.usage, "elapsed_ms": completion.elapsed_ms,
                    "raw_output": completion.text, "error": completion.error,
                    "messages": messages, "text_ref_windows": ref_windows,
                    "raw_response": completion.raw_response}, ensure_ascii=False)+"\n")
        if completion.status != "completed" or completion.finish_reason != "stop":
            raise RuntimeError(f"task interpretation unfinished: {completion.status}/{completion.finish_reason}")
        self.last_metrics = {"input_tokens": completion.input_tokens, "usage": completion.usage,
                             "elapsed_ms": completion.elapsed_ms}
        self.usage["tasks"] += 1
        self.usage["input_tokens"] += completion.input_tokens
        self.usage["output_tokens"] += completion.usage["completion_tokens"]
        self.usage["elapsed_ms"] += completion.elapsed_ms
        self.last_raw_output = completion.text
        value = decode_indexed_task_refs(json.loads(completion.text), ref_windows)
        removals = value.pop("condition_removals")
        value["conditions"] = [{"condition_id": key, **value["conditions"][key]}
            for key in tuple(current_conditions)[:len(value["conditions"])]]
        value["needs"] = [{"need_id": key, **value["needs"][key]}
            for key in tuple(current_needs)[:len(value["needs"])]]
        reverse_conditions = {alias: identity for identity, alias in condition_alias.items()}
        reverse_needs = {alias: identity for identity, alias in need_alias.items()}
        for condition in value["conditions"]:
            condition["condition_id"] = current_conditions[condition["condition_id"]]
            condition["target_refs"] = [reverse_conditions[identity] for identity in condition["target_refs"]]
            condition["supersedes"] = [reverse_conditions[identity] for identity in condition["supersedes"]]
        for need in value["needs"]:
            need["need_id"] = current_needs[need["need_id"]]
            need["condition_ids"] = [{**reverse_conditions, **current_conditions}[identity] for identity in need["condition_ids"]]
            need["predecessor_need_ids"] = [{**reverse_needs, **current_needs}[identity] for identity in need["predecessor_need_ids"]]
        value["conditions"].extend({"condition_id": f"{turn.task_id}:remove{index}", "update": "REMOVE",
            "target_refs": [reverse_conditions[identity] for identity in event["target_refs"]],
            "predicate_span": event["predicate_span"]} for index, event in enumerate(removals))
        if value["resume_task_id"]:
            value["resume_task_id"] = {alias: identity for identity, alias in task_alias.items()}[value["resume_task_id"]]
        value["version_intent"]["version_ids"] = [{alias: identity for identity, alias in version_alias.items()}[alias]
            for alias in value["version_intent"]["version_ids"]]
        value["version_intent"]["resolved_version_ids"] = [{alias: identity for identity, alias in version_alias.items()}[alias]
            for alias in value["version_intent"]["resolved_version_ids"]]
        value["version_intent"]["document_families"] = [{alias: identity for identity, alias in family_alias.items()}[alias]
            for alias in value["version_intent"]["document_families"]]
        return decode(TaskPrediction, value)
