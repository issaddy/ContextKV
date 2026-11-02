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
        needs = tuple(replace(need, condition_ids=tuple(c.condition_id for c in conditions)) for need in needs)
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


class TeacherTaskModel:
    """Structured Qwen teacher for data collection and first integration runs."""
    def __init__(self, client, *, record_path=None):
        self.client = client
        self.record_path = Path(record_path) if record_path is not None else None
        self.usage = {"tasks": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}

    @property
    def manifest(self):
        return {"kind": "qwen3_32b_task_span_teacher", "wire_schema": "owned_task_graph_v3_message_refs",
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
        body = {"turn_id": turn.turn, "question": turn.query,
                "reference_date": memory.source_index.reference_date if memory.source_index is not None else None,
                "history": [{"ref": to_jsonable(w.ref), "text": w.text} for w in windows],
                "prior_states": prior_view,
                "source_versions": [{"version_id": alias, "literal_version": value} for value, alias in version_alias.items()],
                "source_families": [{"family_id": alias, "literal_family": family,
                    "sources": [{**row, "version_id": version_alias[row["version"]]} for row in catalogue.get(family, ())]}
                    for family, alias in family_alias.items()]}
        owners = text_ref_sources(memory.committed_history, current_turn_id=turn.turn, current_user_text=turn.query)
        body["text_ref_sources"] = [{"turn_id": identity, "speaker": speaker, "characters": len(text)}
            for (identity, speaker), text in owners.items()]
        ref_schema = {"anyOf": [{"type": "object", "properties": {
            "turn_id": {"type": "integer", "const": identity},
            "speaker": {"type": "string", "const": speaker},
            "start": {"type": "integer", "minimum": 0, "maximum": len(text)-1},
            "end": {"type": "integer", "minimum": 1, "maximum": len(text)}},
            "required": ["turn_id", "speaker", "start", "end"], "additionalProperties": False}
            for (identity, speaker), text in owners.items() if text]}
        definitions = {"TextRef": ref_schema}
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
        system = ("Interpret the current task using literal half-open Unicode character spans in the supplied actual QA. "
                  "History text starts at ref.start in its original message. Open values must point to original spans. "
                  "TextRefs may name only a listed text_ref_sources message: the current turn has a user message only. "
                  "Never invent turn 0, future turns or a current assistant message. start must be less than end. "
                  "Use null for an absent key_span or predicate_span, and empty arrays when no references apply; "
                  "do not create dummy or zero-width spans. "
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
                  "When dates are unknown, leave date-based resolved_version_ids empty.")
        completion = self.client.complete([{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "contextkv_task_spans", "strict": True, "schema": schema}})
        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"task_id": turn.task_id, "status": completion.status,
                    "finish_reason": completion.finish_reason, "input_tokens": completion.input_tokens,
                    "usage": completion.usage, "elapsed_ms": completion.elapsed_ms,
                    "raw_output": completion.text, "error": completion.error,
                    "text_ref_sources": body["text_ref_sources"],
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
        value = json.loads(completion.text)
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
