"""BGE-M3 query LoRA and bounded owned-task graph heads."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import time

from adapters.history_state import HistoryIndex, TaskModelResult, TaskPrediction, task_source_catalogue, task_catalogue_candidates, TASK_MAX_OBJECTS, TASK_MAX_REFS, TASK_MAX_VERSION_TURNS, task_graph_history, validate_task_anchors, observed_task_sources, budget_task_catalogue
from controller.context_schema import Condition, Need, RetrievalState, TextRef, VersionIntent
from controller.schema import OnlineTurn, TurnRecord
from controller.serialization import decode
from training.task_relations import SCOPES, FORMAT, positions, candidates, targets as relation_targets, apply_edges, graph_items, grouped_indices, grouping_targets, ordered_refs
from training.task_data import read_jsonl, select_cohort_rows


RELATIONS = ("standalone", "followup", "correction", "revisit", "topic_switch")
OPERATIONS = ("lookup", "enumerate", "procedure", "compare", "count", "sum", "group", "infer")
UPDATES = ("ADD", "REPLACE", "REMOVE", "COMPARE")
VERSIONS = ("current", "explicit", "as_of", "compare")
TEMPORAL_RELATIONS = ("at", "before", "after")
CONDITION_RELATIONS = ("PREDICATE", "TIME", "VERSION", "PLATFORM", "LOCATION", "SUBJECT")
SPAN_KINDS = ("NEED", "SUBJECT", "CONDITION_KEY", "CONDITION_VALUE", "CONDITION_PREDICATE", "TIME")
SPAN_LABELS = ("O", *(prefix+kind for kind in SPAN_KINDS for prefix in ("B-", "I-")))


def encode_task(tokenizer, turn, windows, memory, *, catalogue=None, max_input_tokens=8192):
    ids, refs, groups = [tokenizer.cls_token_id], [None], []
    groups.append((TextRef(turn.turn, "user", 0, len(turn.query)), turn.query))
    # Historical graph candidates are encoded from actual QA, independently of labels.
    from adapters.history_state import history_text
    prior_refs = tuple(dict.fromkeys(ref for state in reversed(memory.prior_task_states)
        for refs in ([n.anchor_spans + n.subject_refs for n in state.needs] +
                     [c.value_spans + ((c.key_span,) if c.key_span else ()) +
                      ((c.predicate_span,) if c.predicate_span else ()) for c in state.conditions])
        for ref in refs))
    for ref in prior_refs:
        if not any(owner.turn_id == ref.turn_id and owner.speaker == ref.speaker
                   and owner.start <= ref.start and ref.end <= owner.end for owner, _ in groups):
            groups.append((ref, history_text(ref, memory, turn)))
    catalogue = task_source_catalogue(memory) if catalogue is None else catalogue
    catalogue = budget_task_catalogue(tokenizer, turn.query, catalogue, memory.prior_task_states, max_input_tokens, current_turn_id=turn.turn)
    versions, families = task_catalogue_candidates(catalogue)
    current_size = len(tokenizer(f"Turn {turn.turn} user: ", add_special_tokens=False)["input_ids"]) + len(tokenizer(turn.query, add_special_tokens=False)["input_ids"]) + 1
    if 1 + current_size > max_input_tokens:
        raise ValueError("current question exceeds the encoder input budget")
    catalogue_positions = {}
    for name, identities in (("document_families", families), ("version_ids", versions)):
        catalogue_positions[name] = []
        for identity in identities:
            rows = (catalogue.get(identity, []) if name == "document_families" else
                    [{"family": family, **row} for family, rows in catalogue.items()
                     for row in rows if row["version"] == identity])
            rows = list({json.dumps(row, sort_keys=True, ensure_ascii=False): row for row in rows}.values())
            text = json.dumps({"kind": name, "identity": identity, "sources": rows}, ensure_ascii=False)
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            begin = len(ids)
            ids.extend(tokens + [tokenizer.sep_token_id])
            refs.extend([None]*(len(tokens)+1))
            catalogue_positions[name].append(list(range(begin, begin+len(tokens))))
    metadata_limit = len(ids) + max(0, (max_input_tokens - current_size - len(ids))//3)
    # Pointer candidates have their own observed metadata representations even
    # when the optional history window for that turn is outside the budget.
    def metadata_positions(value):
        tokens = tokenizer(json.dumps(value, ensure_ascii=False), add_special_tokens=False)["input_ids"]
        if len(ids) + len(tokens) + 1 > metadata_limit:
            return []
        begin = len(ids)
        ids.extend(tokens + [tokenizer.sep_token_id])
        refs.extend([None]*(len(tokens)+1))
        return list(range(begin, begin+len(tokens)))
    resume_tasks = tuple(reversed(tuple({state.task_id: state for state in memory.prior_task_states}.values())))
    input_omissions = []
    observation = {}
    if memory.source_index is not None and hasattr(memory.source_index, "reference_date"):
        observation["reference_date"] = memory.source_index.reference_date
    if memory.source_index is not None and hasattr(memory.source_index, "observed_source_scope"):
        observation["source_scope"] = [str(key) for key in memory.source_index.observed_source_scope]
    if observation and not metadata_positions(observation):
        input_omissions.append("source date/scope metadata exceeds the task metadata budget")
    task_positions = [metadata_positions({"task_id": state.task_id, "question": state.raw_question,
        "turn_id": state.current_turn_id}) for state in resume_tasks]
    retained_tasks = [(state, indices) for state, indices in zip(resume_tasks, task_positions) if indices]
    resume_tasks = tuple(state for state, _ in retained_tasks)
    task_positions = [indices for _, indices in retained_tasks]
    version_turn_ids = [row.turn_id for row in memory.committed_history]
    version_turn_positions = [metadata_positions({"turn_id": turn_id,
        "version_intents": [{"mode": state.version_intent.mode, "document_families": state.version_intent.document_families,
                            "version_ids": state.version_intent.version_ids, "resolved_version_ids": state.version_intent.resolved_version_ids} for state in memory.prior_task_states
                            if state.current_turn_id == turn_id]}) for turn_id in version_turn_ids]
    retained_turns = [(identity, indices) for identity, indices in zip(version_turn_ids, version_turn_positions) if indices]
    version_turn_ids = [identity for identity, _ in retained_turns]
    version_turn_positions = [indices for _, indices in retained_turns]
    ranges = []
    # Historical anchors and retrieved windows are optional under the same budget.
    groups.extend((window.ref, window.text) for window in windows)
    included_groups = []
    for group_index, (ref, text) in enumerate(groups):
        prefix = tokenizer(f"Turn {ref.turn_id} {ref.speaker}: ", add_special_tokens=False)["input_ids"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        size = len(prefix) + len(encoded["input_ids"]) + 1
        if len(ids) + size > max_input_tokens:
            if group_index == 0:
                raise ValueError("current question exceeds the encoder input budget")
            continue
        included_groups.append((ref, text))
        ids.extend(prefix)
        refs.extend([None]*len(prefix))
        begin = len(ids)
        ids.extend(encoded["input_ids"])
        refs.extend(TextRef(ref.turn_id, ref.speaker, ref.start+start, ref.start+end) if start < end else None
                    for start, end in encoded["offset_mapping"])
        ranges.append((begin, len(ids), ref))
        ids.append(tokenizer.sep_token_id)
        refs.append(None)
    def visible(spans):
        return all(any(owner.turn_id == span.turn_id and owner.speaker == span.speaker
                       and owner.start <= span.start and span.end <= owner.end
                       for owner, _ in included_groups) for span in spans)
    prior_tasks = tuple(replace(state,
        needs=tuple(n for n in state.needs if visible(n.anchor_spans)),
        conditions=tuple(c for c in state.conditions if visible(c.value_spans +
            ((c.key_span,) if c.key_span else ()) + ((c.predicate_span,) if c.predicate_span else ()))))
        for state in memory.prior_task_states if state.task_id in {s.task_id for s in resume_tasks})
    prior_conditions = tuple(condition for state in prior_tasks for condition in state.conditions)
    prior_conditions = tuple({condition.condition_id: condition for condition in prior_conditions}.values())

    def positions(spans):
        return [i for i, token in enumerate(refs) if token is not None and any(
            token.turn_id == span.turn_id and token.speaker == span.speaker
            and token.start < span.end and span.start < token.end for span in spans)]

    condition_positions = [positions(condition.value_spans + ((condition.key_span,) if condition.key_span else ())
        + ((condition.predicate_span,) if condition.predicate_span else ())) for condition in prior_conditions]
    return {"input_omissions": input_omissions, "input_ids": ids, "refs": refs, "ranges": ranges, "text_groups": included_groups,
            "catalogue_versions": versions, "catalogue_families": families, "catalogue_positions": catalogue_positions,
            "current_turn_id": turn.turn,
            "prior_conditions": prior_conditions, "condition_positions": condition_positions,
            "prior_tasks": prior_tasks, "resume_tasks": resume_tasks, "task_positions": task_positions,
            "version_turn_ids": version_turn_ids, "version_turn_positions": version_turn_positions}


class TaskInputNotVisible(ValueError):
    """Supervision refers to candidates excluded by the observed input budget."""


def task_targets(encoded, prediction):
    if encoded.get("input_omissions"):
        raise TaskInputNotVisible("; ".join(encoded["input_omissions"]))
    refs = [*prediction.history_spans, *prediction.version_intent.time_refs]
    refs.extend(ref for n in prediction.needs for ref in (*n.anchor_spans, *n.subject_refs))
    refs.extend(ref for c in prediction.conditions for ref in (*c.value_spans,
        *((c.key_span,) if c.key_span else ()), *((c.predicate_span,) if c.predicate_span else ())))
    if any(not any(owner.turn_id == ref.turn_id and owner.speaker == ref.speaker
                   and owner.start <= ref.start and ref.end <= owner.end
                   for owner, _ in encoded["text_groups"]) for ref in refs):
        raise TaskInputNotVisible("supervised span is outside the budgeted task input")
    task_ids = {s.task_id for s in encoded["resume_tasks"]}
    if prediction.resume_task_id and prediction.resume_task_id not in task_ids:
        raise TaskInputNotVisible("supervised task pointer is outside the budgeted task input")
    items = candidates(encoded, prediction.needs, prediction.conditions,
        tuple(ref for n in prediction.needs for ref in n.subject_refs),
        relation=prediction.relation, resume_task_id=prediction.resume_task_id)
    if any(not set(n.condition_ids) <= {c.condition_id for c in items["conditions"]}
           or not set(n.predecessor_need_ids) <= {p.need_id for p in items["predecessors"]}
           for n in prediction.needs):
        raise TaskInputNotVisible("supervised graph edge is outside the budgeted task input")
    validate_task_anchors(prediction)
    n = len(encoded["input_ids"])
    targets = {name: [-100]*n for name in ("span", "history", "update", "condition_relation", "operation")}
    for name in ("condition_target", "condition_supersedes"):
        targets[name] = [[-100]*len(encoded["prior_conditions"]) for _ in range(n)]
    targets["source_turns"] = [int(identity in prediction.version_intent.source_turns) for identity in encoded["version_turn_ids"]]
    if not set(prediction.version_intent.source_turns) <= set(encoded["version_turn_ids"]):
        raise TaskInputNotVisible("version source turn is outside the budgeted task input")
    targets["span"] = [[0 if ref is not None else -100 for _ in SPAN_KINDS] for ref in encoded["refs"]]

    def assign(refs, kind, values=None):
        for span in refs:
            first = True
            for i, ref in enumerate(encoded["refs"]):
                if ref is None or ref.turn_id != span.turn_id or ref.speaker != span.speaker:
                    continue
                if ref.start < span.end and span.start < ref.end:
                    targets["span"][i][SPAN_KINDS.index(kind)] = 1 if first else 2
                    first = False
                    for name, value in (values or {}).items():
                        targets[name][i] = value

    for i, ref in enumerate(encoded["refs"]):
        if ref is not None and ref.turn_id != encoded["current_turn_id"]:
            targets["history"][i] = int(any(ref.turn_id == chosen.turn_id and ref.speaker == chosen.speaker
                and ref.start < chosen.end and chosen.start < ref.end for chosen in prediction.history_spans))
    for need in prediction.needs:
        assign(need.anchor_spans, "NEED", {"operation": OPERATIONS.index(need.operation)})
        assign(need.subject_refs, "SUBJECT")
    base_conditions, _ = task_graph_history(encoded["prior_tasks"], prediction.relation, prediction.resume_task_id)
    base_condition_ids = {c.condition_id for c in base_conditions}
    condition_ids = [c.condition_id for c in encoded["prior_conditions"]]
    for condition in prediction.conditions:
        if not set((*condition.target_refs, *condition.supersedes)) <= base_condition_ids:
            raise TaskInputNotVisible("condition update target is outside the budgeted task input")
        values = {"update": UPDATES.index(condition.update),
                  "condition_relation": CONDITION_RELATIONS.index(condition.relation_type),
                  "condition_target": [int(identity in condition.target_refs) if identity in base_condition_ids else -100 for identity in condition_ids],
                  "condition_supersedes": [int(identity in condition.supersedes) if identity in base_condition_ids else -100 for identity in condition_ids]}
        if condition.key_span is not None:
            assign((condition.key_span,), "CONDITION_KEY", values)
        if condition.predicate_span is not None:
            assign((condition.predicate_span,), "CONDITION_PREDICATE", values)
        assign(condition.value_spans, "CONDITION_VALUE", values)
    assign(prediction.version_intent.time_refs, "TIME")
    tasks = [state.task_id for state in encoded["resume_tasks"]]
    targets.update(relation=RELATIONS.index(prediction.relation), version=VERSIONS.index(prediction.version_intent.mode),
                   temporal_relation=TEMPORAL_RELATIONS.index(prediction.version_intent.temporal_relation),
                   resume=tasks.index(prediction.resume_task_id)+1 if prediction.resume_task_id else 0)
    for name, identities in (("document_families", encoded["catalogue_families"]),
                             ("version_ids", encoded["catalogue_versions"]),
                             ("resolved_version_ids", encoded["catalogue_versions"])):
        selected = getattr(prediction.version_intent, name)
        if not set(selected) <= set(identities):
            raise TaskInputNotVisible(f"{name} supervision is outside the budgeted observed catalogue")
        targets[name] = [int(identity in selected) for identity in identities]
    return targets


def create_task_network(base_path, *, adapter_path=None, rank=16, device="cuda", dtype="bfloat16"):
    import torch
    from torch import nn
    from transformers import AutoModel
    from peft import LoraConfig, PeftModel, get_peft_model
    base = AutoModel.from_pretrained(base_path, local_files_only=True, dtype=getattr(torch, dtype))
    encoder = (PeftModel.from_pretrained(base, str(Path(adapter_path)/"adapter"), is_trainable=True)
               if adapter_path else get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank*2,
                   lora_dropout=.05, target_modules=["query", "value"], task_type="FEATURE_EXTRACTION")))

    class TaskNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            width = base.config.hidden_size
            self.heads = nn.ModuleDict({name: nn.Linear(width, size) for name, size in {
                "span": len(SPAN_KINDS)*3, "history": 2, "condition_update": len(UPDATES),
                "condition_relation": len(CONDITION_RELATIONS), "operation": len(OPERATIONS),
                "relation": len(RELATIONS), "version": len(VERSIONS),
                "temporal_relation": len(TEMPORAL_RELATIONS)}.items()})
            self.graph_queries = nn.ModuleDict({name: nn.Linear(width, 64) for name in
                ("need_subject", "need_condition", "need_predecessor")})
            self.graph_keys = nn.ModuleDict({name: nn.Linear(width, 64) for name in self.graph_queries})
            self.catalogue_queries = nn.ModuleDict({name: nn.Linear(width, 64) for name in
                ("document_families", "version_ids", "resolved_version_ids")})
            self.catalogue_keys = nn.ModuleDict({name: nn.Linear(width, 64) for name in self.catalogue_queries})
            self.group_queries = nn.ModuleDict({name: nn.Linear(width, 64) for name in ("need_group", "condition_group")})
            self.group_keys = nn.ModuleDict({name: nn.Linear(width, 64) for name in self.group_queries})
            self.update_queries = nn.ModuleDict({name: nn.Linear(width, 64) for name in ("condition_target", "condition_supersedes", "source_turns")})
            self.update_keys = nn.ModuleDict({name: nn.Linear(width, 64) for name in self.update_queries})
            self.scope_head = nn.Linear(width, len(SCOPES))
            self.pointer_query = nn.Linear(width, width)
            self.pointer_key = nn.Linear(width, width)
            self.null_task = nn.Parameter(torch.zeros(width))

        def graph_scores(self, hidden, encoded, items):
            def pool(refs):
                indices = positions(encoded, refs)
                if not indices:
                    raise ValueError("graph candidate has no observed token anchors")
                return hidden[0, indices].mean(0)
            def matrix(ref_groups):
                return torch.stack([pool(refs) for refs in ref_groups]) if ref_groups else hidden.new_empty((0, hidden.shape[-1]))
            need = matrix([n.anchor_spans for n in items["needs"]])
            objects = {
                "need_subject": matrix([(ref,) for ref in items["subjects"]]),
                "need_condition": matrix([c.value_spans + ((c.key_span,) if c.key_span else ()) +
                    ((c.predicate_span,) if c.predicate_span else ()) for c in items["conditions"]]),
                "need_predecessor": matrix([n.anchor_spans for n in items["predecessors"]])}
            scores = {name: self.graph_queries[name](need) @ self.graph_keys[name](value).T / 8
                      for name, value in objects.items()}
            scores["scope"] = self.scope_head(need)
            return scores

        def group_scores(self, hidden, encoded, name, spans):
            vectors = (torch.stack([hidden[0, positions(encoded, (ref,))].mean(0) for ref in spans])
                       if spans else hidden.new_empty((0, hidden.shape[-1])))
            return self.group_queries[name](vectors) @ self.group_keys[name](vectors).T / 8

        def forward(self, batch, encoded):
            hidden = self.encoder(**batch).last_hidden_state.float()
            output = {name: head(hidden[:, 0] if name in {"relation", "version", "temporal_relation"} else hidden)
                      for name, head in self.heads.items()}
            output["update"] = output.pop("condition_update")
            output["query"] = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
            condition_keys = (torch.stack([hidden[0, indices].mean(0) for indices in encoded["condition_positions"]])
                              if encoded["condition_positions"] else hidden.new_empty((0, hidden.shape[-1])))
            for name in ("condition_target", "condition_supersedes"):
                output[name] = self.update_queries[name](hidden) @ self.update_keys[name](condition_keys).T / 8
            task_keys = [self.null_task]
            task_keys.extend(hidden[0, indices].mean(0) for indices in encoded["task_positions"])
            output["resume"] = self.pointer_query(hidden[:, 0]) @ self.pointer_key(torch.stack(task_keys)).T
            turn_keys = (torch.stack([hidden[0, indices].mean(0) for indices in encoded["version_turn_positions"]])
                         if encoded["version_turn_positions"] else hidden.new_empty((0, hidden.shape[-1])))
            output["source_turns"] = (self.update_queries["source_turns"](hidden[:, 0]) @ self.update_keys["source_turns"](turn_keys).T)[0] / 8
            output["group_scores"] = lambda name, spans: self.group_scores(hidden, encoded, name, spans)
            for name in self.catalogue_queries:
                positions_ = encoded["catalogue_positions"]["document_families" if name == "document_families" else "version_ids"]
                keys = (torch.stack([hidden[0, indices].mean(0) for indices in positions_]) if positions_
                        else hidden.new_empty((0, hidden.shape[-1])))
                output[name] = (self.catalogue_queries[name](hidden[:, 0]) @ self.catalogue_keys[name](keys).T)[0] / 8
            output["graph_scores"] = lambda items: self.graph_scores(hidden, encoded, items)
            return output

    return TaskNetwork().to(device)


def decode_prediction(encoded, output, turn):
    labels = output["span"][0].reshape(-1, len(SPAN_KINDS), 3).argmax(-1).tolist()
    spans = []
    for channel, kind in enumerate(SPAN_KINDS):
        active = None
        for i, (token_labels, ref) in enumerate(zip(labels, encoded["refs"], strict=True)):
            label = token_labels[channel]
            if ref is None or label == 0:
                active = None
                continue
            if (label == 1 or active is None or active[1].turn_id != ref.turn_id
                    or active[1].speaker != ref.speaker):
                spans.append([kind, ref, i])
                active = spans[-1]
            else:
                active[1] = TextRef(ref.turn_id, ref.speaker, active[1].start, ref.end)
    # Score-bounded proposals follow the teacher's 8 objects / 16 refs contract.
    # Retain highest BIO confidence before constructing quadratic pair matrices.
    limits = {"NEED": TASK_MAX_OBJECTS*TASK_MAX_REFS, "SUBJECT": TASK_MAX_OBJECTS*TASK_MAX_REFS,
              "CONDITION_KEY": TASK_MAX_OBJECTS, "CONDITION_PREDICATE": TASK_MAX_OBJECTS+len(encoded["prior_conditions"]),
              "CONDITION_VALUE": TASK_MAX_OBJECTS*TASK_MAX_REFS, "TIME": TASK_MAX_REFS}
    def confidence(item):
        kind, ref, _ = item
        indices = positions(encoded, (ref,))
        channel = SPAN_KINDS.index(kind)
        logits = output["span"][0].reshape(-1, len(SPAN_KINDS), 3)[indices, channel]
        positive = logits[:, 1:].max(-1)
        positive = positive.values if hasattr(positive, "values") else positive
        return float((positive - logits[:, 0]).mean())
    selected_spans = []
    for kind, limit in limits.items():
        unique = {(item[0], item[1]): item for item in spans if item[0] == kind}
        selected_spans.extend(sorted(unique.values(), key=confidence, reverse=True)[:limit])
    spans = selected_spans
    allowed_relations = range(len(RELATIONS)) if encoded["resume_tasks"] else (0, 4)
    relation = RELATIONS[max(allowed_relations, key=lambda i: float(output["relation"][0, i]))]
    resume = (max(range(1, len(encoded["resume_tasks"])+1), key=lambda i: float(output["resume"][0, i]))
              if relation in {"followup", "correction", "revisit"} else 0)
    resume_id = encoded["resume_tasks"][resume-1].task_id if resume else ""
    base_conditions, _ = task_graph_history(encoded["prior_tasks"], relation, resume_id)
    base_condition_ids = {c.condition_id for c in base_conditions}
    version = VERSIONS[int(output["version"][0].argmax())]
    temporal = TEMPORAL_RELATIONS[int(output["temporal_relation"][0].argmax())]
    history_spans = []
    previous_selected = False
    for i, ref in enumerate(encoded["refs"]):
        if ref is None or ref.turn_id == turn.turn or int(output["history"][0, i].argmax()) == 0:
            previous_selected = False
            continue
        if previous_selected and history_spans and history_spans[-1].turn_id == ref.turn_id and history_spans[-1].speaker == ref.speaker and history_spans[-1].end <= ref.start:
            previous = history_spans[-1]
            history_spans[-1] = TextRef(ref.turn_id, ref.speaker, previous.start, ref.end)
        else:
            history_spans.append(ref)
        previous_selected = True
    def history_confidence(ref):
        logits = output["history"][0, positions(encoded, (ref,))]
        return float((logits[:, 1] - logits[:, 0]).mean())
    history = tuple(sorted(dict.fromkeys(history_spans), key=history_confidence, reverse=True)[:TASK_MAX_REFS])
    subjects = tuple(ref for kind, ref, _ in spans if kind == "SUBJECT")
    condition_spans = [(kind, ref, i) for kind, ref, i in spans if kind.startswith("CONDITION_")]
    condition_refs = tuple(dict.fromkeys(ref for _, ref, _ in condition_spans))
    conditions = []
    def compatible_condition(group):
        refs_ = {condition_refs[index] for index in group}
        return all(sum(ref in refs_ and role == kind for role, ref, _ in condition_spans) <= 1
                   for kind in ("CONDITION_KEY", "CONDITION_PREDICATE")) and sum(
                       ref in refs_ and role == "CONDITION_VALUE" for role, ref, _ in condition_spans) <= TASK_MAX_REFS
    condition_groups = grouped_indices(output["group_scores"]("condition_group", condition_refs), len(condition_refs),
        max_size=TASK_MAX_REFS+2, compatible=compatible_condition)
    proposals = []
    for group in condition_groups:
        grouped_refs = {condition_refs[index] for index in group}
        members = [item for item in condition_spans if item[1] in grouped_refs]
        indices = positions(encoded, tuple(ref for _, ref, _ in members))
        update_logits = output["update"][0, indices].mean(0)
        allowed_updates = ["ADD", "COMPARE"]
        if base_conditions:
            allowed_updates.append("REPLACE")
            if len(members) == 1 and members[0][0] == "CONDITION_PREDICATE":
                allowed_updates.append("REMOVE")
        update = max(allowed_updates, key=lambda name: float(update_logits[UPDATES.index(name)]))
        relation_type = CONDITION_RELATIONS[int(output["condition_relation"][0, indices].mean(0).argmax())]
        links = {name: tuple(encoded["prior_conditions"][j].condition_id for j in
                    sorted((j for j, c in enumerate(encoded["prior_conditions"]) if c.condition_id in base_condition_ids), key=lambda j: float(output[head][0, indices, j].mean()), reverse=True)[:TASK_MAX_REFS]
                    if output[head][0, indices, j].mean() > 0)
                 for name, head in (("target_refs", "condition_target"), ("supersedes", "condition_supersedes"))}
        keys = [ref for kind, ref, _ in members if kind == "CONDITION_KEY"]
        predicates = [ref for kind, ref, _ in members if kind == "CONDITION_PREDICATE"]
        values = tuple(ref for kind, ref, _ in members if kind == "CONDITION_VALUE")
        update_index = UPDATES.index(update)
        # Rank object proposals using their own update margin plus span confidence.
        other = [float(value) for j, value in enumerate(update_logits) if j != update_index]
        score = float(update_logits[update_index]) - max(other) + sum(confidence(item) for item in members)/len(members)
        proposals.append((score, Condition("", keys[0] if keys else None,
            values, "PREDICATE" if update == "REMOVE" else relation_type, predicates[0] if predicates else None,
            links["target_refs"], () if update == "REMOVE" else links["supersedes"], update)))
    from dataclasses import replace
    retained = []
    for removal, limit in ((False, TASK_MAX_OBJECTS), (True, len(base_conditions))):
        retained.extend(sorted((item for item in proposals if (item[1].update == "REMOVE") == removal),
                               key=lambda item: item[0], reverse=True)[:limit])
    conditions = [replace(condition, condition_id=f"t{turn.turn}:c{i}") for i, (_, condition) in enumerate(retained)]
    need_refs = tuple(dict.fromkeys(ref for kind, ref, _ in spans if kind == "NEED"))
    needs = tuple(Need(f"t{turn.turn}:n{j}", tuple(need_refs[index] for index in group),
        OPERATIONS[int(output["operation"][0, positions(encoded, tuple(need_refs[index] for index in group))].mean(0).argmax())])
        for j, group in enumerate(grouped_indices(output["group_scores"]("need_group", need_refs), len(need_refs),
            max_size=TASK_MAX_REFS)[:TASK_MAX_OBJECTS]))
    items = candidates(encoded, needs, conditions, subjects, relation=relation, resume_task_id=resume_id)
    needs = apply_edges(items, output["graph_scores"](items))
    times = tuple(ref for kind, ref, _ in spans if kind == "TIME")
    def choose(identities, scores, limit=TASK_MAX_REFS):
        ranked = sorted(range(len(identities)), key=lambda i: float(scores[i]), reverse=True)[:limit]
        selected = {i for i in ranked if scores[i] > 0}
        return tuple(identity for i, identity in enumerate(identities) if i in selected)
    catalogue_selected = {name: choose(
        encoded["catalogue_families"] if name == "document_families" else encoded["catalogue_versions"], output[name])
        for name in ("document_families", "version_ids", "resolved_version_ids")}
    source_turns = choose(encoded["version_turn_ids"], output["source_turns"], TASK_MAX_VERSION_TURNS)
    return TaskPrediction(relation, history, tuple(conditions), needs, VersionIntent(version,
        **catalogue_selected, time_refs=times, source_turns=source_turns, temporal_relation=temporal), resume_id)


class TaskModel:
    def __init__(self, path, *, base_path=None, device="cuda"):
        import torch
        from transformers import AutoTokenizer
        path = Path(path)
        self.manifest = json.loads((path/"task-model.json").read_text())
        if self.manifest["schema_version"] != 3 or self.manifest["graph_format"] != FORMAT or self.manifest["need_scopes"] != list(SCOPES):
            raise ValueError("task model does not implement the current owned Need graph format")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_path or self.manifest["base_path"], local_files_only=True)
        self.network = create_task_network(base_path or self.manifest["base_path"], rank=self.manifest["rank"],
            adapter_path=path, device=device, dtype=self.manifest["dtype"])
        weights = torch.load(path/"task-heads.pt", map_location=device, weights_only=True)
        expected = {name for name in self.network.state_dict() if not name.startswith("encoder.")}
        if set(weights) != expected:
            raise ValueError("task head checkpoint keys do not match the declared graph format")
        state = self.network.state_dict()
        state.update(weights)
        self.network.load_state_dict(state, strict=True)
        self.network.requires_grad_(False)
        self.network.eval()
        self.model = self.network.encoder
        self.max_length = self.manifest["max_input_tokens"]
        self.texts_processed = 0
        self.token_limit_hits = 0
        self.usage = {"tasks": 0, "queries": 0, "input_tokens": 0, "output_tokens": 0,
                      "elapsed_ms": 0.0, "task_ms": 0.0, "query_encode_ms": 0.0}

    def _synchronize(self):
        import torch
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def encode(self, texts):
        """Query-side encoding for the existing frozen DenseIndex space."""
        import numpy as np
        import torch
        self._synchronize()
        started = time.perf_counter()
        vectors = []
        for text in texts:
            self.texts_processed += 1
            encoded = self.tokenizer(text, return_tensors="pt", truncation=False).to(self.device)
            if encoded["input_ids"].shape[1] > self.max_length:
                raise ValueError("task retrieval query exceeds BGE-M3 input budget")
            self.usage["input_tokens"] += int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                hidden = self.network.encoder(**encoded).last_hidden_state[:, 0].float()
                vectors.append(torch.nn.functional.normalize(hidden, dim=-1)[0].cpu().numpy())
        self._synchronize()
        elapsed = (time.perf_counter()-started)*1000
        self.usage["queries"] += len(vectors)
        self.usage["elapsed_ms"] += elapsed
        self.usage["query_encode_ms"] += elapsed
        return np.asarray(vectors)

    def close(self):
        self.network.to("cpu")

    def predict(self, turn, windows, memory, *, catalogue=None):
        return self.predict_result(turn, windows, memory, catalogue=catalogue).prediction

    def predict_result(self, turn, windows, memory, *, catalogue=None):
        import torch
        self._synchronize()
        started = time.perf_counter()
        encoded = encode_task(self.tokenizer, turn, windows, memory, catalogue=catalogue, max_input_tokens=self.max_length)
        if len(encoded["input_ids"]) > self.manifest["max_input_tokens"]:
            raise ValueError("task view exceeds trained encoder budget")
        batch = {"input_ids": torch.tensor([encoded["input_ids"]], device=self.device),
                 "attention_mask": torch.ones((1, len(encoded["input_ids"])), dtype=torch.long, device=self.device)}
        with torch.inference_mode():
            output = self.network(batch, encoded)
        prediction = decode_prediction(encoded, output, turn)
        self._synchronize()
        elapsed = (time.perf_counter()-started)*1000
        self.usage["tasks"] += 1
        self.usage["input_tokens"] += len(encoded["input_ids"])
        self.usage["elapsed_ms"] += elapsed
        self.usage["task_ms"] += elapsed
        metrics = {"input_tokens": len(encoded["input_ids"]), "output_tokens": 0, "elapsed_ms": elapsed,
            "input_omissions": encoded["input_omissions"]}
        return TaskModelResult(prediction, metrics)


def train_task(config):
    complete_started = time.perf_counter()
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from adapters.hybrid_retrieval import DenseIndex, LocalBGE
    from adapters.doc2dial import _identity
    from training.reader_data import file_digest, write_rows
    settings = config.get("training", {})
    task = config.get("task_model", {})
    base_path = task.get("base_path", "models/bge-m3")
    device, dtype = settings.get("device", "cuda"), settings.get("dtype", "bfloat16")
    output = Path(settings.get("task_output", "models/study-task"))
    labels_path = Path(settings.get("task_labels", "runs/study/data/task_labels.jsonl"))
    torch.manual_seed(settings.get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    initial_adapter = task.get("initial_adapter_path", "models/bge-m3-dialogue-query-v1")
    network = create_task_network(base_path, adapter_path=initial_adapter,
                                  rank=task.get("lora_rank", 16), device=device, dtype=dtype)
    frozen_document = LocalBGE(base_path, device=device, dtype=dtype, max_length=8192)
    frozen_document.model.requires_grad_(False)
    dense_path = Path(settings["task_dense_index"])
    dense = DenseIndex.load(dense_path)
    document_vectors = torch.tensor(dense.vectors, dtype=torch.float32, device=device)
    sections = {}
    for index, context in enumerate(dense.contexts):
        domain, document_id, _, start, end = json.loads(context.document_id)
        sections.setdefault(_identity(domain, document_id), []).append((start, end, index))
    labels_paths = [Path(path) for path in settings.get("task_label_paths", (labels_path,))]
    labels = select_cohort_rows((row for path in labels_paths for row in read_jsonl(path)),
                               settings["cohort_manifest"], "train")
    if not labels:
        raise ValueError("task training requires actual-history teacher labels")
    retrieval_targets, retrieval_alignment = {}, []
    for row in labels:
        positives, coverage = set(), []
        for gold in row["verified_grounding"]:
            matches = [(left, right, index) for left, right, index in sections.get(gold["passage_id"], ())
                       if left < gold["end"] and gold["start"] < right]
            positives.update(index for _, _, index in matches)
            covered, cursor = 0, gold["start"]
            for left, right, _ in sorted(matches):
                left, right = max(left, gold["start"]), min(right, gold["end"])
                covered += max(0, right-max(left, cursor))
                cursor = max(cursor, right)
            coverage.append({"passage_id": gold["passage_id"], "start": gold["start"], "end": gold["end"],
                "overlapping_section_ids": [dense.contexts[index].document_id for _, _, index in matches],
                "covered_characters": covered, "gold_characters": gold["end"]-gold["start"],
                "fully_covered": covered == gold["end"]-gold["start"]})
        retrieval_targets[row["task_id"]] = sorted(positives)
        retrieval_alignment.append({"task_id": row["task_id"], "cohort_split": row["cohort_split"],
            "positive_section_ids": [dense.contexts[index].document_id for index in sorted(positives)],
            "query_supervised": bool(positives), "gold_spans": coverage,
            "all_gold_spans_fully_covered": bool(coverage) and all(span["fully_covered"] for span in coverage)})
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"retrieval_alignment.jsonl", retrieval_alignment)
    conversations = {}
    for row in labels:
        conversations.setdefault(row["conversation_id"], []).append(row)
    for session in conversations.values():
        session.sort(key=lambda row: row["turn"])
    conversation_ids = list(conversations)
    order = random.Random(settings.get("seed", 42))
    loss_weights = {name: float(settings.get("task_loss_weights", {}).get(name, 1.0))
        for name in ("retrieval", "history", "span", "update", "condition_relation", "condition_target", "condition_supersedes",
                     "operation", "relation", "version", "temporal_relation", "source_turns", "resume",
                     "need_subject", "need_condition", "need_predecessor", "scope",
                     "document_families", "version_ids", "resolved_version_ids", "need_group", "condition_group")}
    temperature = settings.get("retrieval_temperature", .05)
    adapter_params, head_params = [], []
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            (adapter_params if name.startswith("encoder.") else head_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": settings.get("adapter_lr", 2e-4)},
        {"params": head_params, "lr": settings.get("head_lr", 1e-4)}], weight_decay=.01)
    logs = []
    input_not_visible = []
    span_counts = {kind: Counter() for kind in SPAN_KINDS}
    training_usage = {"task_input_tokens": 0, "history_input_tokens": 0,
                      "document_input_tokens": 0, "query_input_tokens": 0}
    started = time.perf_counter()
    for epoch in range(settings.get("epochs", 1)):
        prior_states = {}
        order.shuffle(conversation_ids)
        for row in (row for identity in conversation_ids for row in conversations[identity]):
            history = tuple(decode(TurnRecord, value) for value in row["actual_history"])
            memory = RetrievalState(len(history), history, tuple(prior_states.get(row["conversation_id"], ())),
                source_index=observed_task_sources(row))
            turn = OnlineTurn(row["conversation_id"], row["task_id"], row["turn"], row["snapshot_id"], row["question"], (), (), "P3")
            # Ranking here uses the frozen document encoder, keeping its space fixed.
            def encode(texts):
                batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(device)
                training_usage["history_input_tokens"] += int(batch["attention_mask"].sum())
                with torch.no_grad():
                    return F.normalize(frozen_document.model(**batch).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
            index = HistoryIndex(tokenizer, encode, unit_tokens=task.get("history_unit_tokens", 384),
                                 candidates=task.get("history_candidates", 16), view_tokens=8192)
            windows = index.select(turn.query, memory, current_turn_id=turn.turn)
            encoded = encode_task(tokenizer, turn, windows, memory, catalogue=row["task_catalogue"])
            if len(encoded["input_ids"]) > 8192:
                raise ValueError("task training view exceeds BGE-M3 input budget")
            prediction = decode(TaskPrediction, row["prediction"])
            try:
                targets = task_targets(encoded, prediction)
            except TaskInputNotVisible as error:
                input_not_visible.append({"task_id": row["task_id"], "epoch": epoch, "reason": str(error)})
                from controller.context_schema import TaskState
                prior_states.setdefault(row["conversation_id"], []).append(decode(TaskState, row["task_state"]))
                continue
            batch = {"input_ids": torch.tensor([encoded["input_ids"]], device=device),
                     "attention_mask": torch.ones((1, len(encoded["input_ids"])), dtype=torch.long, device=device)}
            network.train()
            training_usage["task_input_tokens"] += len(encoded["input_ids"])
            scores = network(batch, encoded)
            losses = {}
            for name, values in targets.items():
                target = torch.tensor(values, device=device)
                if target.ndim == 0:
                    target = target.unsqueeze(0)
                elif not (target != -100).any():
                    continue
                if name in {"document_families", "version_ids", "resolved_version_ids", "source_turns", "condition_target", "condition_supersedes"}:
                    if target.numel():
                        valid = target != -100
                        if valid.any():
                            losses[name] = F.binary_cross_entropy_with_logits(scores[name].reshape(target.shape)[valid], target.float()[valid])
                    continue
                if name == "span":
                    token_losses = F.cross_entropy(scores[name].reshape(-1, 3), target.reshape(-1),
                                                  reduction="none").reshape(target.shape)
                    channel_losses = []
                    for channel, kind in enumerate(SPAN_KINDS):
                        span_counts[kind].update(value[channel] for value in values if value[channel] != -100)
                        channel_losses.append(torch.stack([
                            token_losses[:, channel][target[:, channel] == label].mean()
                            for label in (0, 1, 2) if (target[:, channel] == label).any()]).mean())
                    losses[name] = torch.stack(channel_losses).mean()
                else:
                    losses[name] = F.cross_entropy(scores[name].reshape(-1, scores[name].shape[-1]), target.reshape(-1))
            grouping_objects = {
                "need_group": [need.anchor_spans for need in prediction.needs],
                "condition_group": [condition.value_spans + ((condition.key_span,) if condition.key_span else ()) +
                    ((condition.predicate_span,) if condition.predicate_span else ()) for condition in prediction.conditions]}
            for name, objects in grouping_objects.items():
                spans_, labels_ = grouping_targets(objects)
                if len(spans_) > 1:
                    logits = scores["group_scores"](name, spans_)
                    target = torch.tensor(labels_, device=device, dtype=logits.dtype)
                    mask = ~torch.eye(len(spans_), device=device, dtype=torch.bool)
                    losses[name] = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
            items = candidates(encoded, prediction.needs, prediction.conditions,
                               tuple(ref for n in prediction.needs for ref in n.subject_refs),
                               relation=prediction.relation, resume_task_id=prediction.resume_task_id)
            graph_scores = scores["graph_scores"](items)
            for name, values in relation_targets(items).items():
                logits = graph_scores[name]
                if not logits.numel():
                    continue
                target = torch.tensor(values, device=device)
                losses[name] = (F.cross_entropy(logits, target.long()) if name == "scope"
                                else F.binary_cross_entropy_with_logits(logits, target.float()))
            positives = retrieval_targets[row["task_id"]]
            if positives:
                query_batch = tokenizer(row["task_state"]["query"], return_tensors="pt", truncation=False).to(device)
                training_usage["query_input_tokens"] += int(query_batch["input_ids"].shape[1])
                query_hidden = network.encoder(**query_batch).last_hidden_state[:, 0].float()
                query = F.normalize(query_hidden, dim=-1)
                logits = query @ document_vectors.T / temperature
                losses["retrieval"] = -F.log_softmax(logits, dim=-1)[0, positives].mean()
            total = sum(loss_weights[name]*value for name, value in losses.items())
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            logs.append({"epoch": epoch, "task_id": row["task_id"], "loss": float(total.detach()),
                         "components": {name: float(value.detach()) for name, value in losses.items()}})
            from controller.context_schema import TaskState
            prior_states.setdefault(row["conversation_id"], []).append(decode(TaskState, row["task_state"]))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"input_not_visible.jsonl", input_not_visible)
    if not logs:
        status = {"status": "not_fitted", "reason": "no visible supervised training steps",
            "declared_training_rows": len(labels), "training_rows": 0, "steps": 0,
            "input_not_visible_states": len(input_not_visible)}
        (output/"training-status.json").write_text(json.dumps(status, indent=2)+"\n")
        return status
    network.encoder.save_pretrained(output/"adapter")
    torch.save({name: value.cpu() for name, value in network.state_dict().items()
                if not name.startswith("encoder.")}, output/"task-heads.pt")
    manifest = {"schema_version": 3, "graph_format": FORMAT, "need_scopes": SCOPES, "status": "completed", "base": "BAAI/bge-m3", "base_path": base_path,
        "model_id": "contextkv/task-query-lora", "revision": "study-task-v3",
        "pooling": "CLS then FP32 L2 normalization", "max_length": 8192,
        "document_encoder": {name: dense.manifest["encoder"][name] for name in (
            "model_id", "revision", "max_length", "pooling")},
        "document_encoder_manifest": dense.manifest["encoder"], "document_encoder_training": "frozen",
        "rank": task.get("lora_rank", 16), "dtype": dtype,
        "declared_training_rows": len(labels),
        "training_rows": len({row["task_id"] for row in logs}), "steps": len(logs), "elapsed_ms": (time.perf_counter()-started)*1000,
        "complete_ms": (time.perf_counter()-complete_started)*1000,
        "model_setup_ms": (started-complete_started)*1000,
        "input_tokens": sum(training_usage.values()), "token_usage": training_usage,
        "initial_adapter_path": initial_adapter,
        "labels_paths": [str(path) for path in labels_paths], "max_input_tokens": 8192,
        "retrieval_index": {"path": str(dense_path.resolve()), "sections": len(dense.contexts),
            "manifest_sha256": file_digest(dense_path/"manifest.json"),
            "passages_sha256": file_digest(dense_path/"passages.jsonl"),
            "vectors_sha256": file_digest(dense_path/"vectors.npy"),
            "positive_rule": "same original document and nonempty half-open character intersection with verified gold",
            "loss": "uniform positive-section cross entropy against every frozen index row",
            "alignment_report": str((output/"retrieval_alignment.jsonl").resolve()),
            "supervised_states": sum(row["query_supervised"] for row in retrieval_alignment),
            "fully_covered_states": sum(row["all_gold_spans_fully_covered"] for row in retrieval_alignment)},
        "gradient_scope": {"query_lora": "retrieval and task losses", "task_heads": "task losses"},
        "loss_weights": loss_weights, "retrieval_temperature": temperature,
        "heads": {"span": SPAN_LABELS, "condition_update": UPDATES, "relation": RELATIONS,
                  "operation": OPERATIONS, "version": VERSIONS, "condition_relation": CONDITION_RELATIONS,
                  "document_families": "observed family candidates", "version_ids": "observed version candidates",
                  "resolved_version_ids": "observed version candidates",
                  "need_subject": "independent span edges", "need_condition": "current and historical condition edges",
                  "need_predecessor": "current and historical Need DAG edges", "scope": SCOPES,
                  "history": [0, 1], "condition_target": "multi-label prior condition IDs",
                  "condition_supersedes": "multi-label prior condition IDs", "task_pointer": "prior task IDs",
                  "need_group": "same-Need span edges", "condition_group": "same-Condition span edges",
                  "temporal_relation": TEMPORAL_RELATIONS, "source_turns": "multi-label actual committed turn IDs"},
        "span_scheme": "parallel BIO channels per span kind; nested need/subject/condition spans retained",
        "span_loss": "equal mean across channels and the BIO classes present in each channel; special and zero-width tokens masked",
        "span_supervision_tokens": {kind: {name: counts[label] for label, name in enumerate(("O", "B", "I"))}
                                    for kind, counts in span_counts.items()},
        "training_order": "seeded conversation shuffle per epoch; ascending turns within each conversation",
        "history_supervision": "token-level selected historical spans; current question masked",
        "training_config": settings}
    manifest["input_not_visible_states"] = len(input_not_visible)
    (output/"task-model.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output/"losses.jsonl").write_text("".join(json.dumps(row)+"\n" for row in logs))
    return manifest


def evaluate_task(config):
    """Held-out task-head evaluation on recorded actual-history teacher labels."""
    from adapters.hybrid_retrieval import LocalBGE
    from controller.context_schema import TaskState
    from controller.serialization import to_jsonable
    from training.task_data import write_jsonl
    settings = config["training"]
    task = config.get("task_model", {})
    device = settings.get("device", "cuda")
    base = task.get("base_path", "models/bge-m3")
    model = TaskModel(settings.get("task_output", "models/study-task"), base_path=base, device=device)
    frozen = LocalBGE(base, device=device, dtype=settings.get("dtype", "bfloat16"), max_length=8192)
    index = HistoryIndex(frozen.tokenizer, frozen.encode, unit_tokens=task.get("history_unit_tokens", 384),
                         candidates=task.get("history_candidates", 16), view_tokens=8192)
    prior, rows = {}, []

    def characters(refs):
        return {(ref.turn_id, ref.speaker, position) for ref in refs for position in range(ref.start, ref.end)}

    def f1(predicted, expected):
        if not predicted and not expected:
            return 1.0
        return 2*len(predicted & expected)/max(1, len(predicted)+len(expected))

    evaluation_rows = select_cohort_rows(read_jsonl(settings["task_eval_labels"]), settings["cohort_manifest"], "dev")
    evaluation_rows.sort(key=lambda row: (row["conversation_id"], row["turn"]))
    for row in evaluation_rows:
        history = tuple(decode(TurnRecord, value) for value in row["actual_history"])
        memory = RetrievalState(len(history), history, tuple(prior.get(row["conversation_id"], ())),
            source_index=observed_task_sources(row))
        turn = OnlineTurn(row["conversation_id"], row["task_id"], row["turn"], row["snapshot_id"], row["question"], (), (), "P3")
        expected = decode(TaskPrediction, row["prediction"])
        windows = index.select(turn.query, memory, current_turn_id=turn.turn)
        encoded = encode_task(model.tokenizer, turn, windows, memory, catalogue=row["task_catalogue"])
        input_visibility = {"status": "visible"}
        try:
            task_targets(encoded, expected)
        except TaskInputNotVisible as error:
            input_visibility = {"status": "input_not_visible", "reason": str(error)}
        predicted = model.predict(turn, windows, memory,
                                  catalogue=row["task_catalogue"])
        def condition_items(prediction):
            return {(condition.update, condition.relation_type,
                     tuple((ref.turn_id, ref.speaker, ref.start, ref.end) for ref in condition.value_spans),
                     condition.target_refs, condition.supersedes, condition.key_span, condition.predicate_span) for condition in prediction.conditions}
        predicted_spans = characters(ref for need in predicted.needs for ref in need.anchor_spans)
        expected_spans = characters(ref for need in expected.needs for ref in need.anchor_spans)
        metrics = {"relation_accuracy": float(predicted.relation == expected.relation),
            "version_accuracy": float(predicted.version_intent.mode == expected.version_intent.mode),
            "temporal_relation_accuracy": float(predicted.version_intent.temporal_relation == expected.version_intent.temporal_relation),
            "version_source_turns_accuracy": float(set(predicted.version_intent.source_turns) == set(expected.version_intent.source_turns)),
            "version_time_character_f1": f1(characters(predicted.version_intent.time_refs), characters(expected.version_intent.time_refs)),
            "operation_f1": f1({(ordered_refs(n.anchor_spans), n.operation) for n in predicted.needs},
                               {(ordered_refs(n.anchor_spans), n.operation) for n in expected.needs}),
            "history_character_f1": f1(characters(predicted.history_spans), characters(expected.history_spans)),
            "need_character_f1": f1(predicted_spans, expected_spans),
            "condition_update_f1": f1(condition_items(predicted), condition_items(expected)),
            "operation_accuracy": float(tuple(n.operation for n in predicted.needs) == tuple(n.operation for n in expected.needs)),
            "resume_accuracy": float(predicted.resume_task_id == expected.resume_task_id),
            **{name + "_f1": f1(set(getattr(predicted.version_intent, name)), set(getattr(expected.version_intent, name)))
               for name in ("document_families", "version_ids", "resolved_version_ids")}}
        prior_conditions = tuple(c for state in memory.prior_task_states for c in state.conditions)
        predicted_graph = graph_items(predicted, prior_conditions, memory.prior_task_states)
        expected_graph = graph_items(expected, prior_conditions, memory.prior_task_states)
        metrics.update({name + "_f1": f1(predicted_graph[name], expected_graph[name]) for name in predicted_graph})
        rows.append({"task_id": row["task_id"], "split": row["split"], "cohort_split": row["cohort_split"],
                     "cohort_manifest_sha256": row["cohort_manifest_sha256"], "question": row["question"],
                     "actual_history": row["actual_history"], "metrics": metrics,
                     "input_visibility": input_visibility,
                     "prediction": to_jsonable(predicted), "expected": to_jsonable(expected)})
        prior.setdefault(row["conversation_id"], []).append(decode(TaskState, row["task_state"]))
    output = Path(settings.get("task_eval_output", "runs/study/task-evaluation"))
    write_jsonl(output/"predictions.jsonl", rows)
    summary = {"rows": len(rows), "context": "actual QA with teacher prior task-state replay",
               "labels_path": settings["task_eval_labels"],
               "metrics": {name: sum(row["metrics"][name] for row in rows)/len(rows) for name in rows[0]["metrics"]}}
    if settings.get("task_dense_index") is not None:
        from adapters.doc2dial import _identity
        from adapters.hybrid_retrieval import DenseIndex
        from training.preference_rollout import query_encoder_binding
        from training.reader_data import file_digest
        retrieval_started = time.perf_counter()
        dense = DenseIndex.load(settings["task_dense_index"])
        baseline = LocalBGE(base, device=device, dtype=settings.get("dtype", "bfloat16"), max_length=8192,
            query_adapter_path=task.get("initial_adapter_path", "models/bge-m3-dialogue-query-v1"),
            normalize_float32=True)
        setup_ms = (time.perf_counter()-retrieval_started)*1000

        def union_length(intervals):
            size, end = 0, -1
            for left, right in sorted(intervals):
                size += max(0, right-max(left, end))
                end = max(end, right)
            return size

        declared = {member["task_id"] for member in json.loads(Path(settings["cohort_manifest"]).read_text())["members"]
                    if member["split"] == "dev"}
        cases = []
        for row in evaluation_rows:
            gold = {}
            for span in row["verified_grounding"]:
                gold.setdefault(span["passage_id"], []).append((span["start"], span["end"]))
            denominator = sum(union_length(intervals) for intervals in gold.values())
            case = {"task_id": row["task_id"], "conversation_id": row["conversation_id"], "cohort_split": "dev",
                "query": row["task_state"]["query"], "gold_spans": row["verified_grounding"],
                "gold_characters": denominator, "valid_gold": denominator > 0, "encoders": {}}
            for name, encoder in (("P0", baseline), ("task_query", model)):
                before = dict(encoder.usage)
                started = time.perf_counter()
                hits = dense.search(case["query"], encoder, top_k=50)
                elapsed_ms = (time.perf_counter()-started)*1000
                ranked = []
                for hit in hits:
                    domain, document_id, _, left, right = json.loads(hit.document_id)
                    ranked.append({"section_id": hit.document_id, "document_id": _identity(domain, document_id),
                        "start": left, "end": right, "score": hit.score})
                scores = {}
                for k in (20, 50):
                    intersections = {}
                    for hit in ranked[:k]:
                        for left, right in gold.get(hit["document_id"], ()):
                            lo, hi = max(left, hit["start"]), min(right, hit["end"])
                            if lo < hi:
                                intersections.setdefault(hit["document_id"], []).append((lo, hi))
                    covered = sum(union_length(intervals) for intervals in intersections.values())
                    scores[str(k)] = {"covered_characters": covered,
                        "recall": covered/denominator if denominator else None,
                        "fully_covered": covered == denominator if denominator else None}
                case["encoders"][name] = {"hits": ranked, "coverage": scores,
                    "input_tokens": encoder.usage["input_tokens"]-before["input_tokens"],
                    "query_encode_ms": encoder.usage["elapsed_ms"]-before["elapsed_ms"],
                    "search_complete_ms": elapsed_ms}
            cases.append(case)
        valid = [case for case in cases if case["valid_gold"]]
        aggregate = {}
        for name in ("P0", "task_query"):
            aggregate[name] = {"input_tokens": sum(case["encoders"][name]["input_tokens"] for case in cases),
                "query_encode_ms": sum(case["encoders"][name]["query_encode_ms"] for case in cases),
                "search_complete_ms": sum(case["encoders"][name]["search_complete_ms"] for case in cases),
                "recall": {str(k): {
                    "macro": sum(case["encoders"][name]["coverage"][str(k)]["recall"] for case in valid)/len(valid) if valid else None,
                    "micro": sum(case["encoders"][name]["coverage"][str(k)]["covered_characters"] for case in valid)
                        /sum(case["gold_characters"] for case in valid) if valid else None,
                    "fully_covered_states": sum(case["encoders"][name]["coverage"][str(k)]["fully_covered"] for case in valid)}
                    for k in (20, 50)}}
        report = {"query_source": "fixed independently reviewed task_state.query for both encoders",
            "metric": "union of retrieved section intersections with original-document gold character intervals / union of gold intervals",
            "cohort_manifest": settings["cohort_manifest"], "declared_states": len(declared),
            "evaluated_states": len(cases), "valid_gold_states": len(valid),
            "no_gold_task_ids": [case["task_id"] for case in cases if not case["valid_gold"]],
            "missing_label_task_ids": sorted(declared-{case["task_id"] for case in cases}),
            "valid_gold_characters": sum(case["gold_characters"] for case in valid),
            "index": {"path": settings["task_dense_index"], "manifest_sha256": file_digest(Path(settings["task_dense_index"])/"manifest.json")},
            "query_assets": {"P0": query_encoder_binding({"query_adapter_path": task.get("initial_adapter_path", "models/bge-m3-dialogue-query-v1"),
                                                            "encoder_path": base}),
                "task_query": query_encoder_binding({"task_asset": settings["task_output"]})},
            "setup_ms": setup_ms, "complete_ms": (time.perf_counter()-retrieval_started)*1000,
            "setup_scope": "fixed index and P0 query model; task model was already loaded for task-head evaluation",
            "cost_scope": "observed sequential query encoding/search and setup; no comparable system cost improvement inferred",
            "aggregate": aggregate, "cases": cases}
        path = output/"retrieval.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        summary["retrieval"] = str(path.resolve())
        baseline.close()
    (output/"metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    frozen.close()
    model.close()
    return summary
