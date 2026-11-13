"""One query forward and one forward per evidence batch, shared by task heads."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from concurrent.futures import Future
from threading import Lock
import time

from controller.context_schema import Condition, Need, SourceSpan, TaskState, TextRef, VersionIntent

QUERY_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EVIDENCE_MODEL = "jinaai/jina-reranker-v3.5"


class ReadyBatch:
    """Drain ready jobs together while one thread owns the shared backbone."""
    def __init__(self, execute):
        self.execute, self.lock = execute, Lock()
        self.pending, self.running = [], False

    def submit(self, value):
        future = Future()
        with self.lock:
            self.pending.append((value, future))
            owner = not self.running
            self.running = True
        if owner:
            while True:
                with self.lock:
                    jobs, self.pending = self.pending, []
                    if not jobs:
                        self.running = False
                        break
                try:
                    results = self.execute([value for value, _ in jobs])
                    for (_, target), result in zip(jobs, results):
                        target.set_result(result)
                except Exception as error:
                    for _, target in jobs:
                        target.set_exception(error)
        return future.result()


@dataclass
class QueryEncoding:
    task: TaskState
    query_vector: object
    need_vectors: object
    token_states: object
    token_refs: tuple
    texts: dict
    filter_operators: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    condition_vectors: object = None
    join_needs: frozenset = frozenset()
    distinct_needs: frozenset = frozenset()

    def text(self, refs):
        return " ".join(self.texts[(ref.turn_id, ref.speaker)][ref.start:ref.end] for ref in refs)

    @property
    def search_texts(self):
        return (self.task.raw_question, *(self.text((*need.anchor_spans, *need.subject_refs,
            *(ref for condition in self.task.conditions if condition.condition_id in need.condition_ids
              for ref in condition.value_spans))) for need in self.task.needs))

    @property
    def evidence_query(self):
        needs = "\n".join(dict.fromkeys(self.search_texts[1:]))
        versions = ", ".join(self.task.version_intent.version_ids)
        return self.task.raw_question + ("\nRequested evidence:\n"+needs if needs else "") + ("\nVersions: "+versions if versions else "")


@dataclass(frozen=True)
class FieldBinding:
    need_id: str
    role: str
    source_span: SourceSpan
    text: str
    score: float
    condition_id: str = ""
    relation_group: str = ""
    table_id: str = ""


@dataclass
class EvidenceEncoding:
    unit_ids: tuple[str, ...]
    embeddings: object
    relevance: object
    bindings: tuple[FieldBinding, ...]
    heads: object
    token_states: tuple = ()
    token_sources: tuple = ()
    usage: dict = field(default_factory=dict)
    query_sum: object = None

    def merge(self, other):
        """Extend a shared list using the author's weighted cross-batch query vector."""
        import torch
        embeddings = torch.cat((self.embeddings, other.embeddings))
        query_sum = self.query_sum+other.query_sum
        relevance = torch.nn.functional.cosine_similarity(embeddings, query_sum[None, :], dim=-1)
        return EvidenceEncoding((*self.unit_ids, *other.unit_ids), embeddings, relevance,
            (*self.bindings, *other.bindings), self.heads, usage=other.usage, query_sum=query_sum)

    def summary(self, selected_ids):
        import torch
        selected = set(selected_ids)
        mask = torch.tensor([identity in selected for identity in self.unit_ids],
                            device=self.embeddings.device, dtype=torch.bool)
        chosen, rest = self.embeddings[mask], self.embeddings[~mask]
        zero = self.embeddings.new_zeros(self.embeddings.shape[-1])
        mean = chosen.mean(0) if len(chosen) else zero
        maximum = chosen.max(0).values if len(chosen) else zero
        remaining = rest.mean(0) if len(rest) else zero
        stats = self.embeddings.new_tensor([len(chosen), len(rest),
            self.relevance[mask].mean().item() if len(chosen) else 0.,
            self.relevance[~mask].mean().item() if len(rest) else 0.])
        return torch.cat((mean, maximum, remaining, stats))

    def select(self, units, *, body_tokens):
        """Prune a joint evidence set by generator utility, retaining complementary groups."""
        import numpy as np
        import torch
        by_id = {unit.unit_id: unit for unit in units}
        positions = {identity: index for index, identity in enumerate(self.unit_ids)}
        count = len(self.unit_ids)
        consumers = [[] for _ in self.unit_ids]
        for unit in units:
            for dependency in unit.dependencies:
                consumers[positions[dependency]].append(positions[unit.unit_id])
        costs = np.asarray([by_id[identity].body_tokens for identity in self.unit_ids], dtype=np.int64)
        active = np.ones(count, dtype=bool)
        chosen = (1 << count)-1
        with torch.no_grad():
            encoded = self.heads.set_item(self.embeddings)
            width = encoded.shape[-1]
            vectors = torch.cat((encoded, self.relevance[:, None]), dim=-1).cpu().numpy()
            items = np.empty((count, width+2), dtype=np.float64)
            items[:, :width] = vectors[:, :width]
            items[:, width] = 1
            items[:, width+1] = vectors[:, width]
            removal = np.empty_like(items)
            removal_cost = np.empty(count, dtype=np.int64)
            closures, owners = [], [[] for _ in self.unit_ids]
            for root in range(count):
                pending, members = [root], set()
                while pending:
                    member = pending.pop()
                    if member not in members:
                        members.add(member)
                        pending.extend(consumers[member])
                indices = list(members)
                closures.append(sum(1 << member for member in indices))
                removal[root] = items[indices].sum(axis=0)
                removal_cost[root] = costs[indices].sum()
                for member in indices:
                    owners[member].append(root)
            owners = [np.asarray(group, dtype=np.intp) for group in owners]
            total, cost = items.sum(axis=0), int(costs.sum())
            while chosen:
                candidates = np.flatnonzero(active)
                pooled = np.empty((len(candidates)+1, width+2), dtype=np.float32)
                pooled[0] = total
                pooled[1:] = total-removal[candidates]
                pooled = torch.from_numpy(pooled).to(self.embeddings.device)
                values = self.heads.utility_pooled(pooled[:, :width], pooled[:, width:width+1],
                    pooled[:, width+1:]).cpu().numpy()
                gains = values[1:]-values[0]
                if cost > body_tokens:
                    index = int((gains/removal_cost[candidates].astype(values.dtype)).argmax())
                else:
                    index = int(gains.argmax())
                if cost <= body_tokens and gains[index] <= 0:
                    break
                removed = closures[int(candidates[index])] & chosen
                chosen &= ~removed
                deleted = []
                while removed:
                    bit = removed & -removed
                    deleted.append(bit.bit_length()-1)
                    removed ^= bit
                active[deleted] = False
                total -= items[deleted].sum(axis=0)
                cost -= int(costs[deleted].sum())
                # Each removed item updates only closures that actually contained it.
                for member in deleted:
                    affected = owners[member][active[owners[member]]]
                    removal[affected] -= items[member]
                    removal_cost[affected] -= costs[member]
        return tuple(identity for index, identity in enumerate(self.unit_ids) if active[index])


def _spans(labels, coordinates):
    """Decode independent BIO labels while retaining original source addresses."""
    result, start, previous = [], None, None
    for index, (label, coordinate) in enumerate(zip(labels, coordinates)):
        same_source = coordinate is not None and previous is not None and coordinate[:2] == previous[:2]
        if start is not None and (label != 2 or not same_source):
            result.append((start, index))
            start = None
        if coordinate is not None and label in {1, 2} and start is None:
            start = index
        previous = coordinate
    if start is not None:
        result.append((start, len(coordinates)))
    return result


def _dependencies(scores):
    """Decode scored Need relations as an acyclic dependency graph."""
    ancestors = [set() for _ in scores]
    parents = [set() for _ in scores]
    edges = sorted(((score, child, parent) for child, row in enumerate(scores)
                    for parent, score in enumerate(row) if child != parent and score > 0), reverse=True)
    for _, child, parent in edges:
        if child in ancestors[parent]:
            continue
        parents[child].add(parent)
        additions = {parent, *ancestors[parent]}
        for node in range(len(scores)):
            if node == child or child in ancestors[node]:
                ancestors[node].update(additions)
    return parents


def query_input(tokenizer, history, question, current_turn_id, catalogue=(), max_length=32768,
                history_tokens=4096, history_turns=()):
    """The same token recipe is used by training and online task inference."""
    parts, owners, texts, catalog_spans = [], [], {}, []
    length = 0

    def add(text, owner=None):
        nonlocal length
        start = length
        parts.append(text)
        length += len(text)
        if owner is not None:
            owners.append((start, length, owner))
        return start, length

    add("Instruct: Retrieve evidence for the current question using the conversation.\nAvailable source versions:\n")
    for row in catalogue:
        start, end = add(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        catalog_spans.append((start, end, row))
        add("\n")
    add("Conversation:\n")
    selected, size = set(), 0
    order = sorted(range(len(history)), key=lambda index: (history[index][0] in history_turns, index), reverse=True)
    for index in order:
        turn_id, speaker, text = history[index]
        tokens = len(tokenizer.encode(f"Turn {turn_id} {speaker}: {text}\n", add_special_tokens=False))
        if size+tokens <= history_tokens:
            selected.add(index)
            size += tokens
    for index, (turn_id, speaker, text) in enumerate(history):
        if index not in selected:
            continue
        texts[turn_id, speaker] = text
        add(f"Turn {turn_id} {speaker}: ")
        add(text, (turn_id, speaker))
        add("\n")
    texts[current_turn_id, "user"] = question
    add(f"Current query (turn {current_turn_id}): ")
    add(question, (current_turn_id, "user"))
    encoded = tokenizer("".join(parts), return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded.pop("offset_mapping")
    refs = []
    for start, end in offsets:
        owner = next(((left, right, kind) for left, right, kind in owners if start < right and left < end), None)
        refs.append(None if owner is None else TextRef(*owner[2],
            max(start, owner[0])-owner[0], min(end, owner[1])-owner[0]))
    catalog_positions = [(row, [i for i, (left, right) in enumerate(offsets) if left < end and start < right])
                         for start, end, row in catalog_spans]
    # The caller supplies the actual retained history; a long current input is
    # split by document/evidence processing rather than silently truncated here.
    if len(encoded["input_ids"]) > max_length:
        raise ValueError("shared query input exceeds the encoder context")
    return encoded, tuple(refs), texts, catalog_positions


class SharedQueryEncoder:
    def __init__(self, model_path, asset_path, *, device="cuda:0", dtype="bfloat16", train=False, history_tokens=4096):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from training.shared_heads import QueryHeads
        self.torch = torch
        self.history_tokens = history_tokens
        self.backbone_lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(model_path, dtype=getattr(torch, dtype)).to(device)
        self.heads = QueryHeads(self.model.config.hidden_size).to(device)
        self.asset_path = None if train else Path(asset_path)
        if not train:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.asset_path / "adapter", is_trainable=train)
            self.heads.load_state_dict(torch.load(self.asset_path / "heads.pt", map_location=device, weights_only=True))
        self.model.train(train)
        self.heads.train(train)
        self.manifest = {"model_id": QUERY_MODEL, "asset": str(asset_path), "shared_task_encoding": True}
        self.ready = ReadyBatch(self.encode_many)

    def encode_documents(self, texts):
        return self.encode_documents_with_usage(texts)[0]

    def encode_documents_with_usage(self, texts):
        import torch
        from contextlib import nullcontext
        with self.backbone_lock, torch.no_grad():
            batch = self.tokenizer(texts, padding=True, return_tensors="pt")
            tokens = int(batch["attention_mask"].sum())
            batch = batch.to(self.model.device)
            adapter_context = self.model.disable_adapter() if self.asset_path is not None else nullcontext()
            with adapter_context:
                states = self.model(**batch, use_cache=False).last_hidden_state
            return torch.nn.functional.normalize(states[:, -1].float(), dim=-1).cpu().numpy(), tokens

    def forward_input(self, encoded):
        torch = self.torch
        batch = {key: torch.tensor([value], device=self.model.device) for key, value in encoded.items()}
        states = self.model(**batch, use_cache=False).last_hidden_state[0].float()
        return states, self.heads(states, states[-1])

    def encode(self, history, question, task_id, current_turn_id, *, catalogue=(), source_scope=(), history_turns=()):
        return self.ready.submit({"history": history, "question": question, "task_id": task_id,
            "current_turn_id": current_turn_id, "catalogue": catalogue, "source_scope": source_scope,
            "history_turns": history_turns})

    def encode_many(self, requests):
        import torch
        started = time.perf_counter()
        prepared = [query_input(self.tokenizer, row["history"], row["question"], row["current_turn_id"],
            row.get("catalogue", ()), history_tokens=self.history_tokens, history_turns=row.get("history_turns", ())) for row in requests]
        batch = self.tokenizer.pad([item[0] for item in prepared], padding=True, return_tensors="pt").to(self.model.device)
        with self.backbone_lock, torch.no_grad():
            states = self.model(**batch, use_cache=False).last_hidden_state.float()
            results = [self._encode(**row, _prepared=item, _states=states[index, -len(item[0]["input_ids"]):])
                       for index, (row, item) in enumerate(zip(requests, prepared))]
        elapsed = (time.perf_counter()-started)*1000
        total = sum(item.usage["input_tokens"] for item in results)
        for result in results:
            result.usage["compute_ms"] = elapsed*result.usage["input_tokens"]/total
        return results

    def _encode(self, history, question, task_id, current_turn_id, *, catalogue=(), source_scope=(),
                history_turns=(), _prepared, _states):
        import torch
        from training.shared_heads import (CONDITION_TYPES, EDGE_KINDS, FILTER_OPERATORS, OPERATIONS,
            SCOPES, SPAN_KINDS, TEMPORAL_RELATIONS, UPDATES, VERSION_MODES)
        started = time.perf_counter()
        encoded, refs, texts, catalogue_positions = _prepared
        with torch.no_grad():
            states = _states
            logits = self.heads(states, states[-1])
            coordinates = [(ref.turn_id, ref.speaker, ref.start, ref.end) if ref is not None else None for ref in refs]
            labels = logits["span"].argmax(-1).tolist()
            spans = {kind: _spans([row[index] for row in labels], coordinates)
                     for index, kind in enumerate(SPAN_KINDS)}

            def reference(span):
                first, last = span
                return TextRef(refs[first].turn_id, refs[first].speaker, refs[first].start, refs[last-1].end)

            def vectors(items):
                features = logits["features"]
                return torch.stack([features[first:last].mean(0) for first, last in items]) if items else states.new_empty((0, states.shape[-1]))

            conditions = []
            condition_vectors = vectors(spans["condition"])
            values = vectors(spans["value"])
            value_edges = self.heads.edges(condition_vectors, values)[:, :, EDGE_KINDS.index("value")]
            operators = {}
            for index, span in enumerate(spans["condition"]):
                vector = condition_vectors[index]
                identity = f"c{index}"
                selected_values = tuple(reference(item) for position, item in enumerate(spans["value"])
                                        if value_edges[index, position] > 0)
                conditions.append(Condition(identity, key_span=reference(span), value_spans=selected_values,
                    relation_type=CONDITION_TYPES[int(self.heads.condition_type(vector).argmax())],
                    predicate_span=reference(span), update=UPDATES[int(self.heads.update(vector).argmax())]))
                operators[identity] = FILTER_OPERATORS[int(self.heads.filter_operator(vector).argmax())]
            need_spans = [span for span in spans["need"] if refs[span[0]].turn_id == current_turn_id]
            need_vectors = vectors(need_spans)
            subject_vectors = vectors(spans["subject"])
            subject_edges = self.heads.edges(need_vectors, subject_vectors)[:, :, EDGE_KINDS.index("subject")]
            condition_edges = self.heads.edges(need_vectors, condition_vectors)[:, :, EDGE_KINDS.index("condition")]
            predecessor_edges = self.heads.edges(need_vectors, need_vectors)[:, :, EDGE_KINDS.index("predecessor")]
            dependencies = _dependencies(predecessor_edges.tolist())
            needs = tuple(Need(f"n{index}", anchor_spans=(reference(span),),
                operation=OPERATIONS[int(self.heads.operation(vector).argmax())],
                subject_refs=tuple(reference(item) for position, item in enumerate(spans["subject"])
                                   if subject_edges[index, position] > 0),
                condition_ids=tuple(condition.condition_id for position, condition in enumerate(conditions)
                    if condition_edges[index, position] > 0 and condition.update != "REMOVE"),
                predecessor_need_ids=tuple(f"n{position}" for position in sorted(dependencies[index])),
                scope=SCOPES[int(self.heads.scope(vector).argmax())])
                for index, (span, vector) in enumerate(zip(need_spans, need_vectors)))
            selected_versions, selected_families = [], []
            for row, positions in catalogue_positions:
                vector = states[positions].mean(0)
                if self.heads.version(states[-1], vector).item() > 0:
                    selected_versions.append(row["version"])
                if self.heads.family(states[-1], vector).item() > 0:
                    selected_families.append(row["family"])
            for span in spans["version"]:
                ref = reference(span)
                selected_versions.append(texts[ref.turn_id, ref.speaker][ref.start:ref.end])
            history_refs = []
            for ref, selected in zip(refs, (logits["history"] > 0).tolist()):
                if ref is not None and ref.turn_id != current_turn_id and selected:
                    previous = history_refs[-1] if history_refs else None
                    if previous is not None and (previous.turn_id, previous.speaker, previous.end) == (ref.turn_id, ref.speaker, ref.start):
                        history_refs[-1] = replace(previous, end=ref.end)
                    else:
                        history_refs.append(ref)
            mode = VERSION_MODES[int(logits["version_mode"].argmax())]
            intent = VersionIntent(mode,
                document_families=tuple(dict.fromkeys(selected_families)),
                version_ids=tuple(dict.fromkeys(selected_versions)) if mode != "current" else (),
                time_refs=tuple(reference(span) for span in spans["time"]),
                temporal_relation=TEMPORAL_RELATIONS[int(logits["temporal_relation"].argmax())])
            task = TaskState(task_id, question, tuple(history_refs), tuple(conditions), needs, intent,
                             tuple(source_scope), current_turn_id, question)
            projected = torch.nn.functional.normalize(self.heads.need_projection(need_vectors), dim=-1)
            return QueryEncoding(task, torch.nn.functional.normalize(states[-1], dim=-1), projected,
                states, refs, texts, operators, {"input_tokens": len(encoded["input_ids"]),
                    "compute_ms": (time.perf_counter()-started)*1000, "calls": 1}, condition_vectors,
                frozenset(need.need_id for need, vector in zip(needs, need_vectors)
                          if int(self.heads.join(vector).argmax()) == 1),
                frozenset(need.need_id for need, vector in zip(needs, need_vectors)
                          if int(self.heads.distinct(vector).argmax()) == 1))

    def close(self):
        self.model = self.heads = None


class SharedEvidenceEncoder:
    def __init__(self, model_path, asset_path, *, device="cuda:0", dtype="bfloat16", batch_tokens=32768, train=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from training.shared_heads import EvidenceHeads
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
            dtype=getattr(torch, dtype)).to(device)
        self.heads = EvidenceHeads(self.model.config.hidden_size).to(device)
        if not train or asset_path is not None:
            path = Path(asset_path)
            self.heads.load_state_dict(torch.load(path / "heads.pt", map_location=device, weights_only=True))
        self.model.train(train)
        self.heads.train(train)
        self.batch_tokens = batch_tokens
        self.manifest = {"model_id": EVIDENCE_MODEL, "asset": str(asset_path), "shared_evidence_encoding": True}
        self.ready = ReadyBatch(self.encode_many)

    def prepare(self, query, units):
        """Use Jina's listwise prompt and explicit embedding marker token IDs."""
        prefix = ("<|im_start|>system\nYou are a search relevance expert who can determine a ranking of the passages "
            "based on how relevant they are to the query. If the query is a question, how relevant a passage is "
            "depends on how well it answers the question. If not, try to analyze the intent of the query and "
            "assess how well each passage satisfies the intent. If an instruction is provided, you should follow "
            "the instruction when determining the ranking.<|im_end|>\n<|im_start|>user\n")
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        ids = encode(prefix + f"I will provide you with {len(units)} passages, each indicated by a numerical identifier. "
                     f"Rank the passages based on their relevance to query: {query}\n")
        addresses, doc_positions = [None]*len(ids), []
        for index, unit in enumerate(units):
            header = encode(f'<passage id="{index}">\n'+(unit.heading+"\n" if unit.heading else ""))
            ids.extend(header)
            addresses.extend([None]*len(header))
            body = self.tokenizer(unit.text, add_special_tokens=False, return_offsets_mapping=True,
                                  split_special_tokens=True)
            ids.extend(body["input_ids"])
            source = unit.source_spans[0]
            addresses.extend((unit.unit_id, source.source_key, source.start+left, source.start+right)
                             for left, right in body["offset_mapping"])
            doc_positions.append(len(ids))
            ids.append(self.model.doc_embed_token_id)
            addresses.append(None)
            tail = encode("\n</passage>\n")
            ids.extend(tail)
            addresses.extend([None]*len(tail))
        tail = encode(f"<query>\n{query}")
        ids.extend(tail)
        addresses.extend([None]*len(tail))
        query_position = len(ids)
        ids.append(self.model.query_embed_token_id)
        addresses.append(None)
        tail = encode("\n</query><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        ids.extend(tail)
        addresses.extend([None]*len(tail))
        return ids, tuple(addresses), doc_positions, query_position

    def forward_batch(self, query, units):
        import torch
        ids, addresses, doc_positions, query_position = self.prepare(query, units)
        input_ids = torch.tensor([ids], device=self.model.device)
        states = self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        # The author projector and cosine objective consume the same final layer.
        documents = self.model.projector(states[doc_positions]).float()
        q = self.model.projector(states[query_position]).float()
        scores = torch.nn.functional.cosine_similarity(documents, q.unsqueeze(0), dim=-1)
        return states.float(), torch.nn.functional.normalize(documents, dim=-1), scores, addresses, len(ids), q

    def forward_evidence(self, query, units):
        """Train on the same evidence batches and global cosine scores used online."""
        import torch
        outputs = [self.forward_batch(query.evidence_query, batch) for batch in self._batches(query, units)]
        if not outputs:
            return (torch.empty((0, self.model.config.hidden_size), device=self.model.device),
                    torch.empty((0, 512), device=self.model.device),
                    torch.empty(0, device=self.model.device), (), 0)
        documents = torch.cat([item[1] for item in outputs])
        query_sum = torch.stack([item[5]*((1+item[2])/2).max() for item in outputs]).sum(0)
        scores = torch.nn.functional.cosine_similarity(documents, query_sum[None, :], dim=-1)
        return (torch.cat([item[0] for item in outputs]), documents, scores,
                tuple(address for item in outputs for address in item[3]), sum(item[4] for item in outputs))

    def _batches(self, query, units):
        units = tuple(units)
        batches, current, size = [], [], 0
        overhead = len(self.prepare(query.evidence_query, ())[0])
        for unit in units:
            length = len(self.tokenizer.encode(unit.text, add_special_tokens=False)) + 20
            if current and size+length+overhead > self.batch_tokens:
                batches.append(tuple(current))
                current, size = [], 0
            current.append(unit)
            size += length
        if current:
            batches.append(tuple(current))
        return batches

    def encode(self, query, units):
        return self.ready.submit((query, tuple(units)))

    def encode_many(self, requests):
        import torch
        started = time.perf_counter()
        prepared = [(owner, batch, self.prepare(query.evidence_query, batch))
                    for owner, (query, units) in enumerate(requests) for batch in self._batches(query, units)]
        outputs = [{} for _ in requests]
        groups, current, maximum = [], [], 0
        for item in prepared:
            size = len(item[2][0])
            if current and max(maximum, size)*(len(current)+1) > self.batch_tokens:
                groups.append(current)
                current, maximum = [], 0
            current.append(item)
            maximum = max(maximum, size)
        if current:
            groups.append(current)
        with torch.no_grad():
            for group in groups:
                batch = self.tokenizer.pad([{"input_ids": item[2][0], "attention_mask": [1]*len(item[2][0])}
                    for item in group], padding=True, return_tensors="pt").to(self.model.device)
                hidden = self.model.model(**batch, use_cache=False).last_hidden_state
                for offset, (owner, units, (ids, addresses, positions, query_position)) in enumerate(group):
                    states = hidden[offset, -len(ids):]
                    documents = self.model.projector(states[positions]).float()
                    query = self.model.projector(states[query_position]).float()
                    scores = torch.nn.functional.cosine_similarity(documents, query.unsqueeze(0), dim=-1)
                    outputs[owner][tuple(unit.unit_id for unit in units)] = (states.float(),
                        torch.nn.functional.normalize(documents, dim=-1), scores, addresses, len(ids), query)
            results = [self._encode(query, units, outputs[index]) for index, (query, units) in enumerate(requests)]
        elapsed = (time.perf_counter()-started)*1000
        total = sum(item.usage["input_tokens"] for item in results)
        for result in results:
            result.usage["compute_ms"] = elapsed*result.usage["input_tokens"]/total if total else 0.
        return results

    def _encode(self, query, units, outputs):
        import torch
        from training.shared_heads import FIELD_ROLES
        started = time.perf_counter()
        batches = self._batches(query, units)
        embeddings, queries, bindings, binding_vectors, token_states, token_sources = [], [], [], [], [], []
        token_count = 0
        with torch.no_grad():
            for batch in batches:
                states, documents, scores, addresses, count, query_vector = outputs[tuple(unit.unit_id for unit in batch)]
                token_count += count
                embeddings.append(documents)
                queries.append(query_vector*((1+scores)/2).max())
                source_indices = [i for i, address in enumerate(addresses) if address is not None]
                source_states = states[source_indices]
                sources = tuple(addresses[i] for i in source_indices)
                probabilities = self.heads.field_logits(source_states, query.need_vectors.to(states.device)).softmax(-1)
                confidence, labels = probabilities.max(-1)
                for need_index, need in enumerate(query.task.needs):
                    for role_index, role in enumerate(FIELD_ROLES):
                        for first, last in _spans(labels[need_index, :, role_index].tolist(), sources):
                            owner, key, left, _ = sources[first]
                            right = sources[last-1][3]
                            unit = next(item for item in batch if item.unit_id == owner)
                            base = unit.source_spans[0].start
                            vector = source_states[first:last].mean(0)
                            condition_id = ""
                            if role in {"filter", "period"} and need.condition_ids:
                                indices = [i for i, condition in enumerate(query.task.conditions)
                                           if condition.condition_id in need.condition_ids]
                                conditions = self.heads.query(query.condition_vectors[indices].to(states.device))
                                values = self.heads.binding(conditions, vector.expand_as(conditions))[:, role_index]
                                condition_id = query.task.conditions[indices[int(values.argmax())]].condition_id
                            bindings.append(FieldBinding(need.need_id, role,
                                SourceSpan(f"{owner}:{left}:{right}", key, left, right),
                                unit.text[left-base:right-base], float(confidence[need_index, first:last, role_index].mean()),
                                condition_id, table_id=unit.table_id or ""))
                            binding_vectors.append(vector)
            for need_id in query.join_needs:
                indices = [i for i, binding in enumerate(bindings) if binding.need_id == need_id and binding.role == "entity"]
                groups = {index: index for index in indices}
                def root(index):
                    while groups[index] != index:
                        index = groups[index]
                    return index
                for offset, left in enumerate(indices):
                    for right in indices[offset+1:]:
                        if self.heads.join(binding_vectors[left], binding_vectors[right]).item() > 0:
                            groups[root(right)] = root(left)
                for index in indices:
                    if sum(root(other) == root(index) for other in indices) > 1:
                        bindings[index] = replace(bindings[index], relation_group=f"{need_id}:j{root(index)}")
        documents = torch.cat(embeddings) if embeddings else torch.empty((0, 512), device=self.model.device)
        query_sum = torch.stack(queries).sum(0) if queries else torch.zeros(512, device=self.model.device)
        relevance = torch.nn.functional.cosine_similarity(documents, query_sum[None, :], dim=-1)
        return EvidenceEncoding(tuple(unit.unit_id for unit in units), documents, relevance,
            tuple(bindings), self.heads, tuple(token_states), tuple(token_sources),
            {"input_tokens": token_count, "compute_ms": (time.perf_counter()-started)*1000, "calls": len(batches)},
            query_sum)

    def close(self):
        self.model = self.heads = None
