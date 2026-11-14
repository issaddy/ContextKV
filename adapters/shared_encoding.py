"""One query forward and one forward per evidence batch, shared by task heads."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
import time

from controller.schema import Condition, Need, SourceSpan, TaskState, TextRef, VersionIntent
from controller.templates import ThreadTokenizer
from adapters.jina_encoding import pack_passages, project_markers, merge_rankings

QUERY_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EVIDENCE_MODEL = "jinaai/jina-reranker-v3.5"


class ReadyBatch:
    """One backbone worker batches ready inputs; callers wait only for their result."""
    def __init__(self, execute):
        self.execute, self.lock = execute, Lock()
        self.pending, self.running = [], False
        self.worker = ThreadPoolExecutor(max_workers=1)

    def submit(self, value):
        future = Future()
        with self.lock:
            self.pending.append((value, future))
            if not self.running:
                self.running = True
                self.worker.submit(self._drain)
        return future.result()

    def _drain(self):
        while True:
            with self.lock:
                jobs, self.pending = self.pending, []
                if not jobs:
                    self.running = False
                    return
            try:
                results = self.execute([value for value, _ in jobs])
                for (_, target), result in zip(jobs, results):
                    target.set_result(result)
            except Exception as error:
                for _, target in jobs:
                    target.set_exception(error)

    def close(self):
        self.worker.shutdown(wait=True)


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
    source_titles: tuple = ()

    def text(self, refs):
        merged = []
        for ref in sorted(set(refs), key=lambda r: (r.turn_id, r.speaker, r.start, r.end)):
            if merged and (merged[-1].turn_id, merged[-1].speaker) == (ref.turn_id, ref.speaker) and ref.start <= merged[-1].end:
                merged[-1] = replace(merged[-1], end=max(merged[-1].end, ref.end))
            else:
                merged.append(ref)
        return " ".join(self.texts[(ref.turn_id, ref.speaker)][ref.start:ref.end] for ref in merged)

    @property
    def coverage_vector(self):
        positions = [i for i, ref in enumerate(self.token_refs)
                     if ref is not None and ref.turn_id == self.task.current_turn_id and ref.speaker == "user"]
        return self.token_states[positions].mean(0)

    @property
    def search_texts(self):
        return (self.task.raw_question, *(self.text((*need.anchor_spans, *need.subject_refs,
            *(ref for condition in self.task.conditions if condition.condition_id in need.condition_ids
              for ref in condition.value_spans))) for need in self.task.needs))

    @property
    def evidence_query(self):
        needs = "\n".join(dict.fromkeys(self.search_texts[1:]))
        versions = ", ".join(self.task.version_intent.version_ids)
        return (self.task.raw_question + ("\nRequested evidence:\n"+needs if needs else "")
            + ("\nVersions: "+versions if versions else "")
            + ("\nDocument population:\n"+"\n".join(self.source_titles) if self.source_titles else ""))


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

    def summary(self, selected_ids, units, need_ids, completed=()):
        import torch
        selected = set(selected_ids)
        mask = torch.tensor([identity in selected for identity in self.unit_ids],
                            device=self.embeddings.device, dtype=torch.bool)
        if not len(mask):
            return self.embeddings.new_zeros(16)
        support, known = self.need_support(units, need_ids, completed)
        features = self.heads.features(self.embeddings, self.relevance, mask[None], support, known)[0]
        rest = self.relevance[~mask]
        remaining = self.embeddings.new_tensor([
            float(rest.max()) if len(rest) else 0., float(rest.mean()) if len(rest) else 0.,
            len(selected)/len(mask), len(rest)/len(mask)])
        return torch.cat((features, remaining))

    def need_support(self, units, need_ids, completed=()):
        """Use the shared field bindings as soft support, with computed Needs already known."""
        by_id = {unit.unit_id: unit for unit in units}
        positions = {identity: i for i, identity in enumerate(need_ids)}
        support = self.embeddings.new_zeros((len(self.unit_ids), len(need_ids)))
        for binding in self.bindings:
            if binding.role == "source" and binding.need_id in positions:
                for i, identity in enumerate(self.unit_ids):
                    if any(span.source_key == binding.source_span.source_key
                           and span.start <= binding.source_span.start and binding.source_span.end <= span.end
                           for span in by_id[identity].source_spans):
                        j = positions[binding.need_id]
                        support[i, j] = max(float(support[i, j]), binding.score)
        return support, self.embeddings.new_tensor([identity in completed for identity in need_ids])

    def frontier(self, units):
        """Nested relevance-ranked sets, each containing its source dependencies."""
        by_id = {unit.unit_id: unit for unit in units}
        order = self.relevance.argsort(descending=True).tolist()
        active, seen, sets = [], set(), [()]
        def add(identity):
            if identity in seen:
                return
            seen.add(identity)
            for dependency in by_id[identity].dependencies:
                add(dependency)
            active.append(identity)
        for index in order:
            before = len(active)
            add(self.unit_ids[index])
            if len(active) != before:
                sets.append(tuple(active))
        return tuple(sets)

    def select(self, units, *, body_tokens, cost_model=None, seconds_price=0.0, need_ids=(), completed=()):
        """Price one information frontier, then compile only its chosen context."""
        import torch
        if not self.unit_ids or body_tokens <= 0:
            return ()
        sizes = {unit.unit_id: unit.body_tokens for unit in units}
        candidates = [ids for ids in self.frontier(units) if sum(sizes[i] for i in ids) <= body_tokens]
        masks = torch.tensor([[identity in ids for identity in self.unit_ids] for ids in candidates],
                             device=self.embeddings.device, dtype=torch.bool)
        support, known = self.need_support(units, need_ids, completed)
        with torch.no_grad():
            values = self.heads.utility(self.embeddings, self.relevance, masks, support, known)
            costs = values.new_tensor([cost_model.estimate(ids) if cost_model else 0. for ids in candidates])
            best = int((values-seconds_price*costs/1000).argmax())
        selected = candidates[best]
        return cost_model.order(selected) if cost_model else selected

    def project_population(self, units, *, body_tokens, target_tokens, cost_model=None, seconds_price=0.0):
        """Cover each source, then add query-weighted representative spans within it."""
        import torch
        if not units:
            return ()
        by_id = {unit.unit_id: i for i, unit in enumerate(units)}
        groups = {}
        for i, unit in enumerate(units):
            key = unit.source_spans[0].source_key
            groups.setdefault((key.corpus_id, key.document_id, key.version), []).append(i)
        closures = []
        for unit in units:
            found, pending = set(), [by_id[unit.unit_id]]
            while pending:
                i = pending.pop()
                if i not in found:
                    found.add(i)
                    pending.extend(by_id[identity] for identity in units[i].dependencies)
            closures.append(found)
        selected = set()
        sizes = [unit.body_tokens for unit in units]
        for members in groups.values():
            best = max(members, key=lambda i: float(self.relevance[i]))
            selected.update(closures[best])
        used = sum(sizes[i] for i in selected)
        target_tokens = min(body_tokens, max(used, target_tokens))
        similarity = (self.embeddings@self.embeddings.T).clamp(0, 1)
        same_source = torch.zeros_like(similarity)
        for members in groups.values():
            same_source[torch.tensor(members)[:, None], torch.tensor(members)[None, :]] = 1
        relevance = self.relevance.clamp_min(0)
        affinity = similarity*same_source*relevance[None, :]
        weights = relevance.square()
        represented = affinity[:, sorted(selected)].amax(1)
        def identities(indices):
            return tuple(units[i].unit_id for i in sorted(indices))
        while True:
            price = cost_model.estimate(identities(selected)) if cost_model else 0.
            best, best_score = None, 0.
            for i, closure in enumerate(closures):
                extra = closure-selected
                tokens = sum(sizes[j] for j in extra)
                if not extra or used+tokens > target_tokens:
                    continue
                coverage = torch.maximum(represented, affinity[:, sorted(extra)].amax(1))
                gain = float(((coverage-represented)*weights).sum())
                extra_price = (cost_model.estimate(identities(selected | extra))-price) if cost_model else 0.
                score = (gain-seconds_price*extra_price/1000)/tokens
                if score > best_score:
                    best, best_score = extra, score
            if best is None:
                break
            selected.update(best)
            used += sum(sizes[i] for i in best)
            represented = torch.maximum(represented, affinity[:, sorted(best)].amax(1))
        # Source order preserves qualifiers and the sequence of original facts.
        return identities(selected)

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

    add("Instruct: Retrieve evidence for the current question using the conversation.\nConversation:\n")
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
    if catalogue:
        add("\nAvailable source versions:\n")
        for row in catalogue:
            start, end = add(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            catalog_spans.append((start, end, row))
            add("\n")
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


class QwenDocumentEncoder:
    def __init__(self, model_path, *, device="cpu", dtype="float32", max_length=32768):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer = ThreadTokenizer(self.tokenizer)
        self.model = AutoModel.from_pretrained(model_path, dtype=getattr(torch, dtype)).to(device).eval()
        self.manifest = {"model_id": "Qwen/Qwen3-Embedding-0.6B", "revision": "base", "max_length": max_length,
                         "pooling": "last_nonpadding_token_l2", "model_path": str(model_path)}

    def encode(self, texts):
        return self.encode_with_usage(texts)[0]

    def encode_with_usage(self, texts):
        import torch
        batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        input_tokens = int(batch["attention_mask"].sum())
        batch = batch.to(self.model.device)
        with torch.no_grad():
            states = self.model(**batch, use_cache=False).last_hidden_state
            return torch.nn.functional.normalize(states[:, -1].float(), dim=-1).cpu().numpy(), input_tokens

    def close(self):
        self.model = None


class SharedQueryEncoder:
    def __init__(self, model_path, asset_path, *, device="cuda:0", dtype="bfloat16", train=False, history_tokens=4096):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from adapters.shared_heads import QueryHeads
        self.torch = torch
        self.history_tokens = history_tokens
        self.backbone_lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer = ThreadTokenizer(self.tokenizer)
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

    def forward_input(self, encoded, query_position):
        torch = self.torch
        batch = {key: torch.tensor([value], device=self.model.device) for key, value in encoded.items()}
        states = self.model(**batch, use_cache=False).last_hidden_state[0].float()
        return states, self.heads(states, states[query_position])

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
        from adapters.shared_heads import (CONDITION_TYPES, EDGE_KINDS, FILTER_OPERATORS, OPERATIONS,
            SCOPES, SPAN_KINDS, TEMPORAL_RELATIONS, UPDATES, VERSION_MODES)
        started = time.perf_counter()
        encoded, refs, texts, catalogue_positions = _prepared
        with torch.no_grad():
            states = _states
            current = [i for i, ref in enumerate(refs)
                       if ref is not None and ref.turn_id == current_turn_id and ref.speaker == "user"]
            query_position = current[-1]
            pooled = states[query_position]
            logits = self.heads(states, pooled)
            coverage = self.heads.coverage(torch.nn.functional.layer_norm(
                states[current].mean(0), (states.shape[-1],)))
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
                if self.heads.version(pooled, vector).item() > 0:
                    selected_versions.append(row["version"])
                if self.heads.family(pooled, vector).item() > 0:
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
            # Literal catalogue identifiers are source constraints, independent
            # of the learned intent class. Match complete identifiers only.
            import re
            explicit_versions = tuple(dict.fromkeys(row["version"] for row in catalogue
                if re.search(r"(?<![\w.])"+re.escape(row["version"])+r"(?![\w.])", question, re.IGNORECASE)))
            if explicit_versions:
                selected_versions = list(explicit_versions)
                selected_families = [row["family"] for row in catalogue if row["version"] in explicit_versions]
                if mode == "current":
                    mode = "explicit"
            intent = VersionIntent(mode,
                document_families=tuple(dict.fromkeys(selected_families)),
                version_ids=tuple(dict.fromkeys(selected_versions)) if mode != "current" else (),
                time_refs=tuple(reference(span) for span in spans["time"]),
                temporal_relation=TEMPORAL_RELATIONS[int(logits["temporal_relation"].argmax())])
            task = TaskState(task_id, question, tuple(history_refs), tuple(conditions), needs, intent,
                             tuple(source_scope), current_turn_id, question,
                             coverage_scope=SCOPES[int(coverage.argmax())])
            projected = torch.nn.functional.normalize(self.heads.need_projection(need_vectors), dim=-1)
            return QueryEncoding(task, torch.nn.functional.normalize(pooled, dim=-1), projected,
                states, refs, texts, operators, {"input_tokens": len(encoded["input_ids"]),
                    "compute_ms": (time.perf_counter()-started)*1000, "calls": 1}, condition_vectors,
                frozenset(need.need_id for need, vector in zip(needs, need_vectors)
                          if int(self.heads.join(vector).argmax()) == 1),
                frozenset(need.need_id for need, vector in zip(needs, need_vectors)
                          if int(self.heads.distinct(vector).argmax()) == 1))

    def close(self):
        self.ready.close()
        self.model = self.heads = None


class SharedEvidenceEncoder:
    def __init__(self, model_path, asset_path, *, device="cuda:0", dtype="bfloat16", batch_tokens=32768, train=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from adapters.shared_heads import EvidenceHeads
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer = ThreadTokenizer(self.tokenizer)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
            dtype=getattr(torch, dtype)).to(device)
        self.heads = EvidenceHeads(self.model.config.hidden_size).to(device)
        if not train or asset_path is not None:
            path = Path(asset_path)
            weights = torch.load(path / "heads.pt", map_location=device, weights_only=True)
            if train:
                weights = {**self.heads.state_dict(),
                           **{name: value for name, value in weights.items() if not name.startswith("set_")}}
            self.heads.load_state_dict(weights)
        self.model.train(train)
        self.heads.train(train)
        self.batch_tokens = batch_tokens
        if not train:
            self.heads.set_weights = torch.nn.Parameter(self.heads.set_weights.detach().cpu(), requires_grad=False)
            self.heads.set_bias = torch.nn.Parameter(self.heads.set_bias.detach().cpu(), requires_grad=False)
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
        documents, scores, q = project_markers(self.model, states, doc_positions, query_position)
        return states.float(), documents, scores, addresses, len(ids), q

    def forward_evidence(self, query, units):
        """Train on the same evidence batches and global cosine scores used online."""
        import torch
        outputs = [self.forward_batch(query.evidence_query, batch) for batch in self._batches(query, units)]
        if not outputs:
            return (torch.empty((0, self.model.config.hidden_size), device=self.model.device),
                    torch.empty((0, 512), device=self.model.device),
                    torch.empty(0, device=self.model.device), (), 0)
        documents, scores, _ = merge_rankings([(item[1], item[2], item[5]) for item in outputs])
        return (torch.cat([item[0] for item in outputs]), documents, scores,
                tuple(address for item in outputs for address in item[3]), sum(item[4] for item in outputs))

    def _batches(self, query, units):
        units = tuple(units)
        return pack_passages(units,
            [len(self.tokenizer.encode(unit.text, add_special_tokens=False))+20 for unit in units],
            overhead=len(self.prepare(query.evidence_query, ())[0]), budget=self.batch_tokens)

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
                    documents, scores, query = project_markers(self.model, states, positions, query_position)
                    outputs[owner][tuple(unit.unit_id for unit in units)] = (
                        states.float(), documents, scores, addresses, len(ids), query)
            results = [self._encode(query, units, outputs[index]) for index, (query, units) in enumerate(requests)]
        elapsed = (time.perf_counter()-started)*1000
        total = sum(item.usage["input_tokens"] for item in results)
        for result in results:
            result.usage["compute_ms"] = elapsed*result.usage["input_tokens"]/total if total else 0.
        return results

    def _encode(self, query, units, outputs):
        import torch
        from adapters.shared_heads import FIELD_ROLES
        started = time.perf_counter()
        batches = self._batches(query, units)
        rankings, bindings, binding_vectors, token_states, token_sources = [], [], [], [], []
        token_count = 0
        needs = [(i, need) for i, need in enumerate(query.task.needs)
                 if query.task.coverage_scope != "all_supplied_documents"
                 or need.operation in {"count", "sum", "mean", "min", "max", "group"}]
        with torch.no_grad():
            for batch in batches:
                states, documents, scores, addresses, count, query_vector = outputs[tuple(unit.unit_id for unit in batch)]
                token_count += count
                rankings.append((documents, scores, query_vector))
                if not needs:
                    continue
                source_indices = [i for i, address in enumerate(addresses) if address is not None]
                source_states = states[source_indices]
                sources = tuple(addresses[i] for i in source_indices)
                vectors = query.need_vectors[[i for i, _ in needs]].to(states.device)
                probabilities = self.heads.field_logits(source_states, vectors).softmax(-1)
                confidence, labels = probabilities.max(-1)
                confidence, labels = confidence.cpu(), labels.cpu()
                for need_index, (_, need) in enumerate(needs):
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
        if rankings:
            documents, relevance, query_sum = merge_rankings(rankings)
        else:
            documents = torch.empty((0, 512), device=self.model.device)
            relevance = torch.empty(0, device=self.model.device)
            query_sum = torch.zeros(512, device=self.model.device)
        return EvidenceEncoding(tuple(unit.unit_id for unit in units), documents.cpu(), relevance.cpu(),
            tuple(bindings), self.heads, tuple(token_states), tuple(token_sources),
            {"input_tokens": token_count, "compute_ms": (time.perf_counter()-started)*1000, "calls": len(batches)},
            query_sum.cpu())

    def close(self):
        self.ready.close()
        self.model = self.heads = None
