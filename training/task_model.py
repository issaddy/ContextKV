"""BGE-M3 query LoRA with history, span, condition-pointer and task heads."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import time

from adapters.history_state import HistoryIndex, TaskPrediction
from controller.context_schema import Condition, Need, RetrievalState, TextRef, VersionIntent
from controller.schema import OnlineTurn, TurnRecord
from controller.serialization import decode
from training.task_data import read_jsonl, select_cohort_rows


RELATIONS = ("standalone", "followup", "correction", "revisit", "topic_switch")
OPERATIONS = ("lookup", "enumerate", "procedure", "compare", "count", "sum", "group", "infer")
UPDATES = ("KEEP", "ADD", "REPLACE", "REMOVE", "COMPARE")
VERSIONS = ("current", "explicit", "as_of", "compare")
TEMPORAL_RELATIONS = ("at", "before", "after")
CONDITION_RELATIONS = ("PREDICATE", "TIME", "VERSION", "PLATFORM", "LOCATION", "SUBJECT")
SPAN_KINDS = ("NEED", "SUBJECT", "CONDITION_KEY", "CONDITION_VALUE", "TIME")
SPAN_LABELS = ("O", *(prefix+kind for kind in SPAN_KINDS for prefix in ("B-", "I-")))


def encode_task(tokenizer, turn, windows, memory):
    ids, refs, groups = [tokenizer.cls_token_id], [None], []
    for window in windows:
        groups.append((window.ref, window.text))
    groups.append((TextRef(turn.turn, "user", 0, len(turn.query)), turn.query))
    ranges = []
    for ref, text in groups:
        prefix = tokenizer(f"Turn {ref.turn_id} {ref.speaker}: ", add_special_tokens=False)["input_ids"]
        ids.extend(prefix)
        refs.extend([None]*len(prefix))
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        begin = len(ids)
        ids.extend(encoded["input_ids"])
        refs.extend(TextRef(ref.turn_id, ref.speaker, ref.start+start, ref.start+end) if start < end else None
                    for start, end in encoded["offset_mapping"])
        ranges.append((begin, len(ids), ref))
        ids.append(tokenizer.sep_token_id)
        refs.append(None)
    prior_conditions = tuple(condition for state in memory.prior_task_states for condition in state.conditions)
    prior_conditions = tuple({condition.condition_id: condition for condition in prior_conditions}.values())

    def positions(spans):
        return [i for i, token in enumerate(refs) if token is not None and any(
            token.turn_id == span.turn_id and token.speaker == span.speaker
            and token.start < span.end and span.start < token.end for span in spans)]

    condition_positions = [positions(condition.value_spans) for condition in prior_conditions]
    task_positions = [positions(tuple(ref for need in state.needs for ref in need.anchor_spans))
                      for state in memory.prior_task_states]
    version_turn_ids = [row.turn_id for row in memory.committed_history]
    version_turn_positions = [[i for i, ref in enumerate(refs) if ref is not None and ref.turn_id == turn_id]
                              for turn_id in version_turn_ids]
    return {"input_ids": ids, "refs": refs, "ranges": ranges, "text_groups": groups,
            "current_turn_id": turn.turn,
            "prior_conditions": prior_conditions, "condition_positions": condition_positions,
            "prior_tasks": memory.prior_task_states, "task_positions": task_positions,
            "version_turn_ids": version_turn_ids, "version_turn_positions": version_turn_positions}


def task_targets(encoded, prediction):
    n = len(encoded["input_ids"])
    targets = {name: [-100]*n for name in ("span", "history", "update", "condition_relation", "pointer", "operation")}
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

    for begin, end, ref in encoded["ranges"]:
        if ref.turn_id == encoded["current_turn_id"]:
            continue
        selected = any(ref.turn_id == chosen.turn_id and ref.speaker == chosen.speaker
                       and ref.start < chosen.end and chosen.start < ref.end for chosen in prediction.history_spans)
        targets["history"][begin:end] = [int(selected)]*(end-begin)
    for need in prediction.needs:
        assign(need.anchor_spans, "NEED", {"operation": OPERATIONS.index(need.operation)})
        assign(need.subject_refs, "SUBJECT")
    condition_ids = [c.condition_id for c in encoded["prior_conditions"]]
    for condition in prediction.conditions:
        pointer = condition_ids.index(condition.target_refs[0])+1 if condition.target_refs else 0
        values = {"update": UPDATES.index(condition.update),
                  "condition_relation": CONDITION_RELATIONS.index(condition.relation_type), "pointer": pointer}
        if condition.update == "REMOVE":
            anchors = ((condition.predicate_span,) if condition.predicate_span is not None else
                       (condition.key_span,) if condition.key_span is not None else condition.value_spans)
            assign(anchors, "CONDITION_KEY", values)
            continue
        if condition.key_span is not None:
            assign((condition.key_span,), "CONDITION_KEY", values)
        assign(condition.value_spans, "CONDITION_VALUE", values)
    assign(prediction.version_intent.time_refs, "TIME")
    tasks = [state.task_id for state in encoded["prior_tasks"]]
    targets.update(relation=RELATIONS.index(prediction.relation), version=VERSIONS.index(prediction.version_intent.mode),
                   temporal_relation=TEMPORAL_RELATIONS.index(prediction.version_intent.temporal_relation),
                   version_pointer=encoded["version_turn_ids"].index(prediction.version_intent.source_turns[0])+1
                   if prediction.version_intent.source_turns else 0,
                   resume=tasks.index(prediction.resume_task_id)+1 if prediction.resume_task_id else 0)
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
            self.pointer_query = nn.Linear(width, width)
            self.pointer_key = nn.Linear(width, width)
            self.null_condition = nn.Parameter(torch.zeros(width))
            self.null_task = nn.Parameter(torch.zeros(width))
            self.null_version_turn = nn.Parameter(torch.zeros(width))

        def forward(self, batch, encoded):
            hidden = self.encoder(**batch).last_hidden_state.detach().float()
            output = {name: head(hidden[:, 0] if name in {"relation", "version", "temporal_relation"} else hidden)
                      for name, head in self.heads.items()}
            output["update"] = output.pop("condition_update")
            output["query"] = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
            condition_keys = [self.null_condition]
            condition_keys.extend(hidden[0, positions].mean(0) if positions else hidden[0, 0]
                                  for positions in encoded["condition_positions"])
            task_keys = [self.null_task]
            task_keys.extend(hidden[0, positions].mean(0) if positions else hidden[0, 0]
                             for positions in encoded["task_positions"])
            output["pointer"] = self.pointer_query(hidden) @ self.pointer_key(torch.stack(condition_keys)).T
            output["resume"] = self.pointer_query(hidden[:, 0]) @ self.pointer_key(torch.stack(task_keys)).T
            version_keys = [self.null_version_turn]
            version_keys.extend(hidden[0, positions].mean(0) if positions else hidden[0, 0]
                                for positions in encoded["version_turn_positions"])
            output["version_pointer"] = self.pointer_query(hidden[:, 0]) @ self.pointer_key(torch.stack(version_keys)).T
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
    relation = RELATIONS[int(output["relation"][0].argmax())]
    version = VERSIONS[int(output["version"][0].argmax())]
    temporal = TEMPORAL_RELATIONS[int(output["temporal_relation"][0].argmax())]
    history = tuple(ref for begin, end, ref in encoded["ranges"] if ref.turn_id != turn.turn
                    and float(output["history"][0, begin:end].softmax(-1)[:, 1].mean()) >= .5)
    subjects = tuple(ref for kind, ref, _ in spans if kind == "SUBJECT")
    conditions = []
    keys = [(ref, i) for kind, ref, i in spans if kind == "CONDITION_KEY"]
    removal_keys = [(ref, i) for ref, i in keys if UPDATES[int(output["update"][0, i].argmax())] == "REMOVE"]
    active_keys = [(ref, i) for ref, i in keys if (ref, i) not in removal_keys]
    groups = {i: (ref, []) for ref, i in active_keys}
    for kind, ref, i in spans:
        if kind != "CONDITION_VALUE":
            continue
        key, anchor = min(active_keys, key=lambda item: abs(item[1]-i)) if active_keys else (ref, i)
        groups.setdefault(anchor, (key, []))[1].append(ref)
    for anchor, (key, values) in groups.items():
        update = UPDATES[int(output["update"][0, anchor].argmax())]
        relation_type = CONDITION_RELATIONS[int(output["condition_relation"][0, anchor].argmax())]
        target = int(output["pointer"][0, anchor].argmax())
        targets = (encoded["prior_conditions"][target-1].condition_id,) if target else ()
        conditions.append(Condition(f"t{turn.turn}:c{len(conditions)}", key, tuple(values), relation_type, key,
                                    targets, targets if update == "REPLACE" else (), update))
    for ref, i in removal_keys:
        target = int(output["pointer"][0, i].argmax())
        targets = (encoded["prior_conditions"][target-1].condition_id,) if target else ()
        if any(condition.update == "REMOVE" and condition.target_refs == targets for condition in conditions):
            continue
        relation_type = CONDITION_RELATIONS[int(output["condition_relation"][0, i].argmax())]
        conditions.append(Condition(f"t{turn.turn}:c{len(conditions)}", relation_type=relation_type,
                                    predicate_span=ref, target_refs=targets, update="REMOVE"))
    def span_positions(span):
        return [i for i, ref in enumerate(encoded["refs"]) if ref is not None
                and ref.turn_id == span.turn_id and ref.speaker == span.speaker
                and ref.start < span.end and span.start < ref.end]

    needs = tuple(Need(f"t{turn.turn}:n{j}", (ref,),
                       OPERATIONS[int(output["operation"][0, span_positions(ref)].mean(0).argmax())],
                       subjects, tuple(c.condition_id for c in conditions if c.update not in {"REMOVE", "KEEP"}))
                  for j, (kind, ref, i) in enumerate(spans) if kind == "NEED")
    times = tuple(ref for kind, ref, _ in spans if kind == "TIME")
    resume = int(output["resume"][0].argmax())
    resume_id = encoded["prior_tasks"][resume-1].task_id if resume and relation == "revisit" else ""
    def span_text(span):
        owner, text = next((ref, text) for ref, text in encoded["text_groups"]
                          if ref.turn_id == span.turn_id and ref.speaker == span.speaker
                          and ref.start <= span.start and span.end <= ref.end)
        return text[span.start-owner.start:span.end-owner.start]

    version_ids = tuple(span_text(ref) for condition in conditions if condition.relation_type == "VERSION"
                        for ref in condition.value_spans)
    version_pointer = int(output["version_pointer"][0].argmax())
    source_turns = (encoded["version_turn_ids"][version_pointer-1],) if version_pointer else ()
    return TaskPrediction(relation, history, tuple(conditions), needs, VersionIntent(version,
        version_ids=version_ids, time_refs=times, source_turns=source_turns, temporal_relation=temporal), resume_id)


class TaskModel:
    def __init__(self, path, *, base_path=None, device="cuda"):
        import torch
        from transformers import AutoTokenizer
        path = Path(path)
        self.manifest = json.loads((path/"task-model.json").read_text())
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_path or self.manifest["base_path"], local_files_only=True)
        self.network = create_task_network(base_path or self.manifest["base_path"], rank=self.manifest["rank"],
            adapter_path=path, device=device, dtype=self.manifest["dtype"])
        weights = torch.load(path/"task-heads.pt", map_location=device, weights_only=True)
        self.network.heads.load_state_dict({name.removeprefix("heads."): value for name, value in weights.items()
                                           if name.startswith("heads.")}, strict=True)
        for prefix in ("pointer_query", "pointer_key"):
            getattr(self.network, prefix).load_state_dict({name.removeprefix(prefix+"."): value
                for name, value in weights.items() if name.startswith(prefix+".")}, strict=True)
        with torch.no_grad():
            self.network.null_condition.copy_(weights["null_condition"])
            self.network.null_task.copy_(weights["null_task"])
            self.network.null_version_turn.copy_(weights["null_version_turn"])
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

    def predict(self, turn, windows, memory):
        import torch
        self._synchronize()
        started = time.perf_counter()
        encoded = encode_task(self.tokenizer, turn, windows, memory)
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
        self.last_metrics = {"input_tokens": len(encoded["input_ids"]), "output_tokens": 0, "elapsed_ms": elapsed}
        return prediction


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
        for name in ("retrieval", "history", "span", "update", "condition_relation", "pointer",
                     "operation", "relation", "version", "temporal_relation", "version_pointer", "resume")}
    temperature = settings.get("retrieval_temperature", .05)
    adapter_params, head_params = [], []
    for name, parameter in network.named_parameters():
        if parameter.requires_grad:
            (adapter_params if name.startswith("encoder.") else head_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": settings.get("adapter_lr", 2e-4)},
        {"params": head_params, "lr": settings.get("head_lr", 1e-4)}], weight_decay=.01)
    logs = []
    span_counts = {kind: Counter() for kind in SPAN_KINDS}
    training_usage = {"task_input_tokens": 0, "history_input_tokens": 0,
                      "document_input_tokens": 0, "query_input_tokens": 0}
    started = time.perf_counter()
    for epoch in range(settings.get("epochs", 1)):
        prior_states = {}
        order.shuffle(conversation_ids)
        for row in (row for identity in conversation_ids for row in conversations[identity]):
            history = tuple(decode(TurnRecord, value) for value in row["actual_history"])
            memory = RetrievalState(len(history), history, tuple(prior_states.get(row["conversation_id"], ())))
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
            encoded = encode_task(tokenizer, turn, windows, memory)
            if len(encoded["input_ids"]) > 8192:
                raise ValueError("task training view exceeds BGE-M3 input budget")
            prediction = decode(TaskPrediction, row["prediction"])
            targets = task_targets(encoded, prediction)
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
    network.encoder.save_pretrained(output/"adapter")
    torch.save({name: value.cpu() for name, value in network.state_dict().items()
                if not name.startswith("encoder.")}, output/"task-heads.pt")
    manifest = {"schema_version": 2, "status": "completed", "base": "BAAI/bge-m3", "base_path": base_path,
        "model_id": "contextkv/task-query-lora", "revision": "study-task-v2",
        "pooling": "CLS then FP32 L2 normalization", "max_length": 8192,
        "document_encoder": {name: dense.manifest["encoder"][name] for name in (
            "model_id", "revision", "max_length", "pooling")},
        "document_encoder_manifest": dense.manifest["encoder"], "document_encoder_training": "frozen",
        "rank": task.get("lora_rank", 16), "dtype": dtype,
        "training_rows": len(labels), "steps": len(logs), "elapsed_ms": (time.perf_counter()-started)*1000,
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
        "gradient_scope": {"query_lora": "retrieval loss only", "task_heads": "task losses on detached encoder features"},
        "loss_weights": loss_weights, "retrieval_temperature": temperature,
        "heads": {"span": SPAN_LABELS, "condition_update": UPDATES, "relation": RELATIONS,
                  "operation": OPERATIONS, "version": VERSIONS, "condition_relation": CONDITION_RELATIONS,
                  "history": [0, 1], "condition_pointer": "prior condition IDs", "task_pointer": "prior task IDs",
                  "temporal_relation": TEMPORAL_RELATIONS, "version_turn_pointer": "actual committed turn IDs"},
        "span_scheme": "parallel BIO channels per span kind; nested need/subject/condition spans retained",
        "span_loss": "equal mean across channels and the BIO classes present in each channel; special and zero-width tokens masked",
        "span_supervision_tokens": {kind: {name: counts[label] for label, name in enumerate(("O", "B", "I"))}
                                    for kind, counts in span_counts.items()},
        "training_order": "seeded conversation shuffle per epoch; ascending turns within each conversation",
        "history_supervision": "selected historical windows only; current question masked",
        "training_config": settings}
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
    for row in evaluation_rows:
        history = tuple(decode(TurnRecord, value) for value in row["actual_history"])
        memory = RetrievalState(len(history), history, tuple(prior.get(row["conversation_id"], ())))
        turn = OnlineTurn(row["conversation_id"], row["task_id"], row["turn"], row["snapshot_id"], row["question"], (), (), "P3")
        expected = decode(TaskPrediction, row["prediction"])
        predicted = model.predict(turn, index.select(turn.query, memory, current_turn_id=turn.turn), memory)
        def condition_items(prediction):
            return {(condition.update, condition.relation_type,
                     tuple((ref.turn_id, ref.speaker, ref.start, ref.end) for ref in condition.value_spans),
                     condition.target_refs) for condition in prediction.conditions}
        predicted_spans = characters(ref for need in predicted.needs for ref in need.anchor_spans)
        expected_spans = characters(ref for need in expected.needs for ref in need.anchor_spans)
        metrics = {"relation_accuracy": float(predicted.relation == expected.relation),
            "version_accuracy": float(predicted.version_intent.mode == expected.version_intent.mode),
            "temporal_relation_accuracy": float(predicted.version_intent.temporal_relation == expected.version_intent.temporal_relation),
            "version_turn_pointer_accuracy": float(predicted.version_intent.source_turns == expected.version_intent.source_turns),
            "history_character_f1": f1(characters(predicted.history_spans), characters(expected.history_spans)),
            "need_character_f1": f1(predicted_spans, expected_spans),
            "condition_update_f1": f1(condition_items(predicted), condition_items(expected)),
            "operation_accuracy": float(tuple(n.operation for n in predicted.needs) == tuple(n.operation for n in expected.needs)),
            "resume_accuracy": float(predicted.resume_task_id == expected.resume_task_id)}
        rows.append({"task_id": row["task_id"], "split": row["split"], "cohort_split": row["cohort_split"],
                     "cohort_manifest_sha256": row["cohort_manifest_sha256"], "question": row["question"],
                     "actual_history": row["actual_history"], "metrics": metrics,
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
            "query_assets": {"P0": query_encoder_binding({"query_adapter_path": task.get("initial_adapter_path", "models/bge-m3-dialogue-query-v1")}),
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
