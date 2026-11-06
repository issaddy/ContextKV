"""BGE support heads and a two-layer evidence/need graph utility model.

Heavy dependencies load only when training or loading an asset. The checkpoint
contains the adapted cross encoder, all support heads and the utility graph.
"""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import time

from adapters.support_model import SetPrediction, SupportPrediction
from controller.context_schema import SourceSpan
from controller.serialization import decode
from training.task_data import select_cohort_rows


SUPPORT_CLASSES = ("complete", "partial", "background", "contradict")
ANSWER_CLASSES = ("answer", "partial", "retrieval_gap", "unanswerable", "underspecified")


def _base_provenance(base_path, model_id):
    base_path = Path(base_path)
    observation = base_path/".cache/huggingface/download/model.safetensors.metadata"
    revision = observation.read_text().splitlines()[0] if observation.exists() else None
    return {"model_id": model_id, "revision": revision,
        "revision_observation": str(observation) if observation.exists() else None,
        "model_card_url": f"https://huggingface.co/{model_id}/blob/{revision or 'main'}/README.md",
        "model_card_path": str(base_path/"README.md") if (base_path/"README.md").exists() else None,
        "license": "Apache-2.0" if model_id == "BAAI/bge-reranker-v2-m3" else None}


def preserve_asset_sources(asset_path, *, parent_asset=None):
    """Attach inherited model card/license and the observed base revision."""
    import shutil
    asset_path = Path(asset_path)
    path = asset_path/"support-model.json"
    metadata = json.loads(path.read_text())
    provenance = metadata.get("base_provenance") or _base_provenance(metadata["base_path"], metadata["base"])
    metadata["base_provenance"] = provenance
    inherited = []
    if parent_asset is not None:
        metadata["parent_support_asset"] = str(parent_asset)
    parent_asset = Path(parent_asset) if parent_asset is not None else None
    if provenance["model_card_path"]:
        card = (parent_asset/"BASE_MODEL_CARD.md" if parent_asset is not None
                and (parent_asset/"BASE_MODEL_CARD.md").exists() else Path(provenance["model_card_path"]))
        shutil.copyfile(card, asset_path/"BASE_MODEL_CARD.md")
        inherited.append("BASE_MODEL_CARD.md")
    if provenance["license"] == "Apache-2.0":
        license_path = (parent_asset/"LICENSE" if parent_asset is not None and (parent_asset/"LICENSE").exists()
                        else Path(__file__).resolve().parents[1]/"docs/third_party/APACHE-2.0.txt")
        shutil.copyfile(license_path, asset_path/"LICENSE")
        inherited.append("LICENSE")
        (asset_path/"NOTICE").write_text(
            "ContextKV support and set utility model.\n"
            f"Base: {provenance['model_id']}\nRevision: {provenance['revision']}\n"
            f"Model card: {provenance['model_card_url']}\n"
            "The adapted encoder and additional support/utility heads were trained on the declared training partition.\n")
        inherited.append("NOTICE")
    metadata["inherited_files"] = inherited
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    return metadata


def create_network(encoder, *, graph_width=128, encoder_autocast_dtype="float32"):
    import torch
    from torch import nn

    class GraphLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(graph_width, 4, batch_first=True)
            self.norm1 = nn.LayerNorm(graph_width)
            self.norm2 = nn.LayerNorm(graph_width)
            self.ffn = nn.Sequential(nn.Linear(graph_width, graph_width*2), nn.GELU(),
                                     nn.Linear(graph_width*2, graph_width))

        def forward(self, nodes, blocked):
            attended, _ = self.attention(nodes, nodes, nodes, attn_mask=blocked,
                                        need_weights=False)
            nodes = self.norm1(nodes+attended)
            return self.norm2(nodes+self.ffn(nodes))

    class SupportNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            hidden = encoder.config.hidden_size
            self.scalar_heads = nn.Linear(hidden, 4)
            self.support_head = nn.Linear(hidden, len(SUPPORT_CLASSES))
            self.span_head = nn.Linear(hidden, 3)
            self.node_projection = nn.Linear(hidden+8, graph_width)
            self.node_type = nn.Embedding(3, graph_width)
            self.context_projection = nn.Linear(8, graph_width)
            self.empty_node = nn.Parameter(torch.zeros(1, hidden+8))
            self.graph = nn.ModuleList([GraphLayer(), GraphLayer()])
            self.utility_head = nn.Linear(graph_width, 1)
            self.answer_head = nn.Linear(graph_width, len(ANSWER_CLASSES))

        def forward(self, **encoded):
            with torch.autocast(device_type=self.encoder.device.type, dtype=torch.bfloat16,
                                enabled=encoder_autocast_dtype == "bfloat16"):
                hidden = self.encoder(**encoded, return_dict=True).last_hidden_state
            pooled = hidden[:, 0].float()
            return {"embedding": pooled, "scalars": self.scalar_heads(pooled),
                    "support": self.support_head(pooled), "spans": self.span_head(hidden.float())}

        def set_forward(self, embeddings, features, blocked, node_types, context):
            raw = torch.cat((embeddings, features), dim=-1) if len(embeddings) else self.empty_node
            nodes = (self.node_projection(raw)+self.node_type(node_types)
                     + self.context_projection(context)).unsqueeze(0)
            for layer in self.graph:
                nodes = layer(nodes, blocked)
            pooled = nodes.mean(dim=1)
            return self.utility_head(pooled).squeeze(), self.answer_head(pooled).squeeze(0)

    return SupportNetwork()


def render_document(example):
    parts, locations, cursor = [], [], 0
    for unit in (*example.units, *example.dependencies):
        header = json.dumps({"unit_id": unit.unit_id, "heading": unit.heading,
            "scope": unit.applicability,
            "spans": [asdict(s) for s in unit.source_spans]}, ensure_ascii=False)+"\n"
        parts.append(header)
        cursor += len(header)
        locations.append((cursor, cursor+len(unit.text), unit))
        parts.append(unit.text+"\n\n")
        cursor += len(unit.text)+2
    return "".join(parts), tuple(locations)


def _graph_inputs(state, units, supports, embeddings, torch, device):
    selected = {unit.unit_id for unit in units}
    rows = [support for support in supports if set(support.unit_ids) <= selected]
    if not rows:
        hidden = embeddings.hidden_size
        return (torch.empty((0, hidden), device=device),
                torch.empty((0, 8), device=device),
                torch.zeros((1, 1), dtype=torch.bool, device=device),
                torch.tensor([0], device=device))
    vectors, features = [], []
    text_tokens = sum(unit.body_tokens for unit in units)
    characters = sum(len(unit.text) for unit in units)
    unique_text = sum(len(text) for text in {unit.text for unit in units})
    redundancy = 1-unique_text/characters if characters else 0.0
    bundle_vectors = [embeddings[state.task_id, s.need_id, tuple(s.unit_ids)] for s in rows]
    bundle_features = [[support.applicability, support.joint_support, support.bridge_value,
            len(support.unit_ids)/3, len(support.source_spans)/max(1, len(units)),
            redundancy, text_tokens/32768, len(state.working_history)/16] for support in rows]
    descriptors, types = [], []
    for need in state.needs:
        matching = [i for i, row in enumerate(rows) if row.need_id == need.need_id]
        vectors.append(torch.stack([bundle_vectors[i] for i in matching]).mean(dim=0)
                       if matching else torch.zeros(embeddings.hidden_size))
        features.append([max((bundle_features[i][j] for i in matching), default=0.0) for j in range(8)])
        descriptors.append(("need", need.need_id))
        types.append(0)
    for unit in units:
        matching = [i for i, row in enumerate(rows) if unit.unit_id in row.unit_ids]
        vectors.append(torch.stack([bundle_vectors[i] for i in matching]).mean(dim=0)
                       if matching else torch.zeros(embeddings.hidden_size))
        features.append([max((bundle_features[i][j] for i in matching), default=0.0) for j in range(8)])
        descriptors.append(("unit", unit.unit_id))
        types.append(1)
    bundle_offset = len(vectors)
    vectors.extend(bundle_vectors)
    features.extend(bundle_features)
    descriptors.extend(("bundle", str(i)) for i in range(len(rows)))
    types.extend([2]*len(rows))
    blocked = [[i != j for j in range(len(vectors))] for i in range(len(vectors))]
    for index, (kind, identity) in enumerate(descriptors[:bundle_offset]):
        for j, row in enumerate(rows):
            connected = row.need_id == identity if kind == "need" else identity in row.unit_ids
            if connected:
                blocked[index][bundle_offset+j] = False
                blocked[bundle_offset+j][index] = False
    return (torch.stack(vectors).to(device),
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(blocked, dtype=torch.bool, device=device),
            torch.tensor(types, device=device))


class EmbeddingCache(dict):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size


class TrainedSupportBackend:
    def __init__(self, network, tokenizer, manifest, *, device, batch_size=8,
                 encoder_dtype="float32"):
        import torch
        self.torch = torch
        self.network = network.to(device).eval()
        self.network.encoder.to(dtype=getattr(torch, encoder_dtype))
        self.tokenizer = tokenizer
        self.device, self.batch_size = device, batch_size
        self.manifest = manifest
        self.max_length = manifest["max_length"]
        self.embeddings = EmbeddingCache(network.encoder.config.hidden_size)
        self._predictions = {}
        self.usage = {"examples": 0, "input_tokens": 0, "output_tokens": 0,
                      "compute_ms": 0.0, "support_ms": 0.0, "utility_ms": 0.0,
                      "oversized_examples": 0}
        self.usage["cache_hits"] = 0
        self.oversized_inputs = []
        self.deployment = {"device": str(device),
            "encoder_dtype": str(next(network.encoder.parameters()).dtype),
            "head_dtype": str(network.scalar_heads.weight.dtype),
            "parameter_bytes": sum(parameter.numel()*parameter.element_size()
                                   for parameter in network.parameters())}

    @classmethod
    def load(cls, path, *, device, batch_size=8, encoder_dtype=None, encoder_autocast_dtype="float32"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        path = Path(path)
        manifest = json.loads((path/"support-model.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(path/"encoder", local_files_only=True)
        encoder = AutoModel.from_pretrained(path/"encoder", local_files_only=True)
        network = create_network(encoder, graph_width=manifest["graph_width"],
                                 encoder_autocast_dtype=encoder_autocast_dtype)
        weights = torch.load(path/"heads.pt", map_location="cpu", weights_only=True)
        for name, module in network.named_children():
            if name != "encoder":
                prefix = name+"."
                module.load_state_dict({key[len(prefix):]: value for key, value in weights.items()
                                        if key.startswith(prefix)})
        with torch.no_grad():
            network.empty_node.copy_(weights["empty_node"])
        return cls(network, tokenizer, manifest, device=device, batch_size=batch_size,
            encoder_dtype=encoder_dtype or manifest.get("inference_encoder_dtype", "float32"))

    def _synchronize(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def predict(self, examples):
        keys, pending = [], {}
        for example in examples:
            key = (example.task_id, example.need_id, example.query, render_document(example)[0])
            keys.append(key)
            if key not in self._predictions:
                pending[key] = example
        self.usage["cache_hits"] += len(examples)-len(pending)
        if pending:
            predictions = self._predict_uncached(tuple(pending.values()))
            for key, example, prediction in zip(pending, pending.values(), predictions, strict=True):
                embedding = (self.embeddings[example.task_id, example.need_id, tuple(u.unit_id for u in example.units)]
                             if prediction is not None else None)
                self._predictions[key] = prediction, embedding
        results = []
        for key, example in zip(keys, examples, strict=True):
            prediction, embedding = self._predictions[key]
            if prediction is not None:
                self.embeddings[example.task_id, example.need_id, tuple(u.unit_id for u in example.units)] = embedding
            results.append(prediction)
        return tuple(results)

    def _predict_uncached(self, examples):
        self._synchronize()
        started = time.perf_counter()
        predictions = [None]*len(examples)
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start:start+self.batch_size]
            rendered = [render_document(example) for example in batch]
            measured = self.tokenizer([example.query for example in batch],
                [text for text, _ in rendered], padding=False, truncation=False)["input_ids"]
            eligible = [index for index, ids in enumerate(measured) if len(ids) <= self.max_length]
            for index, ids in enumerate(measured):
                if len(ids) > self.max_length:
                    example = batch[index]
                    self.oversized_inputs.append({"task_id": example.task_id, "need_id": example.need_id,
                        "unit_ids": [unit.unit_id for unit in example.units], "input_tokens": len(ids),
                        "encoder_limit": self.max_length})
                    self.usage["oversized_examples"] += 1
            batch = tuple(batch[index] for index in eligible)
            rendered = [rendered[index] for index in eligible]
            if not batch:
                continue
            encoded = self.tokenizer([example.query for example in batch],
                [text for text, _ in rendered], padding=True, truncation=False,
                return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping")
            sequence_ids = [encoded.sequence_ids(i) for i in range(len(batch))]
            self.usage["input_tokens"] += int(encoded["attention_mask"].sum())
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                output = self.network(**encoded)
            scalars = output["scalars"].sigmoid().cpu().tolist()
            statuses = output["support"].argmax(dim=-1).cpu().tolist()
            span_tags = output["spans"].argmax(dim=-1).cpu().tolist()
            for index, example in enumerate(batch):
                key = (example.task_id, example.need_id, tuple(u.unit_id for u in example.units))
                self.embeddings[key] = output["embedding"][index].detach().cpu()
                selected = []
                for token, tag in enumerate(span_tags[index]):
                    left, right = offsets[index, token].tolist()
                    if tag == 0 or sequence_ids[index][token] != 1 or right <= left:
                        continue
                    for body_start, body_end, unit in rendered[index][1]:
                        if right <= body_start or left >= body_end:
                            continue
                        for span in unit.source_spans:
                            assert span.end-span.start == len(unit.text)
                            original_start = span.start+max(left, body_start)-body_start
                            original_end = span.start+min(right, body_end)-body_start
                            selected.append(SourceSpan(f"{span.span_id}:support:{original_start}:{original_end}",
                                span.source_key, original_start, original_end))
                spans = []
                for span in selected:
                    if spans and spans[-1].source_key == span.source_key and spans[-1].end == span.start:
                        previous = spans.pop()
                        spans.append(SourceSpan(f"{previous.span_id}:through:{span.end}", previous.source_key,
                                                previous.start, span.end))
                    else:
                        spans.append(span)
                applicability, direct, joint, bridge = scalars[index]
                disposition = SUPPORT_CLASSES[statuses[index]]
                predictions[start+eligible[index]] = SupportPrediction(applicability, direct, joint, bridge,
                    disposition, tuple(spans))
            self.usage["examples"] += len(batch)
        self._synchronize()
        elapsed = (time.perf_counter()-started)*1000
        self.usage["support_ms"] += elapsed
        self.usage["compute_ms"] += elapsed
        return tuple(predictions)

    def predict_set(self, state, units, supports, *, context=(0.0,)*8):
        self._synchronize()
        started = time.perf_counter()
        tensors = _graph_inputs(state, units, supports, self.embeddings,
                               self.torch, self.device)
        if self.manifest.get("utility_input_scope") == "evidence_set_only":
            context = (0.0,)*8
        with self.torch.inference_mode():
            quality, answer = self.network.set_forward(*tensors,
                self.torch.tensor(context, dtype=self.torch.float32, device=self.device))
        result = SetPrediction(float(quality.sigmoid()), ANSWER_CLASSES[int(answer.argmax())])
        self._synchronize()
        elapsed = (time.perf_counter()-started)*1000
        self.usage["utility_ms"] += elapsed
        self.usage["compute_ms"] += elapsed
        return result

    def release_task(self, task_id):
        for key in tuple(self._predictions):
            if key[0] == task_id:
                del self._predictions[key]
        for key in tuple(self.embeddings):
            if key[0] == task_id:
                del self.embeddings[key]

    def close(self):
        self._predictions.clear()
        self.embeddings.clear()
        del self.network
        if str(self.device).startswith("cuda"):
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()


def _save(network, tokenizer, directory, metadata, *, encoder_source=None, parent_asset=None):
    import torch
    import os
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if encoder_source is None:
        network.encoder.save_pretrained(directory/"encoder", safe_serialization=True)
        tokenizer.save_pretrained(directory/"encoder")
    else:
        (directory/"encoder").symlink_to(os.path.relpath(Path(encoder_source).resolve(), directory.resolve()),
                                        target_is_directory=True)
    torch.save({key: value.cpu() for key, value in network.state_dict().items()
                if not key.startswith("encoder.")}, directory/"heads.pt")
    (directory/"support-model.json").write_text(json.dumps(metadata, ensure_ascii=False,
                                                         indent=2)+"\n")
    preserve_asset_sources(directory, parent_asset=parent_asset or (Path(encoder_source).parent if encoder_source is not None else None))


def train_support(examples_path, parent_asset, output_dir, *, cohort_manifest, device="cuda:0", epochs=1,
                  batch_size=4, learning_rate=1e-4, encoder_learning_rate=1e-5,
                  max_length=8192, seed=42, base_path=None,
                  encoder_autocast_dtype="float32", inference_encoder_dtype="float32"):
    """Train reviewed support labels from an explicit parent or local BGE base."""
    if parent_asset is None and base_path is None:
        raise ValueError("base_path is required when parent_asset is None")
    complete_started = time.perf_counter()
    import torch
    from adapters.support_model import SupportInput
    from controller.context_schema import EvidenceUnit
    torch.manual_seed(seed)
    initialization_started = time.perf_counter()
    if parent_asset is not None:
        backend = TrainedSupportBackend.load(parent_asset, device=device, batch_size=batch_size,
            encoder_dtype="float32", encoder_autocast_dtype=encoder_autocast_dtype)
        initialization = {"mode": "parent_support_asset", "parent_support_asset": str(parent_asset)}
    else:
        from transformers import AutoModel, AutoTokenizer
        base_path = str(Path(base_path))
        model_id = "BAAI/bge-reranker-v2-m3"
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        encoder = AutoModel.from_pretrained(base_path, local_files_only=True)
        manifest = {"base": model_id, "base_path": base_path,
            "base_provenance": _base_provenance(base_path, model_id),
            "graph_width": 128, "max_length": max_length}
        network = create_network(encoder, graph_width=manifest["graph_width"],
            encoder_autocast_dtype=encoder_autocast_dtype)
        backend = TrainedSupportBackend(network, tokenizer, manifest, device=device,
            batch_size=batch_size, encoder_dtype="float32")
        initialization = {"mode": "base_encoder_new_heads", "base_path": base_path}
    initialization_ms = (time.perf_counter()-initialization_started)*1000
    network, tokenizer = backend.network, backend.tokenizer
    examples = select_cohort_rows((json.loads(line) for line in Path(examples_path).read_text().splitlines() if line),
                                 cohort_manifest, "train")
    assert examples
    assert all(example["label_origin"] != "pending_teacher" for example in examples)
    fitting, oversized = [], []
    for row in examples:
        example = SupportInput(row["task_id"], row["need_id"], row["query"],
            tuple(decode(EvidenceUnit, unit) for unit in row["units"]),
            tuple(decode(EvidenceUnit, unit) for unit in row.get("dependencies", [])))
        document, _ = render_document(example)
        input_tokens = len(tokenizer(row["query"], document, truncation=False)["input_ids"])
        if input_tokens <= max_length:
            fitting.append(row)
        else:
            oversized.append({"task_id": row["task_id"], "need_id": row["need_id"],
                "input_tokens": input_tokens, "encoder_limit": max_length})
    examples = fitting
    assert examples
    encoder_parameters = list(network.encoder.parameters())
    head_parameters = [value for name, value in network.named_parameters()
                       if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": encoder_learning_rate},
                                  {"params": head_parameters, "lr": learning_rate}])
    network.train()
    started = time.perf_counter()
    losses, token_count = [], 0
    for epoch in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(order), batch_size):
            rows = [examples[index] for index in order[start:start+batch_size]]
            inputs = [SupportInput(row["task_id"], row["need_id"], row["query"],
                tuple(decode(EvidenceUnit, unit) for unit in row["units"]),
                tuple(decode(EvidenceUnit, unit) for unit in row.get("dependencies", [])))
                for row in rows]
            documents = [render_document(example) for example in inputs]
            encoded = tokenizer([row["query"] for row in rows], [d[0] for d in documents],
                padding=True, truncation=False,
                return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping")
            span_labels = torch.full(encoded["input_ids"].shape, -100, dtype=torch.long)
            for index, row in enumerate(rows):
                sequence_ids = encoded.sequence_ids(index)
                for token, (left, right) in enumerate(offsets[index].tolist()):
                    if sequence_ids[token] != 1 or right <= left:
                        continue
                    for body_start, body_end, unit in documents[index][1]:
                        if left < body_end and right > body_start:
                            span_labels[index, token] = 0
                            for gold in row["labels"]["source_spans"]:
                                for source in unit.source_spans:
                                    if gold["source_key"] != asdict(source.source_key):
                                        continue
                                    lo = source.start+left-body_start
                                    hi = source.start+right-body_start
                                    if lo < gold["end"] and hi > gold["start"]:
                                        span_labels[index, token] = 1 if lo <= gold["start"] else 2
            token_count += int(encoded["attention_mask"].sum())
            output = network(**{key: value.to(device) for key, value in encoded.items()})
            targets = torch.tensor([[row["labels"][name] for name in (
                "applicability", "direct_support", "joint_support", "bridge_value")]
                for row in rows], dtype=torch.float32, device=device)
            status = torch.tensor([SUPPORT_CLASSES.index(row["labels"]["disposition"])
                                   for row in rows], device=device)
            valid_span_tokens = int((span_labels != -100).sum())
            span_loss = torch.nn.functional.cross_entropy(output["spans"].reshape(-1, 3),
                span_labels.to(device).reshape(-1), reduction="sum")/max(1, valid_span_tokens)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(output["scalars"], targets)
                    + torch.nn.functional.cross_entropy(output["support"], status)
                    + span_loss)
            for index, row in enumerate(rows):
                characters = sum(len(unit["text"]) for unit in row["units"])
                unique_characters = sum(len(text) for text in {unit["text"] for unit in row["units"]})
                redundancy = 1-unique_characters/characters if characters else 0.0
                features = torch.tensor([[*output["scalars"][index, [0, 2, 3]].sigmoid().detach().tolist(),
                    len(row["units"])/3, len(row["labels"]["source_spans"])/max(1, len(row["units"])),
                    redundancy, sum(unit["body_tokens"] for unit in row["units"])/32768,
                    row.get("history_fragment_count", 0)/16]], device=device)
                node_count = len(row["units"])+2
                adjacency = torch.ones((node_count, node_count), dtype=torch.bool, device=device)
                adjacency.fill_diagonal_(False)
                adjacency[-1, :] = False
                adjacency[:, -1] = False
                quality, answer = network.set_forward(output["embedding"][index:index+1].expand(node_count, -1),
                    features.expand(node_count, -1), adjacency,
                    torch.tensor([0, *[1]*len(row["units"]), 2], device=device),
                    torch.zeros(8, device=device))
                if row["labels"]["quality"] is not None:
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        quality, torch.tensor(row["labels"]["quality"], dtype=torch.float32, device=device))/len(rows)
                if row["labels"]["answer_disposition"] is not None:
                    loss = loss + torch.nn.functional.cross_entropy(answer.unsqueeze(0), torch.tensor([
                        ANSWER_CLASSES.index(row["labels"]["answer_disposition"])], device=device))/len(rows)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    metadata = {**backend.manifest,
        "parent_support_asset": str(parent_asset) if parent_asset is not None else None, "max_length": max_length,
        "training_encoder_autocast_dtype": encoder_autocast_dtype,
        "training_parameter_dtypes": sorted({str(parameter.dtype) for parameter in network.parameters()}),
        "optimizer_state_dtypes": sorted({str(value.dtype) for state in optimizer.state.values()
                                           for value in state.values() if isinstance(value, torch.Tensor)}),
        "inference_encoder_dtype": inference_encoder_dtype, "inference_head_dtype": "float32",
        "training_partition": {"split": "train", "examples": len(examples),
            "history_origins": dict(Counter(row.get("history_kind", "explicit_training_fixture") for row in examples))},
        "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
        "support_classes": SUPPORT_CLASSES, "answer_classes": ANSWER_CLASSES,
        "support_training": {"initialization": initialization, "initialization_ms": initialization_ms,
            "examples_path": str(examples_path), "examples": len(examples),
            "epochs": epochs, "batch_size": batch_size, "seed": seed,
            "head_learning_rate": learning_rate, "encoder_learning_rate": encoder_learning_rate,
            "label_origins": dict(Counter(row["label_origin"] for row in examples)),
            "oversized_examples": oversized,
            "losses": losses, "input_tokens": token_count,
            "model_setup_ms": (started-complete_started)*1000,
            "elapsed_ms": (time.perf_counter()-started)*1000},
        "utility_training": {"stage": "support_supervision",
            "examples": sum(row["labels"]["quality"] is not None for row in examples),
            "answer_state_examples": sum(row["labels"]["answer_disposition"] is not None for row in examples),
            "answer_state_origins": dict(Counter(row.get("disposition_supervision_origin", "explicit_labels") for row in examples))}}
    save_started = time.perf_counter()
    _save(network, tokenizer, output_dir, metadata, parent_asset=parent_asset)
    metadata = json.loads((Path(output_dir)/"support-model.json").read_text())
    metadata["save_ms"] = (time.perf_counter()-save_started)*1000
    metadata["complete_ms"] = (time.perf_counter()-complete_started)*1000
    (Path(output_dir)/"support-model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    return metadata


def train_set_utility(preferences_path, asset_path, output_dir, *, cohort_manifest, device="cuda:0",
                      epochs=3, learning_rate=1e-4, seed=42):
    """Learn pairwise actual-answer utility on cached cross-encoder graph nodes.

Rows carry positive/negative graphs from the same real question/history and
the underlying reader answers and component rewards. Cross-encoder inputs are
cached once before optimizing the graph and disposition heads.
"""
    import torch
    from adapters.support_model import SupportInput
    from controller.context_schema import EvidenceUnit, SupportBundle, TaskState
    complete_started = time.perf_counter()
    torch.manual_seed(seed)
    backend = TrainedSupportBackend.load(asset_path, device=device)
    rows = select_cohort_rows((json.loads(line) for line in Path(preferences_path).read_text().splitlines() if line),
                             cohort_manifest, "train")
    assert rows
    assert all(row["label_origin"] == "actual_reader_answers" for row in rows)
    preprocessing_started = time.perf_counter()
    graph_pairs, candidate_graphs = [], {}
    cache_hits = 0
    for row in rows:
        state = decode(TaskState, row["task_state"])
        graphs = []
        for side in ("preferred", "dispreferred"):
            item = row[side]
            candidate_id = (row["task_id"], item["response"]["attempt_id"])
            if candidate_id in candidate_graphs:
                graphs.append(candidate_graphs[candidate_id])
                cache_hits += 1
                continue
            assert item["reader_model"] == "qwen3_32b_nonthinking"
            inputs = tuple(SupportInput(state.task_id, example["need_id"], example["query"],
                tuple(decode(EvidenceUnit, u) for u in example["units"]),
                tuple(decode(EvidenceUnit, u) for u in example.get("dependencies", [])))
                for example in item["support_inputs"])
            backend.predict(inputs)
            units = tuple(decode(EvidenceUnit, u) for u in item["units"])
            supports = tuple(decode(SupportBundle, s) for s in item["supports"])
            graph = _graph_inputs(state, units, supports, backend.embeddings, torch, device)
            context = torch.zeros(8, dtype=torch.float32, device=device)
            entry = ((*graph, context), ANSWER_CLASSES.index(item["disposition"])
                     if item["disposition"] is not None else None)
            candidate_graphs[candidate_id] = entry
            graphs.append(entry)
        graph_pairs.append(graphs)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    preprocessing = {"unique_candidates": len(candidate_graphs), "cache_hits": cache_hits,
        "elapsed_ms": (time.perf_counter()-preprocessing_started)*1000}
    print(json.dumps({"utility_graph_preprocessing": preprocessing}), flush=True)
    network = backend.network
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(not name.startswith(("encoder.", "scalar_heads.", "support_head.", "span_head.")))
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=learning_rate)
    losses = []
    network.train()
    started = time.perf_counter()
    for epoch in range(epochs):
        for index in torch.randperm(len(graph_pairs)).tolist():
            (positive, pos_class), (negative, neg_class) = graph_pairs[index]
            pos, pos_answer = network.set_forward(*positive)
            neg, neg_answer = network.set_forward(*negative)
            loss = -torch.nn.functional.logsigmoid(pos-neg)
            if pos_class is not None:
                loss = loss + torch.nn.functional.cross_entropy(pos_answer.unsqueeze(0), torch.tensor([pos_class], device=device))
            if neg_class is not None:
                loss = loss + torch.nn.functional.cross_entropy(neg_answer.unsqueeze(0), torch.tensor([neg_class], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    metadata = {**backend.manifest, "parent_support_asset": str(asset_path),
        "utility_input_scope": "evidence_set_only", "utility_training": {
        "preferences_path": str(preferences_path), "pairs": len(rows), "epochs": epochs,
        "seed": seed, "learning_rate": learning_rate, "losses": losses,
        "synthetic_fixture_count": sum(bool(row.get("synthetic_fixture")) for row in rows),
        "elapsed_ms": (time.perf_counter()-complete_started)*1000,
        "optimization_ms": (time.perf_counter()-started)*1000,
        "graph_preprocessing": preprocessing,
        "encoding_usage": backend.usage, "frozen_encoder_dtype": backend.deployment["encoder_dtype"],
        "head_parameter_dtypes": sorted({str(parameter.dtype) for parameter in network.parameters() if parameter.requires_grad})}}
    _save(network, backend.tokenizer, output_dir, metadata,
          encoder_source=Path(asset_path)/"encoder")
    return metadata
