"""Train RCEM/task heads and shared field/set heads from source-grounded supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from adapters.shared_encoding import SharedEvidenceEncoder, SharedQueryEncoder, query_input
from controller.schema import Condition, EvidenceUnit, Need, SourceSpan, TaskState, TextRef, VersionIntent
from controller.schema import TurnRecord, history_messages
from controller.serialization import decode


def rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def conversation(row):
    if "actual_history" in row:
        records = tuple(decode(TurnRecord, item) for item in row["actual_history"])
        return tuple((record.turn_id, message.role, message.content) for record in records
                     for message in history_messages((record,)))
    return tuple((item["turn_id"], item["speaker"], item["text"]) for item in row["history"])


def catalogue(row):
    return tuple(row.get("catalogue", ()))


def task_label(row):
    raw = row["task_state"]
    return TaskState(row["task_id"], row["question"],
        tuple(decode(TextRef, ref) for ref in raw.get("working_history", ())),
        tuple(decode(Condition, item) for item in raw["conditions"]),
        tuple(decode(Need, item) for item in raw["needs"]),
        decode(VersionIntent, raw["version_intent"]), current_turn_id=row["turn"], query=row["question"])


def task_loss(encoder, row):
    import torch
    from torch.nn import functional as F
    from adapters.shared_heads import (CONDITION_TYPES, EDGE_KINDS, FILTER_OPERATORS, OPERATIONS,
        SCOPES, SPAN_KINDS, TEMPORAL_RELATIONS, UPDATES, VERSION_MODES)
    task = task_label(row)
    inputs, refs, _, catalog_positions = query_input(encoder.tokenizer, conversation(row), row["question"],
        row["turn"], catalogue(row), history_tokens=encoder.history_tokens, history_turns=row.get("history_turns", ()))
    states, logits = encoder.forward_input(inputs)
    device = states.device
    labels = torch.zeros((len(refs), len(SPAN_KINDS)), dtype=torch.long, device=device)
    labels[[i for i, ref in enumerate(refs) if ref is None]] = -100

    def positions(spans):
        return [i for i, token in enumerate(refs) if token is not None and any(
            token.turn_id == ref.turn_id and token.speaker == ref.speaker
            and token.start < ref.end and ref.start < token.end for ref in spans)]

    def tag(spans, kind):
        for span in spans:
            indices = positions((span,))
            labels[indices, SPAN_KINDS.index(kind)] = 2
            labels[indices[0], SPAN_KINDS.index(kind)] = 1

    for need in task.needs:
        tag(need.anchor_spans, "need")
        tag(need.subject_refs, "subject")
    for condition in task.conditions:
        tag((condition.predicate_span or condition.key_span,), "condition")
        tag(condition.value_spans, "value")
    tag(task.version_intent.time_refs, "time")
    for version in task.version_intent.version_ids:
        for turn_id, speaker, text in conversation(row) + ((row["turn"], "user", row["question"]),):
            start = text.find(version)
            if start >= 0:
                span = TextRef(turn_id, speaker, start, start+len(version))
                if positions((span,)):
                    tag((span,), "version")
    losses = [F.cross_entropy(logits["span"].reshape(-1, 3), labels.reshape(-1),
                              weight=states.new_tensor([.1, 1., 1.]))]
    historical = [i for i, ref in enumerate(refs) if ref is not None and ref.turn_id != row["turn"]]
    if historical:
        wanted = set(positions(task.working_history))
        targets = states.new_tensor([i in wanted for i in historical])
        losses.append(F.binary_cross_entropy_with_logits(logits["history"][historical], targets))
    losses.append(F.cross_entropy(logits["version_mode"].unsqueeze(0), torch.tensor(
        [VERSION_MODES.index(task.version_intent.mode)], device=device)))
    losses.append(F.cross_entropy(logits["temporal_relation"].unsqueeze(0), torch.tensor(
        [TEMPORAL_RELATIONS.index(task.version_intent.temporal_relation)], device=device)))

    def pool(spans):
        return logits["features"][positions(spans)].mean(0)

    need_vectors = torch.stack([pool(need.anchor_spans) for need in task.needs]) if task.needs else states[:0]
    condition_vectors = (torch.stack([pool((condition.predicate_span or condition.key_span,))
                                     for condition in task.conditions]) if task.conditions else states[:0])
    subjects = tuple(dict.fromkeys(ref for need in task.needs for ref in need.subject_refs))
    subject_vectors = torch.stack([pool((ref,)) for ref in subjects]) if subjects else states[:0]
    values = tuple(dict.fromkeys(ref for condition in task.conditions for ref in condition.value_spans))
    value_vectors = torch.stack([pool((ref,)) for ref in values]) if values else states[:0]
    for name, expected in (
        ("operation", [OPERATIONS.index(need.operation) for need in task.needs]),
        ("scope", [SCOPES.index(need.scope) for need in task.needs]),
        ("join", [need.need_id in row.get("join_needs", ()) for need in task.needs]),
        ("distinct", [need.need_id in row.get("distinct_needs", ()) for need in task.needs])):
        if expected:
            losses.append(F.cross_entropy(getattr(encoder.heads, name)(need_vectors), torch.tensor(expected, device=device, dtype=torch.long)))
    for name, expected in (
        ("condition_type", [CONDITION_TYPES.index(condition.relation_type) for condition in task.conditions]),
        ("update", [UPDATES.index(condition.update) for condition in task.conditions])):
        if expected:
            losses.append(F.cross_entropy(getattr(encoder.heads, name)(condition_vectors), torch.tensor(expected, device=device)))
    operator_labels = row.get("filter_operators", {})
    for index, condition in enumerate(task.conditions):
        if condition.condition_id in operator_labels:
            target = torch.tensor([FILTER_OPERATORS.index(operator_labels[condition.condition_id])], device=device)
            losses.append(F.cross_entropy(encoder.heads.filter_operator(condition_vectors[index:index+1]), target))
    for name, left, right, expected in (
        ("subject", need_vectors, subject_vectors, [[ref in need.subject_refs for ref in subjects] for need in task.needs]),
        ("condition", need_vectors, condition_vectors, [[condition.condition_id in need.condition_ids for condition in task.conditions] for need in task.needs]),
        ("predecessor", need_vectors, need_vectors, [[other.need_id in need.predecessor_need_ids for other in task.needs] for need in task.needs]),
        ("value", condition_vectors, value_vectors, [[ref in condition.value_spans for ref in values] for condition in task.conditions])):
        if len(left) and len(right):
            predicted = encoder.heads.edges(left, right)[:, :, EDGE_KINDS.index(name)]
            losses.append(F.binary_cross_entropy_with_logits(predicted, states.new_tensor(expected)))
    for row_metadata, indices in catalog_positions:
        vector = states[indices].mean(0)
        for name, field, expected in (("version", "version", task.version_intent.version_ids),
                                      ("family", "family", task.version_intent.document_families)):
            value = getattr(encoder.heads, name)(states[-1], vector).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(value, states.new_tensor(row_metadata[field] in expected)))

    # Conversational/rewrite and ordinary-query identity pairs share the frozen base space.
    # Documents are always encoded with adapters disabled, so their space stays fixed directly.
    instruction = "Instruct: Retrieve evidence for the question.\nQuery: "
    ordinary = encoder.tokenizer([instruction+row["question"]], return_tensors="pt").to(encoder.model.device)
    ordinary_states = encoder.model(**ordinary, use_cache=False).last_hidden_state[:, -1].float()
    predicted = encoder.heads.need_projection(need_vectors)
    adapted = F.normalize(torch.cat((states[-1:], predicted, ordinary_states)), dim=-1)
    targets = [row["rewritten_query"], *row.get("need_queries", ()), row["question"]]
    batch = encoder.tokenizer([instruction+text for text in targets],
                              padding=True, return_tensors="pt").to(encoder.model.device)
    with torch.no_grad(), encoder.model.disable_adapter():
        base = encoder.model(**batch, use_cache=False).last_hidden_state[:, -1].float()
        base = F.normalize(base, dim=-1)
    losses.append(torch.linalg.vector_norm(adapted-base, dim=-1).mean())
    losses.append((F.pdist(adapted)-F.pdist(base)).abs().mean())
    return torch.stack(losses).mean()


def train_query(args):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(args.seed)
    encoder = SharedQueryEncoder(args.model, None, device=args.device, train=True, history_tokens=args.history_tokens)
    encoder.model = get_peft_model(encoder.model, LoraConfig(r=args.rank, lora_alpha=args.rank*2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="FEATURE_EXTRACTION"))
    parameters = [parameter for module in (encoder.model, encoder.heads) for parameter in module.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    training = [row for row in rows(args.data) if row["split"] == "train"]
    random.seed(args.seed)
    steps, total = 0, 0.
    for _ in range(args.epochs):
        random.shuffle(training)
        for row in training:
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(encoder, row)
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    encoder.model.save_pretrained(output / "adapter")
    torch.save(encoder.heads.state_dict(), output / "heads.pt")
    (output / "training.json").write_text(json.dumps({"model": args.model, "states": len(training),
        "steps": steps, "loss": total/steps, "method": "rcem_shared_task_structure_alignment", "history_tokens": args.history_tokens}, indent=2)+"\n")


def field_loss(encoder, query, units, row):
    import torch
    from torch.nn import functional as F
    from adapters.shared_heads import FIELD_ROLES
    states, embeddings, relevance, addresses, _ = encoder.forward_evidence(query, units)
    source_indices = [i for i, address in enumerate(addresses) if address is not None]
    states = states[source_indices]
    addresses = [addresses[i] for i in source_indices]
    logits = encoder.heads.field_logits(states, query.need_vectors.to(states.device))
    targets = torch.zeros(logits.shape[:-1], device=states.device, dtype=torch.long)
    needs = {need.need_id: index for index, need in enumerate(query.task.needs)}
    bindings, vectors = [], []
    for item in row["bindings"]:
        span = decode(SourceSpan, item["source_span"])
        indices = [i for i, (_, key, left, right) in enumerate(addresses)
                   if key == span.source_key and left < span.end and span.start < right]
        targets[needs[item["need_id"]], indices, FIELD_ROLES.index(item["role"])] = 2
        targets[needs[item["need_id"]], indices[0], FIELD_ROLES.index(item["role"])] = 1
        vectors.append(states[indices].mean(0))
        bindings.append(item)
    losses = ([F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1),
                              weight=states.new_tensor([.1, 1., 1.]))] if query.task.needs and len(states) else [])
    conditions = {condition.condition_id: index for index, condition in enumerate(query.task.conditions)}
    for binding, vector in zip(bindings, vectors):
        if binding.get("condition_id"):
            candidates = encoder.heads.query(query.condition_vectors.to(states.device))
            score = encoder.heads.binding(candidates, vector.expand_as(candidates))[:, FIELD_ROLES.index(binding["role"])]
            losses.append(F.cross_entropy(score.unsqueeze(0), torch.tensor([conditions[binding["condition_id"]]], device=states.device)))
    for left, (binding, vector) in enumerate(zip(bindings, vectors)):
        if binding["role"] != "entity":
            continue
        for other, other_vector in zip(bindings[left+1:], vectors[left+1:]):
            if other["role"] == "entity" and other["need_id"] == binding["need_id"]:
                target = bool(binding.get("relation_group")) and binding["relation_group"] == other.get("relation_group")
                losses.append(F.binary_cross_entropy_with_logits(encoder.heads.join(vector, other_vector).squeeze(-1),
                                                                  states.new_tensor(target)))
    return torch.stack(losses).mean()


def train_evidence(args):
    import torch
    torch.manual_seed(args.seed)
    query_encoder = SharedQueryEncoder(args.query_model, args.query_asset, device=args.device, history_tokens=args.history_tokens)
    encoder = SharedEvidenceEncoder(args.model, args.init_asset, device=args.device, train=True)
    encoder.model.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(encoder.heads.parameters(), lr=args.learning_rate)
    training = [row for row in rows(args.data) if row["split"] == "train"
                and (bool(row["task_state"]["needs"] and row["units"]) if args.field_only else bool(row.get("sets")))]
    steps, total = 0, 0.
    if not args.field_only:
        # The set collector uses inferred tasks. Encode that same query view once;
        # only the lightweight set head changes across fitting epochs.
        encoded = []
        for row in training:
            query = query_encoder.encode(conversation(row), row["question"], row["task_id"], row["turn"],
                catalogue=catalogue(row), history_turns=row.get("history_turns", ()))
            units = tuple(decode(EvidenceUnit, item) for item in row["units"])
            with torch.no_grad():
                _, embeddings, relevance, _, _ = encoder.forward_evidence(query, units)
            masks = embeddings.new_tensor([[unit.unit_id in item["unit_ids"] for unit in units] for item in row["sets"]])
            qualities = embeddings.new_tensor([item["quality"] for item in row["sets"]])
            encoded.append((embeddings, relevance, masks, qualities))
        optimizer = torch.optim.AdamW([*encoder.heads.set_item.parameters(), *encoder.heads.set_utility.parameters()],
                                      lr=args.learning_rate)
        for _ in range(args.epochs):
            for example in encoded:
                optimizer.zero_grad(set_to_none=True)
                loss = encoder.heads.loss(*example)
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
    else:
        for _ in range(args.epochs):
            for row in training:
                query = query_encoder.encode(conversation(row), row["question"], row["task_id"], row["turn"],
                    catalogue=catalogue(row), history_turns=row.get("history_turns", ()))
                # Field supervision uses annotated Need ownership on the same visible input.
                task = task_label(row)
                with torch.no_grad():
                    features = query_encoder.heads.contextualize(query.token_states, query.token_states[-1])
                def pool(refs):
                    indices = [i for i, token in enumerate(query.token_refs) if token is not None and any(
                        token.turn_id == ref.turn_id and token.speaker == ref.speaker
                        and token.start < ref.end and ref.start < token.end for ref in refs)]
                    return features[indices].mean(0)
                query.task = task
                query.need_vectors = (torch.nn.functional.normalize(query_encoder.heads.need_projection(
                    torch.stack([pool(need.anchor_spans) for need in task.needs])), dim=-1).detach()
                    if task.needs else query.token_states[:0])
                query.condition_vectors = torch.stack([pool((condition.predicate_span or condition.key_span,))
                    for condition in task.conditions]).detach() if task.conditions else query.token_states[:0]
                units = tuple(decode(EvidenceUnit, item) for item in row["units"])
                optimizer.zero_grad(set_to_none=True)
                loss = field_loss(encoder, query, units, row)
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.heads.state_dict(), output / "heads.pt")
    (output / "training.json").write_text(json.dumps({"model": args.model, "states": len(training),
        "steps": steps, "loss": total/steps, "method": "shared_fields_generator_set_utility",
        "quality_target": "bounded_answer_gain_listwise", "phase": "fields" if args.field_only else "sets"}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("query", "evidence"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-model")
    parser.add_argument("--query-asset")
    parser.add_argument("--init-asset")
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--history-tokens", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    {"query": train_query, "evidence": train_evidence}[args.stage](args)


if __name__ == "__main__":
    main()
