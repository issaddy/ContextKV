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
        decode(VersionIntent, raw["version_intent"]), current_turn_id=row["turn"], query=row["question"],
        coverage_scope=raw["coverage_scope"])


def task_loss(encoder, row, prepared=None):
    import torch
    from torch.nn import functional as F
    from adapters.shared_heads import (CONDITION_TYPES, EDGE_KINDS, FILTER_OPERATORS, OPERATIONS,
        SCOPES, SPAN_KINDS, TEMPORAL_RELATIONS, UPDATES, VERSION_MODES)
    if prepared is None:
        inputs, refs, _, catalog_positions = query_input(encoder.tokenizer, conversation(row), row["question"],
            row["turn"], catalogue(row), history_tokens=encoder.history_tokens, history_turns=row.get("history_turns", ()))
        query_position = max(i for i, ref in enumerate(refs)
                             if ref is not None and ref.turn_id == row["turn"] and ref.speaker == "user")
        states, logits = encoder.forward_input(inputs, query_position)
    else:
        refs, catalog_positions, states, query_position, alignment_targets = prepared
        logits = encoder.heads(states, states[query_position])
    device = states.device
    current = [i for i, ref in enumerate(refs)
               if ref is not None and ref.turn_id == row["turn"] and ref.speaker == "user"]
    coverage = encoder.heads.coverage(F.layer_norm(states[current].mean(0), (states.shape[-1],)))
    if row.get("coverage_only"):
        return F.cross_entropy(coverage[None], torch.tensor(
            [SCOPES.index(row["task_state"]["coverage_scope"])], device=device))
    task = task_label(row)
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
    losses = [F.cross_entropy(logits["span"].reshape(-1, 3), labels.reshape(-1)),
        F.cross_entropy(coverage[None], torch.tensor(
            [SCOPES.index(task.coverage_scope)], device=device))]
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
            value = getattr(encoder.heads, name)(states[query_position], vector).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(value, states.new_tensor(row_metadata[field] in expected)))

    if prepared is not None:
        if len(need_vectors):
            aligned = F.normalize(torch.cat((states[query_position:query_position+1],
                encoder.heads.need_projection(need_vectors))), dim=-1)
            losses.append(torch.linalg.vector_norm(aligned-alignment_targets, dim=-1).mean())
            losses.append((F.pdist(aligned)-F.pdist(alignment_targets)).abs().mean())
        return torch.stack(losses).mean()
    # Conversational/rewrite and ordinary-query identity pairs share the frozen base space.
    # Documents are always encoded with adapters disabled, so their space stays fixed directly.
    instruction = "Instruct: Retrieve evidence for the question.\nQuery: "
    ordinary = encoder.tokenizer([instruction+row["question"]], return_tensors="pt").to(encoder.model.device)
    ordinary_states = encoder.model(**ordinary, use_cache=False).last_hidden_state[:, -1].float()
    predicted = encoder.heads.need_projection(need_vectors)
    adapted = F.normalize(torch.cat((states[query_position:query_position+1], predicted, ordinary_states)), dim=-1)
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
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.manual_seed(args.seed)
    encoder = SharedQueryEncoder(args.model, None, device=args.device, train=True, history_tokens=args.history_tokens)
    if args.init_asset:
        encoder.model = PeftModel.from_pretrained(encoder.model, Path(args.init_asset)/"adapter", is_trainable=not args.heads_only)
        encoder.heads.load_state_dict({**encoder.heads.state_dict(), **torch.load(
            Path(args.init_asset)/"heads.pt", map_location=args.device, weights_only=True)})
    else:
        encoder.model = get_peft_model(encoder.model, LoraConfig(r=args.rank, lora_alpha=args.rank*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="FEATURE_EXTRACTION"))
    if args.heads_only:
        encoder.model.requires_grad_(False).eval()
    parameters = [parameter for module in (encoder.model, encoder.heads) for parameter in module.parameters()
                  if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    training = [row for row in rows(args.data) if row["split"] == "train"]
    prepared = {}
    if args.heads_only:
        for row in training:
            inputs, refs, _, positions = query_input(encoder.tokenizer, conversation(row), row["question"],
                row["turn"], catalogue(row), history_tokens=encoder.history_tokens, history_turns=row.get("history_turns", ()))
            with torch.no_grad():
                query_position = max(i for i, ref in enumerate(refs)
                    if ref is not None and ref.turn_id == row["turn"] and ref.speaker == "user")
                states, _ = encoder.forward_input(inputs, query_position)
                targets = states[:0]
                if not row.get("coverage_only"):
                    texts = [row["rewritten_query"], *row["need_queries"]]
                    batch = encoder.tokenizer(["Instruct: Retrieve evidence for the question.\nQuery: "+text
                        for text in texts], padding=True, return_tensors="pt").to(encoder.model.device)
                    with encoder.model.disable_adapter():
                        base = encoder.model(**batch, use_cache=False).last_hidden_state[:, -1].float()
                    targets = torch.nn.functional.normalize(base, dim=-1)
            prepared[row["task_id"]] = refs, positions, states, query_position, targets
    random.seed(args.seed)
    steps, total = 0, 0.
    for _ in range(args.epochs):
        random.shuffle(training)
        for row in training:
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(encoder, row, prepared.get(row["task_id"]))
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    encoder.model.save_pretrained(output / "adapter")
    torch.save(encoder.heads.state_dict(), output / "heads.pt")
    (output / "training.json").write_text(json.dumps({"model": args.model, "states": len(training),
        "steps": steps, "loss": total/steps,
        "method": "shared_task_heads" if args.heads_only else "rcem_shared_task_structure_alignment",
        "history_tokens": args.history_tokens}, indent=2)+"\n")


def align_annotations(predicted, annotated):
    """Match predicted and labelled objects by their original message coordinates."""
    edges = []
    for gold, target in enumerate(annotated):
        for guess, spans in enumerate(predicted):
            overlap = sum(max(0, min(a.end, b.end)-max(a.start, b.start))
                          for a in target for b in spans if (a.turn_id, a.speaker) == (b.turn_id, b.speaker))
            if overlap:
                size = sum(ref.end-ref.start for ref in (*target, *spans))
                edges.append((overlap/(size-overlap), gold, guess))
    matched, used = {}, set()
    for _, gold, guess in sorted(edges, reverse=True):
        if gold not in matched and guess not in used:
            matched[gold] = guess
            used.add(guess)
    return matched


def inferred_field_labels(query, row):
    task = task_label(row)
    needs = align_annotations([need.anchor_spans for need in query.task.needs],
                              [need.anchor_spans for need in task.needs])
    conditions = align_annotations([(condition.predicate_span or condition.key_span,)
                                    for condition in query.task.conditions],
                                   [(condition.predicate_span or condition.key_span,) for condition in task.conditions])
    need_ids = {task.needs[gold].need_id: query.task.needs[guess].need_id for gold, guess in needs.items()}
    condition_ids = {task.conditions[gold].condition_id: query.task.conditions[guess].condition_id
                     for gold, guess in conditions.items()}
    return {**row, "annotated_need_ids": tuple(need_ids.values()), "bindings": [{**binding, "need_id": need_ids[binding["need_id"]],
        "condition_id": condition_ids.get(binding.get("condition_id"), "")}
        for binding in row["bindings"] if binding["need_id"] in need_ids]}


def field_loss(encoder, query, units, row):
    import torch
    from torch.nn import functional as F
    from adapters.shared_heads import FIELD_ROLES
    states, embeddings, relevance, addresses, _ = encoder.forward_evidence(query, units)
    source_indices = [i for i, address in enumerate(addresses) if address is not None]
    states = states[source_indices]
    addresses = [addresses[i] for i in source_indices]
    positions = {n.need_id: i for i, n in enumerate(query.task.needs)}
    annotated = row["annotated_need_ids"]
    logits = encoder.heads.field_logits(states, query.need_vectors[[positions[n] for n in annotated]].to(states.device))
    targets = torch.zeros(logits.shape[:-1], device=states.device, dtype=torch.long)
    needs = {n: index for index, n in enumerate(annotated)}
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
    data = [row for row in rows(args.data)
            if (bool(row["task_state"]["needs"] and row["units"]) if args.field_only else bool(row.get("sets")))]
    training = [row for row in data if row["split"] == "train"]
    needs_encoding = args.field_only or any("features" not in row["sets"][0] for row in data)
    if needs_encoding:
        query_encoder = SharedQueryEncoder(args.query_model, args.query_asset, device=args.device, history_tokens=args.history_tokens)
        encoder = SharedEvidenceEncoder(args.model, args.init_asset, device=args.device, train=True)
        encoder.model.requires_grad_(False).eval()
        heads = encoder.heads
    else:
        from adapters.shared_heads import EvidenceHeads
        weights = torch.load(Path(args.init_asset)/"heads.pt", map_location="cpu", weights_only=True)
        heads = EvidenceHeads(weights["field.weight"].shape[1], weights["query.1.weight"].shape[1]).to(args.device)
        heads.load_state_dict(weights)
    steps, total = 0, 0.
    selection = {}
    if not args.field_only:
        encoded = {"train": [], "dev": []}
        tensor = heads.set_bias.new_tensor
        for row in data:
            choices = row["sets"]
            if "features" in choices[0]:
                features = tensor([item["features"] for item in choices])
                completed = tensor(row["known"])
            else:
                query = query_encoder.encode(conversation(row), row["question"], row["task_id"], row["turn"],
                    catalogue=catalogue(row), history_turns=row.get("history_turns", ()))
                units = tuple(decode(EvidenceUnit, item) for item in row["units"])
                with torch.no_grad():
                    evidence = encoder.encode(query, units)
                    support, completed = evidence.need_support(units,
                        tuple(need.need_id for need in query.task.needs), row["completed_needs"])
                    masks = tensor([[key in item["unit_ids"] for key in evidence.unit_ids] for item in choices])
                    features = heads.features(evidence.embeddings, evidence.relevance, masks, support, completed)
            target = tensor([item["quality"] for item in choices])
            empty = tensor([not item["unit_ids"] for item in choices]).bool()
            encoded[row["split"]].append((features, completed, target, empty))
        optimizer = torch.optim.AdamW([heads.set_weights, heads.set_bias], lr=args.learning_rate)
        def gain_loss(example):
            features, completed, target, empty = example
            return heads.gain_loss(heads.utility_from_features(features, completed), target, empty)
        for _ in range(args.epochs):
            for example in encoded["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = gain_loss(example)
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
        with torch.no_grad():
            observed = []
            for (features, completed, _, _), row in zip(encoded["dev"], (r for r in data if r["split"]=="dev")):
                chosen = row["sets"][int(heads.utility_from_features(features, completed).argmax())]
                if "covered_facts" in chosen:
                    observed.append((chosen["quality"], sum(chosen["covered_facts"])/len(chosen["covered_facts"])))
            if observed:
                selection["dev_chosen_quality"], selection["dev_chosen_fact_recall"] = (
                    sum(values)/len(values) for values in zip(*observed))
            if encoded["dev"]:
                selection["dev_loss"] = sum(float(gain_loss(x)) for x in encoded["dev"])/len(encoded["dev"])
        selection.update(dev_states=len(encoded["dev"]), epoch=args.epochs,
                         fact_labelled_states=sum("covered_facts" in r["sets"][0] for r in data))
    else:
        optimizer = torch.optim.AdamW(heads.parameters(), lr=args.learning_rate)
        for _ in range(args.epochs):
            for row in training:
                query = query_encoder.encode(conversation(row), row["question"], row["task_id"], row["turn"],
                    catalogue=catalogue(row), history_turns=row.get("history_turns", ()))
                if not query.task.needs:
                    continue
                labelled = inferred_field_labels(query, row)
                if not labelled["annotated_need_ids"]:
                    continue
                units = tuple(decode(EvidenceUnit, item) for item in row["units"])
                optimizer.zero_grad(set_to_none=True)
                loss = field_loss(encoder, query, units, labelled)
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), output / "heads.pt")
    (output / "training.json").write_text(json.dumps({"model": args.model, "states": len(training),
        "steps": steps, "loss": total/steps, "method": "shared_fields_generator_set_utility",
        "quality_target": "bounded_answer_gain_listwise", "phase": "fields" if args.field_only else "sets",
        **selection}, indent=2)+"\n")


def train_coverage(args):
    """Fit population coverage with equal example weights, independently of task-head count."""
    import torch
    from torch.nn import functional as F
    from adapters.shared_heads import SCOPES
    torch.manual_seed(args.seed)
    encoder = SharedQueryEncoder(args.model, args.init_asset, device=args.device, history_tokens=args.history_tokens)
    data = rows(args.data)
    vectors, need_vectors, need_targets, need_rows, predicted = [], [], [], [], []
    for start in range(0, len(data), 16):
        batch = data[start:start+16]
        encoded = encoder.encode_many([dict(history=conversation(row), question=row["question"],
            task_id=row["task_id"], current_turn_id=row["turn"], catalogue=catalogue(row),
            history_turns=row.get("history_turns", ())) for row in batch])
        for i, (row, item) in enumerate(zip(batch, encoded), start):
            vectors.append(item.coverage_vector.detach())
            predicted.append(item.need_vectors.detach())
            gold = row["task_state"].get("needs", ())
            matched = align_annotations([n.anchor_spans for n in item.task.needs],
                [[decode(TextRef, span) for span in n["anchor_spans"]] for n in gold])
            for target, guess in matched.items():
                need_vectors.append(item.need_vectors[guess].detach())
                need_targets.append(SCOPES.index(gold[target]["scope"]))
                need_rows.append(i)
    features = F.layer_norm(torch.stack(vectors), (vectors[0].shape[-1],))
    labels = torch.tensor([SCOPES.index(row["task_state"]["coverage_scope"]) for row in data], device=args.device)
    train = torch.tensor([row["split"] == "train" for row in data], device=args.device)
    head, scope = encoder.heads.coverage, encoder.heads.scope
    owners = labels.new_tensor([i for i, needs in enumerate(predicted) for _ in needs])
    all_needs = torch.cat(predicted)
    def route_margin(question_logits):
        margins = question_logits[:, 1]-question_logits[:, 0]
        logits = scope(all_needs)
        return margins.scatter_reduce(0, owners, logits[:, 1]-logits[:, 0], reduce="amax")
    has_need_training = False
    if need_vectors:
        need_x = torch.stack(need_vectors)
        need_y = labels.new_tensor(need_targets)
        need_train = train[need_rows]
        train_need_x, train_need_y = need_x[need_train], need_y[need_train]
        has_need_training = len(train_need_y) > 0
    @torch.no_grad()
    def evaluate():
        logits = head(features)
        question = logits.argmax(-1)
        full = route_margin(logits) > 0
        result = {"task_correct": int((question[~train] == labels[~train]).sum()),
                  "routed_correct": int((full[~train] == labels[~train]).sum()),
                  "dev": int((~train).sum())}
        domains = [r.get("dataset", "structured_training") for r in data]
        result["by_dataset"] = {}
        for name in sorted(set(domains)):
            subset = ~train & train.new_tensor([domain == name for domain in domains])
            result["by_dataset"][name] = {"routed_correct": int((full[subset] == labels[subset]).sum()),
                                          "dev": int(subset.sum())}
        if need_vectors:
            result.update(need_correct=int((scope(need_x[~need_train]).argmax(-1)==need_y[~need_train]).sum()),
                          matched_needs=int((~need_train).sum()),
                          annotated_needs=sum(len(r["task_state"].get("needs", ())) for r in data if r["split"]=="dev"))
        return result
    before = evaluate()
    optimizer = torch.optim.AdamW([*head.parameters(), *scope.parameters()], lr=args.learning_rate)
    for _ in range(args.epochs):
        logits = head(features)
        loss = F.cross_entropy(logits[train], labels[train])
        loss += F.binary_cross_entropy_with_logits(route_margin(logits)[train], labels[train].float())
        if has_need_training:
            loss = loss + F.cross_entropy(scope(train_need_x), train_need_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    initial = Path(args.init_asset)
    if not (output/"adapter").exists():
        (output/"adapter").symlink_to((initial/"adapter").resolve(), target_is_directory=True)
    torch.save(encoder.heads.state_dict(), output/"heads.pt")
    metadata = json.loads((initial/"training.json").read_text())
    metadata["coverage"] = {"train": int(train.sum()), "before": before, "after": evaluate(),
                            "epoch": args.epochs}
    (output/"training.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata["coverage"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("query", "coverage", "evidence"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-model")
    parser.add_argument("--query-asset")
    parser.add_argument("--init-asset")
    parser.add_argument("--field-only", action="store_true")
    parser.add_argument("--heads-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--history-tokens", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    {"query": train_query, "coverage": train_coverage, "evidence": train_evidence}[args.stage](args)


if __name__ == "__main__":
    main()
