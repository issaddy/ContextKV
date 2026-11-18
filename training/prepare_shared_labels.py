"""Prepare shared task/field supervision from native MultiDoc2Dial training sources."""
from __future__ import annotations

import argparse
import asyncio
import json
from itertools import cycle
from pathlib import Path
import random

from adapters.doc2dial import benchmark_rows, convert, structural_corpus
from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from adapters.retrieval import BM25SRetriever
from adapters.shared_encoding import query_input
from controller.schema import Message, from_history_messages
from controller.serialization import to_jsonable
from controller.source_windows import numbered_text
from training.shared_collection import numbered_sources, word_reference_schema, word_offsets, source_bindings, object_schema


TASK_INSTRUCTION = """Label a document-QA request for a shared query encoder. Use only the visible conversation and question.
Return JSON with rewritten_query, coverage_scope, needs, conditions, history_refs, version_intent.
Every text reference is {message_id,word_boundaries:[a,b]}, selecting the displayed numbered words.
The two zero-based boundary indices select an inclusive span, in either order.
Use the ID of the message actually containing the intended words. Do not return words or turn IDs as references.
needs: [{need_id,anchor_refs,subject_refs,operation,scope,condition_ids,predecessor_need_ids,query,join,distinct}].
operation: lookup/enumerate/procedure/compare/count/sum/mean/min/max/group/infer; scope: selected_sources/all_supplied_documents.
operation names the requested result: grouped totals use sum, grouped counts use count, and group means listing rows by category.
coverage_scope independently states whether the answer needs support from relevant sources (selected_sources)
or requires examining a complete declared population (all_supplied_documents). Complete enumeration, universal
claims and global relationships can require full coverage without numeric arithmetic. A question using supplied
evidence does not itself require full coverage. Each Need's scope applies the same distinction to that result.
Split independent requested results into separate Needs. query is a standalone search query for that Need.
Give each Need one concise contiguous anchor reference; keep subjects and conditions in their own references.
Every Need anchor must belong to current_message_id; history supplies subject, condition and history_refs.
join means matching records across separate tables; distinct means counting or aggregating unique entities.
conditions: [{condition_id,anchor_ref,value_refs,relation_type,update,filter_operator}].
relation_type: PREDICATE/TIME/VERSION/PLATFORM/LOCATION/SUBJECT; update: ADD/REPLACE/REMOVE/COMPARE.
filter_operator: eq/ne/lt/le/gt/ge/in/contains. Keep entity, period, unit and version restrictions explicit.
history_refs selects only historical text needed to interpret the request.
version_intent: {mode,document_families,version_ids,time_refs,temporal_relation};
mode: current/explicit/as_of/compare, temporal_relation: at/before/after.
Use catalogue identifiers for document_families and version_ids. For current, leave version_ids empty.
For before/after, version_ids names the anchor version whose adjacent version is requested.
For as_of with temporal_relation at, select the version valid on the requested date from the catalogue.
Use empty lists for absent restrictions. Do not add facts or conditions from outside this input."""

FIELD_INSTRUCTION = """Label source evidence for the stated document-QA Needs. Return JSON {bindings:[...]}.
Each binding is {need_id,role,source,condition_id,relation_group}.
source is {source_id,word_boundaries:[a,b]}, selecting the displayed numbered words.
The two zero-based boundary indices select an inclusive span, in either order.
role: source/measure/entity/period/group/filter/unit/version.
source marks a passage span that supports a Need. The other roles mark original column names, field names,
entity identifiers, periods, units and constraints required to compute the requested result.
For lookup and comparison Needs, also label the entity, period and filter bindings needed to qualify their supporting evidence.
Use condition_id to link filter/period columns to a declared query condition.
relation_group identifies matching join keys across tables; leave it empty for non-join fields.
Reuse the relation_group in known_join_keys when the same semantic join field occurs in another source batch.
Label supporting evidence, necessary complementary evidence and explicit computation fields.
The reference answer is an offline labeling aid. Bind only words in the supplied numbered sources."""


def numbered_messages(row):
    shown, locations = [], {}
    for index, item in enumerate([*row["history"], {"turn_id": row["turn"], "speaker": "user", "text": row["question"]}]):
        identity = f"M{index+1}"
        words, offsets = numbered_text(item["text"])
        shown.append({"message_id": identity, "speaker": item["speaker"], "words": words})
        locations[identity] = item, offsets
    return shown, locations


def task_annotations(raw, locations):
    def ref(item):
        message, offsets = locations[item["message_id"]]
        start, end = word_offsets(item, offsets)
        return {"turn_id": message["turn_id"], "speaker": message["speaker"], "start": start, "end": end}

    conditions = [{"condition_id": item["condition_id"], "key_span": ref(item["anchor_ref"]),
        "predicate_span": ref(item["anchor_ref"]), "value_spans": [ref(value) for value in item["value_refs"]],
        "relation_type": item["relation_type"], "update": item["update"]} for item in raw["conditions"]]
    needs = [{"need_id": item["need_id"], "anchor_spans": [ref(value) for value in item["anchor_refs"]],
        "subject_refs": [ref(value) for value in item["subject_refs"]], "operation": item["operation"],
        "scope": item["scope"], "condition_ids": item["condition_ids"],
        "predecessor_need_ids": item["predecessor_need_ids"]} for item in raw["needs"]]
    version = raw["version_intent"]
    return {"task_state": {"conditions": conditions, "needs": needs, "coverage_scope": raw["coverage_scope"],
        "working_history": [ref(item) for item in raw["history_refs"]],
        "version_intent": {**version, "time_refs": [ref(item) for item in version["time_refs"]]}},
        "rewritten_query": raw["rewritten_query"], "need_queries": [item["query"] for item in raw["needs"]],
        "filter_operators": {item["condition_id"]: item["filter_operator"] for item in raw["conditions"]},
        "join_needs": [item["need_id"] for item in raw["needs"] if item["join"]],
        "distinct_needs": [item["need_id"] for item in raw["needs"] if item["distinct"]]}


def task_label_schema(locations, catalogue):
    def array(item):
        return {"type": "array", "items": item}
    string, boolean = {"type": "string"}, {"type": "boolean"}
    ref = word_reference_schema(locations, "message_id")
    current_id = next(reversed(locations))
    current_ref = word_reference_schema({current_id: locations[current_id]}, "message_id")
    def ids(values):
        return array({"enum": list(dict.fromkeys(values))}) if values else {"type": "array", "maxItems": 0, "items": string}
    return object_schema({"rewritten_query": string, "coverage_scope": {"enum": ["selected_sources", "all_supplied_documents"]}, "needs": array(object_schema({
        "need_id": string, "anchor_refs": {**array(current_ref), "minItems": 1, "maxItems": 1},
        "subject_refs": array(ref), "operation": {"enum": "lookup enumerate procedure compare count sum mean min max group infer".split()},
        "scope": {"enum": ["selected_sources", "all_supplied_documents"]}, "condition_ids": array(string),
        "predecessor_need_ids": array(string), "query": string, "join": boolean, "distinct": boolean})),
        "conditions": array(object_schema({"condition_id": string, "anchor_ref": ref, "value_refs": array(ref),
            "relation_type": {"enum": "PREDICATE TIME VERSION PLATFORM LOCATION SUBJECT".split()},
            "update": {"enum": ["ADD", "REPLACE", "REMOVE", "COMPARE"]},
            "filter_operator": {"enum": "eq ne lt le gt ge in contains".split()}})),
        "history_refs": array(ref), "version_intent": object_schema({
            "mode": {"enum": ["current", "explicit", "as_of", "compare"]},
            "document_families": ids([row["family"] for row in catalogue]),
            "version_ids": ids([row["version"] for row in catalogue]), "time_refs": array(ref),
            "temporal_relation": {"enum": ["at", "before", "after"]}})})


def field_label_schema(row, locations):
    variants = []
    for need in row["task_state"]["needs"]:
        properties = {"need_id": {"const": need["need_id"]},
            "role": {"enum": "source measure entity period group filter unit version".split()},
            "source": word_reference_schema(locations, "source_id"),
            "condition_id": {"enum": ["", *need["condition_ids"]]}, "relation_group": {"type": "string"}}
        variants.append(object_schema(properties))
    bindings = {"type": "array", "items": {"oneOf": variants}} if variants else {"type": "array", "maxItems": 0}
    return object_schema({"bindings": bindings})


def prepare_states(args, tokenizer, excluded=()):
    source = Path(args.source)
    documents = json.loads((source/"multidoc2dial_doc.json").read_text())
    records, supervision, _ = convert(documents,
        json.loads((source/"multidoc2dial_dial_train.json").read_text()), dataset_version="archive-f0c034c24966",
        source_uri="https://doc2dial.github.io/multidoc2dial/", split="train", source_format="multidoc2dial")
    if not (args.task_only or args.values_only):
        corpus = CorpusIndex.from_records(structural_corpus(documents, records),
            corpus_id="multidoc2dial", snapshot_id="multidoc2dial-train")
        directory = SourceDirectory(corpus, tokenizer)
        by_id = directory.units_by_id
        sparse = BM25SRetriever({identity: {"text": unit.text, "title": unit.heading} for identity, unit in by_id.items()},
                               corpus_snapshot_id="multidoc2dial-train")
    tasks = list(benchmark_rows(supervision))
    sessions = sorted({row["conversation_id"] for row in tasks})
    random.Random(args.seed).shuffle(sessions)
    dev_sessions = set(sessions[:max(1, len(sessions)//5)])
    chosen, counts = [], {"train": 0, "dev": 0}
    limits = {"train": args.train_states, "dev": args.dev_states}
    grouped = {sid: [] for sid in sessions if sid not in excluded}
    for row in sorted(tasks, key=lambda r: r["turn"]):
        if row["conversation_id"] in grouped:
            grouped[row["conversation_id"]].append(row)
    # Round-robin across conversations, rotating the first turn sampled. A small
    # label budget then includes follow-ups with their actual preceding history.
    tasks = [turns[(i+offset) % len(turns)] for offset in range(max(map(len, grouped.values()), default=0))
             for i, turns in enumerate(grouped.values()) if offset < len(turns)]
    for row in tasks:
        split = "dev" if row["conversation_id"] in dev_sessions else "train"
        if counts[split] >= limits[split]:
            continue
        turns = from_history_messages(tuple(Message(
            "assistant" if item["speaker"] == "agent" else item["speaker"], item["text"])
            for item in row["input"][:-1]))
        history = [{"turn_id": item.turn_id, "speaker": speaker, "text": text}
            for item in turns for speaker, text in (("user", item.user_text), ("assistant", item.assistant_text))
            if text is not None]
        turn = len(turns)+1
        question = row["input"][-1]["text"]
        _, refs, _, _ = query_input(tokenizer, [(item["turn_id"], item["speaker"], item["text"]) for item in history],
                                    question, turn, history_tokens=args.history_tokens)
        visible = {(ref.turn_id, ref.speaker) for ref in refs if ref is not None}
        history = [item for item in history if (item["turn_id"], item["speaker"]) in visible]
        state = {"task_id": row["task_id"], "conversation_id": row["conversation_id"], "turn": turn,
            "split": split, "question": question, "history": history, "catalogue": [], "dataset": "multidoc2dial"}
        chosen.append(state)
        counts[split] += 1
        if args.task_only:
            continue
        if args.values_only:
            state["reference_answer"] = row["targets"][0]["text"]
            continue
        positives = [unit.unit_id for unit in by_id.values() if any(
            span.source_key.document_id == label["passage_id"] and span.start < label["end"] and label["start"] < span.end
            for span in unit.source_spans for label in row["grounding"])]
        recalled = sparse.retrieve([question], top_k=args.candidates)
        candidates = tuple(dict.fromkeys((*positives, *(item.document_id for item in recalled))))[:args.candidates]
        pending, selected = list(candidates), {}
        while pending:
            identity = pending.pop(0)
            if identity not in selected:
                selected[identity] = by_id[identity]
                pending.extend(by_id[identity].dependencies)
        state.update(reference_answer=row["targets"][0]["text"], grounding=row["grounding"],
                     units=to_jsonable(tuple(selected.values())))
    return chosen


async def prepare_labels(args):
    import httpx
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.query_model)
    label_tokenizer = (None if args.task_only or args.values_only
                       else AutoTokenizer.from_pretrained(args.teacher_model))
    excluded = {row["conversation_id"] for path in args.exclude_labelled_sessions
                for row in map(json.loads, Path(path).read_text().splitlines())}
    states = prepare_states(args, tokenizer, excluded)
    from training.controlled_tasks import controlled_states
    extras = controlled_states(tokenizer, train_families=args.controlled_train_families,
        dev_families=args.controlled_dev_families, seed=args.seed)
    for path in args.states:
        extras.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    states.extend(r for r in extras if r["conversation_id"] not in excluded)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["task_id"] for row in (json.loads(line) for line in output.read_text().splitlines() if line.strip())} if output.exists() else set()
    semaphore = asyncio.Semaphore(args.concurrency)
    endpoints = cycle(args.endpoint)
    async with httpx.AsyncClient(timeout=None) as client:
        async def complete(instruction, value, schema):
            response = await client.post(next(endpoints).rstrip("/")+"/v1/chat/completions", json={
                "model": args.teacher_model, "messages": [{"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(value, ensure_ascii=False)}],
                "temperature": 0, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": schema}}})
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])

        async def annotate(row):
            async with semaphore:
                await label_state(row)

        async def label_state(row):
            messages, message_locations = numbered_messages(row)
            task = await complete(TASK_INSTRUCTION, {"messages": messages,
                "current_message_id": messages[-1]["message_id"], "catalogue": row["catalogue"]},
                task_label_schema(message_locations, row["catalogue"]))
            row = {**row, **task_annotations(task, message_locations)}
            if args.task_only:
                labeled = {key: value for key, value in row.items()
                           if key not in {"units", "grounding", "reference_answer"}}
            elif args.values_only:
                labeled = row
            else:
                bindings, join_keys = [], {}
                sources, locations = numbered_sources(row["units"])
                start = 0
                while start < len(sources):
                    value = {"messages": messages, "task": task, "sources": [],
                        "reference_answer": row["reference_answer"],
                        "known_join_keys": [{"need_id": need, "field": text, "relation_group": group}
                                            for (need, text), group in join_keys.items()]}
                    end = start
                    while end < len(sources):
                        trial = {**value, "sources": sources[start:end+1]}
                        tokens = label_tokenizer.apply_chat_template([
                            {"role": "system", "content": FIELD_INSTRUCTION},
                            {"role": "user", "content": json.dumps(trial, ensure_ascii=False)}],
                            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                        if end > start and len(tokens)+4096 > args.label_context_tokens:
                            break
                        value = trial
                        end += 1
                    source_locations = {item["source_id"]: locations[item["source_id"]]
                                        for item in value["sources"]}
                    field = await complete(FIELD_INSTRUCTION, value,
                        field_label_schema(row, source_locations))
                    bindings.extend(source_bindings(source_locations, field["bindings"]))
                    for item in field["bindings"]:
                        if item["relation_group"]:
                            unit, offsets = source_locations[item["source"]["source_id"]]
                            left, right = word_offsets(item["source"], offsets)
                            join_keys[item["need_id"], unit["text"][left:right]] = item["relation_group"]
                    start = end
                labeled = {**row, "bindings": bindings}
            sink.write(json.dumps(labeled, ensure_ascii=False)+"\n")
            sink.flush()
            completed.add(row["task_id"])

        with output.open("a") as sink:
            async with asyncio.TaskGroup() as tasks:
                for row in states:
                    if row["task_id"] not in completed:
                        tasks.create_task(annotate(row))
    return {"train": sum(row["split"] == "train" and row["task_id"] in completed for row in states),
            "dev": sum(row["split"] == "dev" and row["task_id"] in completed for row in states), "output": str(output)}


async def refresh_coverage(args):
    """Relabel scope from the question and conversation, independent of operation labels."""
    import httpx
    from training.controlled_tasks import coverage_states
    datasets = {path: [json.loads(line) for line in Path(path).read_text().splitlines() if line]
                for path in args.refresh_coverage}
    scopes = {"enum": ["selected_sources", "all_supplied_documents"]}
    instruction = ("Classify the evidence scope of the entire question and each named subquestion "
        "individually using the dialogue. selected_sources covers specific lookups, explanations, "
        "or steps in one procedure. all_supplied_documents requires inspecting the declared population, "
        "e.g. exhaustive enumeration, absence, an aggregate over records, or a global relationship. "
        "Using supplied documents alone does not imply full coverage. A local lookup can coexist "
        "with a population-wide subquestion. Classify the subquestion described by its anchor in context.")
    groups = {}
    for dataset in datasets.values():
        for row in dataset:
            if row.get("coverage_only"):
                continue
            texts = {(m["turn_id"], m["speaker"]): m["text"] for m in row["history"]}
            texts[row["turn"], "user"] = row["question"]
            def excerpt(refs):
                return " ".join(texts[r["turn_id"], r["speaker"]][r["start"]:r["end"]] for r in refs)
            needs = {n["need_id"]: {"anchor": excerpt(n["anchor_spans"]), "subject": excerpt(n["subject_refs"]),
                                    "operation": n["operation"]} for n in row["task_state"]["needs"]}
            payload = json.dumps({"question": row["question"], "history": row["history"], "needs": needs},
                                 ensure_ascii=False, sort_keys=True)
            groups.setdefault(payload, []).append(row)
    endpoints = cycle(args.endpoint)
    slots = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=180) as client:
        async def classify(payload, rows):
            ids = [n["need_id"] for n in rows[0]["task_state"]["needs"]]
            schema = object_schema({"coverage_scope": scopes, "needs": object_schema(dict.fromkeys(ids, scopes))})
            async with slots:
                response = await client.post(next(endpoints).rstrip("/")+"/v1/chat/completions", json={
                    "model": args.teacher_model, "temperature": 0, "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": payload}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": "coverage", "schema": schema}}})
                response.raise_for_status()
                labels = json.loads(response.json()["choices"][0]["message"]["content"])
            for row in rows:
                row["task_state"]["coverage_scope"] = labels["coverage_scope"]
                for need in row["task_state"]["needs"]:
                    need["scope"] = labels["needs"][need["need_id"]]
        await asyncio.gather(*(classify(payload, rows) for payload, rows in groups.items()))
    for path, dataset in datasets.items():
        Path(path).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in dataset))
    # Regenerate the controlled contrasts and retain externally grounded scope annotations.
    queries = [row for row in datasets[args.refresh_coverage[0]]
               if not row.get("coverage_only") or row.get("source")]
    queries.extend(coverage_states())
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in queries))
    return {"relabelled": len(groups), "query_states": len(queries), "output": args.output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/multidoc2dial/source")
    parser.add_argument("--states", action="append", default=[], help="Additional source-grounded training states in the same row format")
    parser.add_argument("--query-model", required=True)
    parser.add_argument("--endpoint", nargs="+", default=["http://127.0.0.1:30000"])
    parser.add_argument("--teacher-model", default="/data/models/qwen3-32b")
    parser.add_argument("--label-context-tokens", type=int, default=32768)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-labelled-sessions", nargs="*", default=[],
                        help="Exclude prior labelled conversations when collecting additional training states")
    parser.add_argument("--task-only", action="store_true", help="Label diverse conversation pairs without evidence retrieval or field annotation")
    parser.add_argument("--values-only", action="store_true", help="Prepare task/source states for set and stop collection with the existing field head")
    parser.add_argument("--refresh-coverage", nargs="+", help="Existing task/field/set labels to correct with the fixed teacher")
    parser.add_argument("--train-states", type=int, default=24)
    parser.add_argument("--dev-states", type=int, default=8)
    parser.add_argument("--controlled-train-families", type=int, default=2)
    parser.add_argument("--controlled-dev-families", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--history-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(refresh_coverage(args) if args.refresh_coverage else prepare_labels(args))))


if __name__ == "__main__":
    main()
