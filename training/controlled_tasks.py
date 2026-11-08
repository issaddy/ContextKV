"""Version/platform training dialogues with complete source-scope oracles."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path

import yaml

from controller.evidence import make_evidence
from controller.serialization import to_jsonable
from training.task_data import read_jsonl, write_jsonl


def build_controlled_tasks(config):
    settings = config.get("training", {})
    source_path = Path(settings.get("data_dir", "runs/study/data"))/"sources.jsonl"
    originals = sorted(read_jsonl(source_path), key=lambda row: row["source_id"])
    policy_originals = {}
    for original in originals:
        policy_originals.setdefault(original["document_family"], original)
    policy_originals = tuple(policy_originals.values())
    briefings = sorted(originals, key=lambda row: (-len(row["text"]), row["source_id"]))
    destination = Path(settings.get("controlled_dir", "runs/study/constructed_train"))
    families = settings.get("controlled_families", 24)
    documents, tasks = [], []
    for index in range(families):
        family = f"constructed-train-policy-{index:03}"
        program, other = f"Training Program {index+1}", f"Briefing Program {index+1}"
        original = policy_originals[(index+100) % len(policy_originals)]
        excerpt = original["text"][:1500]
        scope, facts = [], {}
        for version, start, end, delta in (("2018", "2018-01-01", "2024-01-01", 0),
                                          ("2024", "2024-01-01", None, 30)):
            for platform, base in (("Windows", 100), ("Linux", 80)):
                value = base+delta+index
                fact = f"{program} policy {version}: the {platform} filing allowance is {value} credits."
                validity = f"Effective from {start}"+(f" until {end}." if end else "; this is the current supplied policy.")
                prefix = excerpt+"\n\n"+validity+"\n"
                text = prefix+fact
                source_id = f"{family}-{version}-{platform.lower()}"
                source = {"source_id": source_id, "passage_id": source_id, "document_id": source_id,
                    "document_family": family, "version": version, "version_order": [int(version)],
                    "current": version == "2024", "valid_from": start, "valid_to": end,
                    "published_at": start, "applicability": {"PLATFORM": platform}, "split": "train",
                    "track": "constructed_train", "title": f"{program} {version} {platform}", "text": text,
                    "offsets": [0, len(text)], "source_uri": f"study-training://{source_id}",
                    "derived_from": {"source_id": original["source_id"], "start": 0, "end": len(excerpt)}}
                scope.append(source)
                facts[version, platform] = {"passage_id": source_id, "start": len(prefix), "end": len(text),
                    "quote": fact, "value": value, "unit": "credits", "version": version, "platform": platform}
        briefing = briefings[index % len(briefings)]
        text = f"{other}: the following original training-source text is this program's briefing.\n"+briefing["text"][:12000]
        brief_id = family+"-briefing"
        scope.append({"source_id": brief_id, "passage_id": brief_id, "document_id": brief_id,
            "document_family": family+"-other-topic", "version": "2024", "version_order": [2024], "current": True,
            "valid_from": "2024-01-01", "valid_to": None, "applicability": {}, "title": other,
            "text": text, "offsets": [0, len(text)], "source_uri": f"study-training://{brief_id}",
            "split": "train", "track": "constructed_train",
            "derived_from": {"source_id": briefing["source_id"], "start": 0, "end": min(12000, len(briefing["text"]))}})
        scope_ids = [source["source_id"] for source in scope]
        cases = [
            (f"What is the current Windows filing allowance for {program}?", [("2024", "Windows")], "ADD", "current", "at"),
            ("Change the platform to Linux. What is the allowance now?", [("2024", "Linux")], "REPLACE", "current", "at"),
            (f"Remove the platform condition for {program}; list both current platform allowances.", [("2024", "Windows"), ("2024", "Linux")], "REMOVE", "current", "at"),
            (f"Use {program} policy version 2018 on Linux. What is the allowance?", [("2018", "Linux")], "ADD", "explicit", "at"),
            (f"What was {program}'s Linux allowance on 2020-06-01?", [("2018", "Linux")], "REPLACE", "as_of", "at"),
            (f"Now use the current {program} Linux policy and give its allowance.", [("2024", "Linux")], "REPLACE", "current", "at"),
            ("What did its allowance look like before that policy?", [("2018", "Linux")], "KEEP", "as_of", "before"),
            (f"For {program} on Linux, what allowance applies in the version immediately after 2018?", [("2024", "Linux")], "KEEP", "as_of", "after"),
            (f"Compare {program}'s Linux allowances under policy versions 2018 and 2024.", [("2018", "Linux"), ("2024", "Linux")], "COMPARE", "compare", "at"),
            (f"Switch to {other}. Explain its supplied briefing in detail, covering every section, in about 900 words.", [], "ADD", "current", "at"),
            (f"Return to {program}. Give its current Linux filing allowance.", [("2024", "Linux")], "REPLACE", "current", "at"),
            ("Switch to Windows and compare its previous and current policy allowances.", [("2018", "Windows"), ("2024", "Windows")], "COMPARE", "compare", "before"),
            (f"Remove platform and version restrictions for {program}; list both platforms under both supplied policy versions.",
                [("2018", "Windows"), ("2018", "Linux"), ("2024", "Windows"), ("2024", "Linux")], "REMOVE", "compare", "at")]
        for number, (question, keys, update, mode, temporal) in enumerate(cases, 1):
            selected = [facts[key] for key in keys]
            if number == 10:
                selected = [{"passage_id": brief_id, "start": 0, "end": len(text), "quote": text}]
            positive_ids = {fact["passage_id"] for fact in selected}
            answer = (text if number == 10 else "; ".join(
                f"{fact['version']} {fact['platform']}: {fact['value']} credits" for fact in selected))
            tasks.append({"task_id": f"{family}:{number}", "conversation_id": family, "turn": number,
                "split": "train", "track": "constructed_train", "question": question, "reference_answer": answer,
                "grounding": selected, "scope_source_ids": scope_ids,
                "negative_source_ids": [source_id for source_id in scope_ids if source_id not in positive_ids],
                "corpus_has_answer": True, "oracle_scope_complete": True,
                "corpus_answer_origin": "constructed_complete_policy_scope",
                "oracle_condition_update": update, "oracle_version_mode": mode, "oracle_temporal_relation": temporal,
                "source_partition": "native_multidoc2dial_train", "source_document_family": family})
        documents.extend(scope)
    write_jsonl(destination/"sources.jsonl", documents)
    write_jsonl(destination/"questions.jsonl", tasks)
    manifest = {"schema_version": 1, "split": "train", "track": "constructed_train",
        "families": families, "turns": len(tasks), "sources": len(documents),
        "source_path": str(source_path.resolve()), "output_dir": str(destination.resolve()),
        "family_prefix": "constructed-train-policy", "dev_family_prefix": "training-dev-policy",
        "scope_oracle": "complete supplied policy family and separate briefing topic", "public_test_assets_read": []}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def prepare_gain_capture(config, matrix):
    """Reuse the policy dialogue prefix and split by the original document family."""
    settings = config.get("training", {})
    root = Path(settings.get("controlled_dir", "runs/study/constructed_train")).resolve()
    original_path = Path(settings.get("data_dir", "runs/study/data"))/"sources.jsonl"
    originals = {row["source_id"]: row["document_family"] for row in read_jsonl(original_path)}
    source_path = root/"sources.jsonl"
    sources = {row["source_id"]: row for row in read_jsonl(source_path)}
    rows, families = [], {}
    for row in read_jsonl(root/"questions.jsonl"):
        if row["turn"] > 9:
            continue
        scope = [key for key in row["scope_source_ids"]
                 if sources[key]["document_family"] == row["source_document_family"]]
        family, = {originals[sources[key]["derived_from"]["source_id"]] for key in scope}
        families[row["conversation_id"]] = family
        rows.append({**row, "scope_source_ids": scope, "Collection": "constructed_train/conditions",
                     "input": [{"speaker": "user", "text": row["question"]}],
                     "targets": [{"text": row["reference_answer"]}], "contexts": [],
                     "coverage_mode": "selected_sources"})
    native_families = sorted(set(families.values()))
    if len(native_families) < 2:
        raise ValueError("condition gain capture needs distinct original source families for train and dev")
    dev = set(native_families[::5])
    task_path = root/"gain-tasks.jsonl"
    write_jsonl(task_path, rows)
    members = [{"task_id": row["task_id"], "conversation_id": row["conversation_id"], "turn": row["turn"],
                "split": "dev" if families[row["conversation_id"]] in dev else "train"} for row in rows]
    cohort_path = root/"gain-cohort.json"
    cohort_path.write_text(json.dumps({"schema_version": 1, "members": members,
        "original_family_by_conversation": families}, ensure_ascii=False, indent=2)+"\n")
    datasets, cohorts = {}, {}
    for split in ("train", "dev"):
        track = "condition_gain_"+split
        datasets[track] = {"input": str(task_path), "supplied_source_catalogue": str(source_path), "protocol": "P3"}
        cohorts[track] = [row["task_id"] for row in members if row["split"] == split]
    matrix_path = root/"gain-matrix.json"
    matrix_path.write_text(json.dumps({"datasets": datasets, "cohorts": {"train": cohorts},
        "methods": [{"id": "B6_native", "strategy": "B6"}], "resources": matrix["resources"]}, indent=2)+"\n")
    return {"matrix": str(matrix_path), "cohort": str(cohort_path), "sources": str(source_path),
            "source_families": {"train": len(native_families)-len(dev), "dev": len(dev)},
            "tasks": {track: len(ids) for track, ids in cohorts.items()}}


async def rollout_controlled(config, *, concurrency=8):
    from controller.config import create_controller, load_config

    settings = config.get("training", {})
    root = Path(settings.get("controlled_dir", "runs/study/constructed_train"))
    sources = {row["source_id"]: row for row in read_jsonl(root/"sources.jsonl")}
    questions = list(read_jsonl(root/"questions.jsonl"))
    grouped = {}
    for row in questions:
        grouped.setdefault(row["conversation_id"], []).append(row)
    recipe = load_config(settings.get("reader_config", "configs/qwen3_32b.yaml"))
    recipe = replace(recipe, state_dir=str(root/"state"), engine=replace(recipe.engine,
        base_url=settings.get("reader_base_url", "http://127.0.0.1:30001")),
        context=replace(recipe.context, max_new_tokens=settings.get("controlled_output_tokens", 2048)))
    service = create_controller(recipe)
    output = root/"turns.jsonl"
    completed = {row["task_id"] for row in read_jsonl(output)} if output.exists() else set()
    sink = output.open("a", encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)

    async def run(conversation, rows):
        async with semaphore:
            for row in rows:
                if row["task_id"] in completed:
                    continue
                blocks = tuple(make_evidence(sources[source_id]["text"], source_id,
                    corpus_id="constructed_train", passage_id=source_id, version=sources[source_id]["version"],
                    source_uri=sources[source_id]["source_uri"], title=sources[source_id]["title"])
                    for source_id in row["scope_source_ids"])
                session_id = "study-constructed-train:"+conversation
                snapshot = service.store.read_snapshot(session_id, "constructed-train-v1")
                result = await service.serve_turn(session_id, row["question"], blocks, request_id=row["task_id"],
                                                  corpus_snapshot_id="constructed-train-v1")
                record = {**row, "actual_history": to_jsonable(snapshot.raw_history),
                    "snapshot_id": "constructed-train-v1", "answer": result.assistant_text,
                    "response": asdict(result), "rollout_policy": "qwen3_32b_rebuild_complete_constructed_scope",
                    "source_scope": to_jsonable(tuple(block.key for block in blocks))}
                sink.write(json.dumps(record, ensure_ascii=False)+"\n")
                sink.flush()
                completed.add(row["task_id"])
                if len(completed) % 13 == 0:
                    print(json.dumps({"completed": len(completed), "target": len(questions)}), flush=True)
    try:
        async with asyncio.TaskGroup() as tasks:
            for conversation, rows in grouped.items():
                tasks.create_task(run(conversation, rows))
    finally:
        sink.close()
        await service.aclose()
        service.store.close()
    records = list(read_jsonl(output))
    result = {"status": "completed", "track": "constructed_train", "conversations": len(grouped),
        "turns": len(records), "input_tokens": sum(row["response"]["input_tokens"] for row in records),
        "output_tokens": sum(row["response"]["output_tokens"] for row in records),
        "raw_output_path": str(output.resolve()), "source_scope_complete": True}
    (root/"rollout-manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare-gain", "rollout"))
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--matrix", default="configs/study_matrix.yaml")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.command == "prepare":
        result = build_controlled_tasks(config)
    elif args.command == "prepare-gain":
        result = prepare_gain_capture(config, yaml.safe_load(Path(args.matrix).read_text()))
    else:
        result = asyncio.run(rollout_controlled(config, concurrency=args.concurrency))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
