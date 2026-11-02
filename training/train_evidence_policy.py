"""Alternate reader-bound actual-answer preferences and existing utility training.

R1 is explicit: a checkpoint/profile gets its own generation cohort and asset.
The original support encoder/heads remain P0; only the existing graph utility and
answer-disposition heads are optimized by train_set_utility.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

import yaml

from training.reader_data import digest, file_digest, rows, write_rows


def reader_binding(config):
    settings = config["evidence_policy_training"]
    manifest_path = Path(settings["reader_manifest"])
    manifest = json.loads(manifest_path.read_text())
    checkpoint = manifest["reader_id"]
    if manifest["status"] != "trained" or not checkpoint.startswith("contextkv-reader-r1:"):
        raise ValueError("evidence-policy training requires an explicitly trained R1 checkpoint")
    return {"reader_checkpoint": checkpoint, "reader_profile_id": settings["reader_profile_id"],
        "reader_manifest": str(manifest_path.resolve()), "reader_manifest_sha256": file_digest(manifest_path),
        "scorer_identity": settings["scorer_identity"]}


def cohort_directory(config):
    binding = reader_binding(config)
    return Path(config["evidence_policy_training"]["output_dir"])/digest(binding)[:20]


def qualify_collection_rows(label_rows, source_rows):
    """Exclude whole old annotations with missing origins; retain their edges."""
    sources = {row["source_id"]: row["text"] for row in source_rows}
    prior_needs, selected, excluded = {}, [], []
    for row in sorted(label_rows, key=lambda item: (item["conversation_id"], item["turn"])):
        state = row["task_state"]
        history = {turn["turn_id"]: turn for turn in row["actual_history"]}
        available = dict(prior_needs.get(row["conversation_id"], {}))
        available.update((need["need_id"], need) for need in state["needs"])
        reason = ("not_train" if row["split"] != "train" else
            "actual_history_not_reviewed" if row["script_history_review_status"] != "reviewed" else
            "authored_history_binding_mismatch" if row["script_history_mismatch"] else
            "no_verified_grounding" if not row["verified_grounding"] else
            "current_task_mismatch" if state["current_turn_id"] != row["turn"] or state["raw_question"] != row["question"] else None)
        refs = [*state["working_history"], *state["version_intent"]["time_refs"]]
        for condition in state["conditions"]:
            refs.extend(condition["value_spans"])
            refs.extend(ref for ref in (condition["key_span"], condition["predicate_span"]) if ref is not None)
        for need in state["needs"]:
            refs.extend((*need["anchor_spans"], *need["subject_refs"]))
            if not set(need["condition_ids"]) <= {condition["condition_id"] for condition in state["conditions"]}:
                reason = "dangling_active_condition"
        for ref in refs:
            if ref["turn_id"] == row["turn"] and ref["speaker"] == "user":
                text = row["question"]
            elif ref["turn_id"] in history:
                text = history[ref["turn_id"]]["user_text" if ref["speaker"] == "user" else "assistant_text"]
            else:
                reason = "missing_actual_span_origin"
                break
            if not 0 <= ref["start"] < ref["end"] <= len(text):
                reason = "span_outside_actual_message"
                break
        pending = [need["need_id"] for need in state["needs"]]
        visited, active = set(), set()
        def visit(identity):
            if identity in active:
                return "cyclic_need_dependency"
            if identity not in available:
                return "dangling_need_dependency"
            if identity in visited:
                return None
            active.add(identity)
            for predecessor in available[identity]["predecessor_need_ids"]:
                found = visit(predecessor)
                if found is not None:
                    return found
            active.remove(identity)
            visited.add(identity)
            return None
        for identity in pending:
            found = visit(identity)
            if found is not None:
                reason = found
                break
        for span in row["verified_grounding"]:
            if span["passage_id"] not in sources or sources[span["passage_id"]][span["start"]:span["end"]] != span["quote"]:
                reason = "verified_grounding_differs_from_train_original"
                break
        prior_needs[row["conversation_id"]] = available
        if reason is None:
            selected.append(row)
        else:
            excluded.append({"task_id": row["task_id"], "reason": reason})
    return selected, excluded


def audit_collection_inputs(config, *, limit=8):
    settings = config["evidence_policy_training"]
    labels_path = Path(settings["collection"]["turns_path"])
    sources_path = Path(config["training"]["data_dir"])/"sources.jsonl"
    selected, excluded = qualify_collection_rows(rows(labels_path), rows(sources_path))
    selected = selected[:limit]
    directory = cohort_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    cohort = directory/"audited_collection_inputs.jsonl"
    write_rows(cohort, selected)
    write_rows(directory/"collection_input_exclusions.jsonl", excluded)
    actual = json.loads(json.dumps(config))
    actual["evidence_policy_training"]["collection"].update(turns_path=str(cohort.resolve()), turns=len(selected))
    path = directory/"audited_collection_config.yaml"
    path.write_text(yaml.safe_dump(actual, sort_keys=False))
    manifest = {"reader_binding": reader_binding(config), "source_labels": str(labels_path.resolve()),
        "source_labels_sha256": file_digest(labels_path), "source_originals_sha256": file_digest(sources_path),
        "selected_turns": len(selected), "exclusions": dict(Counter(row["reason"] for row in excluded)),
        "selected_cases": [{"task_id": row["task_id"], "actual_history_turns": len(row["actual_history"]),
            "needs": len(row["task_state"]["needs"]), "conditions": len(row["task_state"]["conditions"]),
            "verified_grounding_spans": len(row["verified_grounding"])} for row in selected],
        "cohort_path": str(cohort.resolve()), "collection_config": str(path.resolve()),
        "data_action": "whole rows selected; original spans, dependency edges and actual QA unchanged",
        "model_calls": 0, "purpose": "bounded training collection; not heldout evaluation"}
    (directory/"collection_input_audit.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def validate_reward(row, binding):
    if row["split"] != "train" or row["measurement_kind"] != "hardware":
        raise ValueError("utility supervision requires native-training actual hardware answers")
    if row["history_origin"] not in {"model_actual", "frozen_actual_rollout"}:
        raise ValueError("utility supervision requires actual generated history")
    for name in ("reader_checkpoint", "reader_profile_id", "scorer_identity"):
        if row[name] != binding[name]:
            raise ValueError(f"candidate reward belongs to a different {name}")
    history = [{key: turn[key] for key in ("turn_id", "user_text", "assistant_text")}
               for turn in row["actual_history"]]
    if history != [{key: turn[key] for key in ("turn_id", "user_text", "assistant_text")}
                   for turn in json.loads(row["history_id"])]:
        raise ValueError("candidate answer does not retain its frozen actual history")
    receipt = row["response"]["receipt"]
    if not receipt["generation_complete"] or receipt["input_hash"] != row["input_hash"]:
        raise ValueError("utility labels require their completed native generation receipt")
    for value in row["quality_components"].values():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("quality components must be observed normalized answer scores")
    for name in ("complete_ms", "gpu_gib_seconds", "host_gib_seconds", "total_input_tokens", "total_output_tokens"):
        if row["cost"][name] is None:
            raise ValueError("reader-bound utility data retain full physical and token costs")


def build_reader_bound_pairs(reward_rows, *, binding, weights, minimum_difference=0.01):
    """Only compare actual answers sharing reader, scorer, history and budgets."""
    grouped = {}
    for row in reward_rows:
        validate_reward(row, binding)
        key = (row["task_id"], digest(row["actual_history"]), row["body_budget"], row["output_budget"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in grouped.items():
        ranked = sorted(group, key=lambda row: sum(weights[name]*row["quality_components"][name]
                                                   for name in weights), reverse=True)
        for preferred, dispreferred in combinations(ranked, 2):
            if preferred["input_hash"] == dispreferred["input_hash"]:
                continue
            difference = sum(weights[name]*(preferred["quality_components"][name]-dispreferred["quality_components"][name])
                             for name in weights)
            if difference < minimum_difference:
                continue
            if preferred["task_state"] != dispreferred["task_state"]:
                raise ValueError("utility interventions must keep the same resolved task state")
            result.append({"task_id": key[0], "split": "train", "label_origin": "actual_reader_answers",
                "task_state": preferred["task_state"], "preferred": preferred, "dispreferred": dispreferred,
                **binding, "reward_weights": weights, "reward_difference": difference,
                "comparison_budget": {"body_tokens": key[2], "answer_output_tokens": key[3]},
                "actual_history_sha256": key[1]})
    return result


async def collect_r1_preferences(config, *, limit=None):
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.config import create_controller, load_config
    from training.preference_rollout import collect_preferences
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    directory = cohort_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    run = load_config(settings["run_config"])
    if run.model.revision != binding["reader_checkpoint"] or run.engine.kind != "sglang":
        raise ValueError("R1 preference collection must use its real merged checkpoint and scoped SGLang profile")
    binding_path = directory/"collection_binding.json"
    bound = {**binding, "run_config_sha256": file_digest(settings["run_config"])}
    if binding_path.exists() and json.loads(binding_path.read_text())["binding"] != bound:
        raise ValueError("existing candidate cohort belongs to a different reader/profile/scorer")
    if (directory/"rewards.jsonl").exists() and not binding_path.exists():
        raise ValueError("existing unbound rewards cannot be relabelled as current R1")
    controller = create_controller(replace(run, state_dir=str(directory/"runtime"),
        planner=replace(run.planner, strategy="always_rebuild")))
    info = await controller.engine.info()
    if info.fingerprint.model_revision != binding["reader_checkpoint"] or info.engine_kind != "sglang":
        raise ValueError("active native reader does not match the trained R1 identity")
    import httpx
    async with httpx.AsyncClient(base_url=settings["collection"]["judge_base_url"], timeout=30) as judge:
        response = await judge.post("/context/engine/info", json={})
        response.raise_for_status()
        judge_info = response.json()
    if judge_info["fingerprint"]["model_revision"] != binding["scorer_identity"]["reader_checkpoint"]:
        raise ValueError("actual scorer checkpoint differs from the fixed preference rubric identity")
    binding_path.write_text(json.dumps({"binding": bound, "engine_info": asdict(info),
                                       "scorer_engine_info": judge_info}, indent=2)+"\n")
    preferences = {**config.get("preferences", {}), **settings["collection"],
        "support_asset": settings["parent_support_asset"],
        "run_config": settings["run_config"], "output_dir": str(directory),
        "preferences_path": str(directory/"collector_preferences.jsonl"), **binding}
    actual = {**config, "preferences": preferences}
    ports = [urlparse(run.engine.base_url).port, urlparse(preferences["judge_base_url"]).port]
    sampler = StudyResourceSampler(settings["gpu_devices"], listening_processes(ports))
    await sampler.start()
    started = time.monotonic()
    try:
        result = await collect_preferences(actual, controller=controller, limit=limit)
    finally:
        ended = time.monotonic()
        await sampler.stop()
        await controller.aclose()
        controller.store.close()
        stamp = time.time_ns()
        observation = json.dumps(sampler.observation(), indent=2)+"\n"
        (directory/f"collection_physical.{stamp}.json").write_text(observation)
        (directory/f"collection_cost.{stamp}.json").write_text(json.dumps({"binding": binding,
            "start": started, "end": ended, "physical": integrate_physical(sampler.samples, started, ended)}, indent=2)+"\n")
    attach_collection_costs(directory, binding)
    return {**result, "binding": binding, "cohort_directory": str(directory)}


def attach_collection_costs(directory, binding):
    """Finish saved successes and failed-case accounting after a partial batch."""
    from bench.study_resources import integrate_physical
    directory = Path(directory)
    physical_paths = sorted(directory.glob("collection_physical.*.json"))
    cost_paths = sorted(directory.glob("collection_cost.*.json"))
    samples = sorted((sample for path in physical_paths for sample in json.loads(path.read_text())["samples"]),
                     key=lambda sample: sample["timestamp"])
    rewards = rows(directory/"rewards.jsonl")
    for row in rewards:
        if "cost" not in row:
            start, end = row["request_window"]
            physical = integrate_physical(samples, start, end)
            row["cost"] = {"complete_ms": row["response"]["complete_ms"],
                "total_input_tokens": row["input_total_tokens"], "total_output_tokens": row["output_total_tokens"],
                "gpu_gib_seconds": physical["gpu_gib_seconds"], "host_gib_seconds": physical["host_gib_seconds"]}
        validate_reward(row, binding)
    write_rows(directory/"rewards.jsonl", rewards)
    failure_path = directory/"failed_cases.jsonl"
    if failure_path.exists():
        failures = rows(failure_path)
        for row in failures:
            if row["request_window"] is not None:
                row["physical_cost"] = integrate_physical(samples, *row["request_window"])
            row["batch_cost_paths"] = [str(path) for path in cost_paths]
        write_rows(failure_path, failures)
    return rewards


def prepare_preferences(config):
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    directory = cohort_directory(config)
    rewards = attach_collection_costs(directory, binding)
    pairs = build_reader_bound_pairs(rewards, binding=binding, weights=settings["reward_weights"],
                                    minimum_difference=settings["minimum_reward_difference"])
    write_rows(directory/"preferences.jsonl", pairs)
    manifest = {"binding": binding, "training_split": "train", "answers": len(rewards), "pairs": len(pairs),
        "tasks": len({row["task_id"] for row in pairs}), "reward_weights": settings["reward_weights"],
        "source_rewards_sha256": file_digest(directory/"rewards.jsonl"),
        "reader_updates": "new checkpoint/profile requires newly executed and scored candidate answers"}
    (directory/"preference_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def train_evidence_policy(config):
    from training.support_model import train_set_utility
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    directory = cohort_directory(config)
    pairs = rows(directory/"preferences.jsonl")
    for pair in pairs:
        for name in ("reader_checkpoint", "reader_profile_id", "scorer_identity"):
            if pair[name] != binding[name]:
                raise ValueError(f"preference training data use a different {name}")
        for side in ("preferred", "dispreferred"):
            validate_reward(pair[side], binding)
    if Path(settings["output_asset"]).resolve() == Path(settings["parent_support_asset"]).resolve():
        raise ValueError("P1 asset must retain the P0 support asset as a separate checkpoint")
    optimizer = settings["optimizer"]
    train_set_utility(directory/"preferences.jsonl", settings["parent_support_asset"],
        settings["output_asset"], device=optimizer["device"], epochs=optimizer["epochs"],
        learning_rate=optimizer["learning_rate"], seed=optimizer["seed"])
    path = Path(settings["output_asset"])/"support-model.json"
    metadata = json.loads(path.read_text())
    metadata.update(reader_binding=binding, policy_stage="P1_reader_bound_actual_answer_utility",
        policy_checkpoint="contextkv-evidence-p1:"+file_digest(Path(settings["output_asset"])/"heads.pt"),
        training_preferences_sha256=file_digest(directory/"preferences.jsonl"),
        evaluation_evidence="separate P0/P1 x R0/R1 frozen-history and P3 runs; no quality claim from training loss")
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    return metadata


def require_policy_reader_binding(asset_path, *, reader_checkpoint, reader_profile_id):
    metadata = json.loads((Path(asset_path)/"support-model.json").read_text())
    if metadata["policy_checkpoint"] != "contextkv-evidence-p1:"+file_digest(Path(asset_path)/"heads.pt"):
        raise ValueError("P1 parameter file differs from its recorded checkpoint identity")
    binding = metadata["reader_binding"]
    if binding["reader_checkpoint"] != reader_checkpoint or binding["reader_profile_id"] != reader_profile_id:
        raise ValueError("P1 utility requires the reader checkpoint/profile that generated its rewards")
    return binding


async def rollout_next_round(config, *, limit_conversations=None):
    """Run P1/R1 along its own actual answers, then hand those traces to reader SFT."""
    from bench.study import merge_config, run_dataset
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    require_policy_reader_binding(settings["output_asset"], reader_checkpoint=binding["reader_checkpoint"],
                                  reader_profile_id=binding["reader_profile_id"])
    next_round = settings["next_round"]
    if limit_conversations is None:
        limit_conversations = next_round["conversations"]
    policy = json.loads((Path(settings["output_asset"])/"support-model.json").read_text())
    directory = cohort_directory(config)/"next_reader_rollout"/policy["policy_checkpoint"].split(":")[-1][:20]
    directory.mkdir(parents=True, exist_ok=True)
    questions = rows(next_round["questions_path"])
    if any(row["split"] != "train" for row in questions):
        raise ValueError("the next reader round must use actual native training conversations")
    reader_config = yaml.safe_load(Path(next_round["reader_training_config"]).read_text())
    parent_data_dir = reader_config["reader_training"]["data"]["output_dir"]
    parent_conversations = {row["conversation_id"] for row in rows(Path(parent_data_dir)/"train.jsonl")}
    questions = [row for row in questions if row["conversation_id"] in parent_conversations]
    tasks = [{"dataset": {"name": "reader_policy_train", "revision": "native-training-questions"},
        "task_id": row["task_id"], "conversation_id": row["conversation_id"], "turn": row["turn"],
        "Collection": "reader_policy_train/native", "split": "train",
        "input": [{"speaker": "user", "text": row["question"]}], "contexts": [],
        "targets": [{"text": row["reference_answer"]}]} for row in questions]
    tasks_path = directory/"questions.jsonl"
    write_rows(tasks_path, tasks)
    matrix = yaml.safe_load(Path(next_round["matrix_path"]).read_text())
    from adapters.doc2dial import convert, structural_corpus
    original = Path(config["training"]["multidoc_source"])
    documents = json.loads((original/"multidoc2dial_doc.json").read_text())
    corpus, _, _ = convert(documents, json.loads((original/"multidoc2dial_dial_train.json").read_text()),
        dataset_version="archive-f0c034c24966", source_uri="https://doc2dial.github.io/multidoc2dial/",
        split="train", source_format="multidoc2dial")
    corpus_path = directory/"train_passages.jsonl"
    write_rows(corpus_path, structural_corpus(documents, corpus))
    spec = {**next_round["dataset"], "inputs": {"train": str(tasks_path)},
        "corpus_path": str(corpus_path),
        "split_kind": "native_train_actual_rollout", "weight": 0,
        "primary_metric": "project_condition_version_score", "normalization": 1.0}
    retrieval_recipe = yaml.safe_load(Path(spec["retriever_config"]).read_text())
    retrieval_recipe["device"] = settings["collection"]["encoder_device"]
    retrieval_path = directory/"retriever.yaml"
    retrieval_path.write_text(yaml.safe_dump(retrieval_recipe, sort_keys=False))
    spec["retriever_config"] = str(retrieval_path)
    matrix["datasets"] = {"reader_policy_train": spec}
    matrix["resources"]["gpu_devices"] = settings["gpu_devices"]
    actual = merge_config(config, {"generation": {"run_config": settings["run_config"],
        "teacher_url": settings["collection"]["judge_base_url"],
        "helper_run_config": settings["helper_run_config"]},
        "selection": {"utility": "learned"},
        "reading": {"layout": settings["collection"]["read_layout"], "packing": settings["collection"]["read_packing"]},
        "task_model": {"device": settings["collection"]["encoder_device"]},
        "support_model": {"mode": "trained", "asset": settings["output_asset"],
            "device": settings["collection"]["device"], "preference_asset": settings["output_asset"]}})
    from controller.engine_client import EngineError
    from bench.evidence_policy_comparison import persist_p3_failure, recorded_terminal_budget, terminal_model_budget
    conversations = sorted({row["conversation_id"] for row in tasks})[:limit_conversations]
    turns, rollout_specs, failed = [], [], []
    for conversation in conversations:
        part = directory/"conversations"/digest(conversation)[:20]
        part.mkdir(parents=True, exist_ok=True)
        part_tasks = part/"questions.jsonl"
        write_rows(part_tasks, [row for row in tasks if row["conversation_id"] == conversation])
        local_matrix = merge_config(matrix, {"datasets": {"reader_policy_train": {"inputs": {"train": str(part_tasks)}}}})
        teacher_ledger = actual.get("task_model", {}).get("teacher_record_path",
            str(Path(actual.get("study", {}).get("output_dir", "runs/study"))/"task_teacher_calls.jsonl"))
        recorded = recorded_terminal_budget(part)
        try:
            if recorded:
                results = rows(part/"results.unscored.jsonl")
                failed.append(persist_p3_failure(part, conversation, "; ".join(recorded), teacher_ledger=teacher_ledger))
            else:
                results = await run_dataset(actual, local_matrix, {"id": "P1_R1_train", "strategy": "B7"},
                    "reader_policy_train", "train", part, concurrency=next_round["concurrency"])
        except ExceptionGroup as group:
            known, unknown = group.split(terminal_model_budget)
            if unknown is not None:
                raise unknown
            results = rows(part/"results.unscored.jsonl")
            failed.append(persist_p3_failure(part, conversation, str(known), teacher_ledger=teacher_ledger))
        except EngineError as exc:
            if exc.code != "READ_STEP_OUTPUT_TRUNCATED":
                raise
            results = rows(part/"results.unscored.jsonl")
            failed.append(persist_p3_failure(part, conversation, str(exc), teacher_ledger=teacher_ledger))
        except RuntimeError as exc:
            if not terminal_model_budget(exc):
                raise
            results = rows(part/"results.unscored.jsonl")
            failed.append(persist_p3_failure(part, conversation, str(exc), teacher_ledger=teacher_ledger))
        except ValueError as exc:
            if not terminal_model_budget(exc):
                raise
            results = rows(part/"results.unscored.jsonl")
            failed.append(persist_p3_failure(part, conversation, str(exc), teacher_ledger=teacher_ledger))
        generation = json.loads((part/"generation_manifest.json").read_text())
        snapshot_id = generation["retrieval"]["retrieval"]["corpus_snapshot_id"]
        actual_turns = [{**row, "split": "train", "snapshot_id": snapshot_id, "rollout_policy": "P1_R1_actual_history",
            "reader_checkpoint": binding["reader_checkpoint"], "reader_profile_id": binding["reader_profile_id"]} for row in results]
        turn_path = part/"turns.jsonl"
        write_rows(turn_path, actual_turns)
        turns.extend(actual_turns)
        rollout_specs.append({"turns_path": str(turn_path.resolve()),
            "state_db": str((part/"state"/"sessions.sqlite3").resolve()),
            "source_partition": "native_multidoc2dial_train", "split": "train",
            "reader_identity": binding["reader_checkpoint"], "read_protocol_version": 2})
    write_rows(directory/"turns.jsonl", turns)
    reader = reader_config["reader_training"]
    reader.update(initial_adapter=str(Path(settings["reader_manifest"]).parent/"adapter"),
        output_dir=next_round["reader_output_dir"], merged_model_dir=next_round["reader_merged_dir"])
    reader["data"].update(output_dir=next_round["reader_data_dir"], rollouts=rollout_specs,
        export_turns=None, retrieved_read_turns=0, validation_fraction=0)
    review_path = directory/"reader_p1_review.yaml"
    review_path.write_text(yaml.safe_dump(reader_config, sort_keys=False))
    mix_config = yaml.safe_load(Path(next_round["reader_mix_config"]).read_text())
    mix_config["reader_mix"].update(reader_manifest=settings["reader_manifest"], parent_data_dir=parent_data_dir,
        new_p1_data_dir=next_round["reader_data_dir"], reader_output_dir=next_round["reader_output_dir"],
        reader_merged_dir=next_round["reader_merged_dir"])
    mix_path = directory/"reader_mix.yaml"
    mix_path.write_text(yaml.safe_dump(mix_config, sort_keys=False))
    from training.reader_mix import stage_configs
    _, mixed = stage_configs(mix_config, reader_config)
    next_path = directory/"reader_next_round.yaml"
    next_path.write_text(yaml.safe_dump(mixed, sort_keys=False))
    manifest = {"reader_binding": binding, "planner_asset": settings["output_asset"],
        "planner_checkpoint": policy["policy_checkpoint"],
        "actual_turns": len(turns), "actual_conversations": len({row["conversation_id"] for row in turns}),
        "requested_conversations": conversations, "failed_conversations": failed,
        "history_policy": "P3 along this P1/R1 pair's own committed answers",
        "next_reader_config": str(next_path.resolve()),
        "new_P1_review_config": str(review_path.resolve()), "reader_mix_config": str(mix_path.resolve()),
        "next_steps": [f"python -m training.reader_mix all --config {mix_path} --p1-config {review_path}",
                       f"python -m training.train_reader train --config {next_path}"],
        "supervision": "fresh actual-history/prompt-bound target review required for the new reader round"}
    (directory/"alternation_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


async def measured_policy_training(config):
    from bench.study_resources import StudyResourceSampler, integrate_physical
    import os
    settings = config["evidence_policy_training"]
    sampler = StudyResourceSampler(settings["optimizer"]["physical_gpu_devices"], [os.getpid()])
    await sampler.start()
    start = time.monotonic()
    try:
        metadata = await asyncio.to_thread(train_evidence_policy, config)
    finally:
        end = time.monotonic()
        await sampler.stop()
        output = Path(settings["output_asset"])
        output.mkdir(parents=True, exist_ok=True)
        invocation = time.time_ns()
        (output/f"training_physical.{invocation}.json").write_text(json.dumps(sampler.observation(), indent=2)+"\n")
        (output/f"training_cost.{invocation}.json").write_text(json.dumps({"reader_binding": reader_binding(config),
            "start": start, "end": end, "physical": integrate_physical(sampler.samples, start, end)}, indent=2)+"\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "collect", "prepare", "train", "rollout"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if "base_study_config" in config:
        from bench.study import merge_config
        config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
    if args.command == "audit":
        result = audit_collection_inputs(config, limit=args.limit or 8)
    elif args.command == "collect":
        result = asyncio.run(collect_r1_preferences(config, limit=args.limit))
    elif args.command == "prepare":
        result = prepare_preferences(config)
    elif args.command == "rollout":
        result = asyncio.run(rollout_next_round(config, limit_conversations=args.limit))
    else:
        result = asyncio.run(measured_policy_training(config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
