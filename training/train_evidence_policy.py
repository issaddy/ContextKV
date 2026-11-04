"""Train evidence utility on actual answers bound to the fixed reader.

A checkpoint/profile gets its own generation cohort and asset.
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
from training.task_data import select_cohort_rows


from training.preference_rollout import P1_REWARDS, P1_RUBRIC


def reader_binding(config):
    from controller.config import load_config
    from training.preference_rollout import query_encoder_binding
    settings = config["evidence_policy_training"]
    run = load_config(settings["run_config"])
    scope_path = Path(run.model.scope_path)
    return {"reader_checkpoint": run.model.revision, "reader_profile_id": settings["reader_profile_id"],
        "reader_scope": str(scope_path.resolve()), "reader_scope_sha256": file_digest(scope_path),
        "query_encoder": query_encoder_binding(settings["collection"]),
        "scorer_identity": {**settings["scorer_identity"], "rubric": P1_RUBRIC}}


def cohort_directory(config, *, split=None):
    binding = reader_binding(config)
    settings = config["evidence_policy_training"]
    collection = settings["collection"]
    if collection.get("existing_cohort_dir"):
        return Path(collection["existing_cohort_dir"])
    generation_binding = {key: value for key, value in binding.items() if key != "scorer_identity"}
    identity = digest([generation_binding, file_digest(collection["cohort_manifest"])])[:20]
    return Path(settings["output_dir"])/identity/(split or collection["cohort_split"])


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
    selected = select_cohort_rows(selected, settings["collection"]["cohort_manifest"],
                                 settings["collection"]["cohort_split"])
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
    for name in ("reader_checkpoint", "reader_profile_id", "scorer_identity", "query_encoder"):
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


def parent_support_binding(settings):
    path = Path(settings["parent_support_asset"]).resolve()
    files = [path/"support-model.json", path/"heads.pt",
             *sorted(file for file in (path/"encoder").rglob("*") if file.is_file())]
    return {"path": str(path), "files_sha256": {str(file.relative_to(path)): file_digest(file) for file in files}}


def build_reader_bound_pairs(reward_rows, *, binding, weights, minimum_difference=0.01):
    """Rank distinct actual inputs by observed answer quality at equal budgets."""
    grouped = {}
    for row in reward_rows:
        validate_reward(row, binding)
        key = (row["task_id"], digest(row["actual_history"]), row["body_budget"], row["output_budget"])
        grouped.setdefault(key, []).append(row)

    def utility(row):
        return sum(weights[name]*row["quality_components"][name] for name in weights)

    result = []
    for key, group in grouped.items():
        ranked = sorted(group, key=utility, reverse=True)
        for preferred, dispreferred in combinations(ranked, 2):
            if preferred["input_hash"] == dispreferred["input_hash"]:
                continue
            difference = utility(preferred)-utility(dispreferred)
            if difference < minimum_difference:
                continue
            if preferred["task_state"] != dispreferred["task_state"]:
                raise ValueError("utility interventions must keep the same resolved task state")
            incremental_cost = {name: (preferred["cost"][name]-dispreferred["cost"][name]
                if preferred["cost"][name] is not None and dispreferred["cost"][name] is not None else None)
                for name in ("complete_ms", "gpu_gib_seconds", "host_gib_seconds")}
            incremental_cost.update({name: preferred[name]-dispreferred[name]
                for name in ("reader_input_tokens", "reader_output_tokens")})
            result.append({"task_id": key[0], "split": "train", "label_origin": "actual_reader_answers",
                "conversation_id": preferred["conversation_id"], "cohort_split": preferred["cohort_split"],
                "cohort_manifest_sha256": preferred["cohort_manifest_sha256"],
                "task_state": preferred["task_state"], "preferred": preferred, "dispreferred": dispreferred,
                **binding, "reward_weights": weights,
                "quality_difference": difference, "incremental_request_cost": incremental_cost,
                "reward_difference": difference,
                "comparison_budget": {"body_tokens": key[2], "answer_output_tokens": key[3]},
                "actual_history_sha256": key[1]})
    return result


async def collect_fixed_reader_preferences(config, *, limit=None, rescore=False):
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.config import create_controller, create_model_template, load_config
    from controller.session_store import Attempt, load_record
    from training.preference_rollout import collect_preferences, rescore_saved_answers
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    directory = cohort_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    run = load_config(settings["run_config"])
    if run.model.revision != binding["reader_checkpoint"] or run.engine.kind != "sglang":
        raise ValueError("preference collection must use the fixed reader checkpoint and scoped SGLang profile")
    binding_path = directory/"collection_binding.json"
    bound = {**{key: value for key, value in binding.items() if key != "scorer_identity"}, "run_config_sha256": file_digest(settings["run_config"]),
        "cohort_manifest_sha256": file_digest(settings["collection"]["cohort_manifest"]),
        "cohort_split": settings["collection"]["cohort_split"]}
    if binding_path.exists():
        previous = json.loads(binding_path.read_text())["binding"]
        if {key: value for key, value in previous.items() if key != "scorer_identity"} != bound:
            raise ValueError("existing candidate cohort belongs to a different generation binding")
    elif rescore or (directory/"turns.jsonl").exists():
        raise ValueError("saved answers require their original generation binding")
    controller = None
    if not rescore:
        controller = create_controller(replace(run, state_dir=str(directory/"runtime"),
            planner=replace(run.planner, strategy="always_rebuild")))
        info = await controller.engine.info()
        if info.fingerprint.model_revision != binding["reader_checkpoint"] or info.engine_kind != "sglang":
            raise ValueError("active native reader does not match the fixed reader identity")
    import httpx
    async with httpx.AsyncClient(base_url=settings["collection"]["judge_base_url"], timeout=30) as judge:
        response = await judge.post("/context/engine/info", json={})
        response.raise_for_status()
        judge_info = response.json()
    if judge_info["fingerprint"]["model_revision"] != binding["scorer_identity"]["reader_checkpoint"]:
        raise ValueError("actual scorer checkpoint differs from the fixed preference rubric identity")
    if not binding_path.exists():
        binding_path.write_text(json.dumps({"binding": bound, "engine_info": asdict(info)}, indent=2)+"\n")
    preferences = {**config.get("preferences", {}), **settings["collection"],
        "support_asset": settings["parent_support_asset"],
        "run_config": settings["run_config"], "output_dir": str(directory),
        "preferences_path": str(directory/"collector_preferences.jsonl"), **binding}
    actual = {**config, "preferences": preferences}
    ports = [urlparse(preferences["judge_base_url"]).port]
    if not rescore:
        ports.append(urlparse(run.engine.base_url).port)
    sampler = StudyResourceSampler(settings["gpu_devices"], listening_processes(ports))
    await sampler.start()
    started = time.monotonic()
    try:
        if rescore:
            import sqlite3
            db_path = (directory/"runtime"/"sessions.sqlite3").resolve()
            db = sqlite3.connect(db_path.as_uri()+"?mode=ro", uri=True)
            def get_attempt(attempt_id):
                payload, = db.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
                return load_record(db, Attempt, payload)
            try:
                labels = {row["task_id"]: row for row in rows(preferences["turns_path"])}
                async with httpx.AsyncClient(base_url=preferences["judge_base_url"], timeout=180) as judge:
                    scored, calls = await rescore_saved_answers(judge, create_model_template(run.model), get_attempt,
                        labels, directory, preferences, binding["query_encoder"])
                result = {"scored_answers": len(scored), "unscored_answers": len(calls)-len(scored)}
            finally:
                db.close()
        else:
            result = await collect_preferences(actual, controller=controller, limit=limit)
    finally:
        ended = time.monotonic()
        await sampler.stop()
        if controller is not None:
            await controller.aclose()
            controller.store.close()
        stamp = time.time_ns()
        observation = json.dumps(sampler.observation(), indent=2)+"\n"
        phase = "rescore" if rescore else "collection"
        (directory/f"{phase}_physical.{stamp}.json").write_text(observation)
        (directory/f"{phase}_cost.{stamp}.json").write_text(json.dumps({"binding": binding, "scorer_engine_info": judge_info,
            "start": started, "end": ended, "physical": integrate_physical(sampler.samples, started, ended)}, indent=2)+"\n")
    if (directory/P1_REWARDS).exists():
        attach_collection_costs(directory, binding)
    return {**result, "binding": binding, "cohort_directory": str(directory)}


def attach_collection_costs(directory, binding):
    """Attribute the observed physical window across all concurrent branches."""
    from bench.study_resources import attribute_physical, integrate_physical
    directory = Path(directory)
    physical_paths = sorted(directory.glob("collection_physical.*.json"))
    cost_paths = sorted(directory.glob("collection_cost.*.json"))
    samples = sorted((sample for path in physical_paths for sample in json.loads(path.read_text())["samples"]),
                     key=lambda sample: sample["timestamp"])
    rewards = rows(directory/P1_REWARDS)
    turn_path, failure_path = directory/"turns.jsonl", directory/"failed_cases.jsonl"
    turns = rows(turn_path) if turn_path.exists() else []
    failures = rows(failure_path) if failure_path.exists() else []
    windows = {row["response"]["attempt_id"]: row["request_window"] for row in (*rewards, *turns)
               if row["request_window"] is not None}
    windows.update({row["attempt_id"]: row["request_window"] for row in failures if row["request_window"] is not None})
    shares = attribute_physical(samples, windows)
    envelopes = {key: integrate_physical(samples, *window) for key, window in windows.items()}
    attribution = "physical interval occupancy divided across concurrently active successful and failed requests"

    def attach(row, key):
        envelope = envelopes.get(key)
        covered = envelope is not None and envelope["complete_window_coverage"]
        share = {name: shares[key][name] if covered else None
                 for name in ("gpu_gib_seconds", "host_gib_seconds")}
        row["physical_window"] = envelope
        row["request_share"] = {**share, "attribution": attribution, "complete_window_coverage": covered}
        row["physical_window_scope"] = "non-additive whole-device/process envelope during this request"
        row["batch_cost_paths"] = [str(path) for path in cost_paths]
        return share

    for row in rewards:
        share = attach(row, row["response"]["attempt_id"])
        row["cost"] = {"complete_ms": row["response"]["complete_ms"],
            "total_input_tokens": row["input_total_tokens"], "total_output_tokens": row["output_total_tokens"],
            **share, "attribution": attribution}
    write_rows(directory/P1_REWARDS, rewards)
    for row in failures:
        share = attach(row, row["attempt_id"])
        row["physical_cost"] = share if row["request_window"] is not None else None
    if failure_path.exists():
        write_rows(failure_path, failures)
    return rewards


def candidate_quality_diagnostics(rewards, *, weights, declared_task_ids):
    """Describe saved candidate limits; score ties do not identify a deployed choice."""
    groups = {}
    declared = set(declared_task_ids)
    for row in rewards:
        if row["task_id"] not in declared:
            raise ValueError("candidate diagnostic task is outside the declared cohort")
        key = (row["task_id"], digest(row["actual_history"]), row["body_budget"], row["output_budget"],
               digest({name: row[name] for name in (
                   "reader_checkpoint", "reader_profile_id", "scorer_identity", "query_encoder")}))
        groups.setdefault(key, []).append(row)
    details = []
    for key, candidates in groups.items():
        scored = [(row, sum(weights[name]*row["quality_components"][name] for name in weights))
                  for row in candidates]
        oracle = max(quality for _, quality in scored)
        prediction = max(row["evidence_set"]["quality_prediction"] for row, _ in scored)
        tied = [(row, quality) for row, quality in scored
                if row["evidence_set"]["quality_prediction"] == prediction]
        details.append(dict(task_id=key[0], actual_history_sha256=key[1], body_budget=key[2],
            output_budget=key[3], reader_scorer_query_sha256=key[4], scored_candidates=len(scored),
            distinct_inputs=len({row["input_hash"] for row, _ in scored}), oracle_quality=oracle,
            top_prediction=prediction, top_prediction_ties=len(tied),
            top_prediction_input_hashes=sorted({row["input_hash"] for row, _ in tied}),
            prediction_regret_interval=[oracle-max(quality for _, quality in tied),
                                        oracle-min(quality for _, quality in tied)]))
    observed = {row["task_id"] for row in details}
    return dict(scope="saved scored candidates grouped by actual state and protocol; not deployed P0/P1 regret or full-queue quality",
        declared_tasks=len(declared), observed_tasks=len(observed), missing_task_ids=sorted(declared-observed),
        comparison_groups=len(details), groups=details,
        available_group_oracle_mean=(sum(row["oracle_quality"] for row in details)/len(details)
                                     if details else None),
        available_group_prediction_regret_interval=(
            [sum(row["prediction_regret_interval"][bound] for row in details)/len(details) for bound in (0, 1)]
            if details else None))


def prepare_preferences(config):
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    directory = cohort_directory(config)
    collection = settings["collection"]
    split = collection["cohort_split"]
    cohort_path = Path(collection["cohort_manifest"])
    cohort_hash = file_digest(cohort_path)
    members = {row["task_id"]: row for row in json.loads(cohort_path.read_text())["members"] if row["split"] == split}
    rewards = attach_collection_costs(directory, binding)
    parent = parent_support_binding(settings)
    for row in rewards:
        if (row["cohort_split"] != split or row["cohort_manifest_sha256"] != cohort_hash
                or row["task_id"] not in members or row["conversation_id"] != members[row["task_id"]]["conversation_id"]):
            raise ValueError("reward is outside the registered collection partition")
        if Path(row["support_model_asset"]).resolve() != Path(parent["path"]):
            raise ValueError("reward used a different parent support asset")
    cost_fields = ("complete_ms", "gpu_gib_seconds", "host_gib_seconds", "total_input_tokens", "total_output_tokens")
    measured = [row for row in rewards if all(row["cost"][key] is not None for key in cost_fields)]
    pairs = (build_reader_bound_pairs(rewards, binding=binding, weights=settings["reward_weights"],
                                     minimum_difference=settings["minimum_reward_difference"])
             if split == "train" else [])
    write_rows(directory/"preferences.jsonl", pairs)

    def candidates(rows_):
        return {(row["task_id"], row["category"]) for row in rows_ if row["task_id"] in members}

    from training.preference_rollout import saved_rows
    families = {row["task_id"]: row for row in saved_rows(directory/"candidate_families.jsonl") if row["task_id"] in members}
    declared = {(identity, category) for identity, family in families.items() for category in family["evidence_sets"]}
    turns = saved_rows(directory/"turns.jsonl")
    failed_rows = saved_rows(directory/"failed_cases.jsonl")
    answered = candidates(turns)
    scored = candidates(rewards)
    failed = candidates(failed_rows)-answered
    skipped = candidates(saved_rows(directory/"skipped_cases.jsonl"))-answered-failed
    manifest = {"binding": binding, "parent_support_binding": parent,
        "cohort_split": split, "cohort_manifest_sha256": cohort_hash,
        "declared_states": len(members), "states_without_candidate_family": sorted(set(members)-families.keys()),
        "registered_candidates": len(declared), "answered_candidates": len(answered),
        "scored_candidates": len(scored), "unscored_answers": len(answered-scored),
        "failed_candidates": len(failed), "skipped_candidates": len(skipped),
        "recorded_failed_attempts": len({row["attempt_id"] for row in failed_rows if row["task_id"] in members}),
        "pending_registered_candidates": len(declared-answered-failed-skipped),
        "scored_answers_without_measured_cost": len(rewards)-len(measured),
        "answers": len(rewards), "pairs": len(pairs), "paired_states": len({row["task_id"] for row in pairs}),
        "reward_weights": settings["reward_weights"],
        "candidate_quality_diagnostics": candidate_quality_diagnostics(rewards,
            weights=settings["reward_weights"], declared_task_ids=members),
        "cost_scope": "observed candidate request cost; shared preparation and offline scorer costs stay in collection ledgers",
        "source_rewards_sha256": file_digest(directory/P1_REWARDS),
        "preferences_sha256": file_digest(directory/"preferences.jsonl")}
    (directory/"preference_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def train_evidence_policy(config):
    from training.support_model import train_set_utility
    settings, binding = config["evidence_policy_training"], reader_binding(config)
    if settings["collection"]["cohort_split"] != "train":
        raise ValueError("P1 fitting requires the registered train collection")
    directory = cohort_directory(config, split="train")
    prepared = json.loads((directory/"preference_manifest.json").read_text())
    parent = prepared["parent_support_binding"]
    pairs = select_cohort_rows(rows(directory/"preferences.jsonl"), settings["collection"]["cohort_manifest"], "train")
    for pair in pairs:
        for name in ("reader_checkpoint", "reader_profile_id", "scorer_identity", "query_encoder"):
            if pair[name] != binding[name]:
                raise ValueError(f"preference training data use a different {name}")
        for side in ("preferred", "dispreferred"):
            validate_reward(pair[side], binding)
    if Path(settings["output_asset"]).resolve() == Path(parent["path"]):
        raise ValueError("P1 asset must retain the P0 support asset as a separate checkpoint")
    optimizer = settings["optimizer"]
    train_set_utility(directory/"preferences.jsonl", parent["path"],
        settings["output_asset"], cohort_manifest=settings["collection"]["cohort_manifest"],
        device=optimizer["device"], epochs=optimizer["epochs"],
        learning_rate=optimizer["learning_rate"], seed=optimizer["seed"])
    path = Path(settings["output_asset"])/"support-model.json"
    metadata = json.loads(path.read_text())
    metadata.update(reader_binding=binding, parent_support_binding=parent,
        preference_objective={"reward_weights": prepared["reward_weights"]}, policy_stage="P1_reader_bound_actual_answer_utility",
        policy_checkpoint="contextkv-evidence-p1:"+file_digest(Path(settings["output_asset"])/"heads.pt"),
        training_preferences_sha256=file_digest(directory/"preferences.jsonl"),
        evaluation_evidence="separate planner x reading-protocol frozen-history and P3 runs under fixed Qwen; no quality claim from training loss")
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
    parser.add_argument("command", choices=("audit", "collect", "rescore", "prepare", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if "base_study_config" in config:
        from bench.study import merge_config
        config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
    if args.command == "audit":
        result = audit_collection_inputs(config, limit=args.limit or 8)
    elif args.command in {"collect", "rescore"}:
        result = asyncio.run(collect_fixed_reader_preferences(config, limit=args.limit, rescore=args.command == "rescore"))
    elif args.command == "prepare":
        result = prepare_preferences(config)
    else:
        result = asyncio.run(measured_policy_training(config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
