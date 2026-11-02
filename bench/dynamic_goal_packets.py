"""Offline constructed-dev witnesses from actual answer-driven goal rollouts.

The controlled version corpus supplies literal facts and applicability metadata.
Actual P3 records supply the question, branch, history, E0 and candidate pool.
Public test data and online retrieval never enter this builder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from bench.evidence_diagnosis import (ApplicableScopeReview, OracleAnnotations, RequirementWitnesses,
                                     WitnessGroup, review_binding)
from controller.config import create_model_template, load_config
from controller.context_schema import Condition, ContextPlan, EvidencePool, Need, SourceSpan, TaskState, TextRef, VersionIntent
from controller.evidence import make_evidence
from controller.serialization import decode, to_jsonable


SCENARIOS = {
    "current_windows": ("initial",),
    "correct_linux": ("condition_correction",), "clarify_linux": ("condition_correction",),
    "previous_policy": ("follow_up", "version_change"), "clarify_previous": ("follow_up", "version_change"),
    "compare": ("follow_up", "version_change"), "clarify_old": ("follow_up", "version_change"),
    "remove_platform": ("follow_up", "condition_correction"),
    "topic_switch": ("topic_switch",), "source_revisit": ("revisit",),
}
REQUIRED_SCENARIOS = ("follow_up", "condition_correction", "topic_switch", "revisit", "version_change")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _history_qa(rows):
    return [{name: row[name] for name in ("turn_id", "user_text", "assistant_text")} for row in rows]


def _find_user_ref(text, question, history):
    current = len(history)+1
    for turn, value in ((current, question), *((row["turn_id"], row["user_text"]) for row in reversed(history))):
        start = value.find(text)
        if start >= 0:
            return TextRef(turn, "user", start, start+len(text))
    raise ValueError(f"verified task value has no literal user-history origin: {text}")


def _target_documents(goal, conditions, original, contexts, offline_reference):
    family = original["contexts"][0]["document_family"]
    own = [doc for doc in contexts if doc["document_family"] == family]
    versions = [version for _, version in sorted({(doc["valid_from"], doc["version"]) for doc in own})]
    current = versions[-1]
    if goal == "topic_switch":
        target = conditions["TOPIC"]
        ids = set(offline_reference["source_document_ids"])
        return [doc for doc in contexts if doc["document_family"] == target and doc["document_id"] in ids]
    selected_versions = ([versions[-2]] if goal in {"previous_policy", "clarify_previous"} else
        conditions["VERSIONS"].split(",") if goal in {"compare", "clarify_old"} else [current])
    platform = conditions["PLATFORM"]
    return [doc for doc in own if doc["version"] in selected_versions and
            (platform == "removed" or doc["applicability"]["PLATFORM"] == platform)]


def _full_pool(documents, source_pool, e0, tokenizer):
    existing = (*source_pool.source_blocks, *e0.selected_set.source_blocks)
    corpus_ids = {key.corpus_id for key in (*[block.key for block in existing], *e0.task_state.source_scope)}
    if len(corpus_ids) != 1:
        raise ValueError("constructed diagnostic sources require one original corpus identity")
    corpus_id = next(iter(corpus_ids))
    blocks, metadata = [], {}
    for doc in documents:
        block = next((block for block in existing if block.key.document_id == doc["document_id"]
            and block.key.version == doc["version"] and block.text == doc["text"]), None)
        if block is None:
            block = make_evidence(doc["text"], doc["document_id"], corpus_id=corpus_id,
                passage_id=doc["document_id"], version=doc["version"],
                source_uri=doc.get("source") or doc.get("source_uri"), title=doc["title"])
        blocks.append(block)
        metadata[block.key] = {key: doc[key] for key in ("document_family", "valid_from", "valid_to",
            "published_at", "applicability", "version_order", "current") if key in doc}
    directory = SourceDirectory(CorpusIndex(blocks, corpus_id=corpus_id,
        snapshot_id="constructed-goal-reviewed-applicable-scope", source_metadata=metadata),
        lambda query, **kwargs: (), tokenizer)
    return directory.full_scope(tuple(block.key for block in blocks))


def _verified_state(goal, conditions, documents, leaves, history, question, graph_id):
    current = len(history)+1
    family = documents[0]["document_family"]
    program = (leaves[0]["quote"].partition(" policy version ")[0] if leaves else
               documents[0]["title"].partition(" "+documents[0]["version"])[0])
    subject = _find_user_ref(program, question, history)
    refs, active = [subject], []
    platform = conditions.get("PLATFORM")
    if platform and platform != "removed":
        origin = _find_user_ref(platform, question, history)
        refs.append(origin)
        active.append(Condition("verified:platform", value_spans=(origin,), relation_type="PLATFORM"))
    if goal in {"compare", "clarify_old"}:
        intent = VersionIntent("compare", (family,), tuple(conditions["VERSIONS"].split(",")))
    elif goal in {"previous_policy", "clarify_previous"}:
        anchor = next(row["turn_id"] for row in reversed(history) if "current" in row["user_text"].casefold())
        intent = VersionIntent("as_of", (family,), source_turns=(anchor,), temporal_relation="before")
    elif goal == "topic_switch":
        intent = VersionIntent("explicit", (family,), tuple(dict.fromkeys(doc["version"] for doc in documents)))
    else:
        intent = VersionIntent("current", (family,))
    if goal in {"compare", "clarify_old"}:
        requirements = (("verified:comparison", "compare"),)
    elif goal == "remove_platform":
        requirements = tuple(("verified:allowance:"+doc["applicability"]["PLATFORM"], "lookup") for doc in documents)
    elif goal == "topic_switch" and conditions.get("REQUEST_KIND") != "allowance_lookup":
        requirements = (("verified:topic_summary", "summarize"),)
    else:
        requirements = (("verified:allowance", "lookup"),)
    needs = tuple(Need(identity, (TextRef(current, "user", 0, len(question)),), operation,
        (subject,), tuple(item.condition_id for item in active)) for identity, operation in requirements)
    return TaskState("verified:"+graph_id+":"+family, question,
        tuple(dict.fromkeys(ref for ref in refs if ref.turn_id < current)), tuple(active), needs,
        intent, (), current, program+". "+question)


def _initial_e0(record):
    expanded = record["context_plan"]
    originals = [plan for plan in record["planning_trace"].get("context_plans", ())
                 if plan["plan_id"] == expanded["plan_id"]]
    if len(originals) == 1:
        return originals[0], "planning_trace.context_plans:matched_plan_id", "initial"
    runtime_bound = any(task["input_claim_ids"] or task["kind"] == "REDUCE" for task in expanded["steps"]) or any(
        point["claim_ids"] for point in expanded["answer_plan"]["points"])
    return (expanded, "runtime_expanded_context_plan", "runtime_expanded_only") if runtime_bound else (
        expanded, "record_context_plan_without_runtime_bindings", "initial")


def build_packets(records, dynamic_tasks, version_tasks, version_references, tokenizer, *, provenance):
    if any(row["split"] != "dev" or row["dataset"]["name"] != "version_dev" or
            row["label_origin"] != "constructed_from_native_training_sources" for row in version_tasks):
        raise ValueError("packet witnesses use only independent constructed version_dev from training sources")
    if any(row["split"] != "dev" for row in version_references):
        raise ValueError("public test references cannot enter constructed goal packets")
    if any(row["split"] != "dev" or row["dataset"]["name"] != "dynamic_goal_dev" for row in dynamic_tasks):
        raise ValueError("packet questions use the declared constructed dynamic_goal_dev cohort")
    tasks = {row["task_id"]: row for row in dynamic_tasks}
    originals = {row["conversation_id"]: row for row in version_tasks if row["turn"] == 1}
    leaves_by_source = {}
    reference_records = {}
    for reference in version_references:
        for leaf in reference["source_spans"]:
            key = (leaf["source_id"], leaf["start"], leaf["end"])
            if key in leaves_by_source and leaves_by_source[key] != leaf:
                raise ValueError("constructed literal reference coordinates disagree")
            leaves_by_source[key] = leaf
            reference_records[key] = f"{provenance['version_references']}#{reference['task_id']}:{leaf['span_id']}"
    packets, skipped, scenarios = [], [], Counter()
    for record in records:
        if record["status"] != "completed":
            skipped.append({"task_id": record["task_id"], "reason": "actual_E0_run_not_completed"})
            continue
        if record["split"] != "dev" or record["history_origin"] != "model_actual":
            raise ValueError("packets require an actual P3 development rollout, not author history or public test")
        source = tasks[record["task_id"]]
        decision = record["question_policy"]
        metadata, goal = decision["metadata"], decision["goal_id"]
        if record["question"] != decision["question"] or _history_qa(record["actual_history"]) != metadata["actual_history"]:
            raise ValueError("recorded actual question/history do not match the executed goal policy branch")
        if source["goal_graph_id"] != metadata["graph_id"]:
            raise ValueError("actual goal branch belongs to another frozen conversation")
        conditions = metadata["goal_conditions"]
        original = originals[metadata["graph_id"].removeprefix("dynamic-goal:")]
        documents = _target_documents(goal, conditions, original, source["contexts"], metadata["offline_reference"])
        if not documents:
            raise ValueError("goal conditions have no applicable constructed source")
        initial_e0, e0_origin, e0_stage = _initial_e0(record)
        e0 = decode(ContextPlan, initial_e0)
        if e0.task_state.raw_question != record["question"]:
            raise ValueError("E0 task state no longer contains the actually executed user question")
        pool = decode(EvidencePool, record["candidate_pool"])
        full = _full_pool(documents, pool, e0, tokenizer)
        by_id = {block.key.document_id: block for block in full.source_blocks}
        selected_leaves = [leaf for leaf in leaves_by_source.values() if leaf["source_id"] in by_id]
        witnesses, reviewed = [], []
        for leaf in selected_leaves:
            block = by_id[leaf["source_id"]]
            if block.text[leaf["start"]:leaf["end"]] != leaf["quote"]:
                raise ValueError("reviewed controlled fact is not an exact literal original source span")
            if leaf["version"] != block.key.version or leaf["platform"] != next(
                    doc["applicability"]["PLATFORM"] for doc in documents if doc["document_id"] == leaf["source_id"]):
                raise ValueError("reviewed fact and applicable version/platform metadata disagree")
            witnesses.append(SourceSpan(leaf["span_id"], block.key, leaf["start"], leaf["end"]))
            reviewed.append(reference_records[leaf["source_id"], leaf["start"], leaf["end"]])
        incomplete = ((goal == "topic_switch" and conditions.get("REQUEST_KIND") != "allowance_lookup")
            or {leaf["source_id"] for leaf in selected_leaves} != set(by_id))
        verified = _verified_state(goal, conditions, documents, selected_leaves,
            record["actual_history"], record["question"], metadata["graph_id"])
        verified = replace(verified, source_scope=tuple(by_id[doc["document_id"]].key for doc in documents))
        if goal in {"compare", "clarify_old"}:
            groups = (RequirementWitnesses(verified.needs[0].need_id, (WitnessGroup("both-compared-versions",
                tuple(witnesses), " | ".join(reviewed)),) if not incomplete else ()),)
        elif goal == "remove_platform":
            groups = tuple(RequirementWitnesses(need.need_id, tuple(WitnessGroup("current-"+leaf["platform"],
                (span,), review) for leaf, span, review in zip(selected_leaves, witnesses, reviewed, strict=True)
                if need.need_id.endswith(":"+leaf["platform"]))) for need in verified.needs)
        else:
            groups = (RequirementWitnesses(verified.needs[0].need_id, (WitnessGroup("applicable-controlled-fact",
                tuple(witnesses), " | ".join(reviewed)),) if not incomplete else ()),)
        annotation = OracleAnnotations(**review_binding(e0, record["actual_history"]), requirements=groups,
            review_origin="constructed_goal_policy_and_exact_literal_version_dev_reference_alignment")
        review = ApplicableScopeReview("constructed-goal:"+metadata["graph_id"]+":"+goal,
            tuple(block.key for block in full.source_blocks), tuple(by_id[source_id].key for source_id in
                sorted({leaf["source_id"] for leaf in selected_leaves})),
            tuple(reviewed) if reviewed else (provenance["dynamic_tasks"]+"#"+source["task_id"],),
            "constructed-source-protocol:literal-and-applicability-alignment", not incomplete)
        packets.append({"task_id": record["task_id"], "conversation_id": record["conversation_id"],
            "dataset": "dynamic_goal_dev", "split": "dev", "collection": source["Collection"],
            "actual_history": record["actual_history"], "e0_plan": initial_e0,
            "e0_initial_plan": initial_e0 if e0_stage == "initial" else None,
            "e0_plan_origin": e0_origin, "e0_plan_stage": e0_stage,
            "e0_actual_expanded_plan": record["context_plan"],
            "e0_actual_reading_outcome": record.get("reading_outcome"),
            "source_e0_answer_input_ids": record.get("source_answer_input_ids"),
            "source_context_action": record["response"].get("action"),
            "source_initial_read_inputs": [{key: step[key] for key in ("step_id", "input_ids", "input_messages")}
                for step in (record.get("reading_outcome") or {}).get("steps", ()) if step["step_id"] in
                {task["step_id"] for task in initial_e0["steps"]} and step["input_ids"]],
            "candidate_pool": record["candidate_pool"], "full_applicable_pool": to_jsonable(full),
            "annotations": to_jsonable(annotation), "scope_review": to_jsonable(review),
            "verified_task_state": to_jsonable(verified), "goal_id": goal,
            "goal_conditions": conditions, "offline_reference": metadata["offline_reference"],
            "scenario_tags": SCENARIOS[goal], "review_status": "review_incomplete" if incomplete else "reviewed",
            "review_reason": "generic_topic_summary_has_no_minimal_complete_literal_annotation" if goal == "topic_switch"
                else "applicable_sources_lack_literal_reference" if incomplete else "controlled_facts_and_scope_aligned",
            "review_scope": "constructed_goal_points_in_the_provided_applicable_sources",
            "original_preparation_usage": record["planning_trace"].get("auxiliary_usage"),
            "source_run_cost": record["cost"], "source_run_response": record["response"],
            "source_run_measurement_kind": record.get("measurement_kind"),
            "offline_provenance": provenance})
        scenarios.update(SCENARIOS[goal])
    expected = set(provenance.get("planned_task_ids", tasks))
    missing_actual = sorted(expected-{record["task_id"] for record in records})
    return packets, {"packets": len(packets), "declared_actual_cohort_tasks": len(expected),
        "missing_actual_task_ids": missing_actual,
        "actual_rollout_cohort_complete": not missing_actual and not skipped,
        "reviewed": sum(row["review_status"] == "reviewed" for row in packets),
        "review_incomplete": sum(row["review_status"] == "review_incomplete" for row in packets),
        "scenario_counts": dict(scenarios), "missing_scenarios": [name for name in REQUIRED_SCENARIOS if not scenarios[name]],
        "skipped_actual_runs": skipped, "public_test_assets_read": [], "training_performed": False,
        "scope": "constructed_development_only", "inputs": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dynamic-tasks", required=True)
    parser.add_argument("--version-tasks", required=True)
    parser.add_argument("--version-references", required=True)
    parser.add_argument("--run-config", required=True, help="Current reader tokenizer/model-scope config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source = Path(args.run_dir)
    records = [json.loads(path.read_text()) for path in sorted((source/"records").glob("*.json"))]
    audit_path = source/"audit.jsonl"
    audited_inputs = {row["task_id"]: row["input_ids"] for row in read_jsonl(audit_path)} if audit_path.exists() else {}
    records = [{**record, "source_answer_input_ids": audited_inputs.get(record["task_id"])} for record in records]
    paths = {name: str(Path(getattr(args, name)).resolve()) for name in (
        "run_dir", "dynamic_tasks", "version_tasks", "version_references", "run_config")}
    generation_path = source/"generation_manifest.json"
    generation = json.loads(generation_path.read_text())
    paths.update(source_generation_manifest=str(generation_path.resolve()),
        source_generation_measurement_kind=generation["measurement_kind"], planned_task_ids=generation["task_ids"],
        input_file_sha256={name: hashlib.sha256(Path(paths[name]).read_bytes()).hexdigest()
            for name in ("dynamic_tasks", "version_tasks", "version_references", "run_config")})
    template = create_model_template(load_config(args.run_config).model)
    packets, manifest = build_packets(records, read_jsonl(args.dynamic_tasks), read_jsonl(args.version_tasks),
        read_jsonl(args.version_references), template.tokenizer, provenance=paths)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        for packet in packets:
            stream.write(json.dumps(packet, ensure_ascii=False)+"\n")
    manifest["packets_path"] = str(target.resolve())
    manifest["packets_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    Path(str(target)+".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
