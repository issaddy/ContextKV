"""Offline, answer-bound paired quality readiness and declared-protocol acceptance.

Reads saved runs/audits/scores only. Does not generate ratings or change a matrix gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from bench.judge import DIMENSIONS, evaluation_binding
from bench.official import load_jsonl
from bench.report import bind_quality_rows, paired_quality

JUDGE_TRANSPORT_FIELDS = {"base_url", "api_key_env"}


def judge_protocol(judge):
    """Freeze all recorded scoring controls, including future adapter settings.

    Endpoint locations and credential variable names are transport details; model
    identity remains operator-declared. Everything else, including timeouts,
    thinking, constrained output and full native rubrics, belongs to the protocol.
    """
    return {key: value for key, value in judge.items() if key not in JUDGE_TRANSPORT_FIELDS}


def audit_material_bound(row, result, online):
    """Reuse native bodies or controller content hashes without duplicating them."""
    if "visible_evidence_manifest" in result and "audit_evidence" in row:
        # Matching native result/audit copies can both omit or alter a body.
        # Always bind the complete body set to the original visible keys;
        # controller responses need not duplicate those bodies in the result.
        expected = {json.dumps(span["key"], sort_keys=True)
                    for span in result["visible_evidence_manifest"]}
        actual = set()
        for block in row["audit_evidence"]:
            key = block["key"]
            identity = json.dumps(key, sort_keys=True)
            if identity in actual or sha256(block["text"].encode("utf-8")).hexdigest() != key.get("content_sha256"):
                raise ValueError(f"audit evidence body differs from its original visible key: {row['task_id']}")
            actual.add(identity)
        if actual != expected:
            raise ValueError(f"audit evidence bodies differ from the original visible material: {row['task_id']}")
        return True
    return not any((row.get("visible_evidence_manifest"), row.get("audit_evidence"),
                    online.get("contexts"), online.get("supporting_contexts")))


def served_models(row):
    """Read response identities already retained by ordinary/native judges."""
    values = [row.get("served_model"),
              *(grade.get("served_model") for grade in (row.get("native_results") or {}).values())]
    known = [value for value in values if value is not None]
    if any(not isinstance(value, str) or not value.strip() for value in known):
        raise ValueError("recorded served_model must be a nonempty model name")
    return tuple(sorted(set(known)))


def evaluation_tasks(spec, tasks):
    """One declared applicability map shared by both methods, never score filtering."""
    dimensions = tuple(spec.get("dimensions", DIMENSIONS))
    if not dimensions or len(set(dimensions)) != len(dimensions) or set(dimensions) - set(DIMENSIONS):
        raise ValueError("dimensions must select unique supported quality dimensions")
    declared = spec.get("task_scope", {})
    if set(declared) - set(dimensions):
        raise ValueError("task_scope must name declared quality dimensions")
    selected = {}
    for dimension in dimensions:
        ids = declared.get(dimension, sorted(tasks))
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or set(ids) - tasks):
            raise ValueError("task_scope needs nonempty unique task IDs from the paired cohort")
        selected[dimension] = sorted(ids)
    return dimensions, selected


def evaluator_identity(row, generator):
    """Identity is recorded provenance, not remote weight attestation."""
    source = row.get("source")
    if source in {"human_blind", "human_blind_aggregate"}:
        reviewers = row.get("reviewer_ids", [row.get("reviewer_id")])
        if not reviewers or not all(reviewers):
            return "unknown", None
        return "human_blind", ("human_blind", tuple(sorted(reviewers)), row.get("aggregation", "single_reviewer"), row.get("rubric_version"))
    if source != "offline_judge":
        return "unknown", None
    judge = row.get("judge", {})
    if not all(judge.get(k) for k in ("model", "revision", "prompt", "provider")) or not all(generator.get(k) for k in ("model", "revision")):
        return "unknown", None
    served = served_models(row)
    same = (judge["model"] == generator["model"] or judge["revision"] == generator["revision"]
            or generator["model"] in served)
    if len(served) > 1:
        # One row cannot assert a single evaluator when dimensions disagree.
        return "self_judge" if same else "unknown", None
    signature = ("offline_judge", json.dumps(judge_protocol(judge), sort_keys=True),
                 served,
                 row.get("reference_targets_supplied"),
                 json.dumps(row.get("task_requirements"), sort_keys=True))
    related = bool(judge.get("base_family") and generator.get("base_family")
                   and judge["base_family"] == generator["base_family"])
    kind = "self_judge" if same else "related_model_family_declared" if related else "independent_model_declared"
    return kind, signature


def frozen_evaluator_matches(protocol, row):
    if protocol["kind"] == "human_blind":
        return (row.get("source") in {"human_blind", "human_blind_aggregate"}
                and bool(protocol.get("rubric_version")) and row.get("rubric_version") == protocol["rubric_version"]
                and row.get("aggregation", "single_reviewer") == protocol.get("aggregation"))
    if protocol["kind"] == "offline_judge":
        required = ("provider", "model", "revision", "prompt", "temperature", "max_tokens")
        served = served_models(row)
        expected_model = protocol.get("served_model", protocol.get("judge", {}).get("model"))
        return (row.get("source") == "offline_judge" and all(k in protocol.get("judge", {}) for k in required)
                and judge_protocol(row.get("judge", {})) == judge_protocol(protocol["judge"])
                and (served == (expected_model,) if "served_model" in protocol else not served or served == (expected_model,))
                and "reference_targets_supplied" in protocol
                and row.get("reference_targets_supplied") == protocol["reference_targets_supplied"])
    raise ValueError("frozen evaluator kind must be human_blind or offline_judge")


def _read_method(spec):
    run_path = Path(spec["run"])
    payload = json.loads(run_path.read_text())
    manifest_path = Path(spec["manifest"]) if spec.get("manifest") else (
        run_path.with_name(run_path.name.replace("benchmark-", "manifest-", 1))
        if run_path.name.startswith("benchmark-") else None)
    manifest = json.loads(manifest_path.read_text()) if manifest_path and manifest_path.is_file() else {}
    run = payload.get("report", payload)
    audits = load_jsonl(spec["audit"])
    expected = bind_quality_rows(run["turns"], audits, require_answer=True)
    if {r["task_id"] for r in audits} != expected.keys():
        raise ValueError("audit must contain every successful run answer")
    bindings, unbound_audits = {}, []
    for row in audits:
        task = row["task_id"]
        turn = expected[task]
        online, result = turn.get("online_turn") or {}, turn.get("result") or {}
        for audit_key, online_key in (("query", "query"), ("protocol", "protocol"),
                                      ("actual_history", "history")):
            if online_key in online and audit_key in row and row[audit_key] != online[online_key]:
                raise ValueError(f"audit {audit_key} differs from the original online turn: {task}")
        # Source restoration can expand the reading contexts after retrieval.
        # Bind the ranking to its original rows, and the material actually read
        # to the engine result below; neither is a substitute for the other.
        retrieved = online.get("retrieval_contexts")
        if retrieved is None:
            retrieved = online.get("contexts")
        if retrieved is not None and "retrieved_evidence" in row and row["retrieved_evidence"] != retrieved:
            raise ValueError(f"audit retrieved_evidence differs from the original online turn: {task}")
        material_fields = ("visible_evidence_manifest", "audit_evidence", "citation_scope")
        for key in ("input_ids", "output_ids", *material_fields):
            if key in result and (key not in row or row[key] != result[key]):
                raise ValueError(f"audit {key} differs from the original run result: {task}")
        binding = evaluation_binding(row)
        recorded_hash = result.get("input_hash") or (result.get("receipt") or {}).get("input_hash")
        if recorded_hash and "input_hash" in binding and binding["input_hash"] != recorded_hash:
            raise ValueError(f"audit input_ids differ from the original input_hash: {task}")
        if (not all(k in online for k in ("query", "protocol", "history", "contexts"))
                or not all(k in row for k in ("query", "protocol", "actual_history", "retrieved_evidence"))
                or "input_hash" not in binding or not (recorded_hash or "input_ids" in result)
                or not audit_material_bound(row, result, online)
                or "citation_scope" in row and "citation_scope" not in result):
            unbound_audits.append(task)
        bindings[task] = binding
    scores = load_jsonl(spec["scores"]) if spec.get("scores") else []
    if spec.get("score_method") is not None:
        scores = [r for r in scores if r.get("method") == spec["score_method"]]
    bind_quality_rows(run["turns"], scores, require_answer=True)
    unbound_scores = []
    for row in scores:
        binding = bindings[row["task_id"]]
        for key in ("query", "input_hash"):
            if key in row and row[key] != binding.get(key):
                raise ValueError(f"score {key} differs from its audited input: {row['task_id']}")
        if not all(key in row and row[key] is not None for key in ("query", "input_hash")):
            unbound_scores.append(row["task_id"])
        values = row.get("scores")
        if values is not None and (set(values) - set(DIMENSIONS) or any(
                v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1)
                for v in values.values())):
            raise ValueError("quality scores need supported dimensions, finite [0,1] values or null")
    by_id = {r["task_id"]: r for r in scores}
    identities = {task: evaluator_identity(row, spec.get("generator", {})) for task, row in by_id.items()}
    observed = {payload.get("measurement_kind"), manifest.get("measurement_kind"),
                *(r.get("measurement_kind") for r in audits),
                *((r.get("result") or {}).get("measurement_kind") for r in run["turns"]),
                *(r.get("generation_measurement_kind") for r in scores)}
    simulated = payload.get("engine_kind") == "mock" or bool(observed & {"simulation", "fake_http_correctness"})
    incomplete_run = any(r["status"] != "success" for r in run["turns"]) or bool(run.get("unfinished_turns")) or bool(run.get("timed_out_sessions"))
    coverage = {k: {"scored": sum((r.get("scores") or {}).get(k) is not None for r in scores),
                    "missing_task_ids": sorted(task for task in expected if (by_id.get(task, {}).get("scores") or {}).get(k) is None)}
                for k in DIMENSIONS}
    original_splits = [value["workload"]["split"] for value in (payload, manifest)
                       if (value.get("workload") or {}).get("split") is not None]
    split_hashes = [digest for value in (payload, manifest)
                    for digest in (value.get("split_manifest_sha256"), (value.get("dataset") or {}).get("split_manifest_sha256")) if digest]
    return {"audit": {r["task_id"]: r for r in audits}, "scores": by_id, "identities": identities,
            "original_splits": original_splits, "split_hashes": split_hashes,
            "summary": {"expected_tasks": len(expected), "sessions": len({r["conversation_id"] for r in audits}),
                        "unbound_audit_task_ids": sorted(unbound_audits), "unbound_score_task_ids": sorted(unbound_scores),
                        "measurement_kinds": sorted(kind for kind in observed if kind is not None),
                        "run_manifest": str(manifest_path) if manifest else None,
                        "scored_rows": len(scores), "coverage": coverage,
                        "evaluator_kinds": dict(Counter(kind for kind, _ in identities.values())),
                        "truncated_answers": sum(bool(r.get("truncated")) for r in audits),
                        "simulation": simulated, "generation_incomplete": incomplete_run,
                        "provenance": spec}}


def evaluate(spec):
    scope = spec["scope"]
    if scope not in {"dev", "heldout"}:
        raise ValueError("quality scope must be dev or heldout")
    epsilon = spec.get("epsilon", .01)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be a prespecified margin in (0,1)")
    methods = {name: _read_method(value) for name, value in spec["methods"].items()}
    reference_name = spec["reference"]
    reference = methods[reference_name]
    if len(methods) < 2 or not reference["audit"]:
        raise ValueError("paired acceptance needs at least two nonempty methods")
    declared_dimensions, task_scope = evaluation_tasks(spec, set(reference["audit"]))
    evaluated_tasks = set().union(*task_scope.values())
    for method in methods.values():
        summary = method["summary"]
        summary["all_score_coverage"] = summary["coverage"]
        summary["coverage"] = {dimension: {
            "expected": len(tasks),
            "scored": sum((method["scores"].get(task, {}).get("scores") or {}).get(dimension) is not None for task in tasks),
            "missing_task_ids": [task for task in tasks if (method["scores"].get(task, {}).get("scores") or {}).get(dimension) is None],
        } for dimension, tasks in task_scope.items()}
        summary["evaluator_kinds"] = dict(Counter(kind for task, (kind, _) in method["identities"].items()
                                                 if task in evaluated_tasks))
        summary["unbound_score_task_ids"] = sorted(set(summary["unbound_score_task_ids"]) & evaluated_tasks)
    cohort = sorted({r["conversation_id"] for r in reference["audit"].values()})
    split_verified = False
    split_digest = None
    if spec.get("split_manifest"):
        split_bytes = Path(spec["split_manifest"]).read_bytes()
        split = json.loads(split_bytes)
        split_digest = sha256(split_bytes).hexdigest()
        dev, test = set(split["dev"]), set(split["test"])
        if not dev or not test or dev & test:
            raise ValueError("split manifest must have disjoint nonempty dev/test conversations")
        partition = dev if scope == "dev" else test
        if not set(cohort) <= partition:
            raise ValueError("audit conversations are not in the declared split; dev cannot be relabeled heldout")
        split_verified = True
    for method in methods.values():
        if any(value != ("dev" if scope == "dev" else "test") for value in method["original_splits"]):
            raise ValueError("original run workload.split contradicts the requested quality scope")
        if split_digest and any(digest != split_digest for digest in method["split_hashes"]):
            raise ValueError("original run split_manifest_sha256 differs from the supplied fixed split")
        method["summary"]["original_split_binding_verified"] = bool(split_digest and method["split_hashes"])
    frozen = None
    if spec.get("frozen_protocol"):
        frozen = json.loads(Path(spec["frozen_protocol"]).read_text())
        if any(frozen.get(k) != value for k, value in {
                "scope": scope, "reference": reference_name, "epsilon": epsilon, "dimensions": list(declared_dimensions)}.items()):
            raise ValueError("frozen quality protocol differs from requested comparison")
        if evaluation_tasks(frozen, set(reference["audit"]))[1] != task_scope:
            raise ValueError("frozen quality protocol has different task applicability")
        if not isinstance(frozen.get("evaluator"), dict) or "kind" not in frozen["evaluator"]:
            raise ValueError("frozen quality protocol must include its evaluator/rubric")
    comparisons = {}
    for name, candidate in methods.items():
        if candidate["audit"].keys() != reference["audit"].keys():
            raise ValueError("quality acceptance requires identical task coverage; no intersection filtering")
        for task, row in candidate["audit"].items():
            if any(row.get(k) != reference["audit"][task].get(k) for k in ("conversation_id", "query", "protocol")):
                raise ValueError("paired audit query, conversation or protocol differs")
        if name == reference_name:
            continue
        issues = []
        if scope == "heldout" and frozen is None:
            issues.append("no_frozen_protocol")
        if scope == "heldout" and not split_verified:
            issues.append("heldout_split_unverified")
        if scope == "heldout" and any(not method["summary"]["original_split_binding_verified"] for method in (candidate, reference)):
            issues.append("original_run_split_unbound")
        if any(method["summary"]["unbound_audit_task_ids"] for method in (candidate, reference)):
            issues.append("audit_input_binding_missing")
        if any(method["summary"]["unbound_score_task_ids"] for method in (candidate, reference)):
            issues.append("score_input_binding_missing")
        if frozen:
            declared_methods = {key: {k: v for k, v in value.items() if k not in {"scores", "score_method"}}
                                for key, value in spec["methods"].items()}
            if (frozen.get("cohort") != cohort or frozen.get("methods") != declared_methods
                    or not spec.get("split_manifest") or frozen.get("split_manifest") != spec["split_manifest"]):
                issues.append("frozen_cohort_or_methods_or_split_unmatched")
        if candidate["summary"]["simulation"] or reference["summary"]["simulation"]:
            issues.append("simulation_answers")
        if candidate["summary"]["generation_incomplete"] or reference["summary"]["generation_incomplete"]:
            issues.append("generation_incomplete")
        matched = True
        kinds = set()
        for task in evaluated_tasks:
            a = candidate["identities"].get(task, ("unknown", None))
            b = reference["identities"].get(task, ("unknown", None))
            kinds.update((a[0], b[0]))
            if a[1] is None or a[1] != b[1]:
                matched = False
        if not matched:
            issues.append("missing_or_unmatched_evaluator_protocol")
        if "self_judge" in kinds:
            issues.append("generator_self_judging")
        if "unknown" in kinds:
            issues.append("unknown_evaluator_identity")
        if frozen and any(not frozen_evaluator_matches(frozen["evaluator"], row)
                          for method in (candidate, reference) for task, row in method["scores"].items()
                          if task in evaluated_tasks):
            issues.append("frozen_evaluator_protocol_mismatch")
        dimensions = {}
        for dimension in declared_dimensions:
            tasks = task_scope[dimension]
            missing_left = candidate["summary"]["coverage"][dimension]["missing_task_ids"]
            missing_right = reference["summary"]["coverage"][dimension]["missing_task_ids"]
            if missing_left or missing_right:
                dimensions[dimension] = {"status": "incomplete", "candidate_missing": missing_left, "reference_missing": missing_right}
            elif "simulation_answers" in issues:
                dimensions[dimension] = {"status": "not_validated_simulation"}
            else:
                dimensions[dimension] = {"status": "complete", **paired_quality(
                    [candidate["scores"][task] for task in tasks],
                    [reference["scores"][task] for task in tasks], dimension=dimension,
                    epsilon=epsilon, seed=spec.get("seed", 42), samples=spec.get("bootstrap_samples", 5000))}
            dimensions[dimension]["expected_tasks"] = len(tasks)
        passed = all(row.get("decision") == "noninferior" for row in dimensions.values())
        eligible = passed and not issues
        comparisons[name] = {"reference": reference_name, "dimensions": dimensions, "evaluator_protocol_matched": matched,
                             "acceptance_blockers": issues, "all_dimensions_noninferior_on_evaluator": passed,
                             "dev_quality_eligible": eligible if scope == "dev" else None,
                             "heldout_evaluator_noninferiority_established": eligible and scope == "heldout" and frozen is not None,
                             "quality_noninferiority_established": eligible and scope == "heldout" and frozen is not None and kinds == {"human_blind"}}
    return {"schema_version": 2, "scope": scope, "epsilon": epsilon, "dimensions": list(declared_dimensions),
            "task_scope": task_scope, "quality_claim_scope": "declared_dimensions_and_tasks_only",
            "estimand": "paired_turn_mean_clustered_by_session", "reference": reference_name,
            "freeze_provenance": "operator_supplied_protocol_not_time_attestation" if frozen else "not_supplied",
            "frozen_protocol": spec.get("frozen_protocol"),
            "split_manifest": spec.get("split_manifest"), "split_membership_verified": split_verified,
            "cohort": cohort,
            "generator_identity_provenance": "operator_declared_model_and_revision",
            "interpretation": "development_is_exploratory; automated_scores_are_evaluator_relative; no_matrix_is_modified",
            "methods": {name: row["summary"] for name, row in methods.items()}, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="scope/reference/epsilon and methods with run, audit, optional scores/generator")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = evaluate(json.loads(Path(args.config).read_text()))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
