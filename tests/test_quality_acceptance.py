import json
from pathlib import Path

import pytest

from bench.blind_review import RUBRIC_VERSION, aggregate_ratings, export_blind, import_ratings
from bench.judge import DIMENSIONS, JudgeConfig
from bench.quality_acceptance import evaluate
from controller.templates import token_hash


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def setup_methods(tmp_path, *, source="human_blind", simulation=False, heldout=False):
    split = tmp_path / "split.json"
    cohort = [f"s{i}" for i in range(3)]
    split.write_text(json.dumps({"dev": ["other"] if heldout else cohort, "test": cohort if heldout else ["other"]}))
    methods = {}
    for name in ("candidate", "reference"):
        audits = [{"task_id": f"s{i}:1", "conversation_id": f"s{i}", "query": "Question?", "protocol": "P3",
                   "answer": name + str(i), "measurement_kind": "simulation" if simulation else "hardware",
                   "input_ids": [10+i, 20 if name == "candidate" else 30], "output_ids": [1, 2],
                   "actual_history": [], "audit_evidence": [], "retrieved_evidence": [], "visible_evidence_manifest": []}
                  for i in range(3)]
        run = {"workload": {"split": "test" if heldout else "dev"},
               "report": {"turns": [{"task_id": r["task_id"], "conversation_id": r["conversation_id"],
                                    "online_turn": {"query": r["query"], "protocol": "P3", "history": [], "contexts": []},
                                    "result": {"input_ids": r["input_ids"], "input_hash": token_hash(r["input_ids"]), "output_ids": [1, 2]},
                                    "assistant_text": r["answer"], "status": "success"} for r in audits]}}
        scores = [{"task_id": r["task_id"], "conversation_id": r["conversation_id"], "answer": r["answer"],
                   "query": r["query"], "input_hash": token_hash(r["input_ids"]),
                   "scores": {d: .8 for d in DIMENSIONS}, "source": source, "reviewer_id": "rater-1", "rubric_version": RUBRIC_VERSION} for r in audits]
        run_path, audit_path, score_path = [tmp_path / f"{name}-{kind}.jsonl" for kind in ("run", "audit", "scores")]
        run_path.write_text(json.dumps(run))
        write_jsonl(audit_path, audits)
        write_jsonl(score_path, scores)
        methods[name] = {"run": str(run_path), "audit": str(audit_path), "scores": str(score_path),
                         "generator": {"model": "Generator", "revision": "weight-1"}}
    return {"scope": "dev", "reference": "reference", "epsilon": .01, "bootstrap_samples": 20, "methods": methods,
            "split_manifest": str(split)}


def edit_scores(spec, method, edit):
    path = Path(spec["methods"][method]["scores"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit(rows)
    write_jsonl(path, rows)


def freeze_protocol(spec, tmp_path):
    frozen = tmp_path / "protocol.json"
    frozen.write_text(json.dumps({"scope": spec["scope"], "reference": spec["reference"], "epsilon": .01,
                                 "dimensions": spec.get("dimensions", list(DIMENSIONS)), "task_scope": spec.get("task_scope", {}),
                                 "split_manifest": spec["split_manifest"], "cohort": ["s0", "s1", "s2"],
                                 "methods": {name: {k: v for k, v in value.items() if k != "scores"} for name, value in spec["methods"].items()},
                                 "evaluator": {"kind": "human_blind", "rubric_version": RUBRIC_VERSION, "aggregation": "single_reviewer"}}))
    spec["frozen_protocol"] = str(frozen)


def edit_run(spec, edit):
    path = Path(spec["methods"]["candidate"]["run"])
    payload = json.loads(path.read_text())
    edit(payload)
    path.write_text(json.dumps(payload))


def test_dev_gate_never_becomes_heldout_claim(tmp_path):
    result = evaluate(setup_methods(tmp_path))["comparisons"]["candidate"]
    assert result["dev_quality_eligible"]
    assert result["all_dimensions_noninferior_on_evaluator"]
    assert not result["quality_noninferiority_established"]


def test_missing_dimension_stays_missing_without_pairwise_deletion(tmp_path):
    spec = setup_methods(tmp_path)
    edit_scores(spec, "candidate", lambda rows: rows[0]["scores"].update(citations=None))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert result["dimensions"]["citations"]["candidate_missing"] == ["s0:1"]
    assert result["dimensions"]["correctness"]["status"] == "complete"
    assert not result["dev_quality_eligible"]


def test_declared_task_dimensions_do_not_require_unrelated_citation_scores(tmp_path):
    spec = setup_methods(tmp_path)
    spec["dimensions"] = ["correctness", "groundedness", "conditions_versions"]
    spec["task_scope"] = {"conditions_versions": ["s1:1", "s2:1"]}
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r["scores"].update(
            citations=None, conditions_versions=.8 if r["task_id"] in {"s1:1", "s2:1"} else None) for r in rows])
        edit_scores(spec, method, lambda rows: [r["scores"].pop("answerability") for r in rows])
    result = evaluate(spec)
    assert result["dimensions"] == spec["dimensions"]
    assert result["task_scope"]["correctness"] == ["s0:1", "s1:1", "s2:1"]
    assert result["quality_claim_scope"] == "declared_dimensions_and_tasks_only"
    comparison = result["comparisons"]["candidate"]
    assert comparison["dev_quality_eligible"]
    assert comparison["dimensions"]["conditions_versions"]["expected_tasks"] == 2
    assert "citations" not in comparison["dimensions"]
    assert result["methods"]["candidate"]["all_score_coverage"]["citations"]["missing_task_ids"] == ["s0:1", "s1:1", "s2:1"]


def test_missing_score_on_an_applicable_task_remains_missing(tmp_path):
    spec = setup_methods(tmp_path)
    spec.update(dimensions=["correctness", "conditions_versions"], task_scope={"conditions_versions": ["s1:1"]})
    edit_scores(spec, "candidate", lambda rows: rows[1]["scores"].update(conditions_versions=None))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert result["dimensions"]["conditions_versions"]["candidate_missing"] == ["s1:1"]
    assert not result["dev_quality_eligible"]


def test_unrelated_tasks_need_no_scores_for_a_declared_subset_comparison(tmp_path):
    spec = setup_methods(tmp_path)
    spec.update(dimensions=["conditions_versions"], task_scope={"conditions_versions": ["s1:1", "s2:1"]})
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: rows.pop(0))
    result = evaluate(spec)
    assert result["comparisons"]["candidate"]["dev_quality_eligible"]
    assert result["methods"]["candidate"]["expected_tasks"] == 3
    assert result["methods"]["candidate"]["coverage"]["conditions_versions"]["expected"] == 2
    assert result["task_scope"] == spec["task_scope"]


def test_applicability_is_shared_and_does_not_drop_a_low_score(tmp_path):
    spec = setup_methods(tmp_path)
    spec.update(dimensions=["correctness", "conditions_versions"], task_scope={"conditions_versions": ["s1:1"]})
    edit_scores(spec, "candidate", lambda rows: rows[1]["scores"].update(conditions_versions=0))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert result["dimensions"]["conditions_versions"]["difference"] == pytest.approx(-.8)
    assert not result["dev_quality_eligible"]


def test_heldout_applicability_is_frozen_with_dimensions_and_evaluator(tmp_path):
    spec = setup_methods(tmp_path, heldout=True)
    spec.update(scope="heldout", dimensions=["correctness", "conditions_versions"],
                task_scope={"conditions_versions": ["s1:1", "s2:1"]})
    freeze_protocol(spec, tmp_path)
    assert evaluate(spec)["comparisons"]["candidate"]["quality_noninferiority_established"]
    spec["task_scope"]["conditions_versions"] = ["s2:1"]
    with pytest.raises(ValueError, match="task applicability"):
        evaluate(spec)


@pytest.mark.parametrize("selection", [
    {"dimensions": []}, {"dimensions": ["correctness", "correctness"]},
    {"task_scope": {"unknown": ["s0:1"]}}, {"task_scope": {"correctness": ["absent"]}},
    {"task_scope": {"correctness": []}},
])
def test_task_protocol_identifies_real_dimensions_and_tasks(tmp_path, selection):
    spec = setup_methods(tmp_path) | selection
    with pytest.raises(ValueError, match="dimensions|task_scope"):
        evaluate(spec)


@pytest.mark.parametrize("method", ["candidate", "reference"])
def test_reject_wrong_run_answer_on_either_side(tmp_path, method):
    spec = setup_methods(tmp_path)
    edit_scores(spec, method, lambda rows: rows[0].update(answer="other answer to same task"))
    with pytest.raises(ValueError, match="different answer"):
        evaluate(spec)


def test_missing_answer_and_nan_are_rejected(tmp_path):
    spec = setup_methods(tmp_path)
    edit_scores(spec, "reference", lambda rows: rows[0].pop("answer"))
    with pytest.raises(ValueError, match="retain the evaluated answer"):
        evaluate(spec)
    spec = setup_methods(tmp_path)
    edit_scores(spec, "reference", lambda rows: rows[0]["scores"].update(correctness=float("nan")))
    with pytest.raises(ValueError, match="finite"):
        evaluate(spec)


def test_local_generator_self_judging_cannot_pass_acceptance(tmp_path):
    from dataclasses import asdict
    spec = setup_methods(tmp_path, source="offline_judge")
    # Same weights under a local path alias are still self-judging.
    judge = asdict(JudgeConfig("local", "/models/Generator", "weight-1", "http://localhost/v1"))
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(judge=judge) for r in rows])
    result = evaluate(spec)["comparisons"]["candidate"]
    assert result["all_dimensions_noninferior_on_evaluator"]
    assert "generator_self_judging" in result["acceptance_blockers"]
    assert not result["dev_quality_eligible"]


def test_human_rater_protocol_must_match_and_simulation_is_blocked(tmp_path):
    spec = setup_methods(tmp_path)
    edit_scores(spec, "candidate", lambda rows: rows[0].update(reviewer_id="other-rater"))
    assert not evaluate(spec)["comparisons"]["candidate"]["dev_quality_eligible"]
    result = evaluate(setup_methods(tmp_path, simulation=True))["comparisons"]["candidate"]
    assert result["dimensions"]["correctness"]["status"] == "not_validated_simulation"


def test_heldout_requires_matching_declared_frozen_protocol(tmp_path):
    spec = setup_methods(tmp_path, heldout=True)
    spec["scope"] = "heldout"
    assert not evaluate(spec)["comparisons"]["candidate"]["quality_noninferiority_established"]
    freeze_protocol(spec, tmp_path)
    assert evaluate(spec)["comparisons"]["candidate"]["quality_noninferiority_established"]
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(rubric_version="changed-rubric") for r in rows])
    assert not evaluate(spec)["comparisons"]["candidate"]["quality_noninferiority_established"]
    spec["epsilon"] = .05
    with pytest.raises(ValueError, match="frozen quality protocol"):
        evaluate(spec)


def test_dev_cannot_be_redeclared_heldout_and_missing_split_never_establishes(tmp_path):
    spec = setup_methods(tmp_path)
    spec["scope"] = "heldout"
    with pytest.raises(ValueError, match="dev cannot be relabeled heldout"):
        evaluate(spec)
    spec.pop("split_manifest")
    edit_run(spec, lambda value: value.pop("workload"))
    reference = Path(spec["methods"]["reference"]["run"])
    payload = json.loads(reference.read_text())
    payload.pop("workload")
    reference.write_text(json.dumps(payload))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert "heldout_split_unverified" in result["acceptance_blockers"]
    assert not result["quality_noninferiority_established"]


@pytest.mark.parametrize("location", ["payload", "result", "score"])
def test_original_simulation_flags_cannot_be_hidden_by_hardware_audit(tmp_path, location):
    spec = setup_methods(tmp_path, heldout=True)
    spec["scope"] = "heldout"
    freeze_protocol(spec, tmp_path)
    if location == "payload":
        edit_run(spec, lambda value: value.update(measurement_kind="simulation"))
    elif location == "result":
        edit_run(spec, lambda value: value["report"]["turns"][0]["result"].update(measurement_kind="fake_http_correctness"))
    else:
        edit_scores(spec, "candidate", lambda rows: rows[0].update(generation_measurement_kind="simulation"))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert "simulation_answers" in result["acceptance_blockers"]
    assert not result["quality_noninferiority_established"]


@pytest.mark.parametrize("field", ["workload"])
def test_original_split_declaration_matches_cohort(tmp_path, field):
    spec = setup_methods(tmp_path, heldout=True)
    spec["scope"] = "heldout"
    freeze_protocol(spec, tmp_path)
    if field == "workload":
        edit_run(spec, lambda value: value["workload"].update(split="dev"))
    with pytest.raises(ValueError, match="original run"):
        evaluate(spec)


@pytest.mark.parametrize("field,value", [("query", "Different question"), ("history", [{"speaker": "user", "text": "Different history"}]),
                                        ("contexts", [{"text": "Other evidence"}])])
def test_same_answer_cannot_hide_changed_original_question_history_or_evidence(tmp_path, field, value):
    spec = setup_methods(tmp_path)
    edit_run(spec, lambda raw: raw["report"]["turns"][0]["online_turn"].update({field: value}))
    with pytest.raises(ValueError, match="original online turn"):
        evaluate(spec)


def expanded_material_method(spec, tmp_path, *, native=True):
    """Exercise the real exporter with a ranking snippet and restored source."""
    from dataclasses import asdict
    from controller.schema import OnlineContext, OnlineTurn
    from adapters.retrieval import to_evidence
    from bench.quality import export_evaluation

    path = Path(spec["methods"]["candidate"]["audit"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    original = OnlineContext("source", "retrieved snippet")
    restored = OnlineContext("source", "Complete source including its conditions.")
    supporting = OnlineContext("dependency", "Supporting definition.")
    turn = OnlineTurn("s0", "s0:1", 1, "fixture", "Question?", (), (restored,), "P3",
                      supporting_contexts=(supporting,), retrieval_contexts=(original,))
    bodies = [asdict(to_evidence(c, "fixture")) for c in (restored, supporting)]
    manifest = [{"reference_id": f"E_{i}", "key": block["key"]} for i, block in enumerate(bodies)]
    exported = tmp_path / "expanded-audit.jsonl"
    export_evaluation(turn, rows[0]["answer"], official_path=tmp_path / "expanded-official.jsonl",
                      audit_path=exported, visible_evidence_manifest=manifest, audit_evidence=bodies,
                      input_ids=rows[0]["input_ids"], output_ids=rows[0]["output_ids"],
                      measurement_kind="hardware", citation_scope=("E_0", "E_1"))
    rows[0] = json.loads(exported.read_text())
    write_jsonl(path, rows)

    def update(raw):
        saved = raw["report"]["turns"][0]
        saved["online_turn"] = asdict(turn)
        saved["result"].update(visible_evidence_manifest=manifest, citation_scope=["E_0", "E_1"])
        if native:
            saved["result"]["audit_evidence"] = bodies
    edit_run(spec, update)
    return path


@pytest.mark.parametrize("native", [True, False])
def test_original_ranking_and_restored_reading_material_bind_separately(tmp_path, native):
    spec = setup_methods(tmp_path)
    expanded_material_method(spec, tmp_path, native=native)
    assert evaluate(spec)["comparisons"]["candidate"]["dev_quality_eligible"]


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("tamper", ["retrieved_evidence", "audit_evidence", "visible_evidence_manifest", "citation_scope"])
def test_restored_material_keeps_both_ranking_and_actual_citations_bound(tmp_path, native, tamper):
    spec = setup_methods(tmp_path)
    path = expanded_material_method(spec, tmp_path, native=native)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if tamper == "retrieved_evidence":
        rows[0][tamper][0]["text"] = "a different ranking"
    elif tamper == "audit_evidence":
        rows[0][tamper][0]["text"] = "a different source body"
    else:
        rows[0][tamper].pop()
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="original"):
        evaluate(spec)


@pytest.mark.parametrize("field", ["audit_evidence", "visible_evidence_manifest", "citation_scope"])
def test_original_material_fields_cannot_be_omitted_from_audit(tmp_path, field):
    spec = setup_methods(tmp_path)
    path = expanded_material_method(spec, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].pop(field)
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="original run result"):
        evaluate(spec)


@pytest.mark.parametrize("tamper", ["missing_body", "changed_body"])
def test_equal_native_result_and_audit_copies_must_still_match_visible_material(tmp_path, tamper):
    spec = setup_methods(tmp_path)
    path = expanded_material_method(spec, tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    bodies = rows[0]["audit_evidence"]
    if tamper == "missing_body":
        bodies.pop()
    else:
        bodies[0]["text"] = "Both saved copies contain the same wrong body."
    write_jsonl(path, rows)
    edit_run(spec, lambda raw: raw["report"]["turns"][0]["result"].update(audit_evidence=bodies))
    with pytest.raises(ValueError, match="original visible"):
        evaluate(spec)


@pytest.mark.parametrize("native_ids", [True, False])
def test_audit_ids_match_native_ids_or_controller_receipt(tmp_path, native_ids):
    spec = setup_methods(tmp_path)
    def change(raw):
        result = raw["report"]["turns"][0]["result"]
        if native_ids:
            result["input_ids"] = [999]
        else:
            result.pop("input_ids")
            result.pop("input_hash")
            result["receipt"] = {"input_hash": token_hash([999])}
    edit_run(spec, change)
    with pytest.raises(ValueError, match="audit input_ids"):
        evaluate(spec)


def test_legacy_missing_original_scope_or_score_input_is_diagnostic_only(tmp_path):
    spec = setup_methods(tmp_path, heldout=True)
    spec["scope"] = "heldout"
    freeze_protocol(spec, tmp_path)
    edit_scores(spec, "candidate", lambda rows: rows[0].pop("input_hash"))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert set(result["acceptance_blockers"]) >= {"score_input_binding_missing"}
    assert not result["quality_noninferiority_established"]


def test_scores_cannot_move_between_prompts_with_same_answer(tmp_path):
    spec = setup_methods(tmp_path)
    edit_scores(spec, "candidate", lambda rows: rows[0].update(input_hash=token_hash([999])))
    with pytest.raises(ValueError, match="score input_hash"):
        evaluate(spec)


def test_controller_sidecar_reuses_original_split_and_receipt(tmp_path):
    spec = setup_methods(tmp_path, heldout=True)
    spec["scope"] = "heldout"
    for method in spec["methods"].values():
        path = Path(method["run"])
        raw = json.loads(path.read_text())
        raw.pop("workload")
        for turn in raw["report"]["turns"]:
            turn["result"].pop("input_ids")
            turn["result"]["receipt"] = {"input_hash": turn["result"].pop("input_hash")}
        path.write_text(json.dumps(raw))
        sidecar = path.with_suffix(".manifest.json")
        sidecar.write_text(json.dumps({"dataset": {"split": "test"}}))
        method["manifest"] = str(sidecar)
    freeze_protocol(spec, tmp_path)
    assert evaluate(spec)["comparisons"]["candidate"]["quality_noninferiority_established"]


def test_missing_legacy_run_prompt_stays_diagnostic_instead_of_fabricated(tmp_path):
    spec = setup_methods(tmp_path, heldout=True)
    spec["scope"] = "heldout"
    freeze_protocol(spec, tmp_path)
    edit_run(spec, lambda raw: raw["report"]["turns"][0].pop("online_turn"))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert "audit_input_binding_missing" in result["acceptance_blockers"]
    assert not result["quality_noninferiority_established"]


def test_blind_import_and_aggregate_retain_input_binding(tmp_path):
    import csv
    record = {"task_id": "t", "conversation_id": "s", "query": "q", "answer": "a", "input_ids": [10, 20],
              "actual_history": [], "retrieved_evidence": [], "visible_evidence_manifest": [], "audit_evidence": []}
    export_blind({"A": [record]}, tmp_path)
    path = tmp_path / "ratings.csv"
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(reviewer_id="unit-test-reviewer", script_history_mismatch="false", **{k: ".5" for k in DIMENSIONS})
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = aggregate_ratings(import_ratings(tmp_path / "private-key.json", path))[0]
    assert result["query"] == "q" and result["input_hash"] == token_hash([10, 20])


def test_blind_export_preserves_completed_files_and_has_no_offline_labels(tmp_path):
    row = {"task_id": "task", "conversation_id": "session", "answer": "answer", "query": "question",
           "actual_history": [], "retrieved_evidence": [], "visible_evidence_manifest": [], "audit_evidence": [],
           "targets": ["secret gold"], "enrichments": {"label": "secret gold"}}
    export_blind({"secret_A": [row], "secret_B": [row]}, tmp_path)
    public = (tmp_path / "review.jsonl").read_text()
    assert "secret" not in public
    ratings = tmp_path / "ratings.csv"
    ratings.write_text("completed ratings")
    with pytest.raises(FileExistsError):
        export_blind({"secret_A": [row], "secret_B": [row]}, tmp_path)
    assert ratings.read_text() == "completed ratings"


def test_human_aggregation_is_explicit_equal_rater_and_retains_na_and_disagreement():
    base = {"method": "A", "task_id": "task", "conversation_id": "s", "sample_id": "sample-1", "answer": "answer",
            "source": "human_blind", "script_history_mismatch": False, "comment": "review"}
    first = base | {"reviewer_id": "r1", "scores": {d: .5 for d in DIMENSIONS}}
    second = base | {"reviewer_id": "r2", "scores": {d: 1.0 for d in DIMENSIONS}}
    second["scores"]["citations"] = None
    result = aggregate_ratings([first, second])[0]
    assert result["scores"]["correctness"] == .75 and result["scores"]["citations"] is None
    assert result["score_ranges"]["correctness"] == [.5, 1.0]
    assert result["reviewer_ids"] == ["r1", "r2"] and len(result["reviews"]) == 2
    with pytest.raises(ValueError, match="duplicate reviewer"):
        aggregate_ratings([first, first])
    with pytest.raises(ValueError, match="human blind"):
        aggregate_ratings([base | {"source": "offline_judge"}])


def test_native_rubric_and_parser_changes_are_part_of_evaluator_identity(tmp_path):
    from dataclasses import asdict
    spec = setup_methods(tmp_path, source="offline_judge")
    judge = asdict(JudgeConfig("local", "Judge", "judge-weights", "http://localhost/v1")) | {
        "protocol": "selene_absolute_v1", "base_family": "llama-3.1",
        "protocol_manifest": {"parser": "result-v1", "rubrics": {"correctness": "original"}},
    }
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(judge=judge) for r in rows])
    assert evaluate(spec)["comparisons"]["candidate"]["dev_quality_eligible"]
    edit_scores(spec, "candidate", lambda rows: rows[0]["judge"]["protocol_manifest"]["rubrics"].update(correctness="changed"))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert "missing_or_unmatched_evaluator_protocol" in result["acceptance_blockers"]


def test_current_task_requirements_must_match_for_the_paired_answers(tmp_path):
    from dataclasses import asdict
    spec = setup_methods(tmp_path, source="offline_judge")
    judge = asdict(JudgeConfig("local", "Judge", "judge-weights", "http://localhost/v1"))
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(judge=judge,
            task_requirements={"requirement":r["task_id"]+" current task", "dimensions":["correctness"]}) for r in rows])
    assert evaluate(spec)["comparisons"]["candidate"]["dev_quality_eligible"]
    edit_scores(spec, "candidate", lambda rows: rows[0]["task_requirements"].update(requirement="A future task"))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert "missing_or_unmatched_evaluator_protocol" in result["acceptance_blockers"]


def test_native_frozen_protocol_cannot_omit_rubric_or_na_extension():
    from dataclasses import asdict
    from bench.quality_acceptance import frozen_evaluator_matches
    judge = asdict(JudgeConfig("local", "Judge", "judge-weights", "http://localhost/v1"))
    row = {"source": "offline_judge", "judge": judge | {"na_policy": "explicit_na_v1",
           "protocol_manifest": {"rubric": "v1"}}, "reference_targets_supplied": False}
    frozen = {"kind": "offline_judge", "judge": judge, "reference_targets_supplied": False}
    assert not frozen_evaluator_matches(frozen, row)
    frozen["judge"] = dict(row["judge"])
    assert frozen_evaluator_matches(frozen, row)
    row["judge"]["na_policy"] = "native_numeric_only"
    assert not frozen_evaluator_matches(frozen, row)


@pytest.mark.parametrize("field,value", [("enable_thinking", True), ("output_constraint", "native_regex_v1"),
                                        ("context_window", 2048), ("timeout_seconds", 5),
                                        ("future_scoring_option", {"mode": "changed"})])
def test_every_recorded_scoring_control_binds_paired_and_frozen_evaluators(tmp_path, field, value):
    from dataclasses import asdict
    from bench.quality_acceptance import frozen_evaluator_matches

    spec = setup_methods(tmp_path, source="offline_judge")
    judge = asdict(JudgeConfig("local", "Judge", "judge-weights", "http://localhost/v1"))
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(judge=judge, reference_targets_supplied=False) for r in rows])
    frozen = {"kind": "offline_judge", "judge": judge, "reference_targets_supplied": False}
    row = {"source": "offline_judge", "judge": judge, "reference_targets_supplied": False}
    assert frozen_evaluator_matches(frozen, row)
    assert evaluate(spec)["comparisons"]["candidate"]["dev_quality_eligible"]
    edit_scores(spec, "candidate", lambda rows: rows[0]["judge"].update({field: value}))
    blockers = evaluate(spec)["comparisons"]["candidate"]["acceptance_blockers"]
    assert "missing_or_unmatched_evaluator_protocol" in blockers
    assert not frozen_evaluator_matches(frozen, row | {"judge": judge | {field: value}})


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("changed_model", ["different-served-model", "Generator"])
def test_response_model_drift_and_self_judging_cannot_hide_behind_a_configured_alias(tmp_path, native, changed_model):
    from dataclasses import asdict

    spec = setup_methods(tmp_path, source="offline_judge")
    judge = asdict(JudgeConfig("local", "judge-alias", "judge-weights", "http://localhost/v1"))

    def response_identity(model):
        return ({"native_results": {dimension: {"served_model": model} for dimension in DIMENSIONS}}
                if native else {"served_model": model})

    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(judge=judge, **response_identity("actual-judge")) for r in rows])
    assert evaluate(spec)["comparisons"]["candidate"]["dev_quality_eligible"]
    edit_scores(spec, "candidate", lambda rows: rows[0].update(**response_identity(changed_model)))
    result = evaluate(spec)["comparisons"]["candidate"]
    assert not result["dev_quality_eligible"]
    assert "missing_or_unmatched_evaluator_protocol" in result["acceptance_blockers"]
    if changed_model == "Generator":
        assert "generator_self_judging" in result["acceptance_blockers"]


def test_native_dimensions_cannot_share_a_single_identity_when_models_differ(tmp_path):
    from dataclasses import asdict

    spec = setup_methods(tmp_path, source="offline_judge")
    judge = asdict(JudgeConfig("local", "judge-alias", "judge-weights", "http://localhost/v1"))
    grades = {dimension: {"served_model": "first" if dimension == "correctness" else "second"}
              for dimension in DIMENSIONS}
    for method in spec["methods"]:
        edit_scores(spec, method, lambda rows: [r.update(judge=judge, native_results=grades) for r in rows])
    result = evaluate(spec)["comparisons"]["candidate"]
    assert not result["dev_quality_eligible"]
    assert {"unknown_evaluator_identity", "missing_or_unmatched_evaluator_protocol"} <= set(result["acceptance_blockers"])


def test_frozen_protocol_checks_recorded_response_identity_and_explicit_alias():
    from dataclasses import asdict
    from bench.quality_acceptance import frozen_evaluator_matches

    judge = asdict(JudgeConfig("local", "judge-alias", "judge-weights", "http://localhost/v1"))
    frozen = {"kind": "offline_judge", "judge": judge, "reference_targets_supplied": False}
    row = {"source": "offline_judge", "judge": judge, "reference_targets_supplied": False}
    assert frozen_evaluator_matches(frozen, row)  # Legacy missing response identity remains missing.
    assert frozen_evaluator_matches(frozen, row | {"served_model": "judge-alias"})
    assert not frozen_evaluator_matches(frozen, row | {"served_model": "actual-judge"})
    frozen["served_model"] = "actual-judge"
    assert frozen_evaluator_matches(frozen, row | {"served_model": "actual-judge"})
    assert not frozen_evaluator_matches(frozen, row)
    assert not frozen_evaluator_matches(frozen, row | {"native_results": {
        "correctness": {"served_model": "actual-judge"}, "groundedness": {"served_model": "different"}}})


def test_related_family_is_explicit_without_becoming_self_or_human_judging():
    from dataclasses import asdict
    from bench.quality_acceptance import evaluator_identity
    judge = asdict(JudgeConfig("local", "OtherTunedModel", "different-weights", "http://localhost/v1"))
    row = {"source": "offline_judge", "judge": judge | {"base_family": "qwen2.5"}}
    generator = {"model": "QwenGenerator", "revision": "generator-weights", "base_family": "qwen2.5"}
    assert evaluator_identity(row, generator)[0] == "related_model_family_declared"
    row["judge"]["base_family"] = "llama-3.1"
    assert evaluator_identity(row, generator)[0] == "independent_model_declared"
    row["judge"]["revision"] = generator["revision"]
    assert evaluator_identity(row, generator)[0] == "self_judge"
