"""Operational no-QA failure utility uses actual terminal branch evidence.

CPU protocol fixtures and specified judge values test contracts, not GPU gains.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from training.plan_gain_completion import completion_record, effective_pair, finalize, fit_effective, label_scorer_identity
from training.plan_gain_rollout import execute_branch
from training.train_plan_gain import COST_FIELDS
from training.train_plan_gain import PlanGainModel
from action_helpers import Resources, frozen_fixture


async def measured_fixture(tmp_path, *, direct_terminal_failure=False):
    service, native, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path/"capture")
    collection = tmp_path/"collection"
    values = {}
    original_generate = native.generate
    native.fail_read = False

    async def faulted_generate(**kwargs):
        result = await original_generate(**kwargs)
        if native.fail_read and kwargs.get("response_schema") is not None:
            return replace(result, finish_reason="length", receipt=replace(result.receipt, finish_reason="length"))
        return result

    native.generate = faulted_generate
    if direct_terminal_failure:
        from controller.execution import ReadingBudgetExceeded
        validate = service._validate

        def exhausted_after_generation(attempt, result):
            validate(attempt, result)
            service.store.fail_attempt(attempt.attempt_id, "specified confirmed terminal failure",
                                       force_recovery=True)
            raise ReadingBudgetExceeded("specified terminal failure after the recorded generation")

        service._validate = exhausted_after_generation
    for number, kind in enumerate(("direct_answer", "read_once")):
        native.fail_read = kind == "read_once"
        plan = next(plan for plan in plans if plan.semantic_kind == kind)
        path = collection/"tasks"/"task"/f"branch-{number:02d}"
        run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=path,
            binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
            generation_budget=budget, observed=observed, gap_request=request)
        if run["qa_increment"] == 0:
            assert run["status"] == "failed" and run["qa_increment"] == 0
            run["termination"] = {"terminal": True, "user_paused": False, "native_calls_finished": True,
                "reason": run["failure"]["type"], "checked_at": run["actual_branch_execution_cost"]["window"]["end"]}
        else:
            run["metrics"] = {"plan_gain_quality": .6 if kind == "direct_answer" else .7}
            run["quality_components"] = {"necessary_points": .6 if kind == "direct_answer" else .7}
            run["scoring"] = {"status": "scored", "identity": binding["scorer_identity"]}
        (path/"run.json").write_text(json.dumps(run))
        before = json.loads((path/"before.json").read_text())
        values[kind] = (before, run)
    await service.aclose()
    service.store.close()
    return collection, frozen, values


@pytest.mark.asyncio
async def test_failed_read_lowers_utility_keeps_raw_q_missing_and_paid_signed_costs(tmp_path):
    collection, _, values = await measured_fixture(tmp_path)
    direct = completion_record(*values["direct_answer"])
    failure = completion_record(*values["read_once"])
    assert failure["raw_judge_Q"] is None and failure["effective_utility"] == 0
    assert failure["completion_reason"] == "confirmed_terminal_no_QA"
    pair = effective_pair(direct, failure)
    assert pair["effective_delta_q"] == -.6 and pair["raw_Q_counterfactual"] is None
    raw_direct, raw_failure = values["direct_answer"][1], values["read_once"][1]
    assert raw_failure["actual_native_calls"][0]["receipt"]["generation_complete"]
    assert raw_failure["actual_branch_execution_cost"]["total_input_tokens"] > 0
    for name in COST_FIELDS:
        assert pair["cost_targets"]["branch_complete_cost"][name] == raw_failure["actual_branch_execution_cost"][name]
        assert pair["incremental_complete_cost"][name] == pytest.approx(
            raw_failure["actual_branch_execution_cost"][name]-raw_direct["actual_branch_execution_cost"][name])
    result = finalize(collection, tmp_path/"labels")
    assert result["effective_gain_records"] == 1 and result["cost_labelled_actions"] == 1
    assert result["raw_quality_missing"] == 1

    # A new offline label uses the same executed answer/input and cost. The
    # registered scorer remains the original declaration, not the new rubric.
    registration_path = collection/"tasks/task/branch-00/before.json"
    run_path = registration_path.with_name("run.json")
    original_files = (registration_path.read_bytes(), run_path.read_bytes())
    before, rescored_run = deepcopy(values["direct_answer"])
    new_scorer = {"kind": "specified independent rescore fixture"}
    rescored_run["metrics"]["plan_gain_quality"] = .8
    rescored_run["quality_components"]["necessary_points"] = .8
    rescored_run["scoring"] = {"status": "scored", "identity": new_scorer}
    rescored = completion_record(before, rescored_run)
    assert rescored["scorer_identity"] == new_scorer
    assert rescored["registered_scorer_identity"] == before["scorer_identity"] == raw_direct["scorer_identity"]
    assert rescored["pre_registration"] == values["direct_answer"][0]
    assert rescored["actual_run"]["actual_native_calls"] == raw_direct["actual_native_calls"]
    assert rescored["actual_run"]["actual_branch_execution_cost"] == raw_direct["actual_branch_execution_cost"]
    assert rescored["record_id"] != direct["record_id"]
    assert failure["scorer_identity"] is None  # No-QA utility zero did not call a judge.
    rescored_pair = effective_pair(rescored, failure)
    assert rescored_pair["scorer_identity"] == new_scorer
    assert rescored_pair["effective_delta_q"] == -.8
    assert rescored_pair["cost_targets"] == pair["cost_targets"]
    assert fit_effective([rescored_pair]).provenance["scorer_identity"] == new_scorer
    _, _, failed_values = await measured_fixture(tmp_path/"both-terminal", direct_terminal_failure=True)
    no_qa_pair = effective_pair(*(completion_record(*failed_values[kind])
                                 for kind in ("direct_answer", "read_once")))
    assert no_qa_pair["scorer_identity"] is None and no_qa_pair["effective_delta_q"] == 0
    assert label_scorer_identity([no_qa_pair]) is None
    assert label_scorer_identity([no_qa_pair, rescored_pair]) == new_scorer
    mixed = fit_effective([no_qa_pair, rescored_pair])
    assert mixed.provenance["scorer_identity"] == new_scorer
    assert mixed.provenance["training_records"] == 2
    assert mixed.cost_provenance["training_samples"] == 4
    from training.collect_plan_gains import served_branch
    adapted = served_branch(rescored_run, semantic_kind="direct_answer", plan_id=rescored_run["plan_id"],
        metric="plan_gain_quality", normalization={"scale": 1, "offset": 0, "protocol_source": "fixture"},
        scorer_identity=new_scorer, comparison_budget=rescored_run["comparison_budget"])
    assert adapted["scorer_identity"] == new_scorer
    assert adapted["registered_scorer_identity"] == before["scorer_identity"]
    changed_budget = deepcopy(rescored_run)
    changed_budget["comparison_budget"]["answer_output_tokens"] += 1
    with pytest.raises(ValueError, match="budget/declared scorer"):
        completion_record(before, changed_budget)
    assert (registration_path.read_bytes(), run_path.read_bytes()) == original_files

    assert raw_direct["response"]["read_count"] == 1
    assert raw_direct["new_action_counts"]["READ"] == 0
    before, paused = deepcopy(values["read_once"])
    paused["termination"]["user_paused"] = True
    assert completion_record(before, paused)["effective_utility"] is None
    before, unknown = deepcopy(values["read_once"])
    unknown["actual_branch_execution_cost"]["host_gib_seconds"] = None
    unknown["cost"]["host_gib_seconds"] = None
    uncertain = effective_pair(direct, completion_record(before, unknown))
    assert uncertain["cost_targets"]["branch_complete_cost"]["host_gib_seconds"] is None
    assert uncertain["effective_delta_q"] == -.6


@pytest.mark.asyncio
async def test_whole_plan_completion_asset_round_trips_without_observed_features(tmp_path):
    _, _, values = await measured_fixture(tmp_path)
    branches = {}
    for kind, (before, run) in deepcopy(values).items():
        before["observed_state_binding"] = run["observed_state_binding"] = None
        if kind == "read_once":
            before["plan_candidate"]["semantic_kind"] = "complementary_evidence"
            run["semantic_kind"] = "complementary_evidence"
        branches[kind] = completion_record(before, run)
    pair = effective_pair(branches["direct_answer"], branches["read_once"])
    model = fit_effective([pair])
    path = tmp_path/"whole-plan.json"
    model.save(path)
    loaded = PlanGainModel.load(path, reader_checkpoint=model.reader_checkpoint,
                               reader_profile_id=model.reader_profile_id)
    assert loaded.provenance["observed_feature_schema"] is None
    assert loaded.predict(pair["features"]) == pytest.approx(model.predict(pair["features"]))


@pytest.mark.asyncio
async def test_effective_assets_reject_missing_or_changed_compiler_reader_scope(tmp_path):
    from controller.templates import ByteTemplate

    _, _, values = await measured_fixture(tmp_path)
    pair = effective_pair(*(completion_record(*values[kind]) for kind in ("direct_answer", "read_once")))
    model = fit_effective([pair])
    path = tmp_path/"current.json"
    model.save(path)
    fingerprint = ByteTemplate().fingerprint
    with pytest.raises(ValueError, match="input protocol differs"):
        PlanGainModel.load(path, reader_checkpoint=model.reader_checkpoint,
            reader_profile_id=model.reader_profile_id,
            reader_fingerprint=replace(fingerprint, template_hash="different-template"))
    saved = json.loads(path.read_text())
    del saved["provenance"]["input_protocol"]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="current actual input protocol"):
        PlanGainModel.load(path, reader_checkpoint=model.reader_checkpoint,
            reader_profile_id=model.reader_profile_id)


@pytest.mark.asyncio
async def test_pre_action_cost_quote_must_bind_the_executed_compiler_choice(tmp_path):
    _, _, values = await measured_fixture(tmp_path)
    before, run = deepcopy(values["direct_answer"])
    run["pre_action_input"]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="first executed native input"):
        completion_record(before, run)
