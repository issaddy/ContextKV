"""Operational no-QA failure utility uses actual terminal branch evidence.

CPU protocol fixtures and specified judge values test contracts, not GPU gains.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from training.plan_gain_completion import completion_record, effective_pair, finalize, fit_effective
from training.plan_gain_rollout import execute_branch
from training.train_plan_gain import COST_FIELDS
from training.train_plan_gain import PlanGainModel
from action_helpers import Resources, frozen_fixture


async def measured_fixture(tmp_path):
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
    for number, kind in enumerate(("direct_answer", "read_once")):
        native.fail_read = kind == "read_once"
        plan = next(plan for plan in plans if plan.semantic_kind == kind)
        path = collection/"tasks"/"task"/f"branch-{number:02d}"
        run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=path,
            binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
            generation_budget=budget, observed=observed, gap_request=request)
        if kind == "read_once":
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
