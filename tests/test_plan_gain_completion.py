"""Operational no-QA failure utility uses actual terminal branch evidence.

CPU protocol fixtures and specified judge values test contracts, not GPU gains.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from training.plan_gain_completion import (completion_record, effective_pair, finalize, fit_effective,
    fitted_action_domain, label_scorer_identity, refresh_features)
from training.plan_gain_rollout import execute_branch
from training.train_plan_gain import COST_FIELDS
from training.train_plan_gain import PlanGainModel
from action_helpers import Resources, frozen_fixture


async def measured_fixture(tmp_path, *, direct_terminal_failure=False, read_failure=True):
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
        native.fail_read = read_failure and kind == "read_once"
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
async def test_first_read_quotes_remaining_route_without_executing_every_package(tmp_path):
    from controller.context_schema import SourceSpan
    from controller.evidence import make_evidence
    from training.plan_gain_protocol import build_observed_action_plans
    from training.train_plan_gain import working_state_key

    service, _, budget, state, snapshot, observed, request, _, binding, row, prep = await frozen_fixture(tmp_path/"capture")
    original = state.plan.selected_set.selected_units[-1]
    block = make_evidence(original.text.replace("R2", "R3"), "third-source")
    unit = replace(original, unit_id="R3", text=block.text,
        source_spans=(SourceSpan("R3", block.key, 0, len(block.text)),))
    task = replace(state.plan.steps[-1], step_id="read:third", input_unit_ids=(unit.unit_id,), input_budget=12000)
    material = replace(state.plan.selected_set,
        selected_units=(*state.plan.selected_set.selected_units, unit),
        source_blocks=(*state.plan.selected_set.source_blocks, block))
    state = replace(state, plan=replace(state.plan, selected_set=material,
        steps=(state.plan.steps[0], replace(state.plan.steps[-1], input_budget=12000), task)))
    observed = {**observed, "state_key": working_state_key(state, snapshot)}
    plans, _ = build_observed_action_plans(state, snapshot, service.template,
        reader_checkpoint=binding["reader_checkpoint"], gap_request=request)
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    path = tmp_path/"branch"
    run = await execute_branch(service, plan, state, snapshot, row, case_dir=path,
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request)
    route = run["compiled_route_steps"]
    assert {step["task"]["step_id"] for step in route if step["task"]["kind"] == "READ"} == {
        state.plan.steps[-2].step_id, task.step_id}
    quotes = {quote["candidate_id"] for quote in run["pre_action_quote"]["batch"]["quotes"]}
    assert all(candidate["candidate_id"] in quotes for step in route for candidate in step["variants"]
               if candidate["feasible"])
    assert run["new_action_counts"]["READ"] == 1
    assert task.step_id not in run["new_step_ids"]
    record = completion_record(json.loads((path/"before.json").read_text()), run)
    future_quote = next(candidate for step in route if step["task"]["step_id"] == task.step_id
                        for candidate in step["variants"])
    assert record["cost_features"]["future_input_tokens"] >= len(future_quote["input_ids"])
    assert record["cost_features"]["future_steps"] >= 2
    before = json.loads((path/"before.json").read_text())
    legacy_before, legacy_run = deepcopy(before), deepcopy(run)
    legacy_before["plan_candidate"]["context_plan"]["steps"] = [item
        for item in legacy_before["plan_candidate"]["context_plan"]["steps"] if item["step_id"] != task.step_id]
    legacy_run.pop("registered_context_plan_id")
    legacy_run.pop("compiled_route_steps")
    legacy_run["pre_action_quote"]["batch"]["quotes"] = [quote
        for quote in legacy_run["pre_action_quote"]["batch"]["quotes"]
        if quote["candidate_id"] == legacy_run["pre_action_input"]["candidate_id"]]
    legacy = completion_record(legacy_before, legacy_run)
    assert legacy["features"] == record["features"]
    assert legacy["effective_utility"] == record["effective_utility"]
    assert legacy["actual_run"]["actual_branch_execution_cost"] == record["actual_run"]["actual_branch_execution_cost"]
    assert legacy["cost_features"]["future_steps"] == record["cost_features"]["future_steps"]
    assert legacy["cost_features"]["future_input_tokens"] < record["cost_features"]["future_input_tokens"]
    assert legacy["cost_features"]["future_prefill_tokens"] == legacy["cost_features"]["future_input_tokens"]
    await service.aclose()
    service.store.close()


def test_domain_migration_requires_measured_changed_grouping_not_a_version_relabel():
    from controller.action_proposals import finite_action_domain
    from controller.context_schema import ReadTask
    from controller.serialization import to_jsonable

    current = finite_action_domain((512, 1024), 64)
    legacy = {**current, "protocol": "two_reads_shared_outputs_terminal_representations_v1"}
    tasks = [to_jsonable(ReadTask("a", "READ", ("n",), ("A",))),
             to_jsonable(ReadTask("b", "READ", ("n",), ("B",)))]
    old = {"observed_state_binding": {"action_domain": legacy}, "semantic_kind": "read_once"}
    new = {"observed_state_binding": {"action_domain": current}, "semantic_kind": "read_once",
        "counterfactual": {"pre_registration": {"working_state": {"results": [], "plan": {"steps": tasks}},
                            "plan_candidate": {"context_plan": {"steps": tasks}}},
            "actual_run": {"compiled_route_steps": [{"task": tasks[0]}], "pre_action_input": {"input_ids": [1]},
                           "new_action_counts": {"READ": 1}}}}
    with pytest.raises(ValueError, match="actually executed READ organization"):
        fitted_action_domain([old, new])
    changed = deepcopy(new)
    changed["counterfactual"]["pre_registration"]["plan_candidate"]["context_plan"]["steps"] = [
        to_jsonable(ReadTask("merged", "READ", ("n",), ("A", "B")))]
    with pytest.raises(ValueError, match="actually executed READ organization"):
        fitted_action_domain([old, changed])  # Changing only an unexecuted directory is not a label.
    changed["counterfactual"]["actual_run"].update(new_step_ids=["merged"], steps=[{"step_id": "merged",
        "task": changed["counterfactual"]["pre_registration"]["plan_candidate"]["context_plan"]["steps"][0]}])
    assert fitted_action_domain([old, changed])[0] == current
    assert len(fitted_action_domain([old, changed])[1]) == 2
    incompatible = deepcopy(old)
    incompatible["observed_state_binding"]["action_domain"]["output_budgets"] = [512]
    with pytest.raises(ValueError, match="incompatible"):
        fitted_action_domain([incompatible, changed])
    missing = deepcopy(changed)
    missing["counterfactual"]["pre_registration"]["plan_candidate"]["context_plan"]["steps"][0]["input_unit_ids"] = ["A", "C"]
    with pytest.raises(ValueError, match="frozen source/Need obligations"):
        fitted_action_domain([old, missing])


@pytest.mark.asyncio
async def test_saved_answer_pair_refresh_reads_its_original_registrations(tmp_path):
    _, _, values = await measured_fixture(tmp_path, read_failure=False)
    branches = {kind: completion_record(*values[kind]) for kind in values}
    expected = effective_pair(branches["direct_answer"], branches["read_once"])
    row = {"source": "actual_answer_counterfactual", "record_id": "saved-fixture-pair",
        "delta_q": expected["effective_delta_q"], "quality_direct": expected["raw_Q_direct"],
        "quality_counterfactual": expected["raw_Q_counterfactual"],
        "cost_targets": expected["cost_targets"], "incremental_complete_cost": expected["incremental_complete_cost"],
        "direct_run": {"source_run": values["direct_answer"][1]},
        "counterfactual_run": {"source_run": values["read_once"][1]}}
    source, target = tmp_path/"saved.jsonl", tmp_path/"refreshed.jsonl"
    source.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="original registrations"):
        refresh_features(source, target)
    refresh_features(source, target, registration_root=tmp_path, recorded_root=tmp_path)
    refreshed = json.loads(target.read_text())
    assert refreshed["effective_delta_q"] == expected["effective_delta_q"]
    assert refreshed["cost_targets"] == expected["cost_targets"]
    assert refreshed["features"] == expected["features"]
    assert refreshed["direct"]["source_files"]["run_record_id"] == "saved-fixture-pair"


@pytest.mark.asyncio
async def test_registered_cache_source_must_match_the_actual_primed_tokens(tmp_path):
    from controller.serialization import to_jsonable
    from controller.templates import token_hash
    from training.plan_gain_completion import _cache_fixture_shape

    _, state, values = await measured_fixture(tmp_path)
    source = state.steps[0]
    source = replace(source, input_ids=(11, 22, 33),
        receipt=replace(source.receipt, input_hash=token_hash((11, 22, 33))),
        metrics={"source_view": {"source_prefix_tokens": 1, "input_hash": token_hash((11, 22, 33)),
                                 "source_ids": ["actual-source-A"]}})
    before = {**values["direct_answer"][0], "cache_fixture_mode": "gpu_ready", "cache_source": to_jsonable(source)}
    before["observed_state_binding"] = {"native_condition": {"quotes": []}}
    shape = {"mode": "gpu_ready", "input_hash": token_hash((11, 22)), "cache_config_hash": "fixture"}
    run = {"cache_fixture": {"shape": shape, "prime_input_tokens": 2,
        "prime_receipt": {"generation_complete": True, "input_hash": token_hash((11, 22))}}}
    assert _cache_fixture_shape(before, run) == shape
    tampered = deepcopy(run)
    tampered["cache_fixture"]["shape"]["input_hash"] = token_hash((22,))
    with pytest.raises(ValueError, match="registered executed source prefix"):
        _cache_fixture_shape(before, tampered)
    tampered = deepcopy(run)
    tampered["cache_fixture"]["prime_input_tokens"] = 1
    with pytest.raises(ValueError, match="registered executed source prefix"):
        _cache_fixture_shape(before, tampered)


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
    labelled = tmp_path/"labels/effective-gains.jsonl"
    saved = json.loads(labelled.read_text())
    original_label = deepcopy(saved)
    saved["features"] = {"old_feature": 0}
    labelled.write_text(json.dumps(saved)+"\n")
    refreshed = tmp_path/"refreshed/effective-gains.jsonl"
    assert refresh_features(labelled, refreshed)["records"] == 1
    assert json.loads(refreshed.read_text()) == original_label
    assert json.loads(labelled.read_text())["features"] == {"old_feature": 0}
    changed_action = deepcopy(saved)
    changed_action["counterfactual"]["pre_registration"]["plan_candidate"]["steps"][-2]["task"]["need_ids"] = ["new-need"]
    modified = tmp_path/"modified.jsonl"
    modified.write_text(json.dumps(changed_action)+"\n")
    with pytest.raises(ValueError, match="action/grouping"):
        refresh_features(modified, tmp_path/"invalid.jsonl")

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
    assert mixed.cost_provenance["training_samples"] == 1
    assert mixed.provenance["measured_paid_cost_outcomes"] == {"committed_QA": 1, "confirmed_terminal_no_QA": 3}
    assert not mixed.supports_semantic_kind("read_once", costs=True)
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
