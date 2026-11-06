"""Specified paired scores/costs test CPU regression, not experimental GPU rewards."""
from copy import deepcopy
from dataclasses import replace

import pytest

from controller.claim_graph import ClaimGraph
from controller.context_schema import EvidenceClaim, Need, ReadingOutcome, ReadResult, ReadTask, StepRecord
from controller.plan_policy import CompleteCost, complete_plan
from controller.plan_gain_protocol import input_protocol
from controller.templates import ByteTemplate
from controller.schema import SessionSnapshot, TurnRecord
from training.plan_gain_protocol import ZERO_AUX, observed_state_binding
from training.train_plan_gain import (
    COST_FIELDS, build_gain_record,
)
from context_helpers import context_plan


def paired_records(*, split="train", tag="training"):
    base, _ = context_plan("Infer the qualifying result.")
    task_state = replace(base.task_state, task_id="task:"+tag, needs=(Need("need", operation="infer"),),
                         current_turn_id=2)
    unit = base.selected_set.selected_units[0]
    leaf = EvidenceClaim("past:claim:0", "need", "extract", text="Existing source premise", entity="R1",
                         value="2", unit="g", source_spans=unit.source_spans)
    old_task = ReadTask("past", "READ", ("need",), ("u",))
    old_result = ReadResult("past", (leaf,), covered_units=("u",))
    old_step = StepRecord("specified-source", "past", old_task, input_ids=(1, 2), output_ids=(3,), result=old_result)
    base = replace(base, task_state=task_state, steps=(old_task,), budget=replace(base.budget, output_tokens=64),
                   answer_plan=ClaimGraph((leaf,)).answer_plan(task_state))
    state = ReadingOutcome(base, results=(old_result,), steps=(old_step,), claims=(leaf,), auxiliary_usage=dict(ZERO_AUX))
    snapshot = SessionSnapshot("source:"+tag, raw_history=(TurnRecord(1, "Use the current condition", "Confirmed:"+tag),))
    paid = CompleteCost(10000, 20000, 30000, 5000, 500)
    observed = observed_state_binding(state, snapshot, paid_cost=paid,
        native_condition={"cache_state": "specified paired fixture"}, reader_checkpoint="reader-fixture",
        reader_profile_id="profile-fixture")
    baseline = dict(zip(COST_FIELDS, (100, 2, 3, 1000, 40), strict=True))
    prior = dict(zip(COST_FIELDS, (10000, 20, 30, 5000, 500), strict=True))
    common = {"measurement_kind": "hardware", "history_origin": "frozen_actual_rollout",
        "question": task_state.raw_question, "actual_history": observed["actual_history"],
        "state_key": observed["state_key"], "reader_checkpoint": "reader-fixture",
        "reader_profile_id": "profile-fixture", "split": split, "answer": "specified answer",
        "cohort_split": split, "cohort_manifest_sha256": "specified-fixture",
        "input_protocol": input_protocol(replace(ByteTemplate().fingerprint, model_revision="reader-fixture")),
        "continuation_gap_limits": {"max_units": 2, "max_passages": 2},
        "scorer_identity": {"kind": "specified unit-test scorer"},
        "comparison_budget": {"window_tokens": base.budget.window_tokens, "answer_output_tokens": 64},
        "observed_state_binding": observed, "output_budgets": {"ANSWER": 64}}
    direct = {**deepcopy(common), "semantic_kind": "direct_answer", "plan_id": "direct:"+tag,
        "quality": .2, "read_count": 0, "native_receipts": [
            {"input_hash": "specified-direct:"+tag, "generation_complete": True}],
        "new_action_counts": {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1},
        "remaining_cost": baseline, "cost": {name: prior[name]+baseline[name] for name in COST_FIELDS}}
    rows = []
    for kind, cost_values, quality in (
        ("read_once", (140, 2.5, 3.5, 1300, 80), .4),
        ("reduce_once", (70, 1, 2, 800, 30), .6),
        ("gap_retrieval", (120, 3, 4, 1100, 47), .3),
    ):
        task = (ReadTask("new-read", "READ", ("need",), ("u",), predecessor_step_ids=("past",),
                         input_budget=1000, output_budget=64) if kind == "read_once" else
                ReadTask("new-reduce", "REDUCE", ("need",), input_claim_ids=(leaf.claim_id,),
                         predecessor_step_ids=("past",), input_budget=1000, output_budget=64,
                         operation="infer") if kind == "reduce_once" else None)
        plan = complete_plan(replace(base, plan_id=kind+":"+tag, steps=(old_task, *((task,) if task else ()))),
            (), semantic_kind=kind, reader_checkpoint="reader-fixture")
        future = dict(zip(COST_FIELDS, cost_values, strict=True))
        counts = {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1}
        counts[{"read_once": "READ", "reduce_once": "REDUCE", "gap_retrieval": "GAP_RETRIEVAL"}[kind]] = 1
        candidate = {**deepcopy(common), "semantic_kind": kind, "plan_id": plan.plan_id,
            "quality": quality, "read_count": counts["READ"], "new_action_counts": counts,
            "native_receipts": [{"input_hash": "specified-"+kind+":"+tag, "generation_complete": True}],
            "remaining_cost": future, "cost": {name: prior[name]+future[name] for name in COST_FIELDS}}
        rows.append(build_gain_record(plan, state, snapshot, direct, candidate))
    return rows, state


def test_action_features_and_sizes_are_pre_execution_information():
    rows, state = paired_records()
    for row in rows:
        assert row["features"][row["semantic_kind"]] == 1
        assert sum(row["features"][kind] for kind in ("read_once", "reduce_once", "gap_retrieval")) == 1
        assert row["direct_features"]["direct_answer"] == 1
        assert row["direct_features"]["read_steps"] == 0
        assert row["features"]["existing_claim_chars"] > 0
    assert rows[0]["features"]["planned_read_chars"] > 0
    assert rows[1]["features"]["reduce_input_claim_chars"] > 0
    # Changing future outcomes alters targets and leaves the frozen features intact.
    from controller.serialization import decode
    from controller.plan_policy import PlanCandidate
    row = rows[1]
    candidate = deepcopy(row["counterfactual_run"])
    candidate["quality"] = .9
    candidate["remaining_cost"]["complete_ms"] = 60
    candidate["cost"]["complete_ms"] = 10060
    changed = build_gain_record(decode(PlanCandidate, row["plan"]), state,
        SessionSnapshot("source", raw_history=(TurnRecord(1, "Use the current condition", "Confirmed:training"),)),
        row["direct_run"], candidate)
    assert changed["features"] == row["features"]
    assert changed["delta_q"] != row["delta_q"]


def test_observed_budget_features_use_actual_paid_state_and_hard_steps():
    from controller.plan_policy import PlanCandidate, plan_features
    from controller.serialization import decode
    rows, state = paired_records()
    row = rows[0]
    plan = decode(PlanCandidate, row["plan"])
    paid = decode(CompleteCost, row["observed_state_binding"]["paid_cost"])
    before = plan_features(plan, state, paid_cost=paid)
    after = plan_features(plan, state, paid_cost=replace(paid, input_tokens=paid.input_tokens+10,
                                                       latency_ms=paid.latency_ms+20))
    assert after["remaining_total_tokens"] == before["remaining_total_tokens"]-10
    assert after["paid_complete_ms"] == before["paid_complete_ms"]+20
    assert before["paid_steps"] == len(state.steps)
    limited = replace(state, plan=replace(state.plan, budget=replace(state.plan.budget,
        max_steps=len(state.steps)+2, max_complete_ms=20000)))
    generous = replace(limited, plan=replace(limited.plan, budget=replace(limited.plan.budget,
        max_steps=len(state.steps)+20)))
    assert plan_features(plan, limited, paid_cost=paid)["remaining_hard_steps"] == 2
    assert plan_features(plan, generous, paid_cost=paid)["remaining_hard_steps"] == 20
    assert plan_features(plan, limited, paid_cost=paid)["remaining_complete_ms"] == 10000
    assert before["complete_ms_unbounded"] == 1
    assert plan_features(plan, limited, paid_cost=paid)["complete_ms_unbounded"] == 0
    unbounded = replace(state, plan=replace(state.plan, budget=replace(state.plan.budget,
        max_total_tokens=None)))
    assert plan_features(plan, unbounded, paid_cost=paid)["total_tokens_unbounded"] == 1


def test_observed_value_uses_existing_support_and_bridge_state_without_future_results():
    from controller.context_schema import OpenNeed, SupportBundle
    from controller.plan_policy import PlanCandidate, plan_features
    from controller.serialization import decode
    rows, state = paired_records()
    plan = decode(PlanCandidate, rows[0]["plan"])
    paid = decode(CompleteCost, rows[0]["observed_state_binding"]["paid_cost"])
    support = SupportBundle("known", "need", ("u",), .5, .8, bridge_value=.6)
    material = replace(plan.context_plan.selected_set, need_support=(support,))
    plan = replace(plan, context_plan=replace(plan.context_plan, selected_set=material))
    bridge = replace(state, open_needs=(OpenNeed("need", "founded", ("R1",),
        parent_claim_ids=(state.claims[0].claim_id,)),))
    features = plan_features(plan, bridge, paid_cost=paid)
    assert features["retrieval_support_needs"] == 1
    assert features["retrieval_support_mean"] == pytest.approx(.4)
    assert features["retrieval_bridge_mean"] == pytest.approx(.3)
    assert features["infer_needs"] == features["observed_bridge_gaps"] == 1
    assert features["observed_terminal_needs"] == 0
    terminal = replace(state, results=(replace(state.results[0], support=("need",)),), open_needs=())
    completed = plan_features(plan, terminal, paid_cost=paid)
    assert completed["observed_terminal_needs"] == 1
    assert completed["observed_bridge_gaps"] == 0
    assert completed["existing_claim_chars"] == features["existing_claim_chars"]
    assert completed["remaining_hard_steps"] == features["remaining_hard_steps"]
    absent = plan_features(replace(plan, context_plan=state.plan), state, paid_cost=paid)
    zero = replace(support, joint_support=0, bridge_value=0)
    known_zero = plan_features(replace(plan, context_plan=replace(plan.context_plan,
        selected_set=replace(material, need_support=(zero,)))), state, paid_cost=paid)
    assert absent["retrieval_support_needs"] == 0 and known_zero["retrieval_support_needs"] == 1



def test_frozen_decision_identity_includes_actual_source_and_reader_geometry():
    from training.train_plan_gain import working_state_key

    _, state = paired_records()
    snapshot = SessionSnapshot("actual")
    identity = working_state_key(state, snapshot)
    block = state.plan.selected_set.source_blocks[0]
    changed = replace(state, plan=replace(state.plan, selected_set=replace(state.plan.selected_set,
        source_blocks=(replace(block, title="Different rendered metadata"),))))
    assert working_state_key(changed, snapshot) != identity
    assert working_state_key(replace(state, plan=replace(state.plan, read_layout="source_first")), snapshot) != identity
    assert working_state_key(replace(state, plan=replace(state.plan,
        budget=replace(state.plan.budget, body_tokens=1))), snapshot) != identity
    assert working_state_key(replace(state, claims=()), snapshot) != identity
