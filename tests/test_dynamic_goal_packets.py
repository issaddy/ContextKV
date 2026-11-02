from copy import deepcopy
from dataclasses import replace
import json

import pytest

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from bench.dynamic_goal_packets import REQUIRED_SCENARIOS, build_packets, read_jsonl
from bench.reader_diagnosis import DiagnosisProtocol, prepare_packet
from bench.study_workloads import load_goal_policy, prepare_workloads
from controller.context_schema import ContextPlan, EvidenceSet, Need, ReadTask, TaskState, TextRef
from controller.reading import build_read_plan
from controller.schema import TurnRecord
from controller.serialization import decode, to_jsonable
from controller.templates import ByteTemplate
from test_study_retrieval import OffsetBytes
from training.reading_data import build_reading_data


def constructed_inputs(tmp_path):
    data = tmp_path/"data"
    data.mkdir()
    (data/"sources.jsonl").write_text("".join(json.dumps({"source_id": f"train-{number}",
        "text": f"Literal development background {number}. "*10})+"\n" for number in range(2)))
    output = tmp_path/"study"
    build_reading_data({"study": {"output_dir": str(output)}, "training": {"data_dir": str(data),
        "version_dev_families": 2, "long_dev_cases": 0, "multihop_dev_cases": 0}})
    version = output/"evaluation/version_dev"
    manifest = prepare_workloads({"study": {"output_dir": str(output)}, "workloads": {"conversations": 1}}, {
        "datasets": {"versionqa": {"inputs": {"dev": str(version/"tasks.jsonl")}}},
        "selection": {}, "methods": [{"id": "E0", "strategy": "B7"}]})
    dynamic = read_jsonl(manifest["tasks_path"])
    policy = load_goal_policy(manifest["goal_policy_path"])
    tokenizer = OffsetBytes()
    docs = dynamic[0]["contexts"]
    corpus = CorpusIndex.from_records(({**doc, "passage_id": doc["document_id"]} for doc in docs),
        corpus_id="dynamic_goal_dev", snapshot_id="fixture")
    directory = SourceDirectory(corpus, lambda *args, **kwargs: (), tokenizer)
    pool = directory.full_scope(tuple(block.key for block in corpus.blocks))
    history, records = (), []
    answers = ("Windows 120 credits under 2024", "Linux 100 credits under 2024", "2019 Linux 80 credits",
               "2019 80 credits versus 2024 100 credits", "Windows 120 and Linux 100 credits",
               "The other program is summarized here.", "Linux 100 credits under the current policy")
    for number, (row, answer) in enumerate(zip(dynamic, answers, strict=True)):
        decision = policy.question_fn(row, history)
        state = TaskState("model-inferred-object", decision.question, needs=(Need("model-need",
            (TextRef(number+1, "user", 0, len(decision.question)),)),), current_turn_id=number+1,
            source_scope=tuple(block.key for block in corpus.blocks))
        plan = ContextPlan(f"E0-{number}", state, EvidenceSet(f"selected-{number}", pool.units[:1],
            source_blocks=tuple(block for block in pool.source_blocks if block.key == pool.units[0].source_spans[0].source_key)),
            read_layout="source_first", read_packing="source_windows")
        records.append({"task_id": row["task_id"], "conversation_id": row["conversation_id"],
            "status": "completed", "split": "dev", "history_origin": "model_actual",
            "question": decision.question, "question_policy": to_jsonable(decision),
            "actual_history": to_jsonable(history), "candidate_pool": to_jsonable(pool),
            "context_plan": to_jsonable(plan), "planning_trace": {"auxiliary_usage": {
                "input_tokens": 0, "output_tokens": 0, "compute_ms": 0.0}},
            "cost": {"complete_ms": 1}, "response": {"fixture": "specified simulated rollout"},
            "measurement_kind": "simulation"})
        history = (*history, TurnRecord(number+1, decision.question, answer))
    provenance = {"version_references": str(version/"references.jsonl"),
                  "dynamic_tasks": manifest["tasks_path"], "run_dir": "specified-unit-test-rollout"}
    return records, dynamic, read_jsonl(version/"tasks.jsonl"), read_jsonl(version/"references.jsonl"), tokenizer, provenance


def build(inputs):
    records, dynamic, version, references, tokenizer, provenance = inputs
    return build_packets(records, dynamic, version, references, tokenizer, provenance=provenance)


def test_literal_goal_packets_cover_all_scenarios_and_keep_actual_pools_and_histories(tmp_path):
    inputs = constructed_inputs(tmp_path)
    packets, manifest = build(inputs)
    assert len(packets) == 7 and manifest["reviewed"] == 6 and manifest["review_incomplete"] == 1
    assert all(manifest["scenario_counts"][name] for name in REQUIRED_SCENARIOS)
    assert manifest["missing_scenarios"] == [] and manifest["public_test_assets_read"] == []
    for packet, record in zip(packets, inputs[0], strict=True):
        assert packet["candidate_pool"] == record["candidate_pool"]
        assert packet["e0_plan"] == record["context_plan"]
        assert packet["actual_history"] == record["actual_history"]
        assert packet["verified_task_state"]["raw_question"] == record["question"]
        assert packet["verified_task_state"]["task_id"] != record["context_plan"]["task_state"]["task_id"]
        prepared = prepare_packet(packet, inputs[4], DiagnosisProtocol(100000))
        if packet["review_status"] == "reviewed":
            assert prepared["diagnosis"].e2[0].missing_requirements == ()
            blocks = {block.key: block for block in prepared["diagnosis"].e2[0].evidence_set.source_blocks}
            for unit in prepared["diagnosis"].e2[0].evidence_set.selected_units:
                span = unit.source_spans[0]
                assert unit.text == blocks[span.source_key].text[span.start:span.end]
        else:
            assert packet["goal_id"] == "topic_switch"
            assert packet["scope_review"]["complete"] is False
            assert packet["annotations"]["requirements"][0]["alternatives"] == []


def test_version_comparison_is_one_joint_witness_and_platform_removal_keeps_both_points(tmp_path):
    packets, _ = build(constructed_inputs(tmp_path))
    compared = next(packet for packet in packets if packet["goal_id"] == "compare")
    needs = compared["annotations"]["requirements"]
    assert len(needs) == 1 and len(needs[0]["alternatives"][0]["spans"]) == 2
    assert {span["source_key"]["version"] for span in needs[0]["alternatives"][0]["spans"]} == {"2019", "2024"}
    assert compared["verified_task_state"]["version_intent"]["mode"] == "compare"
    removed = next(packet for packet in packets if packet["goal_id"] == "remove_platform")
    assert len(removed["annotations"]["requirements"]) == 2
    assert removed["verified_task_state"]["conditions"] == []
    revisited = next(packet for packet in packets if packet["goal_id"] == "source_revisit")
    corrected = next(packet for packet in packets if packet["goal_id"] == "correct_linux")
    assert revisited["verified_task_state"]["task_id"] == corrected["verified_task_state"]["task_id"]
    assert all(span["source_key"]["version"] == "2024" for row in revisited["annotations"]["requirements"]
        for witness in row["alternatives"] for span in witness["spans"])


@pytest.mark.parametrize("corruption", ("history", "quote", "test_reference", "test_rollout"))
def test_packets_reject_nonliteral_labels_different_history_and_public_test(tmp_path, corruption):
    values = list(constructed_inputs(tmp_path))
    if corruption == "history":
        values[0][1]["actual_history"][0]["assistant_text"] = "unobserved author answer"
    elif corruption == "quote":
        values[3][0]["source_spans"][0]["quote"] = "invented fact"
    elif corruption == "test_reference":
        values[3][0]["split"] = "test"
    else:
        values[0][0]["split"] = "test"
    with pytest.raises(ValueError):
        build(tuple(values))


def test_unfinished_real_rows_are_skipped_and_scenario_gaps_are_reported(tmp_path):
    values = list(constructed_inputs(tmp_path))
    values[0] = deepcopy(values[0][:2])
    values[0][1]["status"] = "failed"
    packets, manifest = build(tuple(values))
    assert len(packets) == 1
    assert manifest["skipped_actual_runs"][0]["reason"] == "actual_E0_run_not_completed"
    assert set(manifest["missing_scenarios"]) == set(REQUIRED_SCENARIOS)


def test_builder_separates_recorded_initial_plan_from_runtime_gap_expansion(tmp_path):
    values = list(constructed_inputs(tmp_path))
    record = values[0][0]
    old = decode(ContextPlan, record["context_plan"])
    initial = build_read_plan(old.task_state, old.selected_set, old.coverage_scope, old.budget, template=ByteTemplate())
    gap = ReadTask("gap:1:read:0", "READ", ("model-need",), input_claim_ids=("read:0:claim:0",),
                   predecessor_step_ids=("read:0",))
    expanded = replace(initial, steps=(*initial.steps, gap))
    record["context_plan"] = to_jsonable(expanded)
    record["planning_trace"]["context_plans"] = [to_jsonable(initial)]
    record["reading_outcome"] = {"steps": [{"step_id": initial.steps[0].step_id,
        "input_ids": [1, 2], "input_messages": [{"role": "user", "content": "specified original input"}]},
        {"step_id": gap.step_id, "input_ids": [3], "input_messages": []}]}
    packets, _ = build(tuple(values))
    first = packets[0]
    assert first["e0_plan"] == first["e0_initial_plan"] == to_jsonable(initial)
    assert first["e0_actual_expanded_plan"] == to_jsonable(expanded)
    assert first["e0_plan_stage"] == "initial"
    assert len(first["source_initial_read_inputs"]) == 1
    assert first["source_initial_read_inputs"][0]["step_id"] == initial.steps[0].step_id
