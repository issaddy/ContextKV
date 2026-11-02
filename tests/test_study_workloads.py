import asyncio
import copy
import json

import pytest

from bench.study_workloads import (WorkloadDispatcher, build_goal_graph_dataset,
    freeze_workload, load_goal_policy, prepare_workloads, read_workload_manifest)
from controller.schema import TurnRecord


def development_rows():
    rows = []
    for number in (1, 2):
        contexts = [{"document_id": f"program-{number}-{version}-{platform}",
            "document_family": f"program-{number}", "version": version,
            "valid_from": f"{version}-01-01", "title": f"Program {number} {version} {platform}",
            "applicability": {"PLATFORM": platform}, "text": "Public development source."}
            for version in ("2019", "2024") for platform in ("Windows", "Linux")]
        rows.append({"conversation_id": f"program-{number}", "task_id": f"program-{number}:1",
            "turn": 1, "Collection": "version_dev/constructed_dev", "contexts": contexts,
            "input": [{"speaker": "user", "text": f"What is the current allowance for Program {number} on Windows?"}],
            "targets": [{"text": "POISON_REFERENCE_NEVER_VISIBLE_TO_POLICY"}]})
    return rows


def test_dynamic_policy_uses_actual_answer_to_choose_graph_branch_and_keeps_references_offline():
    policy, rows = build_goal_graph_dataset(development_rows(), max_conversations=1)
    initial = policy.question_fn(rows[0], ())
    assert initial.goal_id == "current_windows" and initial.branch == "initial"
    history = (TurnRecord(1, initial.question, "Windows applicants receive 120 credits."),)
    corrected = policy.question_fn(rows[1], history)
    assert corrected.goal_id == "correct_linux" and corrected.branch == "windows_acknowledged"
    missing = (TurnRecord(1, initial.question, "I could not locate that allowance."),)
    clarified = policy.question_fn(rows[1], missing)
    assert clarified.goal_id == "clarify_linux" and "replacing Windows" in clarified.question
    assert corrected.metadata["actual_history"][0]["assistant_text"] == history[0].assistant_text
    assert "POISON_REFERENCE" not in str(policy.graphs) and "POISON_REFERENCE" not in str(rows)
    assert rows[0]["dataset"]["name"] == "dynamic_goal_dev"
    topic_ids = [row["workload_topic"] for row in rows]
    assert topic_ids[5] != topic_ids[0] and topic_ids[6] == topic_ids[0]
    assert all("targets" not in row for row in rows)
    assert corrected.metadata["offline_reference"] is None


def test_goal_graph_replays_conditions_versions_topic_switch_and_old_source_revisit():
    policy, rows = build_goal_graph_dataset(development_rows(), max_conversations=1)
    answers = ("Windows: 120 credits", "Linux: 100 credits under 2024 policy", "2019 Linux: 80 credits",
               "2019: 80; 2024: 100", "Windows: 120; Linux: 100", "Program 2 covers filing policies")
    history, goals = [], []
    for row, answer in zip(rows, (*answers, "Linux: 100 credits"), strict=True):
        decision = policy.question_fn(row, tuple(history))
        goals.append(decision.goal_id)
        history.append(TurnRecord(row["turn"], decision.question, answer))
    assert goals == ["current_windows", "correct_linux", "previous_policy", "compare",
                     "remove_platform", "topic_switch", "source_revisit"]
    restored = policy.question_fn(rows[-1], tuple(history[:-1]))
    assert restored == policy.question_fn(rows[-1], tuple(copy.deepcopy(history[:-1])))
    manifest = freeze_workload(rows, name="topic-return", concurrency=1)
    assert manifest.turns[5].topic_switch and manifest.turns[6].topic_switch
    assert manifest.turns[6].source_revisit and not manifest.turns[1].source_revisit


def test_fixed_manifest_roundtrips_with_exact_interleaving_and_questions_unchanged(tmp_path):
    _, rows = build_goal_graph_dataset(development_rows())
    before = copy.deepcopy(rows)
    manifest = freeze_workload(rows, name="concurrent-paused", concurrency=4,
        think_time_seconds=2.0, pause_after={rows[0]["task_id"]: 30.0},
        session_start_delays={rows[7]["conversation_id"]: 3.0})
    assert manifest.dispatch_order[:4] == (rows[0]["task_id"], rows[7]["task_id"],
                                           rows[1]["task_id"], rows[8]["task_id"])
    assert next(turn for turn in manifest.turns if turn.task_id == rows[1]["task_id"]).think_time_seconds == 30.0
    assert manifest.think_time_fn(rows[2], ()) == 2.0
    path = tmp_path/"workload.json"
    manifest.write(path)
    assert read_workload_manifest(path) == manifest
    assert rows == before
    with pytest.raises(ValueError, match="committed order"):
        freeze_workload(rows, name="invalid", concurrency=1,
            interleaving=[row["task_id"] for row in reversed(rows)])


@pytest.mark.asyncio
async def test_dispatcher_preserves_interleaving_and_pauses_after_actual_completion():
    _, source = build_goal_graph_dataset(development_rows())
    rows = [source[0], source[1], source[7], source[8]]
    manifest = freeze_workload(rows, name="two-sessions", concurrency=2, think_time_seconds=.01)
    dispatcher = WorkloadDispatcher(manifest)
    dispatched = []

    async def session(turns):
        history = []
        for row in turns:
            arrival = await dispatcher.wait_turn(row, tuple(history))
            dispatched.append(row["task_id"])
            assert arrival["committed_history_turns"] == len(history)
            await asyncio.sleep(.002)
            history.append(TurnRecord(row["turn"], "Actual user", "Actual assistant"))
            dispatcher.complete_turn(row["task_id"])

    await asyncio.gather(session(rows[:2]), session(rows[2:]))
    assert tuple(dispatched) == manifest.dispatch_order
    for turn in manifest.turns:
        arrival = dispatcher.arrivals[turn.task_id]
        assert arrival["actual_arrival_seconds"] >= arrival["planned_arrival_seconds"]
        if turn.release_after_task_id is not None:
            done = dispatcher._completion_times[turn.release_after_task_id]
            assert arrival["actual_arrival_seconds"]-done >= .009


@pytest.mark.asyncio
async def test_dispatcher_resume_skips_completed_ordinals_and_records_new_segment_origin():
    _, source = build_goal_graph_dataset(development_rows())
    rows = [source[0], source[1], source[7], source[8]]
    manifest = freeze_workload(rows, name="resume", concurrency=2, think_time_seconds=.001)
    dispatcher = WorkloadDispatcher(manifest)
    dispatcher.restore_completed((rows[0]["task_id"], rows[2]["task_id"]))
    arrivals = await asyncio.gather(dispatcher.wait_turn(rows[1], (TurnRecord(1, "q", "a"),)),
                                    dispatcher.wait_turn(rows[3], (TurnRecord(1, "q2", "a2"),)))
    assert [arrival["dispatch_index"] for arrival in arrivals] == [2, 3]
    assert all(arrival["planned_arrival_seconds"] == .001 for arrival in arrivals)
    assert all(arrival["resume_pause_basis"] == "configured pause reapplied from segment origin" for arrival in arrivals)


def test_prepare_freezes_one_dynamic_cohort_six_arrival_plans_and_separate_reference_metadata(tmp_path):
    import yaml
    originals = []
    for row in development_rows():
        for turn in range(1, 5):
            originals.append({**row, "turn": turn, "task_id": f"{row['conversation_id']}:{turn}",
                              "targets": [{"text": f"OFFLINE_ANSWER_{turn}"}]})
    source = tmp_path/"version_tasks.jsonl"
    source.write_text("".join(json.dumps(row)+"\n" for row in originals))
    manifest = prepare_workloads({"study": {"output_dir": str(tmp_path/"study")}}, {
        "datasets": {"versionqa": {"inputs": {"dev": str(source)}}}, "selection": {},
        "methods": [{"id": "B7", "strategy": "B7"}]})
    assert manifest["conversations"] == 2 and manifest["turns"] == 14
    assert manifest["public_test_assets_read"] == [] and manifest["public_quality_aggregate"] == "excluded"
    assert len(manifest["schedules"]) == 6
    rows = [json.loads(line) for line in open(manifest["tasks_path"])]
    assert "OFFLINE_ANSWER" not in str(rows)
    policy = load_goal_policy(manifest["goal_policy_path"])
    decision = policy.question_fn(rows[0], ())
    assert decision.metadata["offline_reference"]["answers"] == ["OFFLINE_ANSWER_1"]
    assert "OFFLINE_ANSWER" not in decision.question
    hashes, task_ids = set(), set()
    for schedule in manifest["schedules"]:
        arrival = read_workload_manifest(schedule["path"])
        hashes.add(arrival.source_rows_sha256)
        task_ids.add(arrival.dispatch_order)
        assert arrival.concurrency == schedule["concurrency"]
        assert all(turn.think_time_seconds == (schedule["pause_seconds"] if turn.turn == 4 else 0)
                   for turn in arrival.turns)
    assert len(hashes) == len(task_ids) == 1
    matrix = yaml.safe_load(open(manifest["diagnostic_matrix_path"]))
    assert list(matrix["datasets"]) == ["dynamic_goal_dev"]
    assert matrix["selection"]["development_weights"] == {"dynamic_goal_dev": 1.0}
