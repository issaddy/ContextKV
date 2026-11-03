"""The pilot uses the existing runtime traces and reviewed SFT formats end to end."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from controller.context_schema import ContextBudget, EvidencePool, EvidenceSet, Need, SupportScores, TaskState
from controller.serialization import to_jsonable
from training.reader_data import export_reader_traces, file_digest, history_digest, rows, write_rows
from training.reader_stage_workflow import prepare_reader_stage, prepare_reader_phases
from test_reader_stage_audit import trace, review


def stage_fixture(tmp_path):
    members = [{"task_id": name, "conversation_id": name, "turn": 1, "split": split,
        "actual_history_sha256": history_digest([])} for name, split in (("train", "train"), ("rejected", "train"), ("dev", "dev"))]
    cohort = tmp_path/"cohort.json"
    cohort.write_text(json.dumps({"members": members, "minimum_candidates_per_state": 4, "maximum_candidates_per_state": 6}))
    labels = tmp_path/"labels"
    labels.mkdir()
    for split in ("train", "dev"):
        write_rows(labels/f"{split}_task_labels.jsonl", [{"task_id": split, "correct_task_binding": True,
            "source_support_verified": True}])
    (labels/"task_labels_manifest.json").write_text(json.dumps({"status": "incomplete", "cohort_manifest_sha256": file_digest(cohort)}))
    config = {"training": {"data_dir": "runs/study/data"}, "reader_stage": {
        "output_dir": str(tmp_path/"stage"), "cohort_manifest": str(cohort), "labels_dir": str(labels),
        "reader_training_config": "configs/reader_r1.yaml", "reader_output_dir": str(tmp_path/"models"),
        "reader_merged_dir": str(tmp_path/"merged"), "source_data_dir": "runs/study/data", "source_state_db": str(tmp_path/"native.sqlite3"),
        "collection": {"run_config": "configs/qwen3_32b.yaml", "reader_checkpoint": "fixture-r0",
            "reader_profile_id": "fixture-profile", "support_asset": "fixture-support"}}}
    return config


def test_stage_records_rejected_state_and_writes_bound_existing_pipeline_commands(tmp_path):
    prepared = prepare_reader_stage(stage_fixture(tmp_path))
    assert prepared["declared_states"] == 3 and prepared["available_states"] == 2
    assert prepared["available_splits"] == {"train": 1, "dev": 1}
    assert prepared["excluded_states"] == [{"task_id": "rejected", "cohort_split": "train", "reason": "no_accepted_task_state_label"}]
    for split, path in prepared["collection_configs"].items():
        config = yaml.safe_load(Path(path).read_text())
        settings = config["preferences"]
        assert settings["reader_data_only"] and settings["reading"]
        assert settings["eligible_task_ids"] == [split]
        assert settings["read_layout"] == "source_first"
        assert settings["turns_path"].endswith(f"{split}_task_labels.jsonl")
        assert prepared["commands"][f"collect_{split}"][2] == "training.preference_rollout"
    reader = yaml.safe_load(Path(prepared["reader_training_config"]).read_text())["reader_training"]
    assert reader["data"]["minimum_train_turns"] == 1
    assert reader["data"]["read_protocol_versions"] == [2, 3]
    assert reader["data"]["require_independent_review"]
    assert len(reader["data"]["rollouts"]) == 2
    assert all(spec["state_db"].endswith("runtime/sessions.sqlite3") for spec in reader["data"]["rollouts"])


def test_phase_configs_chain_adapters_and_never_invent_absent_reduce_examples(tmp_path):
    prepared = prepare_reader_stage(stage_fixture(tmp_path))
    effective = yaml.safe_load(Path(prepared["reader_stage_config"]).read_text())
    reader = yaml.safe_load(Path(prepared["reader_training_config"]).read_text())["reader_training"]
    data = Path(reader["data"]["output_dir"])
    traces = []
    for task in ("train", "dev"):
        for phase in ("READ", "ANSWER"):
            item = trace(task+phase, phase)
            item.update(task_id=task, conversation_id=task)
            traces.append(item)
    write_rows(data/"traces.jsonl", traces)
    write_rows(data/"reviews.jsonl", [review(item, visible_prompt_support=True,
        review_protocol="visible_target_independent_review_v2") for item in traces])
    result = prepare_reader_phases(effective)
    read, reduce, answer = result["stages"]
    assert read["examples"] == answer["examples"] == {"train": 1, "validation": 1}
    assert reduce["status"] == "no_observed_training_examples" and reduce["train_command"] is None
    read_config = yaml.safe_load(Path(read["training_config"]).read_text())["reader_training"]
    answer_config = yaml.safe_load(Path(answer["training_config"]).read_text())["reader_training"]
    assert answer_config["initial_adapter"] == str(Path(read_config["output_dir"])/"adapter")
    assert rows(data/"reduce"/"train.jsonl") == []
    assert all(item["conversation_id"] == "dev" for item in rows(data/"answer"/"validation.jsonl"))


@pytest.mark.asyncio
async def test_reader_only_collection_replays_four_real_sets_and_exports_native_dev_runtime(tmp_path, monkeypatch):
    from bench.mock_engine import MockEngine
    from controller.planner import PlannerKnobs
    from controller.service import Controller
    from controller.session_store import SessionStore
    from controller.templates import ByteTemplate
    from training.preference_rollout import collect_preferences
    from test_reading_runtime import material, fact, read_output

    first, a = material("Item A is eligible.", "a")
    second, b = material("Item B is eligible.", "b")
    pool = EvidencePool((a, b), (first, second))
    sets = {"current": EvidenceSet("both", (a, b), source_blocks=(first, second)),
        "a_only": EvidenceSet("a", (a,), source_blocks=(first,)),
        "b_only": EvidenceSet("b", (b,), source_blocks=(second,)), "deleted_premise": EvidenceSet("empty", ())}
    usage = {"input_tokens": 0, "output_tokens": 0, "compute_ms": 0}
    model = SimpleNamespace(backend=SimpleNamespace(usage=usage), score_support=lambda *args: SupportScores(),
        release_task=lambda *args: None, close=lambda: None, resolve=None)
    monkeypatch.setattr("training.preference_rollout.SupportModel.from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr("training.preference_rollout.candidate_sets", lambda *args, **kwargs: sets)

    async def no_judge(*args):
        raise AssertionError("reader targets are collected before P1 rewards")

    monkeypatch.setattr("training.preference_rollout._judge", no_judge)
    directory = SimpleNamespace(retrieve_pool=lambda *args: pool, units_by_id={"a": a, "b": b})

    class Engine(MockEngine):
        calls = 0

        async def generate(self, *, response_schema=None, **kwargs):
            self.calls += 1
            if response_schema:
                properties = response_schema["properties"]["claims"]["items"]["properties"]
                spans = properties["span_ids"]["items"]["enum"]
                self.response_text = json.dumps(read_output(*(fact(span, entity=span) for span in spans)))
            else:
                self.response_text = "Supported final answer."
            return await super().generate(**kwargs)

    output = tmp_path/"collection"
    store = SessionStore(output/"runtime"/"sessions.sqlite3")
    template = ByteTemplate()
    engine = Engine(fingerprint=template.fingerprint)
    controller = Controller(store, engine, template=template, knobs=PlannerKnobs(strategy="always_rebuild"))
    state = TaskState("task", "Which items are eligible?", needs=(Need("n"),), current_turn_id=1)
    label = {"task_id": "task", "conversation_id": "session", "turn": 1, "split": "train",
        "actual_history": [], "question": state.raw_question, "snapshot_id": "native", "task_state": to_jsonable(state)}
    labels_path = tmp_path/"dev_task_labels.jsonl"
    write_rows(labels_path, [label])
    cohort = tmp_path/"cohort.json"
    cohort.write_text(json.dumps({"members": [{"task_id": "task", "conversation_id": "session", "turn": 1,
        "split": "dev", "actual_history_sha256": history_digest([])}], "minimum_candidates_per_state": 4, "maximum_candidates_per_state": 6}))
    config = {"preferences": {"turns_path": str(labels_path), "output_dir": str(output),
        "cohort_manifest": str(cohort), "cohort_split": "dev", "reading": True, "reader_data_only": True,
        "read_packing": "mixed", "max_total_tokens": None}}
    manifest = await collect_preferences(config, controller=controller, directory=directory)
    actual = rows(manifest["actual_turns_path"])
    assert len(actual) == 4 and all(item["split"] == "train" and item["cohort_split"] == "dev" for item in actual)
    assert all(item["snapshot_id"] == "native" for item in actual)
    assert not (output/"rewards.jsonl").exists() and manifest["judge_input_tokens"] == 0
    paid_calls = engine.calls
    await collect_preferences(config, controller=controller, directory=directory)
    assert engine.calls == paid_calls and len(rows(manifest["actual_turns_path"])) == 4
    exported = export_reader_traces({"reader_training": {"data": {"output_dir": str(tmp_path/"reader"),
        "cohort_manifest": str(cohort), "rollouts": [{"turns_path": manifest["actual_turns_path"],
            "state_db": str(output/"runtime"/"sessions.sqlite3"), "split": "train",
            "source_partition": "native_multidoc2dial_train", "reader_identity": "explicit-test-engine"}]}}})
    traces = rows(exported["traces_path"])
    assert exported["phases"]["ANSWER"] == 4 and exported["phases"]["READ"] >= 3
    assert all(item["cohort_split"] == "dev" for item in traces)
    assert all(item["messages"] and item["prompt_token_hash"] for item in traces)
    for item in traces:
        if item["phase"] == "READ":
            assert item["read_protocol_version"] == item["output_schema"]["properties"]["format_version"]["const"]
    await controller.aclose()
    store.close()


def test_candidate_expansion_uses_distinct_observed_source_deletions(monkeypatch):
    from adapters.support_model import SetPrediction
    from controller.templates import ByteTokenizer
    from training.preference_rollout import candidate_sets, real_candidate_family
    from test_reading_runtime import material
    blocks, units = zip(*(material(f"Original item {name}.", name) for name in ("a", "b", "c", "d")))
    pool = EvidencePool(units, blocks)
    all_sources = EvidenceSet("all", units, source_blocks=blocks)
    current = EvidenceSet("current", units[:2], source_blocks=blocks[:2])
    monkeypatch.setattr("training.preference_rollout.BudgetedEvidencePlanner.search_sets", lambda *args: (all_sources,))
    model = SimpleNamespace(predict_set=lambda state, selected, supports: SetPrediction(len(selected)/4), resolve=lambda ref: "")
    state = TaskState("task", "Which items?", needs=(Need("n"),))
    candidates = candidate_sets(state, pool, SupportScores(), model, ByteTokenizer(), ContextBudget(body_tokens=8192),
        current_set=current, expand_interventions=True)
    selected, audit = real_candidate_family(state, pool, candidates)
    assert audit["ready"] and 4 <= len(selected) <= 6
    assert "current" in selected and any(category.startswith("source_deletion:") for category in selected)
    assert len({tuple(sorted(unit.unit_id for unit in evidence.selected_units)) for evidence in selected.values()}) == len(selected)
    for evidence in selected.values():
        assert all(unit in units for unit in evidence.selected_units)
        assert all(block in blocks for block in evidence.source_blocks)


def test_saved_original_block_current_set_preserves_source_coordinates():
    from training.preference_rollout import current_evidence_from_blocks
    from controller.evidence import make_evidence
    from test_reader_training import TrainingTokenizer
    block = make_evidence("Allowance: 12 credits.", "policy", corpus_id="native", version="v1")
    block = replace(block, span=(80, 80+len(block.text)))
    current = current_evidence_from_blocks({"task_id": "task", "snapshot_id": "snapshot",
        "current_source_blocks": to_jsonable((block,))}, TrainingTokenizer())
    assert current.source_blocks == (block,)
    spans = [span for unit in current.selected_units for span in unit.source_spans]
    assert spans and all(span.source_key == block.key and span.start >= 80 for span in spans)
    assert "".join(unit.text for unit in current.selected_units) == block.text


def test_stage_can_reuse_existing_teacher_support_without_a_trained_asset(tmp_path):
    config = stage_fixture(tmp_path)
    settings = config["reader_stage"]["collection"]
    settings.pop("support_asset")
    settings.update(support_mode="teacher", support_teacher_url="http://127.0.0.1:30002")
    result = prepare_reader_stage(config)
    collected = yaml.safe_load(Path(result["collection_configs"]["train"]).read_text())["preferences"]
    assert collected["support_mode"] == "teacher" and collected["support_teacher_url"].endswith(":30002")
    assert "support_asset" not in collected
